#!/usr/bin/env python3
"""
Stella Alpha - Main Training Pipeline
======================================

Multi-Timeframe Pure ML pipeline for HIGH R:R SHORT trade prediction.
Extends V4 with D1 context, statistical validation, and loss analysis.

Pipeline Steps:
  1. Load H4 + D1 data
  2. Merge (leakage-safe: previous day D1 only)
  3. Feature engineering (H4 + D1 derived + MTF cross-TF)
  4. Pre-compute labels for all (TP, SL, Hold) combos
  5. Define walk-forward folds
  6. Run experiments (parallel) with trade recording
  7. Select best configuration
  8. Train final model
  9. Run loss analysis
 10. Generate filter recommendations
 11. Publish artifacts

Usage:
    python run_pure_ml_stella_alpha.py \\
        --config config/pure_ml_stella_alpha_settings.yaml \\
        --input  data/EURUSD_H4_features.csv \\
        --d1     data/EURUSD_D1_features.csv \\
        --workers 8

    # Resume interrupted run (checkpoint auto-detected):
    python run_pure_ml_stella_alpha.py --config ... --input ... --d1 ...

    # Debug single-threaded:
    python run_pure_ml_stella_alpha.py --config ... --input ... --d1 ... -w 1 --debug
"""

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

warnings.filterwarnings("ignore")
os.environ["LIGHTGBM_VERBOSITY"] = "-1"

try:
    import lightgbm as lgb
    lgb.register_logger(lgb.basic.NullLogger())
except Exception:
    pass

import numpy as np
import pandas as pd
import yaml
import joblib

# Add src to path
SCRIPT_DIR = Path(__file__).parent.resolve()
SRC_DIR = SCRIPT_DIR / "src"
for p in (str(SCRIPT_DIR), str(SRC_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

# ---------------------------------------------------------------------------
# Core imports
# ---------------------------------------------------------------------------
from pure_ml_labels import (
    precompute_all_labels,
    apply_precomputed_labels,
    get_unique_label_configs,
    estimate_class_imbalance,
    TradeDirection,
)
from experiment import (
    generate_config_space,
    run_all_experiments,
    select_best_result,
    filter_passed_results,
    sort_results_by_ranking,
)
from training import train_model, calibrate_model, save_model, get_best_model_type
from evaluation import compute_metrics, optimize_threshold

# ---------------------------------------------------------------------------
# Stella Alpha – new modules
# ---------------------------------------------------------------------------
try:
    from data_merger import DataMerger
    DATA_MERGER_AVAILABLE = True
except ImportError:
    DATA_MERGER_AVAILABLE = False

try:
    from feature_engineering import FeatureEngineering
    FEATURE_ENGINEERING_AVAILABLE = True
except ImportError:
    FEATURE_ENGINEERING_AVAILABLE = False

try:
    from loss_analysis import LossAnalyzer
    LOSS_ANALYSIS_AVAILABLE = True
except ImportError:
    LOSS_ANALYSIS_AVAILABLE = False

try:
    from filter_recommendations import FilterRecommendationEngine
    FILTER_ENGINE_AVAILABLE = True
except ImportError:
    FILTER_ENGINE_AVAILABLE = False


# =============================================================================
# LOGGER
# =============================================================================

class PipelineLogger:
    """Simple timestamped logger with optional file output."""

    def __init__(self, log_file: Optional[str] = None, verbose: bool = True):
        self.verbose  = verbose
        self.log_file = log_file
        self._start   = datetime.now()
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    def log(self, message: str, level: str = "INFO"):
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] [{level}] {message}"
        if self.verbose:
            print(line)
        if self.log_file:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def header(self, title: str):
        border = "=" * 70
        self.log("")
        self.log(border)
        self.log(f"  {title}")
        self.log(border)

    def subheader(self, title: str):
        self.log("")
        self.log(f"--- {title} ---")

    def elapsed(self) -> float:
        return (datetime.now() - self._start).total_seconds()


# =============================================================================
# DATA LOADING
# =============================================================================

def _load_csv(path: str, config: Dict, logger: PipelineLogger) -> pd.DataFrame:
    """Load a CSV with flexible timestamp parsing."""
    schema = config.get("schema", {})
    ts_col = schema.get("timestamp_column", "timestamp")
    ts_fmt = schema.get("timestamp_format", "%Y.%m.%d %H:%M:%S")

    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    if ts_col in df.columns:
        try:
            df[ts_col] = pd.to_datetime(df[ts_col], format=ts_fmt)
        except Exception:
            df[ts_col] = pd.to_datetime(df[ts_col], infer_datetime_format=True)
        df = df.sort_values(ts_col).reset_index(drop=True)

    logger.log(f"   Loaded {len(df):,} rows | {df.columns[0]} … {df.columns[-1]}")
    if ts_col in df.columns:
        logger.log(f"   Date range: {df[ts_col].min()} → {df[ts_col].max()}")
    return df


def load_h4_data(csv_path: str, config: Dict, logger: PipelineLogger) -> pd.DataFrame:
    logger.log(f"Loading H4 data: {csv_path}")
    return _load_csv(csv_path, config, logger)


def load_d1_data(csv_path: str, config: Dict, logger: PipelineLogger) -> pd.DataFrame:
    logger.log(f"Loading D1 data: {csv_path}")
    return _load_csv(csv_path, config, logger)


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def run_feature_engineering(
    df: pd.DataFrame,
    config: Dict,
    logger: PipelineLogger,
    d1_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Apply feature engineering.
    With D1 data: merge first, then compute H4 + D1 derived + MTF features.
    Without D1 data: compute H4 features only (V4 compatibility).
    """
    feats_config = config.get("features", {})
    mtf_config   = config.get("multi_timeframe", {})

    # ── D1 merge ─────────────────────────────────────────────────────────
    if d1_df is not None and DATA_MERGER_AVAILABLE:
        logger.log("   Merging D1 data (leakage-safe, previous day)…")
        try:
            merger = DataMerger()
            df = merger.merge(df, d1_df)
            if hasattr(merger, "validate_no_leakage"):
                ok = merger.validate_no_leakage(df)
                logger.log(f"   Leakage validation: {'✅ PASSED' if ok else '⚠️  WARNING'}")
            logger.log(f"   After merge: {len(df):,} rows, {len(df.columns)} columns")
        except Exception as e:
            logger.log(f"   ⚠️  D1 merge failed ({e}); continuing with H4 only", "WARN")

    # ── Feature engineering ───────────────────────────────────────────────
    if FEATURE_ENGINEERING_AVAILABLE and feats_config.get("compute_additional", True):
        logger.log("   Running feature engineering (H4 + D1 + MTF)…")
        try:
            fe = FeatureEngineering(verbose=False)
            compute_d1  = mtf_config.get("compute_d1_derived", True) and d1_df is not None
            compute_mtf = mtf_config.get("compute_cross_tf",   True) and d1_df is not None
            df = fe.calculate_features(
                df,
                drop_na=True,
                compute_d1_derived=compute_d1,
                compute_cross_tf=compute_mtf,
            )
            logger.log(f"   After engineering: {len(df):,} rows, {len(df.columns)} columns")
        except Exception as e:
            logger.log(f"   ⚠️  Feature engineering failed ({e})", "WARN")
    else:
        logger.log("   Feature engineering skipped (disabled or module not found)")

    return df


def get_feature_columns(df: pd.DataFrame, config: Dict) -> List[str]:
    """Return numeric columns after excluding non-feature cols."""
    feats_config = config.get("features", {})
    exclude = set(c.lower() for c in feats_config.get("exclude_columns", []))

    # Standard non-feature columns
    exclude.update({
        "timestamp", "open", "high", "low", "close", "volume",
        "d1_timestamp", "d1_open", "d1_high", "d1_low", "d1_close", "d1_volume",
        "label", "label_reason", "regime", "pair", "symbol",
    })

    cols = [
        c for c in df.columns
        if c.lower() not in exclude
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    return cols


# =============================================================================
# WALK-FORWARD SPLITS
# =============================================================================

def define_walk_forward_splits(
    df: pd.DataFrame,
    config: Dict,
    logger: PipelineLogger,
) -> List[Dict[str, Any]]:
    wf = config.get("walk_forward", {})
    n_folds    = wf.get("n_folds",            5)
    cal_ratio  = wf.get("calibration_ratio", 0.10)
    thresh_ratio = wf.get("threshold_ratio", 0.10)
    expanding  = wf.get("expanding_window",  True)

    n_rows   = len(df)
    val_size = cal_ratio + thresh_ratio
    boundaries = []

    if expanding:
        segment = n_rows // (n_folds + 1)
        for fold_idx in range(n_folds):
            train_end   = (fold_idx + 1) * segment
            cal_end     = train_end + int(segment * cal_ratio / val_size)
            thresh_end  = min(train_end + segment, n_rows - 1)

            if thresh_end <= train_end:
                continue

            boundaries.append({
                "fold_number":      fold_idx + 1,
                "train_start_idx":  0,
                "train_end_idx":    min(train_end, n_rows - 1),
                "cal_start_idx":    min(train_end, n_rows - 1),
                "cal_end_idx":      min(cal_end,   n_rows - 1),
                "thresh_start_idx": min(cal_end,   n_rows - 1),
                "thresh_end_idx":   min(thresh_end, n_rows - 1),
            })
    else:
        fold_size = n_rows // n_folds
        for fold_idx in range(n_folds):
            start = fold_idx * fold_size
            end   = start + fold_size
            train_end  = start + int(fold_size * (1 - val_size))
            cal_end    = train_end + int(fold_size * cal_ratio)
            thresh_end = min(end, n_rows - 1)

            boundaries.append({
                "fold_number":      fold_idx + 1,
                "train_start_idx":  start,
                "train_end_idx":    min(train_end, n_rows - 1),
                "cal_start_idx":    min(train_end, n_rows - 1),
                "cal_end_idx":      min(cal_end,   n_rows - 1),
                "thresh_start_idx": min(cal_end,   n_rows - 1),
                "thresh_end_idx":   thresh_end,
            })

    logger.log(f"   {len(boundaries)} walk-forward folds defined")
    for b in boundaries:
        train_sz  = b["train_end_idx"]  - b["train_start_idx"]
        thresh_sz = b["thresh_end_idx"] - b["thresh_start_idx"]
        logger.log(f"      Fold {b['fold_number']}: Train={train_sz:,}  Thresh={thresh_sz:,}")

    return boundaries


# =============================================================================
# LOSS ANALYSIS & FILTER RECOMMENDATIONS
# =============================================================================

def run_loss_analysis(
    trade_db_path: str,
    best_config_id: str,
    artifacts_dir: str,
    logger: PipelineLogger,
) -> None:
    if not LOSS_ANALYSIS_AVAILABLE:
        logger.log("   Loss analysis module not available — skipping", "WARN")
        return

    try:
        analyzer = LossAnalyzer(trade_db_path)
        n_trades = analyzer.load_trades()
        logger.log(f"   Loaded {len(n_trades):,} trades for analysis")

        report = analyzer.analyze(config_id=best_config_id)
        analyzer.print_report(report)

        out = Path(artifacts_dir) / "loss_analysis_stella_alpha.json"
        analyzer.save_report(report, str(out))
        logger.log(f"   Loss report → {out}")
    except Exception as e:
        logger.log(f"   ⚠️  Loss analysis failed: {e}", "WARN")


def run_filter_recommendations(
    trade_db_path: str,
    best_tp: int,
    best_sl: int,
    artifacts_dir: str,
    logger: PipelineLogger,
) -> None:
    if not FILTER_ENGINE_AVAILABLE:
        logger.log("   Filter engine not available — skipping", "WARN")
        return

    try:
        import sqlite3
        conn = sqlite3.connect(trade_db_path)
        trades_df = pd.read_sql_query("SELECT * FROM trades", conn)
        conn.close()

        if len(trades_df) < 50:
            logger.log("   Too few trades for filter analysis — skipping", "WARN")
            return

        engine = FilterRecommendationEngine(
            trades_df,
            config={"tp_pips": best_tp, "sl_pips": best_sl},
        )
        recs   = engine.generate_recommendations(top_n=20)
        combos = engine.test_top_combinations(recs, max_filters=3, top_singles=5)

        engine.print_recommendations(recs, combos)

        out = Path(artifacts_dir) / "filter_recommendations.json"
        engine.save_recommendations(recs, str(out), combos)
        logger.log(f"   Filter recommendations → {out}")
    except Exception as e:
        logger.log(f"   ⚠️  Filter recommendation failed: {e}", "WARN")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline(
    config_path: str,
    h4_csv: str,
    d1_csv: Optional[str],
    output_dir: Optional[str],
    n_workers: Optional[int],
    debug: bool = False,
) -> Optional[Dict[str, Any]]:

    pipeline_start = time.time()

    # ── Config ───────────────────────────────────────────────────────────
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    paths = config.get("paths", {})
    artifacts_dir = output_dir or paths.get("artifacts_dir", "artifacts")
    logs_dir      = paths.get("logs_dir", "logs")

    Path(artifacts_dir).mkdir(parents=True, exist_ok=True)
    Path(logs_dir).mkdir(parents=True, exist_ok=True)

    log_ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(logs_dir) / f"training_stella_alpha_{log_ts}.log"
    logger   = PipelineLogger(log_file=str(log_file), verbose=True)

    logger.header("STELLA ALPHA TRAINING PIPELINE")
    logger.log(f"Config       : {config_path}")
    logger.log(f"H4 input     : {h4_csv}")
    logger.log(f"D1 input     : {d1_csv or 'not provided'}")
    logger.log(f"Artifacts dir: {artifacts_dir}")

    pip_value = config.get("schema", {}).get("pip_value", 0.0001)

    try:
        # ══════════════════════════════════════════════════════════════════
        # STEP 1: Load data
        # ══════════════════════════════════════════════════════════════════
        logger.subheader("STEP 1: Loading Data")

        df    = load_h4_data(h4_csv, config, logger)
        d1_df = load_d1_data(d1_csv, config, logger) if d1_csv else None

        # ══════════════════════════════════════════════════════════════════
        # STEP 2: Feature engineering (merge + H4 + D1 + MTF)
        # ══════════════════════════════════════════════════════════════════
        logger.subheader("STEP 2: Feature Engineering")

        df = run_feature_engineering(df, config, logger, d1_df=d1_df)
        feature_columns = get_feature_columns(df, config)
        logger.log(f"   Feature columns: {len(feature_columns)}")

        h4_feat  = [c for c in feature_columns if not c.startswith("d1_") and not c.startswith("mtf_")]
        d1_feat  = [c for c in feature_columns if c.startswith("d1_")]
        mtf_feat = [c for c in feature_columns if c.startswith("mtf_")]
        logger.log(f"   H4={len(h4_feat)}  D1={len(d1_feat)}  MTF={len(mtf_feat)}")

        # ══════════════════════════════════════════════════════════════════
        # STEP 3: Pre-compute labels
        # ══════════════════════════════════════════════════════════════════
        logger.subheader("STEP 3: Pre-computing Labels")

        config_space = config.get("config_space", {})
        label_configs = get_unique_label_configs(config_space)
        logger.log(f"   Label configurations: {len(label_configs)}")

        label_cache = precompute_all_labels(
            df=df,
            label_configs=label_configs,
            pip_value=pip_value,
            direction=TradeDirection.SHORT,
            verbose=True,
        )

        # Show imbalance for first config
        first = label_configs[0]
        precomp  = label_cache.get(*first)
        imb      = estimate_class_imbalance(precomp.stats)
        logger.log(f"   Imbalance (first cfg): {imb['imbalance_ratio']:.1f}:1 | Win rate: {imb['minority_pct']:.1f}%")

        # ══════════════════════════════════════════════════════════════════
        # STEP 4: Walk-forward splits
        # ══════════════════════════════════════════════════════════════════
        logger.subheader("STEP 4: Walk-Forward Splits")
        fold_boundaries = define_walk_forward_splits(df, config, logger)

        # ══════════════════════════════════════════════════════════════════
        # STEP 5: Generate config space
        # ══════════════════════════════════════════════════════════════════
        logger.subheader("STEP 5: Configuration Space")
        configs = generate_config_space(config)
        logger.log(f"   Total configs to test: {len(configs)}")

        # ══════════════════════════════════════════════════════════════════
        # STEP 6: Run experiments
        # ══════════════════════════════════════════════════════════════════
        logger.subheader("STEP 6: Running Experiments")

        workers = n_workers or config.get("parallel", {}).get("max_workers", 8)
        db_path = Path(artifacts_dir) / "pure_ml_stella_alpha.db"
        trade_db_path = str(Path(artifacts_dir) / "trades_stella_alpha.db")

        results = run_all_experiments(
            df=df,
            label_cache=label_cache,
            configs=configs,
            fold_boundaries=fold_boundaries,
            feature_columns=feature_columns,
            settings=config,
            pip_value=pip_value,
            max_workers=workers,
            db_path=str(db_path),
            trade_db_path=trade_db_path,
        )

        passed_results = filter_passed_results(results)
        logger.log(f"   Passed: {len(passed_results)}/{len(results)}")

        # ══════════════════════════════════════════════════════════════════
        # STEP 7: Select best configuration
        # ══════════════════════════════════════════════════════════════════
        logger.subheader("STEP 7: Best Configuration")

        best = select_best_result(results)
        if best is None:
            logger.log("   ❌ No configs passed acceptance criteria.")
            logger.log("   Relax criteria or check data quality.")
            return None

        agg = best.aggregate_metrics
        logger.log(f"   ✅ {best.config.short_str()}")
        logger.log(f"      Tier     : {best.tier_name}")
        logger.log(f"      R:R      : {best.risk_reward_ratio:.2f}:1")
        logger.log(f"      EV       : {agg.ev_mean:+.2f} pips")
        logger.log(f"      Precision: {agg.precision_mean*100:.1f}%")
        logger.log(f"      Edge     : {best.edge_above_breakeven*100:+.1f}% above breakeven")
        logger.log(f"      Features : {len(best.consensus_features)}")
        logger.log(f"      Threshold: {best.consensus_threshold:.2f}")

        # ══════════════════════════════════════════════════════════════════
        # STEP 8: Train final model
        # ══════════════════════════════════════════════════════════════════
        logger.subheader("STEP 8: Training Final Model")

        last_fold = fold_boundaries[-1]

        df_labeled, _ = apply_precomputed_labels(
            df=df,
            label_cache=label_cache,
            tp_pips=best.config.tp_pips,
            sl_pips=best.config.sl_pips,
            max_holding_bars=best.config.max_holding_bars,
        )

        train_df = df_labeled[df_labeled.index <= last_fold["cal_end_idx"]].copy()
        cal_df   = df_labeled[
            (df_labeled.index > last_fold["cal_end_idx"]) &
            (df_labeled.index <= last_fold["thresh_end_idx"])
        ].copy()

        X_train = train_df[best.consensus_features]
        y_train = train_df["label"]
        logger.log(f"   Training rows: {len(train_df):,}")

        final_model = train_model(
            X_train=X_train,
            y_train=y_train,
            feature_columns=best.consensus_features,
            hyperparameters=best.consensus_hyperparameters,
            model_type=get_best_model_type(),
        )

        # Calibrate
        if len(cal_df) > 20 and len(cal_df["label"].unique()) > 1:
            X_cal = cal_df[best.consensus_features]
            y_cal = cal_df["label"]
            final_model, cal_result = calibrate_model(final_model, X_cal, y_cal)
            logger.log(f"   Calibration improvement: {cal_result.improvement_pct:.1f}%")

        # ══════════════════════════════════════════════════════════════════
        # STEP 9: Save artifacts
        # ══════════════════════════════════════════════════════════════════
        logger.subheader("STEP 9: Saving Artifacts")

        # Model
        model_dir = Path(artifacts_dir) / "models"
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / f"stella_{best.config.config_id}.pkl"
        save_model(final_model, str(model_path))
        logger.log(f"   Model saved → {model_path}")

        # Trading config
        trading_config = {
            "version": "Stella Alpha",
            "generated_at": datetime.utcnow().isoformat(),
            "best_config": {
                "config_id":       best.config.config_id,
                "tier":            best.tier,
                "tier_name":       best.tier_name,
                "tp_pips":         best.config.tp_pips,
                "sl_pips":         best.config.sl_pips,
                "max_holding_bars":best.config.max_holding_bars,
                "threshold":       best.consensus_threshold,
                "classification":  getattr(best, "classification", "BRONZE"),
            },
            "performance": {
                "ev_mean":             round(agg.ev_mean, 4),
                "precision_mean":      round(agg.precision_mean, 4),
                "precision_cv":        round(agg.precision_std / max(agg.precision_mean, 1e-6), 4),
                "total_trades":        agg.total_trades,
                "risk_reward_ratio":   round(best.risk_reward_ratio, 3),
                "min_winrate_required":round(best.min_winrate_required, 4),
                "edge_above_breakeven":round(best.edge_above_breakeven, 4),
            },
            "features": best.consensus_features,
            "model_file": str(model_path),
        }

        tc_path = Path(artifacts_dir) / "trading_config_stella_alpha.json"
        with open(tc_path, "w", encoding="utf-8") as f:
            json.dump(trading_config, f, indent=2)
        logger.log(f"   Trading config → {tc_path}")

        # Metrics summary
        sorted_passed = sort_results_by_ranking(passed_results)
        metrics = {
            "run_summary": {
                "total_configs": len(results),
                "passed":        len(passed_results),
                "failed":        len(results) - len(passed_results),
                "pass_rate":     round(len(passed_results) / max(len(results), 1), 4),
            },
            "classifications": {
                "gold":   sum(1 for r in passed_results if getattr(r, "classification", "") == "GOLD"),
                "silver": sum(1 for r in passed_results if getattr(r, "classification", "") == "SILVER"),
                "bronze": sum(1 for r in passed_results if getattr(r, "classification", "") == "BRONZE"),
            },
            "tiers": {
                "tier_0_runners": sum(1 for r in passed_results if r.tier == 0),
                "tier_1_ideal":   sum(1 for r in passed_results if r.tier == 1),
            },
            "best_configs": {
                "by_ev": {
                    "config_id": sorted_passed[0].config.config_id if sorted_passed else None,
                    "ev":        sorted_passed[0].aggregate_metrics.ev_mean if sorted_passed else None,
                },
            },
            "best_performance": agg.to_dict(),
            "execution": {
                "total_time_seconds": round(time.time() - pipeline_start, 1),
                "workers_used":       workers,
            },
        }

        metrics_path = Path(artifacts_dir) / "metrics_stella_alpha.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        logger.log(f"   Metrics → {metrics_path}")

        # Features list
        features_path = Path(artifacts_dir) / "features_stella_alpha.json"
        with open(features_path, "w", encoding="utf-8") as f:
            json.dump({"features": best.consensus_features, "n_features": len(best.consensus_features)}, f, indent=2)
        logger.log(f"   Features → {features_path}")

        # ══════════════════════════════════════════════════════════════════
        # STEP 10: Loss analysis & filter recommendations
        # ══════════════════════════════════════════════════════════════════
        loss_settings = config.get("loss_analysis", {})
        if loss_settings.get("enabled", True) and Path(trade_db_path).exists():
            logger.subheader("STEP 10: Loss Analysis & Filter Recommendations")

            run_loss_analysis(
                trade_db_path=trade_db_path,
                best_config_id=best.config.config_id,
                artifacts_dir=artifacts_dir,
                logger=logger,
            )

            run_filter_recommendations(
                trade_db_path=trade_db_path,
                best_tp=best.config.tp_pips,
                best_sl=best.config.sl_pips,
                artifacts_dir=artifacts_dir,
                logger=logger,
            )

        # ══════════════════════════════════════════════════════════════════
        # COMPLETE
        # ══════════════════════════════════════════════════════════════════
        duration = time.time() - pipeline_start
        logger.header("PIPELINE COMPLETE")
        logger.log(f"Duration  : {duration:.0f}s ({duration/60:.1f}min)")
        logger.log(f"Best      : {best.config.short_str()}")
        logger.log(f"Tier      : {best.tier_name}")
        logger.log(f"EV        : {agg.ev_mean:+.2f} pips/trade")
        logger.log(f"Precision : {agg.precision_mean*100:.1f}%")
        logger.log(f"Artifacts : {artifacts_dir}")

        return {
            "status":           "success",
            "best_config":      best.config.to_dict(),
            "metrics":          agg.to_dict(),
            "artifacts_dir":    str(artifacts_dir),
            "duration_seconds": duration,
        }

    except Exception as e:
        logger.log(f"❌ Pipeline failed: {e}", "ERROR")
        if debug:
            import traceback
            logger.log(traceback.format_exc(), "ERROR")
        return None


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stella Alpha — Multi-Timeframe Pure ML Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pure_ml_stella_alpha.py \\
      -c config/pure_ml_stella_alpha_settings.yaml \\
      -i data/EURUSD_H4_features.csv \\
      --d1 data/EURUSD_D1_features.csv \\
      -w 8

  # H4-only (no D1):
  python run_pure_ml_stella_alpha.py -c config/... -i data/EURUSD_H4.csv

  # Debug (single-threaded, verbose errors):
  python run_pure_ml_stella_alpha.py -c config/... -i data/... --d1 ... -w 1 --debug
""",
    )
    p.add_argument("--config",   "-c", required=True,  help="YAML config file")
    p.add_argument("--input",    "-i", required=True,  help="H4 input CSV")
    p.add_argument("--d1",             default=None,   help="D1 input CSV (optional)")
    p.add_argument("--output",   "-o", default=None,   help="Override artifacts directory")
    p.add_argument("--workers",  "-w", type=int, default=None, help="Parallel workers (default: from config)")
    p.add_argument("--debug",    "-d", action="store_true",   help="Verbose error output")
    p.add_argument("--version",  "-v", action="version", version="Stella Alpha 1.0.0")
    return p.parse_args()


def main():
    args = parse_args()

    if not Path(args.config).exists():
        print(f"ERROR: Config not found: {args.config}")
        sys.exit(1)
    if not Path(args.input).exists():
        print(f"ERROR: H4 input not found: {args.input}")
        sys.exit(1)
    if args.d1 and not Path(args.d1).exists():
        print(f"ERROR: D1 input not found: {args.d1}")
        sys.exit(1)

    result = run_pipeline(
        config_path=args.config,
        h4_csv=args.input,
        d1_csv=args.d1,
        output_dir=args.output,
        n_workers=args.workers,
        debug=args.debug,
    )

    if result:
        print("\n✅ Pipeline completed successfully!")
        print(f"   Artifacts: {result['artifacts_dir']}")
        sys.exit(0)
    else:
        print("\n❌ Pipeline failed. Check logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
