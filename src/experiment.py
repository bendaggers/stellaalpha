"""
Experiment Runner - Stella Alpha.

CHANGES FROM V4:
────────────────
1. IMPORTS
   - Swapped PureMLCheckpointManager → StellaAlphaCheckpointManager
   - Added tier_classification imports (classify_tier, calculate_risk_reward)
   - Added trade_recorder import (TradeRecorder) for loss analysis hooks

2. ACCEPTANCE CRITERIA  (R:R-aware)
   - process_single_config() now passes tp_pips/sl_pips to check_acceptance_criteria()
   - Uses compute_min_winrate() / compute_edge_above_breakeven() from evaluation.py
   - No longer uses fixed min_precision = 0.55; uses edge-above-breakeven instead

3. TRADE RECORDING HOOKS
   - After each fold's threshold evaluation, winning and losing trades are
     optionally recorded to trades_stella_alpha.db via TradeRecorder
   - Controlled by settings['loss_analysis']['enabled'] flag (default True)
   - Recording is best-effort: failures are silently ignored so the main
     experiment pipeline is never blocked

4. TIER CLASSIFICATION
   - Each passing config is classified into Tier 0 (Runner, R:R >= 2.5:1)
     or Tier 1 (Ideal, R:R 1.67-2.49:1) before saving to checkpoint DB
   - _save_result_to_checkpoint() writes tier data to StellaAlphaCheckpointManager

All parallel/sequential execution, live display, ETA, progress bar, and
walk-forward CV logic are UNCHANGED from V4.
"""

import os
import sys
import pickle
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import warnings
from pathlib import Path
import threading

warnings.filterwarnings('ignore')
os.environ['LIGHTGBM_VERBOSITY'] = '-1'

# ---------------------------------------------------------------------------
# Local imports — try both src/ layout and flat layout
# ---------------------------------------------------------------------------
try:
    from pure_ml_labels import (
        apply_precomputed_labels,
        precompute_all_labels,
        get_unique_label_configs,
        estimate_class_imbalance,
        LabelCache,
        LabelStats,
        TradeDirection,
    )
    from checkpoint_db import StellaAlphaCheckpointManager
    from tier_classification import (
        classify_tier,
        calculate_risk_reward,
        calculate_min_winrate,
        Tier,
    )
    from evaluation import (
        compute_metrics,
        optimize_threshold,
        aggregate_fold_metrics,
        check_acceptance_criteria,
        get_consensus_threshold,
        compute_min_winrate,
        compute_edge_above_breakeven,
        AggregateMetrics,
        MetricsBundle,
        ThresholdResult,
    )
    from training import (
        tune_hyperparameters,
        train_model,
        calibrate_model,
        get_best_model_type,
        HyperparameterResult,
    )
except ImportError:
    from src.pure_ml_labels import (
        apply_precomputed_labels,
        precompute_all_labels,
        get_unique_label_configs,
        estimate_class_imbalance,
        LabelCache,
        LabelStats,
        TradeDirection,
    )
    from src.checkpoint_db import StellaAlphaCheckpointManager
    from src.tier_classification import (
        classify_tier,
        calculate_risk_reward,
        calculate_min_winrate,
        Tier,
    )
    from src.evaluation import (
        compute_metrics,
        optimize_threshold,
        aggregate_fold_metrics,
        check_acceptance_criteria,
        get_consensus_threshold,
        compute_min_winrate,
        compute_edge_above_breakeven,
        AggregateMetrics,
        MetricsBundle,
        ThresholdResult,
    )
    from src.training import (
        tune_hyperparameters,
        train_model,
        calibrate_model,
        get_best_model_type,
        HyperparameterResult,
    )

# Trade recorder is optional — loss analysis is a best-effort feature
try:
    from trade_recorder import TradeRecorder
    TRADE_RECORDER_AVAILABLE = True
except ImportError:
    try:
        from src.trade_recorder import TradeRecorder
        TRADE_RECORDER_AVAILABLE = True
    except ImportError:
        TRADE_RECORDER_AVAILABLE = False
        TradeRecorder = None

# RFE — try both V4 naming conventions
try:
    from feature_engineering import rfe_select as _rfe_select
    def _run_rfe(X, y, feature_columns, settings):
        rfe_settings = settings.get('rfe', {})
        result = _rfe_select(
            X_train=X, y_train=y, feature_columns=feature_columns,
            min_features=rfe_settings.get('min_features', 5),
            max_features=rfe_settings.get('max_features', 20),
        )
        return result.selected_features
except ImportError:
    try:
        from features import rfe_select as _rfe_v4
        def _run_rfe(X, y, feature_columns, settings):
            rfe_settings = settings.get('rfe', {})
            result = _rfe_v4(
                X=X, y=y, feature_columns=feature_columns,
                min_features=rfe_settings.get('min_features', 5),
                max_features=rfe_settings.get('max_features', 20),
            )
            return result.selected_features
    except ImportError:
        def _run_rfe(X, y, feature_columns, settings):
            return feature_columns  # fallback: use all features


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ConfigurationSpec:
    """Specification for a single configuration to test."""
    config_id: str
    tp_pips: int
    sl_pips: int
    max_holding_bars: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            'config_id': self.config_id,
            'tp_pips': self.tp_pips,
            'sl_pips': self.sl_pips,
            'max_holding_bars': self.max_holding_bars,
        }

    def short_str(self) -> str:
        return f"TP={self.tp_pips} SL={self.sl_pips} H={self.max_holding_bars}"

    def __hash__(self):
        return hash(self.config_id)

    def __eq__(self, other):
        if not isinstance(other, ConfigurationSpec):
            return False
        return self.config_id == other.config_id


@dataclass
class FoldResult:
    """Results from a single fold."""
    fold_number: int
    config_id: str
    train_rows: int
    calibration_rows: int
    threshold_rows: int
    train_win_rate: float
    selected_features: List[str]
    feature_importances: Dict[str, float]
    hyperparameters: Dict[str, Any]
    precision: float
    recall: float
    f1_score: float
    auc_pr: float
    expected_value: float
    trade_count: int
    win_count: int
    loss_count: int
    threshold: float
    training_time_seconds: float
    status: str = "success"
    error_message: Optional[str] = None

    # Stella Alpha: trade-level data for recording (not persisted to checkpoint)
    _test_df: Optional[Any] = field(default=None, repr=False)
    _y_true: Optional[Any] = field(default=None, repr=False)
    _y_pred: Optional[Any] = field(default=None, repr=False)
    _y_proba: Optional[Any] = field(default=None, repr=False)


@dataclass
class ExperimentResult:
    """Complete result for a configuration."""
    config: ConfigurationSpec
    fold_results: List[FoldResult]
    aggregate_metrics: Optional[AggregateMetrics]
    consensus_features: List[str]
    consensus_hyperparameters: Dict[str, Any]
    consensus_threshold: float
    passed: bool
    rejection_reasons: List[str]
    total_time_seconds: float

    # Stella Alpha additions
    tier: int = -1
    tier_name: str = ""
    risk_reward_ratio: float = 0.0
    min_winrate_required: float = 0.0
    edge_above_breakeven: float = 0.0


@dataclass
class WorkerStatus:
    """Status of a single parallel worker."""
    worker_id: int
    current_config: Optional[str] = None
    start_time: Optional[float] = None
    last_config: Optional[str] = None
    last_result: Optional[str] = None
    last_duration: float = 0.0


# =============================================================================
# CONFIGURATION GENERATION
# =============================================================================

def generate_config_space(settings: Dict[str, Any]) -> List[ConfigurationSpec]:
    """Generate all configurations to test from YAML settings."""
    config_space = settings.get('config_space', {})

    tp_cfg = config_space.get('tp_pips', {'min': 50, 'max': 150, 'step': 10})
    sl_cfg = config_space.get('sl_pips', 30)
    hold_cfg = config_space.get('max_holding_bars', {'min': 12, 'max': 72, 'step': 12})

    tp_values = (
        list(range(tp_cfg['min'], tp_cfg['max'] + 1, tp_cfg['step']))
        if isinstance(tp_cfg, dict) else [tp_cfg]
    )
    sl_values = (
        list(range(sl_cfg['min'], sl_cfg['max'] + 1, sl_cfg['step']))
        if isinstance(sl_cfg, dict) else [sl_cfg]
    )
    hold_values = (
        list(range(hold_cfg['min'], hold_cfg['max'] + 1, hold_cfg['step']))
        if isinstance(hold_cfg, dict) else [hold_cfg]
    )

    configs = []
    config_id = 1
    for tp in tp_values:
        for sl in sl_values:
            for hold in hold_values:
                configs.append(ConfigurationSpec(
                    config_id=f"cfg_{config_id:04d}",
                    tp_pips=tp,
                    sl_pips=sl,
                    max_holding_bars=hold,
                ))
                config_id += 1

    return configs


# =============================================================================
# SINGLE CONFIG PROCESSING  (runs in worker process)
# =============================================================================

def process_single_config(args: Tuple) -> ExperimentResult:
    """
    Process a single configuration — runs in a worker process.

    args = (config_dict, df_pickle, labels_pickle,
            fold_boundaries, feature_columns, settings, pip_value)
    """
    import warnings
    import os
    warnings.filterwarnings('ignore')
    os.environ['LIGHTGBM_VERBOSITY'] = '-1'

    config_dict, df_pickle, labels_pickle, fold_boundaries, feature_columns, settings, pip_value = args

    config = ConfigurationSpec(**config_dict)
    df = pickle.loads(df_pickle)
    label_cache = pickle.loads(labels_pickle)

    start_time = time.time()

    try:
        # ── Apply labels ──────────────────────────────────────────────────
        df_labeled, label_stats = apply_precomputed_labels(
            df=df,
            label_cache=label_cache,
            tp_pips=config.tp_pips,
            sl_pips=config.sl_pips,
            max_holding_bars=config.max_holding_bars,
        )

        fold_results: List[FoldResult] = []
        threshold_fold_results: List[ThresholdResult] = []
        fold_metrics: List[MetricsBundle] = []

        for boundary in fold_boundaries:
            fold_start = time.time()
            fold_number = boundary['fold_number']

            # ── Split ─────────────────────────────────────────────────────
            train_df = df_labeled[
                (df_labeled.index >= boundary['train_start_idx']) &
                (df_labeled.index <= boundary['train_end_idx'])
            ].copy()

            cal_df = df_labeled[
                (df_labeled.index >= boundary['cal_start_idx']) &
                (df_labeled.index <= boundary['cal_end_idx'])
            ].copy()

            thresh_df = df_labeled[
                (df_labeled.index >= boundary['thresh_start_idx']) &
                (df_labeled.index <= boundary['thresh_end_idx'])
            ].copy()

            if len(train_df) < 100 or len(thresh_df) < 30:
                fold_results.append(FoldResult(
                    fold_number=fold_number, config_id=config.config_id,
                    train_rows=len(train_df), calibration_rows=len(cal_df),
                    threshold_rows=len(thresh_df), train_win_rate=0.0,
                    selected_features=[], feature_importances={},
                    hyperparameters={}, precision=0.0, recall=0.0,
                    f1_score=0.0, auc_pr=0.0, expected_value=0.0,
                    trade_count=0, win_count=0, loss_count=0,
                    threshold=0.5, training_time_seconds=0.0,
                    status="skipped", error_message="Insufficient data",
                ))
                continue

            y_train = train_df['label']
            y_cal   = cal_df['label']
            y_thresh = thresh_df['label']

            X_train  = train_df[feature_columns]
            X_cal    = cal_df[feature_columns]
            X_thresh = thresh_df[feature_columns]

            train_win_rate = float(y_train.mean())

            # ── Statistical pre-filter (Stella Alpha Phase 2) ─────────────
            # Only apply if stat_prefilter enabled in settings
            working_features = list(feature_columns)
            stat_settings = settings.get('statistical_prefilter', {})
            if stat_settings.get('enabled', False):
                try:
                    from statistical_validation import apply_statistical_prefilter
                    kept, _ = apply_statistical_prefilter(
                        df=train_df,
                        feature_columns=working_features,
                        label_column='label',
                        p_threshold=stat_settings.get('p_threshold', 0.05),
                        min_effect_size=stat_settings.get('min_effect_size', 0.15),
                        verbose=False,
                    )
                    if len(kept) >= 10:
                        working_features = kept
                except Exception:
                    pass  # fall back to all features

            # ── RFE ───────────────────────────────────────────────────────
            rfe_settings = settings.get('rfe', {})
            if rfe_settings.get('enabled', True):
                selected_features = _run_rfe(X_train, y_train, working_features, settings)
            else:
                selected_features = working_features

            if not selected_features:
                selected_features = working_features[:20]

            X_train_sel  = X_train[selected_features]
            X_cal_sel    = X_cal[selected_features]
            X_thresh_sel = X_thresh[selected_features]

            # ── Hyperparameter tuning ─────────────────────────────────────
            hp_settings = settings.get('hyperparameter_tuning', {})
            if hp_settings.get('enabled', True):
                try:
                    hp_result = tune_hyperparameters(
                        X_train=X_train_sel,
                        y_train=y_train,
                        feature_columns=selected_features,
                        method=hp_settings.get('method', 'randomized'),
                        n_iter=hp_settings.get('n_iter', 20),
                        cv=hp_settings.get('cv', 3),
                    )
                    hyperparameters = hp_result.best_params
                except Exception:
                    hyperparameters = {}
            else:
                hyperparameters = settings.get('model', {}).get('params', {})

            # ── Train model ───────────────────────────────────────────────
            trained_model = train_model(
                X_train=X_train_sel,
                y_train=y_train,
                feature_columns=selected_features,
                hyperparameters=hyperparameters,
            )

            # ── Calibrate ─────────────────────────────────────────────────
            if len(cal_df) > 20 and len(y_cal.unique()) > 1:
                try:
                    trained_model, _ = calibrate_model(
                        trained_model=trained_model,
                        X_cal=X_cal_sel,
                        y_cal=y_cal,
                    )
                except Exception:
                    pass

            # ── Optimize threshold on thresh set ─────────────────────────
            y_thresh_proba = trained_model.get_proba_positive(X_thresh_sel)
            threshold_result = optimize_threshold(
                y_true=y_thresh.values,
                y_proba=y_thresh_proba,
                tp_pips=float(config.tp_pips),
                sl_pips=float(config.sl_pips),
                min_trades=settings.get('acceptance_criteria', {}).get('min_trades_per_fold', 20),
            )
            threshold_fold_results.append(threshold_result)

            # ── Final evaluation ──────────────────────────────────────────
            y_thresh_pred = (y_thresh_proba >= threshold_result.optimal_threshold).astype(int)
            metrics = compute_metrics(
                y_true=y_thresh.values,
                y_pred=y_thresh_pred,
                y_proba=y_thresh_proba,
                tp_pips=float(config.tp_pips),
                sl_pips=float(config.sl_pips),
                threshold=threshold_result.optimal_threshold,
            )
            fold_metrics.append(metrics)

            # ── Feature importances ───────────────────────────────────────
            try:
                fi = dict(zip(
                    selected_features,
                    trained_model.model.feature_importances_
                    if hasattr(trained_model.model, 'feature_importances_')
                    else [1.0] * len(selected_features),
                ))
            except Exception:
                fi = {}

            fr = FoldResult(
                fold_number=fold_number,
                config_id=config.config_id,
                train_rows=len(train_df),
                calibration_rows=len(cal_df),
                threshold_rows=len(thresh_df),
                train_win_rate=train_win_rate,
                selected_features=selected_features,
                feature_importances=fi,
                hyperparameters=hyperparameters,
                precision=metrics.precision,
                recall=metrics.recall,
                f1_score=metrics.f1_score,
                auc_pr=metrics.auc_pr,
                expected_value=metrics.expected_value,
                trade_count=metrics.trade_count,
                win_count=metrics.win_count,
                loss_count=metrics.loss_count,
                threshold=threshold_result.optimal_threshold,
                training_time_seconds=time.time() - fold_start,
                status="success",
            )

            # ── STELLA ALPHA: attach trade data for recording hook ────────
            # Stored on fold result so the main process can record trades
            # after the worker returns. We keep only trade rows to save memory.
            try:
                trade_mask = y_thresh_pred == 1
                if trade_mask.any():
                    fr._test_df = thresh_df[trade_mask].copy()
                    fr._y_true  = y_thresh.values[trade_mask]
                    fr._y_pred  = y_thresh_pred[trade_mask]
                    fr._y_proba = y_thresh_proba[trade_mask]
            except Exception:
                pass

            fold_results.append(fr)

        # ── Aggregate across folds ────────────────────────────────────────
        successful = [fr for fr in fold_results if fr.status == "success"]

        if successful:
            agg = aggregate_fold_metrics(fold_metrics)
            consensus_features = _get_consensus_features(
                [fr.selected_features for fr in successful]
            )
            consensus_threshold = get_consensus_threshold(threshold_fold_results)
            consensus_hp = (
                successful[0].hyperparameters if successful else {}
            )
        else:
            agg = None
            consensus_features = []
            consensus_threshold = 0.5
            consensus_hp = {}

        # ── STELLA ALPHA: R:R-aware acceptance check ──────────────────────
        criteria = settings.get('acceptance_criteria', {})
        if agg:
            passed, reasons = check_acceptance_criteria(
                aggregate_metrics=agg,
                tp_pips=float(config.tp_pips),
                sl_pips=float(config.sl_pips),
                min_edge_above_breakeven=criteria.get('min_edge_above_breakeven', 0.05),
                min_trades_per_fold=criteria.get('min_trades_per_fold', 30),
                min_expected_value=criteria.get('min_expected_value', 0.0),
                max_precision_cv=criteria.get('max_precision_cv', 0.30),
            )
        else:
            passed = False
            reasons = ["No successful folds"]

        # ── STELLA ALPHA: compute tier + R:R metrics ──────────────────────
        rr = calculate_risk_reward(config.tp_pips, config.sl_pips)
        min_wr = calculate_min_winrate(config.tp_pips, config.sl_pips)
        edge = (agg.precision_mean - min_wr) if agg else 0.0

        tier_result = classify_tier(config.tp_pips, config.sl_pips)
        tier_val = tier_result.value if hasattr(tier_result, 'value') else int(tier_result)
        tier_name = f"TIER {tier_val}"

        result = ExperimentResult(
            config=config,
            fold_results=fold_results,
            aggregate_metrics=agg,
            consensus_features=consensus_features,
            consensus_hyperparameters=consensus_hp,
            consensus_threshold=consensus_threshold,
            passed=passed,
            rejection_reasons=reasons,
            total_time_seconds=time.time() - start_time,
            tier=tier_val,
            tier_name=tier_name,
            risk_reward_ratio=rr,
            min_winrate_required=min_wr,
            edge_above_breakeven=edge,
        )

        return result

    except Exception as e:
        import traceback
        return ExperimentResult(
            config=config,
            fold_results=[],
            aggregate_metrics=None,
            consensus_features=[],
            consensus_hyperparameters={},
            consensus_threshold=0.5,
            passed=False,
            rejection_reasons=[f"Exception: {str(e)}", traceback.format_exc()],
            total_time_seconds=time.time() - start_time,
        )


# =============================================================================
# TRADE RECORDING HOOK  (Stella Alpha)
# =============================================================================

def _record_trades_for_result(
    result: ExperimentResult,
    trade_recorder: Optional[Any],
) -> None:
    """
    Record winning and losing trades to trades_stella_alpha.db.

    Called in the main process after a worker returns its result.
    Best-effort: any failure is silently swallowed so the pipeline
    is never blocked by trade recording issues.
    """
    if trade_recorder is None or not TRADE_RECORDER_AVAILABLE:
        return

    try:
        for fold_result in result.fold_results:
            if fold_result.status != "success":
                continue
            if fold_result._test_df is None or fold_result._y_true is None:
                continue

            trade_df = fold_result._test_df
            y_true   = fold_result._y_true
            y_proba  = fold_result._y_proba if fold_result._y_proba is not None else np.zeros(len(y_true))

            config = result.config

            for i, (_, row) in enumerate(trade_df.iterrows()):
                if i >= len(y_true):
                    break
                outcome = 'WIN' if y_true[i] == 1 else 'LOSS'
                pips = config.tp_pips if y_true[i] == 1 else -config.sl_pips

                trade_record = {
                    'config_id': config.config_id,
                    'fold': fold_result.fold_number,
                    'timestamp': str(row.get('timestamp', '')),
                    'outcome': outcome,
                    'pips_result': pips,
                    'model_probability': float(y_proba[i]) if i < len(y_proba) else 0.0,
                    # H4 features
                    'h4_rsi_value': row.get('rsi_value'),
                    'h4_bb_position': row.get('bb_position'),
                    'h4_trend_strength': row.get('trend_strength'),
                    'h4_atr_pct': row.get('atr_pct'),
                    'h4_volume_ratio': row.get('volume_ratio'),
                    'h4_hour': row.get('hour_sin'),
                    'h4_is_london': row.get('is_london'),
                    'h4_is_ny': row.get('is_ny'),
                    'h4_consecutive_bullish': row.get('consecutive_bullish'),
                    'h4_exhaustion_score': row.get('exhaustion_score'),
                    # D1 features
                    'd1_rsi_value': row.get('d1_rsi_value'),
                    'd1_bb_position': row.get('d1_bb_position'),
                    'd1_trend_strength': row.get('d1_trend_strength'),
                    'd1_trend_direction': row.get('d1_trend_direction'),
                    'd1_is_trending_up': row.get('d1_is_trending_up'),
                    'd1_atr_percentile': row.get('d1_atr_percentile'),
                    'd1_consecutive_bullish': row.get('d1_consecutive_bullish'),
                    # MTF features
                    'mtf_confluence_score': row.get('mtf_confluence_score'),
                    'mtf_rsi_aligned': row.get('mtf_rsi_aligned'),
                    'mtf_bb_aligned': row.get('mtf_bb_aligned'),
                    'mtf_trend_aligned': row.get('mtf_trend_aligned'),
                    'd1_supports_short': row.get('d1_supports_short'),
                    'd1_opposes_short': row.get('d1_opposes_short'),
                    'mtf_strong_short_setup': row.get('mtf_strong_short_setup'),
                    'h4_vs_d1_rsi': row.get('h4_vs_d1_rsi'),
                }

                trade_recorder.insert_trade(trade_record)

    except Exception:
        pass  # Never block the pipeline for trade recording


# =============================================================================
# CHECKPOINT SAVE
# =============================================================================

def _save_result_to_checkpoint(
    checkpoint: StellaAlphaCheckpointManager,
    result: ExperimentResult,
) -> None:
    """Save experiment result to StellaAlphaCheckpointManager."""
    try:
        agg = result.aggregate_metrics
        if agg is None:
            ev_mean = ev_std = precision_mean = precision_std = 0.0
            precision_cv = recall_mean = f1_mean = auc_pr_mean = 0.0
            total_trades = 0
        else:
            ev_mean = agg.ev_mean
            ev_std = agg.ev_std
            precision_mean = agg.precision_mean
            precision_std = agg.precision_std
            precision_cv = (
                agg.precision_std / agg.precision_mean
                if agg.precision_mean > 0 else 0.0
            )
            recall_mean = agg.recall_mean
            f1_mean = agg.f1_mean
            auc_pr_mean = agg.auc_pr_mean
            total_trades = agg.total_trades

        # Classification: GOLD / SILVER / BRONZE
        classification = None
        if result.passed:
            if ev_mean >= 5.0 and precision_cv < 0.15:
                classification = 'GOLD'
            elif ev_mean >= 2.0:
                classification = 'SILVER'
            else:
                classification = 'BRONZE'

        checkpoint.save_result(
            config_id=result.config.config_id,
            tp_pips=result.config.tp_pips,
            sl_pips=result.config.sl_pips,
            max_holding_bars=result.config.max_holding_bars,
            status='PASSED' if result.passed else 'FAILED',
            tier=result.tier,
            tier_name=result.tier_name,
            ev_mean=ev_mean,
            ev_std=ev_std,
            precision_mean=precision_mean,
            precision_std=precision_std,
            precision_cv=precision_cv,
            recall_mean=recall_mean,
            f1_mean=f1_mean,
            auc_pr_mean=auc_pr_mean,
            total_trades=total_trades,
            selected_features=result.consensus_features,
            n_features=len(result.consensus_features),
            consensus_threshold=result.consensus_threshold,
            classification=classification,
            rejection_reasons=result.rejection_reasons if not result.passed else None,
            risk_reward_ratio=result.risk_reward_ratio,
            min_winrate_required=result.min_winrate_required,
            edge_above_breakeven=result.edge_above_breakeven,
            execution_time_seconds=result.total_time_seconds,
        )
    except Exception:
        pass


# =============================================================================
# FEATURE CONSENSUS HELPER
# =============================================================================

def _get_consensus_features(
    fold_feature_lists: List[List[str]],
    min_frequency: float = 0.6,
) -> List[str]:
    """Features selected in >= min_frequency fraction of folds."""
    if not fold_feature_lists:
        return []
    n_folds = len(fold_feature_lists)
    counts: Dict[str, int] = {}
    for fl in fold_feature_lists:
        for f in fl:
            counts[f] = counts.get(f, 0) + 1
    threshold = max(1, int(n_folds * min_frequency))
    selected = [f for f, c in counts.items() if c >= threshold]
    selected.sort(key=lambda x: counts[x], reverse=True)
    return selected


# =============================================================================
# LIVE DISPLAY UTILITIES  (unchanged from V4)
# =============================================================================

def _format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        return f"{seconds/3600:.1f}h"


def _create_progress_bar(pct: float, width: int = 25) -> str:
    filled = int(width * pct / 100)
    return '[' + '█' * filled + '░' * (width - filled) + ']'


def _print_live_status(
    completed: int,
    total: int,
    passed: int,
    failed: int,
    worker_status: Dict[int, WorkerStatus],
    start_time: float,
    n_workers: int,
) -> None:
    elapsed = time.time() - start_time
    pct = 100.0 * completed / total if total > 0 else 0.0

    if completed > 0:
        rate = completed / elapsed
        eta = (total - completed) / rate if rate > 0 else 0.0
        avg_time = elapsed / completed
    else:
        eta = avg_time = 0.0

    lines_to_clear = n_workers + 4
    sys.stdout.write(f"\033[{lines_to_clear}A\033[J")

    bar = _create_progress_bar(pct)
    print(
        f"{'='*80}\n"
        f"{bar} {pct:5.1f}% | {completed:,}/{total:,} | "
        f"ETA: {_format_time(eta)} | Avg: {_format_time(avg_time)}/cfg | "
        f"✓:{passed} ✗:{failed}\n"
        f"{'-'*80}"
    )

    for i in range(1, n_workers + 1):
        ws = worker_status.get(i, WorkerStatus(worker_id=i))
        if ws.current_config and ws.start_time:
            current = f"⟳ {ws.current_config} ({_format_time(time.time() - ws.start_time)})"
        else:
            current = "idle"
        last = f"{ws.last_result} {ws.last_config} ({_format_time(ws.last_duration)})" if ws.last_config else "-"
        print(f"W{i}: {current:40} | Last: {last}")

    sys.stdout.flush()


# =============================================================================
# PARALLEL EXPERIMENT RUNNER
# =============================================================================

def run_parallel_experiments(
    df: pd.DataFrame,
    label_cache: LabelCache,
    configs: List[ConfigurationSpec],
    fold_boundaries: List[Dict[str, Any]],
    feature_columns: List[str],
    settings: Dict[str, Any],
    pip_value: float,
    n_workers: int = 8,
    db_path: str = "artifacts/pure_ml_stella_alpha.db",
    trade_db_path: str = "artifacts/trades_stella_alpha.db",
) -> List[ExperimentResult]:
    """
    Run all experiments in parallel with live display, checkpointing,
    and optional trade recording.
    """
    checkpoint = StellaAlphaCheckpointManager(db_path)
    pending_configs = checkpoint.get_pending_configs(configs)

    # ── If all complete, reload from DB ──────────────────────────────────
    if not pending_configs:
        print("All configs already completed. Loading from checkpoint...")
        return _load_results_from_checkpoint(checkpoint, configs)

    total = len(pending_configs)
    completed = passed = failed = 0
    start_time = time.time()
    worker_status: Dict[int, WorkerStatus] = {
        i: WorkerStatus(worker_id=i) for i in range(1, n_workers + 1)
    }

    # ── Initialise trade recorder (best-effort) ───────────────────────────
    trade_recorder = None
    loss_analysis_settings = settings.get('loss_analysis', {})
    if loss_analysis_settings.get('enabled', True) and TRADE_RECORDER_AVAILABLE:
        try:
            trade_recorder = TradeRecorder(trade_db_path)
        except Exception:
            trade_recorder = None

    # ── Pickle shared objects once ────────────────────────────────────────
    df_pickle = pickle.dumps(df)
    labels_pickle = pickle.dumps(label_cache)

    results: List[ExperimentResult] = []

    # ── Print initial blank lines for live display ────────────────────────
    print('\n' * (n_workers + 4))

    config_queue = [
        (i, (
            c.to_dict(), df_pickle, labels_pickle,
            fold_boundaries, feature_columns, settings, pip_value,
        ))
        for i, c in enumerate(pending_configs)
    ]

    future_to_worker: Dict = {}
    worker_to_config: Dict[int, ConfigurationSpec] = {}
    available_workers = list(range(1, n_workers + 1))

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # ── Seed initial batch ────────────────────────────────────────────
        while available_workers and config_queue:
            worker_id = available_workers.pop(0)
            idx, args = config_queue.pop(0)
            future = executor.submit(process_single_config, args)
            future_to_worker[future] = worker_id
            worker_to_config[worker_id] = pending_configs[idx]
            worker_status[worker_id].current_config = pending_configs[idx].short_str()
            worker_status[worker_id].start_time = time.time()

        last_display_time = 0.0

        while future_to_worker:
            # ── Check for completions ─────────────────────────────────────
            done_futures = [f for f in list(future_to_worker.keys()) if f.done()]

            for future in done_futures:
                worker_id = future_to_worker.pop(future)
                config = worker_to_config.pop(worker_id, None)
                duration = time.time() - (worker_status[worker_id].start_time or time.time())

                try:
                    result = future.result()
                    results.append(result)

                    completed += 1
                    if result.passed:
                        passed += 1
                        result_symbol = "✓"
                    else:
                        failed += 1
                        result_symbol = "✗"

                    # ── Save to checkpoint DB ─────────────────────────────
                    _save_result_to_checkpoint(checkpoint, result)

                    # ── STELLA ALPHA: record trades ───────────────────────
                    if trade_recorder is not None:
                        _record_trades_for_result(result, trade_recorder)

                except Exception as e:
                    completed += 1
                    failed += 1
                    result_symbol = "✗"

                config_name = config.short_str() if config else "?"
                worker_status[worker_id].last_config = config_name
                worker_status[worker_id].last_result = result_symbol
                worker_status[worker_id].last_duration = duration
                worker_status[worker_id].current_config = None
                worker_status[worker_id].start_time = None

                # ── Submit next ───────────────────────────────────────────
                if config_queue:
                    idx, args = config_queue.pop(0)
                    new_future = executor.submit(process_single_config, args)
                    future_to_worker[new_future] = worker_id
                    worker_to_config[worker_id] = pending_configs[idx]
                    worker_status[worker_id].current_config = pending_configs[idx].short_str()
                    worker_status[worker_id].start_time = time.time()
                else:
                    available_workers.append(worker_id)

            # ── Live display (every 2 s) ──────────────────────────────────
            now = time.time()
            if now - last_display_time >= 2.0:
                _print_live_status(
                    completed, total, passed, failed,
                    worker_status, start_time, n_workers,
                )
                last_display_time = now

            time.sleep(0.1)

    # ── Final display ─────────────────────────────────────────────────────
    _print_live_status(completed, total, passed, failed,
                       worker_status, start_time, n_workers)
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"COMPLETED in {_format_time(total_time)}")
    print(f"Passed: {passed}/{total} ({100*passed/max(total,1):.1f}%)")
    print(f"Results saved to: {db_path}")
    if trade_recorder is not None:
        print(f"Trades recorded to: {trade_db_path}")
    print(f"{'='*80}\n")

    # Merge completed results from DB with newly computed ones
    all_results = _load_results_from_checkpoint(checkpoint, configs)
    return all_results


# =============================================================================
# SEQUENTIAL FALLBACK
# =============================================================================

def run_all_experiments(
    df: pd.DataFrame,
    label_cache: LabelCache,
    configs: List[ConfigurationSpec],
    fold_boundaries: List[Dict[str, Any]],
    feature_columns: List[str],
    settings: Dict[str, Any],
    pip_value: float,
    progress_callback: Optional[Callable] = None,
    max_workers: int = 8,
    db_path: str = "artifacts/pure_ml_stella_alpha.db",
    trade_db_path: str = "artifacts/trades_stella_alpha.db",
) -> List[ExperimentResult]:
    """
    Run all experiments — uses parallel if workers > 1, sequential otherwise.
    """
    if max_workers > 1:
        return run_parallel_experiments(
            df=df,
            label_cache=label_cache,
            configs=configs,
            fold_boundaries=fold_boundaries,
            feature_columns=feature_columns,
            settings=settings,
            pip_value=pip_value,
            n_workers=max_workers,
            db_path=db_path,
            trade_db_path=trade_db_path,
        )

    # ── Sequential path ───────────────────────────────────────────────────
    checkpoint = StellaAlphaCheckpointManager(db_path)
    pending_configs = checkpoint.get_pending_configs(configs)

    if not pending_configs:
        return _load_results_from_checkpoint(checkpoint, configs)

    trade_recorder = None
    if settings.get('loss_analysis', {}).get('enabled', True) and TRADE_RECORDER_AVAILABLE:
        try:
            trade_recorder = TradeRecorder(trade_db_path)
        except Exception:
            pass

    df_pickle = pickle.dumps(df)
    labels_pickle = pickle.dumps(label_cache)
    results: List[ExperimentResult] = []

    for i, config in enumerate(pending_configs):
        args = (
            config.to_dict(), df_pickle, labels_pickle,
            fold_boundaries, feature_columns, settings, pip_value,
        )
        result = process_single_config(args)
        results.append(result)
        _save_result_to_checkpoint(checkpoint, result)
        if trade_recorder is not None:
            _record_trades_for_result(result, trade_recorder)

        if progress_callback:
            progress_callback(i + 1, len(pending_configs), result.passed)

    return _load_results_from_checkpoint(checkpoint, configs)


# =============================================================================
# CHECKPOINT RELOAD
# =============================================================================

def _load_results_from_checkpoint(
    checkpoint: StellaAlphaCheckpointManager,
    configs: List[ConfigurationSpec],
) -> List[ExperimentResult]:
    """Rebuild ExperimentResult list from checkpoint DB."""
    results = []
    for config in configs:
        try:
            row = checkpoint.get_result(
                config.tp_pips, config.sl_pips, config.max_holding_bars
            )
            if row:
                passed = str(row.get('status', '')).upper() == 'PASSED'
                features = row.get('selected_features', [])
                if isinstance(features, str):
                    try:
                        features = json.loads(features)
                    except Exception:
                        features = []

                agg = None
                if row.get('ev_mean') is not None:
                    agg = AggregateMetrics(
                        precision_mean=row.get('precision_mean', 0.0),
                        precision_std=row.get('precision_std', 0.0),
                        recall_mean=row.get('recall_mean', 0.0),
                        recall_std=0.0,
                        f1_mean=row.get('f1_mean', 0.0),
                        f1_std=0.0,
                        auc_pr_mean=row.get('auc_pr_mean', 0.0),
                        auc_pr_std=0.0,
                        ev_mean=row.get('ev_mean', 0.0),
                        ev_std=row.get('ev_std', 0.0),
                        total_trades=row.get('total_trades', 0),
                        n_folds=5,
                    )

                rejection_reasons = row.get('rejection_reasons', [])
                if isinstance(rejection_reasons, str):
                    try:
                        rejection_reasons = json.loads(rejection_reasons)
                    except Exception:
                        rejection_reasons = [rejection_reasons]

                results.append(ExperimentResult(
                    config=config,
                    fold_results=[],
                    aggregate_metrics=agg,
                    consensus_features=features or [],
                    consensus_hyperparameters={},
                    consensus_threshold=row.get('consensus_threshold', 0.5),
                    passed=passed,
                    rejection_reasons=rejection_reasons or [],
                    total_time_seconds=row.get('execution_time', 0.0),
                    tier=row.get('tier', -1),
                    tier_name=row.get('tier_name', ''),
                    risk_reward_ratio=row.get('risk_reward_ratio', 0.0),
                    min_winrate_required=row.get('min_winrate_required', 0.0),
                    edge_above_breakeven=row.get('edge_above_breakeven', 0.0),
                ))
        except Exception:
            pass

    return results


# =============================================================================
# RESULT FILTERING AND SELECTION
# =============================================================================

def filter_passed_results(results: List[ExperimentResult]) -> List[ExperimentResult]:
    return [r for r in results if r.passed]


def sort_results_by_ranking(results: List[ExperimentResult]) -> List[ExperimentResult]:
    return sorted(
        results,
        key=lambda r: (
            r.aggregate_metrics.ev_mean if r.aggregate_metrics else 0.0,
            r.aggregate_metrics.f1_mean if r.aggregate_metrics else 0.0,
            r.aggregate_metrics.precision_mean if r.aggregate_metrics else 0.0,
        ),
        reverse=True,
    )


def select_best_result(results: List[ExperimentResult]) -> Optional[ExperimentResult]:
    passed = filter_passed_results(results)
    if not passed:
        return None
    return sort_results_by_ranking(passed)[0]
