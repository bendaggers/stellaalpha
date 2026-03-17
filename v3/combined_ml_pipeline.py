"""
STELLA ALPHA - COMBINED ML PIPELINE
====================================
Uses combined_features.py with ALL ~320 features from both sources.

Usage:
    python combined_ml_pipeline.py --h4 data/EURUSD_H4_features.csv --d1 data/EURUSD_D1_features.csv --workers 6
"""

import pandas as pd
import numpy as np
import argparse
from typing import List, Dict, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import warnings
import time

warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_auc_score
import xgboost as xgb

from combined_features import CombinedFeatureEngineering, get_feature_columns


# =============================================================================
# CONFIGURATION
# =============================================================================

TRADE_CONFIG = {"tp_pips": 100, "sl_pips": 50, "max_hold_bars": 72, "direction": "long"}
PIP_VALUE = 0.0001

WALK_FORWARD_FOLDS = [
    {"name": "Fold1", "train_end": "2010-01-01", "test_end": "2013-01-01"},
    {"name": "Fold2", "train_end": "2013-01-01", "test_end": "2016-01-01"},
    {"name": "Fold3", "train_end": "2016-01-01", "test_end": "2019-01-01"},
    {"name": "Fold4", "train_end": "2019-01-01", "test_end": "2022-01-01"},
    {"name": "Fold5", "train_end": "2022-01-01", "test_end": "2025-01-01"},
]

THRESHOLDS = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
N_FEATURES_TO_SELECT = 25


# =============================================================================
# DATA LOADING
# =============================================================================

def load_and_merge_data(h4_path: str, d1_path: str) -> pd.DataFrame:
    """Load and merge H4/D1 data without leakage."""
    print(f"\n{'='*70}")
    print(f"  LOADING DATA")
    print(f"{'='*70}")
    
    df_h4 = pd.read_csv(h4_path)
    df_d1 = pd.read_csv(d1_path)
    print(f"  H4: {len(df_h4):,} rows | D1: {len(df_d1):,} rows")
    
    df_h4["timestamp"] = pd.to_datetime(df_h4["timestamp"])
    df_d1["timestamp"] = pd.to_datetime(df_d1["timestamp"])
    
    df_h4["hour"] = df_h4["timestamp"].dt.hour
    df_h4["day_of_week"] = df_h4["timestamp"].dt.dayofweek
    df_h4["h4_date"] = df_h4["timestamp"].dt.normalize()
    
    df_d1["d1_date"] = df_d1["timestamp"].dt.normalize()
    df_d1["d1_available_date"] = df_d1["d1_date"] + pd.Timedelta(days=1)
    
    d1_cols_to_keep = ["d1_available_date"]
    exclude_cols = ["timestamp", "d1_date", "d1_available_date"]
    for col in df_d1.columns:
        if col not in exclude_cols:
            new_name = f"d1_{col}" if not col.startswith("d1_") else col
            df_d1[new_name] = df_d1[col]
            if new_name not in d1_cols_to_keep:
                d1_cols_to_keep.append(new_name)
    
    df_d1_slim = df_d1[d1_cols_to_keep].copy()
    df_h4 = df_h4.sort_values("timestamp").reset_index(drop=True)
    df_d1_slim = df_d1_slim.sort_values("d1_available_date").reset_index(drop=True)
    
    df = pd.merge_asof(df_h4, df_d1_slim, left_on="h4_date", right_on="d1_available_date", direction="backward")
    print(f"  Merged: {len(df):,} rows")
    print(f"  ✓ No data leakage (using previous day D1)")
    return df


# =============================================================================
# TRADE LABELING
# =============================================================================

def label_trades(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Label signal bars with trade outcomes."""
    print(f"\n{'='*70}")
    print(f"  LABELING TRADES")
    print(f"{'='*70}")
    
    df = df.copy()
    tp, sl, max_hold = config["tp_pips"], config["sl_pips"], config["max_hold_bars"]
    direction = config["direction"]
    signal_col = "mtf_long_signal" if direction == "long" else "mtf_short_signal"
    
    df["trade_label"] = np.nan
    df["trade_pips"] = np.nan
    
    signal_indices = df[df[signal_col] == 1].index.tolist()
    print(f"  Signal: {signal_col} | Total: {len(signal_indices):,}")
    
    wins, labeled = 0, 0
    for idx in signal_indices:
        if idx + max_hold >= len(df):
            continue
        entry_price = df.loc[idx, "close"]
        tp_price = entry_price + (tp * PIP_VALUE * (1 if direction == "long" else -1))
        sl_price = entry_price - (sl * PIP_VALUE * (1 if direction == "long" else -1))
        
        outcome, pips = None, 0
        for offset in range(1, max_hold + 1):
            bar_idx = idx + offset
            if bar_idx >= len(df):
                break
            bar_high, bar_low = df.loc[bar_idx, "high"], df.loc[bar_idx, "low"]
            
            if direction == "long":
                if bar_low <= sl_price:
                    outcome, pips = 0, -sl
                    break
                if bar_high >= tp_price:
                    outcome, pips = 1, tp
                    break
            else:
                if bar_high >= sl_price:
                    outcome, pips = 0, -sl
                    break
                if bar_low <= tp_price:
                    outcome, pips = 1, tp
                    break
        
        if outcome is None:
            final_price = df.loc[min(idx + max_hold, len(df) - 1), "close"]
            pips = (final_price - entry_price) / PIP_VALUE * (1 if direction == "long" else -1)
            outcome = 1 if pips > 0 else 0
        
        df.loc[idx, "trade_label"] = outcome
        df.loc[idx, "trade_pips"] = pips
        labeled += 1
        if outcome == 1:
            wins += 1
    
    print(f"  Labeled: {labeled:,} | Wins: {wins:,} ({100*wins/labeled:.1f}%)")
    return df


# =============================================================================
# SINGLE FOLD PROCESSING
# =============================================================================

def process_single_fold(args: Tuple) -> Dict:
    """Process a single CV fold."""
    fold_config, df, feature_cols, thresholds = args
    fold_name = fold_config["name"]
    train_end = pd.Timestamp(fold_config["train_end"])
    test_end = pd.Timestamp(fold_config["test_end"])
    
    result = {"fold_name": fold_name, "status": "success", "threshold_results": {}}
    
    try:
        train_df = df[df["timestamp"] < train_end].copy()
        test_df = df[(df["timestamp"] >= train_end) & (df["timestamp"] < test_end)].copy()
        
        train_signals = train_df[train_df["trade_label"].notna()].copy()
        test_signals = test_df[test_df["trade_label"].notna()].copy()
        
        if len(train_signals) < 200 or len(test_signals) < 30:
            result["status"] = "insufficient_data"
            result["train_samples"] = len(train_signals)
            result["test_samples"] = len(test_signals)
            return result
        
        result["train_samples"] = len(train_signals)
        result["test_samples"] = len(test_signals)
        
        valid_features = [f for f in feature_cols if f in train_signals.columns]
        
        X_train = train_signals[valid_features].copy()
        y_train = train_signals["trade_label"].values
        pips_train = train_signals["trade_pips"].values
        
        X_test = test_signals[valid_features].copy()
        y_test = test_signals["trade_label"].values
        pips_test = test_signals["trade_pips"].values
        
        medians = X_train.median()
        X_train = X_train.fillna(medians).replace([np.inf, -np.inf], 0).fillna(0)
        X_test = X_test.fillna(medians).replace([np.inf, -np.inf], 0).fillna(0)
        
        # RFE with XGBoost
        estimator = xgb.XGBClassifier(n_estimators=50, max_depth=4, learning_rate=0.1, 
                                       verbosity=0, random_state=42, n_jobs=1, use_label_encoder=False, eval_metric='logloss')
        n_to_select = min(N_FEATURES_TO_SELECT, len(valid_features))
        selector = RFE(estimator=estimator, n_features_to_select=n_to_select, step=5)
        selector.fit(X_train, y_train)
        
        selected_features = X_train.columns[selector.support_].tolist()
        result["n_features_selected"] = len(selected_features)
        result["selected_features"] = selected_features[:10]
        
        X_train_sel = X_train[selected_features]
        X_test_sel = X_test[selected_features]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_sel)
        X_test_scaled = scaler.transform(X_test_sel)
        
        # Train XGBoost model
        model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, min_child_weight=50, 
                                   subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0, 
                                   n_jobs=1, use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train_scaled, y_train)
        
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        try:
            result["auc"] = roc_auc_score(y_test, y_proba)
        except:
            result["auc"] = 0.5
        
        result["baseline_wr"] = y_test.mean()
        result["baseline_pips"] = pips_test.sum()
        result["baseline_avg_pips"] = pips_test.mean()
        
        for threshold in thresholds:
            mask = y_proba >= threshold
            n_trades = mask.sum()
            
            if n_trades < 5:
                result["threshold_results"][threshold] = {"trades": 0, "wins": 0, "win_rate": 0, "total_pips": 0, "avg_pips": 0, "improved": False}
                continue
            
            filtered_y = y_test[mask]
            filtered_pips = pips_test[mask]
            wins = int(filtered_y.sum())
            wr = wins / n_trades
            total_pips = filtered_pips.sum()
            avg_pips = filtered_pips.mean()
            
            result["threshold_results"][threshold] = {
                "trades": int(n_trades), "wins": wins, "win_rate": float(wr),
                "total_pips": float(total_pips), "avg_pips": float(avg_pips),
                "improved": avg_pips > result["baseline_avg_pips"]
            }
    
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    
    return result


# =============================================================================
# PARALLEL VALIDATION
# =============================================================================

def run_parallel_walk_forward(df: pd.DataFrame, feature_cols: List[str], n_workers: int = 6) -> Tuple[List[Dict], Dict]:
    """Run walk-forward validation in parallel."""
    print(f"\n{'='*70}")
    print(f"  WALK-FORWARD VALIDATION (Parallel)")
    print(f"{'='*70}")
    print(f"  Workers: {n_workers} | Folds: {len(WALK_FORWARD_FOLDS)} | Features: {len(feature_cols)}")
    print(f"  Thresholds: {THRESHOLDS} | RFE: {N_FEATURES_TO_SELECT} features")
    
    fold_args = [(fold, df, feature_cols, THRESHOLDS) for fold in WALK_FORWARD_FOLDS]
    results = []
    start_time = time.time()
    
    print(f"\n  Processing folds...")
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_single_fold, args): args[0]["name"] for args in fold_args}
        for future in as_completed(futures):
            fold_name = futures[future]
            try:
                result = future.result()
                results.append(result)
                if result.get("status") == "success":
                    print(f"    ✓ {fold_name}: {result.get('test_samples', 0)} test, AUC={result.get('auc', 0):.3f}, Features={result.get('n_features_selected', 0)}")
                else:
                    print(f"    ✗ {fold_name}: {result.get('status')}")
            except Exception as e:
                print(f"    ✗ {fold_name}: Error - {e}")
                results.append({"fold_name": fold_name, "status": "error", "error": str(e)})
    
    print(f"\n  Completed in {time.time() - start_time:.1f}s")
    
    # Aggregate
    threshold_aggregates = {t: {"trades": 0, "wins": 0, "total_pips": 0, "baseline_pips": 0, "improved_folds": 0, "valid_folds": 0} for t in THRESHOLDS}
    
    for result in results:
        if result.get("status") != "success":
            continue
        baseline_pips = result.get("baseline_pips", 0)
        for threshold, tres in result.get("threshold_results", {}).items():
            if tres.get("trades", 0) > 0:
                threshold_aggregates[threshold]["trades"] += tres["trades"]
                threshold_aggregates[threshold]["wins"] += tres["wins"]
                threshold_aggregates[threshold]["total_pips"] += tres["total_pips"]
                threshold_aggregates[threshold]["baseline_pips"] += baseline_pips
                threshold_aggregates[threshold]["valid_folds"] += 1
                if tres.get("improved", False):
                    threshold_aggregates[threshold]["improved_folds"] += 1
    
    return results, threshold_aggregates


# =============================================================================
# RESULTS DISPLAY
# =============================================================================

def print_results(fold_results: List[Dict], threshold_aggregates: Dict) -> Tuple[float, Dict]:
    """Print results and find best threshold."""
    print(f"\n{'='*80}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*80}")
    
    valid_results = [r for r in fold_results if r.get("status") == "success"]
    
    print(f"\n  PER-FOLD PERFORMANCE:")
    print(f"  {'Fold':<8} {'Train':>8} {'Test':>8} {'Features':>10} {'AUC':>8} {'Baseline WR':>12}")
    print(f"  {'-'*60}")
    for r in sorted(valid_results, key=lambda x: x["fold_name"]):
        print(f"  {r['fold_name']:<8} {r.get('train_samples', 0):>8} {r.get('test_samples', 0):>8} "
              f"{r.get('n_features_selected', 0):>10} {r.get('auc', 0):>8.3f} {r.get('baseline_wr', 0)*100:>11.1f}%")
    
    print(f"\n  THRESHOLD COMPARISON:")
    print(f"  {'Threshold':>10} {'Trades':>10} {'Win Rate':>10} {'Total Pips':>12} {'Avg Pips':>10} {'Improved':>12}")
    print(f"  {'-'*70}")
    
    best_threshold, best_avg_pips = None, -999
    for threshold in THRESHOLDS:
        agg = threshold_aggregates[threshold]
        trades = agg["trades"]
        if trades == 0:
            continue
        wins, wr = agg["wins"], agg["wins"] / trades
        total_pips, avg_pips = agg["total_pips"], agg["total_pips"] / trades
        improved, valid = agg["improved_folds"], agg["valid_folds"]
        
        if avg_pips > best_avg_pips:
            best_avg_pips, best_threshold = avg_pips, threshold
        
        marker = " ← BEST" if threshold == best_threshold else ""
        print(f"  {threshold:>10.2f} {trades:>10} {wr:>10.1%} {total_pips:>+12.0f} {avg_pips:>+10.2f} {improved:>6}/{valid}{marker}")
    
    if best_threshold:
        agg = threshold_aggregates[best_threshold]
        trades, wins = agg["trades"], agg["wins"]
        wr = wins / trades if trades > 0 else 0
        total_pips = agg["total_pips"]
        avg_pips = total_pips / trades if trades > 0 else 0
        improved, valid = agg["improved_folds"], agg["valid_folds"]
        
        print(f"\n{'='*80}")
        print(f"  BEST THRESHOLD: {best_threshold}")
        print(f"{'='*80}")
        print(f"""
  ╔═══════════════════════════════════════════════════════════════╗
  ║  RECOMMENDED SETTINGS                                         ║
  ╠═══════════════════════════════════════════════════════════════╣
  ║  Signal:        MTF Trend Aligned Long                        ║
  ║  ML Threshold:  {best_threshold:.2f}                                          ║
  ║  TP:            {TRADE_CONFIG['tp_pips']} pips                                         ║
  ║  SL:            {TRADE_CONFIG['sl_pips']} pips                                          ║
  ║  Max Hold:      {TRADE_CONFIG['max_hold_bars']} bars                                         ║
  ╠═══════════════════════════════════════════════════════════════╣
  ║  VALIDATION RESULTS                                           ║
  ║  Total Trades:      {trades:>8,}                                    ║
  ║  Win Rate:          {wr:>8.1%}                                    ║
  ║  Total Pips:        {total_pips:>+8,.0f}                                    ║
  ║  Avg Pips/Trade:    {avg_pips:>+8.2f}                                    ║
  ║  Folds Improved:    {improved}/{valid}                                        ║
  ╚═══════════════════════════════════════════════════════════════╝
        """)
        
        if improved >= valid * 0.6:
            print("  ✅ VALIDATED: ML filter improves performance in majority of folds")
        else:
            print("  ⚠️  CAUTION: Inconsistent improvement across folds")
    
    return best_threshold, threshold_aggregates.get(best_threshold, {})


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Stella Alpha - Combined ML Pipeline")
    parser.add_argument("--h4", required=True, help="Path to H4 CSV")
    parser.add_argument("--d1", required=True, help="Path to D1 CSV")
    parser.add_argument("--workers", "-w", type=int, default=6, help="Parallel workers (default: 6)")
    args = parser.parse_args()
    
    n_workers = max(6, min(args.workers, multiprocessing.cpu_count()))
    
    print(f"\n{'='*70}")
    print(f"  STELLA ALPHA - COMBINED ML PIPELINE")
    print(f"  (~320+ features from BOTH sources)")
    print(f"{'='*70}")
    print(f"  Strategy: MTF Trend Aligned Long")
    print(f"  TP: {TRADE_CONFIG['tp_pips']} | SL: {TRADE_CONFIG['sl_pips']} | MaxHold: {TRADE_CONFIG['max_hold_bars']}")
    print(f"  Workers: {n_workers}")
    
    # Load data
    df = load_and_merge_data(args.h4, args.d1)
    
    # Feature engineering
    print(f"\n{'='*70}")
    print(f"  COMBINED FEATURE ENGINEERING")
    print(f"{'='*70}")
    fe = CombinedFeatureEngineering(verbose=True)
    df = fe.calculate_all_features(df)
    
    # Get feature columns
    feature_cols = get_feature_columns(df)
    print(f"\n  📋 Total ML features: {len(feature_cols)}")
    
    h4_feats = [f for f in feature_cols if not f.startswith('d1_') and not f.startswith('mtf_') and not f.startswith('h4_vs_')]
    d1_feats = [f for f in feature_cols if f.startswith('d1_')]
    mtf_feats = [f for f in feature_cols if f.startswith('mtf_') or f.startswith('h4_vs_')]
    print(f"     H4: {len(h4_feats)} | D1: {len(d1_feats)} | MTF: {len(mtf_feats)}")
    
    # Label trades
    df = label_trades(df, TRADE_CONFIG)
    
    # Validate
    fold_results, threshold_aggregates = run_parallel_walk_forward(df, feature_cols, n_workers)
    
    # Results
    best_threshold, best_stats = print_results(fold_results, threshold_aggregates)
    
    print(f"\n{'='*70}")
    print(f"  PIPELINE COMPLETE")
    print(f"{'='*70}")
    
    return best_threshold, best_stats


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
