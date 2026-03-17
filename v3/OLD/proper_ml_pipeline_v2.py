"""
STELLA ALPHA - PROPER ML PIPELINE V2
=====================================

This script uses YOUR feature_engineering.py module with all ~296 features.

REQUIREMENTS:
    1. Copy feature_engineering.py to the v3 folder
    2. Run this script

Usage:
    python proper_ml_pipeline_v2.py --h4 data/EURUSD_H4_features.csv --d1 data/EURUSD_D1_features.csv --workers 6
"""

import pandas as pd
import numpy as np
import argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import warnings
import time
import sys
import os

warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

# Import YOUR feature engineering module
try:
    from feature_engineering import FeatureEngineering
    HAS_FE_MODULE = True
    print("✓ Imported feature_engineering.py module")
except ImportError:
    HAS_FE_MODULE = False
    print("✗ ERROR: feature_engineering.py not found in current directory!")
    print("  Please copy feature_engineering.py to this folder and try again.")
    sys.exit(1)


# =============================================================================
# CONFIGURATION
# =============================================================================

TRADE_CONFIG = {
    "tp_pips": 100,
    "sl_pips": 50,
    "max_hold_bars": 72,
    "direction": "long",
}

PIP_VALUE = 0.0001

# Walk-forward folds
WALK_FORWARD_FOLDS = [
    {"name": "Fold1", "train_end": "2010-01-01", "test_end": "2013-01-01"},
    {"name": "Fold2", "train_end": "2013-01-01", "test_end": "2016-01-01"},
    {"name": "Fold3", "train_end": "2016-01-01", "test_end": "2019-01-01"},
    {"name": "Fold4", "train_end": "2019-01-01", "test_end": "2022-01-01"},
    {"name": "Fold5", "train_end": "2022-01-01", "test_end": "2025-01-01"},
]

# Thresholds to test
THRESHOLDS = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

# RFE settings
N_FEATURES_TO_SELECT = 25


# =============================================================================
# DATA LOADING & MERGING (No Leakage)
# =============================================================================

def load_and_merge_data(h4_path: str, d1_path: str) -> pd.DataFrame:
    """Load H4 and D1 data, merge safely (no leakage)."""
    print(f"\n{'='*70}")
    print(f"  LOADING DATA")
    print(f"{'='*70}")
    
    df_h4 = pd.read_csv(h4_path)
    df_d1 = pd.read_csv(d1_path)
    print(f"  H4: {len(df_h4):,} rows")
    print(f"  D1: {len(df_d1):,} rows")
    
    # Parse timestamps
    df_h4["timestamp"] = pd.to_datetime(df_h4["timestamp"])
    df_d1["timestamp"] = pd.to_datetime(df_d1["timestamp"])
    
    # Add time features to H4
    df_h4["hour"] = df_h4["timestamp"].dt.hour
    df_h4["day_of_week"] = df_h4["timestamp"].dt.dayofweek
    df_h4["h4_date"] = df_h4["timestamp"].dt.normalize()
    
    # D1: shift by 1 day for no leakage (use PREVIOUS day's D1 data)
    df_d1["d1_date"] = df_d1["timestamp"].dt.normalize()
    df_d1["d1_available_date"] = df_d1["d1_date"] + pd.Timedelta(days=1)
    
    # Rename D1 columns with d1_ prefix
    d1_cols_to_keep = ["d1_available_date"]
    exclude_cols = ["timestamp", "d1_date", "d1_available_date"]
    
    for col in df_d1.columns:
        if col not in exclude_cols:
            new_name = f"d1_{col}" if not col.startswith("d1_") else col
            df_d1[new_name] = df_d1[col]
            if new_name not in d1_cols_to_keep:
                d1_cols_to_keep.append(new_name)
    
    df_d1_slim = df_d1[d1_cols_to_keep].copy()
    
    # Merge using merge_asof (backward direction = use most recent PAST D1)
    df_h4 = df_h4.sort_values("timestamp").reset_index(drop=True)
    df_d1_slim = df_d1_slim.sort_values("d1_available_date").reset_index(drop=True)
    
    df = pd.merge_asof(
        df_h4, df_d1_slim,
        left_on="h4_date",
        right_on="d1_available_date",
        direction="backward"
    )
    
    print(f"  Merged: {len(df):,} rows")
    print(f"  ✓ No data leakage (using previous day D1)")
    
    return df


# =============================================================================
# FEATURE ENGINEERING (Using YOUR module)
# =============================================================================

def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Use the FeatureEngineering class from feature_engineering.py"""
    print(f"\n{'='*70}")
    print(f"  FEATURE ENGINEERING (Using feature_engineering.py)")
    print(f"{'='*70}")
    
    fe = FeatureEngineering(verbose=True)
    
    # Calculate all features (H4 derived + D1 derived + Cross-TF)
    df_features = fe.calculate_features(
        df,
        drop_na=False,  # We'll handle NaN ourselves
        compute_h4_derived=True,
        compute_d1_derived=True,
        compute_cross_tf=True,
    )
    
    # Add MTF signal columns if not present
    if 'mtf_long_signal' not in df_features.columns:
        # MTF Long: H4 uptrend AND D1 uptrend
        h4_up = df_features['trend_strength'] > 0.3
        d1_up = df_features.get('d1_trend_strength', pd.Series(0, index=df_features.index)) > 0.3
        df_features['mtf_long_signal'] = (h4_up & d1_up).astype(int)
    
    if 'mtf_short_signal' not in df_features.columns:
        # MTF Short: H4 downtrend AND D1 downtrend
        h4_down = df_features['trend_strength'] < -0.3
        d1_down = df_features.get('d1_trend_strength', pd.Series(0, index=df_features.index)) < -0.3
        df_features['mtf_short_signal'] = (h4_down & d1_down).astype(int)
    
    return df_features


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Get all valid feature columns for ML."""
    exclude = [
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'h4_date', 'd1_available_date', 'd1_timestamp', 
        'd1_open', 'd1_high', 'd1_low', 'd1_close', 'd1_volume',
        'trade_label', 'trade_pips', 'trade_bars',
        'mtf_long_signal', 'mtf_short_signal',
        'session',  # Categorical
    ]
    
    feature_cols = []
    for col in df.columns:
        if col in exclude:
            continue
        # Skip raw D1 OHLCV
        if col.startswith('d1_') and any(x in col for x in ['_open', '_high', '_low', '_close', '_volume']):
            continue
        # Only numeric columns
        if df[col].dtype in [np.float64, np.int64, np.float32, np.int32, bool, np.bool_]:
            # At least 30% non-null
            if df[col].notna().sum() > len(df) * 0.3:
                feature_cols.append(col)
    
    return sorted(feature_cols)


# =============================================================================
# TRADE LABELING
# =============================================================================

def label_trades(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Label signal bars with trade outcomes."""
    print(f"\n{'='*70}")
    print(f"  LABELING TRADES")
    print(f"{'='*70}")
    
    df = df.copy()
    tp = config["tp_pips"]
    sl = config["sl_pips"]
    max_hold = config["max_hold_bars"]
    direction = config["direction"]
    signal_col = "mtf_long_signal" if direction == "long" else "mtf_short_signal"
    
    df["trade_label"] = np.nan
    df["trade_pips"] = np.nan
    
    # Get signal indices
    if signal_col not in df.columns:
        print(f"  ERROR: Signal column '{signal_col}' not found!")
        return df
    
    signal_indices = df[df[signal_col] == 1].index.tolist()
    print(f"  Signal: {signal_col}")
    print(f"  Total signals: {len(signal_indices):,}")
    print(f"  Config: TP={tp} SL={sl} MaxHold={max_hold}")
    
    wins = 0
    labeled = 0
    
    for idx in signal_indices:
        if idx + max_hold >= len(df):
            continue
        
        entry_price = df.loc[idx, "close"]
        tp_price = entry_price + (tp * PIP_VALUE * (1 if direction == "long" else -1))
        sl_price = entry_price - (sl * PIP_VALUE * (1 if direction == "long" else -1))
        
        outcome = None
        pips = 0
        
        for offset in range(1, max_hold + 1):
            bar_idx = idx + offset
            if bar_idx >= len(df):
                break
            
            bar_high = df.loc[bar_idx, "high"]
            bar_low = df.loc[bar_idx, "low"]
            
            if direction == "long":
                if bar_low <= sl_price:
                    outcome = 0
                    pips = -sl
                    break
                if bar_high >= tp_price:
                    outcome = 1
                    pips = tp
                    break
            else:
                if bar_high >= sl_price:
                    outcome = 0
                    pips = -sl
                    break
                if bar_low <= tp_price:
                    outcome = 1
                    pips = tp
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
    
    wr = 100 * wins / labeled if labeled > 0 else 0
    print(f"  Labeled: {labeled:,}")
    print(f"  Wins: {wins:,} ({wr:.1f}%)")
    
    return df


# =============================================================================
# SINGLE FOLD PROCESSING (for parallel execution)
# =============================================================================

def process_single_fold(args: Tuple) -> Dict:
    """Process a single fold - called by parallel workers."""
    fold_config, df, feature_cols, thresholds = args
    
    fold_name = fold_config["name"]
    train_end = pd.Timestamp(fold_config["train_end"])
    test_end = pd.Timestamp(fold_config["test_end"])
    
    result = {
        "fold_name": fold_name,
        "train_end": str(train_end.date()),
        "test_end": str(test_end.date()),
        "status": "success",
        "threshold_results": {},
    }
    
    try:
        # Split data
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
        
        # Prepare data - only use features that exist
        valid_features = [f for f in feature_cols if f in train_signals.columns]
        
        X_train = train_signals[valid_features].copy()
        y_train = train_signals["trade_label"].values
        pips_train = train_signals["trade_pips"].values
        
        X_test = test_signals[valid_features].copy()
        y_test = test_signals["trade_label"].values
        pips_test = test_signals["trade_pips"].values
        
        # Fill NaN with median
        medians = X_train.median()
        X_train = X_train.fillna(medians)
        X_test = X_test.fillna(medians)
        
        # Replace any remaining NaN/Inf
        X_train = X_train.replace([np.inf, -np.inf], 0).fillna(0)
        X_test = X_test.replace([np.inf, -np.inf], 0).fillna(0)
        
        # RFE Feature Selection
        estimator = lgb.LGBMClassifier(
            n_estimators=50, max_depth=4, learning_rate=0.1,
            verbose=-1, random_state=42, n_jobs=1
        )
        
        n_to_select = min(N_FEATURES_TO_SELECT, len(valid_features))
        selector = RFE(estimator=estimator, n_features_to_select=n_to_select, step=5)
        selector.fit(X_train, y_train)
        
        selected_features = X_train.columns[selector.support_].tolist()
        result["n_features_selected"] = len(selected_features)
        result["selected_features"] = selected_features[:10]  # Top 10 for logging
        
        # Use selected features
        X_train_sel = X_train[selected_features]
        X_test_sel = X_test[selected_features]
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_sel)
        X_test_scaled = scaler.transform(X_test_sel)
        
        # Train model
        model = lgb.LGBMClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.05,
            min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1, n_jobs=1
        )
        model.fit(X_train_scaled, y_train)
        
        # Predict probabilities
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate AUC
        try:
            auc = roc_auc_score(y_test, y_proba)
        except:
            auc = 0.5
        result["auc"] = auc
        
        # Baseline metrics
        baseline_wr = y_test.mean()
        baseline_pips = pips_test.sum()
        baseline_avg = pips_test.mean()
        result["baseline_wr"] = baseline_wr
        result["baseline_pips"] = baseline_pips
        result["baseline_avg_pips"] = baseline_avg
        
        # Test each threshold
        for threshold in thresholds:
            mask = y_proba >= threshold
            n_trades = mask.sum()
            
            if n_trades < 5:
                result["threshold_results"][threshold] = {
                    "trades": 0, "wins": 0, "win_rate": 0,
                    "total_pips": 0, "avg_pips": 0, "improved": False
                }
                continue
            
            filtered_y = y_test[mask]
            filtered_pips = pips_test[mask]
            
            wins = int(filtered_y.sum())
            wr = wins / n_trades
            total_pips = filtered_pips.sum()
            avg_pips = filtered_pips.mean()
            improved = avg_pips > baseline_avg
            
            result["threshold_results"][threshold] = {
                "trades": int(n_trades),
                "wins": wins,
                "win_rate": float(wr),
                "total_pips": float(total_pips),
                "avg_pips": float(avg_pips),
                "improved": improved
            }
    
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        import traceback
        result["traceback"] = traceback.format_exc()
    
    return result


# =============================================================================
# PARALLEL WALK-FORWARD VALIDATION
# =============================================================================

def run_parallel_walk_forward(
    df: pd.DataFrame, 
    feature_cols: List[str], 
    n_workers: int = 6
) -> Tuple[List[Dict], Dict]:
    """Run walk-forward validation in parallel."""
    
    print(f"\n{'='*70}")
    print(f"  WALK-FORWARD VALIDATION (Parallel)")
    print(f"{'='*70}")
    print(f"  Workers: {n_workers}")
    print(f"  Folds: {len(WALK_FORWARD_FOLDS)}")
    print(f"  Total Features: {len(feature_cols)}")
    print(f"  Thresholds: {THRESHOLDS}")
    print(f"  RFE selecting: {N_FEATURES_TO_SELECT} features per fold")
    
    # Prepare arguments for each fold
    fold_args = [
        (fold_config, df, feature_cols, THRESHOLDS)
        for fold_config in WALK_FORWARD_FOLDS
    ]
    
    # Run in parallel
    results = []
    start_time = time.time()
    
    print(f"\n  Processing folds...")
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_single_fold, args): args[0]["name"] 
                   for args in fold_args}
        
        for future in as_completed(futures):
            fold_name = futures[future]
            try:
                result = future.result()
                results.append(result)
                
                status = result.get("status", "unknown")
                if status == "success":
                    print(f"    ✓ {fold_name}: {result.get('test_samples', 0)} test samples, "
                          f"AUC={result.get('auc', 0):.3f}, "
                          f"Features={result.get('n_features_selected', 0)}")
                else:
                    print(f"    ✗ {fold_name}: {status}")
                    if "error" in result:
                        print(f"      Error: {result['error']}")
            except Exception as e:
                print(f"    ✗ {fold_name}: Error - {e}")
                results.append({"fold_name": fold_name, "status": "error", "error": str(e)})
    
    elapsed = time.time() - start_time
    print(f"\n  Completed in {elapsed:.1f}s")
    
    # Aggregate threshold results
    threshold_aggregates = {t: {
        "trades": 0, "wins": 0, "total_pips": 0,
        "baseline_pips": 0, "improved_folds": 0, "valid_folds": 0
    } for t in THRESHOLDS}
    
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
    """Print comprehensive results and find best threshold."""
    
    print(f"\n{'='*80}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*80}")
    
    # Per-fold summary
    valid_results = [r for r in fold_results if r.get("status") == "success"]
    
    print(f"\n  PER-FOLD PERFORMANCE:")
    print(f"  {'Fold':<8} {'Train':>8} {'Test':>8} {'Features':>10} {'AUC':>8} {'Baseline WR':>12}")
    print(f"  {'-'*60}")
    
    for r in sorted(valid_results, key=lambda x: x["fold_name"]):
        print(f"  {r['fold_name']:<8} {r.get('train_samples', 0):>8} {r.get('test_samples', 0):>8} "
              f"{r.get('n_features_selected', 0):>10} {r.get('auc', 0):>8.3f} "
              f"{r.get('baseline_wr', 0)*100:>11.1f}%")
    
    # Threshold comparison
    print(f"\n  THRESHOLD COMPARISON (Aggregated across all folds):")
    print(f"  {'Threshold':>10} {'Trades':>10} {'Win Rate':>10} {'Total Pips':>12} "
          f"{'Avg Pips':>10} {'Improved':>12}")
    print(f"  {'-'*70}")
    
    best_threshold = None
    best_avg_pips = -999
    
    for threshold in THRESHOLDS:
        agg = threshold_aggregates[threshold]
        trades = agg["trades"]
        
        if trades == 0:
            print(f"  {threshold:>10.2f} {'No trades':>10}")
            continue
        
        wins = agg["wins"]
        wr = wins / trades
        total_pips = agg["total_pips"]
        avg_pips = total_pips / trades
        improved = agg["improved_folds"]
        valid = agg["valid_folds"]
        
        if avg_pips > best_avg_pips:
            best_avg_pips = avg_pips
            best_threshold = threshold
        
        marker = " ← BEST" if threshold == best_threshold else ""
        print(f"  {threshold:>10.2f} {trades:>10} {wr:>10.1%} {total_pips:>+12.0f} "
              f"{avg_pips:>+10.2f} {improved:>6}/{valid}{marker}")
    
    # Best threshold details
    print(f"\n{'='*80}")
    print(f"  BEST THRESHOLD: {best_threshold}")
    print(f"{'='*80}")
    
    if best_threshold:
        agg = threshold_aggregates[best_threshold]
        trades = agg["trades"]
        wins = agg["wins"]
        wr = wins / trades if trades > 0 else 0
        total_pips = agg["total_pips"]
        avg_pips = total_pips / trades if trades > 0 else 0
        improved = agg["improved_folds"]
        valid = agg["valid_folds"]
        
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
    parser = argparse.ArgumentParser(description="Stella Alpha - Proper ML Pipeline V2")
    parser.add_argument("--h4", required=True, help="Path to H4 CSV file")
    parser.add_argument("--d1", required=True, help="Path to D1 CSV file")
    parser.add_argument("--workers", "-w", type=int, default=6, 
                        help="Number of parallel workers (default: 6, minimum: 6)")
    args = parser.parse_args()
    
    # Ensure minimum 6 workers
    n_workers = max(6, min(args.workers, multiprocessing.cpu_count()))
    
    print(f"\n{'='*70}")
    print(f"  STELLA ALPHA - PROPER ML PIPELINE V2")
    print(f"  (Using feature_engineering.py with ~296 features)")
    print(f"{'='*70}")
    print(f"  Strategy: MTF Trend Aligned Long")
    print(f"  TP: {TRADE_CONFIG['tp_pips']} pips | SL: {TRADE_CONFIG['sl_pips']} pips | MaxHold: {TRADE_CONFIG['max_hold_bars']} bars")
    print(f"  Workers: {n_workers}")
    print(f"  Thresholds to test: {THRESHOLDS}")
    print(f"  RFE will select: {N_FEATURES_TO_SELECT} features per fold")
    
    # Load and merge data
    df = load_and_merge_data(args.h4, args.d1)
    
    # Engineer ALL features using YOUR module
    df = engineer_all_features(df)
    
    # Get feature columns
    feature_cols = get_feature_columns(df)
    print(f"\n  📋 Total ML features available: {len(feature_cols)}")
    
    # Show feature breakdown
    h4_feats = [f for f in feature_cols if not f.startswith('d1_') and not f.startswith('mtf_') and not f.startswith('h4_vs_')]
    d1_feats = [f for f in feature_cols if f.startswith('d1_')]
    mtf_feats = [f for f in feature_cols if f.startswith('mtf_') or f.startswith('h4_vs_')]
    print(f"     H4 features: {len(h4_feats)}")
    print(f"     D1 features: {len(d1_feats)}")
    print(f"     MTF features: {len(mtf_feats)}")
    
    # Label trades
    df = label_trades(df, TRADE_CONFIG)
    
    # Run parallel walk-forward validation
    fold_results, threshold_aggregates = run_parallel_walk_forward(
        df, feature_cols, n_workers
    )
    
    # Print results
    best_threshold, best_stats = print_results(fold_results, threshold_aggregates)
    
    print(f"\n{'='*70}")
    print(f"  PIPELINE COMPLETE")
    print(f"{'='*70}")
    
    return best_threshold, best_stats


if __name__ == "__main__":
    # Required for Windows multiprocessing
    multiprocessing.freeze_support()
    main()
