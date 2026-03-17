"""
STELLA ALPHA - OPTION A: TP/SL OPTIMIZATION
============================================
Test multiple TP/SL configurations to find which is most predictable.

Hypothesis: Smaller/balanced TP/SL might be easier to predict than 100/50.

Usage:
    python option_a_tpsl_test.py --h4 data/EURUSD_H4_features.csv --d1 data/EURUSD_D1_features.csv --workers 6
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
import lightgbm as lgb

from combined_features import CombinedFeatureEngineering, get_feature_columns


# =============================================================================
# CONFIGURATION
# =============================================================================

PIP_VALUE = 0.0001

# TP/SL configurations to test
TPSL_CONFIGS = [
    {"tp": 30, "sl": 30, "name": "30/30 (1:1)"},
    {"tp": 40, "sl": 40, "name": "40/40 (1:1)"},
    {"tp": 50, "sl": 50, "name": "50/50 (1:1)"},
    {"tp": 50, "sl": 25, "name": "50/25 (2:1)"},
    {"tp": 60, "sl": 30, "name": "60/30 (2:1)"},
    {"tp": 75, "sl": 50, "name": "75/50 (1.5:1)"},
    {"tp": 100, "sl": 50, "name": "100/50 (2:1)"},  # Current
    {"tp": 100, "sl": 100, "name": "100/100 (1:1)"},
    {"tp": 30, "sl": 15, "name": "30/15 (2:1)"},
    {"tp": 20, "sl": 20, "name": "20/20 (1:1)"},
]

MAX_HOLD_BARS = 72

# Walk-forward folds (using 3 for speed)
WALK_FORWARD_FOLDS = [
    {"name": "Fold1", "train_end": "2012-01-01", "test_end": "2016-01-01"},
    {"name": "Fold2", "train_end": "2016-01-01", "test_end": "2020-01-01"},
    {"name": "Fold3", "train_end": "2020-01-01", "test_end": "2025-01-01"},
]

N_FEATURES_TO_SELECT = 25


# =============================================================================
# DATA LOADING
# =============================================================================

def load_and_merge_data(h4_path: str, d1_path: str) -> pd.DataFrame:
    """Load and merge H4/D1 data without leakage."""
    df_h4 = pd.read_csv(h4_path)
    df_d1 = pd.read_csv(d1_path)
    
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
    return df


# =============================================================================
# TRADE LABELING (for specific TP/SL)
# =============================================================================

def label_trades_for_config(df: pd.DataFrame, tp: int, sl: int, signal_col: str = "mtf_long_signal") -> pd.DataFrame:
    """Label trades for specific TP/SL configuration."""
    df = df.copy()
    df["trade_label"] = np.nan
    df["trade_pips"] = np.nan
    
    signal_indices = df[df[signal_col] == 1].index.tolist()
    
    for idx in signal_indices:
        if idx + MAX_HOLD_BARS >= len(df):
            continue
        
        entry_price = df.loc[idx, "close"]
        tp_price = entry_price + (tp * PIP_VALUE)
        sl_price = entry_price - (sl * PIP_VALUE)
        
        outcome, pips = None, 0
        for offset in range(1, MAX_HOLD_BARS + 1):
            bar_idx = idx + offset
            if bar_idx >= len(df):
                break
            bar_high, bar_low = df.loc[bar_idx, "high"], df.loc[bar_idx, "low"]
            
            if bar_low <= sl_price:
                outcome, pips = 0, -sl
                break
            if bar_high >= tp_price:
                outcome, pips = 1, tp
                break
        
        if outcome is None:
            final_price = df.loc[min(idx + MAX_HOLD_BARS, len(df) - 1), "close"]
            pips = (final_price - entry_price) / PIP_VALUE
            outcome = 1 if pips > 0 else 0
        
        df.loc[idx, "trade_label"] = outcome
        df.loc[idx, "trade_pips"] = pips
    
    return df


# =============================================================================
# SINGLE CONFIG TEST
# =============================================================================

def test_single_config(args: Tuple) -> Dict:
    """Test a single TP/SL configuration."""
    config, df, feature_cols = args
    tp, sl, name = config["tp"], config["sl"], config["name"]
    
    result = {
        "name": name,
        "tp": tp,
        "sl": sl,
        "status": "success",
        "fold_aucs": [],
        "fold_win_rates": [],
        "total_signals": 0,
        "total_wins": 0,
    }
    
    try:
        # Label trades for this config
        df_labeled = label_trades_for_config(df.copy(), tp, sl)
        
        signals_df = df_labeled[df_labeled["trade_label"].notna()].copy()
        result["total_signals"] = len(signals_df)
        result["total_wins"] = int(signals_df["trade_label"].sum())
        result["baseline_wr"] = result["total_wins"] / result["total_signals"] if result["total_signals"] > 0 else 0
        
        if result["total_signals"] < 500:
            result["status"] = "insufficient_signals"
            return result
        
        # Walk-forward validation
        for fold in WALK_FORWARD_FOLDS:
            train_end = pd.Timestamp(fold["train_end"])
            test_end = pd.Timestamp(fold["test_end"])
            
            train_df = signals_df[signals_df["timestamp"] < train_end].copy()
            test_df = signals_df[(signals_df["timestamp"] >= train_end) & (signals_df["timestamp"] < test_end)].copy()
            
            if len(train_df) < 200 or len(test_df) < 50:
                continue
            
            valid_features = [f for f in feature_cols if f in train_df.columns]
            
            X_train = train_df[valid_features].copy()
            y_train = train_df["trade_label"].values
            X_test = test_df[valid_features].copy()
            y_test = test_df["trade_label"].values
            
            medians = X_train.median()
            X_train = X_train.fillna(medians).replace([np.inf, -np.inf], 0).fillna(0)
            X_test = X_test.fillna(medians).replace([np.inf, -np.inf], 0).fillna(0)
            
            # RFE
            estimator = lgb.LGBMClassifier(n_estimators=50, max_depth=4, learning_rate=0.1, verbose=-1, random_state=42, n_jobs=1)
            n_select = min(N_FEATURES_TO_SELECT, len(valid_features))
            selector = RFE(estimator=estimator, n_features_to_select=n_select, step=5)
            selector.fit(X_train, y_train)
            
            selected = X_train.columns[selector.support_].tolist()
            X_train_sel = X_train[selected]
            X_test_sel = X_test[selected]
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_sel)
            X_test_scaled = scaler.transform(X_test_sel)
            
            model = lgb.LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.05,
                                        min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
                                        random_state=42, verbose=-1, n_jobs=1)
            model.fit(X_train_scaled, y_train)
            
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            try:
                auc = roc_auc_score(y_test, y_proba)
                result["fold_aucs"].append(auc)
                result["fold_win_rates"].append(y_test.mean())
            except:
                pass
        
        if result["fold_aucs"]:
            result["avg_auc"] = np.mean(result["fold_aucs"])
            result["std_auc"] = np.std(result["fold_aucs"])
            result["min_auc"] = np.min(result["fold_aucs"])
            result["max_auc"] = np.max(result["fold_aucs"])
        else:
            result["status"] = "no_valid_folds"
    
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    
    return result


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Option A: TP/SL Optimization")
    parser.add_argument("--h4", required=True, help="Path to H4 CSV")
    parser.add_argument("--d1", required=True, help="Path to D1 CSV")
    parser.add_argument("--workers", "-w", type=int, default=6, help="Parallel workers")
    args = parser.parse_args()
    
    n_workers = min(args.workers, len(TPSL_CONFIGS), multiprocessing.cpu_count())
    
    print(f"\n{'='*70}")
    print(f"  OPTION A: TP/SL OPTIMIZATION")
    print(f"  Testing {len(TPSL_CONFIGS)} configurations to find most predictable")
    print(f"{'='*70}")
    
    # Load data
    print(f"\n  Loading data...")
    df = load_and_merge_data(args.h4, args.d1)
    print(f"  Loaded: {len(df):,} rows")
    
    # Feature engineering
    print(f"  Computing features...")
    fe = CombinedFeatureEngineering(verbose=False)
    df = fe.calculate_all_features(df)
    feature_cols = get_feature_columns(df)
    print(f"  Features: {len(feature_cols)}")
    
    # Test all configs in parallel
    print(f"\n  Testing {len(TPSL_CONFIGS)} TP/SL configurations...")
    print(f"  Using {n_workers} workers, {len(WALK_FORWARD_FOLDS)} folds each")
    print(f"  {'-'*60}")
    
    config_args = [(config, df, feature_cols) for config in TPSL_CONFIGS]
    results = []
    
    start_time = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(test_single_config, args): args[0]["name"] for args in config_args}
        for future in as_completed(futures):
            name = futures[future]
            try:
                result = future.result()
                results.append(result)
                if result.get("status") == "success" and "avg_auc" in result:
                    print(f"    ✓ {result['name']:<20} AUC={result['avg_auc']:.3f} ± {result['std_auc']:.3f}  WR={result['baseline_wr']*100:.1f}%  ({result['total_signals']:,} signals)")
                else:
                    print(f"    ✗ {name}: {result.get('status', 'unknown')}")
            except Exception as e:
                print(f"    ✗ {name}: Error - {e}")
    
    elapsed = time.time() - start_time
    print(f"  {'-'*60}")
    print(f"  Completed in {elapsed:.1f}s")
    
    # Sort by AUC
    valid_results = [r for r in results if r.get("status") == "success" and "avg_auc" in r]
    valid_results.sort(key=lambda x: x["avg_auc"], reverse=True)
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"  RESULTS RANKED BY AUC")
    print(f"{'='*70}")
    print(f"\n  {'Config':<20} {'TP':>6} {'SL':>6} {'R:R':>6} {'AUC':>8} {'±Std':>8} {'WR%':>8} {'Signals':>10}")
    print(f"  {'-'*80}")
    
    for r in valid_results:
        rr = r['tp'] / r['sl']
        marker = " ← BEST" if r == valid_results[0] else ""
        print(f"  {r['name']:<20} {r['tp']:>6} {r['sl']:>6} {rr:>6.1f} {r['avg_auc']:>8.3f} {r['std_auc']:>8.3f} {r['baseline_wr']*100:>7.1f}% {r['total_signals']:>10,}{marker}")
    
    # Best config
    if valid_results:
        best = valid_results[0]
        print(f"\n{'='*70}")
        print(f"  BEST CONFIGURATION")
        print(f"{'='*70}")
        print(f"""
  ╔═══════════════════════════════════════════════════════════════╗
  ║  {best['name']:<57} ║
  ╠═══════════════════════════════════════════════════════════════╣
  ║  TP: {best['tp']} pips                                                ║
  ║  SL: {best['sl']} pips                                                ║
  ║  R:R Ratio: {best['tp']/best['sl']:.1f}:1                                           ║
  ╠═══════════════════════════════════════════════════════════════╣
  ║  Average AUC:    {best['avg_auc']:.3f} ± {best['std_auc']:.3f}                              ║
  ║  AUC Range:      {best['min_auc']:.3f} - {best['max_auc']:.3f}                              ║
  ║  Baseline WR:    {best['baseline_wr']*100:.1f}%                                       ║
  ║  Total Signals:  {best['total_signals']:,}                                       ║
  ╚═══════════════════════════════════════════════════════════════╝
        """)
        
        if best['avg_auc'] >= 0.65:
            print(f"  🎉 TARGET ACHIEVED! AUC >= 0.65")
        elif best['avg_auc'] >= 0.60:
            print(f"  ✅ Good progress! AUC >= 0.60")
        elif best['avg_auc'] >= 0.55:
            print(f"  ⚠️  Moderate AUC. Consider combining with other improvements.")
        else:
            print(f"  ❌ Low AUC. TP/SL optimization alone is not enough.")
    
    print(f"\n{'='*70}")
    print(f"  TEST COMPLETE")
    print(f"{'='*70}")
    
    return valid_results


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
