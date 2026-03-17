"""
STELLA ALPHA - OPTION B: PREDICT EVERY BAR
==========================================
Instead of filtering by signal, predict on EVERY bar.

Hypothesis: The MTF signal might be adding noise, not value.
Let the ML model decide when to trade based purely on features.

Usage:
    python option_b_every_bar.py --h4 data/EURUSD_H4_features.csv --d1 data/EURUSD_D1_features.csv --workers 6
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
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import lightgbm as lgb

from combined_features import CombinedFeatureEngineering, get_feature_columns


# =============================================================================
# CONFIGURATION
# =============================================================================

PIP_VALUE = 0.0001

# Test multiple TP/SL configs
TPSL_CONFIGS = [
    {"tp": 30, "sl": 30, "name": "30/30"},
    {"tp": 50, "sl": 50, "name": "50/50"},
    {"tp": 50, "sl": 25, "name": "50/25"},
    {"tp": 100, "sl": 50, "name": "100/50"},
]

MAX_HOLD_BARS = 72

# Walk-forward folds
WALK_FORWARD_FOLDS = [
    {"name": "Fold1", "train_end": "2012-01-01", "test_end": "2016-01-01"},
    {"name": "Fold2", "train_end": "2016-01-01", "test_end": "2020-01-01"},
    {"name": "Fold3", "train_end": "2020-01-01", "test_end": "2025-01-01"},
]

N_FEATURES_TO_SELECT = 30  # More features since we have more data
SAMPLE_EVERY_N_BARS = 1  # Use every bar (can increase to reduce data size)


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
# LABEL EVERY BAR
# =============================================================================

def label_every_bar(df: pd.DataFrame, tp: int, sl: int, direction: str = "long") -> pd.DataFrame:
    """
    Label EVERY bar (not just signals) with trade outcome.
    
    This creates much more training data but may have more noise.
    """
    df = df.copy()
    df["trade_label"] = np.nan
    df["trade_pips"] = np.nan
    
    mult = 1 if direction == "long" else -1
    
    # Sample every N bars to reduce computation
    indices = range(0, len(df) - MAX_HOLD_BARS, SAMPLE_EVERY_N_BARS)
    
    for idx in indices:
        entry_price = df.loc[idx, "close"]
        tp_price = entry_price + (tp * PIP_VALUE * mult)
        sl_price = entry_price - (sl * PIP_VALUE * mult)
        
        outcome, pips = None, 0
        for offset in range(1, MAX_HOLD_BARS + 1):
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
            final_price = df.loc[min(idx + MAX_HOLD_BARS, len(df) - 1), "close"]
            pips = (final_price - entry_price) / PIP_VALUE * mult
            outcome = 1 if pips > 0 else 0
        
        df.loc[idx, "trade_label"] = outcome
        df.loc[idx, "trade_pips"] = pips
    
    return df


# =============================================================================
# TEST SINGLE CONFIG
# =============================================================================

def test_config_every_bar(args: Tuple) -> Dict:
    """Test predicting every bar for a specific TP/SL config."""
    config, df, feature_cols, direction = args
    tp, sl, name = config["tp"], config["sl"], config["name"]
    
    result = {
        "name": f"{name} ({direction.upper()})",
        "tp": tp,
        "sl": sl,
        "direction": direction,
        "status": "success",
        "fold_aucs": [],
        "fold_details": [],
    }
    
    try:
        # Label every bar
        df_labeled = label_every_bar(df.copy(), tp, sl, direction)
        labeled_df = df_labeled[df_labeled["trade_label"].notna()].copy()
        
        result["total_bars"] = len(labeled_df)
        result["total_wins"] = int(labeled_df["trade_label"].sum())
        result["baseline_wr"] = result["total_wins"] / result["total_bars"] if result["total_bars"] > 0 else 0
        
        if result["total_bars"] < 1000:
            result["status"] = "insufficient_data"
            return result
        
        # Walk-forward validation
        for fold in WALK_FORWARD_FOLDS:
            train_end = pd.Timestamp(fold["train_end"])
            test_end = pd.Timestamp(fold["test_end"])
            
            train_df = labeled_df[labeled_df["timestamp"] < train_end].copy()
            test_df = labeled_df[(labeled_df["timestamp"] >= train_end) & (labeled_df["timestamp"] < test_end)].copy()
            
            if len(train_df) < 500 or len(test_df) < 200:
                continue
            
            valid_features = [f for f in feature_cols if f in train_df.columns]
            
            X_train = train_df[valid_features].copy()
            y_train = train_df["trade_label"].values
            X_test = test_df[valid_features].copy()
            y_test = test_df["trade_label"].values
            pips_test = test_df["trade_pips"].values
            
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
                
                # Test different thresholds
                fold_detail = {"fold": fold["name"], "auc": auc, "thresholds": {}}
                for thresh in [0.5, 0.55, 0.6, 0.65, 0.7]:
                    mask = y_proba >= thresh
                    n_trades = mask.sum()
                    if n_trades >= 10:
                        wins = y_test[mask].sum()
                        wr = wins / n_trades
                        total_pips = pips_test[mask].sum()
                        avg_pips = pips_test[mask].mean()
                        fold_detail["thresholds"][thresh] = {
                            "trades": int(n_trades),
                            "win_rate": float(wr),
                            "avg_pips": float(avg_pips)
                        }
                
                result["fold_details"].append(fold_detail)
                
            except Exception as e:
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
        import traceback
        result["traceback"] = traceback.format_exc()
    
    return result


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Option B: Predict Every Bar")
    parser.add_argument("--h4", required=True, help="Path to H4 CSV")
    parser.add_argument("--d1", required=True, help="Path to D1 CSV")
    parser.add_argument("--workers", "-w", type=int, default=6, help="Parallel workers")
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"  OPTION B: PREDICT EVERY BAR")
    print(f"  No signal filter - let ML decide when to trade")
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
    
    # Create all test configurations
    all_configs = []
    for config in TPSL_CONFIGS:
        all_configs.append((config, df, feature_cols, "long"))
        all_configs.append((config, df, feature_cols, "short"))
    
    n_workers = min(args.workers, len(all_configs), multiprocessing.cpu_count())
    
    print(f"\n  Testing {len(all_configs)} configurations (LONG + SHORT)...")
    print(f"  Using {n_workers} workers, {len(WALK_FORWARD_FOLDS)} folds each")
    print(f"  Sampling every {SAMPLE_EVERY_N_BARS} bar(s)")
    print(f"  {'-'*60}")
    
    results = []
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(test_config_every_bar, args): args[0]["name"] + f" ({args[3]})" for args in all_configs}
        for future in as_completed(futures):
            name = futures[future]
            try:
                result = future.result()
                results.append(result)
                if result.get("status") == "success" and "avg_auc" in result:
                    print(f"    ✓ {result['name']:<20} AUC={result['avg_auc']:.3f} ± {result['std_auc']:.3f}  WR={result['baseline_wr']*100:.1f}%  ({result['total_bars']:,} bars)")
                else:
                    print(f"    ✗ {name}: {result.get('status', 'unknown')}")
                    if "error" in result:
                        print(f"      Error: {result['error']}")
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
    print(f"\n  {'Config':<25} {'AUC':>8} {'±Std':>8} {'Baseline WR':>12} {'Bars':>10}")
    print(f"  {'-'*70}")
    
    for r in valid_results:
        marker = " ← BEST" if r == valid_results[0] else ""
        print(f"  {r['name']:<25} {r['avg_auc']:>8.3f} {r['std_auc']:>8.3f} {r['baseline_wr']*100:>11.1f}% {r['total_bars']:>10,}{marker}")
    
    # Best config details
    if valid_results:
        best = valid_results[0]
        print(f"\n{'='*70}")
        print(f"  BEST CONFIGURATION: {best['name']}")
        print(f"{'='*70}")
        print(f"""
  Average AUC:    {best['avg_auc']:.3f} ± {best['std_auc']:.3f}
  AUC Range:      {best['min_auc']:.3f} - {best['max_auc']:.3f}
  Baseline WR:    {best['baseline_wr']*100:.1f}%
  Total Bars:     {best['total_bars']:,}
        """)
        
        # Show threshold performance for best config
        print(f"  THRESHOLD PERFORMANCE (from folds):")
        print(f"  {'-'*50}")
        
        # Aggregate threshold performance across folds
        thresh_agg = {}
        for fold_detail in best.get("fold_details", []):
            for thresh, data in fold_detail.get("thresholds", {}).items():
                if thresh not in thresh_agg:
                    thresh_agg[thresh] = {"trades": 0, "wins": 0, "pips": 0, "count": 0}
                thresh_agg[thresh]["trades"] += data["trades"]
                thresh_agg[thresh]["wins"] += data["trades"] * data["win_rate"]
                thresh_agg[thresh]["pips"] += data["trades"] * data["avg_pips"]
                thresh_agg[thresh]["count"] += 1
        
        print(f"  {'Threshold':>10} {'Trades':>10} {'Win Rate':>10} {'Avg Pips':>10}")
        for thresh in sorted(thresh_agg.keys()):
            data = thresh_agg[thresh]
            if data["trades"] > 0:
                wr = data["wins"] / data["trades"]
                avg_pips = data["pips"] / data["trades"]
                print(f"  {thresh:>10.2f} {int(data['trades']):>10} {wr*100:>9.1f}% {avg_pips:>+10.2f}")
        
        # Verdict
        print(f"\n  VERDICT:")
        if best['avg_auc'] >= 0.65:
            print(f"  🎉 TARGET ACHIEVED! AUC >= 0.65")
        elif best['avg_auc'] >= 0.60:
            print(f"  ✅ Good progress! AUC >= 0.60")
        elif best['avg_auc'] >= 0.55:
            print(f"  ⚠️  Moderate AUC. May need additional improvements.")
        else:
            print(f"  ❌ Low AUC. Predicting every bar alone is not enough.")
    
    # Compare with signal-filtered approach
    print(f"\n{'='*70}")
    print(f"  COMPARISON: Every Bar vs Signal-Filtered")
    print(f"{'='*70}")
    print(f"""
  PREVIOUS (Signal-Filtered MTF Trend):
  - AUC: ~0.52
  - Only ~10,000 signals over 25 years
  - Constrained by signal definition

  CURRENT (Every Bar):
  - AUC: {best['avg_auc']:.3f} ({"BETTER" if best['avg_auc'] > 0.52 else "WORSE" if best['avg_auc'] < 0.52 else "SAME"})
  - ~{best['total_bars']:,} bars available
  - Pure ML-driven decisions
    """)
    
    print(f"\n{'='*70}")
    print(f"  TEST COMPLETE")
    print(f"{'='*70}")
    
    return valid_results


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
