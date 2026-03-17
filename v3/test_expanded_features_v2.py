"""
STELLA ALPHA - TEST EXPANDED FEATURES (FIXED)
==============================================
Fixed signal generation and added multiple signal types.

Usage:
    python test_expanded_features_v2.py --h4 data/EURUSD_H4_features.csv --d1 data/EURUSD_D1_features.csv --workers 6
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

from expanded_features import ExpandedFeatureEngineering, get_expanded_feature_columns


# =============================================================================
# CONFIGURATION
# =============================================================================

PIP_VALUE = 0.0001
MAX_HOLD_BARS = 72

# TP/SL configurations to test
TPSL_CONFIGS = [
    {"tp": 50, "sl": 50, "name": "50/50 (1:1)"},
    {"tp": 100, "sl": 50, "name": "100/50 (2:1)"},
    {"tp": 50, "sl": 25, "name": "50/25 (2:1)"},
    {"tp": 75, "sl": 50, "name": "75/50 (1.5:1)"},
]

# Walk-forward folds
WALK_FORWARD_FOLDS = [
    {"name": "Fold1", "train_end": "2010-01-01", "test_end": "2014-01-01"},
    {"name": "Fold2", "train_end": "2014-01-01", "test_end": "2018-01-01"},
    {"name": "Fold3", "train_end": "2018-01-01", "test_end": "2022-01-01"},
    {"name": "Fold4", "train_end": "2022-01-01", "test_end": "2025-01-01"},
]

N_FEATURES_TO_SELECT = 40


# =============================================================================
# DATA LOADING
# =============================================================================

def load_and_merge_data(h4_path: str, d1_path: str) -> pd.DataFrame:
    """Load and merge H4/D1 data without leakage."""
    print("  Loading H4 data...")
    df_h4 = pd.read_csv(h4_path)
    print("  Loading D1 data...")
    df_d1 = pd.read_csv(d1_path)
    
    df_h4["timestamp"] = pd.to_datetime(df_h4["timestamp"])
    df_d1["timestamp"] = pd.to_datetime(df_d1["timestamp"])
    
    df_h4["hour"] = df_h4["timestamp"].dt.hour
    df_h4["day_of_week"] = df_h4["timestamp"].dt.dayofweek
    df_h4["h4_date"] = df_h4["timestamp"].dt.normalize()
    
    df_d1["d1_date"] = df_d1["timestamp"].dt.normalize()
    df_d1["d1_available_date"] = df_d1["d1_date"] + pd.Timedelta(days=1)
    
    # Rename D1 columns
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
    
    # Merge
    print("  Merging H4 + D1...")
    df = pd.merge_asof(df_h4, df_d1_slim, left_on="h4_date", right_on="d1_available_date", direction="backward")
    
    return df


# =============================================================================
# SIGNAL GENERATION (FIXED)
# =============================================================================

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Generate trading signals using the computed features."""
    df = df.copy()
    
    # Ensure we have the necessary base data
    if 'ema_21' not in df.columns:
        df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
    if 'ema_50' not in df.columns:
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    # H4 trend direction
    df['h4_trend_up'] = (df['close'] > df['ema_21']).astype(int)
    df['h4_trend_down'] = (df['close'] < df['ema_21']).astype(int)
    
    # D1 trend direction (if available)
    if 'd1_close' in df.columns:
        d1_ema = df['d1_close'].ewm(span=21, adjust=False).mean()
        df['d1_trend_up'] = (df['d1_close'] > d1_ema).astype(int)
        df['d1_trend_down'] = (df['d1_close'] < d1_ema).astype(int)
    else:
        df['d1_trend_up'] = 1
        df['d1_trend_down'] = 0
    
    # MTF Trend Aligned signals
    df['mtf_long_signal'] = ((df['h4_trend_up'] == 1) & (df['d1_trend_up'] == 1)).astype(int)
    df['mtf_short_signal'] = ((df['h4_trend_down'] == 1) & (df['d1_trend_down'] == 1)).astype(int)
    
    # Alternative signals for testing
    # Signal 1: EMA crossover (8/21)
    if 'ema_8' in df.columns and 'ema_21' in df.columns:
        df['ema_cross_long'] = ((df['ema_8'] > df['ema_21']) & (df['ema_8'].shift(1) <= df['ema_21'].shift(1))).astype(int)
        df['ema_cross_short'] = ((df['ema_8'] < df['ema_21']) & (df['ema_8'].shift(1) >= df['ema_21'].shift(1))).astype(int)
    
    # Signal 2: RSI extreme + reversal
    if 'rsi_value' in df.columns:
        df['rsi_oversold_reversal'] = ((df['rsi_value'] > 30) & (df['rsi_value'].shift(1) <= 30)).astype(int)
        df['rsi_overbought_reversal'] = ((df['rsi_value'] < 70) & (df['rsi_value'].shift(1) >= 70)).astype(int)
    
    # Signal 3: Ichimoku TK cross (if available)
    if 'ichimoku_tenkan' in df.columns and 'ichimoku_kijun' in df.columns:
        df['ichimoku_bull'] = ((df['ichimoku_tenkan'] > df['ichimoku_kijun']) & 
                               (df['ichimoku_tenkan'].shift(1) <= df['ichimoku_kijun'].shift(1))).astype(int)
    
    print(f"  Signals generated:")
    print(f"    MTF Long:  {df['mtf_long_signal'].sum():,} signals")
    print(f"    MTF Short: {df['mtf_short_signal'].sum():,} signals")
    
    return df


# =============================================================================
# TRADE LABELING (WITH DIRECTION)
# =============================================================================

def label_trades_for_config(df: pd.DataFrame, tp: int, sl: int, direction: str = "long") -> pd.DataFrame:
    """Label trades for specific TP/SL configuration."""
    df = df.copy()
    df["trade_label"] = np.nan
    df["trade_pips"] = np.nan
    
    # Choose signal column based on direction
    if direction == "long":
        signal_col = "mtf_long_signal"
        mult = 1
    else:
        signal_col = "mtf_short_signal"
        mult = -1
    
    if signal_col not in df.columns:
        return df
    
    signal_indices = df[df[signal_col] == 1].index.tolist()
    
    for idx in signal_indices:
        if idx + MAX_HOLD_BARS >= len(df):
            continue
        
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
            else:  # short
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
# SINGLE CONFIG TEST
# =============================================================================

def test_single_config(args: Tuple) -> Dict:
    """Test a single TP/SL configuration with expanded features."""
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
        "total_signals": 0,
        "total_wins": 0,
        "top_features": [],
    }
    
    try:
        # Label trades for this config
        df_labeled = label_trades_for_config(df.copy(), tp, sl, direction)
        
        signals_df = df_labeled[df_labeled["trade_label"].notna()].copy()
        result["total_signals"] = len(signals_df)
        result["total_wins"] = int(signals_df["trade_label"].sum())
        result["baseline_wr"] = result["total_wins"] / result["total_signals"] if result["total_signals"] > 0 else 0
        
        if result["total_signals"] < 300:
            result["status"] = f"insufficient_signals ({result['total_signals']})"
            return result
        
        feature_importance_sum = {}
        
        # Walk-forward validation
        for fold in WALK_FORWARD_FOLDS:
            train_end = pd.Timestamp(fold["train_end"])
            test_end = pd.Timestamp(fold["test_end"])
            
            train_df = signals_df[signals_df["timestamp"] < train_end].copy()
            test_df = signals_df[(signals_df["timestamp"] >= train_end) & (signals_df["timestamp"] < test_end)].copy()
            
            if len(train_df) < 200 or len(test_df) < 50:
                continue
            
            # Filter to valid features (no NaN in training data)
            valid_features = []
            for f in feature_cols:
                if f in train_df.columns:
                    nan_pct = train_df[f].isna().mean()
                    if nan_pct < 0.1:
                        valid_features.append(f)
            
            if len(valid_features) < 50:
                continue
            
            X_train = train_df[valid_features].copy()
            y_train = train_df["trade_label"].values
            X_test = test_df[valid_features].copy()
            y_test = test_df["trade_label"].values
            
            # Fill NaN
            medians = X_train.median()
            X_train = X_train.fillna(medians).replace([np.inf, -np.inf], 0).fillna(0)
            X_test = X_test.fillna(medians).replace([np.inf, -np.inf], 0).fillna(0)
            
            # RFE feature selection
            estimator = lgb.LGBMClassifier(
                n_estimators=50, max_depth=4, learning_rate=0.1, 
                verbose=-1, random_state=42, n_jobs=1
            )
            n_select = min(N_FEATURES_TO_SELECT, len(valid_features))
            selector = RFE(estimator=estimator, n_features_to_select=n_select, step=10)
            selector.fit(X_train, y_train)
            
            selected = X_train.columns[selector.support_].tolist()
            X_train_sel = X_train[selected]
            X_test_sel = X_test[selected]
            
            # Scale
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_sel)
            X_test_scaled = scaler.transform(X_test_sel)
            
            # Train model
            model = lgb.LGBMClassifier(
                n_estimators=150, 
                max_depth=6, 
                learning_rate=0.05,
                min_child_samples=30, 
                subsample=0.8, 
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42, 
                verbose=-1, 
                n_jobs=1
            )
            model.fit(X_train_scaled, y_train)
            
            # Predict
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate AUC
            try:
                auc = roc_auc_score(y_test, y_proba)
                result["fold_aucs"].append(auc)
                
                # Track feature importance
                for feat, imp in zip(selected, model.feature_importances_):
                    feature_importance_sum[feat] = feature_importance_sum.get(feat, 0) + imp
                
                # Test thresholds
                fold_detail = {
                    "fold": fold["name"],
                    "auc": auc,
                    "train_size": len(train_df),
                    "test_size": len(test_df),
                    "thresholds": {}
                }
                
                for thresh in [0.45, 0.50, 0.55, 0.60, 0.65]:
                    mask = y_proba >= thresh
                    n_trades = mask.sum()
                    if n_trades >= 10:
                        wins = y_test[mask].sum()
                        wr = wins / n_trades
                        fold_detail["thresholds"][thresh] = {
                            "trades": int(n_trades),
                            "win_rate": float(wr)
                        }
                
                result["fold_details"].append(fold_detail)
                
            except Exception as e:
                pass
        
        # Aggregate results
        if result["fold_aucs"]:
            result["avg_auc"] = np.mean(result["fold_aucs"])
            result["std_auc"] = np.std(result["fold_aucs"])
            result["min_auc"] = np.min(result["fold_aucs"])
            result["max_auc"] = np.max(result["fold_aucs"])
            
            # Top features
            sorted_features = sorted(feature_importance_sum.items(), key=lambda x: x[1], reverse=True)
            result["top_features"] = [f[0] for f in sorted_features[:15]]
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
    parser = argparse.ArgumentParser(description="Test Expanded Features V2")
    parser.add_argument("--h4", required=True, help="Path to H4 CSV")
    parser.add_argument("--d1", required=True, help="Path to D1 CSV")
    parser.add_argument("--workers", "-w", type=int, default=4, help="Parallel workers")
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"  TEST EXPANDED FEATURES V2")
    print(f"  Goal: Reach AUC > 0.65")
    print(f"{'='*70}")
    
    # Load data
    print(f"\n  STEP 1: Loading data...")
    df = load_and_merge_data(args.h4, args.d1)
    print(f"  Loaded: {len(df):,} rows")
    
    # Feature engineering
    print(f"\n  STEP 2: Computing expanded features...")
    fe = ExpandedFeatureEngineering(verbose=True)
    df = fe.calculate_all_features(df)
    
    # Generate signals
    print(f"\n  STEP 3: Generating signals...")
    df = generate_signals(df)
    
    feature_cols = get_expanded_feature_columns(df)
    print(f"\n  Total features: {len(feature_cols)}")
    
    # Create test configurations (LONG + SHORT)
    all_configs = []
    for config in TPSL_CONFIGS:
        all_configs.append((config, df, feature_cols, "long"))
        all_configs.append((config, df, feature_cols, "short"))
    
    # Test configs
    print(f"\n  STEP 4: Testing {len(all_configs)} configurations (LONG + SHORT)...")
    print(f"  Using {len(WALK_FORWARD_FOLDS)} folds, {N_FEATURES_TO_SELECT} features per model")
    print(f"  {'-'*60}")
    
    n_workers = min(args.workers, len(all_configs), multiprocessing.cpu_count())
    results = []
    
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(test_single_config, cfg): cfg[0]["name"] + f" ({cfg[3]})" for cfg in all_configs}
        for future in as_completed(futures):
            name = futures[future]
            try:
                result = future.result()
                results.append(result)
                if result.get("status") == "success" and "avg_auc" in result:
                    auc = result['avg_auc']
                    std = result['std_auc']
                    wr = result['baseline_wr']
                    sigs = result['total_signals']
                    
                    status = "🎉 TARGET!" if auc >= 0.65 else "✅ Good" if auc >= 0.60 else "⚠️" if auc >= 0.55 else "❌"
                    print(f"    {status} {result['name']:<25} AUC={auc:.3f} ± {std:.3f}  WR={wr*100:.1f}%  ({sigs:,} signals)")
                else:
                    status_msg = result.get('status', 'unknown')
                    print(f"    ✗ {name}: {status_msg}")
            except Exception as e:
                print(f"    ✗ {name}: Error - {e}")
    
    elapsed = time.time() - start_time
    print(f"  {'-'*60}")
    print(f"  Completed in {elapsed:.1f}s")
    
    # Sort results
    valid_results = [r for r in results if r.get("status") == "success" and "avg_auc" in r]
    valid_results.sort(key=lambda x: x["avg_auc"], reverse=True)
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"  RESULTS SUMMARY (Ranked by AUC)")
    print(f"{'='*70}")
    
    if valid_results:
        print(f"\n  {'Config':<25} {'AUC':>8} {'±Std':>8} {'WR%':>8} {'Signals':>10}")
        print(f"  {'-'*65}")
        
        for r in valid_results:
            status = "🎉" if r['avg_auc'] >= 0.65 else "✅" if r['avg_auc'] >= 0.60 else "⚠️" if r['avg_auc'] >= 0.55 else "❌"
            print(f"  {status} {r['name']:<22} {r['avg_auc']:>8.3f} {r['std_auc']:>8.3f} {r['baseline_wr']*100:>7.1f}% {r['total_signals']:>10,}")
        
        # Best config details
        best = valid_results[0]
        
        print(f"\n{'='*70}")
        print(f"  BEST CONFIGURATION: {best['name']}")
        print(f"{'='*70}")
        
        print(f"""
  ┌─────────────────────────────────────────────────────────────┐
  │  Average AUC:    {best['avg_auc']:.3f} ± {best['std_auc']:.3f}                            │
  │  AUC Range:      {best['min_auc']:.3f} - {best['max_auc']:.3f}                            │
  │  Baseline WR:    {best['baseline_wr']*100:.1f}%                                     │
  │  Total Signals:  {best['total_signals']:,}                                     │
  └─────────────────────────────────────────────────────────────┘
        """)
        
        # Fold details
        print(f"  Per-Fold AUC:")
        for fd in best.get("fold_details", []):
            print(f"    {fd['fold']}: AUC={fd['auc']:.3f} (train={fd['train_size']}, test={fd['test_size']})")
        
        # Top features
        print(f"\n  Top Features (by importance):")
        for i, feat in enumerate(best.get("top_features", [])[:10], 1):
            # Categorize feature
            if 'ema' in feat.lower():
                cat = "EMA"
            elif 'macd' in feat.lower():
                cat = "MACD"
            elif 'stoch' in feat.lower():
                cat = "Stoch"
            elif 'adx' in feat.lower() or 'di_' in feat:
                cat = "ADX"
            elif 'ichimoku' in feat.lower():
                cat = "Ichi"
            elif 'rsi' in feat.lower():
                cat = "RSI"
            elif 'bb' in feat.lower():
                cat = "BB"
            elif 'lag' in feat.lower():
                cat = "Lag"
            else:
                cat = "Other"
            print(f"    {i:2}. [{cat:5}] {feat}")
        
        # Threshold performance
        print(f"\n  Threshold Performance (aggregated):")
        thresh_agg = {}
        for fd in best.get("fold_details", []):
            for thresh, data in fd.get("thresholds", {}).items():
                if thresh not in thresh_agg:
                    thresh_agg[thresh] = {"trades": 0, "wins": 0}
                thresh_agg[thresh]["trades"] += data["trades"]
                thresh_agg[thresh]["wins"] += data["trades"] * data["win_rate"]
        
        print(f"  {'Threshold':>10} {'Trades':>10} {'Win Rate':>10}")
        for thresh in sorted(thresh_agg.keys()):
            data = thresh_agg[thresh]
            if data["trades"] > 0:
                wr = data["wins"] / data["trades"]
                print(f"  {thresh:>10.2f} {int(data['trades']):>10} {wr*100:>9.1f}%")
        
        # Verdict
        print(f"\n  {'='*60}")
        if best['avg_auc'] >= 0.65:
            print(f"  🎉 TARGET ACHIEVED! AUC >= 0.65")
            print(f"  The expanded features helped reach the goal!")
        elif best['avg_auc'] >= 0.60:
            print(f"  ✅ GOOD PROGRESS! AUC = {best['avg_auc']:.3f}")
            print(f"  Close to target, may need more tuning")
        elif best['avg_auc'] >= 0.55:
            print(f"  ⚠️  MODERATE IMPROVEMENT (AUC = {best['avg_auc']:.3f})")
            print(f"  Better than before (~0.545), but still not enough")
        else:
            print(f"  ❌ NO SIGNIFICANT IMPROVEMENT (AUC = {best['avg_auc']:.3f})")
        print(f"  {'='*60}")
        
        print(f"\n  Comparison:")
        print(f"  Previous best: AUC ~0.545 (215 features)")
        print(f"  Current best:  AUC {best['avg_auc']:.3f} ({len(feature_cols)} features)")
        improvement = best['avg_auc'] - 0.545
        print(f"  Improvement:   {'+' if improvement >= 0 else ''}{improvement:.3f}")
    
    else:
        print(f"\n  ❌ No valid results - all configurations failed")
        print(f"\n  Failed configs:")
        for r in results:
            print(f"    - {r.get('name', 'Unknown')}: {r.get('status', 'unknown')}")
    
    print(f"\n{'='*70}")
    print(f"  TEST COMPLETE")
    print(f"{'='*70}")
    
    return valid_results


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
