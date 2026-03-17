"""
STELLA ALPHA - FULL TEST EXPANDED FEATURES
===========================================
Full walk-forward test with all folds (non-parallel for stability)

Usage:
    python full_test_expanded.py --h4 data/EURUSD_H4_features.csv --d1 data/EURUSD_D1_features.csv
"""

import pandas as pd
import numpy as np
import argparse
import warnings
import time

warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

from expanded_features import ExpandedFeatureEngineering, get_expanded_feature_columns


PIP_VALUE = 0.0001
MAX_HOLD_BARS = 72
N_FEATURES_TO_SELECT = 40

# Walk-forward folds
FOLDS = [
    {"name": "Fold1", "train_end": "2010-01-01", "test_end": "2014-01-01"},
    {"name": "Fold2", "train_end": "2014-01-01", "test_end": "2018-01-01"},
    {"name": "Fold3", "train_end": "2018-01-01", "test_end": "2022-01-01"},
    {"name": "Fold4", "train_end": "2022-01-01", "test_end": "2025-01-01"},
]

# TP/SL configs
CONFIGS = [
    {"tp": 100, "sl": 50, "name": "100/50", "direction": "long"},
    {"tp": 100, "sl": 50, "name": "100/50", "direction": "short"},
    {"tp": 50, "sl": 50, "name": "50/50", "direction": "long"},
    {"tp": 50, "sl": 50, "name": "50/50", "direction": "short"},
    {"tp": 50, "sl": 25, "name": "50/25", "direction": "long"},
    {"tp": 75, "sl": 50, "name": "75/50", "direction": "long"},
]


def load_and_merge_data(h4_path: str, d1_path: str) -> pd.DataFrame:
    """Load and merge H4/D1 data."""
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


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Generate trading signals."""
    df = df.copy()
    
    if 'ema_21' not in df.columns:
        df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
    
    df['h4_trend_up'] = (df['close'] > df['ema_21']).astype(int)
    df['h4_trend_down'] = (df['close'] < df['ema_21']).astype(int)
    
    if 'd1_close' in df.columns:
        d1_ema = df['d1_close'].ewm(span=21, adjust=False).mean()
        df['d1_trend_up'] = (df['d1_close'] > d1_ema).astype(int)
        df['d1_trend_down'] = (df['d1_close'] < d1_ema).astype(int)
    else:
        df['d1_trend_up'] = 1
        df['d1_trend_down'] = 0
    
    df['mtf_long_signal'] = ((df['h4_trend_up'] == 1) & (df['d1_trend_up'] == 1)).astype(int)
    df['mtf_short_signal'] = ((df['h4_trend_down'] == 1) & (df['d1_trend_down'] == 1)).astype(int)
    
    return df


def label_trades(df: pd.DataFrame, tp: int, sl: int, direction: str) -> pd.DataFrame:
    """Label trades."""
    df = df.copy()
    df["trade_label"] = np.nan
    df["trade_pips"] = np.nan
    
    signal_col = "mtf_long_signal" if direction == "long" else "mtf_short_signal"
    mult = 1 if direction == "long" else -1
    
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


def test_config(df, feature_cols, config):
    """Test a single configuration across all folds."""
    tp, sl = config["tp"], config["sl"]
    direction = config["direction"]
    name = f"{config['name']} ({direction.upper()})"
    
    # Label trades
    df_labeled = label_trades(df.copy(), tp, sl, direction)
    signals_df = df_labeled[df_labeled["trade_label"].notna()].copy()
    
    if len(signals_df) < 500:
        return {"name": name, "status": "insufficient_signals", "signals": len(signals_df)}
    
    baseline_wr = signals_df["trade_label"].mean()
    
    fold_aucs = []
    fold_details = []
    all_features = {}
    
    for fold in FOLDS:
        train_end = pd.Timestamp(fold["train_end"])
        test_end = pd.Timestamp(fold["test_end"])
        
        train_df = signals_df[signals_df["timestamp"] < train_end].copy()
        test_df = signals_df[(signals_df["timestamp"] >= train_end) & (signals_df["timestamp"] < test_end)].copy()
        
        if len(train_df) < 200 or len(test_df) < 50:
            continue
        
        # Get valid features
        valid_features = [f for f in feature_cols if f in train_df.columns]
        
        X_train = train_df[valid_features].copy()
        y_train = train_df["trade_label"].values
        X_test = test_df[valid_features].copy()
        y_test = test_df["trade_label"].values
        
        # Clean data
        medians = X_train.median()
        X_train = X_train.fillna(medians).replace([np.inf, -np.inf], 0).fillna(0)
        X_test = X_test.fillna(medians).replace([np.inf, -np.inf], 0).fillna(0)
        
        # RFE
        try:
            estimator = lgb.LGBMClassifier(n_estimators=50, max_depth=4, learning_rate=0.1, verbose=-1, random_state=42, n_jobs=1)
            selector = RFE(estimator=estimator, n_features_to_select=N_FEATURES_TO_SELECT, step=10)
            selector.fit(X_train, y_train)
            
            selected = X_train.columns[selector.support_].tolist()
            X_train_sel = X_train[selected]
            X_test_sel = X_test[selected]
            
            # Scale
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_sel)
            X_test_scaled = scaler.transform(X_test_sel)
            
            # Train
            model = lgb.LGBMClassifier(
                n_estimators=150, max_depth=6, learning_rate=0.05,
                min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=0.1,
                random_state=42, verbose=-1, n_jobs=1
            )
            model.fit(X_train_scaled, y_train)
            
            # Predict
            y_proba = model.predict_proba(X_test_scaled)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
            
            fold_aucs.append(auc)
            
            # Track features
            for feat, imp in zip(selected, model.feature_importances_):
                all_features[feat] = all_features.get(feat, 0) + imp
            
            # Threshold analysis
            thresh_results = {}
            for thresh in [0.50, 0.55, 0.60]:
                mask = y_proba >= thresh
                n = mask.sum()
                if n >= 10:
                    wr = y_test[mask].mean()
                    thresh_results[thresh] = {"trades": n, "wr": wr}
            
            fold_details.append({
                "fold": fold["name"],
                "auc": auc,
                "train": len(train_df),
                "test": len(test_df),
                "thresholds": thresh_results
            })
            
        except Exception as e:
            continue
    
    if not fold_aucs:
        return {"name": name, "status": "no_valid_folds"}
    
    # Sort features
    top_features = sorted(all_features.items(), key=lambda x: x[1], reverse=True)[:10]
    
    return {
        "name": name,
        "status": "success",
        "signals": len(signals_df),
        "baseline_wr": baseline_wr,
        "avg_auc": np.mean(fold_aucs),
        "std_auc": np.std(fold_aucs),
        "min_auc": np.min(fold_aucs),
        "max_auc": np.max(fold_aucs),
        "fold_aucs": fold_aucs,
        "fold_details": fold_details,
        "top_features": top_features
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h4", required=True)
    parser.add_argument("--d1", required=True)
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"  FULL TEST: EXPANDED FEATURES (362 features)")
    print(f"  Goal: Reach AUC > 0.65")
    print(f"{'='*70}")
    
    # Load data
    print(f"\n  Loading data...")
    df = load_and_merge_data(args.h4, args.d1)
    print(f"  Loaded: {len(df):,} rows")
    
    # Feature engineering
    print(f"\n  Computing features...")
    fe = ExpandedFeatureEngineering(verbose=False)
    df = fe.calculate_all_features(df)
    
    # Generate signals
    df = generate_signals(df)
    print(f"  MTF Long: {df['mtf_long_signal'].sum():,} | MTF Short: {df['mtf_short_signal'].sum():,}")
    
    feature_cols = get_expanded_feature_columns(df)
    print(f"  Features: {len(feature_cols)}")
    
    # Test all configs
    print(f"\n  Testing {len(CONFIGS)} configurations across {len(FOLDS)} folds...")
    print(f"  {'-'*60}")
    
    results = []
    start = time.time()
    
    for config in CONFIGS:
        result = test_config(df, feature_cols, config)
        results.append(result)
        
        if result["status"] == "success":
            auc = result["avg_auc"]
            status = "🎉" if auc >= 0.65 else "✅" if auc >= 0.60 else "⚠️" if auc >= 0.55 else "❌"
            print(f"    {status} {result['name']:<20} AUC={auc:.3f} ± {result['std_auc']:.3f}  WR={result['baseline_wr']*100:.1f}%")
        else:
            print(f"    ✗ {result['name']:<20} {result['status']}")
    
    elapsed = time.time() - start
    print(f"  {'-'*60}")
    print(f"  Completed in {elapsed:.1f}s")
    
    # Sort by AUC
    valid = [r for r in results if r["status"] == "success"]
    valid.sort(key=lambda x: x["avg_auc"], reverse=True)
    
    # Summary
    print(f"\n{'='*70}")
    print(f"  RESULTS RANKED BY AUC")
    print(f"{'='*70}")
    
    print(f"\n  {'Config':<25} {'AUC':>8} {'±Std':>8} {'Min':>8} {'Max':>8} {'WR%':>8}")
    print(f"  {'-'*70}")
    
    for r in valid:
        status = "🎉" if r["avg_auc"] >= 0.65 else "✅" if r["avg_auc"] >= 0.60 else "⚠️" if r["avg_auc"] >= 0.55 else "❌"
        print(f"  {status} {r['name']:<22} {r['avg_auc']:>8.3f} {r['std_auc']:>8.3f} {r['min_auc']:>8.3f} {r['max_auc']:>8.3f} {r['baseline_wr']*100:>7.1f}%")
    
    # Best config
    if valid:
        best = valid[0]
        print(f"\n{'='*70}")
        print(f"  BEST: {best['name']}")
        print(f"{'='*70}")
        
        print(f"\n  Average AUC: {best['avg_auc']:.3f} ± {best['std_auc']:.3f}")
        print(f"  Per-fold: {[f'{a:.3f}' for a in best['fold_aucs']]}")
        
        print(f"\n  Top 10 Features:")
        for i, (feat, imp) in enumerate(best["top_features"], 1):
            # Categorize
            if 'ema' in feat.lower(): cat = "EMA"
            elif 'macd' in feat.lower(): cat = "MACD"
            elif 'stoch' in feat.lower(): cat = "Stoch"
            elif 'adx' in feat.lower() or feat.startswith('di_'): cat = "ADX"
            elif 'ichimoku' in feat.lower() or 'tenkan' in feat or 'kijun' in feat: cat = "Ichi"
            elif 'cci' in feat.lower(): cat = "CCI"
            elif 'williams' in feat.lower(): cat = "Will%R"
            elif 'donchian' in feat.lower(): cat = "Donch"
            elif 'rsi' in feat.lower(): cat = "RSI"
            elif 'bb' in feat.lower(): cat = "BB"
            elif 'lag' in feat.lower(): cat = "Lag"
            elif 'd1_' in feat: cat = "D1"
            else: cat = "Other"
            print(f"    {i:2}. [{cat:5}] {feat}")
        
        # Comparison
        print(f"\n{'='*70}")
        print(f"  COMPARISON")
        print(f"{'='*70}")
        print(f"  Previous (215 features): AUC ~0.545")
        print(f"  Current (362 features):  AUC {best['avg_auc']:.3f}")
        
        diff = best['avg_auc'] - 0.545
        if diff > 0:
            print(f"  Improvement: +{diff:.3f} ✅")
        else:
            print(f"  Change: {diff:.3f} ❌")
        
        # Verdict
        print(f"\n  VERDICT:")
        if best['avg_auc'] >= 0.65:
            print(f"  🎉 TARGET ACHIEVED!")
        elif best['avg_auc'] >= 0.60:
            print(f"  ✅ Good progress, close to target")
        elif best['avg_auc'] > 0.545:
            print(f"  ⚠️ Small improvement, not enough")
        else:
            print(f"  ❌ No improvement - expanded features didn't help")
            print(f"     Consider: Option G (profit optimization) or different approach")
    
    print(f"\n{'='*70}")
    print(f"  TEST COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
