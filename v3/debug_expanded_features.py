"""
DEBUG: Test expanded features without multiprocessing to see errors
"""

import pandas as pd
import numpy as np
import argparse
import warnings
import traceback

warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

from expanded_features import ExpandedFeatureEngineering, get_expanded_feature_columns


PIP_VALUE = 0.0001
MAX_HOLD_BARS = 72
N_FEATURES_TO_SELECT = 40


def load_and_merge_data(h4_path: str, d1_path: str) -> pd.DataFrame:
    """Load and merge H4/D1 data."""
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
    if 'ema_50' not in df.columns:
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    
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


def label_trades(df: pd.DataFrame, tp: int, sl: int, direction: str = "long") -> pd.DataFrame:
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h4", required=True)
    parser.add_argument("--d1", required=True)
    args = parser.parse_args()
    
    print("="*70)
    print("  DEBUG: TEST EXPANDED FEATURES")
    print("="*70)
    
    # Load data
    print("\n  Loading data...")
    df = load_and_merge_data(args.h4, args.d1)
    print(f"  Loaded: {len(df):,} rows")
    
    # Feature engineering
    print("\n  Computing features...")
    fe = ExpandedFeatureEngineering(verbose=True)
    df = fe.calculate_all_features(df)
    
    # Generate signals
    print("\n  Generating signals...")
    df = generate_signals(df)
    print(f"  MTF Long signals: {df['mtf_long_signal'].sum():,}")
    print(f"  MTF Short signals: {df['mtf_short_signal'].sum():,}")
    
    # Get feature columns
    feature_cols = get_expanded_feature_columns(df)
    print(f"\n  Total features: {len(feature_cols)}")
    
    # Test ONE config without multiprocessing
    print("\n  Testing 100/50 LONG...")
    tp, sl, direction = 100, 50, "long"
    
    try:
        # Label trades
        print("    Labeling trades...")
        df_labeled = label_trades(df.copy(), tp, sl, direction)
        
        signals_df = df_labeled[df_labeled["trade_label"].notna()].copy()
        print(f"    Labeled signals: {len(signals_df):,}")
        print(f"    Wins: {signals_df['trade_label'].sum():,}")
        print(f"    Baseline WR: {signals_df['trade_label'].mean()*100:.1f}%")
        
        # Walk-forward CV (just one fold for debug)
        print("\n    Testing Fold 2 (2014-2018)...")
        train_end = pd.Timestamp("2014-01-01")
        test_end = pd.Timestamp("2018-01-01")
        
        train_df = signals_df[signals_df["timestamp"] < train_end].copy()
        test_df = signals_df[(signals_df["timestamp"] >= train_end) & (signals_df["timestamp"] < test_end)].copy()
        
        print(f"    Train size: {len(train_df):,}")
        print(f"    Test size: {len(test_df):,}")
        
        if len(train_df) < 100 or len(test_df) < 50:
            print("    ERROR: Not enough data")
            return
        
        # Filter valid features
        print("\n    Filtering features...")
        valid_features = []
        for f in feature_cols:
            if f in train_df.columns:
                nan_pct = train_df[f].isna().mean()
                inf_count = np.isinf(train_df[f].replace([np.inf, -np.inf], np.nan).dropna()).sum() if train_df[f].dtype in ['float64', 'float32'] else 0
                if nan_pct < 0.1:
                    valid_features.append(f)
        
        print(f"    Valid features: {len(valid_features)}")
        
        if len(valid_features) < 50:
            print("    ERROR: Not enough valid features")
            return
        
        X_train = train_df[valid_features].copy()
        y_train = train_df["trade_label"].values
        X_test = test_df[valid_features].copy()
        y_test = test_df["trade_label"].values
        
        print(f"    X_train shape: {X_train.shape}")
        print(f"    X_test shape: {X_test.shape}")
        
        # Check for NaN/Inf
        print(f"\n    Checking for NaN/Inf...")
        nan_cols = X_train.columns[X_train.isna().any()].tolist()
        print(f"    Columns with NaN: {len(nan_cols)}")
        
        # Fill NaN
        print("    Filling NaN...")
        medians = X_train.median()
        X_train = X_train.fillna(medians)
        X_test = X_test.fillna(medians)
        
        # Replace inf
        print("    Replacing inf...")
        X_train = X_train.replace([np.inf, -np.inf], 0)
        X_test = X_test.replace([np.inf, -np.inf], 0)
        
        # Final NaN fill
        X_train = X_train.fillna(0)
        X_test = X_test.fillna(0)
        
        # Check again
        remaining_nan = X_train.isna().sum().sum()
        print(f"    Remaining NaN: {remaining_nan}")
        
        if remaining_nan > 0:
            print("    ERROR: Still have NaN values")
            return
        
        # RFE
        print("\n    Running RFE...")
        estimator = lgb.LGBMClassifier(
            n_estimators=50, max_depth=4, learning_rate=0.1,
            verbose=-1, random_state=42, n_jobs=1
        )
        n_select = min(N_FEATURES_TO_SELECT, len(valid_features))
        selector = RFE(estimator=estimator, n_features_to_select=n_select, step=10)
        selector.fit(X_train, y_train)
        
        selected = X_train.columns[selector.support_].tolist()
        print(f"    Selected features: {len(selected)}")
        
        X_train_sel = X_train[selected]
        X_test_sel = X_test[selected]
        
        # Scale
        print("    Scaling...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_sel)
        X_test_scaled = scaler.transform(X_test_sel)
        
        # Train
        print("    Training model...")
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
        print("    Predicting...")
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # AUC
        auc = roc_auc_score(y_test, y_proba)
        print(f"\n    ✅ SUCCESS! AUC = {auc:.3f}")
        
        # Top features
        print("\n    Top 10 features:")
        feat_imp = list(zip(selected, model.feature_importances_))
        feat_imp.sort(key=lambda x: x[1], reverse=True)
        for i, (feat, imp) in enumerate(feat_imp[:10], 1):
            print(f"      {i:2}. {feat}: {imp:.1f}")
        
        # Threshold analysis
        print("\n    Threshold analysis:")
        for thresh in [0.45, 0.50, 0.55, 0.60, 0.65]:
            mask = y_proba >= thresh
            n = mask.sum()
            if n >= 10:
                wr = y_test[mask].mean()
                print(f"      Threshold {thresh:.2f}: {n:,} trades, {wr*100:.1f}% win rate")
        
    except Exception as e:
        print(f"\n    ❌ ERROR: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()
