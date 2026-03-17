"""
STELLA ALPHA - EXPORT PRODUCTION MODEL
=======================================

Exports the final trained model and all artifacts needed for live trading.

Outputs:
1. stella_model.pkl         - Trained LightGBM model
2. stella_scaler.pkl        - Feature scaler
3. stella_config.json       - Trading configuration
4. stella_features.json     - Feature list and computation logic

Usage:
    python export_production_model.py --h4 data/EURUSD_H4_features.csv --d1 data/EURUSD_D1_features.csv --output models/
"""

import pandas as pd
import numpy as np
import pickle
import json
import argparse
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
import lightgbm as lgb


# =============================================================================
# CONFIGURATION (Validated settings from walk-forward)
# =============================================================================

PRODUCTION_CONFIG = {
    "model_name": "stella_alpha_v1",
    "version": "1.0.0",
    "created_date": datetime.now().isoformat(),
    
    # Signal
    "signal_type": "MTF_TREND_ALIGNED",
    "direction": "long",  # Can add "short" later
    
    # Trade parameters
    "tp_pips": 100,
    "sl_pips": 50,
    "max_hold_bars": 72,
    "timeframe": "H4",
    
    # ML filter
    "ml_threshold": 0.65,
    "use_ml_filter": True,
    
    # Risk management
    "max_trades_per_day": 2,
    "max_open_trades": 1,
    
    # Expected performance (from validation)
    "expected_win_rate": 0.45,
    "expected_avg_pips": 12.0,
    "validated_period": "2016-2025",
    "warning": "Model underperforms in trending bear markets (2010-2013 style)"
}


# =============================================================================
# DATA & FEATURE FUNCTIONS (same as validation)
# =============================================================================

def load_and_merge_data(h4_path: str, d1_path: str) -> pd.DataFrame:
    """Load and merge H4+D1 data."""
    print(f"\n📂 Loading data...")
    df_h4 = pd.read_csv(h4_path)
    df_d1 = pd.read_csv(d1_path)
    
    df_h4["timestamp"] = pd.to_datetime(df_h4["timestamp"])
    df_d1["timestamp"] = pd.to_datetime(df_d1["timestamp"])
    
    df_h4["hour"] = df_h4["timestamp"].dt.hour
    df_h4["day_of_week"] = df_h4["timestamp"].dt.dayofweek
    df_h4["h4_date"] = df_h4["timestamp"].dt.normalize()
    
    df_d1["d1_date"] = df_d1["timestamp"].dt.normalize()
    df_d1["d1_available_date"] = df_d1["d1_date"] + pd.Timedelta(days=1)
    
    d1_feature_cols = ["bb_position", "rsi_value", "trend_strength", "close", "atr_pct",
                       "upper_band", "lower_band", "middle_band", "volume_ratio"]
    d1_cols_to_keep = ["d1_available_date"]
    
    for col in d1_feature_cols:
        if col in df_d1.columns:
            df_d1[f"d1_{col}"] = df_d1[col]
            d1_cols_to_keep.append(f"d1_{col}")
    
    df_d1_slim = df_d1[d1_cols_to_keep].copy()
    
    df_h4 = df_h4.sort_values("timestamp").reset_index(drop=True)
    df_d1_slim = df_d1_slim.sort_values("d1_available_date").reset_index(drop=True)
    
    df = pd.merge_asof(df_h4, df_d1_slim, left_on="h4_date",
                       right_on="d1_available_date", direction="backward")
    
    print(f"   Merged: {len(df):,} rows")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create all features for ML model."""
    df = df.copy()
    
    # H4 trend
    if "trend_strength" not in df.columns:
        df["ma_20"] = df["close"].rolling(20).mean()
        df["ma_50"] = df["close"].rolling(50).mean()
        df["trend_strength"] = (df["close"] - df["ma_50"]) / df["ma_50"]
    
    df["h4_uptrend"] = df["trend_strength"] > 0.3
    df["h4_downtrend"] = df["trend_strength"] < -0.3
    
    # D1 trend
    if "d1_trend_strength" in df.columns:
        df["d1_uptrend"] = df["d1_trend_strength"] > 0.3
        df["d1_downtrend"] = df["d1_trend_strength"] < -0.3
    else:
        df["d1_uptrend"] = False
        df["d1_downtrend"] = False
    
    # MTF signals
    df["mtf_long_signal"] = df["h4_uptrend"] & df["d1_uptrend"]
    df["mtf_short_signal"] = df["h4_downtrend"] & df["d1_downtrend"]
    
    # Price action
    df["price_change_1"] = df["close"].pct_change(1)
    df["price_change_3"] = df["close"].pct_change(3)
    df["price_change_5"] = df["close"].pct_change(5)
    
    # Volatility
    df["atr_14"] = (df["high"] - df["low"]).rolling(14).mean()
    df["atr_pct"] = df["atr_14"] / df["close"]
    df["volatility_ratio"] = df["atr_14"] / df["atr_14"].rolling(50).mean()
    
    # BB features
    if "bb_position" in df.columns:
        df["bb_position_ma5"] = df["bb_position"].rolling(5).mean()
        df["bb_trend"] = df["bb_position"] - df["bb_position"].shift(3)
    
    # RSI features
    if "rsi_value" in df.columns:
        df["rsi_ma5"] = df["rsi_value"].rolling(5).mean()
        df["rsi_trend"] = df["rsi_value"] - df["rsi_value"].shift(3)
        df["rsi_momentum"] = df["rsi_value"] - df["rsi_value"].shift(5)
    
    # Trend features
    df["trend_strength_ma5"] = df["trend_strength"].rolling(5).mean()
    df["trend_acceleration"] = df["trend_strength"] - df["trend_strength"].shift(3)
    
    # D1 features
    if "d1_rsi_value" in df.columns:
        df["d1_rsi_ma3"] = df["d1_rsi_value"].rolling(3).mean()
        df["h4_d1_rsi_diff"] = df["rsi_value"] - df["d1_rsi_value"]
    
    if "d1_bb_position" in df.columns:
        df["h4_d1_bb_diff"] = df["bb_position"] - df["d1_bb_position"]
    
    if "d1_trend_strength" in df.columns:
        df["h4_d1_trend_diff"] = df["trend_strength"] - df["d1_trend_strength"]
        df["mtf_trend_alignment"] = df["trend_strength"] * df["d1_trend_strength"]
    
    # Session features
    df["is_london"] = df["hour"].isin([8, 9, 10, 11]).astype(int)
    df["is_ny"] = df["hour"].isin([13, 14, 15, 16]).astype(int)
    df["is_overlap"] = df["hour"].isin([13, 14, 15, 16]).astype(int)
    df["is_monday"] = (df["day_of_week"] == 0).astype(int)
    df["is_friday"] = (df["day_of_week"] == 4).astype(int)
    
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Get feature columns for ML."""
    exclude = [
        "timestamp", "open", "high", "low", "close", "volume",
        "h4_date", "d1_available_date", "d1_close",
        "trade_label", "trade_pips", "trade_bars",
        "mtf_long_signal", "mtf_short_signal",
        "h4_uptrend", "h4_downtrend", "d1_uptrend", "d1_downtrend"
    ]
    
    feature_cols = []
    for col in df.columns:
        if col in exclude:
            continue
        if df[col].dtype in [np.float64, np.int64, np.float32, np.int32, bool]:
            if df[col].notna().sum() > len(df) * 0.5:
                feature_cols.append(col)
    
    return sorted(feature_cols)  # Sort for consistency


def simulate_trade(df: pd.DataFrame, entry_idx: int, config: dict) -> dict:
    """Simulate a single trade to get label."""
    tp = config["tp_pips"]
    sl = config["sl_pips"]
    max_hold = config["max_hold_bars"]
    direction = config["direction"]
    pip_value = 0.0001
    
    if entry_idx + max_hold >= len(df):
        return None
    
    entry_price = df.loc[entry_idx, "close"]
    tp_price = entry_price + (tp * pip_value * (1 if direction == "long" else -1))
    sl_price = entry_price - (sl * pip_value * (1 if direction == "long" else -1))
    
    for offset in range(1, max_hold + 1):
        idx = entry_idx + offset
        if idx >= len(df):
            break
        
        bar_high = df.loc[idx, "high"]
        bar_low = df.loc[idx, "low"]
        
        if direction == "long":
            if bar_low <= sl_price:
                return {"outcome": 0, "pips": -sl}
            if bar_high >= tp_price:
                return {"outcome": 1, "pips": tp}
        else:
            if bar_high >= sl_price:
                return {"outcome": 0, "pips": -sl}
            if bar_low <= tp_price:
                return {"outcome": 1, "pips": tp}
    
    # Timeout
    final_price = df.loc[min(entry_idx + max_hold, len(df) - 1), "close"]
    if direction == "long":
        pips = (final_price - entry_price) / pip_value
    else:
        pips = (entry_price - final_price) / pip_value
    
    return {"outcome": 1 if pips > 0 else 0, "pips": pips}


def label_signals(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Label all signals with trade outcomes."""
    df = df.copy()
    direction = config["direction"]
    signal_col = "mtf_long_signal" if direction == "long" else "mtf_short_signal"
    
    df["trade_label"] = np.nan
    df["trade_pips"] = np.nan
    
    signal_indices = df[df[signal_col]].index.tolist()
    
    for idx in signal_indices:
        result = simulate_trade(df, idx, config)
        if result:
            df.loc[idx, "trade_label"] = result["outcome"]
            df.loc[idx, "trade_pips"] = result["pips"]
    
    return df


# =============================================================================
# MAIN EXPORT
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h4", required=True)
    parser.add_argument("--d1", required=True)
    parser.add_argument("--output", default="models/")
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    df = load_and_merge_data(args.h4, args.d1)
    
    # Engineer features
    print(f"\n🔧 Engineering features...")
    df = engineer_features(df)
    
    # Label signals
    print(f"\n📊 Labeling signals...")
    df = label_signals(df, PRODUCTION_CONFIG)
    
    # Get labeled data
    labeled_df = df[df["trade_label"].notna()].copy()
    print(f"   Total signals: {len(labeled_df):,}")
    print(f"   Wins: {(labeled_df['trade_label'] == 1).sum():,} ({100*(labeled_df['trade_label'] == 1).mean():.1f}%)")
    
    # Get features
    feature_cols = get_feature_columns(df)
    print(f"\n📋 Features: {len(feature_cols)}")
    
    # Prepare training data (use all data for final model)
    X = labeled_df[feature_cols].copy()
    y = labeled_df["trade_label"].values
    
    # Fill NaN with median
    feature_medians = X.median().to_dict()
    X = X.fillna(X.median())
    
    # Scale
    print(f"\n⚙️ Training final model...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train final model
    model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.05,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    model.fit(X_scaled, y)
    
    # Get feature importance
    importance = dict(zip(feature_cols, model.feature_importances_))
    importance_sorted = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
    
    # ===================
    # EXPORT ARTIFACTS
    # ===================
    
    print(f"\n💾 Exporting to {output_dir}/...")
    
    # 1. Model
    model_path = output_dir / "stella_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"   ✅ {model_path}")
    
    # 2. Scaler
    scaler_path = output_dir / "stella_scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"   ✅ {scaler_path}")
    
    # 3. Config
    config_path = output_dir / "stella_config.json"
    with open(config_path, "w") as f:
        json.dump(PRODUCTION_CONFIG, f, indent=2)
    print(f"   ✅ {config_path}")
    
    # 4. Features
    features_info = {
        "feature_columns": feature_cols,
        "feature_count": len(feature_cols),
        "feature_medians": feature_medians,
        "feature_importance": importance_sorted,
        "top_10_features": list(importance_sorted.keys())[:10],
        
        # Feature computation instructions
        "computation": {
            "price_change_1": "close.pct_change(1)",
            "price_change_3": "close.pct_change(3)",
            "price_change_5": "close.pct_change(5)",
            "atr_14": "(high - low).rolling(14).mean()",
            "atr_pct": "atr_14 / close",
            "volatility_ratio": "atr_14 / atr_14.rolling(50).mean()",
            "bb_position_ma5": "bb_position.rolling(5).mean()",
            "bb_trend": "bb_position - bb_position.shift(3)",
            "rsi_ma5": "rsi_value.rolling(5).mean()",
            "rsi_trend": "rsi_value - rsi_value.shift(3)",
            "rsi_momentum": "rsi_value - rsi_value.shift(5)",
            "trend_strength_ma5": "trend_strength.rolling(5).mean()",
            "trend_acceleration": "trend_strength - trend_strength.shift(3)",
            "h4_d1_rsi_diff": "rsi_value - d1_rsi_value",
            "h4_d1_bb_diff": "bb_position - d1_bb_position",
            "h4_d1_trend_diff": "trend_strength - d1_trend_strength",
            "mtf_trend_alignment": "trend_strength * d1_trend_strength",
            "is_london": "hour in [8, 9, 10, 11]",
            "is_ny": "hour in [13, 14, 15, 16]",
            "is_overlap": "hour in [13, 14, 15, 16]",
            "is_monday": "day_of_week == 0",
            "is_friday": "day_of_week == 4"
        },
        
        # Signal conditions
        "signal_conditions": {
            "mtf_long": "h4_uptrend AND d1_uptrend",
            "h4_uptrend": "trend_strength > 0.3",
            "d1_uptrend": "d1_trend_strength > 0.3",
            "d1_data_rule": "Use PREVIOUS day D1 (no same-day data)"
        }
    }
    
    features_path = output_dir / "stella_features.json"
    with open(features_path, "w") as f:
        json.dump(features_info, f, indent=2)
    print(f"   ✅ {features_path}")
    
    # 5. Summary report
    summary = f"""
================================================================================
  STELLA ALPHA - PRODUCTION MODEL EXPORTED
================================================================================

  Model:     {model_path}
  Scaler:    {scaler_path}
  Config:    {config_path}
  Features:  {features_path}

  TRADING RULES:
  ─────────────────────────────────────────────────
  1. Check MTF Signal:
     - H4 trend_strength > 0.3 (uptrend)
     - D1 trend_strength > 0.3 (uptrend from PREVIOUS day)
     - Both must be true for LONG signal
  
  2. Apply ML Filter:
     - Compute all {len(feature_cols)} features
     - Scale using stella_scaler.pkl
     - Get probability from stella_model.pkl
     - Only trade if probability >= {PRODUCTION_CONFIG['ml_threshold']}
  
  3. Execute Trade:
     - Entry: Market order on signal bar close
     - TP: {PRODUCTION_CONFIG['tp_pips']} pips
     - SL: {PRODUCTION_CONFIG['sl_pips']} pips
     - Max Hold: {PRODUCTION_CONFIG['max_hold_bars']} H4 bars

  TOP 10 FEATURES:
  ─────────────────────────────────────────────────
"""
    for i, (feat, imp) in enumerate(list(importance_sorted.items())[:10], 1):
        summary += f"  {i:2}. {feat:<30} (importance: {imp:.1f})\n"
    
    summary += f"""
  EXPECTED PERFORMANCE:
  ─────────────────────────────────────────────────
  Win Rate:    ~{PRODUCTION_CONFIG['expected_win_rate']*100:.0f}%
  Avg Pips:    ~{PRODUCTION_CONFIG['expected_avg_pips']:.0f} per trade
  Trades/Year: ~50-100
  
  ⚠️ WARNING: {PRODUCTION_CONFIG['warning']}

================================================================================
"""
    print(summary)
    
    # Save summary
    summary_path = output_dir / "README.txt"
    with open(summary_path, "w") as f:
        f.write(summary)
    print(f"   ✅ {summary_path}")
    
    print(f"\n✅ Export complete! All files in: {output_dir}/")
    print(f"\n📋 Next step: Create MQL5 EA using these artifacts")


if __name__ == "__main__":
    main()
