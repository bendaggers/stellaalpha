"""
STELLA ALPHA - PHASE 2: ML-FILTERED MTF TREND SYSTEM
=====================================================

Phase 1 showed:
- MTF Trend Long has edge (+3,409 pips with TP100/SL50)
- But edge is weak (p=0.23) and diluted across many trades

Phase 2 goal:
- Use ML to FILTER signals, not generate them
- Only take high-confidence MTF trend signals
- Concentrate edge in fewer, better trades

Approach:
1. Label trades as WIN (hit TP) or LOSS (hit SL)
2. Train model to predict WIN probability
3. Only take trades where P(WIN) > threshold
4. Compare filtered vs unfiltered performance

Usage:
    python ml_filtered_mtf.py --h4 data/EURUSD_H4_features.csv --d1 data/EURUSD_D1_features.csv
"""

import pandas as pd
import numpy as np
from scipy import stats
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, roc_auc_score
import lightgbm as lgb


# =============================================================================
# CONFIGURATION
# =============================================================================

# Best config from Phase 1
TRADE_CONFIG = {
    "tp_pips": 100,
    "sl_pips": 50,
    "max_hold_bars": 72,
    "direction": "long",  # Long performed better
}

PIP_VALUE = 0.0001

# ML settings
ML_CONFIG = {
    "n_splits": 5,  # Walk-forward splits
    "test_size": 0.2,
    "threshold_range": [0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_and_merge_data(h4_path: str, d1_path: str) -> pd.DataFrame:
    """Load and merge H4+D1 data (same as before)."""
    
    print(f"\n📂 Loading data...")
    df_h4 = pd.read_csv(h4_path)
    df_d1 = pd.read_csv(d1_path)
    print(f"   H4: {len(df_h4):,} rows | D1: {len(df_d1):,} rows")
    
    df_h4["timestamp"] = pd.to_datetime(df_h4["timestamp"])
    df_d1["timestamp"] = pd.to_datetime(df_d1["timestamp"])
    
    df_h4["hour"] = df_h4["timestamp"].dt.hour
    df_h4["day_of_week"] = df_h4["timestamp"].dt.dayofweek
    df_h4["h4_date"] = df_h4["timestamp"].dt.normalize()
    
    # D1: shift by 1 day (no leakage)
    df_d1["d1_date"] = df_d1["timestamp"].dt.normalize()
    df_d1["d1_available_date"] = df_d1["d1_date"] + pd.Timedelta(days=1)
    
    # Rename D1 columns
    d1_feature_cols = ["bb_position", "rsi_value", "trend_strength", "close", "atr_pct",
                       "upper_band", "lower_band", "middle_band", "volume_ratio"]
    d1_cols_to_keep = ["d1_available_date"]
    
    for col in d1_feature_cols:
        if col in df_d1.columns:
            df_d1[f"d1_{col}"] = df_d1[col]
            d1_cols_to_keep.append(f"d1_{col}")
    
    df_d1_slim = df_d1[d1_cols_to_keep].copy()
    
    # Merge
    df_h4 = df_h4.sort_values("timestamp").reset_index(drop=True)
    df_d1_slim = df_d1_slim.sort_values("d1_available_date").reset_index(drop=True)
    
    df = pd.merge_asof(
        df_h4, df_d1_slim,
        left_on="h4_date",
        right_on="d1_available_date",
        direction="backward"
    )
    
    # Validate no leakage
    violations = df[df["timestamp"] < df["d1_available_date"]]
    if len(violations) > 0:
        raise ValueError(f"Data leakage: {len(violations)} rows!")
    print(f"   ✅ No data leakage")
    
    return df


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features for ML model."""
    
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
    
    # === FEATURES FOR ML ===
    
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
    
    # D1 features for ML
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
    
    # Day of week
    df["is_monday"] = (df["day_of_week"] == 0).astype(int)
    df["is_friday"] = (df["day_of_week"] == 4).astype(int)
    
    return df


# =============================================================================
# LABELING (Trade Outcomes)
# =============================================================================

def label_trades(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Label each bar with trade outcome if signal fires."""
    
    df = df.copy()
    tp = config["tp_pips"]
    sl = config["sl_pips"]
    max_hold = config["max_hold_bars"]
    direction = config["direction"]
    
    signal_col = "mtf_long_signal" if direction == "long" else "mtf_short_signal"
    
    # Initialize label column
    df["trade_label"] = np.nan  # NaN = no signal
    df["trade_pips"] = np.nan
    df["trade_bars"] = np.nan
    
    signal_indices = df[df[signal_col]].index.tolist()
    
    print(f"\n📊 Labeling {len(signal_indices):,} {direction.upper()} signals...")
    
    labeled_count = 0
    win_count = 0
    
    for idx in signal_indices:
        if idx + max_hold >= len(df):
            continue
        
        entry_price = df.loc[idx, "close"]
        tp_price = entry_price + (tp * PIP_VALUE * (1 if direction == "long" else -1))
        sl_price = entry_price - (sl * PIP_VALUE * (1 if direction == "long" else -1))
        
        # Simulate trade
        outcome = None
        exit_pips = 0
        bars_held = 0
        
        for offset in range(1, max_hold + 1):
            bar_idx = idx + offset
            if bar_idx >= len(df):
                break
            
            bar_high = df.loc[bar_idx, "high"]
            bar_low = df.loc[bar_idx, "low"]
            
            if direction == "long":
                if bar_low <= sl_price:
                    outcome = 0  # Loss
                    exit_pips = -sl
                    bars_held = offset
                    break
                if bar_high >= tp_price:
                    outcome = 1  # Win
                    exit_pips = tp
                    bars_held = offset
                    break
            else:  # short
                if bar_high >= sl_price:
                    outcome = 0
                    exit_pips = -sl
                    bars_held = offset
                    break
                if bar_low <= tp_price:
                    outcome = 1
                    exit_pips = tp
                    bars_held = offset
                    break
        
        # Timeout
        if outcome is None:
            final_idx = min(idx + max_hold, len(df) - 1)
            final_price = df.loc[final_idx, "close"]
            if direction == "long":
                exit_pips = (final_price - entry_price) / PIP_VALUE
            else:
                exit_pips = (entry_price - final_price) / PIP_VALUE
            outcome = 1 if exit_pips > 0 else 0
            bars_held = max_hold
        
        df.loc[idx, "trade_label"] = outcome
        df.loc[idx, "trade_pips"] = exit_pips
        df.loc[idx, "trade_bars"] = bars_held
        
        labeled_count += 1
        if outcome == 1:
            win_count += 1
    
    print(f"   Labeled: {labeled_count:,} trades")
    print(f"   Wins: {win_count:,} ({100*win_count/labeled_count:.1f}%)")
    print(f"   Losses: {labeled_count - win_count:,} ({100*(labeled_count-win_count)/labeled_count:.1f}%)")
    
    return df


# =============================================================================
# ML MODEL
# =============================================================================

def get_feature_columns(df: pd.DataFrame) -> List[str]:
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
            if df[col].notna().sum() > len(df) * 0.5:  # At least 50% non-null
                feature_cols.append(col)
    
    return feature_cols


def train_and_evaluate(df: pd.DataFrame, feature_cols: List[str]) -> dict:
    """Train ML model with walk-forward validation."""
    
    # Get labeled data only
    labeled_df = df[df["trade_label"].notna()].copy()
    print(f"\n🤖 Training ML model on {len(labeled_df):,} labeled trades...")
    
    X = labeled_df[feature_cols].copy()
    y = labeled_df["trade_label"].values
    pips = labeled_df["trade_pips"].values
    
    # Fill NaN with median
    X = X.fillna(X.median())
    
    # Walk-forward split
    tscv = TimeSeriesSplit(n_splits=ML_CONFIG["n_splits"])
    
    results = []
    all_predictions = []
    all_actuals = []
    all_pips = []
    all_indices = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        pips_test = pips[test_idx]
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train LightGBM
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
        
        model.fit(X_train_scaled, y_train)
        
        # Predict probabilities
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        all_predictions.extend(y_proba)
        all_actuals.extend(y_test)
        all_pips.extend(pips_test)
        all_indices.extend(test_idx)
        
        # Fold metrics
        auc = roc_auc_score(y_test, y_proba)
        print(f"   Fold {fold}: AUC={auc:.3f}, Test size={len(y_test)}")
    
    return {
        "predictions": np.array(all_predictions),
        "actuals": np.array(all_actuals),
        "pips": np.array(all_pips),
        "indices": np.array(all_indices),
        "model": model,
        "feature_cols": feature_cols
    }


def evaluate_thresholds(results: dict) -> pd.DataFrame:
    """Evaluate different probability thresholds."""
    
    predictions = results["predictions"]
    actuals = results["actuals"]
    pips = results["pips"]
    
    threshold_results = []
    
    print(f"\n📈 Evaluating thresholds...")
    print(f"   {'Threshold':<12} {'Trades':>8} {'WinRate':>10} {'TotalPips':>12} {'AvgPips':>10} {'Precision':>10}")
    print("   " + "─" * 65)
    
    # Baseline (no filter)
    baseline_trades = len(pips)
    baseline_wins = (actuals == 1).sum()
    baseline_wr = baseline_wins / baseline_trades
    baseline_total = pips.sum()
    baseline_avg = pips.mean()
    
    print(f"   {'BASELINE':<12} {baseline_trades:>8} {baseline_wr:>10.1%} {baseline_total:>+12.0f} {baseline_avg:>+10.2f} {'-':>10}")
    
    for threshold in ML_CONFIG["threshold_range"]:
        mask = predictions >= threshold
        
        if mask.sum() < 50:
            continue
        
        filtered_trades = mask.sum()
        filtered_pips = pips[mask]
        filtered_actuals = actuals[mask]
        
        wins = (filtered_actuals == 1).sum()
        win_rate = wins / filtered_trades
        total_pips = filtered_pips.sum()
        avg_pips = filtered_pips.mean()
        precision = precision_score(actuals[mask], (filtered_actuals == 1).astype(int), zero_division=0)
        
        # Compare to baseline
        pips_improvement = avg_pips - baseline_avg
        wr_improvement = win_rate - baseline_wr
        
        print(f"   {threshold:<12.2f} {filtered_trades:>8} {win_rate:>10.1%} {total_pips:>+12.0f} {avg_pips:>+10.2f} {precision:>10.1%}")
        
        threshold_results.append({
            "threshold": threshold,
            "trades": filtered_trades,
            "wins": wins,
            "win_rate": win_rate,
            "total_pips": total_pips,
            "avg_pips": avg_pips,
            "precision": precision,
            "trades_reduction": 1 - (filtered_trades / baseline_trades),
            "wr_improvement": wr_improvement,
            "pips_improvement": pips_improvement
        })
    
    return pd.DataFrame(threshold_results)


def find_best_threshold(threshold_df: pd.DataFrame) -> dict:
    """Find best threshold based on avg pips per trade."""
    
    if threshold_df.empty:
        return None
    
    # Best by avg pips
    best_idx = threshold_df["avg_pips"].idxmax()
    best = threshold_df.loc[best_idx]
    
    return {
        "threshold": best["threshold"],
        "trades": int(best["trades"]),
        "win_rate": best["win_rate"],
        "total_pips": best["total_pips"],
        "avg_pips": best["avg_pips"],
        "trades_reduction": best["trades_reduction"],
        "wr_improvement": best["wr_improvement"],
        "pips_improvement": best["pips_improvement"]
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h4", required=True)
    parser.add_argument("--d1", required=True)
    args = parser.parse_args()
    
    # Load data
    df = load_and_merge_data(args.h4, args.d1)
    
    # Engineer features
    print(f"\n🔧 Engineering features...")
    df = engineer_features(df)
    
    # Label trades
    df = label_trades(df, TRADE_CONFIG)
    
    # Get feature columns
    feature_cols = get_feature_columns(df)
    print(f"\n📋 Using {len(feature_cols)} features for ML")
    
    # Train and evaluate
    results = train_and_evaluate(df, feature_cols)
    
    # Evaluate thresholds
    threshold_df = evaluate_thresholds(results)
    
    # Find best threshold
    best = find_best_threshold(threshold_df)
    
    # Print summary
    print("\n" + "=" * 70)
    print("  PHASE 2: ML FILTER RESULTS")
    print("=" * 70)
    
    if best and best["pips_improvement"] > 0:
        print(f"""
  🟢 ML FILTER IMPROVES PERFORMANCE!
  
  Best Threshold: {best['threshold']:.2f}
  
  COMPARISON:
  ─────────────────────────────────────────────
  Metric              Baseline    Filtered    Change
  ─────────────────────────────────────────────
  Trades              {len(results['pips']):>8}    {best['trades']:>8}    {-best['trades_reduction']*100:>+.0f}%
  Win Rate            {(results['actuals']==1).mean():>8.1%}    {best['win_rate']:>8.1%}    {best['wr_improvement']*100:>+.1f}%
  Avg Pips/Trade      {results['pips'].mean():>+8.2f}    {best['avg_pips']:>+8.2f}    {best['pips_improvement']:>+.2f}
  Total Pips          {results['pips'].sum():>+8.0f}    {best['total_pips']:>+8.0f}
  
  ✅ ML filter increases avg pips by {best['pips_improvement']:+.2f} per trade
  ✅ Win rate improved by {best['wr_improvement']*100:+.1f}%
  
  NEXT STEP: Walk-forward backtest with this filter
        """)
    else:
        print(f"""
  🟡 ML FILTER SHOWS MIXED RESULTS
  
  The model can rank trades but improvement is marginal.
  Consider:
  - More features
  - Different model architecture
  - Ensemble approach
        """)
    
    print("=" * 70)
    
    return best


if __name__ == "__main__":
    main()
