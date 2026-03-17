"""
STELLA ALPHA - PHASE 3: WALK-FORWARD OUT-OF-SAMPLE VALIDATION
==============================================================

Phase 2 showed ML filter improves performance:
- Win rate: 37.4% → 44.7%
- Avg pips: +6.03 → +17.04

BUT we need to validate this isn't overfitting.

This script:
1. True walk-forward: Train on past, test on future
2. NO LEAKAGE: Model only sees past data when making predictions
3. Realistic simulation of live trading

Walk-Forward Structure (25 years of data):
- Fold 1: Train [2000-2010], Test [2010-2013]
- Fold 2: Train [2000-2013], Test [2013-2016]
- Fold 3: Train [2000-2016], Test [2016-2019]
- Fold 4: Train [2000-2019], Test [2019-2022]
- Fold 5: Train [2000-2022], Test [2022-2025]

Usage:
    python walk_forward_validation.py --h4 data/EURUSD_H4_features.csv --d1 data/EURUSD_D1_features.csv
"""

import pandas as pd
import numpy as np
from scipy import stats
import argparse
from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, roc_auc_score
import lightgbm as lgb


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

# Walk-forward folds (year-based)
WALK_FORWARD_FOLDS = [
    {"train_end": "2010-01-01", "test_end": "2013-01-01"},
    {"train_end": "2013-01-01", "test_end": "2016-01-01"},
    {"train_end": "2016-01-01", "test_end": "2019-01-01"},
    {"train_end": "2019-01-01", "test_end": "2022-01-01"},
    {"train_end": "2022-01-01", "test_end": "2025-01-01"},
]

# Threshold to use - trying lower for more trades
ML_THRESHOLD = 0.50  # Lower threshold = more trades, still some filtering


# =============================================================================
# DATA LOADING & FEATURE ENGINEERING (same as before)
# =============================================================================

def load_and_merge_data(h4_path: str, d1_path: str) -> pd.DataFrame:
    """Load and merge H4+D1 data."""
    print(f"\n📂 Loading data...")
    df_h4 = pd.read_csv(h4_path)
    df_d1 = pd.read_csv(d1_path)
    print(f"   H4: {len(df_h4):,} rows | D1: {len(df_d1):,} rows")
    
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
    
    violations = df[df["timestamp"] < df["d1_available_date"]]
    if len(violations) > 0:
        raise ValueError(f"Data leakage: {len(violations)} rows!")
    print(f"   ✅ No data leakage")
    
    return df


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
            if df[col].notna().sum() > len(df) * 0.5:
                feature_cols.append(col)
    
    return feature_cols


# =============================================================================
# TRADE SIMULATION
# =============================================================================

def simulate_trade(df: pd.DataFrame, entry_idx: int, config: dict) -> Optional[dict]:
    """Simulate a single trade."""
    tp = config["tp_pips"]
    sl = config["sl_pips"]
    max_hold = config["max_hold_bars"]
    direction = config["direction"]
    
    if entry_idx + max_hold >= len(df):
        return None
    
    entry_price = df.loc[entry_idx, "close"]
    entry_time = df.loc[entry_idx, "timestamp"]
    
    tp_price = entry_price + (tp * PIP_VALUE * (1 if direction == "long" else -1))
    sl_price = entry_price - (sl * PIP_VALUE * (1 if direction == "long" else -1))
    
    for offset in range(1, max_hold + 1):
        idx = entry_idx + offset
        if idx >= len(df):
            break
        
        bar_high = df.loc[idx, "high"]
        bar_low = df.loc[idx, "low"]
        
        if direction == "long":
            if bar_low <= sl_price:
                return {"outcome": 0, "pips": -sl, "bars": offset, "reason": "SL"}
            if bar_high >= tp_price:
                return {"outcome": 1, "pips": tp, "bars": offset, "reason": "TP"}
        else:
            if bar_high >= sl_price:
                return {"outcome": 0, "pips": -sl, "bars": offset, "reason": "SL"}
            if bar_low <= tp_price:
                return {"outcome": 1, "pips": tp, "bars": offset, "reason": "TP"}
    
    # Timeout
    final_idx = min(entry_idx + max_hold, len(df) - 1)
    final_price = df.loc[final_idx, "close"]
    if direction == "long":
        pips = (final_price - entry_price) / PIP_VALUE
    else:
        pips = (entry_price - final_price) / PIP_VALUE
    
    return {"outcome": 1 if pips > 0 else 0, "pips": pips, "bars": max_hold, "reason": "TIMEOUT"}


def label_signal_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Label signal bars with trade outcomes."""
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
# WALK-FORWARD VALIDATION
# =============================================================================

@dataclass
class FoldResult:
    """Results for one walk-forward fold."""
    fold_num: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    
    # Training stats
    train_samples: int
    train_win_rate: float
    
    # Test stats (baseline - all signals)
    test_signals: int
    test_baseline_wins: int
    test_baseline_wr: float
    test_baseline_pips: float
    test_baseline_avg: float
    
    # Test stats (filtered)
    test_filtered_trades: int
    test_filtered_wins: int
    test_filtered_wr: float
    test_filtered_pips: float
    test_filtered_avg: float
    
    # Model performance
    auc: float
    improvement_wr: float
    improvement_pips: float


def run_walk_forward_fold(
    df: pd.DataFrame,
    feature_cols: List[str],
    fold_config: dict,
    fold_num: int
) -> FoldResult:
    """Run one walk-forward fold."""
    
    train_end = pd.Timestamp(fold_config["train_end"])
    test_end = pd.Timestamp(fold_config["test_end"])
    
    # Split data
    train_df = df[df["timestamp"] < train_end].copy()
    test_df = df[(df["timestamp"] >= train_end) & (df["timestamp"] < test_end)].copy()
    
    # Get labeled signals only
    train_signals = train_df[train_df["trade_label"].notna()].copy()
    test_signals = test_df[test_df["trade_label"].notna()].copy()
    
    if len(train_signals) < 100 or len(test_signals) < 50:
        print(f"   Fold {fold_num}: Insufficient data (train={len(train_signals)}, test={len(test_signals)})")
        return None
    
    # Prepare features
    X_train = train_signals[feature_cols].fillna(train_signals[feature_cols].median())
    y_train = train_signals["trade_label"].values
    pips_train = train_signals["trade_pips"].values
    
    X_test = test_signals[feature_cols].fillna(test_signals[feature_cols].median())
    y_test = test_signals["trade_label"].values
    pips_test = test_signals["trade_pips"].values
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
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
    
    # Predict on test
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate AUC
    try:
        auc = roc_auc_score(y_test, y_proba)
    except:
        auc = 0.5
    
    # Baseline (all signals)
    baseline_wins = (y_test == 1).sum()
    baseline_wr = baseline_wins / len(y_test)
    baseline_pips = pips_test.sum()
    baseline_avg = pips_test.mean()
    
    # Filtered (threshold applied)
    filter_mask = y_proba >= ML_THRESHOLD
    
    if filter_mask.sum() < 10:
        # Not enough filtered trades
        filtered_trades = filter_mask.sum()
        filtered_wins = 0
        filtered_wr = 0
        filtered_pips = 0
        filtered_avg = 0
    else:
        filtered_trades = filter_mask.sum()
        filtered_wins = (y_test[filter_mask] == 1).sum()
        filtered_wr = filtered_wins / filtered_trades
        filtered_pips = pips_test[filter_mask].sum()
        filtered_avg = pips_test[filter_mask].mean()
    
    return FoldResult(
        fold_num=fold_num,
        train_start=str(train_df["timestamp"].min().date()),
        train_end=str(train_end.date()),
        test_start=str(train_end.date()),
        test_end=str(test_end.date()),
        train_samples=len(train_signals),
        train_win_rate=(y_train == 1).mean(),
        test_signals=len(test_signals),
        test_baseline_wins=baseline_wins,
        test_baseline_wr=baseline_wr,
        test_baseline_pips=baseline_pips,
        test_baseline_avg=baseline_avg,
        test_filtered_trades=filtered_trades,
        test_filtered_wins=filtered_wins,
        test_filtered_wr=filtered_wr,
        test_filtered_pips=filtered_pips,
        test_filtered_avg=filtered_avg,
        auc=auc,
        improvement_wr=filtered_wr - baseline_wr,
        improvement_pips=filtered_avg - baseline_avg
    )


def run_walk_forward_validation(df: pd.DataFrame, feature_cols: List[str]) -> List[FoldResult]:
    """Run complete walk-forward validation."""
    
    print(f"\n🔄 Running Walk-Forward Validation ({len(WALK_FORWARD_FOLDS)} folds)...")
    print(f"   Threshold: {ML_THRESHOLD}")
    
    results = []
    
    for i, fold_config in enumerate(WALK_FORWARD_FOLDS, 1):
        print(f"\n   Fold {i}: Train→{fold_config['train_end']}, Test→{fold_config['test_end']}")
        
        result = run_walk_forward_fold(df, feature_cols, fold_config, i)
        
        if result:
            results.append(result)
            print(f"      Train: {result.train_samples:,} signals, {result.train_win_rate:.1%} WR")
            print(f"      Test Baseline: {result.test_signals:,} signals, {result.test_baseline_wr:.1%} WR, {result.test_baseline_avg:+.2f} avg pips")
            print(f"      Test Filtered: {result.test_filtered_trades:,} signals, {result.test_filtered_wr:.1%} WR, {result.test_filtered_avg:+.2f} avg pips")
            print(f"      AUC: {result.auc:.3f}, WR Δ: {result.improvement_wr:+.1%}, Pips Δ: {result.improvement_pips:+.2f}")
    
    return results


def print_summary(results: List[FoldResult]):
    """Print walk-forward summary."""
    
    if not results:
        print("\n❌ No valid results")
        return
    
    print("\n" + "=" * 90)
    print("  PHASE 3: WALK-FORWARD OUT-OF-SAMPLE VALIDATION RESULTS")
    print("=" * 90)
    
    # Per-fold table
    print(f"\n  {'Fold':<6} {'Test Period':<25} {'Baseline WR':>12} {'Filtered WR':>12} {'Δ WR':>8} {'Δ Pips':>10} {'AUC':>8}")
    print("  " + "─" * 85)
    
    for r in results:
        period = f"{r.test_start} → {r.test_end}"
        print(f"  {r.fold_num:<6} {period:<25} {r.test_baseline_wr:>12.1%} {r.test_filtered_wr:>12.1%} {r.improvement_wr:>+8.1%} {r.improvement_pips:>+10.2f} {r.auc:>8.3f}")
    
    # Aggregated stats
    total_baseline_signals = sum(r.test_signals for r in results)
    total_baseline_wins = sum(r.test_baseline_wins for r in results)
    total_baseline_pips = sum(r.test_baseline_pips for r in results)
    
    total_filtered_trades = sum(r.test_filtered_trades for r in results)
    total_filtered_wins = sum(r.test_filtered_wins for r in results)
    total_filtered_pips = sum(r.test_filtered_pips for r in results)
    
    avg_auc = np.mean([r.auc for r in results])
    avg_improvement_wr = np.mean([r.improvement_wr for r in results])
    avg_improvement_pips = np.mean([r.improvement_pips for r in results])
    
    # Count positive folds
    positive_wr_folds = sum(1 for r in results if r.improvement_wr > 0)
    positive_pips_folds = sum(1 for r in results if r.improvement_pips > 0)
    
    print("\n  " + "─" * 85)
    print(f"\n  AGGREGATED OUT-OF-SAMPLE RESULTS:")
    print(f"  ─────────────────────────────────────────────────")
    print(f"  Total test signals:     {total_baseline_signals:,}")
    print(f"  Total filtered trades:  {total_filtered_trades:,} ({100*total_filtered_trades/total_baseline_signals:.1f}% of signals)")
    print(f"")
    print(f"  BASELINE (all signals):")
    print(f"    Win Rate:    {100*total_baseline_wins/total_baseline_signals:.1f}%")
    print(f"    Total Pips:  {total_baseline_pips:+,.0f}")
    print(f"    Avg Pips:    {total_baseline_pips/total_baseline_signals:+.2f}")
    print(f"")
    print(f"  ML FILTERED (threshold={ML_THRESHOLD}):")
    print(f"    Win Rate:    {100*total_filtered_wins/total_filtered_trades:.1f}%" if total_filtered_trades > 0 else "    Win Rate:    N/A")
    print(f"    Total Pips:  {total_filtered_pips:+,.0f}")
    print(f"    Avg Pips:    {total_filtered_pips/total_filtered_trades:+.2f}" if total_filtered_trades > 0 else "    Avg Pips:    N/A")
    print(f"")
    print(f"  CONSISTENCY:")
    print(f"    Avg AUC:     {avg_auc:.3f}")
    print(f"    Folds with WR improvement:   {positive_wr_folds}/{len(results)}")
    print(f"    Folds with Pips improvement: {positive_pips_folds}/{len(results)}")
    
    # Final verdict
    print("\n" + "=" * 90)
    print("  VERDICT")
    print("=" * 90)
    
    if avg_auc > 0.52 and positive_pips_folds >= len(results) * 0.6 and total_filtered_pips > total_baseline_pips * 0.5:
        print(f"""
  🟢 WALK-FORWARD VALIDATION PASSED!
  
  The ML filter shows CONSISTENT out-of-sample improvement:
  • Average AUC: {avg_auc:.3f} (> 0.52)
  • {positive_pips_folds}/{len(results)} folds improved avg pips
  • Win rate improved in {positive_wr_folds}/{len(results)} folds
  
  ✅ READY FOR LIVE TESTING
  
  Recommended Settings:
  • Signal: MTF Trend Aligned Long (H4 uptrend + D1 uptrend)
  • Filter: ML probability >= {ML_THRESHOLD}
  • TP: {TRADE_CONFIG['tp_pips']} pips
  • SL: {TRADE_CONFIG['sl_pips']} pips
  • Max Hold: {TRADE_CONFIG['max_hold_bars']} bars (H4)
        """)
    elif avg_auc > 0.50 and positive_pips_folds >= 2:
        print(f"""
  🟡 PARTIAL VALIDATION
  
  The ML filter shows SOME improvement but inconsistent:
  • Average AUC: {avg_auc:.3f}
  • {positive_pips_folds}/{len(results)} folds improved
  
  Consider:
  • More features
  • Different threshold
  • Ensemble models
        """)
    else:
        print(f"""
  🔴 VALIDATION FAILED
  
  The ML filter does NOT show consistent OOS improvement:
  • Average AUC: {avg_auc:.3f}
  • Only {positive_pips_folds}/{len(results)} folds improved
  
  The in-sample improvement was likely overfitting.
        """)
    
    print("=" * 90)


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
    
    # Label all signals
    print(f"\n📊 Labeling signals...")
    df = label_signal_data(df, TRADE_CONFIG)
    
    labeled_count = df["trade_label"].notna().sum()
    win_count = (df["trade_label"] == 1).sum()
    print(f"   Total labeled: {labeled_count:,}")
    print(f"   Wins: {win_count:,} ({100*win_count/labeled_count:.1f}%)")
    
    # Get feature columns
    feature_cols = get_feature_columns(df)
    print(f"\n📋 Using {len(feature_cols)} features")
    
    # Run walk-forward validation
    results = run_walk_forward_validation(df, feature_cols)
    
    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()
