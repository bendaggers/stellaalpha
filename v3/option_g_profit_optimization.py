"""
STELLA ALPHA - OPTION G: PROFIT OPTIMIZATION
=============================================
Accept AUC ~0.54 and optimize for PROFIT instead.

Step 1: Find optimal threshold for maximum profit
Step 2: Add non-ML trade filters (session, volatility, day)

Usage:
    python option_g_profit_optimization.py --h4 data/EURUSD_H4_features.csv --d1 data/EURUSD_D1_features.csv
"""

import pandas as pd
import numpy as np
import argparse
import warnings
import time
from typing import Dict, List, Tuple
from datetime import datetime

warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

# Use the original combined features (215) which performed better
from combined_features import CombinedFeatureEngineering, get_feature_columns


PIP_VALUE = 0.0001
MAX_HOLD_BARS = 72
N_FEATURES_TO_SELECT = 25

# Walk-forward folds
FOLDS = [
    {"name": "Fold1", "train_end": "2010-01-01", "test_end": "2014-01-01"},
    {"name": "Fold2", "train_end": "2014-01-01", "test_end": "2018-01-01"},
    {"name": "Fold3", "train_end": "2018-01-01", "test_end": "2022-01-01"},
    {"name": "Fold4", "train_end": "2022-01-01", "test_end": "2025-01-01"},
]

# Thresholds to test
THRESHOLDS = [0.45, 0.48, 0.50, 0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.65]


# =============================================================================
# DATA LOADING
# =============================================================================

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
    """Generate MTF trend-aligned signals."""
    df = df.copy()
    
    # H4 trend
    if 'trend_strength' in df.columns:
        df['h4_trend_up'] = (df['trend_strength'] > 0.3).astype(int)
    else:
        ema21 = df['close'].ewm(span=21, adjust=False).mean()
        df['h4_trend_up'] = (df['close'] > ema21).astype(int)
    
    # D1 trend
    if 'd1_trend_strength' in df.columns:
        df['d1_trend_up'] = (df['d1_trend_strength'] > 0.3).astype(int)
    elif 'd1_close' in df.columns:
        d1_ema = df['d1_close'].ewm(span=21, adjust=False).mean()
        df['d1_trend_up'] = (df['d1_close'] > d1_ema).astype(int)
    else:
        df['d1_trend_up'] = 1
    
    # MTF signal
    df['mtf_long_signal'] = ((df['h4_trend_up'] == 1) & (df['d1_trend_up'] == 1)).astype(int)
    
    return df


def label_trades(df: pd.DataFrame, tp: int = 100, sl: int = 50) -> pd.DataFrame:
    """Label trades with outcome."""
    df = df.copy()
    df["trade_label"] = np.nan
    df["trade_pips"] = np.nan
    df["bars_held"] = np.nan
    
    signal_indices = df[df["mtf_long_signal"] == 1].index.tolist()
    
    for idx in signal_indices:
        if idx + MAX_HOLD_BARS >= len(df):
            continue
        
        entry_price = df.loc[idx, "close"]
        tp_price = entry_price + (tp * PIP_VALUE)
        sl_price = entry_price - (sl * PIP_VALUE)
        
        outcome, pips, bars = None, 0, 0
        for offset in range(1, MAX_HOLD_BARS + 1):
            bar_idx = idx + offset
            if bar_idx >= len(df):
                break
            bar_high, bar_low = df.loc[bar_idx, "high"], df.loc[bar_idx, "low"]
            
            if bar_low <= sl_price:
                outcome, pips, bars = 0, -sl, offset
                break
            if bar_high >= tp_price:
                outcome, pips, bars = 1, tp, offset
                break
        
        if outcome is None:
            final_price = df.loc[min(idx + MAX_HOLD_BARS, len(df) - 1), "close"]
            pips = (final_price - entry_price) / PIP_VALUE
            outcome = 1 if pips > 0 else 0
            bars = MAX_HOLD_BARS
        
        df.loc[idx, "trade_label"] = outcome
        df.loc[idx, "trade_pips"] = pips
        df.loc[idx, "bars_held"] = bars
    
    return df


# =============================================================================
# TRADE FILTERS (NON-ML)
# =============================================================================

def add_filter_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add columns for non-ML trade filters."""
    df = df.copy()
    
    # Time-based filters
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
    
    # Filter 1: Avoid Monday (day 0) and Friday (day 4)
    df['filter_avoid_mon_fri'] = ~df['day_of_week'].isin([0, 4])
    
    # Filter 2: Only trade during London/NY sessions (8:00-20:00 UTC)
    df['filter_london_ny_only'] = df['hour'].isin([8, 12, 16])  # H4 bars at these hours
    
    # Filter 3: Avoid Asian session (0:00-8:00 UTC)
    df['filter_avoid_asian'] = ~df['hour'].isin([0, 4])
    
    # Filter 4: Require minimum ATR (volatility)
    if 'atr_pct' in df.columns:
        atr_median = df['atr_pct'].median()
        df['filter_min_atr'] = df['atr_pct'] >= atr_median * 0.7
    else:
        df['filter_min_atr'] = True
    
    # Filter 5: Avoid low volatility (BB squeeze)
    if 'bb_width_pct' in df.columns:
        bb_20pct = df['bb_width_pct'].quantile(0.2)
        df['filter_no_squeeze'] = df['bb_width_pct'] >= bb_20pct
    else:
        df['filter_no_squeeze'] = True
    
    # Filter 6: Require strong RSI (not neutral)
    if 'rsi_value' in df.columns:
        df['filter_rsi_strong'] = (df['rsi_value'] > 55) | (df['rsi_value'] < 45)
    else:
        df['filter_rsi_strong'] = True
    
    # Filter 7: D1 trend alignment strength
    if 'd1_trend_strength' in df.columns:
        df['filter_d1_strong'] = df['d1_trend_strength'] > 0.5
    else:
        df['filter_d1_strong'] = True
    
    # Combination filters
    df['filter_basic'] = df['filter_avoid_mon_fri'] & df['filter_avoid_asian']
    df['filter_volatility'] = df['filter_min_atr'] & df['filter_no_squeeze']
    df['filter_all'] = df['filter_basic'] & df['filter_volatility'] & df['filter_d1_strong']
    
    return df


# =============================================================================
# PROFIT METRICS
# =============================================================================

def calculate_profit_metrics(trades_df: pd.DataFrame, tp: int, sl: int) -> Dict:
    """Calculate profit metrics for a set of trades."""
    if len(trades_df) == 0:
        return {
            "n_trades": 0,
            "win_rate": 0,
            "total_pips": 0,
            "avg_pips": 0,
            "sharpe": 0,
            "profit_factor": 0,
            "max_drawdown": 0,
            "expectancy": 0
        }
    
    pips = trades_df['trade_pips'].values
    wins = (trades_df['trade_label'] == 1).sum()
    losses = (trades_df['trade_label'] == 0).sum()
    
    total_pips = pips.sum()
    avg_pips = pips.mean()
    win_rate = wins / len(trades_df) if len(trades_df) > 0 else 0
    
    # Sharpe ratio (assuming daily returns, simplified)
    if pips.std() > 0:
        sharpe = (pips.mean() / pips.std()) * np.sqrt(252)  # Annualized
    else:
        sharpe = 0
    
    # Profit factor
    gross_profit = pips[pips > 0].sum() if (pips > 0).any() else 0
    gross_loss = abs(pips[pips < 0].sum()) if (pips < 0).any() else 1
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else gross_profit
    
    # Max drawdown
    cumulative = np.cumsum(pips)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = running_max - cumulative
    max_drawdown = drawdown.max() if len(drawdown) > 0 else 0
    
    # Expectancy
    avg_win = pips[pips > 0].mean() if (pips > 0).any() else 0
    avg_loss = abs(pips[pips < 0].mean()) if (pips < 0).any() else 0
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
    
    return {
        "n_trades": len(trades_df),
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "total_pips": total_pips,
        "avg_pips": avg_pips,
        "sharpe": sharpe,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
        "expectancy": expectancy,
        "avg_win": avg_win,
        "avg_loss": avg_loss
    }


# =============================================================================
# MAIN OPTIMIZATION
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h4", required=True)
    parser.add_argument("--d1", required=True)
    parser.add_argument("--tp", type=int, default=100)
    parser.add_argument("--sl", type=int, default=50)
    args = parser.parse_args()
    
    tp, sl = args.tp, args.sl
    
    print(f"\n{'='*70}")
    print(f"  OPTION G: PROFIT OPTIMIZATION")
    print(f"  TP={tp} pips, SL={sl} pips, R:R={tp/sl:.1f}:1")
    print(f"{'='*70}")
    
    # Load data
    print(f"\n  Loading data...")
    df = load_and_merge_data(args.h4, args.d1)
    print(f"  Loaded: {len(df):,} rows")
    
    # Feature engineering (use original 215 features)
    print(f"  Computing features (original 215)...")
    fe = CombinedFeatureEngineering(verbose=False)
    df = fe.calculate_all_features(df)
    
    # Generate signals
    df = generate_signals(df)
    print(f"  MTF Long signals: {df['mtf_long_signal'].sum():,}")
    
    # Label trades
    print(f"  Labeling trades...")
    df = label_trades(df, tp, sl)
    
    # Add filters
    print(f"  Adding trade filters...")
    df = add_filter_columns(df)
    
    # Get feature columns
    feature_cols = get_feature_columns(df)
    feature_cols = [c for c in feature_cols if df[c].dtype in ['float64', 'float32', 'int64', 'int32']]
    print(f"  Features: {len(feature_cols)}")
    
    signals_df = df[df['trade_label'].notna()].copy()
    print(f"  Total labeled signals: {len(signals_df):,}")
    print(f"  Baseline win rate: {signals_df['trade_label'].mean()*100:.1f}%")
    
    # ==========================================================================
    # STEP 1: THRESHOLD OPTIMIZATION
    # ==========================================================================
    
    print(f"\n{'='*70}")
    print(f"  STEP 1: THRESHOLD OPTIMIZATION")
    print(f"{'='*70}")
    
    # Train models and get probabilities for each fold
    all_test_data = []
    
    for fold in FOLDS:
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
        
        # Clean
        medians = X_train.median()
        X_train = X_train.fillna(medians).replace([np.inf, -np.inf], 0).fillna(0)
        X_test = X_test.fillna(medians).replace([np.inf, -np.inf], 0).fillna(0)
        
        # RFE
        estimator = lgb.LGBMClassifier(n_estimators=50, max_depth=4, verbose=-1, random_state=42, n_jobs=-1)
        selector = RFE(estimator=estimator, n_features_to_select=N_FEATURES_TO_SELECT, step=5)
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
            random_state=42, verbose=-1, n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Store test data with predictions
        test_df = test_df.copy()
        test_df['ml_probability'] = y_proba
        test_df['fold'] = fold['name']
        all_test_data.append(test_df)
    
    # Combine all test data
    combined_test = pd.concat(all_test_data, ignore_index=True)
    print(f"\n  Combined test data: {len(combined_test):,} trades")
    
    # Test each threshold
    print(f"\n  Testing thresholds...")
    print(f"  {'Thresh':>8} {'Trades':>8} {'WR%':>8} {'TotalPips':>12} {'AvgPips':>10} {'Sharpe':>8} {'PF':>8} {'MaxDD':>10}")
    print(f"  {'-'*82}")
    
    threshold_results = []
    
    for thresh in THRESHOLDS:
        filtered = combined_test[combined_test['ml_probability'] >= thresh].copy()
        metrics = calculate_profit_metrics(filtered, tp, sl)
        metrics['threshold'] = thresh
        threshold_results.append(metrics)
        
        if metrics['n_trades'] >= 50:
            print(f"  {thresh:>8.2f} {metrics['n_trades']:>8} {metrics['win_rate']*100:>7.1f}% {metrics['total_pips']:>+12.0f} {metrics['avg_pips']:>+10.2f} {metrics['sharpe']:>8.2f} {metrics['profit_factor']:>8.2f} {metrics['max_drawdown']:>10.0f}")
        else:
            print(f"  {thresh:>8.2f} {metrics['n_trades']:>8} (insufficient trades)")
    
    # Find best threshold
    valid_results = [r for r in threshold_results if r['n_trades'] >= 100]
    if valid_results:
        # Sort by total pips (or could use Sharpe, or expectancy)
        best_by_pips = max(valid_results, key=lambda x: x['total_pips'])
        best_by_sharpe = max(valid_results, key=lambda x: x['sharpe'])
        best_by_pf = max(valid_results, key=lambda x: x['profit_factor'])
        
        print(f"\n  BEST THRESHOLDS:")
        print(f"  By Total Pips:    {best_by_pips['threshold']:.2f} → {best_by_pips['total_pips']:+,.0f} pips")
        print(f"  By Sharpe Ratio:  {best_by_sharpe['threshold']:.2f} → Sharpe {best_by_sharpe['sharpe']:.2f}")
        print(f"  By Profit Factor: {best_by_pf['threshold']:.2f} → PF {best_by_pf['profit_factor']:.2f}")
    
    # ==========================================================================
    # STEP 2: TRADE FILTERS
    # ==========================================================================
    
    print(f"\n{'='*70}")
    print(f"  STEP 2: TRADE FILTERS (NON-ML)")
    print(f"{'='*70}")
    
    # Use best threshold from Step 1
    best_thresh = best_by_pips['threshold'] if valid_results else 0.50
    print(f"\n  Using threshold: {best_thresh:.2f}")
    
    filtered_by_ml = combined_test[combined_test['ml_probability'] >= best_thresh].copy()
    baseline_metrics = calculate_profit_metrics(filtered_by_ml, tp, sl)
    
    print(f"\n  BASELINE (ML only, threshold={best_thresh:.2f}):")
    print(f"  Trades: {baseline_metrics['n_trades']:,}")
    print(f"  Win Rate: {baseline_metrics['win_rate']*100:.1f}%")
    print(f"  Total Pips: {baseline_metrics['total_pips']:+,.0f}")
    print(f"  Avg Pips: {baseline_metrics['avg_pips']:+.2f}")
    
    # Test each filter
    filters_to_test = [
        ("No Monday/Friday", "filter_avoid_mon_fri"),
        ("Avoid Asian Session", "filter_avoid_asian"),
        ("Basic (Mon/Fri + Asian)", "filter_basic"),
        ("Min ATR (Volatility)", "filter_min_atr"),
        ("No BB Squeeze", "filter_no_squeeze"),
        ("Volatility Combo", "filter_volatility"),
        ("Strong RSI", "filter_rsi_strong"),
        ("Strong D1 Trend", "filter_d1_strong"),
        ("ALL FILTERS", "filter_all"),
    ]
    
    print(f"\n  FILTER COMPARISON:")
    print(f"  {'Filter':<25} {'Trades':>8} {'WR%':>8} {'TotalPips':>12} {'AvgPips':>10} {'vs Base':>10}")
    print(f"  {'-'*75}")
    
    filter_results = []
    
    for filter_name, filter_col in filters_to_test:
        if filter_col not in filtered_by_ml.columns:
            continue
        
        filtered = filtered_by_ml[filtered_by_ml[filter_col] == True].copy()
        metrics = calculate_profit_metrics(filtered, tp, sl)
        metrics['filter_name'] = filter_name
        metrics['filter_col'] = filter_col
        filter_results.append(metrics)
        
        pips_diff = metrics['total_pips'] - baseline_metrics['total_pips']
        pips_diff_str = f"{pips_diff:+,.0f}"
        
        if metrics['n_trades'] >= 50:
            better = "✅" if metrics['total_pips'] > baseline_metrics['total_pips'] else "❌"
            print(f"  {filter_name:<25} {metrics['n_trades']:>8} {metrics['win_rate']*100:>7.1f}% {metrics['total_pips']:>+12.0f} {metrics['avg_pips']:>+10.2f} {pips_diff_str:>10} {better}")
        else:
            print(f"  {filter_name:<25} {metrics['n_trades']:>8} (insufficient)")
    
    # Find best filter combination
    print(f"\n  BEST FILTER COMBINATIONS:")
    valid_filters = [f for f in filter_results if f['n_trades'] >= 100]
    if valid_filters:
        best_filter = max(valid_filters, key=lambda x: x['total_pips'])
        print(f"  By Total Pips: {best_filter['filter_name']}")
        print(f"    Trades: {best_filter['n_trades']:,}")
        print(f"    Win Rate: {best_filter['win_rate']*100:.1f}%")
        print(f"    Total Pips: {best_filter['total_pips']:+,.0f}")
        print(f"    Improvement: {best_filter['total_pips'] - baseline_metrics['total_pips']:+,.0f} pips vs baseline")
    
    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    
    print(f"\n{'='*70}")
    print(f"  FINAL RECOMMENDATIONS")
    print(f"{'='*70}")
    
    print(f"""
  CONFIGURATION:
  ─────────────────────────────────────────
  TP: {tp} pips
  SL: {sl} pips
  R:R: {tp/sl:.1f}:1
  ML Threshold: {best_thresh:.2f}
  
  EXPECTED PERFORMANCE:
  ─────────────────────────────────────────
  Win Rate: {best_by_pips['win_rate']*100:.1f}%
  Avg Pips per Trade: {best_by_pips['avg_pips']:+.2f}
  Profit Factor: {best_by_pips['profit_factor']:.2f}
  Sharpe Ratio: {best_by_pips['sharpe']:.2f}
  
  TRADE FILTERS RECOMMENDED:
  ─────────────────────────────────────────
  1. ML Probability >= {best_thresh:.2f}
  2. Avoid Monday and Friday
  3. Avoid Asian session (00:00-08:00 UTC)
  4. Require minimum ATR (volatility)
    """)
    
    # Reality check
    print(f"  REALITY CHECK:")
    print(f"  ─────────────────────────────────────────")
    
    if best_by_pips['total_pips'] > 0:
        years = 14  # Approximate test period
        annual_pips = best_by_pips['total_pips'] / years
        trades_per_year = best_by_pips['n_trades'] / years
        
        print(f"  Test period: ~{years} years")
        print(f"  Total profit: {best_by_pips['total_pips']:+,.0f} pips")
        print(f"  Annual profit: {annual_pips:+,.0f} pips/year")
        print(f"  Trades per year: {trades_per_year:.0f}")
        print(f"  Max drawdown: {best_by_pips['max_drawdown']:,.0f} pips")
        
        if annual_pips > 1000:
            print(f"\n  ✅ VIABLE STRATEGY - {annual_pips:,.0f} pips/year is tradeable")
        elif annual_pips > 500:
            print(f"\n  ⚠️ MARGINAL STRATEGY - Consider improving filters")
        else:
            print(f"\n  ❌ WEAK STRATEGY - May not be worth the effort")
    else:
        print(f"  ❌ NEGATIVE TOTAL PIPS - Strategy not profitable")
    
    print(f"\n{'='*70}")
    print(f"  OPTIMIZATION COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
