"""
STELLA ALPHA - PHASE 1: SIMPLE MTF TREND BACKTEST
==================================================

We found real edge in:
- MTF Trend Aligned Short: +11.6% edge, p=0.0000
- MTF Trend Aligned Long: +8.5% edge, p=0.0000

This script:
1. Implements the signal as a trading system
2. Backtests with realistic TP/SL in pips
3. Validates the edge translates to profit

NO ML YET - just validate the edge is tradeable.

Usage:
    python backtest_mtf_trend.py --h4 data/EURUSD_H4_features.csv --d1 data/EURUSD_D1_features.csv
"""

import pandas as pd
import numpy as np
from scipy import stats
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TradeConfig:
    """Trading configuration to test."""
    tp_pips: int
    sl_pips: int
    max_hold_bars: int
    
    @property
    def rr_ratio(self) -> float:
        return self.tp_pips / self.sl_pips
    
    def __str__(self):
        return f"TP{self.tp_pips}_SL{self.sl_pips}_H{self.max_hold_bars}"


# Configs to test (various TP/SL combinations)
CONFIGS_TO_TEST = [
    # Tight stops (scalp-like)
    TradeConfig(tp_pips=30, sl_pips=30, max_hold_bars=24),
    TradeConfig(tp_pips=40, sl_pips=30, max_hold_bars=36),
    TradeConfig(tp_pips=50, sl_pips=30, max_hold_bars=48),
    
    # Balanced
    TradeConfig(tp_pips=50, sl_pips=50, max_hold_bars=48),
    TradeConfig(tp_pips=60, sl_pips=40, max_hold_bars=48),
    TradeConfig(tp_pips=80, sl_pips=50, max_hold_bars=72),
    
    # Runners (high R:R)
    TradeConfig(tp_pips=100, sl_pips=50, max_hold_bars=72),
    TradeConfig(tp_pips=100, sl_pips=40, max_hold_bars=72),
    TradeConfig(tp_pips=120, sl_pips=50, max_hold_bars=96),
]

PIP_VALUE = 0.0001  # For EURUSD


# =============================================================================
# DATA LOADING (same as before, with leakage check)
# =============================================================================

def load_and_merge_data(h4_path: str, d1_path: str) -> pd.DataFrame:
    """Load H4 and D1 data, merge safely (no leakage)."""
    
    print(f"\n📂 Loading data...")
    df_h4 = pd.read_csv(h4_path)
    df_d1 = pd.read_csv(d1_path)
    print(f"   H4: {len(df_h4):,} rows | D1: {len(df_d1):,} rows")
    
    # Parse timestamps
    df_h4["timestamp"] = pd.to_datetime(df_h4["timestamp"])
    df_d1["timestamp"] = pd.to_datetime(df_d1["timestamp"])
    
    df_h4["hour"] = df_h4["timestamp"].dt.hour
    df_h4["h4_date"] = df_h4["timestamp"].dt.normalize()
    
    # D1: shift by 1 day (use PREVIOUS day, no leakage)
    df_d1["d1_date"] = df_d1["timestamp"].dt.normalize()
    df_d1["d1_available_date"] = df_d1["d1_date"] + pd.Timedelta(days=1)
    
    # Rename D1 columns
    d1_cols_to_keep = ["d1_available_date"]
    for col in ["bb_position", "rsi_value", "trend_strength", "close"]:
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
    if "d1_available_date" in df.columns:
        violations = df[df["timestamp"] < df["d1_available_date"]]
        if len(violations) > 0:
            raise ValueError(f"Data leakage detected: {len(violations)} rows!")
        print(f"   ✅ No data leakage confirmed")
    
    # Compute features
    df = compute_features(df)
    
    # Drop rows with NaN in key columns
    required = ["close", "high", "low", "d1_trend_strength", "trend_strength"]
    existing = [c for c in required if c in df.columns]
    df = df.dropna(subset=existing).reset_index(drop=True)
    
    print(f"   Final: {len(df):,} rows")
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute trend features."""
    
    # H4 trend
    if "trend_strength" in df.columns:
        df["h4_uptrend"] = df["trend_strength"] > 0.3
        df["h4_downtrend"] = df["trend_strength"] < -0.3
    else:
        df["ma_50"] = df["close"].rolling(50).mean()
        df["h4_uptrend"] = df["close"] > df["ma_50"]
        df["h4_downtrend"] = df["close"] < df["ma_50"]
    
    # D1 trend
    if "d1_trend_strength" in df.columns:
        df["d1_uptrend"] = df["d1_trend_strength"] > 0.3
        df["d1_downtrend"] = df["d1_trend_strength"] < -0.3
    else:
        df["d1_uptrend"] = False
        df["d1_downtrend"] = False
    
    # MTF alignment
    df["mtf_long_signal"] = df["h4_uptrend"] & df["d1_uptrend"]
    df["mtf_short_signal"] = df["h4_downtrend"] & df["d1_downtrend"]
    
    return df


# =============================================================================
# BACKTESTING ENGINE
# =============================================================================

@dataclass
class Trade:
    """Single trade record."""
    entry_idx: int
    entry_time: pd.Timestamp
    entry_price: float
    direction: str  # "long" or "short"
    exit_idx: int
    exit_time: pd.Timestamp
    exit_price: float
    exit_reason: str  # "TP", "SL", "TIMEOUT"
    pips: float
    bars_held: int


def simulate_trade(
    df: pd.DataFrame,
    entry_idx: int,
    direction: str,
    config: TradeConfig
) -> Optional[Trade]:
    """Simulate a single trade from entry to exit."""
    
    if entry_idx + config.max_hold_bars >= len(df):
        return None
    
    entry_price = df.loc[entry_idx, "close"]
    entry_time = df.loc[entry_idx, "timestamp"]
    
    tp_price = entry_price + (config.tp_pips * PIP_VALUE * (1 if direction == "long" else -1))
    sl_price = entry_price - (config.sl_pips * PIP_VALUE * (1 if direction == "long" else -1))
    
    # Simulate bar by bar
    for offset in range(1, config.max_hold_bars + 1):
        idx = entry_idx + offset
        if idx >= len(df):
            break
        
        bar_high = df.loc[idx, "high"]
        bar_low = df.loc[idx, "low"]
        bar_close = df.loc[idx, "close"]
        bar_time = df.loc[idx, "timestamp"]
        
        if direction == "long":
            # Check SL first (conservative)
            if bar_low <= sl_price:
                pips = -config.sl_pips
                return Trade(
                    entry_idx=entry_idx, entry_time=entry_time, entry_price=entry_price,
                    direction=direction, exit_idx=idx, exit_time=bar_time,
                    exit_price=sl_price, exit_reason="SL", pips=pips, bars_held=offset
                )
            # Check TP
            if bar_high >= tp_price:
                pips = config.tp_pips
                return Trade(
                    entry_idx=entry_idx, entry_time=entry_time, entry_price=entry_price,
                    direction=direction, exit_idx=idx, exit_time=bar_time,
                    exit_price=tp_price, exit_reason="TP", pips=pips, bars_held=offset
                )
        else:  # short
            # Check SL first
            if bar_high >= sl_price:
                pips = -config.sl_pips
                return Trade(
                    entry_idx=entry_idx, entry_time=entry_time, entry_price=entry_price,
                    direction=direction, exit_idx=idx, exit_time=bar_time,
                    exit_price=sl_price, exit_reason="SL", pips=pips, bars_held=offset
                )
            # Check TP
            if bar_low <= tp_price:
                pips = config.tp_pips
                return Trade(
                    entry_idx=entry_idx, entry_time=entry_time, entry_price=entry_price,
                    direction=direction, exit_idx=idx, exit_time=bar_time,
                    exit_price=tp_price, exit_reason="TP", pips=pips, bars_held=offset
                )
    
    # Timeout - exit at close
    final_idx = min(entry_idx + config.max_hold_bars, len(df) - 1)
    final_close = df.loc[final_idx, "close"]
    final_time = df.loc[final_idx, "timestamp"]
    
    if direction == "long":
        pips = (final_close - entry_price) / PIP_VALUE
    else:
        pips = (entry_price - final_close) / PIP_VALUE
    
    return Trade(
        entry_idx=entry_idx, entry_time=entry_time, entry_price=entry_price,
        direction=direction, exit_idx=final_idx, exit_time=final_time,
        exit_price=final_close, exit_reason="TIMEOUT", pips=pips,
        bars_held=final_idx - entry_idx
    )


def backtest_signal(
    df: pd.DataFrame,
    signal_mask: pd.Series,
    direction: str,
    config: TradeConfig
) -> List[Trade]:
    """Backtest a signal with given config."""
    
    trades = []
    signal_indices = df[signal_mask].index.tolist()
    
    last_exit_idx = -1
    
    for entry_idx in signal_indices:
        # Skip if we're still in a trade
        if entry_idx <= last_exit_idx:
            continue
        
        trade = simulate_trade(df, entry_idx, direction, config)
        if trade:
            trades.append(trade)
            last_exit_idx = trade.exit_idx
    
    return trades


def backtest_random_baseline(
    df: pd.DataFrame,
    n_trades: int,
    direction: str,
    config: TradeConfig,
    seed: int = 42
) -> List[Trade]:
    """Backtest random entries for baseline comparison."""
    
    np.random.seed(seed)
    
    max_idx = len(df) - config.max_hold_bars - 1
    random_indices = np.random.choice(range(0, max_idx), size=n_trades * 2, replace=False)
    random_indices = sorted(random_indices)
    
    trades = []
    last_exit_idx = -1
    
    for entry_idx in random_indices:
        if len(trades) >= n_trades:
            break
        if entry_idx <= last_exit_idx:
            continue
        
        trade = simulate_trade(df, entry_idx, direction, config)
        if trade:
            trades.append(trade)
            last_exit_idx = trade.exit_idx
    
    return trades


# =============================================================================
# ANALYSIS
# =============================================================================

@dataclass
class BacktestResult:
    """Results for one config."""
    config: TradeConfig
    direction: str
    
    # Signal performance
    signal_trades: int
    signal_wins: int
    signal_win_rate: float
    signal_total_pips: float
    signal_avg_pips: float
    signal_profit_factor: float
    
    # Random baseline
    baseline_trades: int
    baseline_wins: int
    baseline_win_rate: float
    baseline_total_pips: float
    baseline_avg_pips: float
    
    # Edge comparison
    edge_win_rate: float
    edge_pips_per_trade: float
    edge_pct: float
    p_value: float
    
    # Verdict
    is_profitable: bool
    beats_random: bool


def analyze_trades(trades: List[Trade], config: TradeConfig) -> dict:
    """Analyze trade list."""
    if not trades:
        return {
            "n_trades": 0, "wins": 0, "win_rate": 0, "total_pips": 0,
            "avg_pips": 0, "profit_factor": 0, "pips_list": []
        }
    
    pips_list = [t.pips for t in trades]
    wins = sum(1 for p in pips_list if p > 0)
    losses = sum(1 for p in pips_list if p < 0)
    
    gross_profit = sum(p for p in pips_list if p > 0)
    gross_loss = abs(sum(p for p in pips_list if p < 0))
    
    return {
        "n_trades": len(trades),
        "wins": wins,
        "win_rate": wins / len(trades) if trades else 0,
        "total_pips": sum(pips_list),
        "avg_pips": np.mean(pips_list) if pips_list else 0,
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else float('inf'),
        "pips_list": pips_list
    }


def run_backtest_comparison(
    df: pd.DataFrame,
    signal_mask: pd.Series,
    direction: str,
    config: TradeConfig
) -> BacktestResult:
    """Run backtest comparing signal vs random."""
    
    # Backtest signal
    signal_trades = backtest_signal(df, signal_mask, direction, config)
    signal_stats = analyze_trades(signal_trades, config)
    
    # Backtest random baseline (same number of trades)
    baseline_trades = backtest_random_baseline(
        df, signal_stats["n_trades"], direction, config
    )
    baseline_stats = analyze_trades(baseline_trades, config)
    
    # Statistical comparison
    if signal_stats["pips_list"] and baseline_stats["pips_list"]:
        t_stat, p_value = stats.ttest_ind(
            signal_stats["pips_list"], 
            baseline_stats["pips_list"]
        )
    else:
        p_value = 1.0
    
    # Edge calculations
    edge_win_rate = signal_stats["win_rate"] - baseline_stats["win_rate"]
    edge_pips = signal_stats["avg_pips"] - baseline_stats["avg_pips"]
    edge_pct = (edge_pips / abs(baseline_stats["avg_pips"]) * 100) if baseline_stats["avg_pips"] != 0 else 0
    
    return BacktestResult(
        config=config,
        direction=direction,
        signal_trades=signal_stats["n_trades"],
        signal_wins=signal_stats["wins"],
        signal_win_rate=signal_stats["win_rate"],
        signal_total_pips=signal_stats["total_pips"],
        signal_avg_pips=signal_stats["avg_pips"],
        signal_profit_factor=signal_stats["profit_factor"],
        baseline_trades=baseline_stats["n_trades"],
        baseline_wins=baseline_stats["wins"],
        baseline_win_rate=baseline_stats["win_rate"],
        baseline_total_pips=baseline_stats["total_pips"],
        baseline_avg_pips=baseline_stats["avg_pips"],
        edge_win_rate=edge_win_rate,
        edge_pips_per_trade=edge_pips,
        edge_pct=edge_pct,
        p_value=p_value,
        is_profitable=signal_stats["total_pips"] > 0,
        beats_random=signal_stats["avg_pips"] > baseline_stats["avg_pips"] and p_value < 0.1
    )


# =============================================================================
# MAIN
# =============================================================================

def print_results(results: List[BacktestResult]):
    """Print all results."""
    
    print("\n" + "=" * 100)
    print("  PHASE 1: MTF TREND BACKTEST RESULTS")
    print("=" * 100)
    
    # Group by direction
    for direction in ["long", "short"]:
        dir_results = [r for r in results if r.direction == direction]
        if not dir_results:
            continue
        
        print(f"\n  {'📈 LONG' if direction == 'long' else '📉 SHORT'} SIGNALS:")
        print("  " + "─" * 95)
        print(f"  {'Config':<20} {'Trades':>7} {'WinRate':>8} {'TotalPips':>10} {'AvgPips':>8} {'vs Random':>10} {'P-val':>7} {'Status':>10}")
        print("  " + "─" * 95)
        
        for r in sorted(dir_results, key=lambda x: x.signal_total_pips, reverse=True):
            status = "✅ PROFIT" if r.is_profitable and r.beats_random else "⚠️ WEAK" if r.is_profitable else "❌ LOSS"
            edge_str = f"{r.edge_pips_per_trade:+.1f} pip"
            
            print(f"  {str(r.config):<20} {r.signal_trades:>7} {r.signal_win_rate:>7.1%} {r.signal_total_pips:>+10.0f} {r.signal_avg_pips:>+8.2f} {edge_str:>10} {r.p_value:>7.3f} {status:>10}")
    
    # Summary
    print("\n" + "=" * 100)
    print("  SUMMARY")
    print("=" * 100)
    
    profitable = [r for r in results if r.is_profitable and r.beats_random]
    
    if profitable:
        print(f"\n  🟢 FOUND {len(profitable)} PROFITABLE CONFIG(S) THAT BEAT RANDOM:\n")
        
        best = max(profitable, key=lambda x: x.signal_total_pips)
        print(f"     BEST CONFIG: {best.config}")
        print(f"     Direction:   {best.direction.upper()}")
        print(f"     Trades:      {best.signal_trades:,}")
        print(f"     Win Rate:    {best.signal_win_rate:.1%}")
        print(f"     Total Pips:  {best.signal_total_pips:+,.0f}")
        print(f"     Avg Pips:    {best.signal_avg_pips:+.2f} per trade")
        print(f"     vs Random:   {best.edge_pips_per_trade:+.2f} pips better")
        print(f"     P-value:     {best.p_value:.4f}")
        print(f"     PF:          {best.signal_profit_factor:.2f}")
        
        print(f"\n  ✅ NEXT STEP: Add ML to filter signals and improve edge")
    else:
        # Check if any are profitable at all
        any_profit = [r for r in results if r.is_profitable]
        if any_profit:
            print(f"\n  🟡 {len(any_profit)} configs profitable but don't clearly beat random")
            print("     The edge is weak - ML might help extract more signal")
        else:
            print("\n  🔴 No profitable configs found")
            print("     The signal edge doesn't translate to tradeable profit with fixed TP/SL")
    
    print("\n" + "=" * 100)
    
    return profitable


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h4", required=True, help="Path to H4 CSV")
    parser.add_argument("--d1", required=True, help="Path to D1 CSV")
    args = parser.parse_args()
    
    # Load data
    df = load_and_merge_data(args.h4, args.d1)
    
    # Get signals
    long_signal = df["mtf_long_signal"]
    short_signal = df["mtf_short_signal"]
    
    print(f"\n📊 Signal counts:")
    print(f"   Long signals:  {long_signal.sum():,}")
    print(f"   Short signals: {short_signal.sum():,}")
    
    # Run backtests
    print(f"\n🔬 Testing {len(CONFIGS_TO_TEST)} configurations...")
    
    results = []
    
    for config in CONFIGS_TO_TEST:
        print(f"   Testing {config}...")
        
        # Test long
        result_long = run_backtest_comparison(df, long_signal, "long", config)
        results.append(result_long)
        
        # Test short
        result_short = run_backtest_comparison(df, short_signal, "short", config)
        results.append(result_short)
    
    # Print results
    profitable = print_results(results)
    
    return len(profitable) > 0


if __name__ == "__main__":
    main()
