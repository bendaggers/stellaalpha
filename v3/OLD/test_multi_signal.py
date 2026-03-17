"""
STELLA ALPHA - PHASE 0: MULTI-SIGNAL EDGE TEST
===============================================

Tests MULTIPLE signal hypotheses to find ANY edge.

Signals tested:
1. BB/RSI Overbought (original - already failed)
2. BB/RSI Oversold (opposite direction)
3. Volatility Squeeze → Breakout
4. RSI Divergence
5. Price vs MA Distance (mean reversion)
6. Session-based (London open, etc.)
7. Trend Continuation (BB + trend alignment)

Usage:
    python test_multi_signal.py --data path/to/EURUSD_H4_features.csv

Output:
    Summary of ALL signals tested with edge results
"""

import pandas as pd
import numpy as np
from scipy import stats
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Callable
from dataclasses import dataclass


# =============================================================================
# CONFIGURATION
# =============================================================================

FORWARD_CONFIG = {
    "horizon_bars": 48,        # Look 48 bars forward
    "target_atr": 1.0,         # Target: 1.0 ATR move
}

ACCEPTANCE = {
    "min_samples": 200,
    "min_edge_pct": 3.0,
    "p_value": 0.05,
    "effect_size": 0.2,
}


@dataclass
class SignalResult:
    name: str
    direction: str
    n_signals: int
    signal_mean_mfe: float
    baseline_mean_mfe: float
    edge: float
    edge_pct: float
    p_value: float
    effect_size: float
    signal_win_rate: float
    baseline_win_rate: float
    passes: bool


# =============================================================================
# DATA LOADING & PREPARATION
# =============================================================================

def load_and_prepare_data(filepath: str) -> pd.DataFrame:
    """Load and prepare H4 data with all needed features."""
    print(f"\n📂 Loading data from: {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"   Loaded {len(df):,} rows")
    
    # Parse timestamp
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
    
    # Compute ATR if not present
    if "atr_pct" not in df.columns:
        df["atr"] = (df["high"] - df["low"]).rolling(14).mean()
        df["atr_pct"] = df["atr"] / df["close"]
    
    # Compute BB width if not present
    if "bb_width_pct" not in df.columns and "upper_band" in df.columns:
        df["bb_width_pct"] = (df["upper_band"] - df["lower_band"]) / df["middle_band"]
    
    # Compute moving averages if not present
    if "ma_20" not in df.columns:
        df["ma_20"] = df["close"].rolling(20).mean()
    if "ma_50" not in df.columns:
        df["ma_50"] = df["close"].rolling(50).mean()
    
    # Price distance from MA
    df["dist_from_ma20"] = (df["close"] - df["ma_20"]) / df["ma_20"]
    df["dist_from_ma50"] = (df["close"] - df["ma_50"]) / df["ma_50"]
    
    # RSI momentum (for divergence)
    df["rsi_change_5"] = df["rsi_value"].diff(5)
    df["price_change_5"] = df["close"].pct_change(5)
    
    # BB width percentile (for squeeze detection)
    df["bb_width_percentile"] = df["bb_width_pct"].rolling(100).apply(
        lambda x: stats.percentileofscore(x.dropna(), x.iloc[-1]) / 100 if len(x.dropna()) > 0 else 0.5
    )
    
    # Volatility regime
    df["vol_percentile"] = df["atr_pct"].rolling(100).apply(
        lambda x: stats.percentileofscore(x.dropna(), x.iloc[-1]) / 100 if len(x.dropna()) > 0 else 0.5
    )
    df["vol_regime"] = pd.cut(
        df["vol_percentile"].fillna(0.5), 
        bins=[0, 0.33, 0.67, 1.0],
        labels=["low", "medium", "high"]
    )
    
    # Trend direction
    if "trend_strength" in df.columns:
        df["is_uptrend"] = df["trend_strength"] > 0.5
        df["is_downtrend"] = df["trend_strength"] < -0.5
    else:
        df["is_uptrend"] = df["close"] > df["ma_50"]
        df["is_downtrend"] = df["close"] < df["ma_50"]
    
    # Session flags
    if "hour" in df.columns:
        df["is_london_open"] = df["hour"].isin([8, 9, 10, 11])  # London morning
        df["is_ny_open"] = df["hour"].isin([13, 14, 15, 16])    # NY morning (UTC)
        df["is_asian"] = df["hour"].isin([0, 1, 2, 3, 4, 5, 6, 7])
        df["is_overlap"] = df["hour"].isin([13, 14, 15, 16])     # London/NY overlap
    
    # Drop warmup
    df = df.dropna(subset=["atr_pct", "ma_20", "ma_50", "rsi_value"]).reset_index(drop=True)
    print(f"   After preparation: {len(df):,} rows")
    
    return df


# =============================================================================
# SIGNAL DEFINITIONS
# =============================================================================

def signal_bb_rsi_overbought(df: pd.DataFrame) -> Tuple[pd.Series, str]:
    """Original signal: BB high + RSI high → short"""
    mask = (df["bb_position"] >= 0.85) & (df["rsi_value"] >= 65)
    return mask, "short"


def signal_bb_rsi_oversold(df: pd.DataFrame) -> Tuple[pd.Series, str]:
    """Opposite: BB low + RSI low → long"""
    mask = (df["bb_position"] <= 0.15) & (df["rsi_value"] <= 35)
    return mask, "long"


def signal_bb_squeeze_breakout_up(df: pd.DataFrame) -> Tuple[pd.Series, str]:
    """BB squeeze (tight bands) then price breaks up → long"""
    squeeze = df["bb_width_percentile"] <= 0.20  # Tight bands
    breakout = df["close"] > df["upper_band"]     # Price breaks upper band
    prior_squeeze = squeeze.shift(1).fillna(False) | squeeze.shift(2).fillna(False)
    mask = prior_squeeze & breakout
    return mask, "long"


def signal_bb_squeeze_breakout_down(df: pd.DataFrame) -> Tuple[pd.Series, str]:
    """BB squeeze then price breaks down → short"""
    squeeze = df["bb_width_percentile"] <= 0.20
    breakout = df["close"] < df["lower_band"]
    prior_squeeze = squeeze.shift(1).fillna(False) | squeeze.shift(2).fillna(False)
    mask = prior_squeeze & breakout
    return mask, "short"


def signal_rsi_bearish_divergence(df: pd.DataFrame) -> Tuple[pd.Series, str]:
    """Price making higher highs but RSI making lower highs → short"""
    price_up = df["price_change_5"] > 0.005  # Price up 0.5%+
    rsi_down = df["rsi_change_5"] < -5        # RSI down 5+ points
    rsi_high = df["rsi_value"] >= 60          # RSI still elevated
    mask = price_up & rsi_down & rsi_high
    return mask, "short"


def signal_rsi_bullish_divergence(df: pd.DataFrame) -> Tuple[pd.Series, str]:
    """Price making lower lows but RSI making higher lows → long"""
    price_down = df["price_change_5"] < -0.005
    rsi_up = df["rsi_change_5"] > 5
    rsi_low = df["rsi_value"] <= 40
    mask = price_down & rsi_up & rsi_low
    return mask, "long"


def signal_extreme_dist_from_ma_short(df: pd.DataFrame) -> Tuple[pd.Series, str]:
    """Price far above MA20 → mean reversion short"""
    mask = df["dist_from_ma20"] >= 0.015  # 1.5%+ above MA
    return mask, "short"


def signal_extreme_dist_from_ma_long(df: pd.DataFrame) -> Tuple[pd.Series, str]:
    """Price far below MA20 → mean reversion long"""
    mask = df["dist_from_ma20"] <= -0.015  # 1.5%+ below MA
    return mask, "long"


def signal_london_open_trend(df: pd.DataFrame) -> Tuple[pd.Series, str]:
    """London open + existing trend → continuation"""
    if "is_london_open" not in df.columns:
        return pd.Series([False] * len(df)), "long"
    
    london = df["is_london_open"]
    uptrend = df["is_uptrend"]
    mask = london & uptrend
    return mask, "long"


def signal_london_open_reversal(df: pd.DataFrame) -> Tuple[pd.Series, str]:
    """London open + price at BB extreme → reversal"""
    if "is_london_open" not in df.columns:
        return pd.Series([False] * len(df)), "short"
    
    london = df["is_london_open"]
    overbought = df["bb_position"] >= 0.85
    mask = london & overbought
    return mask, "short"


def signal_asian_range_breakout(df: pd.DataFrame) -> Tuple[pd.Series, str]:
    """End of Asian session + breakout → continuation"""
    if "is_asian" not in df.columns:
        return pd.Series([False] * len(df)), "long"
    
    # First non-Asian bar after Asian session
    was_asian = df["is_asian"].shift(1).fillna(False)
    not_asian_now = ~df["is_asian"]
    transition = was_asian & not_asian_now
    
    # Price above recent high (breakout)
    recent_high = df["high"].rolling(6).max().shift(1)  # Last 6 bars high
    breakout = df["close"] > recent_high
    
    mask = transition & breakout
    return mask, "long"


def signal_trend_continuation_long(df: pd.DataFrame) -> Tuple[pd.Series, str]:
    """Strong uptrend + pullback to BB middle → long"""
    uptrend = df["is_uptrend"]
    pullback = (df["bb_position"] >= 0.40) & (df["bb_position"] <= 0.60)  # Near middle
    rsi_ok = df["rsi_value"] >= 40  # Not oversold
    mask = uptrend & pullback & rsi_ok
    return mask, "long"


def signal_trend_continuation_short(df: pd.DataFrame) -> Tuple[pd.Series, str]:
    """Strong downtrend + pullback to BB middle → short"""
    downtrend = df["is_downtrend"]
    pullback = (df["bb_position"] >= 0.40) & (df["bb_position"] <= 0.60)
    rsi_ok = df["rsi_value"] <= 60
    mask = downtrend & pullback & rsi_ok
    return mask, "short"


def signal_momentum_burst_long(df: pd.DataFrame) -> Tuple[pd.Series, str]:
    """Strong momentum + volume → continuation long"""
    strong_move = df["price_change_5"] > 0.01  # 1%+ move up
    rsi_strong = df["rsi_value"] >= 60
    mask = strong_move & rsi_strong
    return mask, "long"


def signal_momentum_burst_short(df: pd.DataFrame) -> Tuple[pd.Series, str]:
    """Strong momentum down → continuation short"""
    strong_move = df["price_change_5"] < -0.01
    rsi_weak = df["rsi_value"] <= 40
    mask = strong_move & rsi_weak
    return mask, "short"


# All signals to test
SIGNALS = [
    ("BB/RSI Overbought → Short", signal_bb_rsi_overbought),
    ("BB/RSI Oversold → Long", signal_bb_rsi_oversold),
    ("BB Squeeze Breakout Up", signal_bb_squeeze_breakout_up),
    ("BB Squeeze Breakout Down", signal_bb_squeeze_breakout_down),
    ("RSI Bearish Divergence", signal_rsi_bearish_divergence),
    ("RSI Bullish Divergence", signal_rsi_bullish_divergence),
    ("Extreme Dist from MA → Short", signal_extreme_dist_from_ma_short),
    ("Extreme Dist from MA → Long", signal_extreme_dist_from_ma_long),
    ("London Open Trend Continuation", signal_london_open_trend),
    ("London Open Reversal", signal_london_open_reversal),
    ("Asian Range Breakout", signal_asian_range_breakout),
    ("Trend Continuation Long", signal_trend_continuation_long),
    ("Trend Continuation Short", signal_trend_continuation_short),
    ("Momentum Burst Long", signal_momentum_burst_long),
    ("Momentum Burst Short", signal_momentum_burst_short),
]


# =============================================================================
# TESTING FUNCTIONS
# =============================================================================

def calculate_forward_mfe(df: pd.DataFrame, indices: np.ndarray, direction: str) -> np.ndarray:
    """Calculate Maximum Favorable Excursion in ATR units."""
    horizon = FORWARD_CONFIG["horizon_bars"]
    mfe_values = []
    
    for idx in indices:
        if idx + horizon >= len(df):
            mfe_values.append(np.nan)
            continue
            
        entry_price = df.loc[idx, "close"]
        entry_atr = df.loc[idx, "atr_pct"] * entry_price
        
        if entry_atr <= 0:
            mfe_values.append(np.nan)
            continue
        
        forward_slice = df.loc[idx+1 : idx+horizon]
        
        if len(forward_slice) < horizon:
            mfe_values.append(np.nan)
            continue
        
        if direction == "short":
            lowest = forward_slice["low"].min()
            mfe = (entry_price - lowest) / entry_atr
        else:  # long
            highest = forward_slice["high"].max()
            mfe = (highest - entry_price) / entry_atr
        
        mfe_values.append(mfe)
    
    return np.array(mfe_values)


def create_matched_baseline(df: pd.DataFrame, signal_mask: pd.Series, n_samples: int) -> np.ndarray:
    """Create volatility-regime matched random baseline."""
    signal_df = df[signal_mask]
    horizon = FORWARD_CONFIG["horizon_bars"]
    non_signal_mask = ~signal_mask & (df.index < len(df) - horizon)
    
    if "vol_regime" not in df.columns:
        # Fallback: simple random
        available = df[non_signal_mask].index.tolist()
        return np.random.choice(available, size=min(n_samples, len(available)), replace=False)
    
    regime_counts = signal_df["vol_regime"].value_counts(normalize=True)
    baseline_indices = []
    
    for regime, pct in regime_counts.items():
        regime_mask = non_signal_mask & (df["vol_regime"] == regime)
        available = df[regime_mask].index.tolist()
        
        if len(available) == 0:
            continue
        
        n_to_sample = int(n_samples * pct)
        sampled = np.random.choice(available, size=min(n_to_sample, len(available)), replace=False)
        baseline_indices.extend(sampled)
    
    return np.array(baseline_indices)


def test_signal(df: pd.DataFrame, name: str, signal_func: Callable) -> SignalResult:
    """Test a single signal hypothesis."""
    horizon = FORWARD_CONFIG["horizon_bars"]
    
    # Get signal mask and direction
    try:
        signal_mask, direction = signal_func(df)
    except Exception as e:
        print(f"   ⚠️ Error in {name}: {e}")
        return None
    
    # Apply horizon limit
    valid_range = df.index < (len(df) - horizon)
    signal_mask = signal_mask & valid_range
    
    n_signals = signal_mask.sum()
    
    if n_signals < 50:
        return SignalResult(
            name=name, direction=direction, n_signals=n_signals,
            signal_mean_mfe=0, baseline_mean_mfe=0, edge=0, edge_pct=0,
            p_value=1.0, effect_size=0, signal_win_rate=0, baseline_win_rate=0,
            passes=False
        )
    
    # Calculate MFE
    signal_indices = df[signal_mask].index.values
    signal_mfe = calculate_forward_mfe(df, signal_indices, direction)
    
    # Create matched baseline
    baseline_indices = create_matched_baseline(df, signal_mask, n_signals)
    baseline_mfe = calculate_forward_mfe(df, baseline_indices, direction)
    
    # Clean NaN
    signal_mfe = signal_mfe[~np.isnan(signal_mfe)]
    baseline_mfe = baseline_mfe[~np.isnan(baseline_mfe)]
    
    if len(signal_mfe) < 50 or len(baseline_mfe) < 50:
        return SignalResult(
            name=name, direction=direction, n_signals=n_signals,
            signal_mean_mfe=0, baseline_mean_mfe=0, edge=0, edge_pct=0,
            p_value=1.0, effect_size=0, signal_win_rate=0, baseline_win_rate=0,
            passes=False
        )
    
    # Statistics
    signal_mean = signal_mfe.mean()
    baseline_mean = baseline_mfe.mean()
    edge = signal_mean - baseline_mean
    edge_pct = (edge / baseline_mean) * 100 if baseline_mean > 0 else 0
    
    t_stat, p_value = stats.ttest_ind(signal_mfe, baseline_mfe)
    
    pooled_std = np.sqrt((signal_mfe.std()**2 + baseline_mfe.std()**2) / 2)
    effect_size = edge / pooled_std if pooled_std > 0 else 0
    
    target = FORWARD_CONFIG["target_atr"]
    signal_win_rate = (signal_mfe >= target).mean()
    baseline_win_rate = (baseline_mfe >= target).mean()
    
    # Check if passes
    passes = (
        n_signals >= ACCEPTANCE["min_samples"] and
        edge_pct >= ACCEPTANCE["min_edge_pct"] and
        p_value < ACCEPTANCE["p_value"] and
        effect_size >= ACCEPTANCE["effect_size"]
    )
    
    return SignalResult(
        name=name,
        direction=direction,
        n_signals=n_signals,
        signal_mean_mfe=signal_mean,
        baseline_mean_mfe=baseline_mean,
        edge=edge,
        edge_pct=edge_pct,
        p_value=p_value,
        effect_size=effect_size,
        signal_win_rate=signal_win_rate,
        baseline_win_rate=baseline_win_rate,
        passes=passes
    )


# =============================================================================
# MAIN
# =============================================================================

def print_results(results: List[SignalResult]):
    """Print summary of all results."""
    
    print("\n" + "=" * 80)
    print("  PHASE 0: MULTI-SIGNAL EDGE TEST RESULTS")
    print("=" * 80)
    
    # Sort by edge percentage
    results = sorted(results, key=lambda x: x.edge_pct if x else -999, reverse=True)
    
    print(f"\n  {'Signal':<35} {'N':>6} {'Edge%':>8} {'P-val':>8} {'Effect':>8} {'Pass':>6}")
    print("  " + "─" * 75)
    
    for r in results:
        if r is None:
            continue
        
        pass_str = "✅ YES" if r.passes else "❌ NO"
        edge_color = "" if r.edge_pct < 0 else ""
        
        print(f"  {r.name:<35} {r.n_signals:>6} {r.edge_pct:>+7.1f}% {r.p_value:>8.4f} {r.effect_size:>+8.3f} {pass_str}")
    
    # Summary
    passing = [r for r in results if r and r.passes]
    
    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    
    if passing:
        print(f"\n  🟢 FOUND {len(passing)} SIGNAL(S) WITH EDGE:\n")
        for r in passing:
            print(f"     ✅ {r.name}")
            print(f"        Direction: {r.direction.upper()}")
            print(f"        Samples: {r.n_signals:,}")
            print(f"        Edge: {r.edge_pct:+.1f}% vs random")
            print(f"        Effect size: {r.effect_size:.3f}")
            print(f"        Win rate: {r.signal_win_rate*100:.1f}% vs {r.baseline_win_rate*100:.1f}% baseline")
            print()
    else:
        print("\n  🔴 NO SIGNALS FOUND WITH EDGE")
        print("\n     All tested signals perform same as or worse than random.")
        print("     Consider:")
        print("     • Different asset class")
        print("     • Different timeframe")
        print("     • Fundamentally different approach")
    
    print("=" * 80 + "\n")
    
    return passing


def main():
    parser = argparse.ArgumentParser(description="Test multiple signal hypotheses for edge")
    parser.add_argument("--data", "-d", required=True, help="Path to H4 CSV file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    # Load data
    df = load_and_prepare_data(args.data)
    
    # Test all signals
    print(f"\n🔬 Testing {len(SIGNALS)} signal hypotheses...\n")
    
    results = []
    for name, signal_func in SIGNALS:
        print(f"   Testing: {name}...")
        result = test_signal(df, name, signal_func)
        results.append(result)
        
        if result:
            status = "✅" if result.passes else "❌"
            print(f"   {status} N={result.n_signals}, Edge={result.edge_pct:+.1f}%")
    
    # Print summary
    passing = print_results(results)
    
    return len(passing) > 0


if __name__ == "__main__":
    main()
