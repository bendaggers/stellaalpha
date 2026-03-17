"""
STELLA ALPHA - PHASE 0: H4 + D1 COMBINED SIGNAL TEST
=====================================================

Tests if D1 alignment AMPLIFIES H4 signal edge.

Key insight from previous test:
- Trend Continuation Long: +11.5% edge, p=0.0000 (significant but small effect)
- London Open Trend: +6.7% edge, p=0.0053 (significant but small effect)

Hypothesis: Adding D1 confirmation will:
1. Filter out weak signals (reduce noise)
2. Amplify edge on remaining signals

Usage:
    python test_h4_d1_combined.py --h4 data/EURUSD_H4_features.csv --d1 data/EURUSD_D1_features.csv
"""

import pandas as pd
import numpy as np
from scipy import stats
import argparse
from typing import Tuple, List
from dataclasses import dataclass


# =============================================================================
# CONFIGURATION
# =============================================================================

FORWARD_CONFIG = {
    "horizon_bars": 48,
    "target_atr": 1.0,
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
# DATA LOADING & MERGING
# =============================================================================

def load_and_merge_data(h4_path: str, d1_path: str) -> pd.DataFrame:
    """Load H4 and D1 data, merge safely (no leakage)."""
    
    print(f"\n📂 Loading H4 data from: {h4_path}")
    df_h4 = pd.read_csv(h4_path)
    print(f"   Loaded {len(df_h4):,} H4 rows")
    
    print(f"📂 Loading D1 data from: {d1_path}")
    df_d1 = pd.read_csv(d1_path)
    print(f"   Loaded {len(df_d1):,} D1 rows")
    
    # Parse timestamps
    df_h4["timestamp"] = pd.to_datetime(df_h4["timestamp"])
    df_d1["timestamp"] = pd.to_datetime(df_d1["timestamp"])
    
    # H4 features
    df_h4["hour"] = df_h4["timestamp"].dt.hour
    df_h4["h4_date"] = df_h4["timestamp"].dt.normalize()
    
    # D1 preparation - shift by 1 day (use PREVIOUS day's D1, no leakage)
    df_d1["d1_date"] = df_d1["timestamp"].dt.normalize()
    df_d1["d1_available_date"] = df_d1["d1_date"] + pd.Timedelta(days=1)
    
    # Rename D1 columns
    d1_cols_to_keep = ["d1_available_date"]
    d1_feature_cols = ["bb_position", "rsi_value", "atr_pct", "close", "high", "low", 
                       "upper_band", "lower_band", "middle_band", "trend_strength"]
    
    for col in d1_feature_cols:
        if col in df_d1.columns:
            df_d1[f"d1_{col}"] = df_d1[col]
            d1_cols_to_keep.append(f"d1_{col}")
    
    df_d1_slim = df_d1[d1_cols_to_keep].copy()
    
    # Merge H4 with D1 (use previous day's D1 data)
    df_h4 = df_h4.sort_values("timestamp").reset_index(drop=True)
    df_d1_slim = df_d1_slim.sort_values("d1_available_date").reset_index(drop=True)
    
    df = pd.merge_asof(
        df_h4,
        df_d1_slim,
        left_on="h4_date",
        right_on="d1_available_date",
        direction="backward"
    )
    
    print(f"   After merge: {len(df):,} rows")
    
    # VALIDATE NO DATA LEAKAGE
    print(f"\n   🔍 Validating no data leakage...")
    df["h4_timestamp"] = df["timestamp"]
    
    # Check: D1 date must be STRICTLY BEFORE H4 date
    # D1 candle closes at midnight of NEXT day, so:
    # - H4 on Jan 16 04:00 can use D1 from Jan 15 (closed at Jan 16 00:00)
    # - H4 on Jan 15 20:00 can only use D1 from Jan 14 (Jan 15 D1 not closed yet)
    
    if "d1_available_date" in df.columns:
        # d1_available_date is already shifted +1 day from actual D1 date
        # So actual D1 date = d1_available_date - 1 day
        df["actual_d1_date"] = df["d1_available_date"] - pd.Timedelta(days=1)
        
        # H4 timestamp must be >= d1_available_date (D1 must be closed before H4 entry)
        violations = df[df["h4_timestamp"] < df["d1_available_date"]]
        
        if len(violations) > 0:
            print(f"   ❌ DATA LEAKAGE DETECTED: {len(violations)} rows use future D1 data!")
            print(f"      Example violation:")
            print(f"      H4 time: {violations.iloc[0]['h4_timestamp']}")
            print(f"      D1 available: {violations.iloc[0]['d1_available_date']}")
            raise ValueError("Data leakage detected! Aborting.")
        else:
            print(f"   ✅ No leakage: All {len(df):,} rows use only PAST D1 data")
            
            # Show example
            sample = df.iloc[len(df)//2]
            print(f"      Example: H4 {sample['h4_timestamp']} uses D1 from {sample['actual_d1_date'].date()}")
    
    # Compute derived features
    df = compute_features(df)
    
    # Drop warmup
    required_cols = ["atr_pct", "ma_20", "ma_50", "rsi_value", "d1_rsi_value"]
    existing_required = [c for c in required_cols if c in df.columns]
    df = df.dropna(subset=existing_required).reset_index(drop=True)
    
    print(f"   After cleanup: {len(df):,} rows")
    
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all needed features."""
    
    # ATR
    if "atr_pct" not in df.columns:
        df["atr"] = (df["high"] - df["low"]).rolling(14).mean()
        df["atr_pct"] = df["atr"] / df["close"]
    
    # Moving averages
    if "ma_20" not in df.columns:
        df["ma_20"] = df["close"].rolling(20).mean()
    if "ma_50" not in df.columns:
        df["ma_50"] = df["close"].rolling(50).mean()
    
    # Volatility regime
    df["vol_percentile"] = df["atr_pct"].rolling(100).apply(
        lambda x: stats.percentileofscore(x.dropna(), x.iloc[-1]) / 100 if len(x.dropna()) > 0 else 0.5
    )
    df["vol_regime"] = pd.cut(
        df["vol_percentile"].fillna(0.5),
        bins=[0, 0.33, 0.67, 1.0],
        labels=["low", "medium", "high"]
    )
    
    # H4 trend
    if "trend_strength" in df.columns:
        df["h4_uptrend"] = df["trend_strength"] > 0.3
        df["h4_downtrend"] = df["trend_strength"] < -0.3
    else:
        df["h4_uptrend"] = df["close"] > df["ma_50"]
        df["h4_downtrend"] = df["close"] < df["ma_50"]
    
    # D1 trend
    if "d1_trend_strength" in df.columns:
        df["d1_uptrend"] = df["d1_trend_strength"] > 0.3
        df["d1_downtrend"] = df["d1_trend_strength"] < -0.3
    elif "d1_close" in df.columns:
        d1_ma20 = df["d1_close"].rolling(20).mean()
        df["d1_uptrend"] = df["d1_close"] > d1_ma20
        df["d1_downtrend"] = df["d1_close"] < d1_ma20
    
    # D1 BB position categories
    if "d1_bb_position" in df.columns:
        df["d1_bb_high"] = df["d1_bb_position"] >= 0.7
        df["d1_bb_low"] = df["d1_bb_position"] <= 0.3
        df["d1_bb_middle"] = (df["d1_bb_position"] > 0.3) & (df["d1_bb_position"] < 0.7)
    
    # D1 RSI categories
    if "d1_rsi_value" in df.columns:
        df["d1_rsi_bullish"] = df["d1_rsi_value"] >= 55
        df["d1_rsi_bearish"] = df["d1_rsi_value"] <= 45
        df["d1_rsi_neutral"] = (df["d1_rsi_value"] > 45) & (df["d1_rsi_value"] < 55)
    
    # Session
    if "hour" in df.columns:
        df["is_london"] = df["hour"].isin([8, 9, 10, 11])
        df["is_ny"] = df["hour"].isin([13, 14, 15, 16])
        df["is_overlap"] = df["hour"].isin([13, 14, 15, 16])
    
    # MTF Alignment
    if "h4_uptrend" in df.columns and "d1_uptrend" in df.columns:
        df["mtf_bullish_aligned"] = df["h4_uptrend"] & df["d1_uptrend"]
        df["mtf_bearish_aligned"] = df["h4_downtrend"] & df["d1_downtrend"]
    
    # H4 pullback to middle BB
    if "bb_position" in df.columns:
        df["h4_pullback_zone"] = (df["bb_position"] >= 0.35) & (df["bb_position"] <= 0.65)
    
    return df


# =============================================================================
# SIGNAL DEFINITIONS (H4 + D1 Combined)
# =============================================================================

def signal_mtf_trend_long(df: pd.DataFrame) -> Tuple[pd.Series, str]:
    """H4 uptrend + D1 uptrend aligned → Long"""
    if "mtf_bullish_aligned" not in df.columns:
        return pd.Series([False] * len(df)), "long"
    mask = df["mtf_bullish_aligned"]
    return mask, "long"


def signal_mtf_trend_short(df: pd.DataFrame) -> Tuple[pd.Series, str]:
    """H4 downtrend + D1 downtrend aligned → Short"""
    if "mtf_bearish_aligned" not in df.columns:
        return pd.Series([False] * len(df)), "short"
    mask = df["mtf_bearish_aligned"]
    return mask, "short"


def signal_mtf_pullback_long(df: pd.DataFrame) -> Tuple[pd.Series, str]:
    """D1 uptrend + H4 pullback to middle BB → Long"""
    if "d1_uptrend" not in df.columns or "h4_pullback_zone" not in df.columns:
        return pd.Series([False] * len(df)), "long"
    mask = df["d1_uptrend"] & df["h4_pullback_zone"]
    return mask, "long"


def signal_mtf_pullback_short(df: pd.DataFrame) -> Tuple[pd.Series, str]:
    """D1 downtrend + H4 pullback to middle BB → Short"""
    if "d1_downtrend" not in df.columns or "h4_pullback_zone" not in df.columns:
        return pd.Series([False] * len(df)), "short"
    mask = df["d1_downtrend"] & df["h4_pullback_zone"]
    return mask, "short"


def signal_d1_trend_h4_rsi_oversold_long(df: pd.DataFrame) -> Tuple[pd.Series, str]:
    """D1 uptrend + H4 RSI oversold → Long (buy the dip)"""
    if "d1_uptrend" not in df.columns:
        return pd.Series([False] * len(df)), "long"
    d1_up = df["d1_uptrend"]
    h4_oversold = df["rsi_value"] <= 35
    mask = d1_up & h4_oversold
    return mask, "long"


def signal_d1_trend_h4_rsi_overbought_short(df: pd.DataFrame) -> Tuple[pd.Series, str]:
    """D1 downtrend + H4 RSI overbought → Short (sell the rally)"""
    if "d1_downtrend" not in df.columns:
        return pd.Series([False] * len(df)), "short"
    d1_down = df["d1_downtrend"]
    h4_overbought = df["rsi_value"] >= 65
    mask = d1_down & h4_overbought
    return mask, "short"


def signal_d1_rsi_bullish_h4_breakout_long(df: pd.DataFrame) -> Tuple[pd.Series, str]:
    """D1 RSI bullish + H4 breaks above upper BB → Long breakout"""
    if "d1_rsi_bullish" not in df.columns:
        return pd.Series([False] * len(df)), "long"
    d1_bullish = df["d1_rsi_bullish"]
    h4_breakout = df["close"] > df["upper_band"]
    mask = d1_bullish & h4_breakout
    return mask, "long"


def signal_d1_rsi_bearish_h4_breakdown_short(df: pd.DataFrame) -> Tuple[pd.Series, str]:
    """D1 RSI bearish + H4 breaks below lower BB → Short breakdown"""
    if "d1_rsi_bearish" not in df.columns:
        return pd.Series([False] * len(df)), "short"
    d1_bearish = df["d1_rsi_bearish"]
    h4_breakdown = df["close"] < df["lower_band"]
    mask = d1_bearish & h4_breakdown
    return mask, "short"


def signal_london_d1_trend_long(df: pd.DataFrame) -> Tuple[pd.Series, str]:
    """London session + D1 uptrend → Long"""
    if "is_london" not in df.columns or "d1_uptrend" not in df.columns:
        return pd.Series([False] * len(df)), "long"
    mask = df["is_london"] & df["d1_uptrend"]
    return mask, "long"


def signal_london_d1_trend_short(df: pd.DataFrame) -> Tuple[pd.Series, str]:
    """London session + D1 downtrend → Short"""
    if "is_london" not in df.columns or "d1_downtrend" not in df.columns:
        return pd.Series([False] * len(df)), "short"
    mask = df["is_london"] & df["d1_downtrend"]
    return mask, "short"


def signal_d1_bb_high_h4_reversal_short(df: pd.DataFrame) -> Tuple[pd.Series, str]:
    """D1 at upper BB + H4 shows reversal signs → Short"""
    if "d1_bb_high" not in df.columns:
        return pd.Series([False] * len(df)), "short"
    d1_high = df["d1_bb_high"]
    h4_reversal = (df["bb_position"] >= 0.8) & (df["rsi_value"] >= 60)
    mask = d1_high & h4_reversal
    return mask, "short"


def signal_d1_bb_low_h4_reversal_long(df: pd.DataFrame) -> Tuple[pd.Series, str]:
    """D1 at lower BB + H4 shows reversal signs → Long"""
    if "d1_bb_low" not in df.columns:
        return pd.Series([False] * len(df)), "long"
    d1_low = df["d1_bb_low"]
    h4_reversal = (df["bb_position"] <= 0.2) & (df["rsi_value"] <= 40)
    mask = d1_low & h4_reversal
    return mask, "long"


def signal_strong_d1_trend_any_h4_long(df: pd.DataFrame) -> Tuple[pd.Series, str]:
    """Strong D1 uptrend (RSI > 60) → Long on any H4"""
    if "d1_rsi_value" not in df.columns:
        return pd.Series([False] * len(df)), "long"
    mask = df["d1_rsi_value"] >= 60
    return mask, "long"


def signal_strong_d1_trend_any_h4_short(df: pd.DataFrame) -> Tuple[pd.Series, str]:
    """Strong D1 downtrend (RSI < 40) → Short on any H4"""
    if "d1_rsi_value" not in df.columns:
        return pd.Series([False] * len(df)), "short"
    mask = df["d1_rsi_value"] <= 40
    return mask, "short"


def signal_mtf_confluence_strong_long(df: pd.DataFrame) -> Tuple[pd.Series, str]:
    """Multiple D1+H4 factors aligned bullish"""
    if "d1_uptrend" not in df.columns or "d1_rsi_bullish" not in df.columns:
        return pd.Series([False] * len(df)), "long"
    
    d1_bullish = df["d1_uptrend"] & df["d1_rsi_bullish"]
    h4_bullish = df["h4_uptrend"] & (df["rsi_value"] >= 50)
    
    mask = d1_bullish & h4_bullish
    return mask, "long"


def signal_mtf_confluence_strong_short(df: pd.DataFrame) -> Tuple[pd.Series, str]:
    """Multiple D1+H4 factors aligned bearish"""
    if "d1_downtrend" not in df.columns or "d1_rsi_bearish" not in df.columns:
        return pd.Series([False] * len(df)), "short"
    
    d1_bearish = df["d1_downtrend"] & df["d1_rsi_bearish"]
    h4_bearish = df["h4_downtrend"] & (df["rsi_value"] <= 50)
    
    mask = d1_bearish & h4_bearish
    return mask, "short"


# All signals
SIGNALS = [
    ("MTF Trend Aligned Long", signal_mtf_trend_long),
    ("MTF Trend Aligned Short", signal_mtf_trend_short),
    ("MTF Pullback Long (D1 up + H4 mid BB)", signal_mtf_pullback_long),
    ("MTF Pullback Short (D1 down + H4 mid BB)", signal_mtf_pullback_short),
    ("D1 Uptrend + H4 RSI Oversold → Long", signal_d1_trend_h4_rsi_oversold_long),
    ("D1 Downtrend + H4 RSI Overbought → Short", signal_d1_trend_h4_rsi_overbought_short),
    ("D1 RSI Bullish + H4 BB Breakout → Long", signal_d1_rsi_bullish_h4_breakout_long),
    ("D1 RSI Bearish + H4 BB Breakdown → Short", signal_d1_rsi_bearish_h4_breakdown_short),
    ("London + D1 Uptrend → Long", signal_london_d1_trend_long),
    ("London + D1 Downtrend → Short", signal_london_d1_trend_short),
    ("D1 BB High + H4 Reversal → Short", signal_d1_bb_high_h4_reversal_short),
    ("D1 BB Low + H4 Reversal → Long", signal_d1_bb_low_h4_reversal_long),
    ("Strong D1 Uptrend (RSI>60) → Long", signal_strong_d1_trend_any_h4_long),
    ("Strong D1 Downtrend (RSI<40) → Short", signal_strong_d1_trend_any_h4_short),
    ("MTF Strong Confluence Long", signal_mtf_confluence_strong_long),
    ("MTF Strong Confluence Short", signal_mtf_confluence_strong_short),
]


# =============================================================================
# TESTING FUNCTIONS
# =============================================================================

def calculate_forward_mfe(df: pd.DataFrame, indices: np.ndarray, direction: str) -> np.ndarray:
    """Calculate MFE in ATR units."""
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
        else:
            highest = forward_slice["high"].max()
            mfe = (highest - entry_price) / entry_atr
        
        mfe_values.append(mfe)
    
    return np.array(mfe_values)


def create_matched_baseline(df: pd.DataFrame, signal_mask: pd.Series, n_samples: int) -> np.ndarray:
    """Create volatility-matched random baseline."""
    horizon = FORWARD_CONFIG["horizon_bars"]
    non_signal_mask = ~signal_mask & (df.index < len(df) - horizon)
    
    signal_df = df[signal_mask]
    
    if "vol_regime" not in df.columns or signal_df["vol_regime"].isna().all():
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


def test_signal(df: pd.DataFrame, name: str, signal_func) -> SignalResult:
    """Test a single signal."""
    horizon = FORWARD_CONFIG["horizon_bars"]
    
    try:
        signal_mask, direction = signal_func(df)
    except Exception as e:
        print(f"   ⚠️ Error: {e}")
        return None
    
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
    
    signal_indices = df[signal_mask].index.values
    signal_mfe = calculate_forward_mfe(df, signal_indices, direction)
    
    baseline_indices = create_matched_baseline(df, signal_mask, n_signals)
    baseline_mfe = calculate_forward_mfe(df, baseline_indices, direction)
    
    signal_mfe = signal_mfe[~np.isnan(signal_mfe)]
    baseline_mfe = baseline_mfe[~np.isnan(baseline_mfe)]
    
    if len(signal_mfe) < 50 or len(baseline_mfe) < 50:
        return SignalResult(
            name=name, direction=direction, n_signals=n_signals,
            signal_mean_mfe=0, baseline_mean_mfe=0, edge=0, edge_pct=0,
            p_value=1.0, effect_size=0, signal_win_rate=0, baseline_win_rate=0,
            passes=False
        )
    
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
    
    passes = (
        n_signals >= ACCEPTANCE["min_samples"] and
        edge_pct >= ACCEPTANCE["min_edge_pct"] and
        p_value < ACCEPTANCE["p_value"] and
        effect_size >= ACCEPTANCE["effect_size"]
    )
    
    return SignalResult(
        name=name, direction=direction, n_signals=n_signals,
        signal_mean_mfe=signal_mean, baseline_mean_mfe=baseline_mean,
        edge=edge, edge_pct=edge_pct, p_value=p_value, effect_size=effect_size,
        signal_win_rate=signal_win_rate, baseline_win_rate=baseline_win_rate,
        passes=passes
    )


def print_results(results: List[SignalResult]):
    """Print all results."""
    print("\n" + "=" * 85)
    print("  PHASE 0: H4 + D1 COMBINED SIGNAL TEST RESULTS")
    print("=" * 85)
    
    results = sorted([r for r in results if r], key=lambda x: x.edge_pct, reverse=True)
    
    print(f"\n  {'Signal':<45} {'N':>6} {'Edge%':>8} {'P-val':>8} {'Effect':>8} {'Pass':>6}")
    print("  " + "─" * 80)
    
    for r in results:
        pass_str = "✅ YES" if r.passes else "❌ NO"
        print(f"  {r.name:<45} {r.n_signals:>6} {r.edge_pct:>+7.1f}% {r.p_value:>8.4f} {r.effect_size:>+8.3f} {pass_str}")
    
    passing = [r for r in results if r.passes]
    
    print("\n" + "=" * 85)
    print("  SUMMARY")
    print("=" * 85)
    
    if passing:
        print(f"\n  🟢 FOUND {len(passing)} SIGNAL(S) WITH EDGE:\n")
        for r in passing:
            print(f"     ✅ {r.name}")
            print(f"        Direction: {r.direction.upper()}")
            print(f"        Samples: {r.n_signals:,}")
            print(f"        Edge: {r.edge_pct:+.1f}% vs matched random")
            print(f"        P-value: {r.p_value:.4f}")
            print(f"        Effect size: {r.effect_size:.3f}")
            print(f"        Win rate: {r.signal_win_rate*100:.1f}% vs {r.baseline_win_rate*100:.1f}% baseline")
            print()
        
        print("  ✅ NEXT STEP: Build simple model using best signal")
    else:
        # Show near-misses
        near_miss = [r for r in results if r.edge_pct >= 3.0 or r.p_value < 0.05]
        if near_miss:
            print(f"\n  🟡 NO FULL PASSES, but {len(near_miss)} near-miss signals:\n")
            for r in near_miss[:5]:
                reasons = []
                if r.n_signals < ACCEPTANCE["min_samples"]:
                    reasons.append(f"samples {r.n_signals} < 200")
                if r.edge_pct < ACCEPTANCE["min_edge_pct"]:
                    reasons.append(f"edge {r.edge_pct:.1f}% < 3%")
                if r.p_value >= ACCEPTANCE["p_value"]:
                    reasons.append(f"p-value {r.p_value:.3f} >= 0.05")
                if r.effect_size < ACCEPTANCE["effect_size"]:
                    reasons.append(f"effect {r.effect_size:.3f} < 0.2")
                
                print(f"     🟡 {r.name}")
                print(f"        Edge: {r.edge_pct:+.1f}%, Effect: {r.effect_size:.3f}")
                print(f"        Failed: {', '.join(reasons)}")
                print()
        else:
            print("\n  🔴 NO SIGNALS FOUND WITH EDGE")
            print("     D1 alignment did not help find edge.")
    
    print("=" * 85 + "\n")
    
    return passing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h4", required=True, help="Path to H4 CSV")
    parser.add_argument("--d1", required=True, help="Path to D1 CSV")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    # Load and merge
    df = load_and_merge_data(args.h4, args.d1)
    
    # Test signals
    print(f"\n🔬 Testing {len(SIGNALS)} H4+D1 combined signals...\n")
    
    results = []
    for name, func in SIGNALS:
        print(f"   Testing: {name}...")
        result = test_signal(df, name, func)
        results.append(result)
        if result:
            status = "✅" if result.passes else "❌"
            print(f"   {status} N={result.n_signals}, Edge={result.edge_pct:+.1f}%, Effect={result.effect_size:+.3f}")
    
    passing = print_results(results)
    return len(passing) > 0


if __name__ == "__main__":
    main()
