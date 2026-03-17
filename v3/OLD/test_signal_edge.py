"""
STELLA ALPHA - PHASE 0: SIGNAL EDGE TEST
=========================================

THE ONLY QUESTION: Does BB/RSI overbought signal beat random entry?

Usage:
    python test_signal_edge.py --data path/to/EURUSD_H4_features.csv

Output:
    SIGNAL HAS EDGE: YES or NO
"""

import pandas as pd
import numpy as np
from scipy import stats
import argparse
from pathlib import Path


# =============================================================================
# CONFIGURATION (Simple)
# =============================================================================

SIGNAL_CONFIG = {
    "bb_threshold": 0.80,      # BB position >= this
    "rsi_threshold": 65,       # RSI >= this
    "direction": "short",      # We're testing mean reversion short
}

FORWARD_CONFIG = {
    "horizon_bars": 48,        # Look 48 bars forward (8 days)
    "target_atr": 1.0,         # Target: 1.0 ATR move
}

ACCEPTANCE = {
    "min_samples": 200,        # Need at least 200 signals
    "min_edge_pct": 3.0,       # Signal must beat random by 3%+
    "p_value": 0.05,           # Statistical significance
    "effect_size": 0.2,        # Cohen's d minimum
}


# =============================================================================
# MAIN FUNCTIONS
# =============================================================================

def load_data(filepath: str) -> pd.DataFrame:
    """Load H4 data."""
    print(f"\n📂 Loading data from: {filepath}")
    
    df = pd.read_csv(filepath)
    print(f"   Loaded {len(df):,} rows")
    
    # Parse timestamp
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Check required columns
    required = ["close", "high", "low", "bb_position", "rsi_value"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    # Compute ATR if not present
    if "atr_pct" not in df.columns:
        print("   Computing ATR...")
        df["atr"] = (df["high"] - df["low"]).rolling(14).mean()
        df["atr_pct"] = df["atr"] / df["close"]
    
    # Compute volatility regime for matching
    df["vol_percentile"] = df["atr_pct"].rolling(100).apply(
        lambda x: stats.percentileofscore(x.dropna(), x.iloc[-1]) / 100
    )
    df["vol_regime"] = pd.cut(
        df["vol_percentile"], 
        bins=[0, 0.33, 0.67, 1.0],
        labels=["low", "medium", "high"]
    )
    
    # Drop warmup rows
    df = df.dropna(subset=["atr_pct", "vol_regime"]).reset_index(drop=True)
    print(f"   After warmup: {len(df):,} rows")
    
    return df


def find_signals(df: pd.DataFrame) -> pd.Series:
    """Find all signal occurrences."""
    bb_cond = df["bb_position"] >= SIGNAL_CONFIG["bb_threshold"]
    rsi_cond = df["rsi_value"] >= SIGNAL_CONFIG["rsi_threshold"]
    
    # Can't be in last N bars (need forward data)
    horizon = FORWARD_CONFIG["horizon_bars"]
    valid_range = df.index < (len(df) - horizon)
    
    signal_mask = bb_cond & rsi_cond & valid_range
    
    return signal_mask


def calculate_forward_mfe(df: pd.DataFrame, indices: np.ndarray) -> np.ndarray:
    """
    Calculate Maximum Favorable Excursion in ATR units.
    
    For SHORT: MFE = (entry - lowest_low) / ATR
    """
    horizon = FORWARD_CONFIG["horizon_bars"]
    mfe_values = []
    
    for idx in indices:
        entry_price = df.loc[idx, "close"]
        entry_atr = df.loc[idx, "atr_pct"] * entry_price
        
        # Get forward bars
        forward_slice = df.loc[idx+1 : idx+horizon]
        
        if len(forward_slice) < horizon:
            mfe_values.append(np.nan)
            continue
        
        if SIGNAL_CONFIG["direction"] == "short":
            lowest = forward_slice["low"].min()
            mfe = (entry_price - lowest) / entry_atr
        else:
            highest = forward_slice["high"].max()
            mfe = (highest - entry_price) / entry_atr
        
        mfe_values.append(mfe)
    
    return np.array(mfe_values)


def create_matched_baseline(df: pd.DataFrame, signal_mask: pd.Series, n_samples: int) -> np.ndarray:
    """
    Create random baseline MATCHED by volatility regime.
    
    This is the key fix for Risk #2.
    """
    signal_df = df[signal_mask]
    non_signal_mask = ~signal_mask & (df.index < len(df) - FORWARD_CONFIG["horizon_bars"])
    
    # Get regime distribution of signals
    regime_counts = signal_df["vol_regime"].value_counts(normalize=True)
    
    baseline_indices = []
    
    for regime, pct in regime_counts.items():
        # Find non-signal candles in this regime
        regime_mask = non_signal_mask & (df["vol_regime"] == regime)
        available = df[regime_mask].index.tolist()
        
        if len(available) == 0:
            continue
        
        # Sample proportionally
        n_to_sample = int(n_samples * pct)
        sampled = np.random.choice(available, size=min(n_to_sample, len(available)), replace=False)
        baseline_indices.extend(sampled)
    
    return np.array(baseline_indices)


def compare_signal_to_baseline(signal_mfe: np.ndarray, baseline_mfe: np.ndarray) -> dict:
    """
    Statistical comparison of signal vs baseline.
    """
    # Clean NaN
    signal_mfe = signal_mfe[~np.isnan(signal_mfe)]
    baseline_mfe = baseline_mfe[~np.isnan(baseline_mfe)]
    
    # Basic stats
    signal_mean = signal_mfe.mean()
    baseline_mean = baseline_mfe.mean()
    edge = signal_mean - baseline_mean
    edge_pct = (edge / baseline_mean) * 100 if baseline_mean > 0 else 0
    
    # T-test
    t_stat, p_value = stats.ttest_ind(signal_mfe, baseline_mfe)
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((signal_mfe.std()**2 + baseline_mfe.std()**2) / 2)
    effect_size = edge / pooled_std if pooled_std > 0 else 0
    
    # Win rate at target
    target = FORWARD_CONFIG["target_atr"]
    signal_win_rate = (signal_mfe >= target).mean()
    baseline_win_rate = (baseline_mfe >= target).mean()
    
    return {
        "signal_mean_mfe": signal_mean,
        "baseline_mean_mfe": baseline_mean,
        "edge": edge,
        "edge_pct": edge_pct,
        "t_stat": t_stat,
        "p_value": p_value,
        "effect_size": effect_size,
        "signal_win_rate": signal_win_rate,
        "baseline_win_rate": baseline_win_rate,
        "win_rate_edge": signal_win_rate - baseline_win_rate,
    }


def print_results(n_signals: int, n_baseline: int, results: dict):
    """Print results in clear format."""
    
    print("\n" + "=" * 60)
    print("  PHASE 0: SIGNAL EDGE TEST RESULTS")
    print("=" * 60)
    
    print(f"""
  SIGNAL DEFINITION:
  ─────────────────────────────────────────
  BB Position >= {SIGNAL_CONFIG['bb_threshold']}
  RSI Value   >= {SIGNAL_CONFIG['rsi_threshold']}
  Direction:     {SIGNAL_CONFIG['direction'].upper()}
  
  SAMPLE SIZES:
  ─────────────────────────────────────────
  Signal occurrences:    {n_signals:,}
  Matched baseline:      {n_baseline:,}
  
  FORWARD MFE ({FORWARD_CONFIG['horizon_bars']} bars, in ATR units):
  ─────────────────────────────────────────
  Signal mean MFE:       {results['signal_mean_mfe']:.3f} ATR
  Baseline mean MFE:     {results['baseline_mean_mfe']:.3f} ATR
  Edge:                  {results['edge']:+.3f} ATR ({results['edge_pct']:+.1f}%)
  
  WIN RATE (reaching {FORWARD_CONFIG['target_atr']} ATR):
  ─────────────────────────────────────────
  Signal win rate:       {results['signal_win_rate']*100:.1f}%
  Baseline win rate:     {results['baseline_win_rate']*100:.1f}%
  Win rate edge:         {results['win_rate_edge']*100:+.1f}%
  
  STATISTICAL SIGNIFICANCE:
  ─────────────────────────────────────────
  T-statistic:           {results['t_stat']:.3f}
  P-value:               {results['p_value']:.4f}
  Effect size (Cohen d): {results['effect_size']:.3f}
    """)
    
    # Verdict
    print("  VERDICT:")
    print("  ─────────────────────────────────────────")
    
    checks = []
    
    # Check 1: Enough samples
    if n_signals >= ACCEPTANCE["min_samples"]:
        checks.append(("✅", f"Sample size: {n_signals} >= {ACCEPTANCE['min_samples']}"))
    else:
        checks.append(("❌", f"Sample size: {n_signals} < {ACCEPTANCE['min_samples']}"))
    
    # Check 2: Positive edge
    if results["edge_pct"] >= ACCEPTANCE["min_edge_pct"]:
        checks.append(("✅", f"Edge: {results['edge_pct']:.1f}% >= {ACCEPTANCE['min_edge_pct']}%"))
    else:
        checks.append(("❌", f"Edge: {results['edge_pct']:.1f}% < {ACCEPTANCE['min_edge_pct']}%"))
    
    # Check 3: Statistical significance
    if results["p_value"] < ACCEPTANCE["p_value"]:
        checks.append(("✅", f"P-value: {results['p_value']:.4f} < {ACCEPTANCE['p_value']}"))
    else:
        checks.append(("❌", f"P-value: {results['p_value']:.4f} >= {ACCEPTANCE['p_value']}"))
    
    # Check 4: Effect size
    if results["effect_size"] >= ACCEPTANCE["effect_size"]:
        checks.append(("✅", f"Effect size: {results['effect_size']:.3f} >= {ACCEPTANCE['effect_size']}"))
    else:
        checks.append(("❌", f"Effect size: {results['effect_size']:.3f} < {ACCEPTANCE['effect_size']}"))
    
    for emoji, msg in checks:
        print(f"  {emoji} {msg}")
    
    # Final verdict
    all_passed = all(c[0] == "✅" for c in checks)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("  🟢 SIGNAL HAS EDGE: YES")
        print("     Proceed to build model")
    else:
        print("  🔴 SIGNAL HAS EDGE: NO")
        print("     DO NOT proceed - signal does not beat random")
    print("=" * 60 + "\n")
    
    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Test if BB/RSI signal has edge vs random")
    parser.add_argument("--data", "-d", required=True, help="Path to H4 CSV file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    # 1. Load data
    df = load_data(args.data)
    
    # 2. Find signals
    signal_mask = find_signals(df)
    n_signals = signal_mask.sum()
    print(f"\n🎯 Found {n_signals:,} signal occurrences")
    
    if n_signals < 50:
        print("❌ Not enough signals to test. Exiting.")
        return False
    
    # 3. Calculate signal MFE
    print("📊 Calculating signal forward MFE...")
    signal_indices = df[signal_mask].index.values
    signal_mfe = calculate_forward_mfe(df, signal_indices)
    
    # 4. Create matched baseline
    print("🎲 Creating matched random baseline...")
    baseline_indices = create_matched_baseline(df, signal_mask, n_signals)
    baseline_mfe = calculate_forward_mfe(df, baseline_indices)
    
    # 5. Compare
    print("📈 Comparing signal to baseline...")
    results = compare_signal_to_baseline(signal_mfe, baseline_mfe)
    
    # 6. Print results
    has_edge = print_results(n_signals, len(baseline_indices), results)
    
    return has_edge


if __name__ == "__main__":
    main()
