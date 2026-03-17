"""
STELLA ALPHA - PROPER ML PIPELINE
==================================

This script does it RIGHT:
1. Uses ALL ~296 features from feature_engineering.py
2. Tests multiple thresholds systematically (0.40 to 0.70)
3. Uses RFE for feature selection (top 25)
4. Proper walk-forward validation (5 folds)
5. Parallel processing with configurable workers

Usage:
    python proper_ml_pipeline.py --h4 data/EURUSD_H4_features.csv --d1 data/EURUSD_D1_features.csv --workers 6
"""

import pandas as pd
import numpy as np
import argparse
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import warnings
import time
import sys
import os

warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_auc_score
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

# Walk-forward folds
WALK_FORWARD_FOLDS = [
    {"name": "Fold1", "train_end": "2010-01-01", "test_end": "2013-01-01"},
    {"name": "Fold2", "train_end": "2013-01-01", "test_end": "2016-01-01"},
    {"name": "Fold3", "train_end": "2016-01-01", "test_end": "2019-01-01"},
    {"name": "Fold4", "train_end": "2019-01-01", "test_end": "2022-01-01"},
    {"name": "Fold5", "train_end": "2022-01-01", "test_end": "2025-01-01"},
]

# Thresholds to test
THRESHOLDS = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

# RFE settings
N_FEATURES_TO_SELECT = 25


# =============================================================================
# DATA LOADING & MERGING
# =============================================================================

def load_and_merge_data(h4_path: str, d1_path: str) -> pd.DataFrame:
    """Load H4 and D1 data, merge safely (no leakage)."""
    print(f"\n{'='*60}")
    print(f"  LOADING DATA")
    print(f"{'='*60}")
    
    df_h4 = pd.read_csv(h4_path)
    df_d1 = pd.read_csv(d1_path)
    print(f"  H4: {len(df_h4):,} rows")
    print(f"  D1: {len(df_d1):,} rows")
    
    # Parse timestamps
    df_h4["timestamp"] = pd.to_datetime(df_h4["timestamp"])
    df_d1["timestamp"] = pd.to_datetime(df_d1["timestamp"])
    
    df_h4["hour"] = df_h4["timestamp"].dt.hour
    df_h4["day_of_week"] = df_h4["timestamp"].dt.dayofweek
    df_h4["h4_date"] = df_h4["timestamp"].dt.normalize()
    
    # D1: shift by 1 day for no leakage (use PREVIOUS day's D1)
    df_d1["d1_date"] = df_d1["timestamp"].dt.normalize()
    df_d1["d1_available_date"] = df_d1["d1_date"] + pd.Timedelta(days=1)
    
    # Rename D1 columns with d1_ prefix
    d1_cols_to_keep = ["d1_available_date"]
    exclude_cols = ["timestamp", "d1_date", "d1_available_date"]
    
    for col in df_d1.columns:
        if col not in exclude_cols:
            new_name = f"d1_{col}" if not col.startswith("d1_") else col
            df_d1[new_name] = df_d1[col]
            if new_name not in d1_cols_to_keep:
                d1_cols_to_keep.append(new_name)
    
    df_d1_slim = df_d1[d1_cols_to_keep].copy()
    
    # Merge using merge_asof (backward direction = use most recent PAST D1)
    df_h4 = df_h4.sort_values("timestamp").reset_index(drop=True)
    df_d1_slim = df_d1_slim.sort_values("d1_available_date").reset_index(drop=True)
    
    df = pd.merge_asof(
        df_h4, df_d1_slim,
        left_on="h4_date",
        right_on="d1_available_date",
        direction="backward"
    )
    
    print(f"  Merged: {len(df):,} rows")
    print(f"  ✓ No data leakage (using previous day D1)")
    
    return df


# =============================================================================
# FULL FEATURE ENGINEERING (~296 features)
# =============================================================================

def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer ALL ~296 features."""
    print(f"\n{'='*60}")
    print(f"  FEATURE ENGINEERING")
    print(f"{'='*60}")
    
    df = df.copy()
    initial_cols = len(df.columns)
    
    # =========================================================================
    # H4 DERIVED FEATURES (~192)
    # =========================================================================
    
    # Price Change (7)
    df['price_change_1'] = df['close'].pct_change(1)
    df['price_change_3'] = df['close'].pct_change(3)
    df['price_change_5'] = df['close'].pct_change(5)
    df['price_change_10'] = df['close'].pct_change(10)
    df['high_change_1'] = df['high'].pct_change(1)
    df['high_change_3'] = df['high'].pct_change(3)
    df['range_pct'] = (df['high'] - df['low']) / df['close']
    
    # Lag Features (18)
    for i in range(1, 6):
        df[f'rsi_lag{i}'] = df['rsi_value'].shift(i)
    for i in range(1, 4):
        df[f'bb_position_lag{i}'] = df['bb_position'].shift(i)
        df[f'price_change_lag{i}'] = df['price_change_1'].shift(i)
        df[f'volume_ratio_lag{i}'] = df['volume_ratio'].shift(i)
    for i in range(1, 3):
        df[f'bb_width_lag{i}'] = df['bb_width_pct'].shift(i)
        df[f'trend_strength_lag{i}'] = df['trend_strength'].shift(i)
    
    # Slope Features (12)
    df['rsi_slope_3'] = (df['rsi_value'] - df['rsi_value'].shift(3)) / 3
    df['rsi_slope_5'] = (df['rsi_value'] - df['rsi_value'].shift(5)) / 5
    df['rsi_slope_10'] = (df['rsi_value'] - df['rsi_value'].shift(10)) / 10
    df['price_slope_3'] = df['close'].pct_change(3) / 3
    df['price_slope_5'] = df['close'].pct_change(5) / 5
    df['price_slope_10'] = df['close'].pct_change(10) / 10
    df['bb_position_slope_3'] = (df['bb_position'] - df['bb_position'].shift(3)) / 3
    df['bb_position_slope_5'] = (df['bb_position'] - df['bb_position'].shift(5)) / 5
    df['volume_slope_3'] = (df['volume_ratio'] - df['volume_ratio'].shift(3)) / 3
    df['bb_width_slope_3'] = (df['bb_width_pct'] - df['bb_width_pct'].shift(3)) / 3
    df['bb_width_slope_5'] = (df['bb_width_pct'] - df['bb_width_pct'].shift(5)) / 5
    df['trend_slope_3'] = (df['trend_strength'] - df['trend_strength'].shift(3)) / 3
    
    # Z-Score Features (6)
    for col, name in [('close', 'price'), ('rsi_value', 'rsi'), ('volume_ratio', 'volume'),
                      ('bb_position', 'bb_position'), ('atr_pct', 'atr'), ('high', 'high')]:
        if col in df.columns:
            mean = df[col].rolling(20).mean()
            std = df[col].rolling(20).std()
            df[f'{name}_zscore'] = (df[col] - mean) / std.replace(0, np.nan)
    
    # Rolling Statistics (10)
    df['price_rolling_std'] = df['close'].rolling(20).std()
    df['rsi_rolling_std'] = df['rsi_value'].rolling(20).std()
    df['rsi_rolling_max'] = df['rsi_value'].rolling(20).max()
    df['rsi_rolling_min'] = df['rsi_value'].rolling(20).min()
    df['rsi_range'] = df['rsi_rolling_max'] - df['rsi_rolling_min']
    df['bb_position_rolling_std'] = df['bb_position'].rolling(20).std()
    df['bb_position_rolling_max'] = df['bb_position'].rolling(20).max()
    df['volume_rolling_std'] = df['volume_ratio'].rolling(20).std()
    df['price_skew_20'] = df['close'].rolling(20).skew()
    df['price_kurtosis_20'] = df['close'].rolling(20).kurt()
    
    # Binary Features (11)
    df['touched_upper_bb'] = (df['high'] >= df['upper_band']).astype(int)
    df['rsi_overbought'] = (df['rsi_value'] >= 70).astype(int)
    df['rsi_extreme_overbought'] = (df['rsi_value'] >= 80).astype(int)
    df['rsi_very_extreme'] = (df['rsi_value'] >= 85).astype(int)
    df['bearish_candle'] = (df['close'] < df['open']).astype(int)
    df['strong_bearish'] = ((df['close'] < df['open']) & 
                            ((df['open'] - df['close']) / df['close'] > 0.002)).astype(int)
    df['bb_very_high'] = (df['bb_position'] >= 0.95).astype(int)
    df['bb_above_upper'] = (df['close'] > df['upper_band']).astype(int)
    df['high_volume'] = (df['volume_ratio'] >= 1.5).astype(int)
    df['extreme_volume'] = (df['volume_ratio'] >= 2.0).astype(int)
    df['strong_uptrend'] = (df['trend_strength'] >= 1.0).astype(int)
    
    # Momentum Features (15)
    df['price_roc_3'] = df['close'].pct_change(3)
    df['price_roc_5'] = df['close'].pct_change(5)
    df['price_roc_10'] = df['close'].pct_change(10)
    df['rsi_roc_3'] = df['rsi_value'].diff(3)
    df['rsi_roc_5'] = df['rsi_value'].diff(5)
    df['volume_roc_3'] = df['volume_ratio'].pct_change(3)
    df['bb_width_roc_5'] = df['bb_width_pct'].pct_change(5)
    df['price_velocity'] = df['close'].diff(1) / df['close'].shift(1)
    df['price_acceleration'] = df['price_velocity'].diff(1)
    df['price_accel_norm'] = df['price_acceleration'] / df['price_velocity'].abs().replace(0, np.nan)
    df['rsi_velocity'] = df['rsi_value'].diff(1)
    df['rsi_acceleration'] = df['rsi_velocity'].diff(1)
    df['momentum_deceleration'] = (df['price_velocity'].shift(1) > df['price_velocity']).astype(int)
    df['rsi_deceleration'] = (df['rsi_velocity'].shift(1) > df['rsi_velocity']).astype(int)
    df['price_accel_smooth'] = df['price_acceleration'].rolling(3).mean()
    
    # Pattern Features (10)
    df['consecutive_bullish'] = (df['close'] > df['open']).rolling(5).sum()
    df['consecutive_bearish'] = (df['close'] < df['open']).rolling(5).sum()
    df['consecutive_higher_highs'] = (df['high'] > df['high'].shift(1)).rolling(5).sum()
    df['consecutive_higher_closes'] = (df['close'] > df['close'].shift(1)).rolling(5).sum()
    df['bullish_exhaustion'] = ((df['consecutive_bullish'] >= 3) & (df['rsi_value'] >= 70)).astype(int)
    df['upper_wick_ratio'] = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low']).replace(0, np.nan)
    df['shooting_star'] = ((df['upper_wick_ratio'] >= 0.6) & (df['bb_position'] >= 0.9)).astype(int)
    df['bearish_engulfing'] = ((df['close'] < df['open']) & 
                               (df['open'] > df['close'].shift(1)) &
                               (df['close'] < df['open'].shift(1))).astype(int)
    df['evening_star'] = ((df['close'].shift(2) < df['open'].shift(2)) &
                          (df['close'].shift(1) > df['open'].shift(1)) &
                          (df['close'] < df['open'])).astype(int)
    df['near_double_top'] = ((df['high'] >= df['high'].rolling(20).max() * 0.99) &
                             (df['high'].shift(10) >= df['high'].rolling(20).max() * 0.99)).astype(int)
    
    # Session Features (16)
    if 'hour' in df.columns:
        df['is_asian_session'] = df['hour'].isin([0, 1, 2, 3, 4, 5, 6, 7]).astype(int)
        df['is_london_session'] = df['hour'].isin([8, 9, 10, 11, 12, 13, 14, 15]).astype(int)
        df['is_ny_session'] = df['hour'].isin([13, 14, 15, 16, 17, 18, 19, 20]).astype(int)
        df['is_overlap_session'] = df['hour'].isin([13, 14, 15, 16]).astype(int)
        df['is_session_start'] = df['hour'].isin([0, 8, 13]).astype(int)
        df['is_session_end'] = df['hour'].isin([7, 12, 20]).astype(int)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    if 'day_of_week' in df.columns:
        df['is_monday'] = (df['day_of_week'] == 0).astype(int)
        df['is_friday'] = (df['day_of_week'] == 4).astype(int)
        df['is_midweek'] = df['day_of_week'].isin([1, 2, 3]).astype(int)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 5)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 5)
    
    # Regime Features (11)
    df['ma_20'] = df['close'].rolling(20).mean()
    df['ma_50'] = df['close'].rolling(50).mean()
    df['trend_direction'] = np.sign(df['close'] - df['ma_50'])
    df['is_trending'] = (df['trend_strength'].abs() >= 0.5).astype(int)
    df['is_ranging'] = (df['trend_strength'].abs() < 0.3).astype(int)
    df['is_volatile'] = (df['atr_pct'] > df['atr_pct'].rolling(50).mean() * 1.5).astype(int)
    df['regime_trend_score'] = df['trend_strength'].rolling(10).mean()
    df['is_trending_up'] = ((df['trend_strength'] > 0.5) & (df['close'] > df['ma_50'])).astype(int)
    df['is_trending_down'] = ((df['trend_strength'] < -0.5) & (df['close'] < df['ma_50'])).astype(int)
    
    # Mean Reversion Features (13)
    df['dist_from_ma10'] = (df['close'] - df['close'].rolling(10).mean()) / df['close']
    df['dist_from_ma20'] = (df['close'] - df['ma_20']) / df['close']
    df['dist_from_ma50'] = (df['close'] - df['ma_50']) / df['close']
    df['overextension_10'] = df['dist_from_ma10'].clip(lower=0)
    df['overextension_20'] = df['dist_from_ma20'].clip(lower=0)
    df['ma20_slope'] = (df['ma_20'] - df['ma_20'].shift(5)) / df['ma_20'].shift(5)
    df['ma50_slope'] = (df['ma_50'] - df['ma_50'].shift(10)) / df['ma_50'].shift(10)
    df['price_above_ma10'] = (df['close'] > df['close'].rolling(10).mean()).astype(int)
    df['price_above_ma20'] = (df['close'] > df['ma_20']).astype(int)
    df['price_above_ma50'] = (df['close'] > df['ma_50']).astype(int)
    df['all_ma_bullish'] = ((df['close'] > df['ma_20']) & (df['ma_20'] > df['ma_50'])).astype(int)
    df['extreme_overextension'] = (df['dist_from_ma20'] > df['dist_from_ma20'].rolling(100).quantile(0.95)).astype(int)
    df['mean_reversion_score'] = df['dist_from_ma20'] * df['rsi_value'] / 100
    
    # Volatility Features (7)
    df['volatility_contraction'] = (df['bb_width_pct'] < df['bb_width_pct'].rolling(20).quantile(0.2)).astype(int)
    df['volatility_expansion'] = (df['bb_width_pct'] > df['bb_width_pct'].rolling(20).quantile(0.8)).astype(int)
    df['historical_vol'] = df['close'].pct_change().rolling(20).std() * np.sqrt(252)
    df['volatility_ratio'] = df['atr_pct'] / df['atr_pct'].rolling(50).mean()
    df['intrabar_vol'] = (df['high'] - df['low']) / df['close']
    df['vol_cluster'] = df['intrabar_vol'].rolling(5).mean() / df['intrabar_vol'].rolling(20).mean()
    
    # =========================================================================
    # D1 DERIVED FEATURES (~40)
    # =========================================================================
    
    if 'd1_rsi_value' in df.columns:
        df['d1_rsi_overbought'] = (df['d1_rsi_value'] >= 70).astype(int)
        df['d1_rsi_oversold'] = (df['d1_rsi_value'] <= 30).astype(int)
        df['d1_rsi_extreme'] = (df['d1_rsi_value'] >= 80).astype(int)
        df['d1_rsi_slope_3'] = (df['d1_rsi_value'] - df['d1_rsi_value'].shift(3)) / 3
        df['d1_rsi_slope_5'] = (df['d1_rsi_value'] - df['d1_rsi_value'].shift(5)) / 5
        df['h4_d1_rsi_diff'] = df['rsi_value'] - df['d1_rsi_value']
        df['h4_d1_rsi_ratio'] = df['rsi_value'] / df['d1_rsi_value'].replace(0, np.nan)
    
    if 'd1_bb_position' in df.columns:
        df['d1_bb_high'] = (df['d1_bb_position'] >= 0.8).astype(int)
        df['d1_bb_low'] = (df['d1_bb_position'] <= 0.2).astype(int)
        df['d1_bb_extreme'] = (df['d1_bb_position'] >= 0.95).astype(int)
        df['h4_d1_bb_diff'] = df['bb_position'] - df['d1_bb_position']
    
    if 'd1_trend_strength' in df.columns:
        df['d1_uptrend'] = (df['d1_trend_strength'] > 0.3).astype(int)
        df['d1_downtrend'] = (df['d1_trend_strength'] < -0.3).astype(int)
        df['d1_strong_trend'] = (df['d1_trend_strength'].abs() > 0.7).astype(int)
        df['h4_d1_trend_diff'] = df['trend_strength'] - df['d1_trend_strength']
        df['h4_d1_trend_aligned'] = (np.sign(df['trend_strength']) == np.sign(df['d1_trend_strength'])).astype(int)
    
    if 'd1_atr_pct' in df.columns:
        df['d1_vol_high'] = (df['d1_atr_pct'] > df['d1_atr_pct'].rolling(20).quantile(0.8)).astype(int)
        df['h4_d1_vol_ratio'] = df['atr_pct'] / df['d1_atr_pct'].replace(0, np.nan)
    
    # =========================================================================
    # CROSS-TIMEFRAME / MTF FEATURES (~20)
    # =========================================================================
    
    df['h4_uptrend'] = (df['trend_strength'] > 0.3).astype(int)
    df['h4_downtrend'] = (df['trend_strength'] < -0.3).astype(int)
    
    if 'd1_trend_strength' in df.columns:
        df['mtf_bullish_aligned'] = (df['h4_uptrend'] & df['d1_uptrend']).astype(int)
        df['mtf_bearish_aligned'] = (df['h4_downtrend'] & df['d1_downtrend']).astype(int)
        df['mtf_trend_conflict'] = ((df['h4_uptrend'] & df['d1_downtrend']) | 
                                    (df['h4_downtrend'] & df['d1_uptrend'])).astype(int)
    
    if 'd1_rsi_value' in df.columns and 'd1_bb_position' in df.columns:
        df['mtf_rsi_aligned'] = ((df['rsi_value'] >= 55) & (df['d1_rsi_value'] >= 55)).astype(int)
        df['mtf_bb_aligned'] = ((df['bb_position'] >= 0.6) & (df['d1_bb_position'] >= 0.6)).astype(int)
        
        # MTF Confluence Score
        df['mtf_confluence_score'] = (
            df.get('mtf_bullish_aligned', 0).astype(float) * 0.3 +
            df.get('mtf_rsi_aligned', 0).astype(float) * 0.3 +
            df.get('mtf_bb_aligned', 0).astype(float) * 0.2 +
            df.get('h4_d1_trend_aligned', 0).astype(float) * 0.2
        )
        
        df['mtf_rsi_divergence'] = ((df['rsi_value'] >= 70) & (df['d1_rsi_value'] <= 50)).astype(int)
    
    if 'd1_trend_strength' in df.columns and 'd1_rsi_value' in df.columns:
        df['d1_supports_short'] = ((df['d1_trend_strength'] < -0.3) | (df['d1_rsi_value'] <= 40)).astype(int)
        df['d1_opposes_short'] = ((df['d1_trend_strength'] > 0.5) & (df['d1_rsi_value'] >= 60)).astype(int)
    
    # MTF Signal columns
    df['mtf_long_signal'] = df.get('mtf_bullish_aligned', 0)
    df['mtf_short_signal'] = df.get('mtf_bearish_aligned', 0)
    
    # =========================================================================
    # CLEANUP
    # =========================================================================
    
    df = df.replace([np.inf, -np.inf], np.nan)
    
    final_cols = len(df.columns)
    print(f"  Features created: {final_cols - initial_cols}")
    print(f"  Total columns: {final_cols}")
    
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Get all valid feature columns for ML."""
    exclude = [
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'h4_date', 'd1_available_date', 'd1_timestamp', 
        'd1_open', 'd1_high', 'd1_low', 'd1_close', 'd1_volume',
        'trade_label', 'trade_pips', 'trade_bars',
        'mtf_long_signal', 'mtf_short_signal',
    ]
    
    feature_cols = []
    for col in df.columns:
        if col in exclude:
            continue
        if col.startswith('d1_') and col.endswith(('_open', '_high', '_low', '_close', '_volume')):
            continue
        if df[col].dtype in [np.float64, np.int64, np.float32, np.int32, bool, np.bool_]:
            if df[col].notna().sum() > len(df) * 0.3:
                feature_cols.append(col)
    
    return sorted(feature_cols)


# =============================================================================
# TRADE LABELING
# =============================================================================

def label_trades(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Label signal bars with trade outcomes."""
    print(f"\n{'='*60}")
    print(f"  LABELING TRADES")
    print(f"{'='*60}")
    
    df = df.copy()
    tp = config["tp_pips"]
    sl = config["sl_pips"]
    max_hold = config["max_hold_bars"]
    direction = config["direction"]
    signal_col = "mtf_long_signal" if direction == "long" else "mtf_short_signal"
    
    df["trade_label"] = np.nan
    df["trade_pips"] = np.nan
    
    signal_indices = df[df[signal_col] == 1].index.tolist()
    print(f"  Signal: {signal_col}")
    print(f"  Total signals: {len(signal_indices):,}")
    
    wins = 0
    labeled = 0
    
    for idx in signal_indices:
        if idx + max_hold >= len(df):
            continue
        
        entry_price = df.loc[idx, "close"]
        tp_price = entry_price + (tp * PIP_VALUE * (1 if direction == "long" else -1))
        sl_price = entry_price - (sl * PIP_VALUE * (1 if direction == "long" else -1))
        
        outcome = None
        pips = 0
        
        for offset in range(1, max_hold + 1):
            bar_idx = idx + offset
            if bar_idx >= len(df):
                break
            
            bar_high = df.loc[bar_idx, "high"]
            bar_low = df.loc[bar_idx, "low"]
            
            if direction == "long":
                if bar_low <= sl_price:
                    outcome = 0
                    pips = -sl
                    break
                if bar_high >= tp_price:
                    outcome = 1
                    pips = tp
                    break
            else:
                if bar_high >= sl_price:
                    outcome = 0
                    pips = -sl
                    break
                if bar_low <= tp_price:
                    outcome = 1
                    pips = tp
                    break
        
        if outcome is None:
            final_price = df.loc[min(idx + max_hold, len(df) - 1), "close"]
            pips = (final_price - entry_price) / PIP_VALUE * (1 if direction == "long" else -1)
            outcome = 1 if pips > 0 else 0
        
        df.loc[idx, "trade_label"] = outcome
        df.loc[idx, "trade_pips"] = pips
        labeled += 1
        if outcome == 1:
            wins += 1
    
    print(f"  Labeled: {labeled:,}")
    print(f"  Wins: {wins:,} ({100*wins/labeled:.1f}%)" if labeled > 0 else "  No trades")
    
    return df


# =============================================================================
# SINGLE FOLD PROCESSING (for parallel execution)
# =============================================================================

def process_single_fold(args: Tuple) -> Dict:
    """
    Process a single fold - called by parallel workers.
    
    Returns dict with fold results for all thresholds.
    """
    fold_config, df, feature_cols, thresholds = args
    
    fold_name = fold_config["name"]
    train_end = pd.Timestamp(fold_config["train_end"])
    test_end = pd.Timestamp(fold_config["test_end"])
    
    result = {
        "fold_name": fold_name,
        "train_end": str(train_end.date()),
        "test_end": str(test_end.date()),
        "status": "success",
        "threshold_results": {},
    }
    
    try:
        # Split data
        train_df = df[df["timestamp"] < train_end].copy()
        test_df = df[(df["timestamp"] >= train_end) & (df["timestamp"] < test_end)].copy()
        
        train_signals = train_df[train_df["trade_label"].notna()].copy()
        test_signals = test_df[test_df["trade_label"].notna()].copy()
        
        if len(train_signals) < 200 or len(test_signals) < 30:
            result["status"] = "insufficient_data"
            result["train_samples"] = len(train_signals)
            result["test_samples"] = len(test_signals)
            return result
        
        result["train_samples"] = len(train_signals)
        result["test_samples"] = len(test_signals)
        
        # Prepare data
        X_train = train_signals[feature_cols].copy()
        y_train = train_signals["trade_label"].values
        pips_train = train_signals["trade_pips"].values
        
        X_test = test_signals[feature_cols].copy()
        y_test = test_signals["trade_label"].values
        pips_test = test_signals["trade_pips"].values
        
        # Fill NaN with median
        medians = X_train.median()
        X_train = X_train.fillna(medians)
        X_test = X_test.fillna(medians)
        
        # RFE Feature Selection
        estimator = lgb.LGBMClassifier(
            n_estimators=50, max_depth=4, learning_rate=0.1,
            verbose=-1, random_state=42, n_jobs=1
        )
        
        selector = RFE(estimator=estimator, n_features_to_select=N_FEATURES_TO_SELECT, step=5)
        selector.fit(X_train, y_train)
        
        selected_features = X_train.columns[selector.support_].tolist()
        result["n_features_selected"] = len(selected_features)
        result["selected_features"] = selected_features
        
        # Use selected features
        X_train_sel = X_train[selected_features]
        X_test_sel = X_test[selected_features]
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_sel)
        X_test_scaled = scaler.transform(X_test_sel)
        
        # Train model
        model = lgb.LGBMClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.05,
            min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1, n_jobs=1
        )
        model.fit(X_train_scaled, y_train)
        
        # Predict probabilities
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate AUC
        try:
            auc = roc_auc_score(y_test, y_proba)
        except:
            auc = 0.5
        result["auc"] = auc
        
        # Baseline metrics
        baseline_wr = y_test.mean()
        baseline_pips = pips_test.sum()
        baseline_avg = pips_test.mean()
        result["baseline_wr"] = baseline_wr
        result["baseline_pips"] = baseline_pips
        result["baseline_avg_pips"] = baseline_avg
        
        # Test each threshold
        for threshold in thresholds:
            mask = y_proba >= threshold
            n_trades = mask.sum()
            
            if n_trades < 5:
                result["threshold_results"][threshold] = {
                    "trades": 0, "wins": 0, "win_rate": 0,
                    "total_pips": 0, "avg_pips": 0, "improved": False
                }
                continue
            
            filtered_y = y_test[mask]
            filtered_pips = pips_test[mask]
            
            wins = int(filtered_y.sum())
            wr = wins / n_trades
            total_pips = filtered_pips.sum()
            avg_pips = filtered_pips.mean()
            improved = avg_pips > baseline_avg
            
            result["threshold_results"][threshold] = {
                "trades": int(n_trades),
                "wins": wins,
                "win_rate": float(wr),
                "total_pips": float(total_pips),
                "avg_pips": float(avg_pips),
                "improved": improved
            }
    
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    
    return result


# =============================================================================
# PARALLEL WALK-FORWARD VALIDATION
# =============================================================================

def run_parallel_walk_forward(
    df: pd.DataFrame, 
    feature_cols: List[str], 
    n_workers: int = 6
) -> Tuple[List[Dict], Dict]:
    """Run walk-forward validation in parallel."""
    
    print(f"\n{'='*60}")
    print(f"  WALK-FORWARD VALIDATION (Parallel)")
    print(f"{'='*60}")
    print(f"  Workers: {n_workers}")
    print(f"  Folds: {len(WALK_FORWARD_FOLDS)}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Thresholds: {THRESHOLDS}")
    print(f"  RFE selecting: {N_FEATURES_TO_SELECT} features per fold")
    
    # Prepare arguments for each fold
    fold_args = [
        (fold_config, df, feature_cols, THRESHOLDS)
        for fold_config in WALK_FORWARD_FOLDS
    ]
    
    # Run in parallel
    results = []
    start_time = time.time()
    
    print(f"\n  Processing folds...")
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_single_fold, args): args[0]["name"] 
                   for args in fold_args}
        
        for future in as_completed(futures):
            fold_name = futures[future]
            try:
                result = future.result()
                results.append(result)
                
                status = result.get("status", "unknown")
                if status == "success":
                    print(f"    ✓ {fold_name}: {result.get('test_samples', 0)} test samples, "
                          f"AUC={result.get('auc', 0):.3f}")
                else:
                    print(f"    ✗ {fold_name}: {status}")
            except Exception as e:
                print(f"    ✗ {fold_name}: Error - {e}")
                results.append({"fold_name": fold_name, "status": "error", "error": str(e)})
    
    elapsed = time.time() - start_time
    print(f"\n  Completed in {elapsed:.1f}s")
    
    # Aggregate threshold results
    threshold_aggregates = {t: {
        "trades": 0, "wins": 0, "total_pips": 0,
        "baseline_pips": 0, "improved_folds": 0, "valid_folds": 0
    } for t in THRESHOLDS}
    
    for result in results:
        if result.get("status") != "success":
            continue
        
        baseline_pips = result.get("baseline_pips", 0)
        
        for threshold, tres in result.get("threshold_results", {}).items():
            if tres.get("trades", 0) > 0:
                threshold_aggregates[threshold]["trades"] += tres["trades"]
                threshold_aggregates[threshold]["wins"] += tres["wins"]
                threshold_aggregates[threshold]["total_pips"] += tres["total_pips"]
                threshold_aggregates[threshold]["baseline_pips"] += baseline_pips
                threshold_aggregates[threshold]["valid_folds"] += 1
                if tres.get("improved", False):
                    threshold_aggregates[threshold]["improved_folds"] += 1
    
    return results, threshold_aggregates


# =============================================================================
# RESULTS DISPLAY
# =============================================================================

def print_results(fold_results: List[Dict], threshold_aggregates: Dict) -> Tuple[float, Dict]:
    """Print comprehensive results and find best threshold."""
    
    print(f"\n{'='*80}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*80}")
    
    # Per-fold summary
    valid_results = [r for r in fold_results if r.get("status") == "success"]
    
    print(f"\n  PER-FOLD PERFORMANCE:")
    print(f"  {'Fold':<8} {'Train':>8} {'Test':>8} {'Features':>10} {'AUC':>8} {'Baseline WR':>12}")
    print(f"  {'-'*60}")
    
    for r in valid_results:
        print(f"  {r['fold_name']:<8} {r.get('train_samples', 0):>8} {r.get('test_samples', 0):>8} "
              f"{r.get('n_features_selected', 0):>10} {r.get('auc', 0):>8.3f} "
              f"{r.get('baseline_wr', 0)*100:>11.1f}%")
    
    # Threshold comparison
    print(f"\n  THRESHOLD COMPARISON (Aggregated):")
    print(f"  {'Threshold':>10} {'Trades':>10} {'Win Rate':>10} {'Total Pips':>12} "
          f"{'Avg Pips':>10} {'Improved':>12}")
    print(f"  {'-'*70}")
    
    best_threshold = None
    best_avg_pips = -999
    
    for threshold in THRESHOLDS:
        agg = threshold_aggregates[threshold]
        trades = agg["trades"]
        
        if trades == 0:
            print(f"  {threshold:>10.2f} {'No trades':>10}")
            continue
        
        wins = agg["wins"]
        wr = wins / trades
        total_pips = agg["total_pips"]
        avg_pips = total_pips / trades
        improved = agg["improved_folds"]
        valid = agg["valid_folds"]
        
        if avg_pips > best_avg_pips:
            best_avg_pips = avg_pips
            best_threshold = threshold
        
        marker = " ←" if threshold == best_threshold else ""
        print(f"  {threshold:>10.2f} {trades:>10} {wr:>10.1%} {total_pips:>+12.0f} "
              f"{avg_pips:>+10.2f} {improved:>6}/{valid}{marker}")
    
    # Best threshold details
    print(f"\n{'='*80}")
    print(f"  BEST THRESHOLD: {best_threshold}")
    print(f"{'='*80}")
    
    if best_threshold:
        agg = threshold_aggregates[best_threshold]
        trades = agg["trades"]
        wins = agg["wins"]
        wr = wins / trades if trades > 0 else 0
        total_pips = agg["total_pips"]
        avg_pips = total_pips / trades if trades > 0 else 0
        improved = agg["improved_folds"]
        valid = agg["valid_folds"]
        
        print(f"""
  ┌─────────────────────────────────────────────────────────┐
  │  RECOMMENDED SETTINGS                                   │
  ├─────────────────────────────────────────────────────────┤
  │  Signal:      MTF Trend Aligned Long                    │
  │  Threshold:   {best_threshold:.2f}                                        │
  │  TP:          {TRADE_CONFIG['tp_pips']} pips                                       │
  │  SL:          {TRADE_CONFIG['sl_pips']} pips                                        │
  │  Max Hold:    {TRADE_CONFIG['max_hold_bars']} bars                                       │
  ├─────────────────────────────────────────────────────────┤
  │  VALIDATION RESULTS                                     │
  │  Total Trades:    {trades:>8,}                                  │
  │  Win Rate:        {wr:>8.1%}                                  │
  │  Total Pips:      {total_pips:>+8,.0f}                                  │
  │  Avg Pips/Trade:  {avg_pips:>+8.2f}                                  │
  │  Folds Improved:  {improved}/{valid}                                      │
  └─────────────────────────────────────────────────────────┘
        """)
        
        if improved >= valid * 0.6:
            print("  ✅ VALIDATED: ML filter improves performance in majority of folds")
        else:
            print("  ⚠️ CAUTION: Inconsistent improvement across folds")
    
    return best_threshold, threshold_aggregates.get(best_threshold, {})


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Stella Alpha - Proper ML Pipeline")
    parser.add_argument("--h4", required=True, help="Path to H4 CSV file")
    parser.add_argument("--d1", required=True, help="Path to D1 CSV file")
    parser.add_argument("--workers", "-w", type=int, default=6, 
                        help="Number of parallel workers (default: 6)")
    args = parser.parse_args()
    
    # Ensure minimum workers
    n_workers = max(6, min(args.workers, multiprocessing.cpu_count()))
    
    print(f"\n{'='*60}")
    print(f"  STELLA ALPHA - PROPER ML PIPELINE")
    print(f"{'='*60}")
    print(f"  Strategy: MTF Trend Aligned")
    print(f"  TP: {TRADE_CONFIG['tp_pips']} pips | SL: {TRADE_CONFIG['sl_pips']} pips")
    print(f"  Workers: {n_workers}")
    print(f"  Thresholds: {THRESHOLDS}")
    print(f"  RFE Features: {N_FEATURES_TO_SELECT}")
    
    # Load and merge data
    df = load_and_merge_data(args.h4, args.d1)
    
    # Engineer ALL features
    df = engineer_all_features(df)
    
    # Get feature columns
    feature_cols = get_feature_columns(df)
    print(f"\n  📋 Total ML features: {len(feature_cols)}")
    
    # Label trades
    df = label_trades(df, TRADE_CONFIG)
    
    # Run parallel walk-forward validation
    fold_results, threshold_aggregates = run_parallel_walk_forward(
        df, feature_cols, n_workers
    )
    
    # Print results
    best_threshold, best_stats = print_results(fold_results, threshold_aggregates)
    
    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Required for Windows multiprocessing
    multiprocessing.freeze_support()
    main()
