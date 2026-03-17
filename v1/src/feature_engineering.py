"""
Feature Engineering - STELLA ALPHA (Phase 2)

VERSION: Stella Alpha - ENHANCED WITH D1 + CROSS-TIMEFRAME FEATURES

PIPELINE:
=========
1. DataExporterEA_SHORT.mq5 calculates BASE features from raw OHLCV (H4 & D1)
2. data_merger.py safely merges D1 into H4 (no leakage)
3. This module calculates:
   - H4 DERIVED features (carried from V4, ~192)
   - D1 DERIVED features (NEW, ~40)
   - CROSS-TIMEFRAME features (NEW, ~20)

FEATURE SUMMARY:
================
H4 Base (from EA):           22
H4 Derived (from V4):       ~192
D1 Base (from EA):           22  (prefixed with d1_)
D1 Derived (NEW):            ~40  (prefixed with d1_)
Cross-Timeframe (NEW):       ~20  (prefixed with mtf_/h4_vs_d1_/d1_supports_*)
────────────────────────────────
TOTAL:                      ~296

After Statistical Pre-Filter:  ~80-120
After RFE Selection:            15-25

BASE FEATURES (from EA CSV - DO NOT RECALCULATE):
─────────────────────────────────────────────────
H4: open, high, low, close, volume, lower_band, middle_band, upper_band,
    bb_touch_strength, bb_position, bb_width_pct, rsi_value, rsi_divergence,
    volume_ratio, candle_rejection, candle_body_pct, atr_pct, trend_strength,
    prev_candle_body_pct, prev_volume_ratio, gap_from_prev_close,
    price_momentum, prev_was_rally, previous_touches, time_since_last_touch,
    resistance_distance_pct, session

D1: same columns prefixed with d1_ (after merge)
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
import warnings
import logging
import os
from scipy import stats as scipy_stats

os.environ['LIGHTGBM_VERBOSITY'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

from sklearn.feature_selection import RFE, RFECV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.exceptions import ConvergenceWarning


# =============================================================================
# PART 1: FEATURE ENGINEERING (STELLA ALPHA - H4 + D1 + MTF)
# =============================================================================

class FeatureEngineering:
    """
    Comprehensive feature engineering for SHORT strategy.
    Stella Alpha version: extends V4 with D1 + cross-timeframe features.

    Usage:
        fe = FeatureEngineering(verbose=True)
        df_features = fe.calculate_features(df_merged)
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.logger = logging.getLogger(__name__)

    def calculate_features(
        self,
        df: pd.DataFrame,
        drop_na: bool = True,
        compute_h4_derived: bool = True,
        compute_d1_derived: bool = True,
        compute_cross_tf: bool = True,
    ) -> pd.DataFrame:
        """
        Calculate all derived features from the merged H4+D1 DataFrame.

        Args:
            df: Merged DataFrame from data_merger.py (H4 rows with d1_ columns)
            drop_na: Drop rows with NaN after feature calculation
            compute_h4_derived: Whether to compute H4 derived features (V4)
            compute_d1_derived: Whether to compute D1 derived features (NEW)
            compute_cross_tf: Whether to compute cross-timeframe features (NEW)

        Returns:
            DataFrame with all features added
        """
        if self.verbose:
            print(f"\n[FeatureEngineering] Input shape: {df.shape}")

        result = df.copy()

        # ── H4 DERIVED FEATURES (V4 carry-forward) ───────────────────────────
        if compute_h4_derived:
            if self.verbose:
                print("  → Computing H4 derived features...")
            result = self._add_price_change_features(result)
            result = self._add_lag_features(result)
            result = self._add_slope_features(result)
            result = self._add_zscore_features(result)
            result = self._add_rolling_stat_features(result)
            result = self._add_percentile_features(result)
            result = self._add_binary_features(result)
            result = self._add_momentum_features(result)
            result = self._add_pattern_features(result)
            result = self._add_session_features(result)
            result = self._add_regime_features(result)
            result = self._add_volatility_features(result)
            result = self._add_mean_reversion_features(result)

        # ── D1 DERIVED FEATURES (NEW) ─────────────────────────────────────────
        if compute_d1_derived and self._has_d1_columns(result):
            if self.verbose:
                print("  → Computing D1 derived features...")
            result = self._add_d1_binary_features(result)
            result = self._add_d1_momentum_features(result)
            result = self._add_d1_mean_reversion_features(result)
            result = self._add_d1_volatility_features(result)
            result = self._add_d1_regime_features(result)
            result = self._add_d1_pattern_features(result)
            result = self._add_d1_zscore_features(result)

        # ── CROSS-TIMEFRAME FEATURES (NEW) ───────────────────────────────────
        if compute_cross_tf and self._has_d1_columns(result):
            if self.verbose:
                print("  → Computing cross-timeframe features...")
            result = self._add_mtf_alignment_features(result)
            result = self._add_mtf_divergence_features(result)
            result = self._add_mtf_relative_position_features(result)
            result = self._add_mtf_volatility_comparison(result)
            result = self._add_d1_context_flags(result)

        # ── REPLACE INF WITH NAN ─────────────────────────────────────────────
        result = result.replace([np.inf, -np.inf], np.nan)

        if drop_na:
            before = len(result)
            result = result.dropna()
            dropped = before - len(result)
            if self.verbose and dropped > 0:
                print(f"  → Dropped {dropped} rows with NaN ({dropped/before*100:.1f}%)")

        if self.verbose:
            print(f"  → Output shape: {result.shape}")
            h4_derived = len([c for c in result.columns if not c.startswith('d1_') and not c.startswith('mtf_') and not c.startswith('h4_vs_')])
            d1_cols = len([c for c in result.columns if c.startswith('d1_')])
            mtf_cols = len([c for c in result.columns if c.startswith('mtf_') or c.startswith('h4_vs_') or c in ('d1_supports_short', 'd1_opposes_short', 'd1_neutral', 'mtf_strong_short_setup')])
            print(f"     H4 columns: {h4_derived} | D1 columns: {d1_cols} | MTF columns: {mtf_cols}")

        return result

    # =========================================================================
    # INTERNAL HELPERS
    # =========================================================================

    def _has_d1_columns(self, df: pd.DataFrame) -> bool:
        """Check if D1 base columns are present (from merge)."""
        return 'd1_rsi_value' in df.columns or 'd1_close' in df.columns

    def _safe_divide(self, num: pd.Series, denom: pd.Series, fill: float = 0.0) -> pd.Series:
        """Safe division avoiding divide-by-zero."""
        return num.div(denom.replace(0, np.nan)).fillna(fill)

    # =========================================================================
    # H4 DERIVED FEATURES (V4 CARRY-FORWARD)
    # =========================================================================

    def _add_price_change_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Price change features."""
        df['price_change_1'] = df['close'].pct_change(1)
        df['price_change_3'] = df['close'].pct_change(3)
        df['price_change_5'] = df['close'].pct_change(5)
        df['price_change_10'] = df['close'].pct_change(10)
        df['high_change_1'] = df['high'].pct_change(1)
        df['high_change_3'] = df['high'].pct_change(3)
        df['range_pct'] = self._safe_divide(df['high'] - df['low'], df['close'])
        return df

    def _add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Lag features."""
        for i in range(1, 6):
            df[f'rsi_lag{i}'] = df['rsi_value'].shift(i)
        for i in range(1, 4):
            df[f'bb_position_lag{i}'] = df['bb_position'].shift(i)
        for i in range(1, 4):
            df[f'price_change_lag{i}'] = df['price_change_1'].shift(i)
        for i in range(1, 4):
            df[f'volume_ratio_lag{i}'] = df['volume_ratio'].shift(i)
        for i in range(1, 3):
            df[f'bb_width_lag{i}'] = df['bb_width_pct'].shift(i)
        for i in range(1, 3):
            df[f'trend_strength_lag{i}'] = df['trend_strength'].shift(i)
        return df

    def _add_slope_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Slope (rate of change) features."""
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
        return df

    def _add_zscore_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Z-score features (20-period rolling)."""
        window = 20
        for col, name in [
            ('close', 'price_zscore'),
            ('rsi_value', 'rsi_zscore'),
            ('volume_ratio', 'volume_zscore'),
            ('bb_position', 'bb_position_zscore'),
            ('atr_pct', 'atr_zscore'),
            ('high', 'high_zscore'),
        ]:
            if col in df.columns:
                roll_mean = df[col].rolling(window).mean()
                roll_std = df[col].rolling(window).std()
                df[name] = self._safe_divide(df[col] - roll_mean, roll_std)
        return df

    def _add_rolling_stat_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rolling statistical features."""
        df['price_rolling_std'] = df['close'].rolling(20).std()
        df['rsi_rolling_std'] = df['rsi_value'].rolling(10).std()
        df['rsi_rolling_max'] = df['rsi_value'].rolling(10).max()
        df['rsi_rolling_min'] = df['rsi_value'].rolling(10).min()
        df['rsi_range'] = df['rsi_rolling_max'] - df['rsi_rolling_min']
        df['bb_position_rolling_std'] = df['bb_position'].rolling(10).std()
        df['bb_position_rolling_max'] = df['bb_position'].rolling(10).max()
        df['volume_rolling_std'] = df['volume_ratio'].rolling(10).std()
        df['price_skew_20'] = df['close'].rolling(20).skew()
        df['price_kurtosis_20'] = df['close'].rolling(20).kurt()
        return df

    def _add_percentile_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Percentile rank features."""
        window = 50
        for col, name in [
            ('close', 'price_percentile'),
            ('rsi_value', 'rsi_percentile'),
            ('volume_ratio', 'volume_percentile'),
            ('atr_pct', 'atr_percentile'),
        ]:
            if col in df.columns:
                df[name] = df[col].rolling(window).rank(pct=True)
        return df

    def _add_binary_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Binary flag features."""
        df['touched_upper_bb'] = (df['high'] >= df['upper_band']).astype(int)
        df['rsi_overbought'] = (df['rsi_value'] >= 70).astype(int)
        df['rsi_extreme_overbought'] = (df['rsi_value'] >= 75).astype(int)
        df['rsi_very_extreme'] = (df['rsi_value'] >= 80).astype(int)
        df['bearish_candle'] = (df['close'] < df['open']).astype(int)
        df['strong_bearish'] = ((df['close'] < df['open']) & (df['candle_body_pct'] > 0.6)).astype(int)
        df['bb_very_high'] = (df['bb_position'] > 0.95).astype(int)
        df['bb_above_upper'] = (df['close'] > df['upper_band']).astype(int)
        df['high_volume'] = (df['volume_ratio'] > 1.5).astype(int)
        df['extreme_volume'] = (df['volume_ratio'] > 2.0).astype(int)
        df['strong_uptrend'] = (df['trend_strength'] > 0.7).astype(int)
        return df

    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Momentum features."""
        df['rsi_momentum'] = df['rsi_value'] - df['rsi_value'].shift(3)
        df['price_momentum_3'] = df['close'].pct_change(3)
        df['price_acceleration'] = df['price_change_1'] - df['price_change_1'].shift(1)
        df['rsi_acceleration'] = df['rsi_slope_3'] - df['rsi_slope_3'].shift(3)
        df['volume_momentum'] = df['volume_ratio'] - df['volume_ratio'].shift(3)
        df['bb_position_momentum'] = df['bb_position'] - df['bb_position'].shift(5)
        df['trend_exhaustion'] = (
            (df['rsi_value'] > 70) &
            (df['rsi_slope_3'] < 0) &
            (df['price_slope_3'] > 0)
        ).astype(int)
        return df

    def _add_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Candlestick pattern features."""
        df['consecutive_bullish'] = 0
        df['consecutive_bearish'] = 0
        bullish = (df['close'] > df['open']).astype(int)
        bearish = (df['close'] < df['open']).astype(int)

        consec_bull = []
        consec_bear = []
        bull_count = bear_count = 0
        for b, be in zip(bullish, bearish):
            if b:
                bull_count += 1
                bear_count = 0
            elif be:
                bear_count += 1
                bull_count = 0
            else:
                bull_count = bear_count = 0
            consec_bull.append(bull_count)
            consec_bear.append(bear_count)

        df['consecutive_bullish'] = consec_bull
        df['consecutive_bearish'] = consec_bear

        df['doji'] = (df['candle_body_pct'] < 0.1).astype(int)
        df['shooting_star'] = (
            (df['candle_rejection'] > 0.6) &
            (df['candle_body_pct'] < 0.3) &
            (df['close'] < df['open'])
        ).astype(int)
        df['upper_wick_ratio'] = self._safe_divide(
            df['high'] - df[['open', 'close']].max(axis=1),
            df['high'] - df['low']
        )
        df['lower_wick_ratio'] = self._safe_divide(
            df[['open', 'close']].min(axis=1) - df['low'],
            df['high'] - df['low']
        )
        df['exhaustion_score'] = (
            df['rsi_overbought'] * 0.3 +
            df['bb_very_high'] * 0.3 +
            df['shooting_star'] * 0.2 +
            df['high_volume'] * 0.2
        )
        return df

    def _add_session_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Session/time features."""
        if 'timestamp' not in df.columns:
            return df
        try:
            ts = pd.to_datetime(df['timestamp'])
            hour = ts.dt.hour
            dow = ts.dt.dayofweek

            df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
            df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
            df['dow_sin'] = np.sin(2 * np.pi * dow / 7)
            df['dow_cos'] = np.cos(2 * np.pi * dow / 7)
            df['is_london'] = ((hour >= 8) & (hour < 16)).astype(int)
            df['is_ny'] = ((hour >= 13) & (hour < 22)).astype(int)
            df['is_asian'] = ((hour >= 0) & (hour < 8)).astype(int)
            df['is_london_ny_overlap'] = ((hour >= 13) & (hour < 16)).astype(int)
            df['is_high_activity'] = (df['is_london'] | df['is_ny']).astype(int)
        except Exception:
            pass
        return df

    def _add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Market regime features."""
        window = 14
        tr = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        ], axis=1).max(axis=1)
        atr = tr.rolling(window).mean()

        plus_dm = df['high'].diff().clip(lower=0)
        minus_dm = (-df['low'].diff()).clip(lower=0)
        plus_dm[plus_dm < minus_dm.abs()] = 0
        minus_dm[minus_dm < plus_dm.abs()] = 0

        plus_di = 100 * plus_dm.rolling(window).mean() / atr
        minus_di = 100 * minus_dm.rolling(window).mean() / atr
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
        df['adx'] = dx.rolling(window).mean()
        df['is_trending'] = (df['adx'] > 25).astype(int)
        df['is_ranging'] = (df['adx'] < 20).astype(int)
        df['trend_direction_h4'] = np.sign(df['close'] - df['close'].rolling(20).mean()).fillna(0)

        df['volatility_regime'] = pd.cut(
            df['atr_percentile'] if 'atr_percentile' in df.columns else df['atr_pct'].rank(pct=True),
            bins=[0, 0.33, 0.66, 1.0],
            labels=[0, 1, 2]
        ).astype(float)
        return df

    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Advanced volatility features."""
        df['atr_change'] = df['atr_pct'].pct_change(3)
        df['bb_squeeze'] = (df['bb_width_pct'] < df['bb_width_pct'].rolling(20).quantile(0.2)).astype(int)
        df['bb_expansion'] = (df['bb_width_pct'] > df['bb_width_pct'].rolling(20).quantile(0.8)).astype(int)
        df['volatility_spike'] = (df['atr_pct'] > df['atr_pct'].rolling(20).mean() * 1.5).astype(int)
        return df

    def _add_mean_reversion_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Mean reversion features."""
        df['dist_from_ma20'] = self._safe_divide(
            df['close'] - df['close'].rolling(20).mean(),
            df['close'].rolling(20).mean()
        )
        df['dist_from_ma50'] = self._safe_divide(
            df['close'] - df['close'].rolling(50).mean(),
            df['close'].rolling(50).mean()
        )
        df['overextension'] = self._safe_divide(
            df['close'] - df['middle_band'],
            df['middle_band']
        )
        df['reversion_probability'] = (1 - df['bb_position']).clip(0, 1)
        return df

    # =========================================================================
    # D1 DERIVED FEATURES (NEW - Stella Alpha)
    # =========================================================================

    def _add_d1_binary_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """D1 binary flag features."""
        if 'd1_high' not in df.columns or 'd1_upper_band' not in df.columns:
            return df
        df['d1_touched_upper_bb'] = (df['d1_high'] >= df['d1_upper_band']).astype(int)
        df['d1_rsi_overbought'] = (df['d1_rsi_value'] >= 70).astype(int)
        df['d1_rsi_extreme'] = (df['d1_rsi_value'] >= 80).astype(int)
        df['d1_bearish_candle'] = (df['d1_close'] < df['d1_open']).astype(int)
        df['d1_strong_bearish'] = (
            (df['d1_close'] < df['d1_open']) &
            (df['d1_candle_body_pct'] > 0.6)
        ).astype(int)
        df['d1_bb_very_high'] = (df['d1_bb_position'] > 0.95).astype(int)
        return df

    def _add_d1_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """D1 momentum features (slopes over multiple D1 bars)."""
        if 'd1_rsi_value' not in df.columns:
            return df
        df['d1_rsi_slope_3'] = (df['d1_rsi_value'] - df['d1_rsi_value'].shift(3)) / 3
        df['d1_rsi_slope_5'] = (df['d1_rsi_value'] - df['d1_rsi_value'].shift(5)) / 5
        df['d1_price_roc_3'] = df['d1_close'].pct_change(3)
        df['d1_price_roc_5'] = df['d1_close'].pct_change(5)
        df['d1_price_roc_10'] = df['d1_close'].pct_change(10)
        df['d1_bb_position_slope_3'] = (df['d1_bb_position'] - df['d1_bb_position'].shift(3)) / 3
        df['d1_volume_slope_3'] = (df['d1_volume_ratio'] - df['d1_volume_ratio'].shift(3)) / 3
        return df

    def _add_d1_mean_reversion_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """D1 mean reversion features."""
        if 'd1_close' not in df.columns or 'd1_middle_band' not in df.columns:
            return df
        df['d1_dist_from_ma10'] = self._safe_divide(
            df['d1_close'] - df['d1_close'].rolling(10).mean(),
            df['d1_close'].rolling(10).mean()
        )
        df['d1_dist_from_ma20'] = self._safe_divide(
            df['d1_close'] - df['d1_close'].rolling(20).mean(),
            df['d1_close'].rolling(20).mean()
        )
        df['d1_overextension'] = self._safe_divide(
            df['d1_close'] - df['d1_middle_band'],
            df['d1_middle_band']
        )
        return df

    def _add_d1_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """D1 volatility features."""
        if 'd1_atr_pct' not in df.columns:
            return df
        df['d1_atr_percentile'] = df['d1_atr_pct'].rolling(50).rank(pct=True)
        df['d1_bb_squeeze'] = (
            df['d1_bb_width_pct'] < df['d1_bb_width_pct'].rolling(20).quantile(0.2)
        ).astype(int)
        df['d1_volatility_regime'] = pd.cut(
            df['d1_atr_pct'].rank(pct=True),
            bins=[0, 0.33, 0.66, 1.0],
            labels=[0, 1, 2]
        ).astype(float)
        return df

    def _add_d1_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """D1 regime/trend features (ADX equivalent)."""
        if 'd1_high' not in df.columns or 'd1_low' not in df.columns:
            return df
        window = 14
        tr = pd.concat([
            df['d1_high'] - df['d1_low'],
            abs(df['d1_high'] - df['d1_close'].shift(1)),
            abs(df['d1_low'] - df['d1_close'].shift(1))
        ], axis=1).max(axis=1)
        atr = tr.rolling(window).mean()
        plus_dm = df['d1_high'].diff().clip(lower=0)
        minus_dm = (-df['d1_low'].diff()).clip(lower=0)
        plus_dm[plus_dm < minus_dm.abs()] = 0
        minus_dm[minus_dm < plus_dm.abs()] = 0
        plus_di = 100 * plus_dm.rolling(window).mean() / atr.replace(0, np.nan)
        minus_di = 100 * minus_dm.rolling(window).mean() / atr.replace(0, np.nan)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
        df['d1_adx'] = dx.rolling(window).mean()
        df['d1_is_trending'] = (df['d1_adx'] > 25).astype(int)
        df['d1_is_ranging'] = (df['d1_adx'] < 20).astype(int)

        ma20 = df['d1_close'].rolling(20).mean()
        df['d1_trend_direction'] = np.sign(df['d1_close'] - ma20).fillna(0)
        df['d1_is_trending_up'] = (df['d1_trend_direction'] > 0).astype(int)
        df['d1_is_trending_down'] = (df['d1_trend_direction'] < 0).astype(int)
        return df

    def _add_d1_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """D1 candlestick pattern features."""
        if 'd1_close' not in df.columns or 'd1_open' not in df.columns:
            return df
        bullish = (df['d1_close'] > df['d1_open']).astype(int)
        bearish = (df['d1_close'] < df['d1_open']).astype(int)

        consec_bull = []
        consec_bear = []
        bull_count = bear_count = 0
        for b, be in zip(bullish, bearish):
            if b:
                bull_count += 1
                bear_count = 0
            elif be:
                bear_count += 1
                bull_count = 0
            else:
                bull_count = bear_count = 0
            consec_bull.append(bull_count)
            consec_bear.append(bear_count)

        df['d1_consecutive_bullish'] = consec_bull
        df['d1_consecutive_bearish'] = consec_bear

        if 'd1_high' in df.columns and 'd1_low' in df.columns:
            df['d1_near_d1_high'] = (
                df['d1_close'] > df['d1_close'].rolling(20).max() * 0.98
            ).astype(int)
            df['d1_shooting_star'] = (
                (df['d1_candle_rejection'] > 0.6) &
                (df['d1_candle_body_pct'] < 0.3) &
                (df['d1_close'] < df['d1_open'])
            ).astype(int) if 'd1_candle_rejection' in df.columns and 'd1_candle_body_pct' in df.columns else 0

        # D1 bearish divergence: price up, RSI down
        df['d1_bearish_divergence'] = (
            (df['d1_close'] > df['d1_close'].shift(5)) &
            (df['d1_rsi_value'] < df['d1_rsi_value'].shift(5))
        ).astype(int) if 'd1_rsi_value' in df.columns else 0

        return df

    def _add_d1_zscore_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """D1 z-score features."""
        window = 20
        for col, name in [
            ('d1_close', 'd1_price_zscore'),
            ('d1_rsi_value', 'd1_rsi_zscore'),
        ]:
            if col in df.columns:
                roll_mean = df[col].rolling(window).mean()
                roll_std = df[col].rolling(window).std()
                df[name] = self._safe_divide(df[col] - roll_mean, roll_std)
        return df

    # =========================================================================
    # CROSS-TIMEFRAME FEATURES (NEW - Stella Alpha)
    # =========================================================================

    def _add_mtf_alignment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Alignment features: Do H4 and D1 agree?"""
        has_rsi = 'd1_rsi_value' in df.columns and 'rsi_value' in df.columns
        has_bb = 'd1_bb_position' in df.columns and 'bb_position' in df.columns
        has_trend = 'd1_trend_direction' in df.columns and 'trend_direction_h4' in df.columns

        df['mtf_rsi_aligned'] = 0
        df['mtf_bb_aligned'] = 0
        df['mtf_trend_aligned'] = 0
        df['mtf_bearish_aligned'] = 0
        df['mtf_overbought_aligned'] = 0

        if has_rsi:
            df['mtf_rsi_aligned'] = (
                (df['rsi_value'] >= 65) & (df['d1_rsi_value'] >= 65)
            ).astype(int)
            df['mtf_overbought_aligned'] = (
                (df['rsi_value'] >= 70) & (df['d1_rsi_value'] >= 70)
            ).astype(int)

        if has_bb:
            df['mtf_bb_aligned'] = (
                (df['bb_position'] > 0.8) & (df['d1_bb_position'] > 0.8)
            ).astype(int)

        if has_trend:
            df['mtf_trend_aligned'] = (
                (df['trend_direction_h4'] == df['d1_trend_direction'])
            ).astype(int)

        if 'd1_bearish_candle' in df.columns and 'bearish_candle' in df.columns:
            df['mtf_bearish_aligned'] = (
                (df['bearish_candle'] == 1) & (df['d1_bearish_candle'] == 1)
            ).astype(int)

        # Confluence score: weighted sum of alignment features
        df['mtf_confluence_score'] = (
            df['mtf_rsi_aligned'] * 0.25 +
            df['mtf_bb_aligned'] * 0.25 +
            df['mtf_trend_aligned'] * 0.25 +
            df['mtf_bearish_aligned'] * 0.25
        )

        return df

    def _add_mtf_divergence_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Divergence features: H4 and D1 disagree."""
        df['mtf_rsi_divergence'] = 0
        df['mtf_bb_divergence'] = 0
        df['mtf_trend_divergence'] = 0
        df['mtf_momentum_divergence'] = 0

        if 'd1_rsi_value' in df.columns and 'rsi_value' in df.columns:
            df['mtf_rsi_divergence'] = (
                (df['rsi_value'] > 70) & (df['d1_rsi_value'] < 50)
            ).astype(int)

        if 'd1_bb_position' in df.columns and 'bb_position' in df.columns:
            df['mtf_bb_divergence'] = (
                (df['bb_position'] > 0.8) & (df['d1_bb_position'] < 0.5)
            ).astype(int)

        if 'd1_trend_direction' in df.columns and 'trend_direction_h4' in df.columns:
            df['mtf_trend_divergence'] = (
                (df['trend_direction_h4'] > 0) & (df['d1_trend_direction'] < 0)
            ).astype(int)

        if 'd1_price_roc_3' in df.columns and 'price_slope_3' in df.columns:
            df['mtf_momentum_divergence'] = (
                (df['price_slope_3'] > 0) & (df['d1_price_roc_3'] < 0)
            ).astype(int)

        return df

    def _add_mtf_relative_position_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Relative position: Where is H4 within D1 context?"""
        if all(c in df.columns for c in ['close', 'd1_high', 'd1_low']):
            d1_range = (df['d1_high'] - df['d1_low']).replace(0, np.nan)
            df['h4_position_in_d1_range'] = (df['close'] - df['d1_low']) / d1_range

        if 'd1_bb_position' in df.columns and 'bb_position' in df.columns:
            df['h4_vs_d1_bb_position'] = df['bb_position'] - df['d1_bb_position']

        if 'd1_rsi_value' in df.columns and 'rsi_value' in df.columns:
            df['h4_vs_d1_rsi'] = df['rsi_value'] - df['d1_rsi_value']

        if 'd1_trend_strength' in df.columns and 'trend_strength' in df.columns:
            df['h4_vs_d1_trend'] = df['trend_strength'] - df['d1_trend_strength']

        return df

    def _add_mtf_volatility_comparison(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volatility comparison: H4 vs D1."""
        if 'd1_atr_pct' in df.columns and 'atr_pct' in df.columns:
            df['h4_vs_d1_atr_ratio'] = self._safe_divide(df['atr_pct'], df['d1_atr_pct'])

        if all(c in df.columns for c in ['high', 'low', 'd1_high', 'd1_low']):
            h4_range = df['high'] - df['low']
            d1_range = (df['d1_high'] - df['d1_low']).replace(0, np.nan)
            df['h4_vs_d1_range_ratio'] = h4_range / d1_range

        return df

    def _add_d1_context_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """D1 context flags: Does D1 support or oppose SHORT?"""
        df['d1_supports_short'] = 0
        df['d1_opposes_short'] = 0
        df['d1_neutral'] = 0
        df['mtf_strong_short_setup'] = 0

        if 'd1_rsi_value' in df.columns and 'd1_bb_position' in df.columns:
            # D1 supports SHORT: overbought daily
            df['d1_supports_short'] = (
                (df['d1_rsi_value'] > 65) & (df['d1_bb_position'] > 0.7)
            ).astype(int)

            # D1 opposes SHORT: oversold daily or bullish trend
            d1_trend_cond = (
                df['d1_trend_direction'] > 0.5
                if 'd1_trend_direction' in df.columns
                else pd.Series(False, index=df.index)
            )
            df['d1_opposes_short'] = (
                (df['d1_rsi_value'] < 40) & d1_trend_cond
            ).astype(int)

            # D1 neutral
            df['d1_neutral'] = (
                (df['d1_supports_short'] == 0) & (df['d1_opposes_short'] == 0)
            ).astype(int)

        # Strong SHORT setup: H4 + D1 both in strong SHORT conditions
        if 'rsi_overbought' in df.columns and 'd1_rsi_overbought' in df.columns:
            df['mtf_strong_short_setup'] = (
                (df['rsi_overbought'] == 1) &
                (df['d1_rsi_overbought'] == 1) &
                (df.get('mtf_confluence_score', pd.Series(0, index=df.index)) >= 0.5)
            ).astype(int)

        return df

    # =========================================================================
    # FEATURE NAME REGISTRY
    # =========================================================================

    def get_feature_names(self) -> Dict[str, List[str]]:
        """Return dictionary of feature categories and names."""
        return {
            # H4 features (V4 carry-forward)
            'h4_binary': [
                'touched_upper_bb', 'rsi_overbought', 'rsi_extreme_overbought',
                'rsi_very_extreme', 'bearish_candle', 'strong_bearish',
                'bb_very_high', 'bb_above_upper', 'high_volume', 'extreme_volume',
                'strong_uptrend',
            ],
            'h4_price_change': [
                'price_change_1', 'price_change_3', 'price_change_5', 'price_change_10',
                'high_change_1', 'high_change_3', 'range_pct',
            ],
            'h4_lags': [
                'rsi_lag1', 'rsi_lag2', 'rsi_lag3', 'rsi_lag4', 'rsi_lag5',
                'bb_position_lag1', 'bb_position_lag2', 'bb_position_lag3',
                'price_change_lag1', 'price_change_lag2', 'price_change_lag3',
                'volume_ratio_lag1', 'volume_ratio_lag2', 'volume_ratio_lag3',
                'bb_width_lag1', 'bb_width_lag2',
                'trend_strength_lag1', 'trend_strength_lag2',
            ],
            'h4_slopes': [
                'rsi_slope_3', 'rsi_slope_5', 'rsi_slope_10',
                'price_slope_3', 'price_slope_5', 'price_slope_10',
                'bb_position_slope_3', 'bb_position_slope_5',
                'volume_slope_3', 'bb_width_slope_3', 'bb_width_slope_5',
                'trend_slope_3',
            ],
            'h4_zscores': [
                'price_zscore', 'rsi_zscore', 'volume_zscore',
                'bb_position_zscore', 'atr_zscore', 'high_zscore',
            ],
            'h4_rolling': [
                'price_rolling_std', 'rsi_rolling_std', 'rsi_rolling_max',
                'rsi_rolling_min', 'rsi_range', 'bb_position_rolling_std',
                'bb_position_rolling_max', 'volume_rolling_std',
                'price_skew_20', 'price_kurtosis_20',
            ],
            'h4_percentiles': [
                'price_percentile', 'rsi_percentile', 'volume_percentile', 'atr_percentile',
            ],
            'h4_momentum': [
                'rsi_momentum', 'price_momentum_3', 'price_acceleration',
                'rsi_acceleration', 'volume_momentum', 'bb_position_momentum',
                'trend_exhaustion',
            ],
            'h4_pattern': [
                'consecutive_bullish', 'consecutive_bearish', 'doji', 'shooting_star',
                'upper_wick_ratio', 'lower_wick_ratio', 'exhaustion_score',
            ],
            'h4_session': [
                'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
                'is_london', 'is_ny', 'is_asian', 'is_london_ny_overlap', 'is_high_activity',
            ],
            'h4_regime': [
                'adx', 'is_trending', 'is_ranging', 'trend_direction_h4', 'volatility_regime',
            ],
            'h4_volatility': [
                'atr_change', 'bb_squeeze', 'bb_expansion', 'volatility_spike',
            ],
            'h4_mean_reversion': [
                'dist_from_ma20', 'dist_from_ma50', 'overextension', 'reversion_probability',
            ],
            # D1 derived features (NEW)
            'd1_binary': [
                'd1_touched_upper_bb', 'd1_rsi_overbought', 'd1_rsi_extreme',
                'd1_bearish_candle', 'd1_strong_bearish', 'd1_bb_very_high',
            ],
            'd1_momentum': [
                'd1_rsi_slope_3', 'd1_rsi_slope_5',
                'd1_price_roc_3', 'd1_price_roc_5', 'd1_price_roc_10',
                'd1_bb_position_slope_3', 'd1_volume_slope_3',
            ],
            'd1_mean_reversion': [
                'd1_dist_from_ma10', 'd1_dist_from_ma20', 'd1_overextension',
            ],
            'd1_volatility': [
                'd1_atr_percentile', 'd1_bb_squeeze', 'd1_volatility_regime',
            ],
            'd1_regime': [
                'd1_adx', 'd1_is_trending', 'd1_is_ranging',
                'd1_trend_direction', 'd1_is_trending_up', 'd1_is_trending_down',
            ],
            'd1_pattern': [
                'd1_consecutive_bullish', 'd1_consecutive_bearish',
                'd1_near_d1_high', 'd1_shooting_star', 'd1_bearish_divergence',
            ],
            'd1_zscores': [
                'd1_price_zscore', 'd1_rsi_zscore',
            ],
            # Cross-timeframe features (NEW)
            'mtf_alignment': [
                'mtf_rsi_aligned', 'mtf_bb_aligned', 'mtf_trend_aligned',
                'mtf_bearish_aligned', 'mtf_overbought_aligned', 'mtf_confluence_score',
            ],
            'mtf_divergence': [
                'mtf_rsi_divergence', 'mtf_bb_divergence',
                'mtf_trend_divergence', 'mtf_momentum_divergence',
            ],
            'mtf_relative': [
                'h4_position_in_d1_range', 'h4_vs_d1_bb_position',
                'h4_vs_d1_rsi', 'h4_vs_d1_trend',
            ],
            'mtf_volatility': [
                'h4_vs_d1_atr_ratio', 'h4_vs_d1_range_ratio',
            ],
            'mtf_context': [
                'd1_supports_short', 'd1_opposes_short', 'd1_neutral',
                'mtf_strong_short_setup',
            ],
        }

    def get_all_feature_names(self) -> List[str]:
        """Get flat list of all feature names."""
        all_features = []
        for cat_features in self.get_feature_names().values():
            all_features.extend(cat_features)
        return all_features


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def calculate_all_features(
    df: pd.DataFrame,
    verbose: bool = False,
    drop_na: bool = True,
    compute_d1_derived: bool = True,
    compute_cross_tf: bool = True,
) -> pd.DataFrame:
    """
    Calculate all derived features from merged H4+D1 CSV output.

    Args:
        df: Merged DataFrame (from data_merger.py)
        verbose: Print progress information
        drop_na: Drop rows with NaN values
        compute_d1_derived: Compute D1 derived features
        compute_cross_tf: Compute cross-timeframe features

    Returns:
        DataFrame with all features added
    """
    fe = FeatureEngineering(verbose=verbose)
    return fe.calculate_features(
        df,
        drop_na=drop_na,
        compute_d1_derived=compute_d1_derived,
        compute_cross_tf=compute_cross_tf,
    )


# =============================================================================
# PART 2: DATA CLASSES FOR RFE
# =============================================================================

@dataclass
class FeatureRanking:
    """Feature ranking information."""
    feature_name: str
    rank: int
    selected: bool
    importance: float = 0.0


@dataclass
class RFEResult:
    """Result of RFE feature selection."""
    selected_features: List[str]
    n_features_selected: int
    n_features_original: int
    feature_rankings: List[FeatureRanking]
    optimal_n_features: Optional[int] = None
    cv_scores: Optional[List[float]] = None

    def to_dataframe(self) -> pd.DataFrame:
        data = [
            {
                'feature_name': r.feature_name,
                'rank': r.rank,
                'selected': r.selected,
                'importance': r.importance,
            }
            for r in self.feature_rankings
        ]
        return pd.DataFrame(data).sort_values('rank').reset_index(drop=True)

    def get_selected_features(self) -> List[str]:
        return self.selected_features.copy()


# =============================================================================
# PART 3: FEATURE UTILITIES
# =============================================================================

def get_feature_columns(
    df: pd.DataFrame,
    exclude_columns: Optional[List[str]] = None,
) -> List[str]:
    """Identify feature columns from DataFrame."""
    if exclude_columns is None:
        exclude_columns = []

    default_exclusions = {
        'timestamp', 'pair', 'symbol',
        'open', 'high', 'low', 'close', 'volume',
        'label', 'label_reason', 'signal', 'regime',
        'lower_band', 'middle_band', 'upper_band',
        # D1 raw price/band columns (keep d1_ derived features)
        'd1_timestamp', 'd1_open', 'd1_high', 'd1_low', 'd1_close', 'd1_volume',
        'd1_lower_band', 'd1_middle_band', 'd1_upper_band',
    }

    exclude_set = set(col.lower() for col in exclude_columns)
    exclude_set.update(default_exclusions)

    feature_columns = [
        col for col in df.columns
        if col.lower() not in exclude_set
        and pd.api.types.is_numeric_dtype(df[col])
    ]

    return feature_columns


def validate_features(
    df: pd.DataFrame,
    feature_columns: List[str],
) -> Tuple[List[str], Dict[str, Any]]:
    """Validate feature columns and return valid ones."""
    report = {
        'original_count': len(feature_columns),
        'valid_count': 0,
        'issues': {},
        'dropped': [],
    }

    valid_features = []

    for col in feature_columns:
        issues = []

        if col not in df.columns:
            issues.append('missing')
            report['issues'][col] = issues
            report['dropped'].append(col)
            continue

        series = df[col]
        nan_count = series.isna().sum()
        if nan_count > 0:
            nan_pct = nan_count / len(series) * 100
            if nan_pct > 10:
                issues.append(f'high_nan ({nan_pct:.1f}%)')
            else:
                issues.append(f'some_nan ({nan_count})')

        if np.isinf(series).any():
            inf_count = np.isinf(series).sum()
            issues.append(f'infinite ({inf_count})')

        if series.std() == 0:
            issues.append('zero_variance')

        has_critical = any(
            issue.startswith(('missing', 'zero_variance', 'high_nan'))
            for issue in issues
        )

        if has_critical:
            report['issues'][col] = issues
            report['dropped'].append(col)
        else:
            valid_features.append(col)
            if issues:
                report['issues'][col] = issues

    report['valid_count'] = len(valid_features)
    return valid_features, report


def prepare_features(
    df: pd.DataFrame,
    feature_columns: List[str],
    handle_nan: str = 'drop',
    handle_inf: str = 'clip',
) -> Tuple[pd.DataFrame, List[str]]:
    """Prepare feature DataFrame for model training."""
    result = df[feature_columns].copy()

    result = result.replace([np.inf, -np.inf], np.nan)

    if handle_nan == 'drop':
        result = result.dropna()
    elif handle_nan == 'fill_zero':
        result = result.fillna(0)
    elif handle_nan == 'fill_median':
        result = result.fillna(result.median())

    valid_cols = [c for c in result.columns if c in result.columns]
    return result, valid_cols


# =============================================================================
# PART 4: RFE FEATURE SELECTION
# =============================================================================

def rfe_select(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_columns: List[str],
    min_features: int = 10,
    max_features: int = 25,
    cv_folds: int = 3,
    use_rfecv: bool = False,
    random_state: int = 42,
) -> RFEResult:
    """
    Recursive Feature Elimination for feature selection.

    Args:
        X_train: Training features
        y_train: Training labels
        feature_columns: List of feature column names
        min_features: Minimum features to select
        max_features: Maximum features to select
        cv_folds: CV folds for RFECV
        use_rfecv: Use RFECV (auto-select n_features)
        random_state: Random seed

    Returns:
        RFEResult with selected features and rankings
    """
    warnings.filterwarnings('ignore', category=ConvergenceWarning)

    estimator = GradientBoostingClassifier(
        n_estimators=50,
        max_depth=3,
        random_state=random_state,
        subsample=0.8,
    )

    n_features = min(max_features, len(feature_columns))

    if use_rfecv:
        selector = RFECV(
            estimator=estimator,
            step=1,
            cv=StratifiedKFold(cv_folds),
            scoring='average_precision',
            min_features_to_select=min_features,
            n_jobs=-1,
        )
    else:
        selector = RFE(
            estimator=estimator,
            n_features_to_select=n_features,
            step=1,
        )

    X = X_train[feature_columns].fillna(0)
    selector.fit(X, y_train)

    rankings = []
    for i, col in enumerate(feature_columns):
        rankings.append(FeatureRanking(
            feature_name=col,
            rank=int(selector.ranking_[i]),
            selected=bool(selector.support_[i]),
            importance=0.0,
        ))

    selected = [col for col, rank in zip(feature_columns, rankings) if rank.selected]

    # Get importances from estimator
    try:
        estimator.fit(X[selected].fillna(0), y_train)
        imp = estimator.feature_importances_
        imp_map = dict(zip(selected, imp))
        for r in rankings:
            r.importance = imp_map.get(r.feature_name, 0.0)
    except Exception:
        pass

    return RFEResult(
        selected_features=selected,
        n_features_selected=len(selected),
        n_features_original=len(feature_columns),
        feature_rankings=rankings,
        optimal_n_features=selector.n_features_ if use_rfecv else n_features,
    )


def select_features_for_fold(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_columns: List[str],
    n_features: int = 20,
    random_state: int = 42,
) -> RFEResult:
    """Convenience wrapper for per-fold feature selection."""
    return rfe_select(
        X_train=X_train,
        y_train=y_train,
        feature_columns=feature_columns,
        min_features=n_features,
        max_features=n_features,
        cv_folds=3,
        use_rfecv=False,
        random_state=random_state,
    )


def get_consensus_features(
    fold_results: List[RFEResult],
    min_fold_frequency: float = 0.8,
    method: str = 'frequency',
) -> List[str]:
    """Get consensus features selected across multiple folds."""
    if not fold_results:
        return []

    n_folds = len(fold_results)

    if method == 'intersection':
        feature_sets = [set(r.selected_features) for r in fold_results]
        consensus = feature_sets[0]
        for fs in feature_sets[1:]:
            consensus = consensus.intersection(fs)
        return list(consensus)

    feature_counts: Dict[str, int] = {}
    for result in fold_results:
        for feat in result.selected_features:
            feature_counts[feat] = feature_counts.get(feat, 0) + 1

    min_count = int(n_folds * min_fold_frequency)
    consensus = [f for f, c in feature_counts.items() if c >= min_count]
    consensus.sort(key=lambda x: feature_counts[x], reverse=True)
    return consensus


def selected_features_to_csv(
    features: List[str],
    importances: Optional[List[float]],
    filepath: str,
) -> None:
    """Save selected features to CSV file."""
    if importances is None:
        importances = [1.0 / len(features)] * len(features)
    if len(importances) != len(features):
        importances = [1.0 / len(features)] * len(features)

    pd.DataFrame({
        'feature_name': features,
        'rank': range(1, len(features) + 1),
        'importance': importances,
    }).to_csv(filepath, index=False)
