"""
COMBINED FEATURE ENGINEERING MODULE
====================================
Combines ALL features from:
1. feature_engineering.py (~150 derived)
2. proper_ml_pipeline.py (~161 derived)

TOTAL: ~320+ unique features
"""

import pandas as pd
import numpy as np
from typing import List


def safe_divide(num, denom, fill=0.0):
    """Safe division avoiding divide-by-zero."""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = num / denom
        if isinstance(result, pd.Series):
            result = result.replace([np.inf, -np.inf], fill).fillna(fill)
        else:
            result = np.where(np.isinf(result) | np.isnan(result), fill, result)
    return result


class CombinedFeatureEngineering:
    """Combined feature engineering from both sources."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
    
    def calculate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate ALL features from both sources."""
        df = df.copy()
        
        if self.verbose:
            print("  Computing H4 features...")
        df = self._add_h4_price_change(df)
        df = self._add_h4_lags(df)
        df = self._add_h4_slopes(df)
        df = self._add_h4_zscores(df)
        df = self._add_h4_rolling_stats(df)
        df = self._add_h4_percentiles(df)
        df = self._add_h4_binary(df)
        df = self._add_h4_momentum(df)
        df = self._add_h4_patterns(df)
        df = self._add_h4_session(df)
        df = self._add_h4_regime(df)
        df = self._add_h4_volatility(df)
        df = self._add_h4_mean_reversion(df)
        
        if self.verbose:
            print("  Computing D1 features...")
        df = self._add_d1_binary(df)
        df = self._add_d1_momentum(df)
        df = self._add_d1_mean_reversion(df)
        df = self._add_d1_volatility(df)
        df = self._add_d1_regime(df)
        df = self._add_d1_patterns(df)
        df = self._add_d1_zscores(df)
        
        if self.verbose:
            print("  Computing MTF/Cross-TF features...")
        df = self._add_mtf_alignment(df)
        df = self._add_mtf_divergence(df)
        df = self._add_mtf_relative(df)
        df = self._add_mtf_volatility(df)
        df = self._add_mtf_context(df)
        df = self._add_mtf_signals(df)
        
        # Replace inf with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        return df
    
    # =========================================================================
    # H4 FEATURES
    # =========================================================================
    
    def _add_h4_price_change(self, df):
        df['price_change_1'] = df['close'].pct_change(1)
        df['price_change_3'] = df['close'].pct_change(3)
        df['price_change_5'] = df['close'].pct_change(5)
        df['price_change_10'] = df['close'].pct_change(10)
        df['high_change_1'] = df['high'].pct_change(1)
        df['high_change_3'] = df['high'].pct_change(3)
        df['range_pct'] = safe_divide(df['high'] - df['low'], df['close'])
        return df
    
    def _add_h4_lags(self, df):
        for i in range(1, 6):
            df[f'rsi_lag{i}'] = df['rsi_value'].shift(i)
        for i in range(1, 4):
            df[f'bb_position_lag{i}'] = df['bb_position'].shift(i)
            df[f'volume_ratio_lag{i}'] = df['volume_ratio'].shift(i)
        for i in range(1, 3):
            df[f'bb_width_lag{i}'] = df['bb_width_pct'].shift(i)
            df[f'trend_strength_lag{i}'] = df['trend_strength'].shift(i)
        return df
    
    def _add_h4_slopes(self, df):
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
    
    def _add_h4_zscores(self, df):
        window = 20
        for col, name in [('close', 'price_zscore'), ('rsi_value', 'rsi_zscore'),
                          ('volume_ratio', 'volume_zscore'), ('bb_position', 'bb_position_zscore'),
                          ('atr_pct', 'atr_zscore'), ('high', 'high_zscore')]:
            if col in df.columns:
                roll_mean = df[col].rolling(window).mean()
                roll_std = df[col].rolling(window).std()
                df[name] = safe_divide(df[col] - roll_mean, roll_std)
        return df
    
    def _add_h4_rolling_stats(self, df):
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
    
    def _add_h4_percentiles(self, df):
        for col, name in [('close', 'price_percentile'), ('rsi_value', 'rsi_percentile'),
                          ('volume_ratio', 'volume_percentile'), ('bb_position', 'bb_position_percentile'),
                          ('high', 'high_percentile'), ('atr_pct', 'atr_percentile')]:
            if col in df.columns:
                df[name] = df[col].rolling(50).rank(pct=True)
        return df
    
    def _add_h4_binary(self, df):
        df['touched_upper_bb'] = (df['high'] >= df['upper_band']).astype(int)
        df['rsi_overbought'] = (df['rsi_value'] >= 70).astype(int)
        df['rsi_extreme_overbought'] = (df['rsi_value'] >= 75).astype(int)
        df['rsi_very_extreme'] = (df['rsi_value'] >= 80).astype(int)
        df['bearish_candle'] = (df['close'] < df['open']).astype(int)
        df['bb_very_high'] = (df['bb_position'] > 0.95).astype(int)
        df['bb_above_upper'] = (df['close'] > df['upper_band']).astype(int)
        df['high_volume'] = (df['volume_ratio'] > 1.5).astype(int)
        df['extreme_volume'] = (df['volume_ratio'] > 2.0).astype(int)
        df['strong_uptrend'] = (df['trend_strength'] > 0.7).astype(int)
        if 'candle_body_pct' in df.columns:
            df['strong_bearish'] = ((df['close'] < df['open']) & (df['candle_body_pct'] > 0.6)).astype(int)
        return df
    
    def _add_h4_momentum(self, df):
        df['rsi_momentum'] = df['rsi_value'] - df['rsi_value'].shift(3)
        df['price_momentum_3'] = df['close'].pct_change(3)
        df['price_acceleration'] = df['price_change_1'] - df['price_change_1'].shift(1)
        df['rsi_acceleration'] = df['rsi_slope_3'] - df['rsi_slope_3'].shift(3)
        df['volume_momentum'] = df['volume_ratio'] - df['volume_ratio'].shift(3)
        df['bb_position_momentum'] = df['bb_position'] - df['bb_position'].shift(5)
        df['trend_exhaustion'] = ((df['rsi_value'] > 70) & (df['rsi_slope_3'] < 0) & (df['price_slope_3'] > 0)).astype(int)
        df['price_roc_3'] = df['close'].pct_change(3)
        df['price_roc_5'] = df['close'].pct_change(5)
        df['price_roc_10'] = df['close'].pct_change(10)
        df['rsi_roc_3'] = df['rsi_value'].diff(3)
        df['rsi_roc_5'] = df['rsi_value'].diff(5)
        df['volume_roc_3'] = df['volume_ratio'].pct_change(3)
        df['price_velocity'] = df['close'].diff(1) / df['close'].shift(1)
        df['rsi_velocity'] = df['rsi_value'].diff(1)
        df['momentum_deceleration'] = (df['price_velocity'].shift(1) > df['price_velocity']).astype(int)
        return df
    
    def _add_h4_patterns(self, df):
        bullish = (df['close'] > df['open']).astype(int)
        bearish = (df['close'] < df['open']).astype(int)
        consec_bull, consec_bear = [], []
        bull_count = bear_count = 0
        for b, be in zip(bullish, bearish):
            if b: bull_count += 1; bear_count = 0
            elif be: bear_count += 1; bull_count = 0
            else: bull_count = bear_count = 0
            consec_bull.append(bull_count)
            consec_bear.append(bear_count)
        df['consecutive_bullish'] = consec_bull
        df['consecutive_bearish'] = consec_bear
        df['consecutive_higher_highs'] = (df['high'] > df['high'].shift(1)).rolling(5).sum()
        df['consecutive_higher_closes'] = (df['close'] > df['close'].shift(1)).rolling(5).sum()
        df['upper_wick_ratio'] = safe_divide(df['high'] - df[['open', 'close']].max(axis=1), df['high'] - df['low'])
        df['lower_wick_ratio'] = safe_divide(df[['open', 'close']].min(axis=1) - df['low'], df['high'] - df['low'])
        df['bullish_exhaustion'] = ((df['consecutive_bullish'] >= 3) & (df['rsi_value'] >= 70)).astype(int)
        df['near_double_top'] = ((df['high'] >= df['high'].rolling(20).max() * 0.99)).astype(int)
        if 'candle_body_pct' in df.columns:
            df['doji'] = (df['candle_body_pct'] < 0.1).astype(int)
        if 'candle_rejection' in df.columns and 'candle_body_pct' in df.columns:
            df['shooting_star'] = ((df['candle_rejection'] > 0.6) & (df['candle_body_pct'] < 0.3) & (df['close'] < df['open'])).astype(int)
        df['exhaustion_score'] = df['rsi_overbought'] * 0.3 + df['bb_very_high'] * 0.3 + df['high_volume'] * 0.2
        return df
    
    def _add_h4_session(self, df):
        if 'hour' in df.columns:
            hour = df['hour']
            df['is_asian_session'] = hour.isin([0,1,2,3,4,5,6,7]).astype(int)
            df['is_london_session'] = hour.isin([8,9,10,11,12,13,14,15]).astype(int)
            df['is_ny_session'] = hour.isin([13,14,15,16,17,18,19,20]).astype(int)
            df['is_overlap_session'] = hour.isin([13,14,15,16]).astype(int)
            df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
            df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
        if 'day_of_week' in df.columns:
            dow = df['day_of_week']
            df['is_monday'] = (dow == 0).astype(int)
            df['is_friday'] = (dow == 4).astype(int)
            df['dow_sin'] = np.sin(2 * np.pi * dow / 7)
            df['dow_cos'] = np.cos(2 * np.pi * dow / 7)
        return df
    
    def _add_h4_regime(self, df):
        df['ma_20'] = df['close'].rolling(20).mean()
        df['ma_50'] = df['close'].rolling(50).mean()
        df['trend_direction'] = np.sign(df['close'] - df['ma_50'])
        df['trend_direction_h4'] = np.sign(df['close'] - df['ma_20']).fillna(0)
        df['is_trending'] = (df['trend_strength'].abs() >= 0.5).astype(int)
        df['is_ranging'] = (df['trend_strength'].abs() < 0.3).astype(int)
        df['is_volatile'] = (df['atr_pct'] > df['atr_pct'].rolling(50).mean() * 1.5).astype(int)
        # ADX
        window = 14
        tr = pd.concat([df['high'] - df['low'], abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1))], axis=1).max(axis=1)
        atr = tr.rolling(window).mean()
        plus_dm = df['high'].diff().clip(lower=0)
        minus_dm = (-df['low'].diff()).clip(lower=0)
        plus_dm[plus_dm < minus_dm.abs()] = 0
        minus_dm[minus_dm < plus_dm.abs()] = 0
        plus_di = 100 * plus_dm.rolling(window).mean() / atr.replace(0, np.nan)
        minus_di = 100 * minus_dm.rolling(window).mean() / atr.replace(0, np.nan)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
        df['adx'] = dx.rolling(window).mean()
        return df
    
    def _add_h4_volatility(self, df):
        df['atr_change'] = df['atr_pct'].pct_change(3)
        df['bb_squeeze'] = (df['bb_width_pct'] < df['bb_width_pct'].rolling(20).quantile(0.2)).astype(int)
        df['bb_expansion'] = (df['bb_width_pct'] > df['bb_width_pct'].rolling(20).quantile(0.8)).astype(int)
        df['volatility_spike'] = (df['atr_pct'] > df['atr_pct'].rolling(20).mean() * 1.5).astype(int)
        df['historical_vol'] = df['close'].pct_change().rolling(20).std() * np.sqrt(252)
        df['volatility_ratio'] = safe_divide(df['atr_pct'], df['atr_pct'].rolling(50).mean())
        df['intrabar_vol'] = (df['high'] - df['low']) / df['close']
        return df
    
    def _add_h4_mean_reversion(self, df):
        df['dist_from_ma10'] = safe_divide(df['close'] - df['close'].rolling(10).mean(), df['close'])
        df['dist_from_ma20'] = safe_divide(df['close'] - df['ma_20'], df['close'])
        df['dist_from_ma50'] = safe_divide(df['close'] - df['ma_50'], df['close'])
        df['overextension_20'] = df['dist_from_ma20'].clip(lower=0)
        if 'middle_band' in df.columns:
            df['overextension'] = safe_divide(df['close'] - df['middle_band'], df['middle_band'])
            df['reversion_probability'] = (1 - df['bb_position']).clip(0, 1)
        df['price_above_ma20'] = (df['close'] > df['ma_20']).astype(int)
        df['price_above_ma50'] = (df['close'] > df['ma_50']).astype(int)
        df['all_ma_bullish'] = ((df['close'] > df['ma_20']) & (df['ma_20'] > df['ma_50'])).astype(int)
        df['mean_reversion_score'] = df['dist_from_ma20'] * df['rsi_value'] / 100
        return df
    
    # =========================================================================
    # D1 FEATURES
    # =========================================================================
    
    def _add_d1_binary(self, df):
        if 'd1_high' in df.columns and 'd1_upper_band' in df.columns:
            df['d1_touched_upper_bb'] = (df['d1_high'] >= df['d1_upper_band']).astype(int)
        if 'd1_rsi_value' in df.columns:
            df['d1_rsi_overbought'] = (df['d1_rsi_value'] >= 70).astype(int)
            df['d1_rsi_oversold'] = (df['d1_rsi_value'] <= 30).astype(int)
            df['d1_rsi_extreme'] = (df['d1_rsi_value'] >= 80).astype(int)
        if 'd1_close' in df.columns and 'd1_open' in df.columns:
            df['d1_bearish_candle'] = (df['d1_close'] < df['d1_open']).astype(int)
            df['d1_bullish_candle'] = (df['d1_close'] > df['d1_open']).astype(int)
        if 'd1_bb_position' in df.columns:
            df['d1_bb_very_high'] = (df['d1_bb_position'] > 0.95).astype(int)
            df['d1_bb_high'] = (df['d1_bb_position'] >= 0.8).astype(int)
            df['d1_bb_low'] = (df['d1_bb_position'] <= 0.2).astype(int)
        return df
    
    def _add_d1_momentum(self, df):
        if 'd1_rsi_value' in df.columns:
            df['d1_rsi_slope_3'] = (df['d1_rsi_value'] - df['d1_rsi_value'].shift(3)) / 3
            df['d1_rsi_slope_5'] = (df['d1_rsi_value'] - df['d1_rsi_value'].shift(5)) / 5
        if 'd1_close' in df.columns:
            df['d1_price_roc_3'] = df['d1_close'].pct_change(3)
            df['d1_price_roc_5'] = df['d1_close'].pct_change(5)
            df['d1_price_roc_10'] = df['d1_close'].pct_change(10)
        if 'd1_bb_position' in df.columns:
            df['d1_bb_position_slope_3'] = (df['d1_bb_position'] - df['d1_bb_position'].shift(3)) / 3
        return df
    
    def _add_d1_mean_reversion(self, df):
        if 'd1_close' in df.columns:
            df['d1_dist_from_ma10'] = safe_divide(df['d1_close'] - df['d1_close'].rolling(10).mean(), df['d1_close'].rolling(10).mean())
            df['d1_dist_from_ma20'] = safe_divide(df['d1_close'] - df['d1_close'].rolling(20).mean(), df['d1_close'].rolling(20).mean())
        if 'd1_close' in df.columns and 'd1_middle_band' in df.columns:
            df['d1_overextension'] = safe_divide(df['d1_close'] - df['d1_middle_band'], df['d1_middle_band'])
        return df
    
    def _add_d1_volatility(self, df):
        if 'd1_atr_pct' in df.columns:
            df['d1_atr_percentile'] = df['d1_atr_pct'].rolling(50).rank(pct=True)
        if 'd1_bb_width_pct' in df.columns:
            df['d1_bb_squeeze'] = (df['d1_bb_width_pct'] < df['d1_bb_width_pct'].rolling(20).quantile(0.2)).astype(int)
        return df
    
    def _add_d1_regime(self, df):
        if 'd1_trend_strength' in df.columns:
            df['d1_uptrend'] = (df['d1_trend_strength'] > 0.3).astype(int)
            df['d1_downtrend'] = (df['d1_trend_strength'] < -0.3).astype(int)
            df['d1_strong_trend'] = (df['d1_trend_strength'].abs() > 0.7).astype(int)
        if 'd1_close' in df.columns:
            d1_ma20 = df['d1_close'].rolling(20).mean()
            df['d1_trend_direction'] = np.sign(df['d1_close'] - d1_ma20).fillna(0)
            df['d1_is_trending_up'] = (df['d1_trend_direction'] > 0).astype(int)
            df['d1_is_trending_down'] = (df['d1_trend_direction'] < 0).astype(int)
        return df
    
    def _add_d1_patterns(self, df):
        if 'd1_close' in df.columns and 'd1_open' in df.columns:
            d1_bullish = (df['d1_close'] > df['d1_open']).astype(int)
            d1_bearish = (df['d1_close'] < df['d1_open']).astype(int)
            d1_consec_bull, d1_consec_bear = [], []
            bull_count = bear_count = 0
            for b, be in zip(d1_bullish, d1_bearish):
                if b: bull_count += 1; bear_count = 0
                elif be: bear_count += 1; bull_count = 0
                else: bull_count = bear_count = 0
                d1_consec_bull.append(bull_count)
                d1_consec_bear.append(bear_count)
            df['d1_consecutive_bullish'] = d1_consec_bull
            df['d1_consecutive_bearish'] = d1_consec_bear
            df['d1_near_d1_high'] = (df['d1_close'] > df['d1_close'].rolling(20).max() * 0.98).astype(int)
            if 'd1_rsi_value' in df.columns:
                df['d1_bearish_divergence'] = ((df['d1_close'] > df['d1_close'].shift(5)) & (df['d1_rsi_value'] < df['d1_rsi_value'].shift(5))).astype(int)
        return df
    
    def _add_d1_zscores(self, df):
        window = 20
        if 'd1_close' in df.columns:
            roll_mean = df['d1_close'].rolling(window).mean()
            roll_std = df['d1_close'].rolling(window).std()
            df['d1_price_zscore'] = safe_divide(df['d1_close'] - roll_mean, roll_std)
        if 'd1_rsi_value' in df.columns:
            roll_mean = df['d1_rsi_value'].rolling(window).mean()
            roll_std = df['d1_rsi_value'].rolling(window).std()
            df['d1_rsi_zscore'] = safe_divide(df['d1_rsi_value'] - roll_mean, roll_std)
        return df
    
    # =========================================================================
    # MTF / CROSS-TIMEFRAME FEATURES
    # =========================================================================
    
    def _add_mtf_alignment(self, df):
        if 'd1_rsi_value' in df.columns and 'rsi_value' in df.columns:
            df['mtf_rsi_aligned'] = ((df['rsi_value'] >= 65) & (df['d1_rsi_value'] >= 65)).astype(int)
            df['mtf_overbought_aligned'] = ((df['rsi_value'] >= 70) & (df['d1_rsi_value'] >= 70)).astype(int)
        if 'd1_bb_position' in df.columns and 'bb_position' in df.columns:
            df['mtf_bb_aligned'] = ((df['bb_position'] > 0.8) & (df['d1_bb_position'] > 0.8)).astype(int)
        if 'd1_trend_direction' in df.columns and 'trend_direction_h4' in df.columns:
            df['mtf_trend_aligned'] = (df['trend_direction_h4'] == df['d1_trend_direction']).astype(int)
        if 'd1_bearish_candle' in df.columns and 'bearish_candle' in df.columns:
            df['mtf_bearish_aligned'] = ((df['bearish_candle'] == 1) & (df['d1_bearish_candle'] == 1)).astype(int)
        # Confluence score
        df['mtf_confluence_score'] = (
            df.get('mtf_rsi_aligned', 0) * 0.25 +
            df.get('mtf_bb_aligned', 0) * 0.25 +
            df.get('mtf_trend_aligned', 0) * 0.25 +
            df.get('mtf_bearish_aligned', 0) * 0.25
        )
        return df
    
    def _add_mtf_divergence(self, df):
        if 'd1_rsi_value' in df.columns and 'rsi_value' in df.columns:
            df['mtf_rsi_divergence'] = ((df['rsi_value'] > 70) & (df['d1_rsi_value'] < 50)).astype(int)
        if 'd1_bb_position' in df.columns and 'bb_position' in df.columns:
            df['mtf_bb_divergence'] = ((df['bb_position'] > 0.8) & (df['d1_bb_position'] < 0.5)).astype(int)
        if 'd1_trend_direction' in df.columns and 'trend_direction_h4' in df.columns:
            df['mtf_trend_divergence'] = ((df['trend_direction_h4'] > 0) & (df['d1_trend_direction'] < 0)).astype(int)
        return df
    
    def _add_mtf_relative(self, df):
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
    
    def _add_mtf_volatility(self, df):
        if 'd1_atr_pct' in df.columns and 'atr_pct' in df.columns:
            df['h4_vs_d1_atr_ratio'] = safe_divide(df['atr_pct'], df['d1_atr_pct'])
        if all(c in df.columns for c in ['high', 'low', 'd1_high', 'd1_low']):
            h4_range = df['high'] - df['low']
            d1_range = (df['d1_high'] - df['d1_low']).replace(0, np.nan)
            df['h4_vs_d1_range_ratio'] = h4_range / d1_range
        return df
    
    def _add_mtf_context(self, df):
        if 'd1_rsi_value' in df.columns and 'd1_bb_position' in df.columns:
            df['d1_supports_short'] = ((df['d1_rsi_value'] > 65) & (df['d1_bb_position'] > 0.7)).astype(int)
            df['d1_supports_long'] = ((df['d1_rsi_value'] < 35) & (df['d1_bb_position'] < 0.3)).astype(int)
        if 'd1_trend_direction' in df.columns:
            df['d1_opposes_short'] = ((df.get('d1_rsi_value', 50) < 40) & (df['d1_trend_direction'] > 0)).astype(int)
            df['d1_opposes_long'] = ((df.get('d1_rsi_value', 50) > 60) & (df['d1_trend_direction'] < 0)).astype(int)
        return df
    
    def _add_mtf_signals(self, df):
        # MTF Long Signal: H4 uptrend AND D1 uptrend
        h4_up = df['trend_strength'] > 0.3
        d1_up = df.get('d1_trend_strength', pd.Series(0, index=df.index)) > 0.3
        df['mtf_long_signal'] = (h4_up & d1_up).astype(int)
        
        # MTF Short Signal: H4 downtrend AND D1 downtrend
        h4_down = df['trend_strength'] < -0.3
        d1_down = df.get('d1_trend_strength', pd.Series(0, index=df.index)) < -0.3
        df['mtf_short_signal'] = (h4_down & d1_down).astype(int)
        
        return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Get all valid ML feature columns."""
    exclude = [
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'h4_date', 'd1_available_date', 'd1_timestamp',
        'd1_open', 'd1_high', 'd1_low', 'd1_close', 'd1_volume',
        'trade_label', 'trade_pips', 'trade_bars',
        'mtf_long_signal', 'mtf_short_signal', 'session', 'ma_20', 'ma_50'
    ]
    
    feature_cols = []
    for col in df.columns:
        if col in exclude:
            continue
        if col.startswith('d1_') and any(x in col for x in ['_open', '_high', '_low', '_close', '_volume']):
            continue
        if df[col].dtype in [np.float64, np.int64, np.float32, np.int32, bool, np.bool_]:
            if df[col].notna().sum() > len(df) * 0.3:
                feature_cols.append(col)
    
    return sorted(feature_cols)
