"""
STELLA ALPHA - EXPANDED FEATURES
=================================
Added indicators:
- EMA (8, 21, 50, 200) + crossovers
- MACD (line, signal, histogram)
- Stochastic (%K, %D)
- ADX (+DI, -DI)
- Ichimoku (Tenkan, Kijun, Senkou A/B, Cloud)
- Donchian Channel (20)
- CCI
- Williams %R
- ROC (20, 50)

Extended lags: up to 20 bars for key indicators
Derived features: crossovers, divergences, momentum combinations

Total features: ~350+
"""

import pandas as pd
import numpy as np
from typing import List, Optional
import warnings
warnings.filterwarnings('ignore')


class ExpandedFeatureEngineering:
    """
    Comprehensive feature engineering with expanded indicators and lags.
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.feature_count = 0
    
    def log(self, msg: str):
        if self.verbose:
            print(f"  [Features] {msg}")
    
    def calculate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all features."""
        df = df.copy()
        initial_cols = len(df.columns)
        
        self.log("Calculating H4 base indicators...")
        df = self._add_h4_base_indicators(df)
        
        self.log("Calculating H4 new indicators (EMA, MACD, etc.)...")
        df = self._add_h4_new_indicators(df)
        
        self.log("Calculating H4 extended lags...")
        df = self._add_h4_extended_lags(df)
        
        self.log("Calculating H4 derived features...")
        df = self._add_h4_derived_features(df)
        
        self.log("Calculating H4 crossover signals...")
        df = self._add_h4_crossover_signals(df)
        
        self.log("Calculating H4 momentum features...")
        df = self._add_h4_momentum_features(df)
        
        self.log("Calculating H4 pattern features...")
        df = self._add_h4_pattern_features(df)
        
        self.log("Calculating H4 session features...")
        df = self._add_h4_session_features(df)
        
        self.log("Calculating H4 regime features...")
        df = self._add_h4_regime_features(df)
        
        self.log("Calculating H4 volatility features...")
        df = self._add_h4_volatility_features(df)
        
        self.log("Calculating H4 mean reversion features...")
        df = self._add_h4_mean_reversion_features(df)
        
        # D1 features
        self.log("Calculating D1 features...")
        df = self._add_d1_features(df)
        
        # MTF features
        self.log("Calculating MTF features...")
        df = self._add_mtf_features(df)
        
        # Indicator-based derived features
        self.log("Calculating indicator combinations...")
        df = self._add_indicator_combinations(df)
        
        self.feature_count = len(df.columns) - initial_cols
        self.log(f"Total new features: {self.feature_count}")
        
        return df
    
    # =========================================================================
    # H4 BASE INDICATORS (from existing)
    # =========================================================================
    
    def _add_h4_base_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure base indicators exist."""
        # These should already exist from EA export
        # But calculate if missing
        
        if 'rsi_value' not in df.columns:
            df['rsi_value'] = self._calculate_rsi(df['close'], 14)
        
        if 'bb_position' not in df.columns:
            df['middle_band'] = df['close'].rolling(20).mean()
            df['bb_std'] = df['close'].rolling(20).std()
            df['upper_band'] = df['middle_band'] + 2 * df['bb_std']
            df['lower_band'] = df['middle_band'] - 2 * df['bb_std']
            df['bb_position'] = (df['close'] - df['lower_band']) / (df['upper_band'] - df['lower_band'])
            df['bb_width_pct'] = (df['upper_band'] - df['lower_band']) / df['middle_band']
        
        if 'atr_pct' not in df.columns:
            df['atr'] = self._calculate_atr(df, 14)
            df['atr_pct'] = df['atr'] / df['close']
        
        return df
    
    # =========================================================================
    # H4 NEW INDICATORS
    # =========================================================================
    
    def _add_h4_new_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all new technical indicators."""
        
        # ----- EMA (8, 21, 50, 200) -----
        for period in [8, 21, 50, 200]:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
            df[f'ema_{period}_slope'] = df[f'ema_{period}'].diff(3) / df[f'ema_{period}'].shift(3)
            df[f'price_vs_ema_{period}'] = (df['close'] - df[f'ema_{period}']) / df[f'ema_{period}']
        
        # EMA distances
        df['ema_8_21_dist'] = (df['ema_8'] - df['ema_21']) / df['ema_21']
        df['ema_21_50_dist'] = (df['ema_21'] - df['ema_50']) / df['ema_50']
        df['ema_50_200_dist'] = (df['ema_50'] - df['ema_200']) / df['ema_200']
        
        # ----- MACD (12, 26, 9) -----
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd_line'] = ema12 - ema26
        df['macd_signal'] = df['macd_line'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd_line'] - df['macd_signal']
        df['macd_hist_slope'] = df['macd_histogram'].diff(3)
        df['macd_line_slope'] = df['macd_line'].diff(3)
        df['macd_normalized'] = df['macd_line'] / df['close'] * 1000
        df['macd_hist_normalized'] = df['macd_histogram'] / df['close'] * 1000
        
        # ----- Stochastic (14, 3, 3) -----
        low_14 = df['low'].rolling(14).min()
        high_14 = df['high'].rolling(14).max()
        df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14)
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        df['stoch_k_slope'] = df['stoch_k'].diff(3)
        df['stoch_d_slope'] = df['stoch_d'].diff(3)
        
        # ----- ADX (14) with +DI, -DI -----
        df = self._calculate_adx(df, 14)
        
        # ----- Ichimoku -----
        # Tenkan-sen (9)
        high_9 = df['high'].rolling(9).max()
        low_9 = df['low'].rolling(9).min()
        df['ichimoku_tenkan'] = (high_9 + low_9) / 2
        
        # Kijun-sen (26)
        high_26 = df['high'].rolling(26).max()
        low_26 = df['low'].rolling(26).min()
        df['ichimoku_kijun'] = (high_26 + low_26) / 2
        
        # Senkou Span A (shifted 26 forward, but we use current for ML)
        df['ichimoku_senkou_a'] = (df['ichimoku_tenkan'] + df['ichimoku_kijun']) / 2
        
        # Senkou Span B (52)
        high_52 = df['high'].rolling(52).max()
        low_52 = df['low'].rolling(52).min()
        df['ichimoku_senkou_b'] = (high_52 + low_52) / 2
        
        # Cloud features
        df['ichimoku_cloud_top'] = df[['ichimoku_senkou_a', 'ichimoku_senkou_b']].max(axis=1)
        df['ichimoku_cloud_bottom'] = df[['ichimoku_senkou_a', 'ichimoku_senkou_b']].min(axis=1)
        df['ichimoku_cloud_width'] = (df['ichimoku_cloud_top'] - df['ichimoku_cloud_bottom']) / df['close']
        df['price_vs_cloud'] = np.where(
            df['close'] > df['ichimoku_cloud_top'], 1,
            np.where(df['close'] < df['ichimoku_cloud_bottom'], -1, 0)
        )
        df['price_vs_tenkan'] = (df['close'] - df['ichimoku_tenkan']) / df['ichimoku_tenkan']
        df['price_vs_kijun'] = (df['close'] - df['ichimoku_kijun']) / df['ichimoku_kijun']
        df['tenkan_vs_kijun'] = (df['ichimoku_tenkan'] - df['ichimoku_kijun']) / df['ichimoku_kijun']
        
        # ----- Donchian Channel (20) -----
        df['donchian_high'] = df['high'].rolling(20).max()
        df['donchian_low'] = df['low'].rolling(20).min()
        df['donchian_mid'] = (df['donchian_high'] + df['donchian_low']) / 2
        df['donchian_width'] = (df['donchian_high'] - df['donchian_low']) / df['close']
        df['donchian_position'] = (df['close'] - df['donchian_low']) / (df['donchian_high'] - df['donchian_low'])
        df['donchian_breakout_up'] = (df['high'] >= df['donchian_high'].shift(1)).astype(int)
        df['donchian_breakout_down'] = (df['low'] <= df['donchian_low'].shift(1)).astype(int)
        
        # ----- CCI (20) -----
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = typical_price.rolling(20).mean()
        mad = typical_price.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
        df['cci'] = (typical_price - sma_tp) / (0.015 * mad)
        df['cci_slope'] = df['cci'].diff(3)
        
        # ----- Williams %R (14) -----
        df['williams_r'] = -100 * (high_14 - df['close']) / (high_14 - low_14)
        df['williams_r_slope'] = df['williams_r'].diff(3)
        
        # ----- ROC (Rate of Change) - Extended -----
        for period in [5, 10, 20, 50]:
            df[f'roc_{period}'] = (df['close'] - df['close'].shift(period)) / df['close'].shift(period) * 100
        
        # ----- Momentum -----
        for period in [10, 20]:
            df[f'momentum_{period}'] = df['close'] - df['close'].shift(period)
        
        return df
    
    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate ADX with +DI and -DI."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed
        atr = pd.Series(tr).rolling(period).mean()
        plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / atr
        
        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        
        df['adx'] = adx
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di
        df['di_diff'] = plus_di - minus_di
        df['di_ratio'] = plus_di / (minus_di + 0.001)
        df['adx_slope'] = df['adx'].diff(3)
        
        return df
    
    # =========================================================================
    # H4 EXTENDED LAGS (up to 20)
    # =========================================================================
    
    def _add_h4_extended_lags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add extended lags for key indicators."""
        
        # RSI lags: 1-20
        for i in range(1, 21):
            df[f'rsi_lag{i}'] = df['rsi_value'].shift(i)
        
        # BB position lags: 1-20
        for i in range(1, 21):
            df[f'bb_position_lag{i}'] = df['bb_position'].shift(i)
        
        # Close price lags (normalized)
        for i in [1, 2, 3, 5, 10, 15, 20]:
            df[f'close_pct_lag{i}'] = (df['close'] - df['close'].shift(i)) / df['close'].shift(i)
        
        # ATR lags
        for i in [1, 3, 5, 10, 20]:
            df[f'atr_lag{i}'] = df['atr_pct'].shift(i)
        
        # Volume lags
        if 'volume_ratio' in df.columns:
            for i in [1, 3, 5, 10, 20]:
                df[f'volume_ratio_lag{i}'] = df['volume_ratio'].shift(i)
        
        # MACD lags
        for i in [1, 3, 5, 10]:
            df[f'macd_hist_lag{i}'] = df['macd_histogram'].shift(i)
            df[f'macd_line_lag{i}'] = df['macd_line'].shift(i)
        
        # Stochastic lags
        for i in [1, 3, 5, 10]:
            df[f'stoch_k_lag{i}'] = df['stoch_k'].shift(i)
        
        # ADX lags
        for i in [1, 3, 5, 10]:
            df[f'adx_lag{i}'] = df['adx'].shift(i)
        
        # EMA lags
        for i in [1, 3, 5, 10]:
            df[f'price_vs_ema_21_lag{i}'] = df['price_vs_ema_21'].shift(i)
            df[f'price_vs_ema_50_lag{i}'] = df['price_vs_ema_50'].shift(i)
        
        # CCI lags
        for i in [1, 3, 5, 10]:
            df[f'cci_lag{i}'] = df['cci'].shift(i)
        
        # Williams %R lags
        for i in [1, 3, 5, 10]:
            df[f'williams_r_lag{i}'] = df['williams_r'].shift(i)
        
        # Ichimoku lags
        for i in [1, 3, 5]:
            df[f'tenkan_vs_kijun_lag{i}'] = df['tenkan_vs_kijun'].shift(i)
        
        return df
    
    # =========================================================================
    # H4 CROSSOVER SIGNALS
    # =========================================================================
    
    def _add_h4_crossover_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add crossover-based signals."""
        
        # EMA crossovers
        df['ema_8_21_cross_up'] = ((df['ema_8'] > df['ema_21']) & (df['ema_8'].shift(1) <= df['ema_21'].shift(1))).astype(int)
        df['ema_8_21_cross_down'] = ((df['ema_8'] < df['ema_21']) & (df['ema_8'].shift(1) >= df['ema_21'].shift(1))).astype(int)
        df['ema_21_50_cross_up'] = ((df['ema_21'] > df['ema_50']) & (df['ema_21'].shift(1) <= df['ema_50'].shift(1))).astype(int)
        df['ema_21_50_cross_down'] = ((df['ema_21'] < df['ema_50']) & (df['ema_21'].shift(1) >= df['ema_50'].shift(1))).astype(int)
        df['ema_50_200_cross_up'] = ((df['ema_50'] > df['ema_200']) & (df['ema_50'].shift(1) <= df['ema_200'].shift(1))).astype(int)
        df['ema_50_200_cross_down'] = ((df['ema_50'] < df['ema_200']) & (df['ema_50'].shift(1) >= df['ema_200'].shift(1))).astype(int)
        
        # EMA alignment (all EMAs aligned)
        df['ema_bullish_alignment'] = ((df['ema_8'] > df['ema_21']) & (df['ema_21'] > df['ema_50']) & (df['ema_50'] > df['ema_200'])).astype(int)
        df['ema_bearish_alignment'] = ((df['ema_8'] < df['ema_21']) & (df['ema_21'] < df['ema_50']) & (df['ema_50'] < df['ema_200'])).astype(int)
        
        # MACD crossovers
        df['macd_cross_up'] = ((df['macd_line'] > df['macd_signal']) & (df['macd_line'].shift(1) <= df['macd_signal'].shift(1))).astype(int)
        df['macd_cross_down'] = ((df['macd_line'] < df['macd_signal']) & (df['macd_line'].shift(1) >= df['macd_signal'].shift(1))).astype(int)
        df['macd_zero_cross_up'] = ((df['macd_line'] > 0) & (df['macd_line'].shift(1) <= 0)).astype(int)
        df['macd_zero_cross_down'] = ((df['macd_line'] < 0) & (df['macd_line'].shift(1) >= 0)).astype(int)
        
        # Stochastic crossovers
        df['stoch_cross_up'] = ((df['stoch_k'] > df['stoch_d']) & (df['stoch_k'].shift(1) <= df['stoch_d'].shift(1))).astype(int)
        df['stoch_cross_down'] = ((df['stoch_k'] < df['stoch_d']) & (df['stoch_k'].shift(1) >= df['stoch_d'].shift(1))).astype(int)
        
        # Ichimoku crossovers
        df['ichimoku_tk_cross_up'] = ((df['ichimoku_tenkan'] > df['ichimoku_kijun']) & (df['ichimoku_tenkan'].shift(1) <= df['ichimoku_kijun'].shift(1))).astype(int)
        df['ichimoku_tk_cross_down'] = ((df['ichimoku_tenkan'] < df['ichimoku_kijun']) & (df['ichimoku_tenkan'].shift(1) >= df['ichimoku_kijun'].shift(1))).astype(int)
        
        # Bars since crossover
        df['bars_since_ema_8_21_cross'] = self._bars_since_event(df['ema_8_21_cross_up'] | df['ema_8_21_cross_down'])
        df['bars_since_macd_cross'] = self._bars_since_event(df['macd_cross_up'] | df['macd_cross_down'])
        
        return df
    
    def _bars_since_event(self, event_series: pd.Series) -> pd.Series:
        """Count bars since last event."""
        event_series = event_series.astype(bool)
        groups = (~event_series).cumsum()
        return groups.groupby(groups).cumcount()
    
    # =========================================================================
    # H4 DERIVED FEATURES (from existing)
    # =========================================================================
    
    def _add_h4_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived features from base indicators."""
        
        # Z-scores
        for col, window in [('rsi_value', 50), ('bb_position', 50), ('close', 50), ('cci', 50), ('stoch_k', 50)]:
            if col in df.columns:
                mean = df[col].rolling(window).mean()
                std = df[col].rolling(window).std()
                df[f'{col}_zscore'] = (df[col] - mean) / (std + 0.0001)
        
        # Percentiles
        for col, window in [('rsi_value', 100), ('bb_position', 100), ('atr_pct', 100), ('adx', 100)]:
            if col in df.columns:
                df[f'{col}_percentile'] = df[col].rolling(window).apply(
                    lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
                )
        
        # Rolling stats
        for window in [10, 20, 50]:
            df[f'close_std_{window}'] = df['close'].rolling(window).std() / df['close']
            df[f'rsi_std_{window}'] = df['rsi_value'].rolling(window).std()
            df[f'rsi_max_{window}'] = df['rsi_value'].rolling(window).max()
            df[f'rsi_min_{window}'] = df['rsi_value'].rolling(window).min()
            df[f'rsi_range_{window}'] = df[f'rsi_max_{window}'] - df[f'rsi_min_{window}']
        
        # Slopes for multiple windows
        for window in [3, 5, 10, 20]:
            df[f'rsi_slope_{window}'] = (df['rsi_value'] - df['rsi_value'].shift(window)) / window
            df[f'price_slope_{window}'] = (df['close'] - df['close'].shift(window)) / df['close'].shift(window)
            df[f'bb_position_slope_{window}'] = (df['bb_position'] - df['bb_position'].shift(window)) / window
        
        return df
    
    # =========================================================================
    # H4 MOMENTUM FEATURES
    # =========================================================================
    
    def _add_h4_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum-based features."""
        
        # Price momentum
        df['price_momentum_3'] = df['close'].diff(3) / df['close'].shift(3)
        df['price_momentum_5'] = df['close'].diff(5) / df['close'].shift(5)
        df['price_momentum_10'] = df['close'].diff(10) / df['close'].shift(10)
        df['price_momentum_20'] = df['close'].diff(20) / df['close'].shift(20)
        
        # Acceleration
        df['price_accel_5'] = df['price_momentum_5'].diff(5)
        df['price_accel_10'] = df['price_momentum_10'].diff(10)
        df['rsi_accel'] = df['rsi_value'].diff(3).diff(3)
        df['macd_accel'] = df['macd_histogram'].diff(3).diff(3)
        
        # Velocity consistency
        df['momentum_consistency'] = df['price_momentum_3'].rolling(10).apply(
            lambda x: (x > 0).sum() / len(x), raw=True
        )
        
        # RSI momentum
        df['rsi_momentum_3'] = df['rsi_value'].diff(3)
        df['rsi_momentum_5'] = df['rsi_value'].diff(5)
        df['rsi_momentum_10'] = df['rsi_value'].diff(10)
        
        return df
    
    # =========================================================================
    # H4 PATTERN FEATURES
    # =========================================================================
    
    def _add_h4_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add pattern recognition features."""
        
        # Candle patterns
        df['candle_body'] = df['close'] - df['open']
        df['candle_range'] = df['high'] - df['low']
        df['candle_body_pct'] = abs(df['candle_body']) / (df['candle_range'] + 0.00001)
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        df['upper_wick_pct'] = df['upper_wick'] / (df['candle_range'] + 0.00001)
        df['lower_wick_pct'] = df['lower_wick'] / (df['candle_range'] + 0.00001)
        
        # Consecutive patterns
        df['bullish_candle'] = (df['close'] > df['open']).astype(int)
        df['bearish_candle'] = (df['close'] < df['open']).astype(int)
        df['consecutive_bullish'] = df['bullish_candle'].groupby((df['bullish_candle'] != df['bullish_candle'].shift()).cumsum()).cumsum()
        df['consecutive_bearish'] = df['bearish_candle'].groupby((df['bearish_candle'] != df['bearish_candle'].shift()).cumsum()).cumsum()
        
        # Higher highs / Lower lows
        df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
        df['higher_close'] = (df['close'] > df['close'].shift(1)).astype(int)
        df['lower_close'] = (df['close'] < df['close'].shift(1)).astype(int)
        
        df['consecutive_higher_highs'] = df['higher_high'].groupby((df['higher_high'] != df['higher_high'].shift()).cumsum()).cumsum()
        df['consecutive_lower_lows'] = df['lower_low'].groupby((df['lower_low'] != df['lower_low'].shift()).cumsum()).cumsum()
        
        # Engulfing patterns
        df['bullish_engulfing'] = ((df['close'] > df['open']) & 
                                    (df['close'].shift(1) < df['open'].shift(1)) &
                                    (df['close'] > df['open'].shift(1)) &
                                    (df['open'] < df['close'].shift(1))).astype(int)
        df['bearish_engulfing'] = ((df['close'] < df['open']) & 
                                    (df['close'].shift(1) > df['open'].shift(1)) &
                                    (df['close'] < df['open'].shift(1)) &
                                    (df['open'] > df['close'].shift(1))).astype(int)
        
        # Doji
        df['is_doji'] = (df['candle_body_pct'] < 0.1).astype(int)
        
        # Pin bars
        df['bullish_pin'] = ((df['lower_wick_pct'] > 0.6) & (df['candle_body_pct'] < 0.3)).astype(int)
        df['bearish_pin'] = ((df['upper_wick_pct'] > 0.6) & (df['candle_body_pct'] < 0.3)).astype(int)
        
        return df
    
    # =========================================================================
    # H4 SESSION FEATURES
    # =========================================================================
    
    def _add_h4_session_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add session-based features."""
        
        if 'hour' not in df.columns:
            df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        if 'day_of_week' not in df.columns:
            df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        
        # Session flags
        df['is_asian'] = df['hour'].isin([0, 4, 8]).astype(int)  # Rough H4 sessions
        df['is_london'] = df['hour'].isin([8, 12]).astype(int)
        df['is_ny'] = df['hour'].isin([12, 16]).astype(int)
        df['is_overlap'] = df['hour'].isin([12, 16]).astype(int)
        
        # Day flags
        df['is_monday'] = (df['day_of_week'] == 0).astype(int)
        df['is_friday'] = (df['day_of_week'] == 4).astype(int)
        df['is_midweek'] = df['day_of_week'].isin([1, 2, 3]).astype(int)
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 5)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 5)
        
        return df
    
    # =========================================================================
    # H4 REGIME FEATURES
    # =========================================================================
    
    def _add_h4_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market regime features."""
        
        # Trend regime
        df['is_uptrend'] = (df['ema_21'] > df['ema_50']).astype(int)
        df['is_downtrend'] = (df['ema_21'] < df['ema_50']).astype(int)
        df['is_strong_trend'] = (df['adx'] > 25).astype(int)
        df['is_weak_trend'] = (df['adx'] < 20).astype(int)
        
        # Volatility regime
        atr_percentile = df['atr_pct'].rolling(100).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)
        df['high_volatility'] = (atr_percentile > 0.7).astype(int)
        df['low_volatility'] = (atr_percentile < 0.3).astype(int)
        
        # Combined regime score
        df['trend_regime_score'] = df['adx'] * np.sign(df['di_diff'])
        df['volatility_regime_score'] = atr_percentile
        
        return df
    
    # =========================================================================
    # H4 VOLATILITY FEATURES
    # =========================================================================
    
    def _add_h4_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based features."""
        
        # ATR-based
        df['atr_ratio_5_20'] = df['atr_pct'].rolling(5).mean() / df['atr_pct'].rolling(20).mean()
        df['atr_expansion'] = (df['atr_pct'] > df['atr_pct'].rolling(20).mean()).astype(int)
        df['atr_contraction'] = (df['atr_pct'] < df['atr_pct'].rolling(20).mean() * 0.7).astype(int)
        
        # BB-based volatility
        df['bb_squeeze'] = (df['bb_width_pct'] < df['bb_width_pct'].rolling(50).quantile(0.2)).astype(int)
        df['bb_expansion'] = (df['bb_width_pct'] > df['bb_width_pct'].rolling(50).quantile(0.8)).astype(int)
        
        # Range-based
        df['daily_range_pct'] = (df['high'] - df['low']) / df['close']
        df['range_vs_atr'] = df['daily_range_pct'] / df['atr_pct']
        
        # Historical volatility
        returns = df['close'].pct_change()
        df['hist_vol_10'] = returns.rolling(10).std() * np.sqrt(252 * 6)  # Annualized
        df['hist_vol_20'] = returns.rolling(20).std() * np.sqrt(252 * 6)
        df['vol_ratio'] = df['hist_vol_10'] / df['hist_vol_20']
        
        return df
    
    # =========================================================================
    # H4 MEAN REVERSION FEATURES
    # =========================================================================
    
    def _add_h4_mean_reversion_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add mean reversion features."""
        
        # Distance from moving averages
        df['dist_from_ema_8'] = (df['close'] - df['ema_8']) / df['ema_8']
        df['dist_from_ema_21'] = (df['close'] - df['ema_21']) / df['ema_21']
        df['dist_from_ema_50'] = (df['close'] - df['ema_50']) / df['ema_50']
        df['dist_from_ema_200'] = (df['close'] - df['ema_200']) / df['ema_200']
        
        # Overextension indicators
        df['price_overextended_up'] = (df['close'] > df['upper_band']).astype(int)
        df['price_overextended_down'] = (df['close'] < df['lower_band']).astype(int)
        df['rsi_overextended_up'] = (df['rsi_value'] > 70).astype(int)
        df['rsi_overextended_down'] = (df['rsi_value'] < 30).astype(int)
        
        # Mean reversion score
        df['mean_reversion_score'] = (
            df['rsi_value_zscore'].clip(-3, 3) / 3 +
            df['bb_position_zscore'].clip(-3, 3) / 3
        ) / 2
        
        return df
    
    # =========================================================================
    # INDICATOR COMBINATIONS
    # =========================================================================
    
    def _add_indicator_combinations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add combined indicator features."""
        
        # RSI + MACD alignment
        df['rsi_macd_bullish'] = ((df['rsi_value'] > 50) & (df['macd_histogram'] > 0)).astype(int)
        df['rsi_macd_bearish'] = ((df['rsi_value'] < 50) & (df['macd_histogram'] < 0)).astype(int)
        
        # Stochastic + RSI
        df['stoch_rsi_oversold'] = ((df['stoch_k'] < 20) & (df['rsi_value'] < 30)).astype(int)
        df['stoch_rsi_overbought'] = ((df['stoch_k'] > 80) & (df['rsi_value'] > 70)).astype(int)
        
        # ADX + EMA alignment
        df['trend_confluence'] = (df['is_strong_trend'] & df['ema_bullish_alignment']).astype(int)
        
        # CCI + BB
        df['cci_bb_extreme_up'] = ((df['cci'] > 100) & (df['bb_position'] > 0.9)).astype(int)
        df['cci_bb_extreme_down'] = ((df['cci'] < -100) & (df['bb_position'] < 0.1)).astype(int)
        
        # Ichimoku + EMA
        df['ichimoku_ema_bullish'] = ((df['price_vs_cloud'] == 1) & (df['close'] > df['ema_50'])).astype(int)
        df['ichimoku_ema_bearish'] = ((df['price_vs_cloud'] == -1) & (df['close'] < df['ema_50'])).astype(int)
        
        # Momentum divergences
        # Price up but RSI down
        df['bearish_divergence_5'] = ((df['close'] > df['close'].shift(5)) & (df['rsi_value'] < df['rsi_value'].shift(5))).astype(int)
        df['bearish_divergence_10'] = ((df['close'] > df['close'].shift(10)) & (df['rsi_value'] < df['rsi_value'].shift(10))).astype(int)
        df['bullish_divergence_5'] = ((df['close'] < df['close'].shift(5)) & (df['rsi_value'] > df['rsi_value'].shift(5))).astype(int)
        df['bullish_divergence_10'] = ((df['close'] < df['close'].shift(10)) & (df['rsi_value'] > df['rsi_value'].shift(10))).astype(int)
        
        # MACD divergences
        df['macd_bearish_div'] = ((df['close'] > df['close'].shift(5)) & (df['macd_histogram'] < df['macd_histogram'].shift(5))).astype(int)
        df['macd_bullish_div'] = ((df['close'] < df['close'].shift(5)) & (df['macd_histogram'] > df['macd_histogram'].shift(5))).astype(int)
        
        # Multi-indicator score (sum of bullish signals)
        df['bullish_indicator_count'] = (
            (df['rsi_value'] > 50).astype(int) +
            (df['macd_histogram'] > 0).astype(int) +
            (df['stoch_k'] > 50).astype(int) +
            (df['di_diff'] > 0).astype(int) +
            (df['price_vs_cloud'] == 1).astype(int) +
            (df['close'] > df['ema_50']).astype(int)
        )
        
        df['bearish_indicator_count'] = (
            (df['rsi_value'] < 50).astype(int) +
            (df['macd_histogram'] < 0).astype(int) +
            (df['stoch_k'] < 50).astype(int) +
            (df['di_diff'] < 0).astype(int) +
            (df['price_vs_cloud'] == -1).astype(int) +
            (df['close'] < df['ema_50']).astype(int)
        )
        
        df['indicator_consensus'] = df['bullish_indicator_count'] - df['bearish_indicator_count']
        
        return df
    
    # =========================================================================
    # D1 FEATURES
    # =========================================================================
    
    def _add_d1_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add D1 timeframe features."""
        
        # D1 indicators (if available)
        d1_cols = [c for c in df.columns if c.startswith('d1_')]
        
        if 'd1_close' in df.columns:
            # D1 trend
            if 'd1_ema_21' not in df.columns:
                df['d1_ema_21'] = df['d1_close'].ewm(span=21, adjust=False).mean()
            if 'd1_ema_50' not in df.columns:
                df['d1_ema_50'] = df['d1_close'].ewm(span=50, adjust=False).mean()
            
            df['d1_trend'] = np.where(df['d1_close'] > df['d1_ema_21'], 1, -1)
            df['d1_trend_strength'] = abs(df['d1_close'] - df['d1_ema_21']) / df['d1_ema_21']
        
        if 'd1_rsi_value' in df.columns:
            df['d1_rsi_overbought'] = (df['d1_rsi_value'] > 70).astype(int)
            df['d1_rsi_oversold'] = (df['d1_rsi_value'] < 30).astype(int)
            df['d1_rsi_momentum'] = df['d1_rsi_value'].diff(1)
        
        if 'd1_bb_position' in df.columns:
            df['d1_bb_extreme_high'] = (df['d1_bb_position'] > 0.9).astype(int)
            df['d1_bb_extreme_low'] = (df['d1_bb_position'] < 0.1).astype(int)
        
        return df
    
    # =========================================================================
    # MTF FEATURES
    # =========================================================================
    
    def _add_mtf_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add multi-timeframe alignment features."""
        
        # H4 trend direction
        df['h4_trend_dir'] = np.where(df['close'] > df['ema_21'], 1, -1)
        
        # D1 trend direction (if available)
        if 'd1_trend' in df.columns:
            df['mtf_trend_aligned'] = (df['h4_trend_dir'] == df['d1_trend']).astype(int)
            df['mtf_trend_aligned_bullish'] = ((df['h4_trend_dir'] == 1) & (df['d1_trend'] == 1)).astype(int)
            df['mtf_trend_aligned_bearish'] = ((df['h4_trend_dir'] == -1) & (df['d1_trend'] == -1)).astype(int)
        
        # RSI alignment
        if 'd1_rsi_value' in df.columns:
            df['mtf_rsi_both_overbought'] = ((df['rsi_value'] > 65) & (df['d1_rsi_value'] > 65)).astype(int)
            df['mtf_rsi_both_oversold'] = ((df['rsi_value'] < 35) & (df['d1_rsi_value'] < 35)).astype(int)
            df['h4_d1_rsi_diff'] = df['rsi_value'] - df['d1_rsi_value']
        
        # BB alignment
        if 'd1_bb_position' in df.columns:
            df['h4_d1_bb_diff'] = df['bb_position'] - df['d1_bb_position']
            df['mtf_bb_both_high'] = ((df['bb_position'] > 0.8) & (df['d1_bb_position'] > 0.8)).astype(int)
            df['mtf_bb_both_low'] = ((df['bb_position'] < 0.2) & (df['d1_bb_position'] < 0.2)).astype(int)
        
        # MTF signal (for labeling)
        if 'd1_trend_strength' in df.columns and 'trend_strength' in df.columns:
            df['mtf_long_signal'] = (
                (df['trend_strength'] > 0.3) & 
                (df['d1_trend_strength'] > 0.3) &
                (df['h4_trend_dir'] == 1)
            ).astype(int)
        else:
            df['mtf_long_signal'] = (df['h4_trend_dir'] == 1).astype(int)
        
        return df
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 0.0001)
        return 100 - (100 / (1 + rs))
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        return tr.rolling(period).mean()


# =============================================================================
# FEATURE COLUMN GETTER
# =============================================================================

def get_expanded_feature_columns(df: pd.DataFrame) -> List[str]:
    """Get list of feature columns (excluding metadata and non-numeric)."""
    exclude = [
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'd1_timestamp', 'd1_open', 'd1_high', 'd1_low', 'd1_close', 'd1_volume',
        'h4_date', 'd1_date', 'd1_available_date',
        'trade_label', 'trade_pips', 'label',
        'mtf_long_signal', 'mtf_short_signal',
        'pair', 'symbol', 'd1_pair', 'd1_symbol'  # String columns
    ]
    
    feature_cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if c.endswith('_signal'):
            continue
        # Only include numeric columns
        if df[c].dtype in ['float64', 'float32', 'int64', 'int32', 'int16', 'int8', 'uint8', 'uint16', 'uint32', 'uint64']:
            feature_cols.append(c)
    
    return feature_cols


# =============================================================================
# MAIN TEST
# =============================================================================

if __name__ == "__main__":
    print("Testing Expanded Feature Engineering...")
    
    # Create sample data
    np.random.seed(42)
    n = 1000
    
    dates = pd.date_range('2020-01-01', periods=n, freq='4H')
    close = 1.1 + np.cumsum(np.random.randn(n) * 0.001)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': close - np.random.rand(n) * 0.001,
        'high': close + np.random.rand(n) * 0.002,
        'low': close - np.random.rand(n) * 0.002,
        'close': close,
        'volume': np.random.randint(1000, 10000, n)
    })
    
    # Add D1 data (simulated)
    df['d1_close'] = df['close'].rolling(6).mean()
    df['d1_rsi_value'] = 50 + np.random.randn(n) * 10
    df['d1_bb_position'] = np.random.rand(n)
    df['d1_trend_strength'] = np.random.rand(n) * 0.5
    
    # Calculate features
    fe = ExpandedFeatureEngineering(verbose=True)
    df = fe.calculate_all_features(df)
    
    feature_cols = get_expanded_feature_columns(df)
    print(f"\nTotal features: {len(feature_cols)}")
    print(f"Sample features: {feature_cols[:20]}")
