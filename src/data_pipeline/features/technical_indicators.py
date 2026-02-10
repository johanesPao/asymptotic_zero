"""
Technical Indicators Calculator

Calculates 200+ technical analysis features from OHLCV data using TA-Lib.
Features are organized into categories:
- Trend (SMA, EMA, MACD, ADX, etc.)
- Momentum (RSI, Stochastic, CCI, etc.)
- Volatility (Bollinger Bands, ATR, etc.)
- Volume (OBV, AD, CMF, etc.)
- Price (Returns, Candle patterns, etc.)
- Pattern (Trend direction, consecutive candles, etc.)

Usage:
    from src.data_pipeline.features import TechnicalIndicators

    calculator = TechnicalIndicators()
    df_with_features = calculator.calculate(df_ohlcv)

    # Or calculate for a specific date from 5m data
    df_features = calculator.calculate_for_date("2024-01-15")
"""

import numpy as np
import polars as pl
from pathlib import Path
from typing import Optional, List, Dict
import talib
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """
    Calculates technical indicators from OHLCV data.

    Uses TA-Lib for efficient calculations and supports configurable
    parameters via YAML config file
    """

    def __init__(
        self,
        config_path: str = "config/features.yaml",
        data_directory: str = "data/volatility/5m",
    ):
        """
        Initialize the calculator.

        Args:
            config_path: Path to features configuration YAML
            data_directory: Directory containing 5m parquet files
        """
        self.config_path = Path(config_path)
        self.data_directory = Path(data_directory)

        # Load configuration
        self.config = self._load_config()

        # Cache for computed indicators (to avoid recalculating for cross features)
        self._cache: Dict[str, np.ndarray] = {}

    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            return yaml.safe_load(f)

    def _to_numpy(self, series: pl.Series) -> np.ndarray:
        """Convert Polars series to numpy array for TA-Lib.

        np.ascontiguousarray is critical here:
        - During training, data comes from parquet files (single Arrow chunk → contiguous).
        - During live trading, data comes from Binance API lists → multiple .with_columns()
          calls create multi-chunk Arrow arrays → .to_numpy() returns a non-contiguous view
          → .astype(float64) on an already-float64 view returns ANOTHER view, not a copy.
        TA-Lib does raw C pointer arithmetic and crashes with SIGABRT / free(): invalid pointer
        on non-contiguous arrays.  np.ascontiguousarray always produces a new, owned,
        C-contiguous copy regardless of the input layout.
        """
        return np.ascontiguousarray(series.to_numpy(), dtype=np.float64)

    def _clear_cache(self):
        """Clear the indicator cache."""
        self._cache = {}

    # ═══════════════════════════════════════════════════════════════════════════
    # TREND INDICATORS
    # ═══════════════════════════════════════════════════════════════════════════

    def _calculate_trend_indicators(
        self, open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Calculate all trend indicators."""
        results = {}
        trend_config = self.config.get("trend", {})

        # SMA - Simple Moving Average
        if "sma" in trend_config:
            for period in trend_config["sma"]["periods"]:
                results[f"sma_{period}"] = talib.SMA(close, timeperiod=period)
                self._cache[f"sma_{period}"] = results[f"sma_{period}"]

        # EMA - Exponential Moving Average
        if "ema" in trend_config:
            for period in trend_config["ema"]["periods"]:
                results[f"ema_{period}"] = talib.EMA(close, timeperiod=period)
                self._cache[f"ema_{period}"] = results[f"ema_{period}"]

        # WMA - Weighted Moving Average
        if "wma" in trend_config:
            for period in trend_config["wma"]["periods"]:
                results[f"wma_{period}"] = talib.WMA(close, timeperiod=period)

        # DEMA - Double Exponential Moving Average
        if "dema" in trend_config:
            for period in trend_config["dema"]["periods"]:
                results[f"dema_{period}"] = talib.DEMA(close, timeperiod=period)

        # TEMA - Triple Exponential Moving Average
        if "tema" in trend_config:
            for period in trend_config["tema"]["periods"]:
                results[f"tema_{period}"] = talib.TEMA(close, timeperiod=period)

        # KAMA - Kaufman Adaptive Moving Average
        if "kama" in trend_config:
            for period in trend_config["kama"]["periods"]:
                results[f"kama_{period}"] = talib.KAMA(close, timeperiod=period)

        # T3 - Triple Exponential Moving Average (T3)
        if "t3" in trend_config:
            vfactor = trend_config["t3"].get("vfactor", 0.7)
            for period in trend_config["t3"]["periods"]:
                results[f"t3_{period}"] = talib.T3(
                    close, timeperiod=period, vfactor=vfactor
                )

        # MACD - Moving Average Convergence Divergence
        if "macd" in trend_config:
            macd_cfg = trend_config["macd"]
            macd, macd_signal, macd_hist = talib.MACD(
                close,
                fastperiod=macd_cfg["fast_period"],
                slowperiod=macd_cfg["slow_period"],
                signalperiod=macd_cfg["signal_period"],
            )
            results["macd_line"] = macd
            results["macd_signal"] = macd_signal
            results["macd_histogram"] = macd_hist

        # ADX - Average Directional Index
        if "adx" in trend_config:
            for period in trend_config["adx"]["periods"]:
                results[f"adx_{period}"] = talib.ADX(
                    high, low, close, timeperiod=period
                )

        # Plus/Minus Directional Index
        if "di" in trend_config:
            for period in trend_config["di"]["periods"]:
                results[f"plus_di_{period}"] = talib.PLUS_DI(
                    high, low, close, timeperiod=period
                )
                results[f"minus_di_{period}"] = talib.MINUS_DI(
                    high, low, close, timeperiod=period
                )

        # Aroon Indicator
        if "aroon" in trend_config:
            for period in trend_config["aroon"]["periods"]:
                aroon_down, aroon_up = talib.AROON(high, low, timeperiod=period)
                results[f"aroon_up_{period}"] = aroon_up
                results[f"aroon_down_{period}"] = aroon_down
                results[f"aroon_osc_{period}"] = talib.AROONOSC(
                    high, low, timeperiod=period
                )

        # Parabolic SAR
        if "sar" in trend_config:
            sar_cfg = trend_config["sar"]
            results["sar"] = talib.SAR(
                high,
                low,
                acceleration=sar_cfg["acceleration"],
                maximum=sar_cfg["maximum"],
            )

        return results

    # ═══════════════════════════════════════════════════════════════════════════
    # MOMENTUM INDICATORS
    # ═══════════════════════════════════════════════════════════════════════════

    def _calculate_momentum_indicators(
        self,
        open_: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Calculate all momentum indicators."""
        results = {}
        momentum_config = self.config.get("momentum", {})

        # RSI - Relative Strength Index
        if "rsi" in momentum_config:
            for period in momentum_config["rsi"]["periods"]:
                results[f"rsi_{period}"] = talib.RSI(close, timeperiod=period)
                if period == 14:
                    self._cache["rsi_14"] = results[f"rsi_{period}"]

        # Stochastic Oscillator
        if "stochastic" in momentum_config:
            stoch_cfg = momentum_config["stochastic"]
            for period in stoch_cfg["periods"]:
                slowk, slowd = talib.STOCH(
                    high,
                    low,
                    close,
                    fastk_period=period,
                    slowk_period=stoch_cfg["k_smooth"],
                    slowk_matype=0,
                    slowd_period=stoch_cfg["d_smooth"],
                    slowd_matype=0,
                )
                results[f"stoch_k_{period}"] = slowk
                results[f"stoch_d_{period}"] = slowd

        # Stochastic RSI
        if "stochastic_rsi" in momentum_config:
            stochrsi_cfg = momentum_config["stochastic_rsi"]
            fastk, fastd = talib.STOCHRSI(
                close,
                timeperiod=stochrsi_cfg["period"],
                fastk_period=stochrsi_cfg["k_smooth"],
                fastd_period=stochrsi_cfg["d_smooth"],
                fastd_matype=0,
            )
            results["stochrsi_k"] = fastk
            results["stochrsi_d"] = fastd

        # CCI - Commodity Channel Index
        if "cci" in momentum_config:
            for period in momentum_config["cci"]["periods"]:
                results[f"cci_{period}"] = talib.CCI(
                    high, low, close, timeperiod=period
                )

        # Williams %R
        if "willr" in momentum_config:
            for period in momentum_config["willr"]["periods"]:
                results[f"willr_{period}"] = talib.WILLR(
                    high, low, close, timeperiod=period
                )

        # MFI - Money Flow Index
        if "mfi" in momentum_config:
            for period in momentum_config["mfi"]["periods"]:
                results[f"mfi_{period}"] = talib.MFI(
                    high, low, close, volume, timeperiod=period
                )

        # ROC - Rate of Change
        if "roc" in momentum_config:
            for period in momentum_config["roc"]["periods"]:
                results[f"roc_{period}"] = talib.ROC(close, timeperiod=period)

        # MOM - Momentum
        if "mom" in momentum_config:
            for period in momentum_config["mom"]["periods"]:
                results[f"mom_{period}"] = talib.MOM(close, timeperiod=period)

        # TRIX - Triple Exponential Average
        if "trix" in momentum_config:
            for period in momentum_config["trix"]["periods"]:
                results[f"trix_{period}"] = talib.TRIX(close, timeperiod=period)

        # Ultimate Oscillator
        if "ultosc" in momentum_config:
            ultosc_cfg = momentum_config["ultosc"]
            results["ultosc"] = talib.ULTOSC(
                high,
                low,
                close,
                timeperiod1=ultosc_cfg["period1"],
                timeperiod2=ultosc_cfg["period2"],
                timeperiod3=ultosc_cfg["period3"],
            )

        # CMO - Chande Momentum Oscillator
        if "cmo" in momentum_config:
            for period in momentum_config["cmo"]["periods"]:
                results[f"cmo_{period}"] = talib.CMO(close, timeperiod=period)

        # PPO - Percentage Price Oscillator
        if "ppo" in momentum_config:
            ppo_cfg = momentum_config["ppo"]
            results["ppo"] = talib.PPO(
                close,
                fastperiod=ppo_cfg["fast_period"],
                slowperiod=ppo_cfg["slow_period"],
                matype=0,
            )

        return results

    # ═══════════════════════════════════════════════════════════════════════════
    # VOLATILITY INDICATORS
    # ═══════════════════════════════════════════════════════════════════════════

    def _calculate_volatility_indicators(
        self, open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Calculate all volatility indicators."""
        results = {}
        volatility_config = self.config.get("volatility", {})

        # Bollinger Bands
        if "bbands" in volatility_config:
            bbands_cfg = volatility_config["bbands"]
            for period in bbands_cfg["periods"]:
                upper, middle, lower = talib.BBANDS(
                    close,
                    timeperiod=period,
                    nbdevup=bbands_cfg["std_dev"],
                    nbdevdn=bbands_cfg["std_dev"],
                    matype=0,
                )
                results[f"bb_upper_{period}"] = upper
                results[f"bb_middle_{period}"] = middle
                results[f"bb_lower_{period}"] = lower

                # %B = (price - lower) / (upper - lower)
                bb_range = upper - lower
                bb_range = np.where(bb_range == 0, np.nan, bb_range)
                results[f"bb_pctb_{period}"] = (close - lower) / bb_range

                # Bandwidth = (upper - lower) / middle
                middle_safe = np.where(middle == 0, np.nan, middle)
                results[f"bb_width_{period}"] = (upper - lower) / middle_safe

                # Cache for cross features
                if period == 20:
                    self._cache["bb_upper_20"] = upper
                    self._cache["bb_lower_20"] = lower

        # ATR - Average True Range
        if "atr" in volatility_config:
            for period in volatility_config["atr"]["periods"]:
                results[f"atr_{period}"] = talib.ATR(
                    high, low, close, timeperiod=period
                )
                if period == 14:
                    self._cache["atr_14"] = results[f"atr_{period}"]

        # NATR - Normalized Average True Range
        if "natr" in volatility_config:
            for period in volatility_config["natr"]["periods"]:
                results[f"natr_{period}"] = talib.NATR(
                    high, low, close, timeperiod=period
                )

        # True Range
        if "trange" in volatility_config and volatility_config["trange"]["enabled"]:
            results["trange"] = talib.TRANGE(high, low, close)

        # Keltner Channels
        if "keltner" in volatility_config:
            kc_cfg = volatility_config["keltner"]
            kc_middle = talib.EMA(close, timeperiod=kc_cfg["period"])
            kc_atr = talib.ATR(high, low, close, timeperiod=kc_cfg["atr_period"])
            results["kc_upper"] = kc_middle + (kc_cfg["multiplier"] * kc_atr)
            results["kc_middle"] = kc_middle
            results["kc_lower"] = kc_middle - (kc_cfg["multiplier"] * kc_atr)

        # Donchian Channels
        if "donchian" in volatility_config:
            dc_period = volatility_config["donchian"]["period"]
            results["dc_upper"] = talib.MAX(high, timeperiod=dc_period)
            results["dc_lower"] = talib.MIN(low, timeperiod=dc_period)
            results["dc_middle"] = (results["dc_upper"] + results["dc_lower"]) / 2

        # Standard Deviation
        if "stddev" in volatility_config:
            for period in volatility_config["stddev"]["periods"]:
                results[f"stddev_{period}"] = talib.STDDEV(
                    close, timeperiod=period, nbdev=1
                )

        # Variance
        if "variance" in volatility_config:
            for period in volatility_config["variance"]["periods"]:
                results[f"variance_{period}"] = talib.VAR(
                    close, timeperiod=period, nbdev=1
                )

        return results

    # ═══════════════════════════════════════════════════════════════════════════
    # VOLUME INDICATORS
    # ═══════════════════════════════════════════════════════════════════════════

    def _calculate_volume_indicators(
        self,
        open_: np.ndarray,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        volume: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Calculate all volume indicators."""
        results = {}
        volume_config = self.config.get("volume", {})

        # OBV - On-Balance Volume
        if "obv" in volume_config and volume_config["obv"]["enabled"]:
            results["obv"] = talib.OBV(close, volume)

        # Accumulation/Distribution Line
        if "ad" in volume_config and volume_config["ad"]["enabled"]:
            results["ad"] = talib.AD(high, low, close, volume)

        # AD - Accumulation/Distribution Oscillator
        if "adosc" in volume_config:
            adosc_cfg = volume_config["adosc"]
            results["adosc"] = talib.ADOSC(
                high,
                low,
                close,
                volume,
                fastperiod=adosc_cfg["fast_period"],
                slowperiod=adosc_cfg["slow_period"],
            )

        # Volume SMA
        if "volume_sma" in volume_config:
            for period in volume_config["volume_sma"]["periods"]:
                results[f"volume_sma_{period}"] = talib.SMA(volume, timeperiod=period)

        # Volume EMA
        if "volume_ema" in volume_config:
            for period in volume_config["volume_ema"]["periods"]:
                results[f"volume_ema_{period}"] = talib.EMA(volume, timeperiod=period)

        # Volume Ratio
        if "volume_ratio" in volume_config:
            period = volume_config["volume_ratio"]["period"]
            vol_sma = talib.SMA(volume, timeperiod=period)
            vol_sma_safe = np.where(vol_sma == 0, np.nan, vol_sma)
            results["volume_ratio"] = volume / vol_sma_safe

        # VWAP - Volume Weighted Average Price
        if "vwap" in volume_config and volume_config["vwap"]["enabled"]:
            typical_price = (high + low + close) / 3
            cumulative_tp_vol = np.cumsum(typical_price * volume)
            cumulative_vol = np.cumsum(volume)
            cumulative_vol_safe = np.where(cumulative_vol == 0, np.nan, cumulative_vol)
            results["vwap"] = cumulative_tp_vol / cumulative_vol_safe

        # CMF - Chaikin Money Flow
        if "cmf" in volume_config:
            for period in volume_config["cmf"]["periods"]:
                # Money Flow Multiplier
                hl_range = high - low
                hl_range_safe = np.where(hl_range == 0, np.nan, hl_range)
                mf_multiplier = ((close - low) - (high - close)) / hl_range_safe
                mf_volume = mf_multiplier * volume

                # CMF = sum(mf_volume, n) / sum(volume, n)
                mf_sum = talib.SMA(mf_volume, timeperiod=period) * period
                vol_sum = talib.SMA(volume, timeperiod=period) * period
                vol_sum_safe = np.where(vol_sum == 0, np.nan, vol_sum)
                results[f"cmf_{period}"] = mf_sum / vol_sum_safe

        # EMV - Ease of Movement
        if "emv" in volume_config:
            period = volume_config["emv"]["period"]
            distance = ((high + low) / 2) - ((np.roll(high, 1) + np.roll(low, 1)) / 2)
            box_ratio = (volume / 1e8) / (high - low + 1e-10)
            emv = distance / box_ratio
            emv[0] = np.nan
            results["emv"] = talib.SMA(emv, timeperiod=period)

        # Force Index
        if "force" in volume_config:
            for period in volume_config["force"]["periods"]:
                force = (close - np.roll(close, 1)) * volume
                force[0] = np.nan
                results[f"force_{period}"] = talib.EMA(force, timeperiod=period)

        return results

    # ═══════════════════════════════════════════════════════════════════════════
    # PRICE FEATURES
    # ═══════════════════════════════════════════════════════════════════════════

    def _calculate_price_features(
        self, open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Calculate price-based features."""
        results = {}
        price_config = self.config.get("price", {})

        # Returns (percentage change)
        if "returns" in price_config:
            for period in price_config["returns"]["periods"]:
                prev_close = np.roll(close, period)
                prev_close_safe = np.where(prev_close == 0, np.nan, prev_close)
                returns = (close - prev_close) / prev_close_safe * 100
                returns[:period] = np.nan
                results[f"return_{period}"] = returns

        # Log Returns
        if "log_returns" in price_config:
            for period in price_config["log_returns"]["periods"]:
                prev_close = np.roll(close, period)
                # Avoid log(0) and log(negative)
                safe_close = np.where(close <= 0, np.nan, close)
                safe_prev = np.where(prev_close <= 0, np.nan, prev_close)
                log_returns = np.log(safe_close / safe_prev)
                log_returns[:period] = np.nan
                results[f"log_return_{period}"] = log_returns

        # High-Low Range
        if "hl_range" in price_config and price_config["hl_range"]["enabled"]:
            results["hl_range"] = high - low
            close_safe = np.where(close == 0, np.nan, close)
            results["hl_range_pct"] = (high - low) / close_safe * 100

        # Close Position in Range
        if (
            "close_position" in price_config
            and price_config["close_position"]["enabled"]
        ):
            hl_range = high - low
            hl_range_safe = np.where(hl_range == 0, np.nan, hl_range)
            results["close_position"] = (close - low) / hl_range_safe

        # Gap
        if "gap" in price_config and price_config["gap"]["enabled"]:
            prev_close = np.roll(close, 1)
            prev_close_safe = np.where(prev_close == 0, np.nan, prev_close)
            results["gap"] = (open_ - prev_close) / prev_close_safe * 100
            results["gap"][0] = np.nan

        # Candle Components
        if "candle" in price_config and price_config["candle"]["enabled"]:
            body = close - open_
            results["body_size"] = np.abs(body)
            results["body_direction"] = np.sign(body)

            # Upper shadow
            results["upper_shadow"] = high - np.maximum(open_, close)

            # Lower shadow
            results["lower_shadow"] = np.minimum(open_, close) - low

            # Body ratio (body / total range)
            total_range = high - low
            total_range_safe = np.where(total_range == 0, np.nan, total_range)
            results["body_ratio"] = np.abs(body) / total_range_safe

        return results

    # ═══════════════════════════════════════════════════════════════════════════
    # CROSS/RATIO FEATURES
    # ═══════════════════════════════════════════════════════════════════════════

    def _calculate_cross_features(self, close: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate cross and ratio features using cached indicators."""
        results = {}
        cross_config = self.config.get("cross", {})

        # Price / SMA Ratio
        if "price_sma_ratio" in cross_config:
            for period in cross_config["price_sma_ratio"]["periods"]:
                sma_key = f"sma_{period}"
                if sma_key in self._cache:
                    sma = self._cache[sma_key]
                    sma_safe = np.where(sma == 0, np.nan, sma)
                    results[f"price_sma_ratio_{period}"] = close / sma_safe

        # Price / EMA Ratio
        if "price_ema_ratio" in cross_config:
            for period in cross_config["price_ema_ratio"]["periods"]:
                ema_key = f"ema_{period}"
                if ema_key in self._cache:
                    ema = self._cache[ema_key]
                    ema_safe = np.where(ema == 0, np.nan, ema)
                    results[f"price_ema_ratio_{period}"] = close / ema_safe

        # SMA Crossover Distance
        if "sma_cross" in cross_config:
            for fast, slow in cross_config["sma_cross"]["pairs"]:
                fast_key = f"sma_{fast}"
                slow_key = f"sma_{slow}"
                if fast_key in self._cache and slow_key in self._cache:
                    fast_sma = self._cache[fast_key]
                    slow_sma = self._cache[slow_key]
                    slow_safe = np.where(slow_sma == 0, np.nan, slow_sma)
                    results[f"sma_cross_{fast}_{slow}"] = (
                        fast_sma - slow_sma
                    ) / slow_safe

        # EMA Crossover Distance
        if "ema_cross" in cross_config:
            for fast, slow in cross_config["ema_cross"]["pairs"]:
                fast_key = f"ema_{fast}"
                slow_key = f"ema_{slow}"
                if fast_key in self._cache and slow_key in self._cache:
                    fast_ema = self._cache[fast_key]
                    slow_ema = self._cache[slow_key]
                    slow_safe = np.where(slow_ema == 0, np.nan, slow_ema)
                    results[f"ema_cross_{fast}_{slow}"] = (
                        fast_ema - slow_ema
                    ) / slow_safe

        # RSI Distance from 50
        if "rsi_distance" in cross_config:
            if "rsi_14" in self._cache:
                results["rsi_distance_50"] = self._cache["rsi_14"] - 50

        # BB Distance
        if "bb_distance" in cross_config:
            if "bb_upper_20" in self._cache and "bb_lower_20" in self._cache:
                bb_upper = self._cache["bb_upper_20"]
                bb_lower = self._cache["bb_lower_20"]
                bb_range = bb_upper - bb_lower
                bb_range_safe = np.where(bb_range == 0, np.nan, bb_range)
                results["bb_dist_upper"] = (bb_upper - close) / bb_range_safe
                results["bb_dist_lower"] = (close - bb_lower) / bb_range_safe

        return results

    # ═══════════════════════════════════════════════════════════════════════════
    # PATTERN FEATURES
    # ═══════════════════════════════════════════════════════════════════════════

    def _calculate_pattern_features(
        self, open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """Calculate pattern-based features."""
        results = {}
        pattern_config = self.config.get("pattern", {})

        # MA Trend Direction
        if "ma_trend" in pattern_config:
            lookback = pattern_config["ma_trend"]["lookback"]
            for period in pattern_config["ma_trend"]["periods"]:
                sma_key = f"sma_{period}"
                if sma_key in self._cache:
                    sma = self._cache[sma_key]
                    sma_prev = np.roll(sma, lookback)
                    trend = np.zeros_like(sma)
                    trend[sma > sma_prev] = 1
                    trend[sma < sma_prev] = -1
                    trend[:lookback] = np.nan
                    results[f"ma_trend_{period}"] = trend

        # Higher Highs / Lower Lows Count
        if "hl_count" in pattern_config:
            lookback = pattern_config["hl_count"]["lookback"]
            n = len(high)
            higher_highs = np.zeros(n)
            lower_lows = np.zeros(n)

            for i in range(lookback, n):
                hh_count = 0
                ll_count = 0
                for j in range(1, lookback):
                    if high[i - j + 1] > high[i - j]:
                        hh_count += 1
                    if low[i - j + 1] < low[i - j]:
                        ll_count += 1
                higher_highs[i] = hh_count
                lower_lows[i] = ll_count

            higher_highs[:lookback] = np.nan
            lower_lows[:lookback] = np.nan
            results["higher_highs"] = higher_highs
            results["lower_lows"] = lower_lows

        # Consecutive Up/Down Candles
        if "consecutive" in pattern_config and pattern_config["consecutive"]["enabled"]:
            n = len(close)
            consec_up = np.zeros(n)
            consec_down = np.zeros(n)

            for i in range(1, n):
                if close[i] > close[i - 1]:
                    consec_up[i] = consec_up[i - 1] + 1
                    consec_down[i] = 0
                elif close[i] < close[i - 1]:
                    consec_down[i] = consec_down[i - 1] + 1
                    consec_up[i] = 0
                else:
                    consec_up[i] = 0
                    consec_down[i] = 0

            results["consecutive_up"] = consec_up
            results["consecutive_down"] = consec_down

        # Volatility Regime
        if "volatility_regime" in pattern_config:
            vr_cfg = pattern_config["volatility_regime"]
            if "atr_14" in self._cache:
                atr = self._cache["atr_14"]
                lookback = vr_cfg["lookback"]
                n = len(atr)
                regime = np.zeros(n)

                for i in range(lookback, n):
                    window = atr[i - lookback : i]
                    valid_window = window[~np.isnan(window)]
                    if len(valid_window) > 0:
                        percentile = np.sum(valid_window < atr[i]) / len(valid_window)
                        regime[i] = percentile
                    else:
                        regime[i] = np.nan

                regime[:lookback] = np.nan
                results["volatility_regime"] = regime

        return results

    # ═══════════════════════════════════════════════════════════════════════════
    # MAIN CALCULATE METHOD
    # ═══════════════════════════════════════════════════════════════════════════

    def calculate(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Calculate all technical indicators for a DataFrame.

        Args:
            df: DataFrame with columns: open_time, open, high, low, close, volume

        Returns:
            DataFrame with all original columns plus calculated features
        """
        self._clear_cache()

        # Extract numpy arrays
        open_ = self._to_numpy(df["open"])
        high = self._to_numpy(df["high"])
        low = self._to_numpy(df["low"])
        close = self._to_numpy(df["close"])
        volume = self._to_numpy(df["volume"])

        # Calculate all indicator categories
        all_features = {}

        logger.debug("Calculating trend indicators...")
        all_features.update(self._calculate_trend_indicators(open_, high, low, close))

        logger.debug("Calculating momentum indicators...")
        all_features.update(
            self._calculate_momentum_indicators(open_, high, low, close, volume)
        )

        logger.debug("Calculating volatility indicators...")
        all_features.update(
            self._calculate_volatility_indicators(open_, high, low, close)
        )

        logger.debug("Calculating volume indicators...")
        all_features.update(
            self._calculate_volume_indicators(open_, high, low, close, volume)
        )

        logger.debug("Calculating price features...")
        all_features.update(self._calculate_price_features(open_, high, low, close))

        logger.debug("Calculating cross features...")
        all_features.update(self._calculate_cross_features(close))

        logger.debug("Calculating pattern indicators...")
        all_features.update(self._calculate_pattern_features(open_, high, low, close))

        # Add features to DataFrame
        for name, values in all_features.items():
            df = df.with_columns(pl.Series(name=name, values=values))

        # logger.info(f"Calculated {len(all_features)} features")

        return df

    def calculate_for_date(
        self, target_date: str, symbols: Optional[List[str]] = None
    ) -> pl.DataFrame:
        """
        Calculate features for all symbols on a specific date.

        Args:
            target_date: Date string (YYYY-MM-DD)
            symbols: Optional list of symbols to process

        Returns:
            DataFrame with features for all symbols
        """
        all_data = []

        for symbol_dir in self.data_directory.iterdir():
            if not symbol_dir.is_dir():
                continue
            if symbol_dir.name.startswith("_"):
                continue

            symbol = symbol_dir.name

            if symbols and symbol not in symbols:
                continue

            date_file = symbol_dir / f"{target_date}.parquet"
            if not date_file.exists():
                continue

            # Load and calculate
            df = pl.read_parquet(date_file)
            df = df.with_columns(pl.lit(symbol).alias("symbol"))
            df = self.calculate(df)
            all_data.append(df)

        if not all_data:
            logger.warning(f"No data found for date {target_date}")
            return pl.DataFrame()

        return pl.concat(all_data)

    def get_feature_names(self) -> List[str]:
        """
        Get list of all feature names that will be calculated.

        Returns:
            List of feature column names
        """
        # Create dummy data to get feature names
        n = 300  # Need enough data for longest indicator period
        dummy_df = pl.DataFrame(
            {
                "open_time": range(n),
                "open": np.random.randn(n) + 100,
                "high": np.random.randn(n) + 101,
                "low": np.random.randn(n) + 99,
                "close": np.random.randn(n) + 100,
                "volume": np.abs(np.random.randn(n)) * 1000,
            }
        )

        df_with_features = self.calculate(dummy_df)

        # Return only the new columns (not original OHLCV)
        original_cols = {"open_time", "open", "high", "low", "close", "volume"}
        return [col for col in df_with_features.columns if col not in original_cols]


if __name__ == "__main__":
    # Test calculation
    calculator = TechnicalIndicators()

    # Get feature names
    feature_names = calculator.get_feature_names()
    print(f"Total features: {len(feature_names)}")
    print(f"First 20 features: {feature_names[:20]}")

    # Test with real data
    from pathlib import Path

    data_dir = Path("data/volatility/5m")

    # Find a file to test
    for symbol_dir in data_dir.iterdir():
        if symbol_dir.is_dir() and not symbol_dir.name.startswith("_"):
            files = list(symbol_dir.glob("*.parquet"))
            if files:
                test_file = files[0]
                print(f"\nTesting with: {test_file}")

                df = pl.read_parquet(test_file)
                print(f"Original shape: {df.shape}")

                df_features = calculator.calculate(df)
                print(f"With features shape: {df_features.shape}")
                print(
                    f"Sample features:\n{df_features.select(feature_names[:10]).head(5)}"
                )
                break
