"""
Fitur Teknikal Analisis

Menghitung indikator teknikal dan fitur untuk trading menggunakan TA-Lib
Semua periode dan parameter dibaca dari konfigurasi.yaml untuk fleksibilitas.
"""

import polars as pl
import numpy as np
import talib
from pathlib import Path
from typing import List, Dict, Optional
import logging
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FiturTA:
    """Menghitung indikator teknikal dan fitur menggunakan TA-Lib."""

    def __init__(
        self,
        direktori_input: str = "data/proses",
        direktori_output: str = "data/fitur",
        konfigurasi_pengguna: Optional[Dict] = None,
    ):
        """
        Inisialisasi insinyur fitur.

        Args:
            direktori_input: Direktori yang berisi data timeframe terproses
            direktori_output: Direktori untuk menyimpan data fitur
            konfigurasi_pengguna: Override konfigurasi dari konfigurasi/konfigurasi.yaml
        """
        self.direktori_input = Path(direktori_input)
        self.direktori_output = Path(direktori_output)
        self.direktori_output.mkdir(parents=True, exist_ok=True)
        self.konfigurasi = self._ambil_konfigurasi_default()

        if konfigurasi_pengguna:
            self._gabungkan_konfigurasi(konfigurasi_pengguna)

        logger.info(
            f"InsinyurFitur diinisialisasi dengan {len(self.konfigurasi)} grup konfigurasi"
        )

    def _ambil_konfigurasi_default(self) -> Dict:
        """Konfigurasi default jika tidak ada konfigurasi yang diberikan."""
        return {
            # ===========================================================================
            # MOVING AVERAGES
            # ===========================================================================
            "sma_periods": [7, 14, 21, 50, 200],
            "ema_periods": [9, 12, 21, 26, 50],
            "wma_periods": [14, 21],
            "dema_periods": [14, 21],
            "tema_periods": [14, 21],
            # ===========================================================================
            # MOMENTUM INDICATORS
            # ===========================================================================
            "rsi_periods": [14, 21],
            "macd": {"cepat": 12, "lambat": 26, "sinyal": 9},
            "stochastic": {"k_period": 14, "d_period": 3, "perlambatan": 3},
            "stochrsi": {"timeperiod": 14, "fastk_period": 5, "fastd_period": 3},
            "ultosc": {"timeperiod1": 7, "timeperiod2": 14, "timeperiod3": 20},
            "cci_periods": [14, 20],
            "williams_r_period": 14,
            "mfi_period": 14,
            "roc_periods": [10, 21],
            "momentum_periods": [10, 21],
            # ===========================================================================
            # VOLATILITY INDICATORS
            # ===========================================================================
            "bollinger": {"period": 20, "std_dev": 2.0},
            "atr_periods": [14, 21],
            "natr_period": 14,
            "keltner": {"ema_period": 20, "atr_period": 10, "multiplier": 2.0},
            "donchian_period": 20,
            # ===========================================================================
            # VOLUME INDICATORS
            # ===========================================================================
            "volume_ma_periods": [10, 20, 50],
            "volume_rasio_ma_period": 20,
            "hitung_obv": True,
            "obv_roc_period": 10,
            "hitung_ad": True,
            "hitung_adosc": True,
            "adosc": {"cepat": 3, "lambat": 10},
            # ===========================================================================
            # TREND INDICATORS
            # ===========================================================================
            "adx_period": 14,
            "aaron_period": 14,
            "sar": {"percepatan": 0.02, "maksimum": 0.2},
            "hitung_ichimoku": True,
            "ichimoku": {"tenkan": 9, "kijun": 26, "senkou_b": 52},
            # ===========================================================================
            # PRICE FEATURES
            # ===========================================================================
            "price_change_periods": [1, 4, 12, 24, 48, 96],
            "hitung_candle_patterns": True,
            "hitung_price_position": True,
            # ===========================================================================
            # FITUR TURUNAN
            # ===========================================================================
            "hitung_ma_crossovers": True,
            "ma_crossovers": {
                "golden_death": {"cepat": 50, "lambat": 200},
                "ema_cross": {"cepat": 9, "lambat": 21},
            },
            "hitung_ma_slope": True,
            "ma_slope_base": 21,
            "slope_periods": [5, 10, 20],
            "hitung_price_vs_ma": True,
            "price_vs_ma_periods": [7, 21, 50, 200],
            # RSI zones
            "rsi_zone_period": 14,
            "rsi_overbought": 70,
            "rsi_oversold": 30,
            # Volatility regime
            "volatility_atr_period": 14,  # ATR period untuk volatility regime
            "volatility_ma_period": 50,  # MA period untuk ATR
            "volatility_high_mult": 1.5,  # Multiplier untuk high volatility
            "volatility_low_mult": 0.5,  # Multiplier untuk low volatility
        }

    def _gabungkan_konfigurasi(self, konfigurasi_pengguna: Dict) -> None:
        """
        Deep merge konfigurasi pengguna ke dalam konfigurasi default.

        Args:
            konfigurasi_pengguna: Konfigurasi dari konfigurasi/konfigurasi.yaml
        """
        for kunci, nilai in konfigurasi_pengguna.items():
            if kunci in self.konfigurasi:
                # Jika nilai adalah dict, merge secara rekursif
                if isinstance(nilai, dict) and isinstance(
                    self.konfigurasi[kunci], dict
                ):
                    self.konfigurasi[kunci].update(nilai)
                else:
                    self.konfigurasi[kunci] = nilai
            else:
                # Tambah kunci baru dari konfigurasi_pengguna
                self.konfigurasi[kunci] = nilai
                logger.debug(f"Menambah konfigurasi baru: {kunci}")

    # ===================================================================================================
    # MOVING AVERAGES
    # ===================================================================================================

    def _hitung_moving_average(
        self, df: pl.DataFrame, tutup: np.ndarray
    ) -> pl.DataFrame:
        """Menghitung berbagai moving average."""

        hasil = []

        # Simple Moving Average (SMA)
        for periode in self.konfigurasi["sma_periods"]:
            hasil.append(
                pl.lit(talib.SMA(tutup, timeperiod=periode)).alias(f"sma_{periode}")
            )

        # Exponential Moving Average (EMA)
        for periode in self.konfigurasi["ema_periods"]:
            hasil.append(
                pl.lit(talib.EMA(tutup, timeperiod=periode)).alias(f"ema_{periode}")
            )

        # Weighted Moving Average (WMA)
        for periode in self.konfigurasi["wma_periods"]:
            hasil.append(
                pl.lit(talib.WMA(tutup, timeperiod=periode)).alias(f"wma_{periode}")
            )

        # Double Exponential Moving Average (DEMA)
        for periode in self.konfigurasi["dema_periods"]:
            hasil.append(
                pl.lit(talib.DEMA(tutup, timeperiod=periode)).alias(f"dema_{periode}")
            )

        # Triple Exponential Moving Average (TEMA)
        for periode in self.konfigurasi["tema_periods"]:
            hasil.append(
                pl.lit(talib.TEMA(tutup, timeperiod=periode)).alias(f"tema_{periode}")
            )

        df = df.with_columns(hasil)
        return df

    # ===================================================================================================
    # MOMENTUM INDICATORS
    # ===================================================================================================

    def _hitung_indikator_momentum(
        self,
        df: pl.DataFrame,
        tinggi: np.ndarray,
        rendah: np.ndarray,
        tutup: np.ndarray,
        volume: np.ndarray,
    ) -> pl.DataFrame:
        """Menghitung indikator momentum."""

        hasil = []

        # RSI (Relative Strength Index)
        for periode in self.konfigurasi["rsi_periods"]:
            hasil.append(
                pl.lit(talib.RSI(tutup, timeperiod=periode)).alias(f"rsi_{periode}")
            )

        # MACD (Moving Average Convergence Divergence)
        konfigurasi_macd = self.konfigurasi["macd"]
        macd, sinyal_macd, macd_hist = talib.MACD(
            tutup,
            fastperiod=konfigurasi_macd["cepat"],
            slowperiod=konfigurasi_macd["lambat"],
            signalperiod=konfigurasi_macd["sinyal"],
        )
        hasil.append(pl.lit(macd).alias("macd"))
        hasil.append(pl.lit(sinyal_macd).alias("sinyal_macd"))
        hasil.append(pl.lit(macd_hist).alias("macd_hist"))

        # Stochastic
        konfigurasi_stochastic = self.konfigurasi["stochastic"]
        slowk, slowd = talib.STOCH(
            tinggi,
            rendah,
            tutup,
            fastk_period=konfigurasi_stochastic["k_period"],
            slowk_period=konfigurasi_stochastic["perlambatan"],
            slowk_matype=0,
            slowd_period=konfigurasi_stochastic["d_period"],
            slowd_matype=0,
        )
        hasil.append(pl.lit(slowk).alias("stoch_k"))
        hasil.append(pl.lit(slowd).alias("stoch_d"))

        # Stochastic RSI
        konfigurasi_stochrsi = self.konfigurasi["stochrsi"]
        fastk, fastd = talib.STOCHRSI(
            tutup,
            timeperiod=konfigurasi_stochrsi["timeperiod"],
            fastk_period=konfigurasi_stochrsi["fastk_period"],
            fastd_period=konfigurasi_stochrsi["fastd_period"],
            fastd_matype=0,
        )
        hasil.append(pl.lit(fastk).alias("stochrsi_k"))
        hasil.append(pl.lit(fastd).alias("stochrsi_d"))

        # Commodity Channel Index (CCI)
        for periode in self.konfigurasi["cci_periods"]:
            hasil.append(
                pl.lit(talib.CCI(tinggi, rendah, tutup, timeperiod=periode)).alias(
                    f"cci_{periode}"
                )
            )

        # Williams %R
        hasil.append(
            pl.lit(
                talib.WILLR(
                    tinggi,
                    rendah,
                    tutup,
                    timeperiod=self.konfigurasi["williams_r_period"],
                )
            ).alias("williams_r")
        )

        # Money Flow Index (MFI)
        hasil.append(
            pl.lit(
                talib.MFI(
                    tinggi,
                    rendah,
                    tutup,
                    volume,
                    timeperiod=self.konfigurasi["mfi_period"],
                )
            ).alias("mfi")
        )

        # Rate of Change (ROC)
        for periode in self.konfigurasi["roc_periods"]:
            hasil.append(
                pl.lit(talib.ROC(tutup, timeperiod=periode)).alias(f"roc_{periode}")
            )

        # Momentum
        for periode in self.konfigurasi["momentum_periods"]:
            hasil.append(
                pl.lit(talib.MOM(tutup, timeperiod=periode)).alias(f"mom_{periode}")
            )

        # Ultimate Oscillator
        konfigurasi_ultosc = self.konfigurasi["ultosc"]
        hasil.append(
            pl.lit(
                talib.ULTOSC(
                    tinggi,
                    rendah,
                    tutup,
                    timeperiod1=konfigurasi_ultosc["timeperiod1"],
                    timeperiod2=konfigurasi_ultosc["timeperiod2"],
                    timeperiod3=konfigurasi_ultosc["timeperiod3"],
                )
            ).alias("ultosc")
        )

        df = df.with_columns(hasil)
        return df

    # ===================================================================================================
    # VOLATILITY INDICATORS
    # ===================================================================================================

    def _hitung_indikator_volatilitas(
        self,
        df: pl.DataFrame,
        tinggi: np.ndarray,
        rendah: np.ndarray,
        tutup: np.ndarray,
    ) -> pl.DataFrame:
        """Menghitung indikator volatilitas."""

        hasil = []

        # Bollinger Bands
        konfigurasi_bb = self.konfigurasi["bollinger"]
        upper, middle, lower = talib.BBANDS(
            tutup,
            timeperiod=konfigurasi_bb["period"],
            nbdevup=konfigurasi_bb["std_dev"],
            nbdevdn=konfigurasi_bb["std_dev"],
            matype=0,
        )
        hasil.append(pl.lit(upper).alias("bb_upper"))
        hasil.append(pl.lit(middle).alias("bb_middle"))
        hasil.append(pl.lit(lower).alias("bb_lower"))
        hasil.append(pl.lit((upper - lower) / middle).alias("bb_width"))

        # Average True Range (ATR)
        for periode in self.konfigurasi["atr_periods"]:
            hasil.append(
                pl.lit(talib.ATR(tinggi, rendah, tutup, timeperiod=periode)).alias(
                    f"atr_{periode}"
                )
            )

        # Normalized ATR (NATR)
        hasil.append(
            pl.lit(
                talib.NATR(
                    tinggi, rendah, tutup, timeperiod=self.konfigurasi["natr_period"]
                )
            ).alias("natr")
        )

        # True Range
        hasil.append(pl.lit(talib.TRANGE(tinggi, rendah, tutup)).alias("trange"))

        # Keltner Channels
        konfigurasi_keltner = self.konfigurasi["keltner"]
        ema_kelt = talib.EMA(tutup, timeperiod=konfigurasi_keltner["ema_period"])
        atr_kelt = talib.ATR(
            tinggi, rendah, tutup, timeperiod=konfigurasi_keltner["atr_period"]
        )
        hasil.append(
            pl.lit(ema_kelt + (konfigurasi_keltner["multiplier"] * atr_kelt)).alias(
                "keltner_upper"
            )
        )
        hasil.append(
            pl.lit(ema_kelt - (konfigurasi_keltner["multiplier"] * atr_kelt)).alias(
                "keltner_lower"
            )
        )
        hasil.append(pl.lit(ema_kelt).alias("keltner_middle"))

        # Donchian Channels
        periode_donch = self.konfigurasi["donchian_period"]
        hasil.append(
            pl.col("tinggi")
            .rolling_max(window_size=periode_donch)
            .alias("donchian_upper")
        )
        hasil.append(
            pl.col("rendah")
            .rolling_min(window_size=periode_donch)
            .alias("donchian_lower")
        )

        df = df.with_columns(hasil)

        # bb_persen needs bb_lower from the newly added columns
        df = df.with_columns(
            [
                (
                    (pl.col("tutup") - pl.col("bb_lower"))
                    / (pl.col("bb_upper") - pl.col("bb_lower"))
                ).alias("bb_persen"),
                ((pl.col("donchian_upper") + pl.col("donchian_lower")) / 2).alias(
                    "donchian_middle"
                ),
            ]
        )

        return df

    # ===================================================================================================
    # VOLUME INDICATORS
    # ===================================================================================================

    def _hitung_indikator_volume(
        self,
        df: pl.DataFrame,
        tinggi: np.ndarray,
        rendah: np.ndarray,
        tutup: np.ndarray,
        volume: np.ndarray,
    ) -> pl.DataFrame:
        """Menghitung indikator volume."""

        hasil = []

        # Volume Moving Averages
        for periode in self.konfigurasi["volume_ma_periods"]:
            hasil.append(
                pl.lit(talib.SMA(volume, timeperiod=periode)).alias(
                    f"volume_ma_{periode}"
                )
            )

        df = df.with_columns(hasil)

        # Rasio volume terhadap MA
        vol_rasio_period = self.konfigurasi["volume_rasio_ma_period"]
        col_vol_ma = f"volume_ma_{vol_rasio_period}"
        if col_vol_ma in df.columns:
            df = df.with_columns(
                (pl.col("volume") / pl.col(col_vol_ma)).alias("volume_rasio")
            )

        # On-Balance Volume (OBV)
        if self.konfigurasi["hitung_obv"]:
            obv = talib.OBV(tutup, volume)
            df = df.with_columns(pl.lit(obv).alias("obv"))
            # OBV rate of change
            obv_roc_period = self.konfigurasi["obv_roc_period"]
            df = df.with_columns(
                pl.lit(talib.ROC(obv, timeperiod=obv_roc_period)).alias("obv_roc")
            )

        # Accumulation Distribution Line (A/D Line)
        if self.konfigurasi["hitung_ad"]:
            df = df.with_columns(
                pl.lit(talib.AD(tinggi, rendah, tutup, volume)).alias("ad")
            )

        # Accumulation Distribution Oscillator (A/D Oscillator)
        if self.konfigurasi["hitung_adosc"]:
            konfigurasi_adosc = self.konfigurasi["adosc"]
            df = df.with_columns(
                pl.lit(
                    talib.ADOSC(
                        tinggi,
                        rendah,
                        tutup,
                        volume,
                        fastperiod=konfigurasi_adosc["cepat"],
                        slowperiod=konfigurasi_adosc["lambat"],
                    )
                ).alias("adosc")
            )

        return df

    # ===================================================================================================
    # TREND INDICATORS
    # ===================================================================================================

    def _hitung_ichimoku(self, df: pl.DataFrame) -> pl.DataFrame:
        """Menghitung komponen Ichimoku Cloud."""

        konfigurasi_ichimoku = self.konfigurasi["ichimoku"]

        df = df.with_columns(
            [
                # Tenkan-sen (Conversion Line)
                (
                    (
                        pl.col("tinggi").rolling_max(
                            window_size=konfigurasi_ichimoku["tenkan"]
                        )
                        + pl.col("rendah").rolling_min(
                            window_size=konfigurasi_ichimoku["tenkan"]
                        )
                    )
                    / 2
                ).alias("tenkan_sen"),
                # Kijun-sen (Base Line)
                (
                    (
                        pl.col("tinggi").rolling_max(
                            window_size=konfigurasi_ichimoku["kijun"]
                        )
                        + pl.col("rendah").rolling_min(
                            window_size=konfigurasi_ichimoku["kijun"]
                        )
                    )
                    / 2
                ).alias("kijun_sen"),
            ]
        )

        df = df.with_columns(
            [
                # Senkou Span A (Leading Span A)
                ((pl.col("tenkan_sen") + pl.col("kijun_sen")) / 2)
                .shift(konfigurasi_ichimoku["kijun"])
                .alias("senkou_span_a"),
                # Senkou Span B (Leading Span B)
                (
                    (
                        pl.col("tinggi").rolling_max(
                            window_size=konfigurasi_ichimoku["senkou_b"]
                        )
                        + pl.col("rendah").rolling_min(
                            window_size=konfigurasi_ichimoku["senkou_b"]
                        )
                    )
                    / 2
                )
                .shift(konfigurasi_ichimoku["kijun"])
                .alias("senkou_span_b"),
                # Chikou Span (Lagging Span)
                pl.col("tutup")
                .shift(-konfigurasi_ichimoku["kijun"])
                .alias("chikou_span"),
            ]
        )

        df = df.with_columns(
            [
                # Cloud signals
                pl.when(
                    (pl.col("tutup") > pl.col("senkou_span_a"))
                    & (pl.col("tutup") > pl.col("senkou_span_b"))
                )
                .then(1)
                .otherwise(0)
                .alias("ichi_above_cloud"),
                pl.when(
                    (pl.col("tutup") < pl.col("senkou_span_a"))
                    & (pl.col("tutup") < pl.col("senkou_span_b"))
                )
                .then(1)
                .otherwise(0)
                .alias("ichi_below_cloud"),
            ]
        )

        return df

    def _hitung_indikator_trend(
        self,
        df: pl.DataFrame,
        tinggi: np.ndarray,
        rendah: np.ndarray,
        tutup: np.ndarray,
    ) -> pl.DataFrame:
        """Menghitung indikator trend."""

        hasil = []

        # Average Directional Index (ADX) & Directional Index (DI)
        adx_period = self.konfigurasi["adx_period"]
        hasil.append(
            pl.lit(talib.ADX(tinggi, rendah, tutup, timeperiod=adx_period)).alias("adx")
        )
        hasil.append(
            pl.lit(talib.PLUS_DI(tinggi, rendah, tutup, timeperiod=adx_period)).alias(
                "plus_di"
            )
        )
        hasil.append(
            pl.lit(talib.MINUS_DI(tinggi, rendah, tutup, timeperiod=adx_period)).alias(
                "minus_di"
            )
        )
        hasil.append(
            pl.lit(talib.DX(tinggi, rendah, tutup, timeperiod=adx_period)).alias("dx")
        )

        # Aroon
        aroon_period = self.konfigurasi["aroon_period"]
        aroon_up, aroon_down = talib.AROON(tinggi, rendah, timeperiod=aroon_period)
        hasil.append(pl.lit(aroon_up).alias("aroon_up"))
        hasil.append(pl.lit(aroon_down).alias("aroon_down"))
        hasil.append(
            pl.lit(talib.AROONOSC(tinggi, rendah, timeperiod=aroon_period)).alias(
                "aroon_osc"
            )
        )

        # Parabolic SAR
        konfigurasi_sar = self.konfigurasi["sar"]
        sar = talib.SAR(
            tinggi,
            rendah,
            acceleration=konfigurasi_sar["percepatan"],
            maximum=konfigurasi_sar["maksimum"],
        )
        hasil.append(pl.lit(sar).alias("sar"))

        df = df.with_columns(hasil)
        df = df.with_columns(
            pl.when(pl.col("tutup") > pl.col("sar"))
            .then(1)
            .otherwise(-1)
            .alias("sar_sinyal")
        )

        # Ichimoku Cloud
        if self.konfigurasi["hitung_ichimoku"]:
            df = self._hitung_ichimoku(df)

        return df

    # ===================================================================================================
    # PRICE FEATURES
    # ===================================================================================================

    def _hitung_fitur_harga(self, df: pl.DataFrame) -> pl.DataFrame:
        """Menghitung fitur berbasis harga."""

        hasil = []

        # Price changes (return)
        for periode in self.konfigurasi["price_change_periods"]:
            hasil.append(
                (
                    (pl.col("tutup") - pl.col("tutup").shift(periode))
                    / pl.col("tutup").shift(periode)
                ).alias(f"return_{periode}")
            )
            hasil.append(
                (pl.col("tutup") / pl.col("tutup").shift(periode))
                .log()
                .alias(f"log_return_{periode}")
            )

        # High-Low range
        hasil.append(
            ((pl.col("tinggi") - pl.col("rendah")) / pl.col("tutup")).alias("hl_range")
        )

        # Open-Close range
        hasil.append(
            ((pl.col("tutup") - pl.col("buka")) / pl.col("buka")).alias("oc_range")
        )

        # Upper/Lower shadows
        hasil.append(
            (
                (pl.col("tinggi") - pl.max_horizontal("buka", "tutup"))
                / pl.col("tutup")
            ).alias("upper_shadow")
        )
        hasil.append(
            (
                (pl.min_horizontal("buka", "tutup") - pl.col("rendah"))
                / pl.col("tutup")
            ).alias("lower_shadow")
        )

        # Body size (absolute)
        hasil.append(
            ((pl.col("tutup") - pl.col("buka")).abs() / pl.col("tutup")).alias(
                "body_size"
            )
        )

        # Bullish/Bearish candle
        hasil.append(
            pl.when(pl.col("tutup") > pl.col("buka"))
            .then(1)
            .otherwise(0)
            .alias("is_bullish")
        )

        # Price position dalam daily range
        if self.konfigurasi["hitung_price_position"]:
            hasil.append(
                (
                    (pl.col("tutup") - pl.col("rendah"))
                    / (pl.col("tinggi") - pl.col("rendah") + 1e-10)
                ).alias("price_position")
            )

        # Gap (jarak dari tutup sebelumnya ke buka sekarang)
        hasil.append(
            (
                (pl.col("buka") - pl.col("tutup").shift(1)) / pl.col("tutup").shift(1)
            ).alias("gap")
        )

        df = df.with_columns(hasil)
        return df

    def hitung_fitur(self, simbol: str, timeframe: str = "15m") -> pl.DataFrame:
        """
        Menghitung semua fitur untuk sebuah simbol

        Args:
            simbol: Pasangan trading
            timeframe: Timeframe yang akan digunakan

        Returns:
            DataFrame dengan semua fitur atau None jika gagal
        """
        logger.info(f"Menghitung fitur untuk {simbol} {timeframe}")

        # Muat data
        file_input = self.direktori_input / f"{simbol}_{timeframe}.parquet"

        if not file_input.exists():
            logger.error(f"File input tidak ditemukan: {file_input}")
            return None

        df = pl.read_parquet(file_input)

        # Pastikan datetime column
        if "waktu_buka" in df.columns:
            if df["waktu_buka"].dtype != pl.Datetime:
                df = df.with_columns(
                    pl.col("waktu_buka").cast(pl.Datetime).alias("datetime")
                )
            else:
                df = df.with_columns(pl.col("waktu_buka").alias("datetime"))
        elif "datetime" in df.columns:
            if df["datetime"].dtype != pl.Datetime:
                df = df.with_columns(pl.col("datetime").cast(pl.Datetime))

        # Konversi ke numpy arrays
        buka = df["buka"].to_numpy().astype(np.float64)
        tinggi = df["tinggi"].to_numpy().astype(np.float64)
        rendah = df["rendah"].to_numpy().astype(np.float64)
        tutup = df["tutup"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)

        # Hitung semua grup fitur
        df = self._hitung_moving_average(df, tutup)
        df = self._hitung_indikator_momentum(df, tinggi, rendah, tutup, volume)
        df = self._hitung_indikator_volatilitas(df, tinggi, rendah, tutup)
        df = self._hitung_indikator_volume(df, tinggi, rendah, tutup, volume)
        df = self._hitung_indikator_trend(df, tinggi, rendah, tutup)
        df = self._hitung_fitur_harga(df)
        df = self._hitung_fitur_turunan(df)

        if self.konfigurasi["hitung_candle_patterns"]:
            df = self._hitung_pola_candle(df, buka, tinggi, rendah, tutup)

        # Buang baris NaN (dari perhitungan indikator)
        jumlah_awal = len(df)
        df = df.drop_nulls()
        jumlah_akhir = len(df)

        logger.info(f"Dropped {jumlah_awal - jumlah_akhir} baris NaN")

        # Simpan ke parquet
        file_output = self.direktori_output / f"{simbol}_{timeframe}_fitur.parquet"
        df.write_parquet(file_output)

        logger.info(
            f"Tersimpan {len(df)} baris dengan {len(df.columns)} fitur ke {file_output}"
        )

        return df

    # ===================================================================================================
    # DERIVED FEATURES
    # ===================================================================================================

    def _hitung_fitur_turunan(self, df: pl.DataFrame) -> pl.DataFrame:
        """Menghitung fitur turunan dari indikator lain."""

        hasil = []

        # MA Crossovers
        if self.konfigurasi["hitung_ma_crossovers"]:
            konfigurasi_cross = self.konfigurasi["ma_crossovers"]

            # Golden/Death Cross signals (SMA)
            gd_fast = konfigurasi_cross["golden_death"]["cepat"]
            gd_slow = konfigurasi_cross["golden_death"]["lambat"]
            col_fast = f"sma_{gd_fast}"
            col_slow = f"sma_{gd_slow}"

            if col_fast in df.columns and col_slow in df.columns:
                hasil.append(
                    pl.when(
                        (pl.col(col_fast) > pl.col(col_slow))
                        & (pl.col(col_fast).shift(1) <= pl.col(col_slow).shift(1))
                    )
                    .then(1)
                    .otherwise(0)
                    .alias("golden_cross")
                )
                hasil.append(
                    pl.when(
                        (pl.col(col_fast) < pl.col(col_slow))
                        & (pl.col(col_fast).shift(1) >= pl.col(col_slow).shift(1))
                    )
                    .then(1)
                    .otherwise(0)
                    .alias("death_cross")
                )

            # EMA cross
            ema_fast = konfigurasi_cross["ema_cross"]["cepat"]
            ema_slow = konfigurasi_cross["ema_cross"]["lambat"]
            col_ema_fast = f"ema_{ema_fast}"
            col_ema_slow = f"ema_{ema_slow}"

            if col_ema_fast in df.columns and col_ema_slow in df.columns:
                hasil.append(
                    pl.when(pl.col(col_ema_fast) > pl.col(col_ema_slow))
                    .then(1)
                    .otherwise(0)
                    .alias(f"ema_{ema_fast}_{ema_slow}_bullish")
                )

        # MA Slopes (trend direction)
        if self.konfigurasi.get("hitung_ma_slope", False):
            ma_base = self.konfigurasi["ma_slope_base"]
            col_ma_base = f"sma_{ma_base}"

            if col_ma_base in df.columns:
                for periode in self.konfigurasi["slope_periods"]:
                    hasil.append(
                        (
                            (pl.col(col_ma_base) - pl.col(col_ma_base).shift(periode))
                            / pl.col(col_ma_base).shift(periode)
                        ).alias(f"sma_{ma_base}_slope_{periode}")
                    )

        # Price vs MA (normalized distance)
        if self.konfigurasi["hitung_price_vs_ma"]:
            for periode in self.konfigurasi["price_vs_ma_periods"]:
                col_sma = f"sma_{periode}"
                if col_sma in df.columns:
                    hasil.append(
                        ((pl.col("tutup") - pl.col(col_sma)) / pl.col(col_sma)).alias(
                            f"tutup_vs_sma_{periode}"
                        )
                    )

        # RSI overbought/oversold zones
        rsi_period = self.konfigurasi["rsi_zone_period"]
        col_rsi = f"rsi_{rsi_period}"
        if col_rsi in df.columns:
            hasil.append(
                pl.when(pl.col(col_rsi) > self.konfigurasi["rsi_overbought"])
                .then(1)
                .otherwise(0)
                .alias("rsi_overbought")
            )
            hasil.append(
                pl.when(pl.col(col_rsi) < self.konfigurasi["rsi_oversold"])
                .then(1)
                .otherwise(0)
                .alias("rsi_oversold")
            )

        # MACD histogram direction
        if "macd_hist" in df.columns:
            hasil.append(
                pl.when(pl.col("macd_hist") > 0)
                .then(1)
                .otherwise(0)
                .alias("macd_hist_positive")
            )
            hasil.append(
                pl.when(pl.col("macd_hist") > pl.col("macd_hist").shift(1))
                .then(1)
                .otherwise(0)
                .alias("macd_hist_rising")
            )

        if hasil:
            df = df.with_columns(hasil)

        # Volatility regime
        vol_atr_period = self.konfigurasi["volatility_atr_period"]
        col_atr = f"atr_{vol_atr_period}"
        if col_atr in df.columns:
            vol_ma_period = self.konfigurasi["volatility_ma_period"]
            high_mult = self.konfigurasi["volatility_high_mult"]
            low_mult = self.konfigurasi["volatility_low_mult"]

            df = df.with_columns(
                [
                    pl.col(col_atr)
                    .rolling_mean(window_size=vol_ma_period)
                    .alias("atr_ma_temp")
                ]
            )
            df = df.with_columns(
                [
                    pl.when(pl.col(col_atr) > pl.col("atr_ma_temp") * high_mult)
                    .then(1)
                    .otherwise(0)
                    .alias("high_volatility"),
                    pl.when(pl.col(col_atr) < pl.col("atr_ma_temp") * low_mult)
                    .then(1)
                    .otherwise(0)
                    .alias("low_volatility"),
                ]
            )
            df = df.drop("atr_ma_temp")

        return df

    # ===================================================================================================
    # CANDLESTICK PATTERNS
    # ===================================================================================================

    def _hitung_pola_candle(
        self,
        df: pl.DataFrame,
        buka: np.ndarray,
        tinggi: np.ndarray,
        rendah: np.ndarray,
        tutup: np.ndarray,
    ) -> pl.DataFrame:
        """Menghitung pola candlestick menggunakan TA-Lib pattern recognition."""

        hasil = []

        # Pola bullish
        hasil.append(
            pl.lit(talib.CDLHAMMER(buka, tinggi, rendah, tutup) / 100).alias(
                "cdl_hammer"
            )
        )
        hasil.append(
            pl.lit(talib.CDLINVERTEDHAMMER(buka, tinggi, rendah, tutup) / 100).alias(
                "cdl_inverted_hammer"
            )
        )
        hasil.append(
            pl.lit(talib.CDLENGULFING(buka, tinggi, rendah, tutup) / 100).alias(
                "cdl_engulfing"
            )
        )
        hasil.append(
            pl.lit(talib.CDLPIERCING(buka, tinggi, rendah, tutup) / 100).alias(
                "cdl_piercing"
            )
        )
        hasil.append(
            pl.lit(talib.CDLMORNINGSTAR(buka, tinggi, rendah, tutup) / 100).alias(
                "cdl_morning_star"
            )
        )
        hasil.append(
            pl.lit(talib.CDL3WHITESOLDIERS(buka, tinggi, rendah, tutup) / 100).alias(
                "cdl_three_white_soldiers"
            )
        )

        # Pola bearish
        hasil.append(
            pl.lit(talib.CDLHANGINGMAN(buka, tinggi, rendah, tutup) / 100).alias(
                "cdl_hanging_man"
            )
        )
        hasil.append(
            pl.lit(talib.CDLSHOOTINGSTAR(buka, tinggi, rendah, tutup) / 100).alias(
                "cdl_shooting_star"
            )
        )
        hasil.append(
            pl.lit(talib.CDLEVENINGSTAR(buka, tinggi, rendah, tutup) / 100).alias(
                "cdl_evening_star"
            )
        )
        hasil.append(
            pl.lit(talib.CDL3BLACKCROWS(buka, tinggi, rendah, tutup) / 100).alias(
                "cdl_three_black_crows"
            )
        )
        hasil.append(
            pl.lit(talib.CDLDARKCLOUDCOVER(buka, tinggi, rendah, tutup) / 100).alias(
                "cdl_dark_cloud"
            )
        )

        # Pola netral/reversal
        hasil.append(
            pl.lit(talib.CDLDOJI(buka, tinggi, rendah, tutup) / 100).alias("cdl_doji")
        )
        hasil.append(
            pl.lit(talib.CDLDRAGONFLYDOJI(buka, tinggi, rendah, tutup) / 100).alias(
                "cdl_dragonfly_doji"
            )
        )
        hasil.append(
            pl.lit(talib.CDLGRAVESTONEDOJI(buka, tinggi, rendah, tutup) / 100).alias(
                "cdl_gravestone_doji"
            )
        )
        hasil.append(
            pl.lit(talib.CDLSPINNINGTOP(buka, tinggi, rendah, tutup) / 100).alias(
                "cdl_spinning_top"
            )
        )
        hasil.append(
            pl.lit(talib.CDLMARUBOZU(buka, tinggi, rendah, tutup) / 100).alias(
                "cdl_marubozu"
            )
        )

        # Pola kelanjutan
        hasil.append(
            pl.lit(talib.CDLRISEFALL3METHODS(buka, tinggi, rendah, tutup) / 100).alias(
                "cdl_rising_three"
            )
        )
        hasil.append(
            pl.lit(talib.CDLTASUKIGAP(buka, tinggi, rendah, tutup) / 100).alias(
                "cdl_tasuki_gap"
            )
        )

        df = df.with_columns(hasil)
        return df

    # ===================================================================================================
    # BATCH PROCESSING
    # ===================================================================================================

    def hitung_semua_simbol(
        self, daftar_simbol: List[str], timeframe: str = "15m"
    ) -> Dict[str, pl.DataFrame]:
        """
        Menghitung fitur untuk semua simbol dalam daftar_simbol.

        Ards:
            daftar_simbol: List pasangan trading
            timeframe: Timeframe yang akan diproses

        Returns:
            Dictionary dengan simbol sebagai kunci dan DataFrame sebagai nilai
        """
        hasil = {}

        for indeks, simbol in enumerate(daftar_simbol, 1):
            logger.info(f"[{indeks}/{len(daftar_simbol)}] Memproses {simbol}")

            try:
                df = self.hitung_fitur(simbol, timeframe)
                if df is not None:
                    hasil[simbol] = df
            except Exception as e:
                logger.error(f"Error menghitung fitur untuk {simbol}: {e}")
                continue

        logger.info(f"Selesai memproses {len(hasil)}/{len(daftar_simbol)} simbol")
        return hasil

    def ambil_daftar_fitur(self) -> List[str]:
        """
        Mengembalikan daftar nama fitur yang akan dihitung.
        Fungsi ini berguna untuk mengetahui dimensi input DQN.

        Returns:
            List nama kolom fitur
        """
        # Hitung dummy untuk mendapatkan nama kolom
        dummy_data = pl.DataFrame(
            {
                "buka": np.random.randn(300) + 100,
                "tinggi": np.random.randn(300) + 101,
                "rendah": np.random.randn(300) + 99,
                "tutup": np.random.randn(300) + 100,
                "volume": np.abs(np.random.randn(300)) * 1000,
            }
        )

        # Fix high/low konsistensi
        dummy_data = dummy_data.with_columns(
            [
                pl.max_horizontal("buka", "tinggi", "tutup").add(0.5).alias("tinggi"),
                pl.min_horizontal("buka", "rendah", "tutup").sub(0.5).alias("rendah"),
            ]
        )

        buka = dummy_data["buka"].to_numpy().astype(np.float64)
        tinggi = dummy_data["tinggi"].to_numpy().astype(np.float64)
        rendah = dummy_data["rendah"].to_numpy().astype(np.float64)
        tutup = dummy_data["tutup"].to_numpy().astype(np.float64)
        volume = dummy_data["volume"].to_numpy().astype(np.float64)

        # Hitung semua fitur
        dummy_data = self._hitung_moving_average(dummy_data, tutup)
        dummy_data = self._hitung_indikator_momentum(
            dummy_data, tinggi, rendah, tutup, volume
        )
        dummy_data = self._hitung_indikator_volatilitas(
            dummy_data, tinggi, rendah, tutup
        )
        dummy_data = self._hitung_indikator_volume(
            dummy_data, tinggi, rendah, tutup, volume
        )
        dummy_data = self._hitung_indikator_trend(dummy_data, tinggi, rendah, tutup)
        dummy_data = self._hitung_fitur_harga(dummy_data)
        dummy_data = self._hitung_fitur_turunan(dummy_data)

        if self.konfigurasi["hitung_candle_patterns"]:
            dummy_data = self._hitung_pola_candle(
                dummy_data, buka, tinggi, rendah, tutup
            )

        # Exclude kolom OHLCV
        fitur_cols = [
            col
            for col in dummy_data.columns
            if col not in ["buka", "tinggi", "rendah", "tutup", "volume", "datetime"]
        ]

        return fitur_cols


def main():
    """Contoh penggunaan."""

    # Coba muat konfigurasi dari file
    path_konfigurasi = Path("konfigurasi/konfigurasi.yaml")
    konfigurasi_pengguna = None

    if path_konfigurasi.exists():
        with open(path_konfigurasi, "r") as f:
            konfigurasi = yaml.safe_load(f)
            konfigurasi_pengguna = konfigurasi.get("fitur", {})

            direktori_input = konfigurasi["data"].get(
                "direktori_data_proses", "data/proses"
            )
            direktori_output = konfigurasi["data"].get(
                "direktori_data_fitur", "data/fitur"
            )
            daftar_simbol = konfigurasi["data"].get("simbol", ["BTCUSDT"])
            timeframe = konfigurasi["pelatihan"].get("timeframe_keputusan", "15m")
    else:
        # Fallback jika tidak ada file konfigurasi
        direktori_input = "data/proses"
        direktori_output = "data/fitur"
        daftar_simbol = ["BTCUSDT"]
        timeframe = "15m"

    # Inisialisasi - penggabungan konfigurasi
    feat_engineer = FiturTA(
        direktori_input=direktori_input,
        direktori_output=direktori_output,
        konfigurasi_pengguna=konfigurasi_pengguna,
    )

    # Tampilkan jumlah fitur yang akan dihitung
    fitur = feat_engineer.ambil_daftar_fitur()
    logger.info(f"Akan menghitung {len(fitur)} fitur")

    # Hitung fitur untuk semua simbol
    feat_engineer.hitung_semua_simbol(daftar_simbol=daftar_simbol, timeframe=timeframe)

    logger.info("Perhitungan fitur selesai!")


if __name__ == "__main__":
    main()
