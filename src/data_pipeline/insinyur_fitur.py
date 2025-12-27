"""
Insinyur Fitur

Menghitung indikator teknikal dan fitur untuk trading
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InsinyurFitur:
    """Menghitung indikator teknikal dan fitur."""

    def __init__(
        self, direktori_input: str = "data/proses", direktori_output: str = "data/fitur"
    ):
        """
        Inisialisasi insinyur fitur.

        Args:
            direktori_input: Direktori yang berisi data timeframe terproses
            direktori_output: Direktori untuk menyimpan data fitur
        """
        self.direktori_input = Path(direktori_input)
        self.direktori_output = Path(direktori_output)
        self.direktori_output.mkdir(parents=True, exist_ok=True)

    # ===================================================================================================
    # Metode helper untuk indikator
    # ===================================================================================================
    @staticmethod
    def _rsi(series: pd.Series, periode: int = 14) -> pd.Series:
        """Menghitung RSI."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periode).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periode).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def _stocahstic(
        df: pd.DataFrame, periode_k: int = 14, periode_d: int = 3
    ) -> pd.DataFrame:
        """Menghitung Stochastic Oscillator."""
        rendah_min = df["rendah"].rolling(window=periode_k).min()
        tinggi_max = df["tinggi"].rolling(window=periode_k).max()
        df["stoch_k"] = 100 * (df["tutup"] - rendah_min) / (tinggi_max - rendah_min)
        df["stoch_d"] = df["stoch_k"].rolling(window=periode_d).mean()
        return df

    @staticmethod
    def _mfi(df: pd.DataFrame, periode: int = 14) -> pd.Series:
        """Menghitung Money Flow Index."""
        harga_tipikal = (df["tinggi"] + df["rendah"] + df["tutup"]) / 3
        aliran_uang = harga_tipikal * df["volume"]

        aliran_positif = aliran_uang.where(harga_tipikal > harga_tipikal.shift(1), 0)
        aliran_negatif = aliran_uang.where(harga_tipikal < harga_tipikal.shift(1), 0)

        mf_positif = aliran_positif.rolling(window=periode).sum()
        mf_negatif = aliran_negatif.rolling(window=periode).sum()

        mfi = 100 - (100 / (1 + mf_positif / mf_negatif))
        return mfi

    @staticmethod
    def _cci(df: pd.DataFrame, periode: int = 20) -> pd.Series:
        """Menghitung Commodity Channel Index."""
        harga_tipikal = (df["tinggi"] + df["rendah"] + df["tutup"]) / 3
        sma = harga_tipikal.rolling(window=periode).mean()
        mad = harga_tipikal.rolling(window=periode).apply(
            lambda x: np.abs(x - x.mean()).mean()
        )
        cci = (harga_tipikal - sma) / (0.015 * mad)
        return cci

    @staticmethod
    def _atr(df: pd.DataFrame, periode: int = 14) -> pd.Series:
        """Menghitung Average True Range."""
        tinggi_rendah = df["tinggi"] - df["rendah"]
        tinggi_tutup = np.abs(df["tinggi"] - df["tutup"].shift())
        rendah_tutup = np.abs(df["rendah"] - df["tutup"].shift())

        tr = pd.concat([tinggi_rendah, tinggi_tutup, rendah_tutup], axis=1).max(axis=1)
        atr = tr.rolling(window=periode).mean()
        return atr

    @staticmethod
    def _adx(df: pd.DataFrame, periode: int = 14) -> pd.DataFrame:
        """Menghitung Average Directional Index."""
        # Hitung pergerakan arah
        diff_tinggi = df["tinggi"].diff()
        diff_rendah = df["rendah"].diff()

        pos_dm = diff_tinggi.where((diff_tinggi > diff_rendah) & (diff_tinggi > 0), 0)
        neg_dm = diff_rendah.where((diff_rendah > diff_tinggi) & (diff_rendah > 0), 0)

        # Hitung ATR
        atr = InsinyurFitur._atr(df, periode)

        # Hitung indikator arah
        pos_di = 100 * pos_dm.rolling(window=periode).mean() / atr
        neg_di = 100 * neg_dm.rolling(window=periode).mean() / atr

        # Hitung ADX
        dx = 100 * np.abs(pos_di - neg_di) / (pos_di + neg_di)
        df["adx"] = dx.rolling(window=periode).mean()
        df["plus_di"] = pos_di
        df["minus_di"] = neg_di

        return df

    @staticmethod
    def _parabolic_sar(
        df: pd.DataFrame, af_mulai: float = 0.02, af_maks: float = 0.2
    ) -> pd.Series:
        """Menghitung Parabolic SAR (versi sederhana)."""
        sar = df["tutup"].copy()
        # Ini placeholder sederhana karena SAR yang sebenarnya sangat kompleks
        # Untuk saat ini, kita akan menggunakan EMA sebagai proxy
        sar = df["tutup"].ewm(span=10).mean()
        return sar

    @staticmethod
    def _ichimoku(df: pd.DataFrame) -> pd.DataFrame:
        """Menghitung indikator Ichimoku Cloud."""
        # Tenkan-sen (Conversion Line)
        tinggi_9 = df["tinggi"].rolling(window=9).max()
        rendah_9 = df["rendah"].rolling(window=9).min()
        df["tenkan_sen"] = (tinggi_9 + rendah_9) / 2

        # Kijun-sen (Base Line)
        tinggi_26 = df["tinggi"].rolling(window=26).max()
        rendah_26 = df["rendah"].rolling(window=26).min()
        df["kijun_sen"] = (tinggi_26 + rendah_26) / 2

        # Senkou Span A (Leading Span A)
        df["senkou_span_a"] = ((df["tenkan_sen"] + df["kijun_sen"]) / 2).shift(26)

        # Senkou Span B (Leading Line B)
        tinggi_52 = df["tinggi"].rolling(window=52).max()
        rendah_52 = df["rendah"].rolling(window=52).min()
        df["senkou_span_b"] = ((tinggi_52 + rendah_52) / 2).shift(26)

        return df

    # ===================================================================================================

    def _hitung_moving_average(self, df: pd.DataFrame) -> pd.DataFrame:
        """Menghitung moving average."""

        # Simple Moving Average (SMA)
        for periode in [7, 14, 21, 50, 100, 200]:
            df[f"sma_{periode}"] = df["tutup"].rolling(window=periode).mean()

        # Exponential Moving Average (EMA)
        for periode in [9, 12, 21, 26, 50]:
            df[f"ema_{periode}"] = df["tutup"].ewm(span=periode, adjust=False).mean()

        # Harga relatif terhadap MA (dinormalisasi)
        for periode in [7, 21, 50, 200]:
            df[f"tutup_ke_sma_{periode}"] = (df["tutup"] - df[f"sma_{periode}"]) / df[
                f"sma_{periode}"
            ]

        return df

    def _hitung_indikator_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Menghitung indikator momentum."""

        # RSI (Relative Strength Index)
        for periode in [14, 21]:
            df[f"rsi_{periode}"] = self._rsi(df["tutup"], periode)

        # MACD (Moving Average Convergence Divergence)
        ema_12 = df["tutup"].ewm(span=12, adjust=False).mean()
        ema_26 = df["tutup"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema_12 - ema_26
        df["sinyal_macd"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_diff"] = df["macd"] - df["sinyal_macd"]

        # Stochastic Oscillator
        df = self._stocahstic(df, periode_k=14, periode_d=3)

        # Rate of Change (ROC)
        for periode in [12, 24]:
            df[f"roc_{periode}"] = (
                (df["tutup"] - df["tutup"].shift(periode))
                / df["tutup"].shift(periode)
                * 100
            )

        # Money Flow Index (MFI)
        df["mfi"] = self._mfi(df, periode=14)

        # Commodity Channel Index (CCI)
        df["cci"] = self._cci(df, periode=20)

        return df

    def _hitung_indikator_volatilitas(self, df: pd.DataFrame) -> pd.DataFrame:
        """Menghitung indikator volatilitas."""

        # Bollinger Bands
        for periode, std in [(20, 2), (20, 3)]:
            sma = df["tutup"].rolling(window=periode).mean()
            std_dev = df["tutup"].rolling(window=periode).std()
            df[f"bb_atas_{periode}_{std}"] = sma + (std_dev * std)
            df[f"bb_bawah_{periode}_{std}"] = sma - (std_dev * std)
            df[f"bb_lebar_{periode}_{std}"] = (
                df[f"bb_atas_{periode}_{std}"] - df[f"bb_bawah_{periode}_{std}"]
            ) / sma
            df[f"bb_posisi_{periode}_{std}"] = (
                df["tutup"] - df[f"bb_bawah_{periode}_{std}"]
            ) / (df[f"bb_atas_{periode}_{std}"] - df[f"bb_bawah_{periode}_{std}"])

        # Average True Range (ATR)
        df["atr"] = self._atr(df, periode=14)
        df["atr_persen"] = df["atr"] / df["tutup"] * 100

        # Historical Volatility
        df["volatilitas_20"] = df["tutup"].pct_change().rolling(
            window=20
        ).std() * np.sqrt(20)

        return df

    def _hitung_indikator_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        """Menghitung indikator volume."""

        # Volume Moving Averages
        for periode in [20, 50]:
            df[f"volume_sma_{periode}"] = df["volume"].rolling(window=periode).mean()

        # Rasio volume
        df["rasio_volume"] = df["volume"] / df["volume_sma_20"]

        # On-Balance Volume (OBV)
        df["obv"] = (np.sign(df["tutup"].diff()) * df["volume"]).fillna(0).cumsum()

        # Volume Weighted Average Price (VWAP)
        df["vwap"] = (
            df["volume"] * (df["tinggi"] + df["rendah"] + df["tutup"]) / 3
        ).cumsum() / df["volume"].cumsum()

        return df

    def _hitung_indikator_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """Menghitung indikator trend."""

        # Average Directional Index (ADX)
        df = self._adx(df, periode=14)

        # Parabolic SAR
        df["psar"] = self._parabolic_sar(df)

        # Ichimoku Cloud
        df = self._ichimoku(df)

        return df

    def _hitung_fitur_harga(self, df: pd.DataFrame) -> pd.DataFrame:
        """Menghitung fitur berbasis harga."""

        # Perubahan harga (return)
        for periode in [1, 4, 24, 168]:
            df[f"perubahan_harga_{periode}"] = df["tutup"].pct_change(periode)

        # Range Tinggi-Rendah
        df["range_tr"] = (df["tinggi"] - df["rendah"]) / df["tutup"]

        # Posisi tutup dalam candle
        df["posisi_tutup"] = (df["tutup"] - df["rendah"]) / (
            df["tinggi"] - df["rendah"]
        )

        # Ukuran body dan shadow candle
        df["ukuran_body"] = abs(df["buka"] - df["tutup"]) / df["tutup"]
        df["shadow_atas"] = (df["tinggi"] - df[["buka", "tutup"]].max(axis=1)) / df[
            "tutup"
        ]
        df["shadow_bawah"] = (df[["buka", "tutup"]].min(axis=1) - df["rendah"]) / df[
            "tutup"
        ]

        return df

    def hitung_fitur(
        self, simbol: str, timeframe: str = "15m", config: Dict = None
    ) -> pd.DataFrame:
        """
        Menghitung semua fitur untuk sebuah simbol

        Args:
            simbol: Pasangan trading
            timeframe: Timeframe yang akan digunakan
            config: Dictionary konfigurasi fitur
        """
        logger.info(f"Menghitung fitur untuk {simbol} {timeframe}")

        # Muat data
        file_input = self.direktori_input / f"{simbol}_{timeframe}.parquet"

        if not file_input.exists():
            logger.error(f"File input tidak ditemukan: {file_input}")
            return None

        df = pd.read_parquet(file_input)

        # Pastikan datetime index
        if "waktu_buka" in df.columns:
            df["datetime"] = pd.to_datetime(df["waktu_buka"])
            df = df.set_index("datetime")

        # Hitung semua grup fitur
        df = self._hitung_moving_average(df)
        df = self._hitung_indikator_momentum(df)
        df = self._hitung_indikator_volatilitas(df)
        df = self._hitung_indikator_volume(df)
        df = self._hitung_indikator_trend(df)
        df = self._hitung_fitur_harga(df)

        # Buang barus NaN (dari perhitungan indikator)
        df = df.dropna()

        # Simpan ke parquet
        file_output = self.direktori_output / f"{simbol}_{timeframe}_fitur.parquet"
        df.reset_index().to_parquet(file_output, index=False)

        logger.info(
            f"Tersimpan {len(df)} baris dengan {len(df.columns)} fitur ke {file_output}"
        )

        return df

    def hitung_semua_simbol(
        self, daftar_simbol: List[str], timeframe: str = "15m"
    ) -> None:
        """Menghitung fitur untuk semua simbol."""
        for simbol in daftar_simbol:
            try:
                self.hitung_fitur(simbol, timeframe)
            except Exception as e:
                logger.error(f"Error menghitung fitur untuk {simbol}: {e}")
                continue


def main():
    """Contoh penggunaan."""
    import yaml

    # Muat konfigurasi
    with open("konfigurasi/konfigurasi.yaml", "r") as f:
        konfigurasi = yaml.safe_load(f)

    # Inisialisasi insinyur fitur
    insinyur = InsinyurFitur(
        direktori_input=konfigurasi["data"]["direktori_data_proses"],
        direktori_output=konfigurasi["data"]["direktori_data_fitur"],
    )

    # Hitung fitur untuk semua simbol
    timeframe = konfigurasi["pelatihan"]["timeframe_keputusan"]
    insinyur.hitung_semua_simbol(
        daftar_simbol=konfigurasi["data"]["simbol"], timeframe=timeframe
    )

    logger.info("Perhitungan fitur selesai!")


if __name__ == "__main__":
    main()
