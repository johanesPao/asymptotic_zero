"""
Agregator Timeframe

Mengkonversi data 1-menit ke berbagai timeframe di atasnya (5m, 15m, 1h, 4h, 1d).
"""

import polars as pl
from pathlib import Path
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgregatorTimeframe:
    """Agregasi data OHLCV 1-menit ke berbagai timeframe di atasnya."""

    # Mapping string timeframe ke aturan resample pandas
    PETA_TIMEFRAME = {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "1h",
        "2h": "2h",
        "4h": "4h",
        "6h": "6h",
        "12h": "12h",
        "1d": "1d",
        "1w": "1w",
    }

    def __init__(
        self,
        direktori_input: str = "data/mentah",
        direktori_output: str = "data/proses",
    ):
        """
        Inisiasi agragator

        Args:
            direktori_input: Direktori yang berisi data 1m mentah
            direktori_output: Direktori untuk menyimpan data ter-agregasi
        """
        self.direktori_input = Path(direktori_input)
        self.direktori_output = Path(direktori_output)
        self.direktori_output.mkdir(parents=True, exist_ok=True)

    def _resample_ohlcv(self, df: pl.DataFrame, timeframe: str) -> pl.DataFrame:
        """
        Resample data OHLCV ke timeframe berbeda.

        Args:
            df: DataFrame dengan data OHLCV
            timeframe: Timeframe target (contoh: "15m", "1h")

        Returns:
            DataFrame yang di-resample
        """
        if timeframe not in self.PETA_TIMEFRAME:
            raise ValueError(f"Timeframe tidak didukung: {timeframe}")

        aturan = self.PETA_TIMEFRAME[timeframe]

        # Resample dengan fungsi agregasi yang sesuai
        df_agregasi = df.group_by_dynamic("waktu_buka", every=aturan).agg(
            [
                pl.col("buka").first(),  # Buka pertama dalam periode
                pl.col("tinggi").max(),  # Harga tertinggi
                pl.col("rendah").min(),  # Harga terendah
                pl.col("tutup").last(),  # Tutup terakhir
                pl.col("volume").sum(),  # Total volume
                pl.col("jumlah_trade").sum(),  # Total trade
            ]
        )

        # Hapus candle yang tidak lengkap (baris terakhir mungkin tidak lengkap)
        df_agregasi = df_agregasi.drop_nulls()

        return df_agregasi

    def agregasi_simbol(
        self,
        simbol: str,
        timeframe_sumber: str = "1m",
        daftar_timeframe_target: List[str] = None,
    ) -> None:
        """
        Agregasi sebuah simbol ke berbagai timeframe.

        Args:
            simbol: Pasangan trading (contoh: "BTCUSDT)
            timeframe_sumber: Timeframe sumber (default "1m")
            daftar_timeframe_target: Daftar timeframe target
        """
        if daftar_timeframe_target is None:
            daftar_timeframe_target = ["5m", "15m", "1h", "4h", "1d"]

        logger.info(
            f"Agregasi {simbol} dari {timeframe_sumber} ke {daftar_timeframe_target}"
        )

        # Muat data sumber
        file_sumber = self.direktori_input / simbol / f"{timeframe_sumber}.parquet"

        if not file_sumber.exists():
            logger.error(f"File sumber tidak ditemukan: {file_sumber}")
            return

        df = pl.read_parquet(file_sumber)

        # Pastikan waktu_buka adalah datetime
        if df["waktu_buka"].dtype != pl.Datetime:
            df = df.with_columns(pl.col("waktu_buka").str.to_datetime())

        # Sort by waktu_buka untuk resampling
        df = df.sort("waktu_buka")

        # Agregasi ke setiap timeframe target
        for tf_target in daftar_timeframe_target:
            try:
                df_agregasi = self._resample_ohlcv(df, tf_target)

                # Simpan ke parquet
                file_output = self.direktori_output / f"{simbol}_{tf_target}.parquet"
                df_agregasi.write_parquet(file_output)

                logger.info(f"Dibuat {file_output} dengan {len(df_agregasi)} candle")
            except Exception as e:
                logger.error(f"Error agregasi {simbol} ke {tf_target}: {e}")

    def agregasi_semua_simbol(
        self,
        daftar_simbol: List[str],
        timeframe_sumber: str = "1m",
        daftar_timeframe_target: List[str] = None,
    ) -> None:
        """
        Agregasi banyak simbol.

        Args:
            daftar_simbol: Daftar pasangan trading
            timeframe_sumber: Timeframe sumber
            daftar_timeframe_target: Daftar timeframe target
        """
        for simbol in daftar_simbol:
            try:
                self.agregasi_simbol(simbol, timeframe_sumber, daftar_timeframe_target)
            except Exception as e:
                logger.error(f"Error agregasi {simbol}: {e}")
                continue


def main():
    """Contoh penggunaan."""
    import yaml

    # Muat konfigurasi
    with open("konfigurasi/konfigurasi.yaml", "r") as f:
        konfigurasi = yaml.safe_load(f)

    # Inisialisasi agregator
    agregator = AgregatorTimeframe(
        direktori_input=konfigurasi["data"]["direktori_data_mentah"],
        direktori_output=konfigurasi["data"]["direktori_data_proses"],
    )

    # Agregasi semua simbol
    agregator.agregasi_semua_simbol(
        daftar_simbol=konfigurasi["data"]["simbol"],
        timeframe_sumber=konfigurasi["data"]["interval_dasar"],
        daftar_timeframe_target=konfigurasi["data"]["timeframe"],
    )

    logger.info("Agregasi selesai!")


if __name__ == "__main__":
    main()
