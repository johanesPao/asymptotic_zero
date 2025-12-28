"""
Pengunduh Data Publik Binance

Mengunduh data OHLCV historis dari repositori data publik Binance.
Sumber data: https://data.binance.vision/
"""

import os
import requests
import zipfile
import polars as pl
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PengunduhDataHistoris:
    """Mengunduh data historis dari repositori data publik Binance"""

    URL_DASAR = "https://data.binance.vision/data/futures/um"

    def __init__(self, direktori_output: str = "data/mentah"):
        """
        Inisialisasi pengunduh

        Args:
            direktori_output: Direktori untuk menyimpan data yang diunduh
        """
        self.direktori_output = Path(direktori_output)
        self.direktori_output.mkdir(parents=True, exist_ok=True)

    def _generate_rentang_tanggal(
        self, tanggal_mulai: str, tanggal_selesai: str, tipe_data: str
    ) -> List[str]:
        """Generate daftar string tanggal untuk pengunduhan."""

        mulai = datetime.strptime(tanggal_mulai, "%Y-%m-%d")
        selesai = datetime.strptime(tanggal_selesai, "%Y-%m-%d")

        daftar_tanggal = []
        saat_ini = mulai

        if tipe_data == "monthly":
            # Generate format YYYY-MM
            while saat_ini <= selesai:
                daftar_tanggal.append(saat_ini.strftime("%Y-%m"))
                # Pindah ke bulan berikutnya
                if saat_ini.month == 12:
                    saat_ini = saat_ini.replace(year=saat_ini.year + 1, month=1)
                else:
                    saat_ini = saat_ini.replace(month=saat_ini.month + 1)
        else:  # daily
            # Generate format YYYY-MM-DD
            while saat_ini <= selesai:
                daftar_tanggal.append(saat_ini.strftime("%Y-%m-%d"))
                saat_ini += timedelta(days=1)

        return daftar_tanggal

    def _unduh_potongan(
        self, simbol: str, interval: str, string_tanggal: str, tipe_data: str
    ) -> Optional[pl.DataFrame]:
        """Mengunduh satu potongan (daily atau monthly)."""

        # Buat URL
        url = f"{self.URL_DASAR}/{tipe_data}/klines/{simbol}/{interval}/{simbol}-{interval}-{string_tanggal}.zip"

        # Unduh
        respon = requests.get(url, timeout=30)

        if respon.status_code == 404:
            # Data mungkin tidak ada untuk tanggal ini
            return None

        respon.raise_for_status()

        # Simpan zip sementara
        zip_sementara = self.direktori_output / f"temp_{simbol}_{string_tanggal}.zip"
        with open(zip_sementara, "wb") as f:
            f.write(respon.content)

        # Ekstrak CSV dari zip
        with zipfile.ZipFile(zip_sementara, "r") as zip_ref:
            nama_csv = zip_ref.namelist()[0]
            zip_ref.extract(nama_csv, self.direktori_output)

        # Baca CSV
        path_csv = self.direktori_output / nama_csv
        df = pl.read_csv(path_csv, has_header=False,
                        new_columns=[
                            "waktu_buka",
                            "buka",
                            "tinggi",
                            "rendah",
                            "tutup",
                            "volume",
                            "waktu_tutup",
                            "quote_volume",
                            "jumlah_trade",
                            "taker_buy_base",
                            "taker_buy_quote",
                            "ignore",
                        ])

        # Bersihkan
        zip_sementara.unlink()
        path_csv.unlink()

        # Format CSV:
        # 0: Waktu buka, 1: Buka, 2: Tinggi, 3: Rendah, 4: Tutup, 5: Volume,
        # 6: Waktu tutup, 7: Quote volume, 8: Jumlah trade,
        # 9: Taker buy base volume, 10: Taker buy quote volume, 11: Ignore

        # Periksa apakah baris pertama adalah header (beberapa file Binance punya header)
        first_val = df.select(pl.col("waktu_buka").first()).item()
        if first_val == "open_time" or not str(first_val).isdigit():
            # Buang baris header
            df = df.slice(1)

        # Konversi timestamp ke datetime
        df = df.with_columns([
            pl.col("waktu_buka").cast(pl.Int64, strict=False).cast(pl.Datetime("ms")),
            pl.col("waktu_tutup").cast(pl.Int64, strict=False).cast(pl.Datetime("ms")),
        ])

        # Konversi ke tipe numerik
        kolom_numerik = [
            "buka",
            "tinggi",
            "rendah",
            "tutup",
            "volume",
            "quote_volume",
            "taker_buy_base",
            "taker_buy_quote",
        ]
        df = df.with_columns([pl.col(kolom).cast(pl.Float64, strict=False) for kolom in kolom_numerik])

        # Konversi jumlah_trade ke integer
        df = df.with_columns(pl.col("jumlah_trade").cast(pl.Int64, strict=False))

        # Buang kolom yang tidak perlu
        df = df.select([
            "waktu_buka",
            "buka",
            "tinggi",
            "rendah",
            "tutup",
            "volume",
            "jumlah_trade",
        ])

        return df

    def unduh_simbol(
        self,
        simbol: str,
        interval: str = "1m",
        tanggal_mulai: str = "2020-01-01",
        tanggal_selesai: Optional[str] = None,
        tipe_data: str = "monthly",  # "daily" atau "monthly"
    ) -> None:
        """
        Mengunduh data historis untuk sebuah simbol

        Args:
            simbol: Pasangan trading (contoh: "BTCUSDT")
            interval: Timeframe (1m, 5m, 15m, 1h, 4h, 1d)
            tanggal_mulai: Tanggal mulai (YYYY-MM-DD)
            tanggal_selesai: Tanggal selesai (YYYY-MM-DD), default hari ini
            tipe_data: Potongan "daily" atau "monthly"
        """
        if tanggal_selesai is None:
            tanggal_selesai = datetime.now().strftime("%Y-%m-%d")

        logger.info(
            f"Mengunduh {simbol} {interval} dari {tanggal_mulai} sampai {tanggal_selesai}"
        )

        # Buat direktori simbol
        direktori_simbol = self.direktori_output / simbol
        direktori_simbol.mkdir(exist_ok=True)

        # Generate rentang tanggal
        daftar_tanggal = self._generate_rentang_tanggal(
            tanggal_mulai, tanggal_selesai, tipe_data
        )

        # Unduh setiap potongan
        semua_data = []
        for string_tanggal in tqdm(daftar_tanggal, desc=f"Mengunduh {simbol}"):
            try:
                df = self._unduh_potongan(simbol, interval, string_tanggal, tipe_data)
                if df is not None and not df.empty:
                    semua_data.append(df)
            except Exception as e:
                logger.warning(f"Gagal mengunduh {simbol} untuk {string_tanggal}: {e}")

        if not semua_data:
            logger.error(f"Tidak ada data yang diunduh untuk {simbol}")
            return

        # Gabungkan semua potongan
        df_gabungan = pl.concat(semua_data)
        df_gabungan = df_gabungan.unique(subset=["waktu_buka"], maintain_order=True)
        df_gabungan = df_gabungan.sort("waktu_buka")

        # Simpan sebagai parquet (jauh lebih efisien dari CSV)
        file_output = direktori_simbol / f"{interval}.parquet"
        df_gabungan.write_parquet(file_output)

        logger.info(f"Tersimpan {len(df_gabungan)} candle ke {file_output}")
        logger.info(
            f"Rentang tanggal: {df_gabungan['waktu_buka'].min()} sampai {df_gabungan['waktu_buka'].max()}"
        )

    def unduh_banyak_simbol(
        self,
        daftar_simbol: List[str],
        interval: str = "1m",
        tanggal_mulai: str = "2020-01-01",
        tanggal_selesai: Optional[str] = None,
    ) -> None:
        """
        Mengunduh data untuk banyak simbol.

        Args:
            daftar_simbol: Daftar pasangan trading
            interval: Timeframe
            tanggal_mulai: Tanggal mulai
            tanggal_selesai: Tanggal selesai
        """
        for simbol in daftar_simbol:
            try:
                self.unduh_simbol(simbol, interval, tanggal_mulai, tanggal_selesai)
            except Exception as e:
                logger.error(f"Error mengunduh {simbol}: {e}")
                continue


def main():
    """Contoh penggunaan"""
    import yaml

    # Muat konfigurasi
    with open("konfigurasi/konfigurasi.yaml", "r") as f:
        konfigurasi = yaml.safe_load(f)

    # Inisialisasi pengunduh
    pengunduh = PengunduhDataHistoris(
        direktori_output=konfigurasi["data"]["direktori_data_mentah"]
    )

    # Unduh data untuk semua simbol yang dikonfigurasi
    pengunduh.unduh_banyak_simbol(
        daftar_simbol=konfigurasi["data"]["simbol"],
        interval=konfigurasi["data"]["interval_dasar"],
        tanggal_mulai=konfigurasi["data"]["tanggal_mulai"],
        tanggal_selesai=konfigurasi["data"]["tanggal_selesai"],
    )

    logger.info("Pengunduhan selesai!")


if __name__ == "__main__":
    main()
