"""
Pipeline Data Utama

Menjalankan pipeline data lengkap
1. Mengunduh data historis dari Binance
2. Agregasi ke berbagai timeframe
3. Menghitung indikator teknikal dan fitur
"""

import yaml
import logging
import argparse
from pathlib import Path
import sys

# Tambahkan src ke path
sys.path.append(str(Path(__file__).parent.parent))

from data_pipeline.pengunduh import PengunduhDataHistoris
from data_pipeline.agregator import AgregatorTimeframe
from data_pipeline.insinyur_fitur import InsinyurFitur

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def muat_konfigurasi(path_konfigurasi: str = "konfigurasi/konfigurasi.yaml") -> dict:
    """Muat file konfigurasi."""
    with open(path_konfigurasi, "r") as f:
        return yaml.safe_load(f)


def jalankan_pengunduhan(konfigurasi: dict, daftar_simbol: list = None) -> None:
    """Jalankan langkah pengunduhan data."""
    logger.info("=" * 60)
    logger.info("LANGKAH 1: Mengunduh data historis dari Binance")
    logger.info("=" * 60)

    pengunduh = PengunduhDataHistoris(
        direktori_output=konfigurasi["data"]["direktori_data_mentah"]
    )

    simbol_untuk_unduh = daftar_simbol or konfigurasi["data"]["simbol"]

    pengunduh.unduh_banyak_simbol(
        daftar_simbol=simbol_untuk_unduh,
        interval=konfigurasi["data"]["interval_dasar"],
        tanggal_mulai=konfigurasi["data"]["tanggal_mulai"],
        tanggal_selesai=konfigurasi["data"]["tanggal_selesai"],
    )

    logger.info("Pengunduhan selesai!")


def jalankan_agregasi(konfigurasi: dict, daftar_simbol: list = None) -> None:
    """Jalankan langkah agregasi timeframe."""
    logger.info("=" * 60)
    logger.info("LANGKAH 2: Agregasi ke berbagai timeframe")
    logger.info("=" * 60)

    agregator = AgregatorTimeframe(
        direktori_input=konfigurasi["data"]["direktori_data_mentah"],
        direktori_output=konfigurasi["data"]["direktori_data_proses"],
    )

    simbol_untuk_agregasi = daftar_simbol or konfigurasi["data"]["simbol"]

    agregator.agregasi_semua_simbol(
        daftar_simbol=simbol_untuk_agregasi,
        timeframe_sumber=konfigurasi["data"]["interval_dasar"],
        daftar_timeframe_target=konfigurasi["data"]["timeframe"],
    )

    logger.info("Agregasi selesai!")


def jalankan_rekayasa_fitur(konfigurasi: dict, daftar_simbol: list = None) -> None:
    """Jalankan langkah rekayasa fitur."""
    logger.info("=" * 60)
    logger.info("LANGKAH 3: Menghitung indikator teknikal dan fitur")
    logger.info("=" * 60)

    insinyur = InsinyurFitur(
        direktori_input=konfigurasi["data"]["direktori_data_proses"],
        direktori_output=konfigurasi["data"]["direktori_data_fitur"],
    )

    simbol_untuk_proses = daftar_simbol or konfigurasi["data"]["simbol"]
    timeframe = konfigurasi["pelatihan"]["timeframe_keputusan"]

    insinyur.hitung_semua_simbol(daftar_simbol=simbol_untuk_proses, timeframe=timeframe)

    logger.info("Rekayasa fitur selesai!")


def jalankan_pipeline_lengkap(
    konfigurasi: dict, daftar_simbol: list = None, lewati_unduh: bool = False
) -> None:
    """Jalankan pipeline data lengkap"""
    logger.info("=" * 60)
    logger.info("MEMULAI PIPELINE DATA LENGKAP")
    logger.info("=" * 60)

    if not lewati_unduh:
        jalankan_pengunduhan(konfigurasi, daftar_simbol)
    else:
        logger.info("Melewati langkah pengunduhan (flag --lewati-unduh)")

    jalankan_agregasi(konfigurasi, daftar_simbol)
    jalankan_rekayasa_fitur(konfigurasi, daftar_simbol)

    logger.info("=" * 60)
    logger.info("PIPELINE SELESAI!")
    logger.info("=" * 60)
    logger.info(f"Data mentah: {konfigurasi['data']['direktori_data_mentah']}")
    logger.info(f"Data proses: {konfigurasi['data']['direktori_data_proses']}")
    logger.info(f"Data fitur: {konfigurasi['data']['direktori_data_fitur']}")


def verifikasi_data(konfigurasi: dict) -> None:
    """Verifikasi bahwa data ada dan tampilkan ringkasan."""
    logger.info("=" * 60)
    logger.info("VERIFIKASI DATA")
    logger.info("=" * 60)

    direktori_fitur = Path(konfigurasi["data"]["direktori_data_fitur"])

    if not direktori_fitur.exists():
        logger.error(f"Direktori fitur tidak ditemukan: {direktori_fitur}")
        return

    file_fitur = list(direktori_fitur.glob("*_fitur.parquet"))

    if not file_fitur:
        logger.warning("Tidak ada file fitur yang ditemukan!")
        return

    logger.info(f"Ditemukan {len(file_fitur)} file fitur:")

    import pandas as pd

    for file in sorted(file_fitur):
        df = pd.read_parquet(file)
        logger.info(f"{file.name}: {len(df)} baris, {len(df.columns)} fitur")
        logger.info(
            f"Rentang tanggal: {df['datetime'].min()} sampai {df['datetime'].max()}"
        )


def main():
    """Titik masuk utama file eksekusi pipeline."""
    parser = argparse.ArgumentParser(description="Pipeline Data DQN Kripto")
    parser.add_argument(
        "--konfigurasi",
        default="konfigurasi/konfigurasi.yaml",
        help="Path ke file konfigurasi yaml",
    )
    parser.add_argument(
        "--langkah",
        choices=["unduh", "agregasi", "fitur", "semua", "verifikasi"],
        default="semua",
        help="Langkah mana pada eksekusi pipeline yang akan dijalankan",
    )
    parser.add_argument(
        "--simbol",
        nargs="+",
        help="Simbol spesifik untuk diproses (menimpa konfigurasi)",
    )
    parser.add_argument(
        "--lewati-unduh",
        action="store_true",
        help="Lewati langkah pengunduhan (gunakan data existing)",
    )

    args = parser.parse_args()

    # Muat konfigurasi
    konfigurasi = muat_konfigurasi(args.konfigurasi)

    # Jalankan langkah yang diminta user
    match args.langkah:
        case "unduh":
            jalankan_pengunduhan(konfigurasi, args.simbol)
        case "agregasi":
            jalankan_agregasi(konfigurasi, args.simbol)
        case "fitur":
            jalankan_rekayasa_fitur(konfigurasi, args.simbol)
        case "verifikasi":
            verifikasi_data(konfigurasi)
        case _:
            jalankan_pipeline_lengkap(konfigurasi, args.simbol, args.lewati_unduh)
            verifikasi_data(konfigurasi)


if __name__ == "__main__":
    main()
