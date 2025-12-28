"""
Eksplorasi Data

Utilitas untuk eksplorasi dan visualisasi data yang telah diunduh.
"""

import pandas as pd
from pathlib import Path
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def eksplorasi_fitur(
    direktori_fitur: str, simbol: str = "BTCUSDT", timeframe: str = "15m"
) -> pd.DataFrame:
    """
    Muat dan tampilkan informasi tentang data fitur.

    Args:
        direktori_fitur: Direktori yang berisi file fitur
        simbol: Simbol untuk di-eksplorasi
        timeframe: Timeframe untuk di-eksplorasi
    """
    path_fitur = Path(direktori_fitur)
    path_file = path_fitur / f"{simbol}_{timeframe}_fitur.parquet"

    if not path_file.exists():
        logger.error(f"File tidak ditemukan: {path_file}")
        logger.info(f"File yang tersedia di {path_fitur}:")
        for file in sorted(path_fitur.glob("*.parquet")):
            logger.info(f"- {file.name}")
        return

    # Muat data
    logger.info(f"Memuat {path_file}")
    df = pd.read_parquet(path_file)

    print("\n" + "=" * 80)
    print(f"RINGKASAN DATA: {simbol} {timeframe}")
    print("=" * 80)

    # Info dasar
    print(f"\nBentuk: {df.shape[0]:,} baris x {df.shape[1]} kolom")
    print(f"Rentang Tanggal: {df['datetime'].min()} sampai {df['datetime'].max()}")
    print(f"Durasi: {(df['datetime'].max() - df['datetime'].min()).days} hari")

    # Penggunaan memori
    memori_mb = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Penggunaan Memori: {memori_mb:.2f} MB")

    # Statistik harga
    print("\n" + "-" * 80)
    print("STATISTIK HARGA")
    print("-" * 80)
    print(f"Rentang Harga: ${df['rendah'].min():.2f} - ${df['tinggi'].max():.2f}")
    print(f"Harga Terakhir: ${df['tutup'].iloc[-1]:.2f}")
    print(
        f"Return Total: {((df['tutup'].iloc[-1] / df['tutup'].iloc[0]) -1) * 100:.2f}%"
    )

    # Statistik volume
    print("\n" + "-" * 80)
    print("STATISTIK VOLUME")
    print("-" * 80)
    print(f"Volume Rata-rata: {df['volume'].mean():,.0f}")
    print(f"Volume Maksimal: {df['volume'].max():,.0f}")

    # Indikator teknikal
    print("\n" + "-" * 80)
    print("INDIKATOR TEKNIKAL")
    print("-" * 80)

    # Kelompokkan fitur berdasarkan tipe
    grup_fitur = {
        "Moving Average": [c for c in df.columns if "sma" in c or "ema" in c],
        "Momentum": [
            c
            for c in df.columns
            if any(x in c for x in ["rsi", "macd", "stoch", "roc", "mfi", "cci"])
        ],
        "Volatilitas": [
            c for c in df.columns if any(x in c for x in ["bb_", "atr", "volatilitas"])
        ],
        "Volume": [c for c in df.columns if "volume" in c or "obv" in c or "vwap" in c],
        "Trend": [
            c
            for c in df.columns
            if any(x in c for x in ["adx", "psar", "ichimoku", "kijun", "senkou"])
        ],
        "Fitur Harga": [c for c in df],
    }

    for grup, fitur in grup_fitur.items():
        if fitur:
            print(f"\n{grup} {len(fitur)} fitur")
            for feat in sorted(fitur)[:5]:  # Tampilkan 5 pertama
                print(f"- {feat}")
            if len(fitur) > 5:
                print(f"... dan {len(fitur) - 5} lagi")

    # Contoh data
    print("\n" + "-" * 80)
    print("CONTOH DATA (5 baris terakhir)")
    print("-" * 80)

    # Tampilakn kolom kunci saja untuk keterbacaan
    kolom_kunci = [
        "datetime",
        "buka",
        "tinggi",
        "rendah",
        "tutup",
        "volume",
        "rsi_14",
        "macd",
        "bb_posisi_20_2",
    ]
    kolom_tersedia = [c for c in kolom_kunci if c in df.columns]

    print(df[kolom_tersedia].tail().to_string(index=False))

    # Pemeriksaan nilai yang hilang
    hilang = df.isnull().sum()
    if hilang.any():
        print("\n" + "-" * 80)
        print("PERINGATAN NILAI YANG HILANG")
        print("-" * 80)
        print(hilang[hilang > 0])
    else:
        print("\n✅ Tidak ada nilai yang hilang!")

    print("\n" + "=" * 80)

    return df


def daftar_data_tersedia(direktori_fitur: str):
    """Daftar semua file fitur yang tersedia."""
    path_fitur = Path(direktori_fitur)

    if not path_fitur.exists():
        logger.error(f"Direktori tidak ditemukan: {path_fitur}")
        return

    file = sorted(path_fitur.glob("*_fitur.parquet"))

    print("\n" + "=" * 80)
    print("FILE DATA YANG TERSEDIA")
    print("=" * 80)

    if not file:
        print("Tidak ada file fitur yang ditemukan!")
        return

    # Kelompokkan berdasarkan simbol
    data_simbol = {}
    for file_item in file:
        bagian = file_item.stem.split("_")
        simbol = bagian[0]
        timeframe = bagian[1] if len(bagian) > 1 else "tidak diketahui"

        if simbol not in data_simbol:
            data_simbol[simbol] = []
        data_simbol[simbol].append(timeframe)

    print(f"\nDitemukan data untuk {len(data_simbol)} simbol:\n")

    for simbol in sorted(data_simbol.keys()):
        timeframe_list = ", ".join(sorted(data_simbol[simbol]))
        print(f"{simbol:15} → Timeframe: {timeframe_list}")

    print("\n" + "=" * 80)


def bandingkan_simbol(
    direktori_fitur: str, daftar_simbol: list, timeframe: str = "15m"
) -> None:
    """Bandingkan statistik di berbagai simbol."""
    path_fitur = Path(direktori_fitur)

    print("\n" + "=" * 80)
    print(f"PERBANDINGAN: {', '.join(daftar_simbol)} ({timeframe})")
    print("=" * 80)

    data = {}
    for simbol in daftar_simbol:
        path_file = path_fitur / f"{simbol}_{timeframe}_fitur.parquet"
        if path_file.exists():
            data[simbol] = pd.read_parquet(path_file)
        else:
            logger.warning(f"Data tidak ditemukan untuk {simbol}")

    if not data:
        logger.error("Tidak ada data yang dimuat!")
        return

    # Buat tabel perbandingan
    perbandingan = []
    for simbol, df in data.items():
        perbandingan.append(
            {
                "Simbol": simbol,
                "Baris": f"{len(df):,}",
                "Mulai": df["datetime"].min().strftime("%Y-%m-%d"),
                "Selesai": df["datetime"].max().strftime("%Y-%m-%d"),
                "Harga Terakhir": f"${df['tutup'].iloc[-1]:.2f}",
                "Return Total": f"{((df['tutup'].iloc[-1] / df['tutup'].iloc[0]) -1) * 100:.1f}%",
                "Volume Rata-rata": f"{df['volume'].mean():.0f}",
                "Volatilitas": (
                    f"{df['volatilitas_20'].iloc[-1]:.2f}"
                    if "volatilitas_20" in df
                    else "N/A"
                ),
            }
        )

    df_perbandingan = pd.DataFrame(perbandingan)
    print("\n", df_perbandingan.to_string(index=False))
    print("\n" + "=" * 80)


def main():
    """Titik masuk file eksplorasi_data."""
    parser = argparse.ArgumentParser(
        description="Eksplorasi data histori trading kripto"
    )
    parser.add_argument(
        "--direktori-fitur",
        default="data/fitur",
        help="Direktori yang berisi file fitur",
    )
    parser.add_argument(
        "--aksi",
        choices=["daftar", "eksplorasi", "bandingkan"],
        default="daftar",
        help="Tindakan yang dapat anda lakukan",
    )
    parser.add_argument(
        "--simbol", default="BTCUSDT", help="Simbol untuk di-eksplorasi"
    )
    parser.add_argument("--simbol-list", nargs="+", help="Simbol untuk dibandingkan")
    parser.add_argument(
        "--timeframe", default="15m", help="Timeframe untuk di-eksplorasi"
    )

    args = parser.parse_args()

    match args.aksi:
        case "daftar":
            daftar_data_tersedia(args.direktori_fitur)
        case "eksplorasi":
            eksplorasi_fitur(args.direktori_fitur, args.simbol, args.timeframe)
        case "bandingkan":
            if not args.simbol_list:
                logger.error("Harap berikan --simbol-list untuk perbandingan")
                return
            bandingkan_simbol(args.direktori_fitur, args.simbol_list, args.timeframe)


if __name__ == "__main__":
    main()
