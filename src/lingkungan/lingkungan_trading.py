"""
Lingkungan Trading untuk DQN Agent

Environment yang kompatibel dengan OpenAI Gym untuk simulasi trading cryptocurrency futures.
Agent berinteraksi dengan historical market data dan belajar strategi trading yang optimal.
"""

import numpy as np
import polars as pl
from typing import Tuple, Dict, Optional, List
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LingkunganTrading:
    """
    Trading environment untuk cryptocurrency futures.

    Observasi (State)
        - 70+ indikator teknikal dari fitur_ta.py
        - Informasi posisi saat ini
        - Account balance dan P&L

    Aksi (Action)
        0: HOLD     - Tidak melakukan apa-apa / flat
        1: BUY      - Buka posisi LONG (jika flat) atau tambah LONG
        2: SELL     - Buka posisi SHORT (jika flat) atau tambah SHORT
        3: CLOSE    - Tutup posisi saat ini
        4: REVERSE  - Balik posisi (LONG+SHORT atau SHORT+LONG)

    Reward:
        - Profit/loss dari trading
        - Penalti untuk fee trading
        - Penalti untuk volatilitas (risk management)
    """

    # Konstanta aksi
    HOLD = 0
    BUY = 1
    SELL = 2
    CLOSE = 3
    REVERSE = 4

    # Konstanta posisi
    FLAT = 0
    LONG = 1
    SHORT = -1

    def __init__(
        self,
        data: pl.DataFrame,
        saldo_awal: float = 10000.0,
        biaya_trading: float = 0.0004,
        leverage: int = 1,
        max_posisi: int = 1,
        panjang_episode_max: int = 1500,
        kolom_fitur: List[str] = None,
    ) -> None:
        """
        Inisialisasi trading environment

        Args:
            data: Polars DataFrame dengan OHLCV + fitur teknikal
            saldo_wal: Saldo awal dalam USD
            biaya_trading: Fee trading (0.0004 = 0.04% untuk Binance future maker)
            leverage: Leverage untuk futures trading (1 = tidak ada leverage)
            max_posisi: Maksimal posisi bersamaan
            panjang_episode_max: Maksimal candle per episode
            kolom_fitur: List nama kolom untuk state (jika None, auto-deteksi)
        """
        self.data = data
        self.saldo_awal = saldo_awal
        self.biaya_trading = biaya_trading
        self.leverage = leverage
        self.max_posisi = max_posisi
        self.panjang_episode_max = panjang_episode_max

        # Auto deteksi kolom fitur (exclude OHLCV dan datetime)
        if kolom_fitur is None:
            kolom_exclude = [
                "datetime",
                "waktu_buka",
                "buka",
                "tinggi",
                "rendah",
                "tutup",
                "volume",
            ]
            self.kolom_fitur = [col for col in data.columns if col not in kolom_exclude]
        else:
            self.kolom_fitur = kolom_fitur

        # Dimensi observasi dan ruang tindakan
        # Observasi = fitur teknikal + info posisi (4 nilai: posisi, harga_masuk, ukuran, pnl_persen)
        self.dim_observasi = len(self.kolom_fitur) + 4
        self.dim_aksi = 5  # HOLD, BUY, SELL, CLOSE, REVERSE

        # State variabel (di-reset pada setiap episode)
        self.indeks_saat_ini = 0
        self.saldo = saldo_awal
        self.posisi = self.FLAT  # FLAT, LONG atau SHORT
        self.harga_masuk = 0.0
        self.ukuran_posisi = 0.0
        self.pnl_terealisasi = 0.0
        self.pnl_belum_terealisasi = 0.0

        # Histori untuk tracking
        self.histori_saldo = []
        self.histori_aksi = []
        self.histori_reward = []

        logger.info("LingkunganTrading diinisialisasi:")
        logger.info(f"  - Data points: {len(self.data)}")
        logger.info(f"  - Fitur: {len(self.kolom_fitur)}")
        logger.info(f"  - Observasi dim: {self.dim_observasi}")
        logger.info(f"  - Aksi dim: {self.dim_aksi}")
        logger.info(f"  - Panjang episdoe maksimum: {self.panjang_episode_max}")

    def _dapatkan_observasi(self) -> np.ndarray:
        """
        Dapatkan vektor keadaan (state) untuk timestamp saat ini.

        Returns:
            observasi: Array numpy dengan shape (dim_observasi,)
                - fitur teknikal (70+ nilai)
                - info posisi (4 nilai: posisi, norm_harga_masuk, norm_ukuran, pnl_persen)
        """
        # Ambil fitur teknikal dari data menggunakan Polars
        # Polars: df.select(columns).row(index) atau df[index, columns]
        fitur = self.data.select(self.kolom_fitur).row(self.indeks_saat_ini)
        fitur = np.array(fitur, dtype=np.float32)

        # Hitung harga sekarang untuk normalisasi
        harga_sekarang = self._dapatkan_harga_sekarang()

        # Info posisi (di-normalisasi)
        if self.posisi == self.FLAT:
            info_posisi = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        else:
            # Normalisasi harga masuk relatif terhadap harga sekarang
            norm_harga_masuk = (self.harga_masuk - harga_sekarang) / harga_sekarang

            # Normalisasi ukuran posisi relatif terhadap saldo
            norm_ukuran = self.ukuran_posisi / self.saldo if self.saldo > 0 else 0.0

            # P&L sebagai persentase
            pnl_persen = (
                self.pnl_belum_terealisasi / self.saldo if self.saldo > 0 else 0.0
            )

            info_posisi = np.array(
                [
                    float(self.posisi),  # -1, 0 atau 1
                    norm_harga_masuk,
                    norm_ukuran,
                    pnl_persen,
                ],
                dtype=np.float32,
            )

        # Gabungkan fitur + info posisi
        observasi = np.concatenate([fitur, info_posisi])

        # Replace NaN dengan 0 (safety check)
        observasi = np.nan_to_num(observasi, nan=0.0, posinf=1.0, neginf=-1.0)

        return observasi

    def _dapatkan_harga_sekarang(self) -> float:
        """Dapatkan harga close saat ini."""
        return self.data[self.indeks_saat_ini, "tutup"]

    def _nama_aksi(self, aksi: int) -> str:
        """Konversi aksi integer ke nama aksi."""
        nama = {0: "HOLD", 1: "BUY", 2: "SELL", 3: "CLOSE", 4: "REVERSE"}
        return nama.get(aksi, "UNKNOWN")

    def _nama_posisi(self) -> str:
        """Konversi integer posisi ke nama."""
        match self.posisi:
            case self.FLAT:
                return "FLAT"
            case self.LONG:
                return "LONG"
            case self.SHORT:
                return "SHORT"

    def _buka_posisi(self, tipe_posisi: int, harga: float) -> None:
        """
        Buka posisi baru (LONG atau SHORT).

        Args:
            tipe_posisi: LONG (1) atau SHORT (-1)
            harga: Harga masuk
        """
        # Hitung ukuran posisi berdasarkan saldo dan leverage
        # Gunakan persentsae dari saldo (misal 100% dari saldo tersedia)
        ukuran = (self.saldo * self.leverage) / harga

        # Kurangi fee dari saldo
        fee = self.saldo * self.biaya_trading
        self.saldo -= fee

        # Set posisi
        self.posisi = tipe_posisi
        self.harga_masuk = harga
        self.ukuran_posisi = ukuran
        self.pnl_belum_terealisasi = 0.0

    def _tutup_posisi(self, harga: float) -> None:
        """
        Tutup posisi saat ini.

        Args:
            harga: Harga keluar
        """
        match self.posisi:
            case self.FLAT:
                return
            case self.LONG:
                pnl = (harga - self.harga_masuk) * self.ukuran_posisi
            case self.SHORT:
                pnl = (self.harga_masuk - harga) * self.ukuran_posisi

        # Kurangi fee
        nilai_posisi = harga * self.ukuran_posisi
        fee = nilai_posisi * self.biaya_trading
        pnl -= fee

        # Update saldo
        self.saldo += pnl
        self.pnl_terealisasi += pnl

        # Reset posisi
        self.posisi = self.FLAT
        self.harga_masuk = 0.0
        self.ukuran_posisi = 0.0
        self.pnl_belum_terealisasi = 0.0

    def _eksekusi_aksi(self, aksi: int, harga: float) -> None:
        """
        Eksekusi tindakan trading.

        Args:
            aksi: integer 0-4
            harga: Harga eksekusi saat ini
        """
        match aksi:
            case self.HOLD:
                # Tidak melakukan apa-apa
                pass
            case self.BUY:
                match self.posisi:
                    case self.FLAT:
                        # Buka posisi LONG
                        self._buka_posisi(self.LONG, harga)
                    case self.LONG:
                        # Sudah LONG, hold
                        pass
                    case self.SHORT:
                        # Sedang SHORT, tapi aksi BUY → tutup SHORT dulu
                        self._tutup_posisi(harga)
            case self.SELL:
                match self.posisi:
                    case self.FLAT:
                        # Buka posisi SHORT
                        self._buka_posisi(self.SHORT, harga)
                    case self.SHORT:
                        # Sudah SHORT, hold
                        pass
                    case self.LONG:
                        # Sedang LONG tapi aksi SELL → tutup LONG dulu
                        self._tutup_posisi(harga)
            case self.CLOSE:
                # Tutup posisi apapun
                if self.posisi != self.FLAT:
                    self._tutup_posisi(harga)
            case self.REVERSE:
                # Balik posisi
                match self.posisi:
                    case self.LONG:
                        self._tutup_posisi(harga)
                        self._buka_posisi(self.SHORT, harga)
                    case self.SHORT:
                        self._tutup_posisi(harga)
                        self._buka_posisi(self.LONG, harga)

    def _update_pnl_belum_terealisasi(self, harga: float) -> None:
        """
        Update P&L belum terealisasi untuk posisi terbuka.

        Args:
            harga: Harga saat ini
        """
        match self.posisi:
            case self.FLAT:
                self.pnl_belum_terealisasi = 0.0
            case self.LONG:
                self.pnl_belum_terealisasi = (
                    harga - self.harga_masuk
                ) * self.ukuran_posisi
            case self.SHORT:
                self.pnl_belum_terealisasi = (
                    self.harga_masuk - harga
                ) * self.ukuran_posisi

    def _hitung_reward(self, saldo_sebelum: float) -> float:
        """
        Hitung reward untuk timestep ini.

        Reward function:
            reward = delta_ekuitas - (penalti_risiko + volatilitas)

        Args:
            saldo_sebelum: Saldo sebelum aksi dieksekusi

        Returns:
            reward: Nilai reward (bisa positif atau negatif)
        """
        # Delta ekuitas (realized + unrealized P&L)
        ekuitas_sekarang = self.saldo + self.pnl_belum_terealisasi
        ekuitas_sebelum = saldo_sebelum  # Sebelum ada unrealized P&L
        delta_ekuitas = ekuitas_sekarang - ekuitas_sebelum

        # Normalisasi terhadap saldo awal (percentage return)
        delta_ekuitas_persen = delta_ekuitas / self.saldo_awal

        # Penalti risiko berdasarkan volatilitas
        # Gunakan ATR dari data jika ada, atau hitung dari price range
        if "atr_14" in self.data.columns:
            volatilitas = self.data[self.indeks_saat_ini, "atr_14"]
            harga = self._dapatkan_harga_sekarang()
            volatilitas_norm = volatilitas / harga if harga > 0 else 0.0
        else:
            # Fallback: gunakan (high - low) / close
            tinggi = self.data[self.indeks_saat_ini, "tinggi"]
            rendah = self.data[self.indeks_saat_ini, "rendah"]
            tutup = self.data[self.indeks_saat_ini, "tutup"]
            volatilitas_norm = (tinggi - rendah) / tutup if tutup > 0 else 0.0

        # Penalti risiko (penalti jika volatilitas tinggi)
        penalti_risiko = 0.1 * volatilitas_norm if self.posisi != self.FLAT else 0.0

        # Reward akhir
        reward = delta_ekuitas_persen - penalti_risiko

        # Scale reward untuk mempermudah learning
        reward = reward * 100  # Scale up

        return float(reward)

    def _cek_selesai(self) -> bool:
        """
        Cek apakah episode sudah selesai.

        Returns:
            Boolean True jika episode selesai
        """
        # Selesai jika:
        # 1. Sudah mencapai akhir data pada episode
        if self.indeks_saat_ini >= len(self.data) - 1:
            return True

        # 2. Sudah mencapai panjang episode max
        if len(self.histori_aksi) >= self.panjang_episode_max:
            return True

        # 3. Account balance habis (blown account)
        ekuitas_total = self.saldo + self.pnl_belum_terealisasi
        if ekuitas_total <= 0:
            logger.warning(f"Account terlikuidasi sempuran! Ekuitas: {ekuitas_total}")
            return True

        return False

    def reset(self, indeks_mulai: Optional[int] = None) -> np.ndarray:
        """
        Reset lingkungan ke state awal untuk episode baru.

        Args:
            indeks_mulai: Indeks untuk memulai episode (jika None, random)

        Returns:
            observasi_awal: State vector untuk timestamp pertama
        """
        # Titik mulai acak (hindari terlalu dekat dengan akhir data)
        if indeks_mulai is None:
            max_start = len(self.data) - self.panjang_episode_max - 1
            self.indeks_saat_ini = np.random.randint(0, max(1, max_start))
        else:
            self.indeks_saat_ini = indeks_mulai

        # Reset account state
        self.saldo = self.saldo_awal
        self.posisi = self.FLAT
        self.harga_masuk = 0.0
        self.ukuran_posisi = 0.0
        self.pnl_terealisasi = 0.0
        self.pnl_belum_terealisasi = 0.0

        # Reset histori
        self.histori_saldo = [self.saldo]
        self.histori_aksi = []
        self.histori_reward = []

        return self._dapatkan_observasi()

    def langkah(self, aksi: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Eksekusi satu timestep dalam environment.

        Args:
            aksi: Integer 0-4 (HOLD, BUY, SELL, CLOSE, REVERSE)

        Returns:
            observasi: Vektor keadaan (state vector) untuk timestep berikutnya
            reward: Reward yang didapat dari aksi ini
            selesai: Boolean apakah episode selesai
            info: Dictionary dengan informasi tambahan
        """
        # Validasi aksi
        if aksi not in range(self.dim_aksi):
            raise ValueError(f"Aksi tidak valid: {aksi}. Harus 0-4.")

        # Simpan state sebelum aksi
        saldo_sebelum = self.saldo
        harga_sekarang = self._dapatkan_harga_sekarang()

        # Eksekusi state sebelum aksi
        self._eksekusi_aksi(aksi, harga_sekarang)

        # Update P&L belum terealisasi
        self._update_pnl_belum_terealisasi(harga_sekarang)

        # Hitung reward
        reward = self._hitung_reward(saldo_sebelum)

        # Pindah ke timestep berikutnya
        self.indeks_saat_ini += 1

        # Cek apakah episode selesai
        selesai = self._cek_selesai()

        # Update histori
        self.histori_saldo.append(self.saldo + self.pnl_belum_terealisasi)
        self.histori_aksi.append(aksi)
        self.histori_reward.append(reward)

        # Info tambahan
        info = {
            "indeks": self.indeks_saat_ini,
            "harga": harga_sekarang,
            "posisi": self.posisi,
            "saldo": self.saldo,
            "pnl_belum_terealisasi": self.pnl_belum_terealisasi,
            "ekuitas_total": self.saldo + self.pnl_belum_terealisasi,
            "aksi": self._nama_aksi(aksi),
        }

        return self._dapatkan_observasi(), reward, selesai, info

    def dapatkan_statistik_episode(self) -> Dict:
        """
        Dapatkan statistik untuk episode yang baru selesai.

        Returns:
            Dictionary dengan metrics episode
        """
        total_ekuitas = self.saldo + self.pnl_belum_terealisasi
        total_hasil = (total_ekuitas - self.saldo_awal) / self.saldo_awal

        stats = {
            "saldo_awal": self.saldo_awal,
            "saldo_akhir": self.saldo,
            "total_ekuitas": total_ekuitas,
            "pnl_terealisasi": self.pnl_terealisasi,
            "pnl_belum_terealisasi": self.pnl_belum_terealisasi,
            "total_hasil": total_hasil,
            "total_hasil_persen": total_hasil * 100,
            "jumlah_langkah": len(self.histori_aksi),
            "total_reward": sum(self.histori_reward),
            "avg_reward": np.mean(self.histori_reward) if self.histori_reward else 0.0,
        }

        # Hitung Sharpe ratio (versi sederhana)
        if len(self.histori_reward) > 1:
            pendapatan = np.diff(self.histori_saldo) / np.array(self.histori_saldo[:-1])
            pendapatan = pendapatan[~np.isnan(pendapatan)]  # Remove NaN
            if len(pendapatan) > 0 and np.std(pendapatan) > 0:
                sharpe = (
                    np.mean(pendapatan) / np.std(pendapatan) * np.sqrt(252)
                )  # Anualisasi
                stats["sharpe_ratio"] = sharpe
            else:
                stats["sharpe_ratio"] = 0.0
        else:
            stats["sharpe_ratio"] = 0.0

        return stats

    def render(self, mode: str = "debug") -> None:
        """
        Render keadaan (state) saat ini (untuk debugging).

        Args:
            mode: 'debugging' untuk print ke terminal
        """
        if mode == "debug":
            harga = self._dapatkan_harga_sekarang()
            ekuitas = self.saldo + self.pnl_belum_terealisasi

            print(f"\n{"="*60}")
            print(f"Langkah: {self.indeks_saat_ini} | Harga: ${harga:.2f}")
            print(f"Posisi: {self._nama_posisi()} | Entry: ${self.harga_masuk:.2f}")
            print(f"Saldo: ${self.saldo:.2f} | Ekuitas: ${ekuitas:.2f}")
            print(f"P&L Unrealized: ${self.pnl_belum_terealisasi:.2f}")
            print(
                f"Total Return: {((ekuitas - self.saldo_awal) / self.saldo_awal * 100):.2f}%"
            )


# =============================================================================
# FUNGSI UTILITAS
# =============================================================================


def muat_data_untuk_training(
    path_fitur: str,
    simbol: str = "BTCUSDT",
    timeframe: str = "15m",
    rasio_train: float = 0.6,
    rasio_validasi: float = 0.2,
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Muat data dan split menjadi train/val/test.

    Args:
        path_fitur: Path ke direktori data fitur
        simbol: Simbol pasangan crypto
        timeframe: Timeframe data
        rasio_train: Rasio data untuk training
        rasio_validasi: Rasio data untuk validation

    Returns:
        df_train, df_val, df_test
    """
    # Muat data
    file_path = Path(path_fitur) / f"{simbol}_{timeframe}_fitur.parquet"

    if not file_path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {file_path}")

    df = pl.read_parquet(file_path)

    # Sort by datetime jika ada
    if "datetime" in df.columns:
        df = df.sort("datetime")

    # Split data
    n = len(df)
    train_akhir = int(n * rasio_train)
    validation_akhir = int(n * (rasio_train + rasio_validasi))

    df_train = df.slice(0, train_akhir)
    df_val = df.slice(train_akhir, validation_akhir - train_akhir)
    df_test = df.slice(validation_akhir, n - validation_akhir)

    logger.info(f"Data split untuk {simbol}:")
    logger.info(f"  Train:  {len(df_train)} baris ({rasio_train*100:.0f}%)")
    logger.info(f"  Val:    {len(df_val)} baris ({rasio_validasi*100:.0f}%)")
    logger.info(
        f"  Test:   {len(df_test)} baris ({(1-rasio_train-rasio_validasi)*100:.0f}%)"
    )

    return df_train, df_val, df_test


def test_environment():
    """Test environment dengan random actions."""
    logger.info("Testing LingkunganTrading dengan aksi acak ...")

    # Muat data
    df_train, df_val, df_test = muat_data_untuk_training(
        path_fitur="data/fitur", simbol="BTCUSDT", timeframe="15m"
    )

    logger.info(
        f"Data types - Train: {type(df_train)}, Val: {type(df_val)}, Test: {type(df_test)}"
    )

    # Buat environment
    env = LingkunganTrading(
        data=df_train,
        saldo_awal=10000.0,
        panjang_episode_max=100,  # Episode singkat untuk percobaan
    )

    # Jalankan episdoe acak
    observasi = env.reset()
    print(f"\nBentuk observasi: {observasi.shape}")
    print(f"Dim observasi: {env.dim_observasi}")
    print(f"Contoh observasi (10 pertama): \n{observasi[:10]}")

    total_hasil = 0
    for langkah in range(100):
        # Aksi acak
        aksi = np.random.randint(0, env.dim_aksi)
        observasi, hasil, selesai = env.langkah(aksi)
        total_hasil += hasil

        if langkah % 20 == 0:
            env.render()

        if selesai:
            break

    # Print stats
    stats = env.dapatkan_statistik_episode()
    print("\nStatistik Episode:")
    for kunci, nilai in stats.items():
        if isinstance(nilai, float):
            print(f"    {kunci}: {nilai:.4f}")
        else:
            print(f"    {kunci}: {nilai}")

    logger.info("Test selesai! ✅")


if __name__ == "__main__":
    test_environment()
