"""
Script untuk pelatihan Agen DQN.

Script utama untuk melatih agen DQN pada data historis cryptocurrency.
Menggabungkan lingkungan, agen dan loop pelatihan.
"""

import sys
from pathlib import Path

# Tambahkan direktori src ke Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, List
import logging
import json

# Import komponen
from lingkungan.lingkungan_trading import LingkunganTrading, muat_data_untuk_training
from agen_dqn import AgenDQN
from gpu import setup_gpu, print_info_gpu, monitor_memori_gpu

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PelatihDQN:
    """
    Pelatih untuk agen DQN.

    Mengelola:
    - Loop pelatihan
    - Validasi
    - Checkpoint
    - Metriks log
    """

    def __init__(
        self,
        path_konfigurasi: str = "konfigurasi/konfigurasi.yaml",
        direktori_output: str = "model/dqn",
    ):
        """
        Inisialisasi pelatih.

        Args:
            path_konfigurasi: Path ke file konfigurasi YAML
            direktori_output: Direktori untuk menyimpan model dan logs
        """
        self.path_konfigurasi = path_konfigurasi
        self.direktori_output = Path(direktori_output)
        self.direktori_output.mkdir(parents=True, exist_ok=True)

        # Muat konfigurasi
        with open(path_konfigurasi, "r") as f:
            self.konfig = yaml.safe_load(f)

        # Histori untuk training
        self.histori_pelatihan: Dict = {
            "episode": [],
            "reward": [],
            "return": [],
            "epsilon": [],
            "loss": [],
            "langkah": [],
        }

        self.histori_validasi = {
            "episode": [],
            "reward": [],
            "return": [],
            "sharpe": [],
        }

        logger.info("Inisialisasi PelatihDQN")
        logger.info(f"  - Konfigurasi: {path_konfigurasi}")
        logger.info(f"  - Output: {direktori_output}")

    def siapkan_data(self, simbol: str = "BTCUSDT") -> tuple:
        """
        Siapkan data pelatihan, validasi dan tes

        Args:
            simbol: Pasangan koin cryptocurrency

        Returns:
            (df_latih, df_val, df_tes)
        """
        logger.info(f"Memuat data untuk {simbol}...")

        df_latih, df_val, df_tes = muat_data_untuk_training(
            path_fitur=self.konfig["data"]["direktori_data_fitur"],
            simbol=simbol,
            timeframe=self.konfig["pelatihan"]["timeframe_keputusan"],
            rasio_pelatihan=0.6,
            rasio_validasi=0.2,
        )

        logger.info("Data siap:")
        logger.info(f"  - Latih: {len(df_latih):,} candles")
        logger.info(f"  - Validasi: {len(df_val):,} candles")
        logger.info(f"  - Tes: {len(df_tes):,} candles")

        return df_latih, df_val, df_tes

    def buat_lingkungan(self, data) -> LingkunganTrading:
        """
        Buat lingkungan trading.

        Args:
            data: Polars DataFrame dengan data market

        Returns:
            LingkunganTrading instance
        """
        return LingkunganTrading(
            data=data,
            saldo_awal=self.konfig["lingkungan"]["saldo_awal"],
            biaya_trading=self.konfig["lingkungan"]["biaya_trading"],
            leverage=self.konfig["lingkungan"]["leverage"],
            panjang_episode_max=1500,  # 15 hari untuk 15 timeframe
        )

    def buat_agen(self, dim_state: int, use_gpu: bool = False) -> AgenDQN:
        """
        Buat agen DQN.

        Args:
            dim_state: Dimensi state vektor
            use_gpu: Apakah menggunakan optimisasi GPU

        Returns:
            AgenDQN instance
        """
        return AgenDQN(
            dim_state=dim_state,
            dim_aksi=5,
            hidden_layers=self.konfig["agen"]["hidden_layers"],
            learning_rate=self.konfig["agen"]["learning_rate"],
            gamma=self.konfig["agen"]["gamma"],
            epsilon_mulai=self.konfig["agen"]["epsilon_mulai"],
            epsilon_akhir=self.konfig["agen"]["epsilon_akhir"],
            epsilon_decay=self.konfig["agen"]["epsilon_decay"],
            ukuran_buffer=self.konfig["agen"]["ukuran_buffer"],
            ukuran_batch=self.konfig["agen"]["ukuran_batch"],
            frekuensi_upd_target=self.konfig["agen"]["frekuensi_upd_target"],
            use_mixed_precision=use_gpu,  # Enable mixed precision jika ada GPU
        )

    def jalankan_episode(
        self, lingkungan: LingkunganTrading, agen: AgenDQN, pelatihan: bool = True
    ) -> Dict:
        """
        Jalankan satu episode pelatihan.

        Args:
            lingkungan: Lingkungan trading
            agen: Agen DQN
            pelatihan: Jika True, agen akan melakukan pelatihan

        Returns:
            Dictionary dengan episode metriks
        """
        state = lingkungan.reset()
        episode_reward = 0
        episode_loss = []
        langkah = 0

        while True:
            # Pilih aksi
            aksi = agen.pilih_aksi(state, pelatihan=pelatihan)

            # Eksekusi aksi
            next_state, reward, selesai, info = lingkungan.langkah(aksi)

            episode_reward += reward
            langkah += 1

            if pelatihan:
                # Simpan pengalaman
                agen.simpan_pengalaman(state, aksi, reward, next_state, selesai)

                # Langkah pelatihan
                loss = agen.latih()
                if loss is not None:
                    episode_loss.append(loss)

            state = next_state

            if selesai:
                break

        # Mendapatkan statistik episode
        statistik = lingkungan.dapatkan_statistik_episode()
        statistik["episode_reward"] = episode_reward
        statistik["langkah"] = langkah

        if pelatihan and episode_loss:
            statistik["avg_loss"] = np.mean(episode_loss)
        else:
            statistik["avg_loss"] = 0.0

        return statistik

    def latih(
        self,
        jumlah_episode: int = 10000,
        frekuensi_validasi: int = 100,
        frekuensi_checkpoint: int = 500,
        simbol: str = "BTCUSDT",
    ) -> None:
        """
        Latih agen DQN.

        Args:
            jumlah_episode: Total episode pelatihan
            frekuensi_validasi: Frekuensi validasi (setiap N episode)
            frekuensi_checkpoint: Frekuensi simpan checkpoint (setiap N episode)
            simbol: Simbol cryptocurrency untuk training
        """
        logger.info(f"\n{'='*80}")
        logger.info("MEMULAI PELATIHAN AGEN DQN")
        logger.info(f"{'='*80}")
        logger.info(f"Simbol: {simbol}")
        logger.info(f"Total episode: {jumlah_episode}")
        logger.info(f"Validasi setiap: {frekuensi_validasi} episode")
        logger.info(f"Checkpoint setiap {frekuensi_checkpoint} episode")

        # Setup GPU
        logger.info(f"\n{'='*80}")
        logger.info("SETUP GPU")
        logger.info(f"{'='*80}")
        print_info_gpu()
        gpu_tersedia = setup_gpu(dinamis=True)
        if gpu_tersedia:
            logger.info("✅ GPU akan digunakan untuk pelatihan")
        else:
            logger.info("⚠️ Pelatihan menggunakan CPU")

        # Siapkan data
        df_latih, df_val, df_tes = self.siapkan_data(simbol)

        # Buat lingkungan
        lingkungan_latih = self.buat_lingkungan(df_latih)
        lingkungan_validasi = self.buat_lingkungan(df_val)

        # Buat agen dengan optimisasi GPU jika tersedia
        agen = self.buat_agen(
            dim_state=lingkungan_latih.dim_observasi, use_gpu=gpu_tersedia
        )

        # Loop pelatihan
        logger.info(f"\n{'='*80}")
        logger.info("PELATIHAN DIMULAI")
        logger.info(f"{'='*80}")

        hasil_terbaik_val = -np.inf

        for episode in range(1, jumlah_episode + 1):
            # Episode pelatihan
            statistik = self.jalankan_episode(lingkungan_latih, agen, pelatihan=True)

            # Decay epsilon
            agen.decay_epsilon()

            # Metriks log
            self.histori_pelatihan["episode"].append(episode)
            self.histori_pelatihan["reward"].append(statistik["episode_reward"])
            self.histori_pelatihan["return"].append(statistik["total_hasil"])
            self.histori_pelatihan["epsilon"].append(agen.epsilon)
            self.histori_pelatihan["loss"].append(statistik["avg_loss"])
            self.histori_pelatihan["langkah"].append(statistik["langkah"])

            # Print progress
            if episode % 10 == 0:
                logger.info(
                    f"Episode: {episode:5d} | "
                    f"Reward: {statistik["episode_reward"]:8.2f} | "
                    f"Return: {statistik["total_hasil_persen"]:6.2f}% | "
                    f"Epsilon: {agen.epsilon:.4f} | "
                    f"Loss: {statistik["avg_loss"]:.6f} | "
                    f"Langkah: {statistik["langkah"]:4d}"
                )

            # Validasi
            if episode % frekuensi_validasi == 0:
                logger.info(f"\n{'='*80}")
                logger.info(f"VALIDASI - Episode {episode}")
                logger.info(f"{'='*80}")

                # Monitor memori GPU
                if gpu_tersedia:
                    info_memori = monitor_memori_gpu()
                    if info_memori:
                        logger.info(
                            f"Memori GPU - Saat ini: {info_memori['mb_saat_ini']:.0f} MB, Tertinggi: {info_memori['mb_tertinggi']:.0f} MB"
                        )

                statistik_validasi = self.jalankan_episode(
                    lingkungan_validasi, agen, pelatihan=False
                )

                self.histori_validasi["episode"].append(episode)
                self.histori_validasi["reward"].append(
                    statistik_validasi["episode_reward"]
                )
                self.histori_validasi["return"].append(
                    statistik_validasi["total_hasil"]
                )
                self.histori_validasi["sharpe"].append(
                    statistik_validasi["sharpe_ratio"]
                )

                logger.info("Hasil Validasi:")
                logger.info(f"  - Reward: {statistik_validasi['episode_reward']:.2f}")
                logger.info(
                    f"  - Hasil: {statistik_validasi['total_hasil_persen']:.2f}%"
                )
                logger.info(f"  - Sharpe: {statistik_validasi['sharpe_ratio']:.4f}")
                logger.info(f"{'='*80}\n")

                # Simpan model terbaik
                if statistik_validasi["total_hasil"] > hasil_terbaik_val:
                    hasil_terbaik_val = statistik_validasi["total_hasil"]
                    agen.simpan_model(self.direktori_output / "best_model")
                    logger.info(
                        f"✅ Model terbaik disimpan! Return: {hasil_terbaik_val*100:.2f}%"
                    )

            # Checkpoint
            if episode % frekuensi_checkpoint == 0:
                path_checkpoint = self.direktori_output / f"checkpoint_{episode}"
                agen.simpan_model(path_checkpoint)
                self.simpan_histori()
                logger.info(f"💾 Checkpoint disimpan: {path_checkpoint}")

        # Pelatihan selesai
        logger.info(f"\n{'='*80}")
        logger.info("PELATIHAN SELESAI")
        logger.info(f"{'='*80}")
        logger.info(f"Total episode: {jumlah_episode}")
        logger.info(f"Hasil validasi terbaik: {hasil_terbaik_val*100:.2f}%")

        # Simpan final model
        agen.simpan_model(self.direktori_output / "final_model")
        self.simpan_histori()

        logger.info(f"\nModel dan histori disimpan di: {self.direktori_output}")

    def simpan_histori(self) -> None:
        """Simpan histori pelatihan ke JSON."""
        path_histori = self.direktori_output / "history.json"

        histori = {
            "pelatihan": self.histori_pelatihan,
            "validasi": self.histori_validasi,
        }

        with open(path_histori, "w") as f:
            json.dump(histori, f, indent=2)

        logger.debug(f"Histori disimpan ke {path_histori}")

    def muat_histori(self) -> None:
        """Muat histori pelatihan dan validasi dari JSON."""
        path_histori = self.direktori_output / "histori.json"

        if path_histori.exists():
            with open(path_histori, "r") as f:
                histori = json.load(f)

            self.histori_pelatihan = histori["pelatihan"]
            self.histori_validasi = histori["validasi"]

            logger.info(f"Histori dimuat dari {path_histori}")


# =============================================================================
# FUNGSI MAIN PELATIHAN
# =============================================================================
def main():
    """Fungsi utama untuk menjalankan pelatihan."""
    import argparse

    parser = argparse.ArgumentParser(description="Latih Agen Trading DQN")
    parser.add_argument(
        "--konfigurasi",
        default="konfigurasi/konfigurasi.yaml",
        help="Path ke file konfigurasi",
    )
    parser.add_argument("--simbol", default="BTCUSDT", help="Simbol cryptocurrency")
    parser.add_argument(
        "--episode", type=int, default=10000, help="Jumlah episode pelatihan"
    )
    parser.add_argument("--output", default="model/dqn", help="Direktori output model")

    args = parser.parse_args()

    # Buat pelatih
    pelatih = PelatihDQN(
        path_konfigurasi=args.konfigurasi, direktori_output=args.output
    )

    # Mulai pelatihan
    pelatih.latih(
        jumlah_episode=args.episode,
        frekuensi_validasi=100,
        frekuensi_checkpoint=500,
        simbol=args.simbol,
    )

    logger.info("\n✅ Pelatihan selesai! Model siap untuk dievaluasi.")


if __name__ == "__main__":
    main()
