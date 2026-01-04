"""
Experience Replay Buffer untuk DQN

Buffer menyimpan pengalaman (state, action, reward, next_state, done) dan
menyediakan random sampling untuk training.

Menggunakan numpy arryas untuk efisiensi memori dan kecepatan.
"""

import numpy as np
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BufferPengalaman:
    """
    Buffer untuk menyimpan dan sample pengalaman trading agen.
    """

    def __init__(self, kapasitas_max: int = 100000, dim_state: int = 74):
        """
        Inisialisasi buffer pengalaman

        Args:
            kapasitas_max: Maksimal jumlah pengalaman yang disimpan
            dim_state: Dimensi vektor state (jumlah fitur)
        """
        self.kapasitas_max = kapasitas_max
        self.dim_state = dim_state
        self.posisi = 0  # Posisi tulis saat ini
        self.penuh = False  # Flag yang akan mengindikasikan apakah buffer sudah penuh

        # Pre-alokasi numpy arrays memori untuk efisiensi read dan write
        self.states = np.zeros((kapasitas_max, dim_state), dtype=np.float32)
        self.aksi = np.zeros(kapasitas_max, dtype=np.int32)
        self.reward = np.zeros(kapasitas_max, dtype=np.float32)
        self.next_states = np.zeros((kapasitas_max, dim_state), dtype=np.float32)
        self.selesai = np.zeros(kapasitas_max, dtype=np.bool)

        logger.info("BufferPengalaman di-inisialisasi:")
        logger.info(f"  - Kapasitas: {kapasitas_max}")
        logger.info(f"  - State dim: {dim_state}")
        logger.info(f"  - Memory: ~{self._estimasi_memori():.2f} MB")

    def _estimasi_memori(self) -> float:
        """
        Estimasi penggunaan memori dalam MB

        Returns:
            Estimasi memori dalam MB
        """
        # Hitung bytes setiap array
        bytes_states = self.states.nbytes
        bytes_aksi = self.states.nbytes
        bytes_reward = self.reward.nbytes
        bytes_next_states = self.next_states.nbytes
        bytes_selesai = self.selesai.nbytes

        total_bytes = (
            bytes_states + bytes_aksi + bytes_reward + bytes_next_states + bytes_selesai
        )

        return total_bytes / (1024 * 1024)  # Konversi ke MB

    def simpan(
        self,
        state: np.ndarray,
        aksi: int,
        reward: float,
        next_state: np.ndarray,
        selesai: bool,
    ) -> None:
        """
        Simpan satu set pengalaman ke buffer.

        Args:
            state: State saat ini (array shape: dim_state)
            aksi: Aksi yang diambil (0-4)
            reward: Reward yang didapat
            next_state: State berikutnya (array_shape: dim_state)
            selesai: Boolean apakah episode selesai
        """
        # Simpan ke posisi saat ini (circular buffer)
        self.states[self.posisi] = state
        self.aksi[self.posisi] = aksi
        self.reward[self.posisi] = reward
        self.next_states[self.posisi] = next_state
        self.selesai[self.posisi] = selesai

        # Update posisi (circular)
        self.posisi = (self.posisi + 1) % self.kapasitas_max

        # Cek apakah buffer sudah penuh
        if self.posisi == 0:
            self.penuh = True

    def sample(self, batch_size: int = 64) -> Tuple[np.ndarray, ...]:
        """
        Sample random batch dari buffer untuk training.

        Args:
            batch_size: Jumlah sample yang diambil

        Returns:
            Tuple dari states, aksi, reward, next_states dan selesai
        """
        # Tentukan range valid untuk sampling
        if self.penuh:
            max_idx = self.kapasitas_max
        else:
            max_idx = self.posisi

        # Random sampling
        indices = np.random.randint(0, max_idx, size=batch_size)

        # Return batch sample
        return (
            self.states[indices],
            self.aksi[indices],
            self.reward[indices],
            self.next_states[indices],
            self.selesai[indices],
        )

    def __len__(self) -> int:
        """Return jumlah pengalaman yang tersimpan sekarang."""
        if self.penuh:
            return self.kapasitas_max
        else:
            return self.posisi

    def kosongkan(self) -> None:
        """Reset buffer (untuk testing atau re-training)."""
        self.posisi = 0
        self.penuh = False
        self.states.fill(0)
        self.aksi.fill(0)
        self.reward.fill(0)
        self.next_states.fill(0)
        self.selesai.fill(0)
        logger.info("Buffer dikosongkan")

    def dapatkan_statistik(self) -> dict:
        """
        Dapatkan statistik buffer untuk monitoring.

        Returns:
            Dictionary dengan statistik buffer
        """
        ukuran = len(self)

        if ukuran == 0:
            return {
                "ukuran": 0,
                "kapasitas": self.kapasitas_max,
                "penggunaan_persen": 0.0,
                "avg_reward": 0.0,
                "min_reward": 0.0,
                "max_reward": 0.0,
            }

        # Ambil data yang valid
        if self.penuh:
            reward_valid = self.reward
        else:
            reward_valid = self.reward[: self.posisi]

        return {
            "ukuran": ukuran,
            "kapasitas": self.kapasitas_max,
            "penggunaan_persen": (ukuran / self.kapasitas_max) * 100,
            "avg_reward": float(np.mean(reward_valid)),
            "min_reward": float(np.min(reward_valid)),
            "max_reward": float(np.max(reward_valid)),
            "std_reward": float(np.std(reward_valid)),
        }


# =====================================================
# TESTING
# =====================================================


def test_buffer():
    """Test buffer pengalaman."""
    logger.info("Testing BufferPengalaman...")

    # Buat buffer
    dim_state = 74  # 70 fitur + 4 info posisi
    buffer = BufferPengalaman(kapasitas_max=1000, dim_state=dim_state)

    print(f"\n{'='*60}")
    print("TEST 1: Simpan pengalaman")
    print("=" * 60)

    # Simulasi 500 pengalaman
    for i in range(500):
        state = np.random.randn(dim_state).astype(np.float32)
        aksi = np.random.randint(0, 5)
        reward = np.random.randn() * 10  # Random reward
        next_state = np.random.randn(dim_state).astype(np.float32)
        selesai = i % 100 == 0  # Setiap 100 step = episode selesai

        buffer.simpan(state, aksi, reward, next_state, selesai)

    print(f"Buffer size: {len(buffer)}")
    print(f"Buffer penuh: {buffer.penuh}")

    stats = buffer.dapatkan_statistik()
    print("\nStatistik buffer:")
    for kunci, nilai in stats.items():
        if isinstance(nilai, float) and kunci != "penggunaan_persen":
            print(f"    {kunci}: {nilai:.4f}")
        elif isinstance(nilai, float):
            print(f"    {kunci}: {nilai:.2f}%")
        else:
            print(f"    {kunci}: {nilai}")

    # Test sampling
    print(f"\n{'='*60}")
    print("TEST 2: Sample random batch")
    print(f"{'='*60}")

    batch_size = 64
    states, aksi, reward, next_states, selesai = buffer.sample(batch_size)

    print(f"Dimensi states: {states.shape}")
    print(f"Dimensi aksi: {aksi.shape}")
    print(f"Dimensi reward: {reward.shape}")
    print(f"Dimensi next states: {next_states.shape}")
    print(f"Dimensi selesai: {selesai.shape}")

    # Test circular buffer (simpan lebih dari kapasitas)
    print(f"\n{'='*60}")
    print("TEST 3: Circular buffer (overwrite)")
    print(f"{'='*60}")

    # Simpan 600 lagi (total 1100, dengan kapasitas max 1000)
    for i in range(600):
        state = np.random.randn(dim_state).astype(np.float32)
        aksi = np.random.randint(0, 5)
        reward = np.random.randn() * 10
        next_state = np.random.randn(dim_state).astype(np.float32)
        selesai = False

        buffer.simpan(state, aksi, reward, next_state, selesai)

    print(f"Buffer size setelah 1100 penyimpanan: {len(buffer)}")
    print(f"Buffer penuh: {buffer.penuh}")
    print("Expected: 1000 (circular overwrite)")

    stats = buffer.dapatkan_statistik()
    print("\nStatistik buffer final:")
    for kunci, nilai in stats.items():
        if isinstance(nilai, float) and kunci != "penggunaan_persen":
            print(f"    {kunci}: {nilai:.4f}")
        elif isinstance(nilai, float):
            print(f"    {kunci}: {nilai:.2f}%")
        else:
            print(f"    {kunci}: {nilai}")

    # Test kosongkan
    print(f"\n{'='*60}")
    print("TEST 4: Kosongkan buffer")
    print(f"{'='*60}")

    buffer.kosongkan()
    print(f"Buffer size setelah dikosongkan: {len(buffer)}")
    print(f"Buffer penuh: {buffer.penuh}")

    logger.info("Test selesai! ✅")
    print(f"\n{'='*60}")
    print("SEMUA TEST PASSED! 🚀")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    test_buffer()
