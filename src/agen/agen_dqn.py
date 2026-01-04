"""
Agen DQN untuk Trading

Agen yang mengimplementasikan algoritma Deep Q-Network untuk trading.
Menggabungkan Q-Network, Target Network, Experience Replay dan eksplorasi Epsilon-Greedy.
"""

import numpy as np
from typing import Tuple, Optional
import logging
from pathlib import Path

# Import komponen DQN
from jaringan_q import JaringanQ
from buffer_pengalaman import BufferPengalaman

# from src.agen import jaringan_q

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgenDQN:
    """
    Agen Deep Q-Network untuk cryptocurrency futures trading.

    Komponen utama:
    1. Q-Network (utama): Memprediksi nilai Q dan di-update setiap step
    2. Target Network: Menyalin dari JaringanQ utama, di-update lebih lambat untuk stabilisasi
    3. Experience Replay: Menyimpan dan sampling pengalaman untuk training
    4. Epsilon-Greedy: Strategi eksplorasi dan eksploitasi

    Algoritma DQN:
        1. Amati state s
        2. Pilih aksi a (epsilon-greedy)
        3. Eksekusi aksi, mendapatkan reward r dan state selanjutnya s'
        4. Simpan pengalaman (s, a, r, s', selesai) dalam buffer replay
        5. Sample batch acak dari buffer
        6. Hitung target: r + gamma + max_a' Q_target(s', a')
        7. Update Q-network untuk minimisasi: (Q(s,a) - target)^2
        8. Secara periodik meng-update jaringan target
    """

    def __init__(
        self,
        dim_state: int = 74,
        dim_aksi: int = 5,
        hidden_layers: list = [256, 128, 64],
        learning_rate: float = 0.0001,
        gamma: float = 0.99,
        epsilon_mulai: float = 1.0,
        epsilon_akhir: float = 0.01,
        epsilon_decay: float = 0.995,
        ukuran_buffer: int = 100000,
        ukuran_batch: int = 64,
        frekuensi_upd_target: int = 100,
        use_mixed_precision: bool = False,
    ):
        """
        Inisialisasi Agen DQN.

        Args:
            dim_state: Dimensi vektor state
            dim_aksi: Jumlah aksi (5: HOLD, BUY, SELL, CLOSE, REVERSE)
            hidden_layers: Arsitektur hidden layers
            learning_rate: Learning rate untuk optimizer
            gamma: Discount faktor untuk future rewards
            epsilon_mulai: Epsilon awal untuk eksplorasi
            epsilon_akhir: Epsilon minimum (eksploitasi)
            epsilon_decay: Rate penurunan nilai epsilon
            ukuran_buffer: Ukuran replay buffer
            ukuran_batch: Ukuran batch dalam pelatihan
            frekuensi_upd_target: Frekuensi update jaringan target
            use_mixed_precision: Gunakan mixed precision (FP16) untuk GPU speedup
        """
        self.dim_state = dim_state
        self.dim_aksi = dim_aksi
        self.gamma = gamma
        self.epsilon = epsilon_mulai
        self.epsilon_akhir = epsilon_akhir
        self.epsilon_decay = epsilon_decay
        self.ukuran_batch = ukuran_batch
        self.frekuensi_upd_target = frekuensi_upd_target

        # Penghitung untuk tracking
        self.jumlah_training = 0
        self.jumlah_aksi = 0

        # Jaringan Q (Jaringan utama yang di-latih)
        self.jaringan_q = JaringanQ(
            dim_state=dim_state,
            dim_aksi=dim_aksi,
            hidden_layers=hidden_layers,
            learning_rate=learning_rate,
            use_mixed_precision=use_mixed_precision,
        )

        # Target Network (untuk stabilitas training)
        self.jaringan_target = JaringanQ(
            dim_state=dim_state,
            dim_aksi=dim_aksi,
            hidden_layers=hidden_layers,
            learning_rate=learning_rate,
            use_mixed_precision=use_mixed_precision,
        )

        # Salin bobot dari jaringan_q ke jaringan_target
        self.jaringan_target.salin_bobot_dari(self.jaringan_q)

        # Buffer pengalaman
        self.buffer = BufferPengalaman(kapasitas_max=ukuran_buffer, dim_state=dim_state)

        logger.info(f"AgenDQN di-inisialisasi:")
        logger.info(f"  - Dimensi State: {dim_state}")
        logger.info(f"  - Dimensi Aksi: {dim_aksi}")
        logger.info(f"  - Gamma: {gamma}")
        logger.info(
            f"  - Epsilon: {epsilon_mulai} → {epsilon_akhir} (decay: {epsilon_decay})"
        )
        logger.info(f"  - UKuran Buffer: {ukuran_buffer}")
        logger.info(f"  - Ukuran Batch: {ukuran_batch}")
        logger.info(f"  - Frek. Upd. Target: {frekuensi_upd_target}")

    def pilih_aksi(self, state: np.ndarray, pelatihan: bool = True) -> int:
        """
        Pilih aksi menggunakan strategi epsilon-greedy.

        Args:
            state: State saat ini
            pelatihan: Jika True, gunakan epsilon-greedy decay. Jika False, selalu greedy.

        Returns:
            Aksi yang dipilih (0-4)
        """
        self.jumlah_aksi += 1

        # Eksplorasi (aksi acak)
        if pelatihan and np.random.random() < self.epsilon:
            return np.random.randint(0, self.dim_aksi)

        # Eksploitasi (aksi terbaik berdasar nilai Q)
        nilai_q = self.jaringan_q.prediksi(state)
        return int(np.argmax(nilai_q))

    def simpan_pengalaman(
        self,
        state: np.ndarray,
        aksi: int,
        reward: float,
        state_selanjutnya: np.ndarray,
        selesai: bool,
    ) -> None:
        """
        Simpan pengalaman ke buffer pengalaman.

        Args:
            state: State saat ini
            aksi: Aksi yang diambil
            reward: Reward yang didapat
            state_selanjutnya: State selanjutnya
            selesai: Boolean mengenai apakah episode selesai
        """
        self.buffer.simpan(state, aksi, reward, state_selanjutnya, selesai)

    def latih(self) -> Optional[float]:
        """
        Latih agen dengan sample batch dari replay buffer.

        Returns:
            Loss jika training dilakukan, None jika buffer belum cukup
        """
        # Cek apakah buffer sudah cukup untuk training
        if len(self.buffer) < self.ukuran_batch:
            return None

        # Sample batch dari buffer
        states, aksi, reward, next_states, selesai = self.buffer.sample(
            self.ukuran_batch
        )

        # Hitung target nilai Q
        # Target = r + gamma + max_a' Q_targets(s', a') untuk non-terminal states
        # Target = r untuk terminal states

        # Prediksi nilai Q untuk next states menggunakan target network
        next_q_values = self.jaringan_target.prediksi(
            next_states
        )  # (ukuran_batch, dim_aksi)
        max_next_q = np.max(next_q_values, axis=1)  # (ukuran_batch,)

        # Hitung nilai Q target
        target_q = reward + self.gamma * max_next_q * (1 - selesai)

        # Nilai Q saat ini untuk semua aksi
        nilai_q_saat_ini = self.jaringan_q.prediksi(states)  # (ukuran_batch, dim_aksi)

        # Update hanya nilai Q untuk aksi yang diambil
        nilai_target_q = nilai_q_saat_ini.copy()
        for i in range(self.ukuran_batch):
            nilai_target_q[i, aksi[i]] = target_q[i]

        # Latih jaringan Q
        loss = self.jaringan_q.latih_batch(states, nilai_target_q)

        # Update counter
        self.jumlah_training += 1

        # Update target network secara periodik
        if self.jumlah_training % self.frekuensi_upd_target == 0:
            self.jaringan_target.salin_bobot_dari(self.jaringan_q)
            logger.debug(f"Jaringan target diupdate pada step {self.jumlah_training}")

        return loss

    def decay_epsilon(self) -> None:
        """Kurangi epsilon untuk mengurangi eksplorasi secara bertahap."""
        self.epsilon = max(self.epsilon_akhir, self.epsilon * self.epsilon_decay)

    def simpan_model(self, path: str) -> None:
        """
        Simpan jaringan Q dan metadata.

        Args:
            path: Path direktori untuk menyimpan model
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Simpan jaringan Q
        self.jaringan_q.simpan(str(path / "jaringan_q.keras"))

        # Simpen jaringan target
        self.jaringan_target.simpan(str(path / "jaringan_target.keras"))

        # Simpan metadata
        metadata = {
            "dim_state": self.dim_state,
            "dim_aksi": self.dim_aksi,
            "epsilon": self.epsilon,
            "jumlah_training": self.jumlah_training,
            "jumlah_aksi": self.jumlah_aksi,
        }

        import json

        with open(path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Model disimpan ke {path}")

    def muat_model(self, path: str) -> None:
        """
        Muat jaringan Q dan metadata.

        Args:
            path: Path direktori model disimpan
        """
        path = Path(path)

        # Muat jaringan_q
        self.jaringan_q.muat(str(path / "jaringan_q.keras"))

        # Muat jaringan_target
        self.jaringan_target.muat(str(path / "jaringan_target.keras"))

        # Muat metadata
        import json

        with open(path / "metadata.json", "r") as f:
            metadata = json.load(f)

        self.epsilon = metadata.get("epsilon", self.epsilon)
        self.jumlah_training = metadata.get("jumlah_training", 0)
        self.jumlah_aksi = metadata.get("jumlah_aksi", 0)

        logger.info(f"Model dimuat dari {path}")
        logger.info(f"  - Jumlah training: {self.jumlah_training}")
        logger.info(f"  - Jumlah aksi: {self.jumlah_aksi}")
        logger.info(f"  - Epsilon: {self.epsilon:.4f}")

    def dapatkan_statistik(self) -> dict:
        """
        Dapatkan statistik agen untuk monitoring.

        Returns:
            Dictionary dengna statistik agen
        """
        buffer_stats = self.buffer.dapatkan_statistik()

        return {
            "epsilon": self.epsilon,
            "jumlah_training": self.jumlah_training,
            "jumlah_aksi": self.jumlah_aksi,
            "ukuran_buffer": self.buffer,
            "buffer_penuh": self.buffer.penuh,
            **buffer_stats,
        }


# =============================================================================
# TESTING
# =============================================================================


def test_agen():
    """Test Agen DQN."""
    logger.info("Testing AgenDQN...")

    print(f"\n{'='*60}")
    print("TEST 1: Inisialisasi agen")
    print(f"{'='*60}")

    agen = AgenDQN(
        dim_state=74,
        dim_aksi=5,
        hidden_layers=[256, 128, 64],
        learning_rate=0.0001,
        gamma=0.99,
        epsilon_mulai=1.0,
        epsilon_akhir=0.01,
        epsilon_decay=0.995,
        ukuran_buffer=10000,
        ukuran_batch=64,
        frekuensi_upd_target=100,
    )

    stats = agen.dapatkan_statistik()
    print(f"\nStatistik awal:")
    for kunci, nilai in stats.items():
        print(f"    {kunci}:{nilai}")

    # Tes Pemilihan aksi
    print(f"\n{'='*60}")
    print("TEST 2: Pilih aksi (epsilon-greedy)")
    print(f"{'='*60}")

    state = np.random.rand(74).astype(np.float32)

    # Test dengan epsilon tinggi (banyak random)
    agen.epsilon = 1.0
    aksi_random = [agen.pilih_aksi(state, pelatihan=True) for _ in range(10)]
    print(f"Aksi dengan epsilon=1.0 (random): {aksi_random}")

    # Test dengan epsilon rendah (mostly greedy)
    agen.epsilon = 0.0
    aksi_greedy = [agen.pilih_aksi(state, pelatihan=True) for _ in range(10)]
    print(f"Aksi dengan epsilon=0.0 (greedy): {aksi_greedy}")

    # Reset epsilon
    agen.epsilon = 1.0

    # Test save experiences
    print(f"\n{'='*60}")
    print("TEST 3: Simpan pengalaman ke buffer")
    print(f"{'='*60}")

    for i in range(200):
        state = np.random.randn(74).astype(np.float32)
        aksi = np.random.randint(0, 5)
        reward = np.random.randn() * 10
        next_state = np.random.randn(74).astype(np.float32)
        selesai = i % 50 == 0

        agen.simpan_pengalaman(state, aksi, reward, next_state, selesai)

    stats = agen.dapatkan_statistik()
    print(f"\nBuffer size after 200 experiences: {stats['ukuran_buffer']}")

    # Test training
    print(f"\n{'='*60}")
    print("TEST 4: Training agen")
    print(f"{'='*60}")

    losses = []
    for i in range(10):
        loss = agen.latih()
        if loss is not None:
            losses.append(loss)

    print(f"Training iterations: {len(losses)}")
    if losses:
        print(f"Average loss: {np.mean(losses):.6f}")
        print(f"Loss trend: {losses[:5]}")

    # Test epsilon decay
    print(f"\n{'='*60}")
    print("TEST 5: Epsilon decay")
    print(f"{'='*60}")

    epsilons = [agen.epsilon]
    for _ in range(100):
        agen.decay_epsilon()
        epsilons.append(agen.epsilon)

    print(f"Epsilon decay (first 10): {epsilons[:10]}")
    print(f"Epsilon after 100 decays: {epsilons[-1]:.6f}")

    # Test save/load
    print(f"\n{'='*60}")
    print("TEST 6: Save and load model")
    print(f"{'='*60}")

    import tempfile
    import shutil

    # Create temp directory
    temp_dir = tempfile.mkdtemp()

    try:
        # Save
        agen.simpan_model(temp_dir)

        # Get Q-values before
        q_before = agen.jaringan_q.prediksi(state)
        epsilon_before = agen.epsilon

        # Create new agen and load
        agen2 = AgenDQN(dim_state=74, dim_aksi=5)
        agen2.muat_model(temp_dir)

        # Get Q-values after
        q_after = agen2.jaringan_q.prediksi(state)
        epsilon_after = agen2.epsilon

        print(f"Q-values match: {np.allclose(q_before, q_after)}")
        print(f"Epsilon match: {epsilon_before == epsilon_after}")
        print(f"Epsilon loaded: {epsilon_after:.6f}")

    finally:
        # Cleanup
        shutil.rmtree(temp_dir)

    logger.info("Test selesai! ✅")
    print(f"\n{'='*60}")
    print("SEMUA TEST PASSED! 🚀")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    test_agen()
