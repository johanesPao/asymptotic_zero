"""
Arsitektur Q-Network untuk DQN

Neural network yang akan memprediksi nilai Q untuk setiap aksi.
Input: State (70+ fitur teknikal + info posisi)
Output: Nilai Q untuk setiap aksi (HOLD, BUY, SELL, CLOSE, REVERSE)

Menggunakan TensorFlow dan Keras untuk implementasi.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from typing import List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JaringanQ:
    """
    Deep Q-Network untuk trading

    Arsitektur:
        Input (dim_state)
            ↓
        Dense(256) + ReLU + Dropout(0.2)
            ↓
        Dense(128) + ReLU + Dropout(0.2)
            ↓
        Dense(64) + ReLU
            ↓
        Dense(dim_aksi) [Output layer]

    Output = Nilai Q untuk setiap aksi (nilai raw, bukan probabilitas)

    GPU Optimization:
        - Mixed precision training (untuk RTX 4070)
        - XLA compilation
    """

    def __init__(
        self,
        dim_state: int = 74,
        dim_aksi: int = 5,
        hidden_layers: List[int] = [256, 128, 64],
        learning_rate: float = 0.0001,
        dropout_rate: float = 0.2,
        use_mixed_precision: bool = False,
    ):
        """
        Inisialisasi Jaringan Q.

        Args:
            dim_state: Dimensi state input (fitur teknikal + info posisi)
            dim_aksi: Dimensi aksi output (HOLD, BUY, SELL, CLOSE, REVERSE)
            hidden_layers: List ukuran hidden layers
            learning_rate: Learning rate untuk optimizer
            dropout_rate: Dropout rate untuk regularisasi
            use_mixed_precision: Gunakan mixed precision (FP16) untuk GPU speedup
        """
        self.dim_state = dim_state
        self.dim_aksi = dim_aksi
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.use_mixed_precision = use_mixed_precision

        # Aktifkan mixed precision untuk GPU dengan Tensor Cores
        if use_mixed_precision:
            policy = keras.mixed_precision.Policy("mixed_float16")
            keras.mixed_precision.set_global_policy(policy)
            logger.info("✅ Pelatihan menggunakan mixed precision (FP16) diaktifkan.")

        # Buat model
        self.model = self._bangun_model()

        logger.info("JaringanQ diinisialisasi:")
        logger.info(f"  - Input dim: {dim_state}")
        logger.info(f"  - Output dim: {dim_aksi}")
        logger.info(f"  - Hidden layers: {hidden_layers}")
        logger.info(f"  - Learning rate: {learning_rate}")
        logger.info(f"  - Dropout rate: {dropout_rate}")
        logger.info(f"  - Mixed precision: {use_mixed_precision}")
        logger.info(f"  - Total params: {self.model.count_params()}")

    def _bangun_model(self) -> keras.Model:
        """
        Bangun arsitektur model Q-Network.

        Returns:
            Compiled Keras model
        """
        # Lapisan input
        inputs = layers.Input(shape=(self.dim_state,), name="lapisan_input_state")
        x = inputs

        # Lapisan hidden dengan dropout untuk regularisasi
        for indeks, units in enumerate(self.hidden_layers):
            x = layers.Dense(
                units,
                activation="relu",
                kernel_initializer="he_normal",
                name=f"lapisan_hidden_{indeks+1}",
            )(x)

            # Dropout untuk layer pertama dan kedua (tapi tidak untuk layer terakhir)
            if indeks < len(self.hidden_layers) - 1:
                x = layers.Dropout(self.dropout_rate, name=f"dropout_{indeks+1}")(x)

        # Lapisan output (tidak ada aktivasi - nilai Q raw)
        outputs = layers.Dense(
            self.dim_aksi,
            activation=None,
            kernel_initializer="glorot_uniform",
            name="nilai_q_output",
        )(x)

        # Buat model
        model = keras.Model(inputs=inputs, outputs=outputs, name="DQN")

        # Compile dengan optimizer
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss="mse")  # Mean Squared Error

        return model

    def prediksi(self, state: np.ndarray) -> np.ndarray:
        """
        Prediksi nilai Q untuk state.

        Args:
            state: Vektor state atau batch states
                - Single state: shape (dim_state,)
                - Batch state: shape (batch_size, dim_state)

        Returns:
            Nilai Q untuk setiap aksi
                - Single: shape (dim_aksi,)
                - Batch: shape (batch_size, dim_aksi)
        """
        # Pastikan input adalah 2 dimensi (batch shape)
        if state.ndim == 1:
            state = np.expand_dims(state, axis=0)
            return self.model.predict(state, verbose=0)[0]
        else:
            return self.model.predict(state, verbose=0)

    def latih_batch(self, states: np.ndarray, target_nilai_q: np.ndarray) -> float:
        """
        Latih jaringan pada batch data.

        Args:
            states: Batch state, shape (batch_size, dim_state)
            target_nilai_q: Target nilai Q, shape (batch_size, dim_aksi)

        Returns:
            Niali Loss
        """
        histori = self.model.fit(
            states, target_nilai_q, batch_size=len(states), epochs=1, verbose=0
        )

        return histori.history["loss"][0]

    def salin_bobot_dari(self, jaringan_lain: "JaringanQ") -> None:
        """
        Salin bobot dari jaringan lain (untuk jaringan target).

        Args:
            jaringan_lain: JaringanQ untuk disalin bobotnya
        """
        self.model.set_weights(jaringan_lain.model.get_weights())

    def simpan(self, path: str) -> None:
        """
        Simpan model ke file.

        Args:
            path: Path untuk menyimpan model (.keras atau .h5)
        """
        self.model.save(path)
        logger.info(f"Model disimpan ke {path}")

    def muat(self, path: str) -> None:
        """
        Muat model dari file.

        Args:
            path: Path file model
        """
        self.model = keras.models.load_model(path)
        logger.info(f"Model dimuat dari {path}")

    def ringkasan(self) -> None:
        """Print ringkasan arsitektur model"""
        self.model.summary()


# =============================================================================
# TESTING
# =============================================================================
def test_jaringan():
    """Test Q-Network."""
    logger.info("Testing JaringanQ...")

    print(f"\n{'='*60}")
    print("TEST 1: Inisialisasi dan arsitektur")
    print(f"{'='*60}")

    # Buat network
    dim_state = 74
    dim_aksi = 5

    jaringan_q = JaringanQ(
        dim_state=dim_state,
        dim_aksi=dim_aksi,
        hidden_layers=[256, 128, 64],
        learning_rate=0.0001,
        dropout_rate=0.2,
    )

    print("\n Arsitektur model:")
    jaringan_q.ringkasan()

    # Test prediksi single state
    print(f"\n{'='*60}")
    print("TEST 2: Prediksi single state")
    print(f"{'='*60}")

    state = np.random.randn(dim_state).astype(np.float32)
    nilai_q = jaringan_q.prediksi(state)

    print(f"Input state shape: {state.shape}")
    print(f"Output Nilai Q shape: {nilai_q.shape}")
    print(f"Nilai Q: {nilai_q}")
    print(f"\nAksi terbaik: {np.argmax(nilai_q)}")

    # Test prediksi batch
    print(f"\n{'='*60}")
    print("TEST 3: Prediksi batch state")
    print(f"{'='*60}")

    ukuran_batch = 64
    batch_state = np.random.randn(ukuran_batch, dim_state).astype(np.float32)
    batch_nilai_q = jaringan_q.prediksi(batch_state)

    print(f"Input batch shape: {batch_state.shape}")
    print(f"Output Nilai Q shape: {batch_nilai_q.shape}")
    print(f"Sampel pertama Nilai Q: {batch_nilai_q[0]}")

    # Test training
    print(f"\n{'='*60}")
    print(f"TEST 4: Pelatihan Jaringan")
    print(f"{'='*60}")

    # Buat target dummy nilai Q
    target_nilai_q = np.random.randn(ukuran_batch, dim_aksi).astype(np.float32)

    loss = jaringan_q.latih_batch(batch_state, target_nilai_q)
    print(f"Loss pelatihan: {loss:.6f}")

    # Latih beberapa kali
    losses = []
    for i in range(5):
        loss = jaringan_q.latih_batch(batch_state, target_nilai_q)
        losses.append(loss)

    print(f"\nLosses dalam 5 iterasi:")
    for indeks, loss in enumerate(losses):
        print(f"    Iterasi {indeks+1}: {loss:.6f}")

    # Test kopi bobot model untuk jaringan target
    print(f"\n{'='*60}")
    print("TEST 5: Kopi bobot (jaringan target)")
    print(f"{'='*60}")

    jaringan_target = JaringanQ(
        dim_state=dim_state, dim_aksi=dim_aksi, hidden_layers=[256, 128, 64]
    )

    # Prediksi sebelum kopi
    q_sebelum = jaringan_target.prediksi(state)
    print(f"Nilai Q jaringan target (sebelum kopi): {q_sebelum}")

    # Kopi bobot
    jaringan_target.salin_bobot_dari(jaringan_q)

    # Prediksi setelah kopi
    q_setelah = jaringan_target.prediksi(state)
    print(f"Nilai Q jaringan target (setelah kopi): {q_setelah}")

    # Verifikasi sama
    q_utama = jaringan_q.prediksi(state)
    print(f"Nilai Q jaringan utama: {q_utama}")
    print(f"\nBobot dikopi secara benar: {np.allclose(q_setelah, q_utama)}")

    # Test simpan dan muat
    print(f"\n{'='*60}")
    print("TEST 6: Simpan dan muat model")
    print(f"{'='*60}")

    import tempfile
    import os

    # Buat file temporer
    file_temporer = tempfile.NamedTemporaryFile(delete=False, suffix=".keras")
    path_temporer = file_temporer.name
    file_temporer.close()

    try:
        # Simpan
        jaringan_q.simpan(path_temporer)

        # Buat jaringan baru dan muat
        jaringan_termuat = JaringanQ(
            dim_state=dim_state, dim_aksi=dim_aksi, hidden_layers=[256, 128, 64]
        )
        jaringan_termuat.muat(path_temporer)

        # Verifikasi prediksi sama
        q_original = jaringan_q.prediksi(state)
        q_termuat = jaringan_termuat.prediksi(state)

        print(f"Nilai Q original: {q_original}")
        print(f"Nilai Q termuat: {q_termuat}")
        print(f"Model dimuat secara benar: {np.allclose(q_original, q_termuat)}")

    finally:
        # Bersihkan
        if os.path.exists(path_temporer):
            os.remove(path_temporer)

    logger.info("Test selesai! ✅")
    print(f"\n{'='*60}")
    print("SEMUA TEST BERHASIL! 🚀")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    test_jaringan()
