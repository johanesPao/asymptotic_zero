"""
Utilitas GPU untuk DQN Training

Konfigurasi dan monitoring GPU (RTX 4070) untuk training yang lebih optimal.
"""

import tensorflow as tf
import logging
from typing import Optional, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_gpu(
    batas_memori: Optional[int] = None, dinamis: bool = True, id_device: int = 0
) -> bool:
    """
    Setup GPU untuk training TensorFlow.

    Args:
        batas_memori: Batas memori GPU (MB). None = gunakan semua.
        dinamis: Alokasi memori secara dinamis
        id_device: GPU device ID (0 untuk GPU pertama).

    Returns:
        True jika GPU berhasil dikonfigurasi, False jika tidak ada GPU
    """
    gpus = tf.config.list_physical_devices("GPU")

    if not gpus:
        logger.warning("⚠️ GPU tidak terdeteksi! Training akan menggunakan CPU.")
        logger.warning("Pastikan:")
        logger.warning("    1. NVIDIA driver terinstall")
        logger.warning("    2. CUDA toolkit terinstall")
        logger.warning("    3. TensorFlow versi GPU terinstall")
        return False

    try:
        # Pilih GPU
        if id_device < len(gpus):
            gpu = gpus[id_device]
        else:
            logger.warning(f"GPU {id_device} tidak ditemukan, menggunakan GPU 0")
            gpu = gpus[0]

        # Konfigurasi GPU
        if batas_memori:
            # Set batas memori
            tf.config.set_logical_device_configuration(
                gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=batas_memori)]
            )
            logger.info(f"Batas memori GPU: {batas_memori} MB")

        if dinamis:
            # Pertumbuhan memori: alokasi bertahap
            # Menghindari TensorFlow mengambil semua VRAM di awal
            tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("Pertumbuhan memori GPU: ENABLED (alokasi dinamis)")

        # Visible devices (untuk multi-GPU, berikan pilihan GPU yang bisa dipakai)
        tf.config.set_visible_devices([gpu], "GPU")

        logger.info("✅ Konfigurasi GPU berhasil!")
        logger.info(f"GPU: {gpu.name}")

        return True

    except RuntimeError as e:
        logger.error(f"Error saat konfigurasi GPU: {e}")
        return False


def print_info_gpu() -> Dict:
    """
    Print informsi detail tentang GPU yang tersedia

    Returns:
        Dictionary dengan informasi GPU
    """
    print(f"\n{'='*80}")
    print("INFORMASI GPU")
    print(f"{'='*80}")

    # Versi TensorFlow
    print(f"\nTensorFlow Version: {tf.__version__}")
    print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")

    # GPU devices
    gpus = tf.config.list_physical_devices("GPU")
    print(f"\nDevice GPU: {len(gpus)}")

    info = {
        "versi_tensorflow": tf.__version__,
        "cuda_tersedia": tf.test.is_built_with_cuda(),
        "jumlah_gpu": len(gpus),
        "nama_gpu": [],
    }

    if gpus:
        for indeks, gpu in enumerate(gpus):
            print(f"    [{indeks}] {gpu.name}")
            info["nama_gpu"].append(gpu.name)

        # Info memori
        try:
            detail_gpus = tf.config.experimental.get_device_details(gpus[0])
            print("\nDetail GPU:")
            for kunci, nilai in detail_gpus.items():
                print(f"    {kunci}: {nilai}")
                info[kunci] = nilai
        except:
            pass

        # Cek setting dinamika memori
        try:
            dinamis = tf.config.experimental.get_memory_growth(gpus[0])
            print(f"\nMemori Dinamis: {dinamis}")
            info["memori_dinamis"] = dinamis
        except:
            pass
    else:
        print("\n⚠️ Tidak ada GPU terdeteksi!")
        print("Pelatihan akan menggunakan CPU (lebih lambat)")

    print(f"{'='*80}\n")

    return info


def komparasi_gpu_vs_cpu(ukuran_matriks: int = 5000) -> Dict:
    """
    Komparasi performa GPU vs CPU dengan operasi multiplikasi matriks.

    Args:
        ukuran_matriks: Ukuran matriks untuk test (default 5000x5000)

    Returns:
        Dictionary dengan hasil benchmark
    """
    import time

    print(f"\n{'='*80}")
    print(f"Komparasi GPU vs CPU (Matriks {ukuran_matriks}x{ukuran_matriks})")
    print(f"{'='*80}\n")

    hasil = {}

    # Benchmark CPU
    print("Testing performa CPU...")
    with tf.device("/CPU:0"):
        cpu_mulai = time.time()
        a = tf.random.normal([ukuran_matriks, ukuran_matriks])
        b = tf.random.normal([ukuran_matriks, ukuran_matriks])
        c = tf.matmul(a, b)
        _ = c.numpy()
        waktu_cpu = time.time() - cpu_mulai

    print(f"Waktu CPU: {waktu_cpu:.4f} detik")
    hasil["waktu_cpu"] = waktu_cpu

    # Benchmark GPU
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print("\nTesting performa GPU...")
        with tf.device("/GPU:0"):
            gpu_mulai = time.time()
            a = tf.random.normal([ukuran_matriks, ukuran_matriks])
            b = tf.random.normal([ukuran_matriks, ukuran_matriks])
            c = tf.matmul(a, b)
            _ = c.numpy()
            waktu_gpu = time.time() - gpu_mulai

        print(f"Waktu GPU: {waktu_gpu:.4f} detik")
        hasil["waktu_gpu"] = waktu_gpu

        # Percepatan
        percepatan = waktu_cpu / waktu_gpu
        print(f"\n🚀 Percepatan GPU: {percepatan:.2f}x lebih cepat dari CPU")
        hasil["percepatan"] = percepatan

        if percepatan < 2:
            print("\n Peringatan: percepatan GPU kurang dari 2x")
            print("Kemungkinan masalah:")
            print(" - Matriks terlalu kecil untuk GPU")
            print(" - CUDA setup tidak optiomal")
            print(" - Driver perlu update")
    else:
        print("\n⚠️ GPU tidak tersedia untuk komparasi")

    print(f"{'='*80}\n")

    return hasil


def monitor_memori_gpu():
    """
    Monitor penggunaan memori GPU saat pelatihan.
    Panggil fungsi ini secara berkala untuk tracking.
    """
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return None

    try:
        # TensorFlow 2.x memory stats
        info_memori = tf.config.experimental.get_memory_info("GPU:0")

        mb_saat_ini = info_memori["current"] / (1024**2)
        mb_tertinggi = info_memori["peak"] / (1024**2)

        return {"mb_saat_ini": mb_saat_ini, "mb_tertinggi": mb_tertinggi}
    except:
        return None


def tes_setup_gpu():
    """Tes lengkap setup GPU"""
    logger.info("Melakukan pengetesan setup GPU...")

    # 1. Print info GPU
    info = print_info_gpu()

    # 2. Setup GPU dengan memori dinamis
    sukses = setup_gpu(dinamis=True)

    if not sukses:
        logger.error("Setup GPU gagal!")
        return False

    # 3. Komparasi
    hasil = komparasi_gpu_vs_cpu(ukuran_matriks=10000)

    if "percepatan" in hasil and hasil["percepatan"] > 2:
        logger.info(f"✅ Setup GPU SUKSES! Percepatan: {hasil['percepatan']:.2f}x")
        return True
    else:
        logger.warning("⚠️ GPU terdeteksi tapi performa tidak optimal")
        return True


if __name__ == "__main__":
    # Jalankan test
    tes_setup_gpu()
