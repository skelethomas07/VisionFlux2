import numpy as np

from pipeline.compute import compute_structure_tensor, detect_compute_backend


def test_detect_compute_backend_can_force_cpu():
    info = detect_compute_backend(prefer_gpu=False)
    assert info.name == "CPU"
    assert info.gpu_available is False


def test_compute_structure_tensor_returns_numpy_arrays_on_cpu():
    y, x = np.mgrid[:32, :32]
    image = np.exp(-((y - 16.0) ** 2) / 12.0).astype(np.float32)
    jyy, jxy, jxx, info = compute_structure_tensor(
        image,
        derivative_sigma=1.2,
        tensor_sigma=3.0,
        prefer_gpu=False,
    )
    assert info.name == "CPU"
    assert isinstance(jyy, np.ndarray)
    assert jyy.shape == image.shape
    assert jxy.shape == image.shape
    assert jxx.shape == image.shape
    assert np.isfinite(jyy).all()
