from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from scipy import ndimage as scipy_ndimage


@dataclass(frozen=True)
class ComputeBackend:
    name: str
    gpu_available: bool
    detail: str


@lru_cache(maxsize=1)
def _load_cupy_modules() -> tuple[Any, Any] | None:
    try:
        import cupy as cp
        from cupyx.scipy import ndimage as cupy_ndimage

        if int(cp.cuda.runtime.getDeviceCount()) < 1:
            return None
        # Force a tiny allocation so driver/runtime problems are caught here.
        cp.asarray([0.0], dtype=cp.float32)
        return cp, cupy_ndimage
    except Exception:
        return None


def detect_compute_backend(prefer_gpu: bool = True) -> ComputeBackend:
    if not prefer_gpu:
        return ComputeBackend("CPU", False, "GPU 사용 안 함")
    modules = _load_cupy_modules()
    if modules is None:
        return ComputeBackend("CPU", False, "CUDA/CuPy를 찾지 못해 CPU 사용")
    cp, _ = modules
    try:
        device_id = int(cp.cuda.Device().id)
        props = cp.cuda.runtime.getDeviceProperties(device_id)
        raw_name = props.get("name", b"CUDA GPU") if isinstance(props, dict) else b"CUDA GPU"
        name = raw_name.decode(errors="replace") if isinstance(raw_name, bytes) else str(raw_name)
    except Exception:
        name = "CUDA GPU"
    return ComputeBackend("GPU", True, name)


def _cpu_structure_tensor(
    image: np.ndarray,
    derivative_sigma: float,
    tensor_sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    img = np.asarray(image, dtype=np.float32)
    gy = scipy_ndimage.gaussian_filter(img, derivative_sigma, order=(1, 0), mode="reflect")
    gx = scipy_ndimage.gaussian_filter(img, derivative_sigma, order=(0, 1), mode="reflect")
    jyy = scipy_ndimage.gaussian_filter(gy * gy, tensor_sigma, mode="reflect")
    jxy = scipy_ndimage.gaussian_filter(gy * gx, tensor_sigma, mode="reflect")
    jxx = scipy_ndimage.gaussian_filter(gx * gx, tensor_sigma, mode="reflect")
    return (
        np.asarray(jyy, dtype=np.float32),
        np.asarray(jxy, dtype=np.float32),
        np.asarray(jxx, dtype=np.float32),
    )


def compute_structure_tensor(
    image: np.ndarray,
    *,
    derivative_sigma: float,
    tensor_sigma: float,
    prefer_gpu: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, ComputeBackend]:
    backend = detect_compute_backend(prefer_gpu=prefer_gpu)
    modules = _load_cupy_modules() if backend.gpu_available else None
    if modules is not None:
        cp, cupy_ndimage = modules
        try:
            img = cp.asarray(np.asarray(image, dtype=np.float32))
            gy = cupy_ndimage.gaussian_filter(
                img, derivative_sigma, order=(1, 0), mode="reflect"
            )
            gx = cupy_ndimage.gaussian_filter(
                img, derivative_sigma, order=(0, 1), mode="reflect"
            )
            jyy = cupy_ndimage.gaussian_filter(gy * gy, tensor_sigma, mode="reflect")
            jxy = cupy_ndimage.gaussian_filter(gy * gx, tensor_sigma, mode="reflect")
            jxx = cupy_ndimage.gaussian_filter(gx * gx, tensor_sigma, mode="reflect")
            return (
                cp.asnumpy(jyy).astype(np.float32, copy=False),
                cp.asnumpy(jxy).astype(np.float32, copy=False),
                cp.asnumpy(jxx).astype(np.float32, copy=False),
                backend,
            )
        except Exception as exc:
            backend = ComputeBackend("CPU", False, f"GPU 처리 실패 후 CPU 전환: {type(exc).__name__}")

    jyy, jxy, jxx = _cpu_structure_tensor(image, derivative_sigma, tensor_sigma)
    return jyy, jxy, jxx, backend
