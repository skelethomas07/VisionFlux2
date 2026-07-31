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


def compute_multiscale_bright_ridge(
    image: np.ndarray,
    *,
    sigmas: tuple[float, ...],
    gradient_sigma: float = 1.0,
    prefer_gpu: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, ComputeBackend]:
    """Return bright-line Hessian response, best scale, and first derivatives.

    All scale-space operations stay on the GPU until the final arrays are copied back.
    The response uses the negative principal Hessian eigenvalue with a blob penalty.
    """
    if not sigmas:
        raise ValueError("sigmas must not be empty")
    backend = detect_compute_backend(prefer_gpu=prefer_gpu)
    modules = _load_cupy_modules() if backend.gpu_available else None

    def _ridge_for_backend(xp, ndi, img):
        gy = ndi.gaussian_filter(img, gradient_sigma, order=(1, 0), mode="reflect")
        gx = ndi.gaussian_filter(img, gradient_sigma, order=(0, 1), mode="reflect")
        best = xp.zeros_like(img, dtype=xp.float32)
        best_scale = xp.full_like(img, float(sigmas[0]), dtype=xp.float32)
        eps = xp.asarray(1e-12, dtype=xp.float32)
        for sigma in sigmas:
            scale2 = float(sigma) ** 2
            dyy = ndi.gaussian_filter(img, sigma, order=(2, 0), mode="reflect") * scale2
            dxy = ndi.gaussian_filter(img, sigma, order=(1, 1), mode="reflect") * scale2
            dxx = ndi.gaussian_filter(img, sigma, order=(0, 2), mode="reflect") * scale2
            trace = dyy + dxx
            disc = xp.sqrt(xp.maximum((dyy - dxx) ** 2 + 4.0 * dxy**2, 0.0))
            lam_min = 0.5 * (trace - disc)
            lam_max = 0.5 * (trace + disc)
            line_strength = xp.maximum(-lam_min, 0.0)
            blob_ratio = xp.abs(lam_max) / (xp.abs(lam_min) + eps)
            response = line_strength * xp.exp(-0.5 * (blob_ratio / 0.65) ** 2)
            replace = response > best
            best = xp.where(replace, response, best)
            best_scale = xp.where(replace, float(sigma), best_scale)
        return best, best_scale, gy, gx

    if modules is not None:
        cp, cupy_ndimage = modules
        try:
            img = cp.asarray(np.asarray(image, dtype=np.float32))
            best, best_scale, gy, gx = _ridge_for_backend(cp, cupy_ndimage, img)
            return (
                cp.asnumpy(best).astype(np.float32, copy=False),
                cp.asnumpy(best_scale).astype(np.float32, copy=False),
                cp.asnumpy(gy).astype(np.float32, copy=False),
                cp.asnumpy(gx).astype(np.float32, copy=False),
                backend,
            )
        except Exception as exc:
            backend = ComputeBackend("CPU", False, f"GPU ridge 처리 실패 후 CPU 전환: {type(exc).__name__}")

    img = np.asarray(image, dtype=np.float32)
    best, best_scale, gy, gx = _ridge_for_backend(np, scipy_ndimage, img)
    return (
        np.asarray(best, dtype=np.float32),
        np.asarray(best_scale, dtype=np.float32),
        np.asarray(gy, dtype=np.float32),
        np.asarray(gx, dtype=np.float32),
        backend,
    )
