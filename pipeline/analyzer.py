from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import re
import tempfile
from typing import Callable

import numpy as np
import pandas as pd
from PIL import Image
from skimage.transform import resize
import tifffile

from pipeline import legacy_pipeline
from pipeline.fast_detector import FastDetectionResult, detect_fibers_fast
from pipeline.image_metadata import (
    ImageQuality,
    ScaleCalibration,
    extract_sem_content_and_metadata,
)
from pipeline.orientation import (
    OrientationResult,
    analyze_orientation,
    annotate_measurement_directions,
    orientation_guided_rescue,
    orientation_summary_dict,
)


@dataclass(frozen=True)
class DecodedImage:
    image: np.ndarray
    analysis_scale: float
    original_shape: tuple[int, int]
    full_original_shape: tuple[int, int]
    footer_start_y: int | None
    calibration: ScaleCalibration
    quality: ImageQuality
    original_image: np.ndarray


@dataclass
class AnalysisResult:
    image: np.ndarray
    image_name: str
    analysis_scale: float
    original_shape: tuple[int, int]
    summary: dict
    measurements: pd.DataFrame
    regions: pd.DataFrame
    representatives: pd.DataFrame
    candidates: pd.DataFrame
    diagnostic_png: bytes | None = None
    orientation: OrientationResult | None = None
    calibration: ScaleCalibration | None = None
    quality: ImageQuality | None = None
    footer_start_y: int | None = None
    full_original_shape: tuple[int, int] | None = None
    original_image: np.ndarray | None = None


def _load_image_bytes(data: bytes, filename: str) -> np.ndarray:
    suffix = Path(filename).suffix.lower()
    if suffix in {".tif", ".tiff"}:
        arr = tifffile.imread(BytesIO(data))
    else:
        arr = np.asarray(Image.open(BytesIO(data)))
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        rgb = arr[..., :3].astype(np.float32)
        arr = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    elif arr.ndim == 3:
        arr = arr[arr.shape[0] // 2]
    arr = np.squeeze(np.asarray(arr, np.float32))
    if arr.ndim != 2:
        raise ValueError(f"2-D SEM image expected, got shape {arr.shape}")
    finite = np.isfinite(arr)
    if not finite.any():
        raise ValueError("image contains no finite pixels")
    if not finite.all():
        arr[~finite] = np.median(arr[finite])
    return arr


def decode_and_scale_image(
    data: bytes,
    filename: str,
    max_dimension: int | None = None,
    *,
    auto_calibrate: bool = True,
) -> DecodedImage:
    raw = _load_image_bytes(data, filename)
    full_original_shape = (int(raw.shape[0]), int(raw.shape[1]))
    metadata = extract_sem_content_and_metadata(raw, auto_calibrate=auto_calibrate)
    content = np.asarray(metadata.content, np.float32)
    original_display = _display_uint8(content)
    original_shape = (int(content.shape[0]), int(content.shape[1]))
    longest = max(original_shape)
    scale = 1.0
    if max_dimension is not None and max_dimension > 0 and longest > max_dimension:
        scale = float(max_dimension) / float(longest)
        target = (max(1, round(content.shape[0] * scale)), max(1, round(content.shape[1] * scale)))
        content = resize(content, target, order=1, mode="reflect", anti_aliasing=True, preserve_range=True).astype(np.float32)
    return DecodedImage(
        content.astype(np.float32),
        scale,
        original_shape,
        full_original_shape,
        metadata.footer_start_y,
        metadata.calibration,
        metadata.quality,
        original_display,
    )


def _add_scaled_width_columns(df: pd.DataFrame, analysis_scale: float, nm_per_px: float | None) -> pd.DataFrame:
    out = df.copy()
    for col in list(out.columns):
        if col.endswith("_width_px") or col in {"width_px", "median_width_px", "min_width_px", "max_width_px", "p10_width_px", "p90_width_px", "representative_width_px"}:
            values = pd.to_numeric(out[col], errors="coerce")
            if col == "width_px":
                new_col = "width_original_px"
            else:
                new_col = col.replace("_px", "_original_px")
            out[new_col] = values / analysis_scale
            if nm_per_px is not None:
                nm_col = col.replace("_px", "_nm")
                out[nm_col] = out[new_col] * nm_per_px
    return out


def normalize_analysis_tables(
    local_df: pd.DataFrame,
    region_df: pd.DataFrame,
    representative_df: pd.DataFrame,
    analysis_scale: float,
    nm_per_px: float | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if analysis_scale <= 0:
        raise ValueError("analysis_scale must be positive")
    local = _add_scaled_width_columns(local_df, analysis_scale, nm_per_px)
    regions = _add_scaled_width_columns(region_df, analysis_scale, nm_per_px)
    reps = _add_scaled_width_columns(representative_df, analysis_scale, nm_per_px)

    if not local.empty:
        local = local.reset_index(drop=True)
        ids = []
        seen: dict[str, int] = {}
        for i, row in local.iterrows():
            rid = row.get("fiber_region_id", "x")
            sample = row.get("region_sample_index", i)
            try:
                rid_value = float(rid)
                rid_text = str(int(rid_value)) if rid_value.is_integer() else str(rid)
            except (TypeError, ValueError):
                rid_text = str(rid)
            try:
                sample_text = str(int(float(sample)))
            except (TypeError, ValueError):
                sample_text = str(i)
            base = f"auto-r{rid_text}-s{sample_text}"
            count = seen.get(base, 0)
            seen[base] = count + 1
            ids.append(base if count == 0 else f"{base}-{count}")
        local.insert(0, "measurement_id", ids)
        local["status"] = "active"
        local["source"] = "auto"
        local["review_label"] = "unreviewed"
        if "width_nm" not in local.columns:
            local["width_nm"] = np.nan if nm_per_px is None else local["width_original_px"] * nm_per_px
    return local, regions, reps




def _region_text(value) -> str:
    try:
        number = float(value)
        if np.isfinite(number) and number.is_integer():
            return str(int(number))
    except (TypeError, ValueError):
        pass
    return str(value)


def merge_orientation_measurements(
    local: pd.DataFrame,
    rescue: pd.DataFrame,
    analysis_scale: float,
    nm_per_px: float | None,
) -> pd.DataFrame:
    if rescue is None or rescue.empty:
        return local.copy(deep=True)
    added = rescue.copy(deep=True).reset_index(drop=True)
    ids = []
    seen: dict[str, int] = {}
    for i, row in added.iterrows():
        region = _region_text(row.get("fiber_region_id", "x"))
        sample = _region_text(row.get("region_sample_index", i))
        base = f"orientation-r{region}-s{sample}"
        count = seen.get(base, 0)
        seen[base] = count + 1
        ids.append(base if count == 0 else f"{base}-{count}")
    added.insert(0, "measurement_id", ids)
    added["status"] = "active"
    added["source"] = "orientation"
    added["review_label"] = "unreviewed"
    added["width_px"] = pd.to_numeric(added["width_px"], errors="coerce")
    added["width_original_px"] = added["width_px"] / float(analysis_scale)
    added["width_nm"] = np.nan if nm_per_px is None else added["width_original_px"] * float(nm_per_px)
    return pd.concat([local, added], ignore_index=True, sort=False)

def _safe_stem(filename: str) -> str:
    stem = Path(filename).stem
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    return stem or "uploaded_sem"


def _display_uint8(image: np.ndarray) -> np.ndarray:
    finite = image[np.isfinite(image)]
    if not finite.size:
        return np.zeros(image.shape, np.uint8)
    lo, hi = np.percentile(finite, [0.5, 99.5])
    norm = np.clip((image - lo) / max(float(hi - lo), 1e-9), 0, 1)
    return np.round(norm * 255).astype(np.uint8)


def _emit_progress(
    callback: Callable[[float, str], None] | None,
    fraction: float,
    message: str,
) -> None:
    if callback is not None:
        callback(float(min(1.0, max(0.0, fraction))), str(message))


def _run_legacy_fallback(
    decoded: DecodedImage,
    filename: str,
    nm_per_px: float | None,
) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, bytes | None]:
    """Run the previous detector only when explicitly requested as an emergency fallback."""
    with tempfile.TemporaryDirectory(prefix="sem-fiber-legacy-") as tmp:
        root = Path(tmp)
        image_path = root / f"{_safe_stem(filename)}_analysis.tif"
        output_dir = root / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(image_path, decoded.image.astype(np.float32))
        legacy_pipeline.OUTPUT_DIR = output_dir
        legacy_pipeline.SHOW_INLINE = False
        legacy_pipeline.NM_PER_PX = {}
        if nm_per_px is not None:
            legacy_pipeline.NM_PER_PX[image_path.name] = float(nm_per_px) / decoded.analysis_scale
        summary, local, regions, reps, candidates = legacy_pipeline.process_one(image_path)
        analysis_img, _ = legacy_pipeline.prepare_image(image_path)
        diagnostic_path = output_dir / f"{image_path.stem}_fiber_regions_sem_refined.png"
        diagnostic = diagnostic_path.read_bytes() if diagnostic_path.exists() else None
    return summary, local, regions, reps, candidates, analysis_img, diagnostic


def run_uploaded_analysis(
    data: bytes,
    filename: str,
    nm_per_px: float | None = None,
    max_dimension: int | None = 1600,
    *,
    prefer_gpu: bool = True,
    progress_callback: Callable[[float, str], None] | None = None,
    fallback_to_legacy: bool = False,
    auto_calibrate: bool = True,
) -> AnalysisResult:
    """Analyze one uploaded SEM image with the fast direction-graph detector.

    The structure tensor, ridge maps, pore mask, centerline paths, and normal edge
    profiles are each computed once. The old beam-search pipeline remains available
    only through ``fallback_to_legacy=True``.
    """
    _emit_progress(progress_callback, 0.0, "이미지 읽는 중")
    decoded = decode_and_scale_image(
        data, filename, max_dimension=max_dimension, auto_calibrate=auto_calibrate
    )
    effective_nm_per_px = (
        float(nm_per_px)
        if nm_per_px is not None
        else decoded.calibration.nm_per_px
    )
    analysis_img = decoded.image.astype(np.float32, copy=False)
    _emit_progress(progress_callback, 0.06, "분석 이미지 준비")

    def detector_progress(fraction: float, message: str) -> None:
        _emit_progress(progress_callback, 0.08 + 0.84 * float(fraction), message)

    diagnostic = None
    used_legacy = False
    try:
        fast: FastDetectionResult = detect_fibers_fast(
            analysis_img,
            prefer_gpu=prefer_gpu,
            progress_callback=detector_progress,
        )
        summary = dict(fast.summary)
        local = fast.measurements
        regions = fast.regions
        reps = fast.representatives
        candidates = fast.candidates
        orientation = fast.orientation
    except Exception:
        if not fallback_to_legacy:
            raise
        used_legacy = True
        _emit_progress(progress_callback, 0.10, "고속 검출 실패 · 기존 검출기로 전환")
        summary, local, regions, reps, candidates, analysis_img, diagnostic = _run_legacy_fallback(
            decoded, filename, effective_nm_per_px,
        )
        orientation = analyze_orientation(
            analysis_img,
            sigma_px=4.0,
            prefer_gpu=prefer_gpu,
        )

    _emit_progress(progress_callback, 0.93, "두께와 방향 결과 정리")
    local, regions, reps = normalize_analysis_tables(
        local, regions, reps, decoded.analysis_scale, effective_nm_per_px,
    )
    local = annotate_measurement_directions(local, orientation)

    summary = dict(summary)
    summary.update(orientation_summary_dict(orientation))
    summary.update(
        uploaded_filename=filename,
        original_height=decoded.original_shape[0],
        original_width=decoded.original_shape[1],
        analysis_scale=decoded.analysis_scale,
        nm_per_original_px=effective_nm_per_px,
        calibration_source=(
            "manual" if nm_per_px is not None
            else ("footer_ocr" if decoded.calibration.detected else "pixels")
        ),
        scale_bar_detected=bool(decoded.calibration.detected),
        scale_bar_value=decoded.calibration.scale_value,
        scale_bar_unit=decoded.calibration.scale_unit,
        scale_bar_length_px=decoded.calibration.bar_length_px,
        scale_bar_confidence=decoded.calibration.confidence,
        footer_start_y=decoded.footer_start_y,
        footer_removed_px=(decoded.full_original_shape[0] - decoded.original_shape[0]),
        full_original_height=decoded.full_original_shape[0],
        full_original_width=decoded.full_original_shape[1],
        image_quality=decoded.quality.label,
        quality_messages=list(decoded.quality.messages),
        quality_sharpness=decoded.quality.sharpness,
        quality_contrast=decoded.quality.contrast,
        estimated_min_fiber_width_px=decoded.quality.estimated_min_fiber_width_px,
        analysis_height=int(analysis_img.shape[0]),
        analysis_width=int(analysis_img.shape[1]),
        legacy_fallback_used=bool(used_legacy),
    )
    result = AnalysisResult(
        image=_display_uint8(analysis_img),
        image_name=filename,
        analysis_scale=decoded.analysis_scale,
        original_shape=decoded.original_shape,
        summary=summary,
        measurements=local,
        regions=regions,
        representatives=reps,
        candidates=candidates,
        diagnostic_png=diagnostic,
        orientation=orientation,
        calibration=decoded.calibration,
        quality=decoded.quality,
        footer_start_y=decoded.footer_start_y,
        full_original_shape=decoded.full_original_shape,
        original_image=decoded.original_image,
    )
    _emit_progress(progress_callback, 1.0, "분석 완료")
    return result
