from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import re
from typing import Any

import numpy as np
from scipy import ndimage as ndi


@dataclass(frozen=True)
class ScaleCalibration:
    detected: bool
    nm_per_px: float | None
    scale_value: float | None
    scale_unit: str | None
    scale_value_nm: float | None
    bar_length_px: float | None
    bar_bbox: tuple[int, int, int, int] | None
    ocr_text: str = ""
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ImageQuality:
    label: str
    width_px: int
    height_px: int
    longest_side_px: int
    sharpness: float
    contrast: float
    saturation_fraction: float
    estimated_min_fiber_width_px: float | None
    messages: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MetadataExtraction:
    content: np.ndarray
    footer: np.ndarray | None
    footer_start_y: int | None
    calibration: ScaleCalibration
    quality: ImageQuality


def robust_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if not finite.size:
        return np.zeros(arr.shape, np.uint8)
    lo, hi = np.percentile(finite, [0.5, 99.5])
    out = np.clip((arr - lo) / max(float(hi - lo), 1e-9), 0.0, 1.0)
    return np.round(out * 255).astype(np.uint8)


def detect_footer_boundary(image: np.ndarray) -> int | None:
    """Locate a dark SEM metadata strip near the image bottom.

    Returns the first row of the footer. The detector deliberately requires a
    sustained, abrupt transition so dark pores inside the SEM field are not cut.
    """
    gray = robust_uint8(image)
    h, w = gray.shape
    if h < 80 or w < 120:
        return None

    row_mean = gray.mean(axis=1)
    row_dark = (gray < 40).mean(axis=1)
    start = int(round(h * 0.62))
    max_footer = int(round(h * 0.32))
    min_footer = max(14, int(round(h * 0.035)))

    candidates: list[tuple[float, int]] = []
    for y in range(start, h - min_footer):
        footer_h = h - y
        if footer_h > max_footer:
            continue
        after = slice(y, min(h, y + max(8, min_footer // 2)))
        before = slice(max(0, y - 14), y)
        after_dark = float(np.median(row_dark[after]))
        before_mean = float(np.median(row_mean[before])) if y > 0 else float(row_mean[y])
        after_mean = float(np.median(row_mean[after]))
        abrupt_drop = before_mean - after_mean
        bottom_dark = float(np.mean(gray[y:] < 45))
        # Metadata strips may contain white text, therefore the whole footer is
        # not required to be black. The first few rows must still be strongly dark.
        if after_dark >= 0.55 and abrupt_drop >= 24 and bottom_dark >= 0.58:
            score = 1.5 * after_dark + 0.012 * abrupt_drop + 0.5 * bottom_dark - 0.0005 * y
            candidates.append((score, y))

    if not candidates:
        return None
    # Prefer the earliest plausible abrupt boundary, while rejecting isolated rows.
    plausible = sorted(candidates, key=lambda item: item[1])
    first_y = plausible[0][1]
    nearby = [item for item in candidates if abs(item[1] - first_y) <= 8]
    return int(max(nearby)[1] if nearby else first_y)


def _component_candidates(binary: np.ndarray) -> list[tuple[float, tuple[int, int, int, int]]]:
    labels, count = ndi.label(binary)
    objects = ndi.find_objects(labels)
    h, w = binary.shape
    rows: list[tuple[float, tuple[int, int, int, int]]] = []
    for label_id, sl in enumerate(objects, start=1):
        if sl is None:
            continue
        y0, y1 = sl[0].start, sl[0].stop
        x0, x1 = sl[1].start, sl[1].stop
        height = y1 - y0
        width = x1 - x0
        if width < max(20, int(0.025 * w)) or width > int(0.45 * w):
            continue
        if height < 2 or height > max(22, int(0.20 * h)):
            continue
        aspect = width / max(height, 1)
        if aspect < 4.5:
            continue
        component = labels[y0:y1, x0:x1] == label_id
        fill = float(component.mean())
        if fill < 0.48:
            continue
        # Scale bars are usually in the right half and below the vertical center.
        rightness = (x0 + x1) / (2 * max(w, 1))
        lowerness = (y0 + y1) / (2 * max(h, 1))
        score = math.log1p(width) + 0.8 * fill + 0.8 * rightness + 0.35 * lowerness
        rows.append((score, (x0, y0, x1, y1)))
    return rows


def detect_scale_bar(footer: np.ndarray) -> tuple[float | None, tuple[int, int, int, int] | None, float]:
    gray = robust_uint8(footer)
    # Keep bright solid horizontal structures but suppress thin text strokes.
    bright = gray >= max(195, int(np.percentile(gray, 93)))
    horizontal = ndi.binary_opening(bright, structure=np.ones((2, 11), bool))
    horizontal = ndi.binary_closing(horizontal, structure=np.ones((2, 7), bool))
    candidates = _component_candidates(horizontal)
    if not candidates:
        return None, None, 0.0
    score, bbox = max(candidates, key=lambda item: item[0])
    x0, y0, x1, y1 = bbox
    return float(x1 - x0), bbox, float(min(1.0, score / 9.0))


_UNIT_TO_NM = {
    "nm": 1.0,
    "um": 1000.0,
    "µm": 1000.0,
    "μm": 1000.0,
    "micron": 1000.0,
}


def _normalize_ocr_token(text: str) -> str:
    return (
        text.lower()
        .replace("μ", "µ")
        .replace("u m", "um")
        .replace("µ m", "µm")
        .replace(" ", "")
        .replace(",", ".")
    )


def _parse_scale_token(text: str) -> tuple[float, str] | None:
    token = _normalize_ocr_token(text)
    match = re.search(r"(?<![\dx])(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>nm|um|µm|micron)\b", token)
    if not match:
        return None
    try:
        value = float(match.group("value"))
    except ValueError:
        return None
    unit = match.group("unit")
    if value <= 0:
        return None
    return value, unit


def ocr_scale_value(
    footer: np.ndarray,
    bar_bbox: tuple[int, int, int, int] | None,
) -> tuple[float | None, str | None, str, float]:
    """Read a scale label such as 100 nm or 1 µm from the footer.

    Tesseract is optional at runtime. When it is unavailable or uncertain, the
    caller receives ``None`` and can ask the user for confirmation.
    """
    try:
        import pytesseract
        from pytesseract import Output
    except Exception:
        return None, None, "", 0.0

    gray = robust_uint8(footer)
    # First read a compact region around the scale bar. This prevents an FOV
    # dimension elsewhere in the footer from being mistaken for the bar label.
    nearby_text = ""
    if bar_bbox is not None:
        bx0, by0, bx1, by1 = bar_bbox
        bw = max(1, bx1 - bx0)
        bh = max(1, by1 - by0)
        x0 = max(0, int(bx0 - 0.25 * bw))
        x1 = min(gray.shape[1], int(bx1 + 1.6 * bw))
        y0 = max(0, int(by0 - 1.2 * bh))
        y1 = min(gray.shape[0], int(by1 + 2.0 * bh))
        crop = gray[y0:y1, x0:x1]
        if crop.size:
            enlarged_crop = np.repeat(np.repeat(crop, 4, axis=0), 4, axis=1)
            try:
                nearby_text = pytesseract.image_to_string(
                    enlarged_crop, config="--psm 7"
                ).strip()
                parsed_nearby = _parse_scale_token(nearby_text)
                if parsed_nearby is not None:
                    return parsed_nearby[0], parsed_nearby[1], nearby_text, 0.85
            except Exception:
                nearby_text = ""

    # Upscaling the whole footer supports FOV calibration and a fallback scale read.
    zoom = 2
    enlarged = np.repeat(np.repeat(gray, zoom, axis=0), zoom, axis=1)
    try:
        data = pytesseract.image_to_data(
            enlarged,
            output_type=Output.DICT,
            config="--psm 11",
        )
        full_text = " ".join(str(t) for t in data.get("text", []) if str(t).strip())
        if nearby_text:
            full_text = nearby_text + " " + full_text
    except Exception:
        return None, None, nearby_text, 0.0

    candidates: list[tuple[float, float, str]] = []
    n = len(data.get("text", []))
    for i in range(n):
        text = str(data["text"][i]).strip()
        if not text:
            continue
        # Unit and number are often split into neighboring OCR boxes. Test a small
        # window of consecutive tokens.
        for j in range(i, min(n, i + 3)):
            combined = "".join(str(data["text"][k]).strip() for k in range(i, j + 1))
            parsed = _parse_scale_token(combined)
            if parsed is None:
                continue
            value, unit = parsed
            left = min(int(data["left"][k]) for k in range(i, j + 1)) / zoom
            top = min(int(data["top"][k]) for k in range(i, j + 1)) / zoom
            right = max(int(data["left"][k]) + int(data["width"][k]) for k in range(i, j + 1)) / zoom
            bottom = max(int(data["top"][k]) + int(data["height"][k]) for k in range(i, j + 1)) / zoom
            conf_values = []
            for k in range(i, j + 1):
                try:
                    conf_values.append(max(0.0, float(data["conf"][k])))
                except Exception:
                    pass
            conf = float(np.mean(conf_values) / 100.0) if conf_values else 0.0
            score = conf
            if bar_bbox is not None:
                bx0, by0, bx1, by1 = bar_bbox
                cx, cy = 0.5 * (left + right), 0.5 * (top + bottom)
                bcx, bcy = 0.5 * (bx0 + bx1), 0.5 * (by0 + by1)
                distance = math.hypot(cx - bcx, cy - bcy)
                score += math.exp(-distance / max(footer.shape[1] * 0.15, 1.0))
            # FOV dimensions contain an 'x' and should not be mistaken for the bar.
            if "x" in _normalize_ocr_token(combined):
                score -= 1.0
            candidates.append((score, value, unit))

    if not candidates:
        # A final full-text parser helps when OCR boxes are irregular.
        parsed = _parse_scale_token(full_text)
        if parsed:
            return parsed[0], parsed[1], full_text, 0.25
        return None, None, full_text, 0.0
    score, value, unit = max(candidates, key=lambda item: item[0])
    return float(value), str(unit), full_text, float(np.clip(score / 2.0, 0.0, 1.0))


def parse_fov_calibration(text: str, content_shape: tuple[int, int]) -> tuple[float | None, float]:
    token = _normalize_ocr_token(text).replace(":", "")
    match = re.search(
        r"fov[^0-9]*(?P<w>\d+(?:\.\d+)?)x(?P<h>\d+(?:\.\d+)?)(?P<unit>nm|um|µm)",
        token,
    )
    if not match:
        return None, 0.0
    width_value = float(match.group("w"))
    height_value = float(match.group("h"))
    unit = match.group("unit")
    factor = _UNIT_TO_NM[unit]
    h_px, w_px = content_shape
    if width_value <= 0 or height_value <= 0 or w_px <= 0 or h_px <= 0:
        return None, 0.0
    x_nm_per_px = width_value * factor / float(w_px)
    y_nm_per_px = height_value * factor / float(h_px)
    ratio = max(x_nm_per_px, y_nm_per_px) / max(min(x_nm_per_px, y_nm_per_px), 1e-12)
    if ratio > 1.18:
        return None, 0.0
    confidence = float(np.clip(1.0 - abs(x_nm_per_px - y_nm_per_px) / max(0.5 * (x_nm_per_px + y_nm_per_px), 1e-12), 0.0, 1.0))
    return float(0.5 * (x_nm_per_px + y_nm_per_px)), confidence


def estimate_quality(content: np.ndarray) -> ImageQuality:
    gray = robust_uint8(content)
    h, w = gray.shape
    lap = ndi.laplace(gray.astype(np.float32))
    sharpness = float(np.var(lap))
    p5, p95 = np.percentile(gray, [5, 95])
    contrast = float((p95 - p5) / 255.0)
    saturation = float(((gray <= 2) | (gray >= 253)).mean())

    # A rough minimum fiber-width estimate from gradient autocorrelation. It is
    # intentionally advisory and never used as a hard detector threshold.
    smooth = ndi.gaussian_filter(gray.astype(np.float32), 1.0)
    gy, gx = np.gradient(smooth)
    gm = np.hypot(gx, gy)
    high = gm > np.percentile(gm, 85)
    distance = ndi.distance_transform_edt(~high)
    values = 2.0 * distance[(distance > 0.5) & (distance < 12)]
    estimated = float(np.percentile(values, 18)) if values.size else None

    messages: list[str] = []
    if max(h, w) < 1200:
        messages.append("긴 변 1200px 이상을 권장합니다.")
    if sharpness < 90:
        messages.append("영상이 흐릴 수 있습니다. 초점이 더 선명한 원본을 권장합니다.")
    if contrast < 0.30:
        messages.append("fiber와 배경의 명암 대비가 낮습니다.")
    if estimated is not None and estimated < 6:
        messages.append("가장 얇은 fiber가 약 6px 미만으로 보여 edge 오차가 커질 수 있습니다.")
    if saturation > 0.12:
        messages.append("검정 또는 흰색으로 포화된 픽셀이 많습니다.")

    if not messages:
        label = "양호"
    elif len(messages) <= 2:
        label = "보통"
    else:
        label = "주의"
    return ImageQuality(
        label=label,
        width_px=int(w),
        height_px=int(h),
        longest_side_px=int(max(h, w)),
        sharpness=sharpness,
        contrast=contrast,
        saturation_fraction=saturation,
        estimated_min_fiber_width_px=estimated,
        messages=tuple(messages),
    )


def extract_sem_content_and_metadata(
    image: np.ndarray,
    *,
    auto_calibrate: bool = True,
) -> MetadataExtraction:
    footer_start = detect_footer_boundary(image)
    if footer_start is None:
        content = np.asarray(image).copy()
        footer = None
    else:
        content = np.asarray(image)[:footer_start].copy()
        footer = np.asarray(image)[footer_start:].copy()

    calibration = ScaleCalibration(False, None, None, None, None, None, None, "", 0.0)
    if auto_calibrate and footer is not None and footer.size:
        bar_length, bbox, bar_conf = detect_scale_bar(footer)
        value, unit, text, ocr_conf = ocr_scale_value(footer, bbox)
        value_nm = None
        nm_per_px = None
        fov_nm_per_px, fov_conf = parse_fov_calibration(text, content.shape)
        if fov_nm_per_px is not None:
            nm_per_px = fov_nm_per_px
        elif value is not None and unit in _UNIT_TO_NM and bar_length is not None and bar_length > 0:
            value_nm = float(value) * _UNIT_TO_NM[unit]
            nm_per_px = value_nm / float(bar_length)
        confidence = (
            float(np.clip(0.75 + 0.25 * fov_conf, 0.0, 1.0))
            if fov_nm_per_px is not None
            else float(np.clip(0.55 * bar_conf + 0.45 * ocr_conf, 0.0, 1.0))
        )
        calibration = ScaleCalibration(
            detected=nm_per_px is not None,
            nm_per_px=nm_per_px,
            scale_value=value,
            scale_unit=unit,
            scale_value_nm=value_nm,
            bar_length_px=bar_length,
            bar_bbox=bbox,
            ocr_text=text,
            confidence=confidence,
        )

    return MetadataExtraction(
        content=content,
        footer=footer,
        footer_start_y=footer_start,
        calibration=calibration,
        quality=estimate_quality(content),
    )
