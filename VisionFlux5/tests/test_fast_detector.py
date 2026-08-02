import time

import numpy as np
import pandas as pd

from pipeline.fast_detector import (
    FastDetectorConfig,
    build_pore_core,
    detect_fibers_fast,
)


def _draw_gaussian_curve(image, xs, ys, sigma=2.2, amplitude=0.9):
    yy, xx = np.mgrid[: image.shape[0], : image.shape[1]]
    for x, y in zip(xs, ys):
        image += amplitude * np.exp(-0.5 * (((xx - x) ** 2 + (yy - y) ** 2) / sigma**2))


def curved_fiber_image(h=180, w=240):
    image = np.full((h, w), 0.08, np.float32)
    xs = np.linspace(20, w - 20, 180)
    ys = 85 + 34 * np.sin((xs - 20) / (w - 40) * np.pi)
    _draw_gaussian_curve(image, xs, ys, sigma=2.4)
    return np.clip(image, 0, 1)


def parallel_fiber_image(h=150, w=220):
    yy, xx = np.mgrid[:h, :w]
    image = np.full((h, w), 0.05, np.float32)
    image += 0.9 * np.exp(-0.5 * ((yy - 58) / 2.3) ** 2)
    image += 0.9 * np.exp(-0.5 * ((yy - 72) / 2.3) ** 2)
    # Strong dark pore core between the two bright fibers.
    image -= 0.06 * np.exp(-0.5 * ((yy - 65) / 2.0) ** 2)
    return np.clip(image, 0, 1)


def test_pore_core_marks_dark_gap_but_not_bright_fibers():
    image = parallel_fiber_image()
    result = detect_fibers_fast(image, prefer_gpu=False)
    pore = result.pore_core
    assert pore[:, 65:155].mean() > 0.005
    assert pore[62:69, 65:155].mean() > pore[55:61, 65:155].mean()


def test_curved_fiber_keeps_one_path_and_multiple_direction_segments():
    result = detect_fibers_fast(
        curved_fiber_image(),
        config=FastDetectorConfig(sample_spacing_px=6.0, min_path_length_px=24.0),
        prefer_gpu=False,
    )
    local = result.measurements
    assert len(local) >= 12
    main = local.groupby("fiber_path_id").size().sort_values(ascending=False).index[0]
    main_rows = local[local.fiber_path_id == main]
    assert main_rows.direction_segment_id.nunique() >= 2
    assert main_rows.direction_deg.max() - main_rows.direction_deg.min() >= 12


def test_parallel_fibers_are_not_merged_into_one_thick_measurement():
    result = detect_fibers_fast(
        parallel_fiber_image(),
        config=FastDetectorConfig(sample_spacing_px=8.0),
        prefer_gpu=False,
    )
    local = result.measurements
    assert len(local) >= 10
    assert local.width_px.median() < 10.0
    assert (local.width_px > 13.0).mean() < 0.1
    assert local.fiber_path_id.nunique() >= 2


def test_fast_detector_returns_review_compatible_schema():
    result = detect_fibers_fast(curved_fiber_image(120, 160), prefer_gpu=False)
    required = {
        "fiber_region_id", "fiber_path_id", "direction_segment_id",
        "region_sample_index", "center_x", "center_y", "x1", "y1", "x2", "y2",
        "width_px", "direction_deg", "local_coherency", "confidence", "grade", "detector",
    }
    assert required.issubset(result.measurements.columns)
    assert {"fiber_region_id", "median_width_px"}.issubset(result.regions.columns)
    assert {"fiber_region_id", "representative_width_px", "fiber_count_weight"}.issubset(
        result.representatives.columns
    )


def test_fast_detector_finishes_small_image_quickly():
    image = curved_fiber_image(160, 220)
    started = time.perf_counter()
    result = detect_fibers_fast(image, prefer_gpu=False)
    elapsed = time.perf_counter() - started
    assert len(result.measurements) > 0
    assert elapsed < 8.0
