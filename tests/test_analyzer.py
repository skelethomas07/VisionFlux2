from io import BytesIO

import numpy as np
import pandas as pd
from PIL import Image

from pipeline.analyzer import decode_and_scale_image, normalize_analysis_tables


def png_bytes(width=100, height=50):
    arr = np.arange(width * height, dtype=np.uint8).reshape(height, width)
    bio = BytesIO()
    Image.fromarray(arr).save(bio, format="PNG")
    return bio.getvalue()


def test_decode_and_scale_image_preserves_aspect_ratio():
    decoded = decode_and_scale_image(png_bytes(), "sample.png", max_dimension=50)
    assert decoded.image.shape == (25, 50)
    assert decoded.analysis_scale == 0.5
    assert decoded.original_shape == (50, 100)


def test_normalize_analysis_tables_adds_stable_review_columns():
    local = pd.DataFrame([
        dict(fiber_region_id=2, region_sample_index=3, width_px=5.0,
             center_x=4.0, center_y=6.0, x1=2.0, y1=6.0, x2=7.0, y2=6.0),
    ])
    regions = pd.DataFrame([dict(fiber_region_id=2, median_width_px=5.0)])
    reps = pd.DataFrame([dict(
        fiber_region_id=2, subregion_id=1, representative_width_px=5.0,
        fiber_count_weight=1.0,
    )])
    local_n, regions_n, reps_n = normalize_analysis_tables(
        local, regions, reps, analysis_scale=0.5, nm_per_px=2.0,
    )
    assert local_n.measurement_id.iloc[0] == "auto-r2-s3"
    assert local_n.status.iloc[0] == "active"
    assert local_n.width_original_px.iloc[0] == 10.0
    assert local_n.width_nm.iloc[0] == 20.0
    assert regions_n.median_width_original_px.iloc[0] == 10.0
    assert reps_n.representative_width_original_px.iloc[0] == 10.0


def test_merge_orientation_measurements_adds_rescue_rows_with_review_schema():
    from pipeline.analyzer import merge_orientation_measurements

    local = pd.DataFrame([
        dict(measurement_id="auto-r1-s0", fiber_region_id=1, region_sample_index=0,
             x1=1.0, y1=0.0, x2=1.0, y2=4.0, center_x=1.0, center_y=2.0,
             width_px=4.0, width_original_px=8.0, status="active", source="auto"),
    ])
    rescue = pd.DataFrame([
        dict(fiber_region_id="orientation-1", region_sample_index=0,
             x1=5.0, y1=0.0, x2=5.0, y2=6.0, center_x=5.0, center_y=3.0,
             width_px=6.0, confidence=0.8),
    ])
    merged = merge_orientation_measurements(local, rescue, analysis_scale=0.5, nm_per_px=2.0)
    added = merged[merged.source.astype(str) == "orientation"].iloc[0]
    assert added.measurement_id.startswith("orientation-r")
    assert added.status == "active"
    assert added.width_original_px == 12.0
    assert added.width_nm == 24.0


def test_run_uploaded_analysis_emits_monotonic_stage_progress(monkeypatch):
    import pipeline.analyzer as analyzer

    local = pd.DataFrame([
        dict(fiber_region_id=1, region_sample_index=0, width_px=4.0,
             center_x=4.0, center_y=4.0, x1=4.0, y1=2.0, x2=4.0, y2=6.0,
             confidence=0.8),
    ])
    regions = pd.DataFrame([dict(fiber_region_id=1, median_width_px=4.0)])
    reps = pd.DataFrame([dict(fiber_region_id=1, subregion_id=1,
                              representative_width_px=4.0, fiber_count_weight=1.0)])
    candidates = pd.DataFrame()
    image = np.zeros((16, 16), np.float32)

    monkeypatch.setattr(
        analyzer.legacy_pipeline,
        "process_one",
        lambda path: ({"ok": True}, local.copy(), regions.copy(), reps.copy(), candidates.copy()),
    )
    monkeypatch.setattr(
        analyzer.legacy_pipeline,
        "prepare_image",
        lambda path: (image.copy(), None),
    )
    monkeypatch.setattr(analyzer, "orientation_guided_rescue", lambda *args, **kwargs: pd.DataFrame())

    events = []
    result = analyzer.run_uploaded_analysis(
        png_bytes(width=16, height=16),
        "sample.png",
        max_dimension=16,
        prefer_gpu=False,
        progress_callback=lambda fraction, message: events.append((fraction, message)),
    )

    fractions = [fraction for fraction, _ in events]
    assert fractions[0] == 0.0
    assert fractions[-1] == 1.0
    assert fractions == sorted(fractions)
    assert len(events) >= 7
    assert result.summary["compute_backend"] == "CPU"
