import numpy as np
import pandas as pd

from pipeline.orientation import (
    analyze_orientation,
    annotate_measurement_directions,
    axial_error_deg,
)


def horizontal_fibres(h=128, w=160, period=16):
    yy, xx = np.mgrid[:h, :w]
    img = 0.15 + 0.8 * (np.cos(2 * np.pi * yy / period) > 0.75)
    return img.astype(np.float32)


def test_orientation_analysis_finds_horizontal_direction():
    result = analyze_orientation(horizontal_fibres(), sigma_px=2.0)
    assert axial_error_deg(result.dominant_direction_deg, 0.0) < 8.0
    assert result.order_parameter > 0.65
    assert result.theta.shape == (128, 160)
    assert result.color_map.shape == (128, 160, 3)


def test_measurement_direction_is_tangent_not_thickness_chord():
    image = horizontal_fibres()
    result = analyze_orientation(image, sigma_px=2.0)
    measurements = pd.DataFrame([
        {
            "measurement_id": "m1",
            "x1": 40.0,
            "y1": 30.0,
            "x2": 40.0,
            "y2": 42.0,
            "center_x": 40.0,
            "center_y": 36.0,
            "width_px": 12.0,
            "confidence": 0.5,
        }
    ])
    annotated = annotate_measurement_directions(measurements, result)
    assert axial_error_deg(annotated.loc[0, "direction_deg"], 0.0) < 1e-6
    assert annotated.loc[0, "orientation_error_deg"] < 12.0
    assert 0.0 <= annotated.loc[0, "orientation_score"] <= 1.0


def test_orientation_guided_rescue_finds_distinct_bright_fibres():
    from pipeline.orientation import orientation_guided_rescue

    h, w = 120, 180
    yy, xx = np.mgrid[:h, :w]
    image = np.full((h, w), 0.08, np.float32)
    for center in (28, 60, 94):
        image += 0.85 * np.exp(-0.5 * ((yy - center) / 3.2) ** 2)
    image = np.clip(image, 0, 1)
    orientation = analyze_orientation(image, sigma_px=2.5)
    rescued = orientation_guided_rescue(
        image,
        orientation,
        existing_measurements=pd.DataFrame(),
        sample_spacing_px=10,
        max_candidates=120,
    )
    assert rescued.fiber_region_id.nunique() >= 3
    assert rescued.width_px.between(4.0, 14.0).mean() > 0.75
    assert np.nanmedian(np.abs(rescued.direction_deg)) < 8.0
