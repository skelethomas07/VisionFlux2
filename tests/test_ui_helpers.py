import numpy as np
import pandas as pd

from pipeline.orientation import analyze_orientation
from ui.figures import (
    build_distribution_figure,
    build_orientation_histogram,
    build_orientation_rose,
)


def test_distribution_figure_builds_from_representatives():
    reps = pd.DataFrame([
        dict(representative_width_original_px=4.0, fiber_count_weight=1.0),
        dict(representative_width_original_px=8.0, fiber_count_weight=1.0),
    ])
    distribution = build_distribution_figure(reps)
    assert len(distribution.data) == 1


def test_orientation_figures_build_from_analysis_result():
    image = np.tile(np.sin(np.linspace(0, 8 * np.pi, 80))[:, None], (1, 100)).astype(np.float32)
    result = analyze_orientation(image, sigma_px=2.0)
    histogram = build_orientation_histogram(result)
    rose = build_orientation_rose(result)
    assert len(histogram.data) >= 1
    assert len(rose.data) >= 1


def test_direction_segment_figure_uses_length_weighted_local_directions():
    from ui.figures import build_direction_segment_figure

    measurements = pd.DataFrame([
        {"direction_deg": 5.0, "sample_length_px": 20.0, "status": "active", "fiber_path_id": 1, "direction_segment_id": 0},
        {"direction_deg": 25.0, "sample_length_px": 5.0, "status": "active", "fiber_path_id": 1, "direction_segment_id": 1},
        {"direction_deg": -40.0, "sample_length_px": 100.0, "status": "rejected", "fiber_path_id": 2, "direction_segment_id": 0},
    ])
    fig = build_direction_segment_figure(measurements)
    assert len(fig.data) == 1
    assert float(np.sum(fig.data[0].y)) == 25.0
    assert "구간 길이" in fig.layout.yaxis.title.text


def test_thickness_direction_3d_counts_and_titles():
    from ui.figures import build_thickness_direction_3d

    lines = [
        {"direction_deg": 5.0, "width_original_px": 4.0},
        {"direction_deg": 7.0, "width_original_px": 4.2},
        {"direction_deg": 35.0, "width_original_px": 8.0},
    ]
    fig = build_thickness_direction_3d(lines, use_nm=False)
    assert len(fig.data) == 1
    assert float(np.asarray(fig.data[0].z).sum()) == 3.0
    assert "방향" in fig.layout.scene.xaxis.title.text
    assert "두께" in fig.layout.scene.yaxis.title.text
    assert "개수" in fig.layout.scene.zaxis.title.text
