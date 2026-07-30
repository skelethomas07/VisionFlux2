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
