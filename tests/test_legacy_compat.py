import numpy as np

from pipeline.legacy_pipeline import weighted_histogram_bin_count


def test_weighted_histogram_uses_explicit_bin_count():
    values = np.array([1.0, 2.0, 3.0])
    weights = np.array([1.0, 0.5, 1.0])
    bins = weighted_histogram_bin_count(values)
    counts, edges = np.histogram(values, bins=bins, weights=weights)
    assert isinstance(bins, int)
    assert counts.sum() == weights.sum()
    assert len(edges) == bins + 1
