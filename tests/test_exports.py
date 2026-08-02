from io import BytesIO

import numpy as np
from PIL import Image

from pipeline.exports import build_export_bundle, build_imagej_results


def test_imagej_results_use_intensity_and_straight_line_geometry():
    image = np.tile(np.arange(20, dtype=np.uint8), (20, 1))
    lines = [{
        "label": 1,
        "x1": 2.0,
        "y1": 10.0,
        "x2": 8.0,
        "y2": 10.0,
        "direction_deg": 90.0,
        "fiber_region_id": 4,
        "source": "auto",
    }]
    table, directions, length_unit, area_unit = build_imagej_results(
        image, lines, analysis_scale=1.0, nm_per_px=None,
    )
    assert list(table.columns) == ["label", "Area", "Mean", "Min", "Max", "Angle", "Length"]
    row = table.iloc[0]
    assert row["label"] == 1
    assert row["Length"] == 6.0
    assert row["Area"] == 7.0
    assert row["Min"] == 2.0
    assert row["Max"] == 8.0
    assert row["Mean"] == 5.0
    assert row["Angle"] == 0.0
    assert directions.iloc[0].fiber_direction_deg == 90.0
    assert length_unit == "px"
    assert area_unit == "px^2"


def test_calibrated_export_uses_original_coordinates_and_nm_units():
    original = np.full((100, 100), 128, np.uint8)
    lines = [{"label": 1, "x1": 5.0, "y1": 5.0, "x2": 10.0, "y2": 5.0, "source": "manual"}]
    bundle = build_export_bundle(
        original,
        lines,
        analysis_scale=0.5,
        nm_per_px=2.0,
        image_coordinates_are_original=True,
    )
    row = bundle.imagej_table.iloc[0]
    assert row.Length == 20.0  # 5 analysis px -> 10 original px -> 20 nm
    assert row.Area == 11.0 * 4.0  # 11 one-pixel samples, each 2nm x 2nm
    assert bundle.unit_length == "nm"
    assert bundle.unit_area == "nm^2"
    assert Image.open(BytesIO(bundle.annotated_png)).size == (100, 100)


def test_export_bundle_contains_labeled_and_unlabeled_images():
    image = np.full((40, 40), 120, np.uint8)
    lines = [{"label": 7, "x1": 8.0, "y1": 8.0, "x2": 20.0, "y2": 8.0, "source": "auto"}]
    bundle = build_export_bundle(image, lines, analysis_scale=1.0, nm_per_px=None)
    assert bundle.annotated_labeled_png.startswith(b"\x89PNG")
    assert bundle.annotated_unlabeled_png.startswith(b"\x89PNG")
    assert bundle.annotated_labeled_png != bundle.annotated_unlabeled_png
    assert bundle.annotated_png == bundle.annotated_labeled_png
