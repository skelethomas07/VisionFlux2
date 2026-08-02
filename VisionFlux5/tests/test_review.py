from io import BytesIO
import zipfile

import numpy as np
import pandas as pd

from pipeline.review import (
    build_session_zip,
    distance_measurement,
    recompute_representatives,
    reject_measurement,
    replace_with_manual,
    split_region_at_measurement,
)


def sample_measurements():
    return pd.DataFrame([
        dict(measurement_id="m1", fiber_region_id=1, region_sample_index=0,
             center_x=1.0, center_y=1.0, x1=0.0, y1=1.0, x2=2.0, y2=1.0,
             width_px=2.0, status="active", source="auto"),
        dict(measurement_id="m2", fiber_region_id=1, region_sample_index=1,
             center_x=2.0, center_y=1.0, x1=1.0, y1=1.0, x2=3.0, y2=1.0,
             width_px=2.0, status="active", source="auto"),
        dict(measurement_id="m3", fiber_region_id=1, region_sample_index=2,
             center_x=3.0, center_y=1.0, x1=2.0, y1=1.0, x2=4.0, y2=1.0,
             width_px=2.0, status="active", source="auto"),
    ])


def test_distance_measurement_converts_analysis_pixels_to_original_and_nm():
    result = distance_measurement((0, 0), (3, 4), analysis_scale=0.5, nm_per_px=2.0)
    assert result["analysis_width_px"] == 5.0
    assert result["original_width_px"] == 10.0
    assert result["width_nm"] == 20.0


def test_reject_measurement_excludes_only_selected_row():
    updated, event = reject_measurement(sample_measurements(), "m2", "bundle")
    assert updated.loc[updated.measurement_id == "m2", "status"].item() == "rejected"
    assert updated.loc[updated.measurement_id == "m1", "status"].item() == "active"
    assert event["action"] == "bundle"


def test_replace_with_manual_marks_old_row_and_inherits_region():
    updated, event, new_id = replace_with_manual(
        sample_measurements(), "m2", (10, 10), (16, 18),
        analysis_scale=0.5, nm_per_px=1.5,
    )
    old = updated.loc[updated.measurement_id == "m2"].iloc[0]
    new = updated.loc[updated.measurement_id == new_id].iloc[0]
    assert old.status == "corrected"
    assert new.source == "manual"
    assert new.status == "active"
    assert new.fiber_region_id == 1
    assert np.isclose(new.width_px, 10.0)
    assert np.isclose(new.width_original_px, 20.0)
    assert np.isclose(new.width_nm, 30.0)
    assert event["action"] == "manual_replace"


def test_split_region_at_selected_measurement_moves_selected_and_later_rows():
    updated, event, new_region = split_region_at_measurement(sample_measurements(), "m2")
    assert updated.loc[updated.measurement_id == "m1", "fiber_region_id"].item() == 1
    assert updated.loc[updated.measurement_id == "m2", "fiber_region_id"].item() == new_region
    assert updated.loc[updated.measurement_id == "m3", "fiber_region_id"].item() == new_region
    assert event["action"] == "split_region"


def test_representatives_give_each_region_total_weight_one_and_split_taper():
    rows = []
    widths = [4, 4, 4.2, 4.1, 4, 8, 8.1, 8, 8.2, 8]
    for i, width in enumerate(widths):
        rows.append(dict(
            measurement_id=f"m{i}", fiber_region_id=7, region_sample_index=i,
            center_x=float(i), center_y=0.0, width_px=float(width),
            status="active", source="auto",
        ))
    reps = recompute_representatives(pd.DataFrame(rows), analysis_scale=1.0, nm_per_px=2.0)
    assert len(reps) == 2
    assert np.isclose(reps.fiber_count_weight.sum(), 1.0)
    assert reps.representative_width_px.iloc[0] < reps.representative_width_px.iloc[1]
    assert np.allclose(
        reps.representative_width_nm,
        reps.representative_width_original_px * 2.0,
    )


def test_session_zip_contains_corrected_tables_and_feedback():
    measurements = sample_measurements()
    reps = recompute_representatives(measurements, analysis_scale=1.0, nm_per_px=None)
    blob = build_session_zip("sample.tif", measurements, reps, [{"action": "accepted"}])
    with zipfile.ZipFile(BytesIO(blob)) as archive:
        names = set(archive.namelist())
    assert "sample_corrected_measurements.csv" in names
    assert "sample_region_representatives.csv" in names
    assert "sample_feedback.json" in names
