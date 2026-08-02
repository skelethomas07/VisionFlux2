import numpy as np
import pandas as pd

from pipeline.review import (
    apply_canvas_edits,
    build_representative_lines,
    recompute_representatives,
)


def sample_measurements():
    rows = []
    for region, widths in [(1, [4.0, 4.2, 4.1]), (2, [8.0, 8.2, 7.9])]:
        for i, width in enumerate(widths):
            rows.append({
                "measurement_id": f"r{region}s{i}",
                "fiber_region_id": region,
                "region_sample_index": i,
                "center_x": 10.0 * region + i,
                "center_y": 20.0 * region,
                "x1": 10.0 * region + i,
                "y1": 20.0 * region - width / 2,
                "x2": 10.0 * region + i,
                "y2": 20.0 * region + width / 2,
                "width_px": width,
                "confidence": 0.8,
                "status": "active",
                "source": "auto",
            })
    return pd.DataFrame(rows)


def test_representative_lines_show_one_line_per_region_segment():
    measurements = sample_measurements()
    reps = recompute_representatives(measurements)
    lines = build_representative_lines(measurements, reps)
    assert len(lines) == 2
    assert all(len(item["erase_ids"]) == 3 for item in lines)
    assert {item["source"] for item in lines} == {"auto"}


def test_canvas_batch_adds_multiple_manual_lines_and_erases_region():
    measurements = sample_measurements()
    reps = recompute_representatives(measurements)
    lines = build_representative_lines(measurements, reps)
    region_one = next(item for item in lines if str(item["fiber_region_id"]) == "1")
    updated, events = apply_canvas_edits(
        measurements,
        new_measurements=[
            {"p1": [50.0, 50.0], "p2": [50.0, 56.0]},
            {"p1": [70.0, 70.0], "p2": [78.0, 70.0]},
        ],
        delete_ids=region_one["erase_ids"],
        analysis_scale=0.5,
        nm_per_px=2.0,
    )
    deleted = updated[updated.measurement_id.isin(region_one["erase_ids"])]
    manual = updated[updated.source.astype(str) == "manual"]
    assert set(deleted.status) == {"rejected"}
    assert len(manual) == 2
    assert manual.fiber_region_id.nunique() == 2
    assert np.allclose(sorted(manual.width_original_px), [12.0, 16.0])
    assert np.allclose(sorted(manual.width_nm), [24.0, 32.0])
    assert [event["action"] for event in events].count("manual_add") == 2


def test_representative_lines_relabel_without_gaps_after_region_removed():
    measurements = sample_measurements()
    measurements.loc[measurements.fiber_region_id == 1, "status"] = "rejected"
    reps = recompute_representatives(measurements)
    lines = build_representative_lines(measurements, reps)
    assert [line["label"] for line in lines] == [1]
    assert lines[0]["fiber_region_id"] == "2"
    assert len(lines[0]["path_points"]) == 3
