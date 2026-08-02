from types import SimpleNamespace

import pandas as pd

from pipeline.review_state import build_review_item, recompute_review_item


def fake_analysis(name, width):
    measurements = pd.DataFrame([
        dict(
            measurement_id=f"{name}-m1",
            fiber_region_id=1,
            region_sample_index=0,
            status="active",
            source="auto",
            width_px=width,
            width_original_px=width,
            x1=0.0,
            y1=0.0,
            x2=0.0,
            y2=width,
            center_x=0.0,
            center_y=width / 2,
        )
    ])
    return SimpleNamespace(
        image_name=name,
        analysis_scale=1.0,
        measurements=measurements,
    )


def test_build_review_item_copies_measurements_and_computes_representatives():
    analysis = fake_analysis("a.png", 6.0)
    item = build_review_item("a", analysis, nm_per_px=2.0)
    assert item.analysis.image_name == "a.png"
    assert item.measurements is not analysis.measurements
    assert len(item.representatives) == 1
    assert item.representatives.representative_width_nm.iloc[0] == 12.0


def test_review_items_are_isolated_from_each_other():
    first = build_review_item("a", fake_analysis("a.png", 6.0), nm_per_px=None)
    second = build_review_item("b", fake_analysis("b.png", 10.0), nm_per_px=None)
    first.measurements.loc[:, "status"] = "rejected"
    recompute_review_item(first)
    assert first.representatives.empty
    assert len(second.representatives) == 1
    assert second.measurements.status.iloc[0] == "active"
