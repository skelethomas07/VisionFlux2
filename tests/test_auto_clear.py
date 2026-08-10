import pandas as pd

from pipeline.review import apply_canvas_edits
from ui.measurement_canvas import _HTML, _JS


def test_delete_all_auto_token_rejects_automatic_but_preserves_manual():
    measurements = pd.DataFrame([
        {"measurement_id": "auto-1", "status": "active", "review_label": "", "source": "auto"},
        {"measurement_id": "ori-1", "status": "accepted", "review_label": "", "source": "orientation"},
        {"measurement_id": "manual-1", "status": "active", "review_label": "manual", "source": "manual"},
    ])

    updated, events = apply_canvas_edits(
        measurements,
        new_measurements=[],
        delete_ids=["__VISIONFLUX_DELETE_ALL_AUTO__"],
    )

    by_id = updated.set_index("measurement_id")
    assert by_id.loc["auto-1", "status"] == "rejected"
    assert by_id.loc["ori-1", "status"] == "rejected"
    assert by_id.loc["manual-1", "status"] == "active"
    assert by_id.loc["manual-1", "review_label"] == "manual"
    assert any(
        event.get("action") == "erase_all_automatic_measurements" and event.get("count") == 2
        for event in events
    )


def test_canvas_has_confirmed_auto_only_clear_action():
    assert 'data-action="clear-auto"' in _HTML
    assert "__VISIONFLUX_DELETE_ALL_AUTO__" in _JS
    assert "window.confirm" in _JS
    assert "isAutoClearHidden" in _JS
    assert "line.source==='manual'" in _JS or "line.source !== 'manual'" in _JS
