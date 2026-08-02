import json
from types import SimpleNamespace

import pandas as pd

from services.collaboration import (
    CollaborationConfig,
    apply_snapshot_to_item,
    deserialize_snapshot,
    serialize_snapshot,
)


def test_collaboration_config_from_mapping():
    config = CollaborationConfig.from_mapping({
        "url": "https://example.supabase.co",
        "key": "anon-key",
        "project_id": "chem-frontier",
        "table": "visionflux_reviews",
    })
    assert config.project_id == "chem-frontier"
    assert config.table == "visionflux_reviews"


def test_snapshot_is_deterministic_and_roundtrips():
    item = SimpleNamespace(
        item_id="abc",
        analysis=SimpleNamespace(image_name="2-7.jpg"),
        measurements=pd.DataFrame([{"measurement_id": "m1", "status": "active", "width_px": 4.0}]),
        feedback=[{"action": "manual_add"}],
        revision=3,
        nm_per_px=10.0,
    )
    first = serialize_snapshot(item, worker_name="A", status="in_progress")
    second = serialize_snapshot(item, worker_name="A", status="in_progress")
    assert first == second
    parsed = deserialize_snapshot(first)
    assert parsed["revision"] == 3
    assert parsed["measurements"][0]["measurement_id"] == "m1"


def test_apply_snapshot_replaces_review_state_and_recomputes(monkeypatch):
    item = SimpleNamespace(
        measurements=pd.DataFrame([{"measurement_id": "old", "status": "active", "width_px": 2.0}]),
        feedback=[],
        revision=0,
        nm_per_px=None,
        representatives=pd.DataFrame(),
        last_apply_token="x",
    )
    called = {"value": False}
    monkeypatch.setattr("services.collaboration.recompute_review_item", lambda current: called.__setitem__("value", True))
    snapshot = json.dumps({
        "measurements": [{"measurement_id": "new", "status": "active", "width_px": 7.0}],
        "feedback": [{"action": "loaded"}],
        "revision": 4,
        "nm_per_px": 2.5,
    })
    apply_snapshot_to_item(item, snapshot)
    assert item.measurements.iloc[0].measurement_id == "new"
    assert item.revision == 4
    assert item.nm_per_px == 2.5
    assert item.last_apply_token is None
    assert called["value"]
