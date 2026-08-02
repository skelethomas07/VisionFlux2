import base64

import numpy as np

from ui.measurement_canvas import image_to_data_url, normalize_canvas_payload


def test_image_to_data_url_is_png():
    url = image_to_data_url(np.zeros((10, 12), dtype=np.uint8))
    assert url.startswith("data:image/png;base64,")
    raw = base64.b64decode(url.split(",", 1)[1])
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"


def test_normalize_canvas_payload_filters_invalid_lines_and_ids():
    payload = normalize_canvas_payload({
        "new_measurements": [
            {"p1": [1, 2], "p2": [4, 6]},
            {"p1": [1], "p2": [2, 3]},
        ],
        "delete_ids": ["a", "a", None, "b"],
    })
    assert payload["new_measurements"] == [{"p1": [1.0, 2.0], "p2": [4.0, 6.0]}]
    assert payload["delete_ids"] == ["a", "b"]


def test_canvas_source_contains_requested_interactions():
    from ui.measurement_canvas import _HTML, _JS

    assert 'data-tool="modify"' in _HTML
    assert 'data-action="labels"' in _HTML
    assert '1500' in _JS
    assert 'path_points' in _JS
    assert 'magnifier.style.left' in _JS
    assert 'drawMagnifier' in _JS
    assert 'normalGuide' in _JS
    assert 'labelsEnabled' in _JS
    assert "line.source!=='manual'" in _JS
    assert "on_autosave_change" not in _JS  # Python wrapper owns callback registration


def test_normalize_canvas_payload_keeps_model_path_metadata_and_state():
    payload = normalize_canvas_payload({
        "new_measurements": [{
            "p1": [1, 2], "p2": [4, 8],
            "fiber_region_id": 7, "fiber_path_id": "path-7",
            "direction_deg": 12.5, "replacement_for": "rep-7",
        }],
        "delete_ids": ["m1"],
        "canvas_state": {"sectorLayout": "3x2", "labelsEnabled": False},
    })
    row = payload["new_measurements"][0]
    assert row["fiber_region_id"] == "7"
    assert row["fiber_path_id"] == "path-7"
    assert row["direction_deg"] == 12.5
    assert payload["canvas_state"]["sectorLayout"] == "3x2"
