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
