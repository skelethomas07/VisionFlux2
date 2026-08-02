from pathlib import Path

import numpy as np
from PIL import Image

from pipeline.image_metadata import (
    detect_footer_boundary,
    extract_sem_content_and_metadata,
    parse_fov_calibration,
)


def test_footer_boundary_removes_dark_metadata_strip():
    image = np.full((200, 300), 140, np.uint8)
    rng = np.random.default_rng(3)
    image[:170] = np.clip(image[:170] + rng.normal(0, 25, (170, 300)), 0, 255)
    image[170:] = 0
    image[177:181, 210:280] = 255
    image[185:194, 20:100:4] = 255
    boundary = detect_footer_boundary(image)
    assert boundary is not None
    assert 168 <= boundary <= 172


def test_fov_calibration_uses_content_dimensions():
    nm_per_px, confidence = parse_fov_calibration("FOV:12.8x9.6um", (960, 1280))
    assert nm_per_px == 10.0
    assert confidence > 0.95


def test_uploaded_sem_example_excludes_footer_and_reads_fov_when_available():
    path = Path("/mnt/data/2-7_수작업.jpg")
    if not path.exists():
        return
    image = np.asarray(Image.open(path).convert("L"), np.float32)
    result = extract_sem_content_and_metadata(image, auto_calibrate=True)
    assert result.footer_start_y is not None
    assert result.content.shape[0] < image.shape[0]
    assert result.calibration.detected
    assert abs(result.calibration.nm_per_px - 10.0) < 0.2
