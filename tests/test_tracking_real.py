import os
from pathlib import Path

import pytest

from visionflow.tracking import UltralyticsTrackSource

pytestmark = pytest.mark.slow


def test_real_model_detects_at_least_one_vehicle():
    sample = os.environ.get("VISIONFLOW_SAMPLE_VIDEO")
    if not sample or not Path(sample).is_file():
        pytest.skip("set VISIONFLOW_SAMPLE_VIDEO to a short traffic video to run this test")
    source = UltralyticsTrackSource()
    detections = 0
    for frame in source.frames(Path(sample), "yolo11n.pt", 0.25):
        assert frame.image.shape[:2] == (frame.height, frame.width)
        detections += len(frame.detections)
        if detections:
            break
    assert detections > 0
