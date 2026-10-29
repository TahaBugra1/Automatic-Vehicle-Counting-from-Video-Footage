"""Offline tests for the Ultralytics adapter: no weights, no network, ultralytics not imported."""

import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pytest
import yaml
from fakes import make_video

from visionflow import tracking
from visionflow.models import VehicleClass
from visionflow.tracking import (
    TRACK_LOW_THRESH,
    UltralyticsTrackSource,
    _to_frame_result,
    tracker_config,
)

BASE = {
    "tracker_type": "bytetrack",
    "track_high_thresh": 0.25,
    "track_low_thresh": 0.1,
    "new_track_thresh": 0.25,
    "track_buffer": 30,
    "match_thresh": 0.8,
    "fuse_score": True,
}


class FakeTensor:
    """Just enough of a torch tensor: `.int()` and `.tolist()`."""

    def __init__(self, values):
        self.values = values

    def int(self):
        return FakeTensor([[int(v) for v in row] if isinstance(row, list) else int(row)
                           for row in self.values])  # fmt: skip

    def tolist(self):
        return list(self.values)


def boxes(ids, classes, confs, xyxy):
    return types.SimpleNamespace(
        id=None if ids is None else FakeTensor(ids),
        cls=FakeTensor([float(c) for c in classes]),
        conf=FakeTensor(confs),
        xyxy=FakeTensor(xyxy),
    )


def result(box_set, width=64, height=48):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    return types.SimpleNamespace(orig_shape=(height, width), boxes=box_set, orig_img=image)


# --- _to_frame_result ---------------------------------------------------------------------


def test_vehicles_are_mapped_with_ids_classes_and_bottom_center():
    b = boxes(
        [4, 7, 9, 11],
        [2, 3, 5, 7],
        [0.9, 0.8, 0.7, 0.6],
        [[0, 0, 10, 20], [10, 10, 30, 40], [1, 2, 3, 4], [5, 6, 7, 8]],
    )
    frame = _to_frame_result(3, 0.12, result(b))
    assert (frame.frame, frame.time_s, frame.width, frame.height) == (3, 0.12, 64, 48)
    assert [d.track_id for d in frame.detections] == [4, 7, 9, 11]
    assert [d.cls for d in frame.detections] == [
        VehicleClass.car,
        VehicleClass.motorcycle,
        VehicleClass.bus,
        VehicleClass.truck,
    ]
    first = frame.detections[0]
    assert first.bottom_center == (5.0, 20.0)
    assert first.confidence == 0.9 and first.frame == 3 and first.time_s == 0.12
    assert frame.image.shape == (48, 64, 3)


def test_boxes_without_track_ids_give_no_detections():
    b = boxes(None, [2], [0.9], [[0, 0, 10, 10]])
    assert _to_frame_result(0, 0.0, result(b)).detections == []


def test_non_vehicle_class_is_filtered():
    b = boxes([1, 2], [0, 2], [0.9, 0.8], [[0, 0, 1, 1], [0, 0, 2, 2]])  # 0 = person
    dets = _to_frame_result(0, 0.0, result(b)).detections
    assert [(d.track_id, d.cls) for d in dets] == [(2, VehicleClass.car)]


def test_empty_result_gives_no_detections():
    assert _to_frame_result(0, 0.0, result(None)).detections == []
    empty = boxes([], [], [], [])
    assert _to_frame_result(0, 0.0, result(empty)).detections == []


# --- tracker config (I3) -------------------------------------------------------------------


def test_tracker_config_uses_user_confidence_for_high_and_new_track():
    cfg = tracker_config(0.4, BASE)
    assert cfg["track_high_thresh"] == 0.4
    assert cfg["new_track_thresh"] == 0.4
    assert cfg["track_low_thresh"] == TRACK_LOW_THRESH == 0.1
    for key in ("tracker_type", "track_buffer", "match_thresh", "fuse_score"):
        assert cfg[key] == BASE[key]
    assert BASE["track_high_thresh"] == 0.25  # input not mutated


def test_tracker_config_clamps_low_threshold_below_high():
    cfg = tracker_config(0.05, BASE)
    assert cfg["track_high_thresh"] == cfg["new_track_thresh"] == 0.05
    assert cfg["track_low_thresh"] == 0.05


def test_installed_bytetrack_defaults_are_read():
    defaults = tracking.bytetrack_defaults()
    assert defaults["tracker_type"] == "bytetrack"
    assert {"track_high_thresh", "track_low_thresh", "new_track_thresh"} <= set(defaults)


def test_bytetrack_defaults_fall_back_when_ultralytics_is_missing(monkeypatch):
    monkeypatch.setattr(tracking.importlib.util, "find_spec", lambda name: None)
    defaults = tracking.bytetrack_defaults()
    assert defaults == tracking.FALLBACK_BYTETRACK


# --- UltralyticsTrackSource with a fake YOLO --------------------------------------------------


class FakeResults:
    def __init__(self, items):
        self.items = items
        self.closed = False

    def __iter__(self):
        return iter(self.items)

    def close(self):
        self.closed = True


class FakeYOLO:
    instances: list["FakeYOLO"] = []

    def __init__(self, model_name):
        self.model_name = model_name
        self.kwargs = None
        self.tracker_yaml = None
        self.results = FakeResults([result(boxes([1], [2], [0.9], [[0, 0, 10, 10]]))] * 3)
        FakeYOLO.instances.append(self)

    def track(self, **kwargs):
        self.kwargs = kwargs
        tracker = Path(kwargs["tracker"])
        self.tracker_yaml = yaml.safe_load(tracker.read_text(encoding="utf-8"))
        return self.results


@pytest.fixture
def fake_ultralytics(monkeypatch):
    FakeYOLO.instances = []
    module = types.ModuleType("ultralytics")
    module.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", module)
    monkeypatch.setattr(tracking, "bytetrack_defaults", lambda: dict(BASE))
    return FakeYOLO


def test_track_is_called_with_low_detector_conf_and_per_run_tracker(tmp_path, fake_ultralytics):
    video = make_video(tmp_path / "v.avi")
    frames = list(UltralyticsTrackSource().frames(video, "yolo11n.pt", 0.4))
    assert len(frames) == 3
    (model,) = fake_ultralytics.instances
    assert model.model_name == "yolo11n.pt"
    kw = model.kwargs
    assert kw["conf"] == pytest.approx(0.1)
    assert kw["stream"] is True and kw["persist"] is True
    assert kw["classes"] == [2, 3, 5, 7]
    assert kw["source"] == str(video)
    assert model.tracker_yaml["track_high_thresh"] == 0.4
    assert model.tracker_yaml["new_track_thresh"] == 0.4
    assert model.tracker_yaml["track_low_thresh"] == 0.1
    tracker_path = Path(kw["tracker"])
    assert not tracker_path.exists()  # temp YAML removed after the run
    assert Path(tempfile.gettempdir()).resolve() in tracker_path.resolve().parents
    assert model.results.closed


def test_detector_conf_follows_a_low_user_threshold(tmp_path, fake_ultralytics):
    video = make_video(tmp_path / "v.avi")
    list(UltralyticsTrackSource().frames(video, "yolo11s.pt", 0.05))
    assert fake_ultralytics.instances[0].kwargs["conf"] == pytest.approx(0.05)


def test_generator_closes_results_and_removes_yaml_when_consumer_stops(tmp_path, fake_ultralytics):
    video = make_video(tmp_path / "v.avi")
    gen = UltralyticsTrackSource().frames(video, "yolo11s.pt", 0.25)
    next(gen)
    gen.close()
    (model,) = fake_ultralytics.instances
    assert model.results.closed
    assert not Path(model.kwargs["tracker"]).exists()
