import contextlib
import importlib.util
import math
import os
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
import yaml

from .models import Detection, VehicleClass

DEFAULT_FPS = 25.0
COCO_TO_VEHICLE = {
    2: VehicleClass.car,
    3: VehicleClass.motorcycle,
    5: VehicleClass.bus,
    7: VehicleClass.truck,
}


class VideoReadError(Exception):
    """Raised when a video file cannot be opened or decoded."""


@dataclass(frozen=True, slots=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int
    duration_s: float


@dataclass(frozen=True, slots=True)
class FrameResult:
    frame: int
    time_s: float
    width: int
    height: int
    detections: list[Detection]
    image: np.ndarray


class TrackSource(Protocol):
    def frames(
        self, video_path: Path, model_name: str, confidence: float
    ) -> Iterator[FrameResult]: ...


def _valid_fps(fps: float) -> float:
    return fps if math.isfinite(fps) and fps > 0 else DEFAULT_FPS


def probe_video(path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise VideoReadError(f"cannot open video: {path.name}")
        ok, image = cap.read()
        if not ok or image is None:
            raise VideoReadError(f"cannot read frames from video: {path.name}")
        height, width = image.shape[:2]
        fps = _valid_fps(float(cap.get(cv2.CAP_PROP_FPS)))
        raw_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = int(raw_count) if math.isfinite(raw_count) and raw_count > 0 else 0
    finally:
        cap.release()
    return VideoInfo(width, height, fps, frame_count, frame_count / fps)


# ByteTrack's second (low-score) association stage only sees boxes above this score.
TRACK_LOW_THRESH = 0.1
# Copy of Ultralytics 8.4 `cfg/trackers/bytetrack.yaml`, used only if that file is not found.
FALLBACK_BYTETRACK: dict[str, object] = {
    "tracker_type": "bytetrack",
    "track_high_thresh": 0.25,
    "track_low_thresh": 0.1,
    "new_track_thresh": 0.25,
    "track_buffer": 30,
    "match_thresh": 0.8,
    "fuse_score": True,
}


def bytetrack_defaults() -> dict[str, object]:
    """Keys of the installed Ultralytics `bytetrack.yaml`, located without importing ultralytics."""
    try:
        spec = importlib.util.find_spec("ultralytics")
    except (ImportError, ValueError):
        spec = None
    locations = list(spec.submodule_search_locations or []) if spec is not None else []
    for location in locations:
        path = Path(location) / "cfg" / "trackers" / "bytetrack.yaml"
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError):
            continue
        if isinstance(data, dict) and data.get("tracker_type") == "bytetrack":
            return data
    return dict(FALLBACK_BYTETRACK)


def tracker_config(confidence: float, base: dict[str, object]) -> dict[str, object]:
    """Per-run ByteTrack settings: the user's threshold applies inside the tracker.

    `track_high_thresh` (first association stage) and `new_track_thresh` (starting a track)
    use the user's confidence; `track_low_thresh` stays at 0.1 (never above the high one), so
    the detector can deliver low-score boxes for ByteTrack's second stage.
    """
    cfg = dict(base)
    cfg["track_high_thresh"] = confidence
    cfg["new_track_thresh"] = confidence
    cfg["track_low_thresh"] = min(TRACK_LOW_THRESH, confidence)
    return cfg


def _write_tracker_yaml(cfg: dict[str, object]) -> Path:
    """Write the config to a temp file in the system temp folder (never inside the repo)."""
    fd, name = tempfile.mkstemp(prefix="visionflow_bytetrack_", suffix=".yaml")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return Path(name)


class UltralyticsTrackSource:
    """YOLO11 detection + ByteTrack tracking, restricted to COCO vehicle classes."""

    def frames(self, video_path: Path, model_name: str, confidence: float) -> Iterator[FrameResult]:
        from ultralytics import YOLO

        fps = probe_video(video_path).fps
        cfg = tracker_config(confidence, bytetrack_defaults())
        tracker_path = _write_tracker_yaml(cfg)
        try:
            # A fresh model per video so tracker state never leaks between videos.
            model = YOLO(model_name)
            results = model.track(
                source=str(video_path),
                stream=True,
                persist=True,
                tracker=str(tracker_path),
                classes=list(COCO_TO_VEHICLE),
                conf=min(confidence, float(cfg["track_low_thresh"])),
                verbose=False,
            )
            # Close the stream deterministically (releases the video reader) when the consumer
            # stops early or raises, instead of waiting for garbage collection.
            with contextlib.closing(results):
                for index, result in enumerate(results):
                    yield _to_frame_result(index, index / fps, result)
        finally:
            tracker_path.unlink(missing_ok=True)


def _to_frame_result(index: int, time_s: float, result) -> FrameResult:
    height, width = result.orig_shape[:2]
    detections: list[Detection] = []
    boxes = result.boxes
    if boxes is not None and boxes.id is not None:
        ids = boxes.id.int().tolist()
        classes = boxes.cls.int().tolist()
        confs = boxes.conf.tolist()
        for track_id, cls, conf, xyxy in zip(ids, classes, confs, boxes.xyxy.tolist()):
            vehicle = COCO_TO_VEHICLE.get(cls)
            if vehicle is None:
                continue
            bbox = (xyxy[0], xyxy[1], xyxy[2], xyxy[3])
            detections.append(Detection(index, time_s, track_id, vehicle, conf, bbox))
    return FrameResult(index, time_s, width, height, detections, result.orig_img)
