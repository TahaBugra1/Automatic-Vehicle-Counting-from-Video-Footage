import time
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import pytest

from visionflow.geometry import Line
from visionflow.models import Detection, Direction, VehicleClass
from visionflow.pipeline import ProcessOptions, process_video
from visionflow.summary import Reliability
from visionflow.suspicion import FlagKind
from visionflow.tracking import FrameResult, VideoReadError

FPS = 10.0
N_FRAMES = 10
LINE = Line((0.0, 0.5), (1.0, 0.5))
# Normalized bottom-center y per frame: track 1 moves down, track 2 moves up.
SCRIPT = {
    1: [0.2 + 0.06 * i for i in range(N_FRAMES)],
    2: [0.8 - 0.06 * i for i in range(N_FRAMES)],
}


def make_video(path: Path, width: int, height: int, frames: int = N_FRAMES) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (width, height))
    for _ in range(frames):
        writer.write(np.zeros((height, width, 3), dtype=np.uint8))
    writer.release()
    return path


class FakeSource:
    def __init__(self, width: int, height: int, script: dict[int, list[float]]):
        self.width = width
        self.height = height
        self.script = script

    def frames(self, video_path: Path, model_name: str, confidence: float) -> Iterator[FrameResult]:
        w, h = self.width, self.height
        for i in range(N_FRAMES):
            t = i / FPS
            dets = []
            for track_id, ys in self.script.items():
                y = ys[i] * h
                bbox = (0.4 * w, y - 0.1 * h, 0.5 * w, y)
                dets.append(Detection(i, t, track_id, VehicleClass.car, 0.9, bbox))
            image = np.zeros((h, w, 3), dtype=np.uint8)
            yield FrameResult(i, t, w, h, dets, image)


def run(tmp_path, width=640, height=360, script=SCRIPT, **kwargs):
    video = make_video(tmp_path / f"v_{width}x{height}.avi", width, height)
    return process_video(video, LINE, ProcessOptions(), FakeSource(width, height, script), **kwargs)


def test_pipeline_counts_scripted_crossings(tmp_path):
    seen = []
    result = run(tmp_path, on_frame=lambda fr, events: seen.extend(events))
    assert result.summary.total[Direction.ONE] == 1
    assert result.summary.total[Direction.TWO] == 1
    assert {e.track_id for e in result.events} == {1, 2}
    assert seen == result.events
    assert result.fps == FPS
    assert result.duration_s == pytest.approx(N_FRAMES / FPS)
    assert result.processing_fps > 0


def test_pipeline_with_no_detections_returns_zero_and_unknown_reliability(tmp_path):
    result = run(tmp_path, script={})
    assert result.summary.total_count == 0
    assert result.events == []
    assert result.summary.reliability == Reliability.UNKNOWN


def test_line_scales_with_resolution(tmp_path):
    small = run(tmp_path, 640, 360)
    large = run(tmp_path, 1280, 720)
    assert small.summary.total == large.summary.total
    assert small.summary.total_count == large.summary.total_count == 2


def test_progress_reaches_one(tmp_path):
    progress: list[float] = []
    run(tmp_path, on_progress=progress.append)
    assert progress[-1] == 1.0
    assert all(0.0 <= p <= 1.0 for p in progress)
    assert progress == sorted(progress)


def test_unreadable_video_raises_video_read_error(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"this is not a video" * 100)
    with pytest.raises(VideoReadError):
        process_video(bad, LINE, ProcessOptions(), FakeSource(640, 360, SCRIPT))


class SlowStartSource:
    """Waits (like loading a model) before the first frame, then yields frames quickly."""

    def __init__(self, n_frames: int, delay_s: float):
        self.n_frames = n_frames
        self.delay_s = delay_s

    def frames(self, video_path: Path, model_name: str, confidence: float) -> Iterator[FrameResult]:
        time.sleep(self.delay_s)
        for i in range(self.n_frames):
            yield FrameResult(i, i / FPS, 64, 48, [], np.zeros((48, 64, 3), dtype=np.uint8))


class LostTrackSource:
    """One car approaches the line for 12 frames and vanishes; the video goes on to 30 frames."""

    def frames(self, video_path: Path, model_name: str, confidence: float) -> Iterator[FrameResult]:
        w, h = 640, 360
        for i in range(30):
            dets = []
            if i < 12:
                y = 40 + 10 * i  # pixel line at y=180; ends at y=150, 30 px before it
                dets.append(Detection(i, i / FPS, 1, VehicleClass.car, 0.9, (280, y - 30, 320, y)))
            yield FrameResult(i, i / FPS, w, h, dets, np.zeros((h, w, 3), dtype=np.uint8))


def test_track_lost_before_video_end_is_flagged_missed(tmp_path):
    video = make_video(tmp_path / "v.avi", 640, 360, frames=30)
    result = process_video(video, LINE, ProcessOptions(), LostTrackSource())
    missed = [f for f in result.summary.flags if f.kind == FlagKind.MISSED]
    assert [(f.track_id, f.direction) for f in missed] == [(1, Direction.ONE)]
    assert result.summary.too_few == 1


def test_processing_fps_excludes_time_before_first_frame(tmp_path):
    n, delay = 20, 0.3
    video = make_video(tmp_path / "v.avi", 64, 48)
    result = process_video(video, LINE, ProcessOptions(), SlowStartSource(n, delay))
    assert result.processing_fps > 5 * n / delay
