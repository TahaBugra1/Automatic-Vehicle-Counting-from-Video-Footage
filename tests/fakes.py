"""Shared test doubles: tiny synthesized videos and a fake TrackSource."""

import threading
import time
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from visionflow.geometry import Line
from visionflow.jobs import JobRunner, JobStatus, VideoJob
from visionflow.models import Detection, VehicleClass
from visionflow.pipeline import ProcessOptions
from visionflow.tracking import FrameResult

W, H, FPS, N_FRAMES = 64, 48, 10.0, 6
LINE = Line((0.0, 0.5), (1.0, 0.5))
OPTIONS = ProcessOptions()
TIMEOUT_S = 15.0
# TestClient sends `Host: testserver`; the app's default allow-list deliberately excludes it.
TEST_HOSTS = {"testserver", "127.0.0.1", "localhost", "[::1]"}


def make_video(path: Path) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
    for _ in range(N_FRAMES):
        writer.write(np.zeros((H, W, 3), dtype=np.uint8))
    writer.release()
    return path


def make_bad_video(path: Path) -> Path:
    path.write_bytes(b"this is not a video" * 100)
    return path


class FakeSource:
    """One car moving down across the horizontal middle line; records call order."""

    def __init__(self, gate: threading.Event | None = None):
        self.gate = gate
        self.calls: list[str] = []
        self.active = 0
        self.max_active = 0
        self.started = threading.Event()
        self._lock = threading.Lock()

    def frames(self, video_path: Path, model_name: str, confidence: float) -> Iterator[FrameResult]:
        with self._lock:
            self.calls.append(video_path.name)
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        self.started.set()
        try:
            if self.gate is not None:
                assert self.gate.wait(TIMEOUT_S)
            for i in range(N_FRAMES):
                y = (0.2 + 0.12 * i) * H
                det = Detection(i, i / FPS, 1, VehicleClass.car, 0.9, (20.0, y - 10, 40.0, y))
                image = np.zeros((H, W, 3), dtype=np.uint8)
                yield FrameResult(i, i / FPS, W, H, [det], image)
        finally:
            with self._lock:
                self.active -= 1


def wait_finished(runner: JobRunner, job_ids: list[str]) -> list[VideoJob]:
    deadline = time.monotonic() + TIMEOUT_S
    while time.monotonic() < deadline:
        jobs = [runner.get(j) for j in job_ids]
        if all(j.status in (JobStatus.DONE, JobStatus.ERROR) for j in jobs):
            return jobs
        time.sleep(0.02)
    raise AssertionError(f"jobs did not finish: {[runner.get(j) for j in job_ids]}")
