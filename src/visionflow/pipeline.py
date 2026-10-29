import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .config import load_reliability
from .counter import LineCounter
from .geometry import Line
from .models import CrossingEvent
from .summary import VideoSummary, summarize
from .suspicion import TrackRecorder
from .tracking import FrameResult, TrackSource, probe_video


@dataclass(frozen=True, slots=True)
class ProcessOptions:
    model_name: str = "yolo11s.pt"
    confidence: float = 0.25


@dataclass(frozen=True, slots=True)
class VideoResult:
    summary: VideoSummary
    events: list[CrossingEvent]
    fps: float
    duration_s: float
    processing_fps: float


def process_video(
    video_path: Path,
    line: Line,
    options: ProcessOptions,
    source: TrackSource,
    on_frame: Callable[[FrameResult, list[CrossingEvent]], None] | None = None,
    on_progress: Callable[[float], None] | None = None,
) -> VideoResult:
    """Count line crossings in one video. `line` is normalized (0-1) and scaled per frame size."""
    info = probe_video(video_path)
    thresholds = load_reliability()
    counter: LineCounter | None = None
    recorder: TrackRecorder | None = None
    events: list[CrossingEvent] = []
    processed = 0
    started = time.perf_counter()

    first_frame_at: float | None = None
    last_frame = -1

    for frame in source.frames(video_path, options.model_name, options.confidence):
        last_frame = max(last_frame, frame.frame)
        if first_frame_at is None:
            # Model loading (and any weight download) happens before the first frame.
            first_frame_at = time.perf_counter()
        if counter is None or recorder is None:
            pixel_line = line.to_pixels(frame.width, frame.height)
            counter = LineCounter(pixel_line)
            recorder = TrackRecorder(pixel_line)
        new_events: list[CrossingEvent] = []
        for det in frame.detections:
            recorder.observe(det)
            event = counter.update(det)
            if event is not None:
                new_events.append(event)
        events.extend(new_events)
        processed += 1
        if on_frame is not None:
            on_frame(frame, new_events)
        if on_progress is not None and info.frame_count > 0:
            on_progress(min(processed / info.frame_count, 1.0))

    finished = time.perf_counter()
    processing_fps = _processing_fps(processed, started, first_frame_at, finished)
    flags = recorder.finish(events, last_frame=last_frame) if recorder is not None else []
    summary = summarize(events, flags, info.duration_s, thresholds)
    if on_progress is not None:
        on_progress(1.0)
    return VideoResult(
        summary=summary,
        events=events,
        fps=info.fps,
        duration_s=info.duration_s,
        processing_fps=processing_fps,
    )


def _processing_fps(
    processed: int, started: float, first_frame_at: float | None, finished: float
) -> float:
    """Frames per second from the first frame's arrival (excludes model start-up).

    Falls back to the whole run for a single frame or a zero-length interval.
    """
    if first_frame_at is not None and processed > 1 and finished > first_frame_at:
        return (processed - 1) / (finished - first_frame_at)
    elapsed = finished - started
    return processed / elapsed if elapsed > 0 else 0.0
