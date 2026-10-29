import math
from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum

from .counter import DEADBAND_PX, PROJECTION_MARGIN
from .geometry import Line
from .models import CrossingEvent, Detection, Direction


class FlagKind(StrEnum):
    MISSED = "olası kaçırılan araç"
    DOUBLE_COUNT = "olası çift sayım"
    UNSTABLE = "kararsız geçiş"
    WEAK = "zayıf tespit"
    CLASS_UNCERTAIN = "sınıf belirsiz"
    CROWDED = "kalabalık geçiş"


@dataclass(frozen=True, slots=True)
class Flag:
    kind: FlagKind
    track_id: int
    frame: int
    time_s: float
    counted: bool
    detail: str
    # MISSED only: the direction the track was heading to (the side it was approaching).
    direction: Direction | None = None


@dataclass(frozen=True, slots=True)
class FlagConfig:
    short_track_frames: int = 10
    low_confidence: float = 0.35
    unstable_side_changes: int = 3
    unstable_window_frames: int = 30
    lost_near_line_px: float = 60.0
    min_approach_px: float = 20.0
    end_of_video_frames: int = 5
    double_count_px: float = 40.0
    double_count_frames: int = 15
    crowd_iou: float = 0.5
    # Same as LineCounter: points this close to the line have no side; crossings are only
    # judged inside the segment extended by this fraction of its length at both ends.
    deadband_px: float = DEADBAND_PX
    projection_margin: float = PROJECTION_MARGIN


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class TrackRecorder:
    """Keeps per-track history and flags suspicious crossings when the video ends.

    `line` must be in pixel coordinates (the same line given to LineCounter).
    """

    def __init__(self, line: Line, config: FlagConfig = FlagConfig()):
        self.line = line
        self.config = config
        self._tracks: dict[int, list[Detection]] = defaultdict(list)
        self._by_frame: dict[int, list[Detection]] = defaultdict(list)
        self._events: dict[int, CrossingEvent] = {}
        self._last_frame = -1

    def observe(self, det: Detection) -> None:
        self._tracks[det.track_id].append(det)
        self._by_frame[det.frame].append(det)

    def finish(self, events: list[CrossingEvent], last_frame: int | None = None) -> list[Flag]:
        """Flags for all tracks. `last_frame` is the video's final frame index (default: the
        last frame with a detection); tracks still visible near it are not "missed"."""
        counted = {e.track_id: e for e in events}
        self._events = counted
        observed = max(self._by_frame, default=-1)
        self._last_frame = observed if last_frame is None else max(last_frame, observed)
        rules = (
            self._missed,
            self._double_count,
            self._unstable,
            self._weak,
            self._class_uncertain,
            self._crowded,
        )
        flags: list[Flag] = []
        for rule in rules:
            for track_id, history in self._tracks.items():
                flag = rule(track_id, history, counted.get(track_id))
                if flag is not None:
                    flags.append(flag)
        return flags

    # Each rule returns at most one flag per track.

    def _side(self, distance: float) -> int:
        """-1 / +1 like LineCounter; 0 inside the deadband (no side)."""
        if abs(distance) <= self.config.deadband_px:
            return 0
        return 1 if distance > 0 else -1

    def _missed(
        self, track_id: int, history: list[Detection], event: CrossingEvent | None
    ) -> Flag | None:
        cfg = self.config
        if event is not None or len(history) < cfg.short_track_frames:
            return None
        end = history[-1]
        if self._last_frame - end.frame <= cfg.end_of_video_frames:
            return None  # still visible when the video ended: not lost
        first = self.line.signed_distance(history[0].bottom_center)
        last = self.line.signed_distance(end.bottom_center)
        if abs(last) > cfg.lost_near_line_px:
            return None
        if abs(first) - abs(last) <= cfg.min_approach_px:
            return None  # no real approach (e.g. a car parked near the line)
        start_side, end_side = self._side(first), self._side(last)
        if start_side == 0 or (end_side != 0 and end_side != start_side):
            return None  # already on the other side: it was moving away, not approaching
        projection = self.line.projection(end.bottom_center)
        if not -cfg.projection_margin <= projection <= 1 + cfg.projection_margin:
            return None  # lost next to the extended line, outside the counting segment
        approach = Direction.ONE if start_side < 0 else Direction.TWO
        detail = f"çizgiye {abs(last):.0f} px kala kayboldu, yaklaşma yönü {approach}"
        return Flag(FlagKind.MISSED, track_id, end.frame, end.time_s, False, detail, approach)

    def _double_count(self, track_id, history, event) -> Flag | None:
        # Only a real double count if the earlier track was itself counted before this one.
        if event is None:
            return None
        start = history[0]
        for other_id, other in self._tracks.items():
            other_event = self._events.get(other_id)
            if other_id == track_id or other_event is None or other_event.frame >= event.frame:
                continue
            end = other[-1]
            gap = start.frame - end.frame
            if not 0 < gap <= self.config.double_count_frames:
                continue
            dist = math.dist(start.bottom_center, end.bottom_center)
            if dist <= self.config.double_count_px:
                detail = f"{gap} kare önce {dist:.0f} px yakında kaybolan sayılmış takip {other_id}"
                return Flag(
                    FlagKind.DOUBLE_COUNT, track_id, event.frame, event.time_s, True, detail
                )
        return None

    def _unstable(self, track_id, history, event) -> Flag | None:
        changes: list[Detection] = []
        previous = 0
        for det in history:
            side = self._side(self.line.signed_distance(det.bottom_center))
            if side == 0:  # inside the counter's deadband: jitter on the line is no side change
                continue
            if previous and side != previous:
                changes.append(det)
            previous = side
        n = self.config.unstable_side_changes
        for i in range(len(changes) - n + 1):
            first, last = changes[i], changes[i + n - 1]
            if last.frame - first.frame <= self.config.unstable_window_frames:
                detail = f"{last.frame - first.frame} kare içinde {n} taraf değişimi"
                return Flag(
                    FlagKind.UNSTABLE, track_id, last.frame, last.time_s, event is not None, detail
                )
        return None

    def _weak(self, track_id, history, event) -> Flag | None:
        if event is None:
            return None
        mean_conf = sum(d.confidence for d in history) / len(history)
        too_short = len(history) < self.config.short_track_frames
        if not too_short and mean_conf >= self.config.low_confidence:
            return None
        detail = f"{len(history)} kare, ortalama güven {mean_conf:.2f}"
        return Flag(FlagKind.WEAK, track_id, event.frame, event.time_s, True, detail)

    def _class_uncertain(self, track_id, history, event) -> Flag | None:
        classes = sorted({d.cls for d in history})
        if len(classes) < 2:
            return None
        ref = event if event is not None else history[-1]
        detail = "sınıflar: " + ", ".join(classes)
        return Flag(
            FlagKind.CLASS_UNCERTAIN, track_id, ref.frame, ref.time_s, event is not None, detail
        )

    def _crowded(self, track_id, history, event) -> Flag | None:
        if event is None:
            return None
        own = next((d for d in history if d.frame == event.frame), None)
        if own is None:
            return None
        for other in self._by_frame[event.frame]:
            if other.track_id == track_id:
                continue
            iou = _iou(own.bbox, other.bbox)
            if iou > self.config.crowd_iou:
                detail = f"takip {other.track_id} ile IoU {iou:.2f}"
                return Flag(FlagKind.CROWDED, track_id, event.frame, event.time_s, True, detail)
        return None
