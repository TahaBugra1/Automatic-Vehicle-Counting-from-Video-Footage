import math
from dataclasses import dataclass
from enum import StrEnum

from .config import ReliabilityThresholds
from .models import CrossingEvent, Direction, VehicleClass
from .suspicion import Flag, FlagKind


class Reliability(StrEnum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True, slots=True)
class TimelineBin:
    minute: int
    one: int
    two: int


@dataclass(frozen=True, slots=True)
class VideoSummary:
    total: dict[Direction, int]
    total_count: int
    by_class: dict[VehicleClass, int]
    timeline: list[TimelineBin]
    too_many: int
    too_few: int
    reliability: Reliability
    flags: list[Flag]


_NOT_TOO_MANY = frozenset({FlagKind.MISSED, FlagKind.CLASS_UNCERTAIN})


def suggests_too_many(flag: Flag) -> bool:
    """True when the flag hints that a counted crossing may be wrong ("too many")."""
    return flag.counted and flag.kind not in _NOT_TOO_MANY


def _reliability(
    ratio: float, total_count: int, thresholds: ReliabilityThresholds
) -> Reliability:
    if total_count == 0:
        return Reliability.UNKNOWN
    if ratio < thresholds.high_below:
        return Reliability.HIGH
    if ratio < thresholds.medium_below:
        return Reliability.MEDIUM
    return Reliability.LOW


def summarize(
    events: list[CrossingEvent],
    flags: list[Flag],
    duration_s: float,
    thresholds: ReliabilityThresholds,
) -> VideoSummary:
    total = {d: sum(1 for e in events if e.direction == d) for d in Direction}
    by_class: dict[VehicleClass, int] = {}
    for e in events:
        by_class[e.cls] = by_class.get(e.cls, 0) + 1

    valid_duration = math.isfinite(duration_s) and duration_s > 0
    n_bins = max(1, math.ceil(duration_s / 60)) if valid_duration else 1
    n_bins = max([n_bins] + [int(e.time_s // 60) + 1 for e in events])
    ones = [0] * n_bins
    twos = [0] * n_bins
    for e in events:
        target = ones if e.direction == Direction.ONE else twos
        target[max(int(e.time_s // 60), 0)] += 1
    timeline = [TimelineBin(m, ones[m], twos[m]) for m in range(n_bins)]

    too_many = len({f.track_id for f in flags if suggests_too_many(f)})
    too_few = sum(1 for f in flags if f.kind == FlagKind.MISSED)
    total_count = len(events)
    ratio = (too_many + too_few) / max(total_count + too_few, 1)
    return VideoSummary(
        total=total,
        total_count=total_count,
        by_class=by_class,
        timeline=timeline,
        too_many=too_many,
        too_few=too_few,
        reliability=_reliability(ratio, total_count, thresholds),
        flags=flags,
    )


def format_total(summary: VideoSummary, lang: str) -> str:
    if lang == "tr":
        return (
            f"{summary.total_count} araç "
            f"({summary.too_many} fazla, {summary.too_few} eksik olabilir)"
        )
    return (
        f"{summary.total_count} vehicles "
        f"({summary.too_many} too many, {summary.too_few} too few possible)"
    )
