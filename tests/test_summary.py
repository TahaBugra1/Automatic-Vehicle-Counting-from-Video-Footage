import pytest

from visionflow.config import ReliabilityThresholds, load_reliability
from visionflow.models import CrossingEvent, Direction, VehicleClass
from visionflow.summary import Reliability, format_total, summarize
from visionflow.suspicion import Flag, FlagKind

TH = ReliabilityThresholds(high_below=0.05, medium_below=0.15)


def ev(track_id, time_s=1.0, cls=VehicleClass.car, direction=Direction.ONE):
    return CrossingEvent(int(time_s * 30), time_s, track_id, cls, direction, 0.9)


def fl(kind, track_id, counted=True):
    return Flag(kind, track_id, 1, 1.0, counted, "x")


def test_timeline_bins_by_minute():
    events = [ev(1, 10), ev(2, 70, direction=Direction.TWO), ev(3, 75)]
    s = summarize(events, [], 120.0, TH)
    assert [(b.minute, b.one, b.two) for b in s.timeline] == [(0, 1, 0), (1, 1, 1)]
    assert s.total == {Direction.ONE: 2, Direction.TWO: 1}
    assert s.total_count == 3


def test_short_video_has_single_bin():
    s = summarize([ev(1, 10)], [], 30.0, TH)
    assert len(s.timeline) == 1 and s.timeline[0].one == 1


def test_zero_duration_does_not_divide_by_zero():
    s = summarize([ev(1, 0.0)], [], 0.0, TH)
    assert len(s.timeline) == 1
    assert summarize([], [], 0.0, TH).total_count == 0


def test_by_class_counts():
    events = [ev(1), ev(2), ev(3, cls=VehicleClass.bus)]
    s = summarize(events, [], 60.0, TH)
    assert s.by_class[VehicleClass.car] == 2
    assert s.by_class[VehicleClass.bus] == 1


def test_uncertainty_counts_missed_as_too_few():
    events = [ev(1), ev(2), ev(3)]
    flags = [
        fl(FlagKind.DOUBLE_COUNT, 2),
        fl(FlagKind.WEAK, 2),
        fl(FlagKind.MISSED, 9, counted=False),
        fl(FlagKind.MISSED, 10, counted=False),
    ]
    s = summarize(events, flags, 60.0, TH)
    assert s.too_many == 1
    assert s.too_few == 2
    assert s.total_count == 3
    assert s.flags == flags


def test_class_uncertain_does_not_affect_totals():
    s = summarize([ev(1)], [fl(FlagKind.CLASS_UNCERTAIN, 1)], 60.0, TH)
    assert s.too_many == 0 and s.too_few == 0


def test_uncounted_flags_do_not_count_as_too_many():
    s = summarize([ev(1)], [fl(FlagKind.UNSTABLE, 5, counted=False)], 60.0, TH)
    assert s.too_many == 0


@pytest.mark.parametrize(
    ("too_many", "level"),
    [(4, Reliability.HIGH), (10, Reliability.MEDIUM), (30, Reliability.LOW)],
)
def test_reliability_levels_use_thresholds(too_many, level):
    events = [ev(i) for i in range(100)]
    flags = [fl(FlagKind.UNSTABLE, i) for i in range(too_many)]
    assert summarize(events, flags, 60.0, TH).reliability == level


def test_no_events_reliability_unknown():
    assert summarize([], [], 60.0, TH).reliability == Reliability.UNKNOWN


def test_format_total_tr():
    events = [ev(i) for i in range(222)]
    flags = [fl(FlagKind.WEAK, i) for i in range(3)]
    flags += [fl(FlagKind.MISSED, 1000 + i, counted=False) for i in range(4)]
    s = summarize(events, flags, 600.0, TH)
    assert format_total(s, "tr") == "222 araç (3 fazla, 4 eksik olabilir)"


def test_format_total_en():
    events = [ev(i) for i in range(222)]
    flags = [fl(FlagKind.WEAK, i) for i in range(3)]
    flags += [fl(FlagKind.MISSED, 1000 + i, counted=False) for i in range(4)]
    s = summarize(events, flags, 600.0, TH)
    assert format_total(s, "en") == "222 vehicles (3 too many, 4 too few possible)"


@pytest.mark.parametrize(
    ("too_many", "level"),
    [(5, Reliability.MEDIUM), (15, Reliability.LOW)],
)
def test_threshold_boundary_is_strict_less_than(too_many, level):
    events = [ev(i) for i in range(100)]
    flags = [fl(FlagKind.UNSTABLE, i) for i in range(too_many)]
    assert summarize(events, flags, 60.0, TH).reliability == level


def test_duration_exactly_60_has_one_bin():
    assert len(summarize([ev(1, 10)], [], 60.0, TH).timeline) == 1


def test_event_past_duration_extends_timeline():
    s = summarize([ev(1, 130)], [], 30.0, TH)
    assert len(s.timeline) == 3 and s.timeline[2].one == 1


@pytest.mark.parametrize("duration", [float("nan"), float("inf"), -5.0])
def test_nan_duration_falls_back_to_one_bin(duration):
    s = summarize([ev(1, 10)], [], duration, TH)
    assert len(s.timeline) == 1 and s.timeline[0].one == 1


def test_load_reliability_tolerates_extra_keys(tmp_path):
    path = tmp_path / "reliability.json"
    path.write_text('{"high_below": 0.1, "medium_below": 0.3, "placeholder": true}')
    t = load_reliability(path)
    assert t.high_below == 0.1 and t.medium_below == 0.3


def test_load_reliability_default_file():
    t = load_reliability()
    assert t.high_below == 0.05 and t.medium_below == 0.15
