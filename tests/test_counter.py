import pytest

from visionflow.counter import LineCounter
from visionflow.geometry import Line
from visionflow.models import Detection, Direction, VehicleClass

LINE = Line((0, 100), (200, 100))


def det(frame: int, track_id: int, x: float, y: float) -> Detection:
    """Detection whose bottom-center is (x, y)."""
    return Detection(
        frame=frame,
        time_s=frame / 25,
        track_id=track_id,
        cls=VehicleClass.car,
        confidence=0.9,
        bbox=(x - 10, y - 20, x + 10, y),
    )


def feed(counter: LineCounter, track_id: int, points: list[tuple[float, float]]) -> list:
    events = [counter.update(det(i, track_id, x, y)) for i, (x, y) in enumerate(points)]
    return [e for e in events if e is not None]


def test_crossing_counts_once_per_track():
    counter = LineCounter(LINE)
    events = feed(counter, 1, [(100, 80), (100, 120), (100, 80), (100, 120)])
    assert len(events) == 1
    assert sum(counter.counts.values()) == 1


def test_direction_one_when_moving_to_positive_side():
    counter = LineCounter(LINE)
    events = feed(counter, 1, [(100, 80), (100, 120)])
    assert [e.direction for e in events] == [Direction.ONE]
    assert counter.counts[Direction.ONE] == 1
    assert counter.counts[Direction.TWO] == 0


def test_direction_two_when_moving_to_negative_side():
    counter = LineCounter(LINE)
    events = feed(counter, 1, [(100, 120), (100, 80)])
    assert [e.direction for e in events] == [Direction.TWO]
    assert counter.counts[Direction.TWO] == 1


def test_deadband_suppresses_jitter():
    counter = LineCounter(LINE, deadband=2.0)
    events = feed(counter, 1, [(100, 98.5), (100, 101.5), (100, 98.5), (100, 101.5)])
    assert events == []
    assert sum(counter.counts.values()) == 0


def test_crossing_outside_segment_is_ignored():
    counter = LineCounter(LINE, projection_margin=0.05)
    events = feed(counter, 1, [(400, 80), (400, 120)])
    assert events == []
    assert sum(counter.counts.values()) == 0


def test_out_of_segment_then_in_segment_crossing_counts_once():
    counter = LineCounter(LINE, projection_margin=0.05)
    events = feed(counter, 1, [(400, 80), (400, 120), (100, 120), (100, 80), (100, 120)])
    assert [e.frame for e in events] == [3]
    assert [e.direction for e in events] == [Direction.TWO]
    assert sum(counter.counts.values()) == 1


def test_two_distinct_tracks_counted_separately():
    counter = LineCounter(LINE)
    feed(counter, 2, [(50, 80), (50, 120)])
    feed(counter, 4, [(150, 120), (150, 80)])
    assert sum(counter.counts.values()) == 2


def test_zero_length_line_raises_value_error():
    with pytest.raises(ValueError):
        Line((5, 5), (5, 5))
