from visionflow.counter import LineCounter
from visionflow.geometry import Line
from visionflow.models import Detection, Direction, VehicleClass
from visionflow.suspicion import Flag, FlagConfig, FlagKind, TrackRecorder

# Horizontal line y=100 in pixels; y > 100 is the positive side.
LINE = Line((0, 100), (200, 100))


def det(
    frame: int,
    track_id: int,
    x: float,
    y: float,
    cls: VehicleClass = VehicleClass.car,
    confidence: float = 0.9,
    width: float = 20,
) -> Detection:
    """Detection whose bottom-center is (x, y)."""
    return Detection(
        frame=frame,
        time_s=frame / 25,
        track_id=track_id,
        cls=cls,
        confidence=confidence,
        bbox=(x - width / 2, y - 20, x + width / 2, y),
    )


def run(
    detections: list[Detection],
    config: FlagConfig = FlagConfig(),
    last_frame: int | None = None,
):
    """Feed detections (sorted by frame) to a counter and a recorder; return (events, flags).

    `last_frame` is the video's final frame index (default: the last observed frame).
    """
    counter = LineCounter(LINE)
    recorder = TrackRecorder(LINE, config)
    events = []
    for d in sorted(detections, key=lambda d: d.frame):
        recorder.observe(d)
        event = counter.update(d)
        if event is not None:
            events.append(event)
    return events, recorder.finish(events, last_frame=last_frame)


VIDEO_END = 100  # the video goes on well after the short test tracks end


def straight_track(track_id: int, x: float, start_frame: int, ys: list[float], **kw):
    return [det(start_frame + i, track_id, x, y, **kw) for i, y in enumerate(ys)]


def kinds(flags) -> list[FlagKind]:
    return [f.kind for f in flags]


def test_missed_flag_when_track_ends_near_line_uncounted():
    # 12 frames moving down from y=0 to y=55, ends 45 px above the line.
    ys = [i * 5 for i in range(12)]
    events, flags = run(straight_track(1, 100, 0, ys), last_frame=VIDEO_END)
    assert events == []
    assert kinds(flags) == [FlagKind.MISSED]
    assert flags[0].track_id == 1
    assert flags[0].frame == 11
    # Heading down = towards the positive side = Direction.ONE.
    assert flags[0].direction == Direction.ONE


def test_missed_flag_direction_for_track_moving_up():
    ys = [200 - i * 5 for i in range(12)]  # ends at y=145, 45 px below the line
    _, flags = run(straight_track(1, 100, 0, ys), last_frame=VIDEO_END)
    assert kinds(flags) == [FlagKind.MISSED]
    assert flags[0].direction == Direction.TWO


def test_missed_flag_not_counted_in_total():
    ys = [i * 5 for i in range(12)]
    _, flags = run(straight_track(1, 100, 0, ys), last_frame=VIDEO_END)
    assert len(flags) == 1
    assert flags[0].counted is False


def test_no_missed_flag_when_lost_beyond_the_segment_end():
    # Same approach, but next to the extended line: x=260 projects to 1.3 (> 1 + margin).
    ys = [i * 5 for i in range(12)]
    _, flags = run(straight_track(1, 260, 0, ys), last_frame=VIDEO_END)
    assert flags == []


def test_missed_flag_inside_the_projection_margin():
    ys = [i * 5 for i in range(12)]
    _, flags = run(straight_track(1, 208, 0, ys), last_frame=VIDEO_END)  # projection 1.04
    assert kinds(flags) == [FlagKind.MISSED]


def test_no_missed_flag_for_parked_car_near_the_line():
    # A car parked 30 px from the line with a few px of box jitter: no real approach.
    ys = [70 + (i % 3) for i in range(40)]
    _, flags = run(straight_track(1, 100, 0, ys), last_frame=VIDEO_END)
    assert flags == []


def test_no_missed_flag_for_too_small_approach():
    ys = [60 + i for i in range(15)]  # from 40 px to 26 px: approach 14 px < 20 px
    _, flags = run(straight_track(1, 100, 0, ys), last_frame=VIDEO_END)
    assert flags == []


def test_no_missed_flag_when_track_already_changed_side():
    # Crossed outside the segment (x=300), then walked back in on the other side and was lost.
    track = [det(i, 1, 300, i * 15) for i in range(10)]  # y 0..135, crosses at x=300
    track += [det(10 + i, 1, 300 - i * 20, 130) for i in range(9)]  # ends (140, 130): +30 px
    events, flags = run(track, last_frame=VIDEO_END)
    assert events == []
    assert [f for f in flags if f.kind == FlagKind.MISSED] == []


def test_no_missed_flag_for_track_alive_at_video_end():
    ys = [i * 5 for i in range(12)]  # last seen at frame 11
    for end in (11, 16):  # video ends at or within 5 frames after the track's last frame
        _, flags = run(straight_track(1, 100, 0, ys), last_frame=end)
        assert flags == [], end
    _, flags = run(straight_track(1, 100, 0, ys), last_frame=17)
    assert kinds(flags) == [FlagKind.MISSED]


def test_end_of_video_defaults_to_last_observed_frame():
    ys = [i * 5 for i in range(12)]
    _, flags = run(straight_track(1, 100, 0, ys))  # nothing seen after frame 11
    assert flags == []
    later = straight_track(2, 20, 30, [10] * 3)  # another car far away, seen until frame 32
    _, flags = run(straight_track(1, 100, 0, ys) + later)
    assert [(f.kind, f.track_id) for f in flags] == [(FlagKind.MISSED, 1)]


def test_flag_direction_defaults_to_none():
    flag = Flag(FlagKind.WEAK, 1, 0, 0.0, True, "x")
    assert flag.direction is None


def test_no_unstable_flag_for_stopped_vehicle_jittering_on_the_line():
    # Stopped on the line: bottom-center jitters +-1 px around y=100 (inside the 2 px deadband).
    ys = [60, 70, 80, 90] + [99, 101] * 10 + [130, 140]
    events, flags = run(straight_track(1, 100, 0, ys))
    assert len(events) == 1
    assert [f for f in flags if f.kind == FlagKind.UNSTABLE] == []


def test_double_count_flag_when_new_track_starts_where_another_ended():
    # Track 1 crosses (counted) and ends at (100, 110), frame 10; track 2 starts 3 frames later
    # at (105, 85), about 25 px away, and crosses again.
    first = straight_track(1, 100, 0, [20 + i * 9 for i in range(11)])
    second = straight_track(2, 105, 13, [85 + i * 8 for i in range(12)])
    events, flags = run(first + second)
    assert [e.track_id for e in events] == [1, 2]
    doubles = [f for f in flags if f.kind == FlagKind.DOUBLE_COUNT]
    assert len(doubles) == 1
    assert doubles[0].track_id == 2
    assert doubles[0].counted is True


def test_no_double_count_flag_when_earlier_track_was_not_counted():
    # ID switch: track 1 is lost 45 px before the line (uncounted); track 2 continues and crosses.
    first = straight_track(1, 100, 0, [i * 5 for i in range(12)])  # ends at (100, 55), frame 11
    second = straight_track(2, 105, 15, [60 + i * 8 for i in range(12)])  # starts at (105, 60)
    events, flags = run(first + second)
    assert [e.track_id for e in events] == [2]
    assert [f for f in flags if f.kind == FlagKind.DOUBLE_COUNT] == []
    assert [(f.kind, f.track_id) for f in flags] == [(FlagKind.MISSED, 1)]


def test_unstable_flag_on_flip_flopping_track():
    ys = [60, 70, 80, 90, 95, 105, 95, 105, 95, 105, 120, 140]
    events, flags = run(straight_track(1, 100, 0, ys))
    assert len(events) == 1
    unstable = [f for f in flags if f.kind == FlagKind.UNSTABLE]
    assert len(unstable) == 1
    assert unstable[0].counted is True


def test_weak_flag_for_short_counted_track():
    events, flags = run(straight_track(1, 100, 0, [80, 90, 110, 120]))
    assert len(events) == 1
    assert kinds(flags) == [FlagKind.WEAK]
    assert flags[0].counted is True
    assert flags[0].frame == events[0].frame


def test_class_uncertain_when_class_changes():
    ys = [40 + i * 10 for i in range(12)]
    track = straight_track(1, 100, 0, ys)
    track[3] = det(3, 1, 100, ys[3], cls=VehicleClass.truck)
    events, flags = run(track)
    assert len(events) == 1
    assert kinds(flags) == [FlagKind.CLASS_UNCERTAIN]


def test_crowded_flag_on_overlapping_boxes_at_crossing():
    ys = [40 + i * 10 for i in range(12)]
    a = straight_track(1, 100, 0, ys, width=40)
    b = straight_track(2, 104, 0, [y - 2 for y in ys], width=40)
    events, flags = run(a + b)
    assert len(events) == 2
    crowded = [f for f in flags if f.kind == FlagKind.CROWDED]
    assert sorted(f.track_id for f in crowded) == [1, 2]
    assert all(f.counted for f in crowded)


def test_clean_track_produces_no_flags():
    ys = [40 + i * 10 for i in range(12)]
    events, flags = run(straight_track(1, 100, 0, ys))
    assert len(events) == 1
    assert flags == []
