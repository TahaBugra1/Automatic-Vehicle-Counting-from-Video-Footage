import csv
import json
import math
import tempfile
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import pytest
from fakes import make_video

from eval.evaluate import (
    compute_metrics,
    evaluate_sequence,
    load_lines,
    main,
    match_events,
    run,
)
from eval.ground_truth import GtCrossing, derive_ground_truth_crossings, frames_to_video
from visionflow import tracking
from visionflow.geometry import Line
from visionflow.models import CrossingEvent, Detection, Direction, VehicleClass
from visionflow.pipeline import ProcessOptions
from visionflow.suspicion import Flag, FlagKind
from visionflow.tracking import FrameResult, probe_video

W, H = 64, 48
LINE = Line((0.0, 0.5), (1.0, 0.5))  # pixel y = 24; moving down is Direction.ONE
DOWN = [10, 16, 22, 28, 34]  # box bottoms; crosses at the 4th sample (y=22 is in the deadband)
UP = list(reversed(DOWN))  # crosses at the 4th sample too (y=16)


def track(first_frame: int, bottoms: list[int], x: float = 20.0) -> list[tuple[int, tuple]]:
    """0-based frames with pixel bboxes (x1, y1, x2, y2) of a 10x10 box."""
    return [(first_frame + i, (x, y - 10.0, x + 10.0, float(y))) for i, y in enumerate(bottoms)]


def write_xml(path: Path, tracks: dict[int, list], ignored: list[tuple] = ()) -> Path:
    """UA-DETRAC-style annotations; frame `num` is written 1-based."""
    by_frame: dict[int, list[str]] = {}
    for tid, samples in tracks.items():
        for frame, (x1, y1, x2, y2) in samples:
            by_frame.setdefault(frame, []).append(
                f'<target id="{tid}"><box left="{x1}" top="{y1}" width="{x2 - x1}" '
                f'height="{y2 - y1}"/><attribute orientation="0" speed="1" trunc_ratio="0" '
                f'vehicle_type="car"/></target>'
            )
    regions = "".join(
        f'<box left="{x}" top="{y}" width="{w}" height="{h}"/>' for x, y, w, h in ignored
    )
    frames = "".join(
        f'<frame density="{len(t)}" num="{f + 1}"><target_list>{"".join(t)}</target_list></frame>'
        for f, t in sorted(by_frame.items())
    )
    path.write_text(
        f'<?xml version="1.0" encoding="utf-8"?><sequence name="{path.stem}">'
        f'<sequence_attribute camera_state="unstable" sence_weather="sunny"/>'
        f"<ignored_region>{regions}</ignored_region>{frames}</sequence>",
        encoding="utf-8",
    )
    return path


class ScriptedSource:
    """Yields the given pixel tracks frame by frame (frame indices are 0-based)."""

    def __init__(self, tracks: dict[int, list], n_frames: int):
        self.tracks = tracks
        self.n_frames = n_frames

    def frames(self, video_path: Path, model_name: str, confidence: float) -> Iterator[FrameResult]:
        for i in range(self.n_frames):
            dets = [
                Detection(i, i / 25.0, tid, VehicleClass.car, 0.9, bbox)
                for tid, samples in self.tracks.items()
                for frame, bbox in samples
                if frame == i
            ]
            yield FrameResult(i, i / 25.0, W, H, dets, np.zeros((H, W, 3), dtype=np.uint8))


def event(frame: int, track_id: int, direction: Direction = Direction.ONE) -> CrossingEvent:
    return CrossingEvent(frame, frame / 25.0, track_id, VehicleClass.car, direction, 0.9)


def flag(kind: FlagKind, track_id: int, frame: int, counted: bool) -> Flag:
    return Flag(kind, track_id, frame, frame / 25.0, counted, "")


def write_frames(folder: Path, n: int) -> Path:
    folder.mkdir(parents=True)
    for i in range(n):
        image = np.full((H, W, 3), 40 * i % 255, dtype=np.uint8)
        # imencode + tofile: cv2.imwrite cannot write non-ASCII paths on Windows.
        cv2.imencode(".jpg", image)[1].tofile(folder / f"img{i + 1:05d}.jpg")
    return folder


# --- ground truth ---------------------------------------------------------------------------


def test_ground_truth_counts_scripted_crossings(tmp_path):
    xml = write_xml(
        tmp_path / "MVI_1.xml",
        {1: track(0, DOWN), 2: track(10, DOWN, x=40.0), 3: track(0, [5, 8, 11, 14])},
    )
    crossings = derive_ground_truth_crossings(xml, LINE, W, H)
    assert crossings == [
        GtCrossing(frame=3, track_id=1, direction=Direction.ONE),
        GtCrossing(frame=13, track_id=2, direction=Direction.ONE),
    ]


def test_ground_truth_respects_direction(tmp_path):
    xml = write_xml(tmp_path / "MVI_1.xml", {1: track(0, DOWN), 2: track(0, UP, x=40.0)})
    by_id = {c.track_id: c.direction for c in derive_ground_truth_crossings(xml, LINE, W, H)}
    assert by_id == {1: Direction.ONE, 2: Direction.TWO}
    flipped = derive_ground_truth_crossings(xml, LINE.flipped(), W, H)
    assert {c.track_id: c.direction for c in flipped} == {1: Direction.TWO, 2: Direction.ONE}


def test_ground_truth_frame_numbers_are_zero_based(tmp_path):
    # Written as num="6".."10"; the crossing sample is num="9", i.e. 0-based frame 8.
    xml = write_xml(tmp_path / "MVI_1.xml", {7: track(5, DOWN)})
    assert derive_ground_truth_crossings(xml, LINE, W, H) == [GtCrossing(8, 7, Direction.ONE)]


def test_ignored_region_targets_are_skipped(tmp_path):
    ignored = [(30.0, 0.0, 34.0, 48.0)]  # covers x in [30, 64]: track 2 lives there
    xml = write_xml(
        tmp_path / "MVI_1.xml", {1: track(0, DOWN), 2: track(0, DOWN, x=40.0)}, ignored
    )
    assert [c.track_id for c in derive_ground_truth_crossings(xml, LINE, W, H)] == [1]


def test_malformed_xml_raises_clear_error(tmp_path):
    bad = tmp_path / "MVI_BAD.xml"
    bad.write_text('<sequence name="MVI_BAD"><frame num="1"><target_list>', encoding="utf-8")
    with pytest.raises(ValueError, match="MVI_BAD.xml"):
        derive_ground_truth_crossings(bad, LINE, W, H)
    bad.write_text('<sequence><frame num="one"></frame></sequence>', encoding="utf-8")
    with pytest.raises(ValueError, match="MVI_BAD.xml"):
        derive_ground_truth_crossings(bad, LINE, W, H)


def test_ground_truth_tolerates_missing_optional_elements(tmp_path):
    xml = tmp_path / "MVI_1.xml"
    targets = [
        f'<frame num="{i + 1}"><target_list><target id="1">'
        f'<box left="20" top="{y - 10}" width="10" height="10"/></target>'
        f'<target id="2"/></target_list></frame>'
        for i, y in enumerate(DOWN)
    ]
    xml.write_text(f"<sequence>{''.join(targets)}<frame num='9'/></sequence>", encoding="utf-8")
    assert derive_ground_truth_crossings(xml, LINE, W, H) == [GtCrossing(3, 1, Direction.ONE)]


# --- frames to video ------------------------------------------------------------------------


def test_frames_to_video_makes_readable_mp4(tmp_path):
    frames = write_frames(tmp_path / "MVI_1", 5)
    out = frames_to_video(frames, tmp_path / "MVI_1.mp4")
    info = probe_video(out)
    assert (info.width, info.height, info.frame_count) == (W, H, 5)
    assert info.fps == pytest.approx(25.0)
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="empty"):
        frames_to_video(tmp_path / "empty", tmp_path / "empty.mp4")


# --- metrics --------------------------------------------------------------------------------


def test_metrics_perfect_match(tmp_path):
    tracks = {1: track(0, DOWN), 2: track(8, UP, x=40.0)}
    xml = write_xml(tmp_path / "MVI_1.xml", tracks)
    video = make_video(tmp_path / "MVI_1.avi")  # 64x48, the same size as ScriptedSource
    m = evaluate_sequence(video, xml, LINE, ScriptedSource(tracks, 14), ProcessOptions(), 2)
    assert (m.gt_count, m.system_count, m.total_error, m.relative_error) == (2, 2, 0, 0.0)
    assert (m.precision, m.recall, m.f1) == (1.0, 1.0, 1.0)
    # Both tracks are short, so both counts carry a WEAK flag; both are real: precision 0.
    assert m.flag_precision == 0.0
    assert math.isnan(m.missed_flag_hit_rate)
    assert m.median_fps > 0


def test_metrics_with_one_miss_and_one_extra(tmp_path):
    gt_tracks = {1: track(0, DOWN), 2: track(40, DOWN)}
    # The system sees track 1 (with other ids, 10 frames late), misses track 2 and
    # counts an extra vehicle moving the other way.
    sys_tracks = {11: track(10, DOWN), 12: track(40, UP, x=40.0)}
    xml = write_xml(tmp_path / "MVI_1.xml", gt_tracks)
    video = make_video(tmp_path / "MVI_1.avi")
    m = evaluate_sequence(video, xml, LINE, ScriptedSource(sys_tracks, 50), ProcessOptions())
    assert (m.gt_count, m.system_count, m.total_error, m.relative_error) == (2, 2, 0, 0.0)
    assert (m.precision, m.recall, m.f1) == (0.5, 0.5, 0.5)


def test_matching_is_one_to_one_nearest_first():
    gt = [GtCrossing(100, 1, Direction.ONE)]
    events = [event(90, 5), event(104, 6), event(100, 7, Direction.TWO)]
    assert match_events(events, gt) == {1: 0}
    two_gt = [GtCrossing(100, 1, Direction.ONE), GtCrossing(112, 2, Direction.ONE)]
    # Event 1 (frame 101) takes gt 0 first; event 0 (frame 106) is equally near both
    # crossings but gt 0 is taken, so it gets gt 1. No crossing is used twice.
    assert match_events([event(106, 5), event(101, 6)], two_gt) == {1: 0, 0: 1}
    assert match_events([event(116, 5)], gt) == {}


def test_metrics_signed_error_and_empty_cases():
    m = compute_metrics([event(10, 1), event(50, 2)], [], [GtCrossing(10, 1, Direction.ONE)], [5])
    assert (m.total_error, m.relative_error, m.precision, m.recall) == (1, 1.0, 0.5, 1.0)
    empty = compute_metrics([], [], [], [3.0, 1.0, 2.0])
    assert (empty.precision, empty.recall, empty.f1, empty.median_fps) == (1.0, 1.0, 1.0, 2.0)
    none_found = compute_metrics([], [], [GtCrossing(10, 1, Direction.ONE)], [1.0])
    assert (none_found.total_error, none_found.precision, none_found.recall) == (-1, 1.0, 0.0)
    assert none_found.f1 == 0.0


def test_flag_precision_computation():
    gt = [GtCrossing(10, 1, Direction.ONE), GtCrossing(205, 9, Direction.TWO)]
    events = [event(10, 1), event(50, 2), event(90, 3)]
    flags = [
        flag(FlagKind.WEAK, 1, 10, True),  # flagged but matched: a false alarm
        flag(FlagKind.DOUBLE_COUNT, 2, 50, True),  # flagged and unmatched: correct
        flag(FlagKind.CROWDED, 2, 50, True),  # same event again: counted once
        flag(FlagKind.CLASS_UNCERTAIN, 3, 90, True),  # not a "too many" kind
        flag(FlagKind.MISSED, 20, 200, False),  # unmatched gt crossing at 205: a hit
        flag(FlagKind.MISSED, 21, 400, False),  # nothing near frame 400
    ]
    m = compute_metrics(events, flags, gt, [10.0])
    assert m.flag_precision == 0.5
    assert m.missed_flag_hit_rate == 0.5
    # A MISSED flag next to a gt crossing that the system did count is not a hit.
    hit = compute_metrics([event(10, 1)], [flag(FlagKind.MISSED, 5, 12, False)], gt[:1], [1.0])
    assert hit.missed_flag_hit_rate == 0.0
    assert math.isnan(hit.flag_precision)


# --- lines file and CLI ---------------------------------------------------------------------


def write_lines(path: Path, sequences: dict, frozen: bool = False) -> Path:
    path.write_text(json.dumps({"frozen": frozen, "sequences": sequences}), encoding="utf-8")
    return path


def test_lines_file_must_cover_all_sequences(tmp_path):
    lines = write_lines(tmp_path / "lines.json", {"MVI_1": [0, 0.5, 1, 0.5]})
    assert load_lines(lines, ["MVI_1"], "train") == {"MVI_1": LINE}
    with pytest.raises(ValueError, match="MVI_2.*MVI_3|MVI_3.*MVI_2"):
        load_lines(lines, ["MVI_1", "MVI_2", "MVI_3"], "train")


def test_test_split_requires_frozen_lines(tmp_path):
    lines = write_lines(tmp_path / "lines.json", {"MVI_1": [0, 0.5, 1, 0.5]})
    with pytest.raises(ValueError, match="frozen"):
        load_lines(lines, ["MVI_1"], "test")
    write_lines(lines, {"MVI_1": [0, 0.5, 1, 0.5]}, frozen=True)
    assert load_lines(lines, ["MVI_1"], "test") == {"MVI_1": LINE}


def test_report_has_row_per_model(tmp_path, monkeypatch):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    images, annotations = tmp_path / "images", tmp_path / "annotations"
    annotations.mkdir()
    tracks = {1: track(0, DOWN)}
    for seq in ("MVI_1", "MVI_2"):
        write_frames(images / seq, 6)
        write_xml(annotations / f"{seq}.xml", tracks)
    lines = write_lines(
        tmp_path / "lines.json", {"MVI_1": [0, 0.5, 1, 0.5], "MVI_2": [0, 0.5, 1, 0.5]}
    )
    out = tmp_path / "report.csv"
    run(
        images,
        annotations,
        lines,
        out,
        models=["yolo11n.pt", "yolo11s.pt"],
        source=ScriptedSource(tracks, 6),
    )
    with out.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    keys = [(r["model"], r["sequence"]) for r in rows]
    # Rows are written as produced (sequence by sequence), then one ALL row per model.
    assert keys == [
        ("yolo11n.pt", "MVI_1"),
        ("yolo11s.pt", "MVI_1"),
        ("yolo11n.pt", "MVI_2"),
        ("yolo11s.pt", "MVI_2"),
        ("yolo11n.pt", "ALL"),
        ("yolo11s.pt", "ALL"),
    ]
    assert all(float(r["f1"]) == 1.0 and float(r["relative_error"]) == 0.0 for r in rows)
    assert all(r["error"] == "" and r["warnings"] == "" for r in rows)
    assert list(scratch.iterdir()) == []  # temporary videos are cleaned up


class PerSequenceSource:
    """Delegates to one ScriptedSource per sequence; raises for sequences in `fail`."""

    def __init__(self, by_sequence: dict[str, ScriptedSource], fail: set[str] = frozenset()):
        self.by_sequence = by_sequence
        self.fail = fail
        self.on_call = None

    def frames(self, video_path: Path, model_name: str, confidence: float) -> Iterator[FrameResult]:
        if self.on_call is not None:
            self.on_call(video_path.stem)
        if video_path.stem in self.fail:
            raise RuntimeError(f"tracker crashed on {video_path.stem}")
        return self.by_sequence[video_path.stem].frames(video_path, model_name, confidence)


def make_dataset(tmp_path: Path, gt: dict[str, dict], ignored: dict[str, list] | None = None):
    images, annotations = tmp_path / "images", tmp_path / "annotations"
    annotations.mkdir()
    for seq, tracks in gt.items():
        write_frames(images / seq, 6)
        write_xml(annotations / f"{seq}.xml", tracks, (ignored or {}).get(seq, []))
    lines = write_lines(tmp_path / "lines.json", {s: [0, 0.5, 1, 0.5] for s in gt})
    return images, annotations, lines


def read_rows(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return {(r["model"], r["sequence"]): r for r in csv.DictReader(f)}


def test_all_row_uses_pooled_metrics_and_abs_error(tmp_path):
    one, two = track(0, DOWN), track(0, DOWN, x=40.0)
    gt = {"MVI_1": {1: one}, "MVI_2": {1: one, 2: two}}
    system = {"MVI_1": {1: one, 2: two}, "MVI_2": {1: one}}  # +1 and -1
    images, annotations, lines = make_dataset(tmp_path, gt)
    source = PerSequenceSource({s: ScriptedSource(t, 6) for s, t in system.items()})
    out = tmp_path / "report.csv"
    assert run(images, annotations, lines, out, models=["m.pt"], source=source) == 0
    rows = read_rows(out)
    assert [rows["m.pt", s]["total_error"] for s in ("MVI_1", "MVI_2")] == ["1", "-1"]
    assert [rows["m.pt", s]["abs_error"] for s in ("MVI_1", "MVI_2")] == ["1", "1"]
    total = rows["m.pt", "ALL"]
    assert (total["total_error"], total["abs_error"]) == ("0", "2")
    assert (total["gt_count"], total["system_count"], total["matched"]) == ("3", "3", "2")
    for key in ("relative_error", "precision", "recall", "f1"):
        assert float(total[key]) == pytest.approx(2 / 3)
    assert float(total["mean_relative_error"]) == pytest.approx((1 / 1 + 1 / 2) / 2)
    # Every count is a short track (WEAK); only MVI_1's extra one is really wrong.
    assert (total["flagged"], total["flagged_wrong"]) == ("3", "1")
    assert float(total["flag_precision"]) == pytest.approx(1 / 3)
    assert math.isnan(float(total["missed_flag_hit_rate"]))  # no MISSED flags anywhere


def test_failed_sequence_is_recorded_and_others_continue(tmp_path, monkeypatch, capsys):
    tracks = {1: track(0, DOWN)}
    images, annotations, lines = make_dataset(tmp_path, {s: tracks for s in ("A", "B", "C")})
    source = PerSequenceSource({s: ScriptedSource(tracks, 6) for s in "AC"}, fail={"B"})
    monkeypatch.setattr(tracking, "UltralyticsTrackSource", lambda: source)
    out = tmp_path / "report.csv"
    argv = ["--images", str(images), "--annotations", str(annotations), "--lines", str(lines),
            "--out", str(out), "--models", "m.pt"]
    with pytest.raises(SystemExit) as exit_info:
        main(argv)
    assert exit_info.value.code == 1
    rows = read_rows(out)
    assert list(rows) == [("m.pt", "A"), ("m.pt", "B"), ("m.pt", "C"), ("m.pt", "ALL")]
    assert rows["m.pt", "B"]["error"] == "RuntimeError: tracker crashed on B"
    assert rows["m.pt", "B"]["f1"] == ""
    assert rows["m.pt", "A"]["error"] == rows["m.pt", "C"]["error"] == ""
    assert rows["m.pt", "ALL"]["gt_count"] == "2"  # B is excluded from the aggregate
    assert "Traceback" in capsys.readouterr().err


def test_rows_are_written_incrementally(tmp_path):
    tracks = {1: track(0, DOWN)}
    images, annotations, lines = make_dataset(tmp_path, {"MVI_1": tracks, "MVI_2": tracks})
    source = PerSequenceSource({"MVI_1": ScriptedSource(tracks, 6)}, fail={"MVI_2"})
    out = tmp_path / "report.csv"
    seen: dict[str, str] = {}
    source.on_call = lambda seq: seen.setdefault(seq, out.read_text(encoding="utf-8"))
    assert run(images, annotations, lines, out, models=["m.pt"], source=source) == 1
    assert "m.pt,MVI_1," in seen["MVI_2"]  # MVI_1's row was on disk before MVI_2 failed


def test_line_intersecting_ignored_region_adds_warning(tmp_path, capsys):
    tracks = {1: track(0, DOWN)}
    # A parked-car region on the right crosses the line y=24; MVI_2's region does not.
    ignored = {"MVI_1": [(50.0, 10.0, 14.0, 30.0)], "MVI_2": [(50.0, 0.0, 14.0, 10.0)]}
    images, annotations, lines = make_dataset(tmp_path, {"MVI_1": tracks, "MVI_2": tracks}, ignored)
    out = tmp_path / "report.csv"
    run(images, annotations, lines, out, models=["m.pt"], source=ScriptedSource(tracks, 6))
    rows = read_rows(out)
    assert "ignored region" in rows["m.pt", "MVI_1"]["warnings"]
    assert rows["m.pt", "MVI_2"]["warnings"] == ""
    assert float(rows["m.pt", "MVI_1"]["f1"]) == 1.0  # other columns are unaffected
    assert "MVI_1" in capsys.readouterr().err
