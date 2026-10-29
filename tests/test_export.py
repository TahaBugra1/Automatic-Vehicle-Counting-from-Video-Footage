import csv
import re
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import pytest

from visionflow import export
from visionflow.export import AnnotatedWriter, write_csv, write_summary_csv
from visionflow.geometry import Line
from visionflow.models import CrossingEvent, Detection, Direction, VehicleClass
from visionflow.pipeline import VideoResult
from visionflow.summary import Reliability, VideoSummary
from visionflow.suspicion import Flag, FlagKind
from visionflow.tracking import FrameResult


def _result(events: list[CrossingEvent], flags: list[Flag]) -> VideoResult:
    summary = VideoSummary(
        total={d: sum(1 for e in events if e.direction == d) for d in Direction},
        total_count=len(events),
        by_class={},
        timeline=[],
        too_many=0,
        too_few=0,
        reliability=Reliability.HIGH,
        flags=flags,
    )
    return VideoResult(summary, events, fps=25.0, duration_s=10.0, processing_fps=30.0)


def _events() -> list[CrossingEvent]:
    return [
        CrossingEvent(50, 2.0, 7, VehicleClass.truck, Direction.TWO, 0.8),
        CrossingEvent(25, 1.0, 3, VehicleClass.car, Direction.ONE, 0.9),
    ]


def _flags() -> list[Flag]:
    return [
        Flag(FlagKind.MISSED, 9, 80, 3.2, False, "kayboldu", Direction.TWO),
        Flag(FlagKind.WEAK, 3, 25, 1.0, True, "zayıf"),
        Flag(FlagKind.MISSED, 5, 40, 1.6, False, "kayboldu", Direction.ONE),
    ]


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def test_csv_has_header_and_crossing_rows(tmp_path):
    path = tmp_path / "out.csv"
    write_csv(_result(_events(), _flags()), "a.mp4", path)
    assert path.read_bytes()[:3] == b"\xef\xbb\xbf"  # UTF-8 BOM, so Excel detects UTF-8
    header = path.read_text(encoding="utf-8-sig").splitlines()[0]
    assert header == "tür,video,kare,zaman_s,takip_kimliği,sınıf,yön,işaretler,güven"
    rows = _read(path)
    crossings = [r for r in rows if r["tür"] == "geçiş"]
    assert [r["takip_kimliği"] for r in crossings] == ["3", "7"]
    first = crossings[0]
    assert first["video"] == "a.mp4"
    assert first["kare"] == "25"
    assert float(first["zaman_s"]) == 1.0
    assert first["sınıf"] == "car"
    assert first["işaretler"] == FlagKind.WEAK.value
    assert float(first["güven"]) == 0.9
    assert crossings[1]["işaretler"] == ""
    assert rows.index(crossings[1]) < len(crossings)  # crossings come first


def test_csv_missed_rows_carry_the_approach_direction(tmp_path):
    path = tmp_path / "out.csv"
    write_csv(_result(_events(), _flags()), "a.mp4", path)
    missed = [r for r in _read(path) if r["tür"] == "olası_kaçırılan"]
    assert [r["takip_kimliği"] for r in missed] == ["5", "9"]
    assert [r["yön"] for r in missed] == ["Yön 1", "Yön 2"]
    for r in missed:
        assert r["kare"] == "" and r["sınıf"] == ""
        assert r["işaretler"] == FlagKind.MISSED.value
    assert float(missed[0]["zaman_s"]) == 1.6


def test_csv_missed_row_without_direction_leaves_it_blank(tmp_path):
    path = tmp_path / "out.csv"
    write_csv(_result([], [Flag(FlagKind.MISSED, 1, 5, 0.2, False, "x")]), "a.mp4", path)
    (row,) = _read(path)
    assert row["yön"] == ""


def _summary_result() -> VideoResult:
    events = _events() + [CrossingEvent(60, 2.4, 8, VehicleClass.car, Direction.ONE, 0.7)]
    summary = VideoSummary(
        total={Direction.ONE: 2, Direction.TWO: 1},
        total_count=3,
        by_class={VehicleClass.car: 2, VehicleClass.truck: 1},
        timeline=[],
        too_many=1,
        too_few=2,
        reliability=Reliability.LOW,
        flags=_flags(),
    )
    return VideoResult(summary, events, fps=25.0, duration_s=12.346, processing_fps=31.26)


def test_summary_csv_has_all_metrics(tmp_path):
    path = tmp_path / "summary.csv"
    write_summary_csv(_summary_result(), "a.mp4", "yolo11n.pt", path)
    assert path.read_bytes()[:3] == b"\xef\xbb\xbf"
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["metrik", "deger"]
    metrics = dict(rows[1:])
    assert len(metrics) == len(rows) - 1  # no duplicate metric names
    assert metrics == {
        "video": "a.mp4",
        "Yön 1": "2",
        "Yön 2": "1",
        "toplam": "3",
        "fazla": "1",
        "eksik": "2",
        "guvenilirlik": "LOW",
        "sure_s": "12.35",
        "fps_isleme": "31.3",
        "model": "yolo11n.pt",
        "car": "2",
        "bus": "0",
        "truck": "1",
        "motorcycle": "0",
    }


def test_summary_csv_neutralizes_formula_video_name(tmp_path):
    path = tmp_path / "summary.csv"
    write_summary_csv(_summary_result(), "=cmd", "yolo11n.pt", path)
    with path.open(encoding="utf-8-sig", newline="") as f:
        assert dict(list(csv.reader(f))[1:])["video"] == "'=cmd"


def test_csv_uses_yon_labels(tmp_path):
    path = tmp_path / "out.csv"
    write_csv(_result(_events(), []), "a.mp4", path)
    assert [r["yön"] for r in _read(path)] == ["Yön 1", "Yön 2"]


def test_csv_video_name_formula_prefix_is_neutralized(tmp_path):
    path = tmp_path / "out.csv"
    for name in ["=cmd", "+1", "-2", "@sum"]:
        write_csv(_result(_events(), _flags()), name, path)
        assert {r["video"] for r in _read(path)} == {"'" + name}
    write_csv(_result(_events(), []), "plain-name", path)
    assert {r["video"] for r in _read(path)} == {"plain-name"}


def _frames(n: int, w: int, h: int) -> list[FrameResult]:
    out = []
    for i in range(n):
        image = np.full((h, w, 3), 40, dtype=np.uint8)
        det = Detection(i, i / 10, 1, VehicleClass.car, 0.9, (10.0 + i, 20.0, 80.0 + i, 70.0))
        out.append(FrameResult(i, i / 10, w, h, [det], image))
    return out


def test_annotated_video_is_h264_and_readable(tmp_path):
    w, h, n = 160, 120, 12
    path = tmp_path / "annotated.mp4"
    line = Line((0.0, 0.5), (1.0, 0.5))
    events: list[CrossingEvent] = []
    with AnnotatedWriter(path, w, h, 10.0, line) as writer:
        for i, frame in enumerate(_frames(n, w, h)):
            if i == 5:
                events.append(CrossingEvent(i, i / 10, 1, VehicleClass.car, Direction.ONE, 0.9))
            writer.write(frame, events)
    assert path.exists()
    assert not list(tmp_path.glob("*.tmp*")) and len(list(tmp_path.iterdir())) == 1

    proc = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-i", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert re.search(r"Video: h264", proc.stderr)
    assert "yuv420p" in proc.stderr

    cap = cv2.VideoCapture(str(path))
    try:
        assert cap.isOpened()
        count = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            count += 1
    finally:
        cap.release()
    assert count == n


def test_writer_cleans_temp_file_on_exception(tmp_path):
    path = tmp_path / "annotated.mp4"
    line = Line((0.0, 0.5), (1.0, 0.5))
    try:
        with AnnotatedWriter(path, 160, 120, 10.0, line) as writer:
            writer.write(_frames(1, 160, 120)[0], [])
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert list(tmp_path.iterdir()) == []


def test_writer_rejects_frame_of_wrong_size(tmp_path):
    path = tmp_path / "annotated.mp4"
    line = Line((0.0, 0.5), (1.0, 0.5))
    with pytest.raises(ValueError, match="size"):
        with AnnotatedWriter(path, 160, 120, 10.0, line) as writer:
            writer.write(_frames(1, 100, 80)[0], [])
    assert list(tmp_path.iterdir()) == []


def test_replace_failure_raises_clear_error_and_cleans_temp(tmp_path, monkeypatch):
    path = tmp_path / "annotated.mp4"
    line = Line((0.0, 0.5), (1.0, 0.5))

    def locked(src, dst):
        raise PermissionError("locked")

    monkeypatch.setattr(export.os, "replace", locked)
    monkeypatch.setattr(export, "REPLACE_DELAY_S", 0.0)
    with pytest.raises(OSError, match="in use") as info:
        with AnnotatedWriter(path, 160, 120, 10.0, line) as writer:
            writer.write(_frames(1, 160, 120)[0], [])
    assert str(path) in str(info.value)
    assert list(tmp_path.iterdir()) == []
