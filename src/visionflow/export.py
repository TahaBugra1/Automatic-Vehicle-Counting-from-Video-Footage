import csv
import math
import os
import subprocess
import time
from pathlib import Path
from types import TracebackType

import cv2
import imageio_ffmpeg

from .geometry import Line
from .models import CrossingEvent, Direction, VehicleClass
from .pipeline import VideoResult
from .suspicion import FlagKind
from .tracking import FrameResult

CSV_COLUMNS = (
    "tür",
    "video",
    "kare",
    "zaman_s",
    "takip_kimliği",
    "sınıf",
    "yön",
    "işaretler",
    "güven",
)
SUMMARY_COLUMNS = ("metrik", "deger")
CSV_ENCODING = "utf-8-sig"  # BOM: Excel opens the file as UTF-8
CROSSING = "geçiş"
MISSED = "olası_kaçırılan"
REPLACE_ATTEMPTS = 4
REPLACE_DELAY_S = 0.2
_FORMULA_PREFIXES = ("=", "+", "-", "@")


def _neutralize_formula(value: str) -> str:
    """Spreadsheet apps execute cells starting with these characters as formulas."""
    return "'" + value if value.startswith(_FORMULA_PREFIXES) else value


def write_csv(result: VideoResult, video_name: str, path: Path) -> None:
    """Crossings first (by time), then possibly missed vehicles (by last-seen time).

    UTF-8 with BOM (Excel then detects the encoding), comma-separated. A missed row's `yön` is
    the direction the vehicle was approaching.
    """
    video_name = _neutralize_formula(video_name)
    flags_by_track: dict[int, list[str]] = {}
    for flag in result.summary.flags:
        if flag.counted:
            flags_by_track.setdefault(flag.track_id, []).append(flag.kind.value)

    with path.open("w", encoding=CSV_ENCODING, newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_COLUMNS)
        for e in sorted(result.events, key=lambda e: (e.time_s, e.track_id)):
            writer.writerow(
                [
                    CROSSING,
                    video_name,
                    e.frame,
                    f"{e.time_s:.2f}",
                    e.track_id,
                    e.cls.value,
                    e.direction.value,
                    ";".join(flags_by_track.get(e.track_id, [])),
                    f"{e.confidence:.2f}",
                ]
            )
        missed = [f for f in result.summary.flags if f.kind == FlagKind.MISSED]
        for flag in sorted(missed, key=lambda f: (f.time_s, f.track_id)):
            direction = flag.direction.value if flag.direction is not None else ""
            writer.writerow(
                [MISSED, video_name, "", f"{flag.time_s:.2f}", flag.track_id, "", direction,
                 FlagKind.MISSED.value, ""]
            )  # fmt: skip


def write_summary_csv(result: VideoResult, video_name: str, model_name: str, path: Path) -> None:
    """Summary metrics of one video as `metrik,deger` rows (UTF-8 with BOM)."""
    s = result.summary
    rows: list[tuple[str, object]] = [
        ("video", _neutralize_formula(video_name)),
        *((d.value, s.total.get(d, 0)) for d in Direction),
        ("toplam", s.total_count),
        ("fazla", s.too_many),
        ("eksik", s.too_few),
        ("guvenilirlik", s.reliability.value),
        ("sure_s", f"{result.duration_s:.2f}"),
        ("fps_isleme", f"{result.processing_fps:.1f}"),
        ("model", _neutralize_formula(model_name)),
        *((cls.value, s.by_class.get(cls, 0)) for cls in VehicleClass),
    ]
    with path.open("w", encoding=CSV_ENCODING, newline="") as f:
        writer = csv.writer(f)
        writer.writerow(SUMMARY_COLUMNS)
        writer.writerows(rows)


_GREEN = (0, 200, 0)
_RED = (0, 0, 220)
_YELLOW = (0, 220, 220)
_WHITE = (255, 255, 255)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


class AnnotatedWriter:
    """Writes annotated frames and transcodes the result to browser-playable H.264 on exit."""

    def __init__(self, path: Path, width: int, height: int, fps: float, line: Line):
        self.path = Path(path)
        self.width = width
        self.height = height
        self.fps = fps
        self.line = line.to_pixels(width, height)
        self._raw = self.path.with_name(f"{self.path.stem}.raw.tmp.mp4")
        self._out = self.path.with_name(f"{self.path.stem}.h264.tmp.mp4")
        self._writer: cv2.VideoWriter | None = None
        self._counts: dict[Direction, int] = {d: 0 for d in Direction}
        self._seen: set[tuple[int, int]] = set()

    def __enter__(self) -> "AnnotatedWriter":
        writer = cv2.VideoWriter(
            str(self._raw), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (self.width, self.height)
        )
        if not writer.isOpened():
            raise OSError(f"cannot open video writer: {self._raw}")
        self._writer = writer
        return self

    def write(self, frame: FrameResult, events: list[CrossingEvent]) -> None:
        if self._writer is None:
            raise RuntimeError("AnnotatedWriter must be used as a context manager")
        if frame.image.shape[:2] != (self.height, self.width):
            raise ValueError(
                f"frame size {frame.image.shape[1]}x{frame.image.shape[0]} does not match "
                f"writer size {self.width}x{self.height}"
            )
        # on_frame may pass only this frame's events or all so far; count each crossing once.
        for e in events:
            key = (e.track_id, e.frame)
            if key not in self._seen:
                self._seen.add(key)
                self._counts[e.direction] += 1
        image = frame.image.copy()
        self._draw_line(image)
        for det in frame.detections:
            x1, y1, x2, y2 = (round(v) for v in det.bbox)
            cv2.rectangle(image, (x1, y1), (x2, y2), _GREEN, 2)
            label = f"#{det.track_id} {det.cls.value}"
            cv2.putText(image, label, (x1, max(y1 - 6, 12)), _FONT, 0.5, _GREEN, 1, cv2.LINE_AA)
        # ASCII labels: cv2 Hershey fonts cannot render "ö".
        cv2.putText(
            image, f"Yon 1: {self._counts[Direction.ONE]}", (10, 25), _FONT, 0.7, _WHITE, 2
        )
        cv2.putText(
            image, f"Yon 2: {self._counts[Direction.TWO]}", (10, 55), _FONT, 0.7, _WHITE, 2
        )
        self._writer.write(image)

    def _draw_line(self, image) -> None:
        (sx, sy), (ex, ey) = self.line.start, self.line.end
        cv2.line(image, (round(sx), round(sy)), (round(ex), round(ey)), _RED, 2)
        # Arrow perpendicular to the line, towards the positive side (Direction.ONE).
        dx, dy = ex - sx, ey - sy
        length = math.hypot(dx, dy)
        nx, ny = -dy / length, dx / length
        mx, my = (sx + ex) / 2, (sy + ey) / 2
        size = 0.08 * min(self.width, self.height)
        tail = (round(mx), round(my))
        tip = (round(mx + nx * size), round(my + ny * size))
        cv2.arrowedLine(image, tail, tip, _YELLOW, 2, tipLength=0.3)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        try:
            if exc_type is None:
                self._transcode()
        finally:
            for tmp in (self._raw, self._out):
                tmp.unlink(missing_ok=True)

    def _transcode(self) -> None:
        cmd = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(self._raw),
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(self._out),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg transcode failed: {proc.stderr.strip()}")
        for attempt in range(REPLACE_ATTEMPTS):
            try:
                os.replace(self._out, self.path)
                return
            except PermissionError as err:
                if attempt == REPLACE_ATTEMPTS - 1:
                    raise OSError(
                        f"cannot write output video, the file is in use by another program: "
                        f"{self.path}"
                    ) from err
                time.sleep(REPLACE_DELAY_S)
