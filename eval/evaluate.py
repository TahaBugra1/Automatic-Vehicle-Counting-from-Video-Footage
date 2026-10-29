"""Evaluate the counting pipeline on UA-DETRAC sequences and compare YOLO11 models.

Usage:
    python -m eval.evaluate --images DIR --annotations DIR --lines eval/lines.json \
        --out report.csv [--models yolo11n.pt yolo11s.pt yolo11m.pt] [--sequences MVI_20011 ...] \
        [--split train|test] [--repeats 3] [--confidence 0.25]

`--images` holds one folder of img*.jpg frames per sequence and `--annotations` the matching
<SEQ>.xml files. UA-DETRAC ships train and test as separate folders, which keeps the official
split: count lines are chosen on train sequences only, and `--split test` runs only once the
lines file is marked frozen.

Metric definitions (one sequence, the system's crossing events vs ground-truth crossings):
- matching: one-to-one, greedy nearest-first, same direction, |frame difference| <= 15;
  classes and track ids are ignored.
- total_error: system count minus ground-truth count (signed); abs_error: |total_error|;
  relative_error: abs_error / max(gt count, 1).
- precision: matched / system events (1.0 when there are no events: nothing was wrong);
  recall: matched / gt crossings (1.0 when there are no crossings: nothing was missed);
  f1: harmonic mean, 0.0 when precision + recall is 0.
- flag_precision: among events with a counted "too many" flag (the kinds `summarize` counts
  as too many), the fraction that is unmatched, i.e. really wrong; nan when none was flagged.
- missed_flag_hit_rate: among MISSED flags, the fraction with an unmatched gt crossing within
  +-15 frames of the flag's frame; nan when there are no MISSED flags.
- median_fps: median processing_fps over `repeats` runs of `process_video`.

The per-model "ALL" row pools the raw counts of the successful sequences: total_error and
abs_error are sums, relative_error = sum abs_error / max(sum gt, 1), precision, recall, f1,
flag_precision and missed_flag_hit_rate come from the summed counts, mean_relative_error is
the unweighted mean of the per-sequence relative errors and median_fps the median of the
per-sequence medians. Failed sequences get a row with the `error` column and are left out.

Exit status: 0 when every (model, sequence) succeeded, 1 when any failed, 2 for argument,
annotation-folder or lines-file errors (nothing is evaluated then).
"""

import argparse
import contextlib
import csv
import json
import math
import statistics
import sys
import tempfile
import traceback
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from visionflow.geometry import Line
from visionflow.models import CrossingEvent
from visionflow.pipeline import ProcessOptions, process_video
from visionflow.summary import suggests_too_many
from visionflow.suspicion import Flag, FlagKind
from visionflow.tracking import TrackSource, probe_video

from .ground_truth import (
    GtCrossing,
    derive_ground_truth_crossings,
    frames_to_video,
    load_ignored_regions,
    segment_hits_box,
)

MATCH_FRAMES = 15
DEFAULT_MODELS = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"]
AGGREGATE = "ALL"


@dataclass(frozen=True, slots=True)
class SequenceMetrics:
    gt_count: int
    system_count: int
    matched: int
    total_error: int
    abs_error: int
    relative_error: float
    precision: float
    recall: float
    f1: float
    flagged: int
    flagged_wrong: int
    flag_precision: float
    missed_flags: int
    missed_hits: int
    missed_flag_hit_rate: float
    median_fps: float


COLUMNS = (
    ["model", "sequence"]
    + [f.name for f in fields(SequenceMetrics)]
    + ["mean_relative_error", "warnings", "error"]
)


def match_events(events: Sequence[CrossingEvent], gt: Sequence[GtCrossing]) -> dict[int, int]:
    """One-to-one greedy matching, nearest frames first. Returns {event index: gt index}."""
    pairs = sorted(
        (abs(e.frame - g.frame), ei, gi)
        for ei, e in enumerate(events)
        for gi, g in enumerate(gt)
        if e.direction == g.direction and abs(e.frame - g.frame) <= MATCH_FRAMES
    )
    matched: dict[int, int] = {}
    used_gt: set[int] = set()
    for _, ei, gi in pairs:
        if ei not in matched and gi not in used_gt:
            matched[ei] = gi
            used_gt.add(gi)
    return matched


def _ratio(numerator: int, denominator: int, empty: float) -> float:
    return numerator / denominator if denominator else empty


def _from_counts(
    *,
    gt_count: int,
    system_count: int,
    matched: int,
    abs_error: int,
    flagged: int,
    flagged_wrong: int,
    missed_flags: int,
    missed_hits: int,
    median_fps: float,
) -> SequenceMetrics:
    precision = _ratio(matched, system_count, 1.0)
    recall = _ratio(matched, gt_count, 1.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return SequenceMetrics(
        gt_count=gt_count,
        system_count=system_count,
        matched=matched,
        total_error=system_count - gt_count,
        abs_error=abs_error,
        relative_error=abs_error / max(gt_count, 1),
        precision=precision,
        recall=recall,
        f1=f1,
        flagged=flagged,
        flagged_wrong=flagged_wrong,
        flag_precision=_ratio(flagged_wrong, flagged, math.nan),
        missed_flags=missed_flags,
        missed_hits=missed_hits,
        missed_flag_hit_rate=_ratio(missed_hits, missed_flags, math.nan),
        median_fps=median_fps,
    )


def compute_metrics(
    events: Sequence[CrossingEvent],
    flags: Sequence[Flag],
    gt: Sequence[GtCrossing],
    fps_values: Sequence[float],
) -> SequenceMetrics:
    matched = match_events(events, gt)
    flagged_tracks = {f.track_id for f in flags if suggests_too_many(f)}
    flagged = [i for i, e in enumerate(events) if e.track_id in flagged_tracks]
    matched_gt = set(matched.values())
    unmatched_gt = [g for i, g in enumerate(gt) if i not in matched_gt]
    missed = [f for f in flags if f.kind == FlagKind.MISSED]
    hits = sum(
        1 for f in missed if any(abs(g.frame - f.frame) <= MATCH_FRAMES for g in unmatched_gt)
    )
    return _from_counts(
        gt_count=len(gt),
        system_count=len(events),
        matched=len(matched),
        abs_error=abs(len(events) - len(gt)),
        flagged=len(flagged),
        flagged_wrong=sum(1 for i in flagged if i not in matched),
        missed_flags=len(missed),
        missed_hits=hits,
        median_fps=statistics.median(fps_values),
    )


def pool_metrics(rows: Sequence[SequenceMetrics]) -> dict[str, float | int]:
    """The ALL row: metrics from the summed counts of the given (successful) sequences."""
    pooled = _from_counts(
        **{
            name: sum(getattr(r, name) for r in rows)
            for name in (
                "gt_count",
                "system_count",
                "matched",
                "abs_error",
                "flagged",
                "flagged_wrong",
                "missed_flags",
                "missed_hits",
            )
        },
        median_fps=statistics.median(r.median_fps for r in rows),
    )
    return {
        **asdict(pooled),
        "mean_relative_error": statistics.mean(r.relative_error for r in rows),
    }


def evaluate_sequence(
    video_path: Path,
    xml_path: Path,
    line: Line,
    source: TrackSource,
    options: ProcessOptions,
    repeats: int = 1,
) -> SequenceMetrics:
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    info = probe_video(video_path)
    gt = derive_ground_truth_crossings(xml_path, line, info.width, info.height)
    results = [process_video(video_path, line, options, source) for _ in range(repeats)]
    first = results[0]
    return compute_metrics(
        first.events, first.summary.flags, gt, [r.processing_fps for r in results]
    )


def load_lines(path: Path, sequences: Sequence[str], split: str) -> dict[str, Line]:
    """Normalized count line per selected sequence from `{"frozen": bool, "sequences": {}}`."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if split == "test" and data.get("frozen") is not True:
        raise ValueError(
            f"{path.name} is not frozen: choose lines on train sequences, set "
            '"frozen": true, then evaluate the test split'
        )
    stored = data.get("sequences", {})
    missing = [s for s in sequences if s not in stored]
    if missing:
        raise ValueError(f"{path.name} has no line for: {', '.join(missing)}")
    lines: dict[str, Line] = {}
    for seq in sequences:
        coords = stored[seq]
        if not (isinstance(coords, list) and len(coords) == 4):
            raise ValueError(f"{path.name}: line for {seq} must be [x1, y1, x2, y2]")
        x1, y1, x2, y2 = (float(c) for c in coords)
        lines[seq] = Line((x1, y1), (x2, y2))
    return lines


def ignored_region_warning(xml_path: Path, line: Line, width: int, height: int) -> str:
    """Non-empty when the count line crosses an ignored region: the system still counts
    vehicles there but the ground truth has none, so the sequence's errors are inflated."""
    pixel_line = line.to_pixels(width, height)
    hits = sum(1 for box in load_ignored_regions(xml_path) if segment_hits_box(pixel_line, box))
    if not hits:
        return ""
    return f"count line crosses {hits} ignored region(s); counts there have no ground truth"


def _short_error(exc: BaseException) -> str:
    text = str(exc).strip().splitlines()
    message = f"{type(exc).__name__}: {text[0]}" if text else type(exc).__name__
    return message[:200]


def run(
    images: Path,
    annotations: Path,
    lines_path: Path,
    out: Path,
    models: Sequence[str] = DEFAULT_MODELS,
    sequences: Sequence[str] | None = None,
    split: str = "train",
    repeats: int = 1,
    confidence: float = 0.25,
    source: TrackSource | None = None,
) -> int:
    """Evaluate every model on every sequence and write the CSV report row by row.

    Returns the number of failed (model, sequence) runs. Setup problems (no annotations,
    missing lines, unfrozen test split) raise ValueError before anything is evaluated.
    """
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    if sequences is None:
        sequences = sorted(p.stem for p in annotations.glob("*.xml"))
    if not sequences:
        raise ValueError(f"no <SEQ>.xml annotation files in {annotations}")
    missing_xml = [s for s in sequences if not (annotations / f"{s}.xml").is_file()]
    if missing_xml:
        raise ValueError(f"no annotation file for: {', '.join(missing_xml)}")
    lines = load_lines(lines_path, sequences, split)
    if source is None:
        from visionflow.tracking import UltralyticsTrackSource

        source = UltralyticsTrackSource()

    results: dict[str, list[SequenceMetrics]] = {m: [] for m in models}
    failures = 0
    with (
        out.open("w", encoding="utf-8", newline="") as f,
        tempfile.TemporaryDirectory(prefix="visionflow-eval-", ignore_cleanup_errors=True) as tmp,
    ):
        writer = csv.DictWriter(f, fieldnames=COLUMNS, restval="")
        writer.writeheader()
        f.flush()

        def write(row: dict) -> None:
            writer.writerow(row)
            f.flush()

        for seq in sequences:
            xml_path = annotations / f"{seq}.xml"
            video: Path | None = None
            warning, prepare_error = "", None
            try:
                video = frames_to_video(images / seq, Path(tmp) / f"{seq}.mp4")
                info = probe_video(video)
                warning = ignored_region_warning(xml_path, lines[seq], info.width, info.height)
                if warning:
                    print(f"warning: {seq}: {warning}", file=sys.stderr)
            except Exception as exc:
                prepare_error = exc
                print(f"error: {seq}: preparing the sequence failed", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
            for model in models:
                row: dict = {"model": model, "sequence": seq, "warnings": warning}
                if prepare_error is not None:
                    row["error"] = _short_error(prepare_error)
                else:
                    options = ProcessOptions(model_name=model, confidence=confidence)
                    try:
                        metrics = evaluate_sequence(
                            video, xml_path, lines[seq], source, options, repeats
                        )
                    except Exception as exc:
                        row["error"] = _short_error(exc)
                        print(f"error: {model} on {seq} failed", file=sys.stderr)
                        traceback.print_exc(file=sys.stderr)
                    else:
                        results[model].append(metrics)
                        row.update(asdict(metrics))
                if "error" in row:
                    failures += 1
                write(row)
            if video is not None:
                with contextlib.suppress(OSError):
                    video.unlink(missing_ok=True)

        for model in models:
            row = {"model": model, "sequence": AGGREGATE}
            if results[model]:
                row.update(pool_metrics(results[model]))
            else:
                row["error"] = "no successful sequences"
            write(row)
    return failures


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate vehicle counting on UA-DETRAC")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--lines", type=Path, default=Path("eval/lines.json"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--sequences", nargs="+")
    parser.add_argument("--split", choices=["train", "test"], default="train")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--confidence", type=float, default=0.25)
    args = parser.parse_args(argv)
    try:
        failures = run(
            args.images,
            args.annotations,
            args.lines,
            args.out,
            args.models,
            args.sequences,
            args.split,
            args.repeats,
            args.confidence,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
    print(f"report written to {args.out}")
    if failures:
        print(f"{failures} evaluation(s) failed; see the error column", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
