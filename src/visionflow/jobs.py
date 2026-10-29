"""Sequential job queue: one worker thread processes videos in submit (FIFO) order.

Per job, everything lives in `output_root/<job_id>/`:
- `job.json`      small status record (status, error, video_name, line, created), written at
                  every status change so a new runner can list old jobs;
- `result.json`   the fixed result schema read by the API and UI;
- `annotated.mp4`, `crossings.csv` and `summary.csv`.
The source video is deleted after processing, on success and on error (and, best effort, when a
job interrupted by a crash or shutdown is found on start-up), together with its upload folder
once that is empty.
"""

import dataclasses
import json
import os
import queue
import re
import shutil
import threading
import time
import unicodedata
import uuid
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from .export import AnnotatedWriter, write_csv, write_summary_csv
from .geometry import Line
from .models import CrossingEvent, Direction
from .paths import remove_empty_upload_folder
from .pipeline import ProcessOptions, VideoResult, process_video
from .summary import format_total
from .tracking import FrameResult, TrackSource, probe_video

MAX_STEM = 80
MAX_ERROR = 200
PROGRESS_INTERVAL_S = 0.5
PROGRESS_STEP = 0.01
SHUTDOWN_TIMEOUT_S = 10.0
DELETE_ATTEMPTS = 4
DELETE_DELAY_S = 0.2
JOB_FILE = "job.json"
RESULT_FILE = "result.json"
VIDEO_FILE = "annotated.mp4"
CSV_FILE = "crossings.csv"
SUMMARY_FILE = "summary.csv"
INTERRUPTED = "interrupted"
# Phases of a RUNNING job, shown next to the progress bar.
PHASE_LOADING = "loading"  # model start-up (and a first-run weight download)
PHASE_PROCESSING = "processing"  # frames are being tracked and counted
PHASE_ENCODING = "encoding"  # the annotated video is being transcoded

_JOB_ID = re.compile(r"[0-9a-f]{32}")
_UNSAFE = re.compile(r"[^A-Za-z0-9_-]+")
# Letters that NFKD does not reduce to ASCII; the rest (ç, ğ, ö, ş, ü, İ) decompose.
_TRANSLIT = str.maketrans({"ı": "i", "İ": "I"})


def safe_stem(name: str) -> str:
    """File-system-safe stem: only [A-Za-z0-9_-], at most 80 chars, `video` if nothing is left.

    Directory parts (`/`, `\\`) and the extension are dropped; Turkish letters are
    transliterated to ASCII (ç→c, ğ→g, ı→i, İ→I, ö→o, ş→s, ü→u), other non-ASCII letters
    are dropped, and runs of other characters become a single `_`.
    """
    base = re.split(r"[\\/]", name)[-1].strip()
    stem, dot, _ext = base.rpartition(".")
    if not dot or not stem.strip("."):
        stem = base
    ascii_stem = (
        unicodedata.normalize("NFKD", stem.translate(_TRANSLIT))
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    cleaned = _UNSAFE.sub("_", ascii_stem).strip("_-")
    return cleaned[:MAX_STEM].strip("_-") or "video"


class JobStatus(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    DONE = "DONE"
    ERROR = "ERROR"


@dataclass(slots=True)
class VideoJob:
    job_id: str
    video_name: str
    line: Line
    status: JobStatus
    progress: float
    error: str | None
    phase: str | None = None  # one of the PHASE_* values while RUNNING, else None


@dataclass(slots=True)
class _Entry:
    job: VideoJob
    created: float
    video_path: Path | None = None
    options: ProcessOptions | None = None
    last_progress_at: float = 0.0


class _Interrupted(Exception):
    pass


def _line_list(line: Line) -> list[float]:
    return [line.start[0], line.start[1], line.end[0], line.end[1]]


def _write_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _remove_quietly(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass  # e.g. still locked on Windows; never let cleanup kill the worker


def _remove_source(path: Path) -> None:
    """Delete a source video and its `uploads/<video_id>/` folder if that is now empty."""
    _remove_quietly(path)
    remove_empty_upload_folder(path)


def _delete_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _with_retries(action, *args) -> None:
    """Run `action(*args)`; on OSError retry (Windows locks are often brief), then re-raise."""
    for attempt in range(DELETE_ATTEMPTS):
        try:
            action(*args)
            return
        except OSError:
            if attempt == DELETE_ATTEMPTS - 1:
                raise
            time.sleep(DELETE_DELAY_S)


def _remove_job_folder(folder: Path) -> None:
    """Remove a job folder: every other file first, `job.json` and the folder last.

    Raises OSError if a file stays locked; `job.json` is then still there, so the job is
    listed again after a restart and can be deleted later.
    """
    if not folder.is_dir():
        return
    for path in sorted(folder.iterdir()):
        if path.name != JOB_FILE:
            _with_retries(_delete_path, path)
    _with_retries(_delete_path, folder / JOB_FILE)
    _with_retries(folder.rmdir)


def _short_error(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}".strip()
    return text if len(text) <= MAX_ERROR else text[: MAX_ERROR - 3] + "..."


def build_result(job: VideoJob, options: ProcessOptions, result: VideoResult) -> dict[str, Any]:
    s = result.summary
    return {
        "job_id": job.job_id,
        "video_name": job.video_name,
        "line": _line_list(job.line),
        "model_name": options.model_name,
        "fps": result.fps,
        "duration_s": result.duration_s,
        "processing_fps": result.processing_fps,
        "totals": {d.value: s.total.get(d, 0) for d in Direction},
        "total_count": s.total_count,
        "by_class": {cls.value: n for cls, n in s.by_class.items()},
        "timeline": [{"minute": b.minute, "one": b.one, "two": b.two} for b in s.timeline],
        "too_many": s.too_many,
        "too_few": s.too_few,
        "reliability": s.reliability.value,
        "total_text": {"tr": format_total(s, "tr"), "en": format_total(s, "en")},
        "flags": [
            {
                "kind": f.kind.value,
                "track_id": f.track_id,
                "time_s": f.time_s,
                "counted": f.counted,
                "detail": f.detail,
            }
            for f in s.flags
        ],
    }


class JobRunner:
    """Thread-safe job registry with a single FIFO worker thread.

    `get`/`delete`/`result` raise KeyError for unknown or malformed ids (ids are validated as
    uuid4 hex before any file-system access); `result` raises ValueError if the job has no
    result (not DONE); `delete` raises ValueError for a RUNNING job.
    """

    def __init__(self, output_root: Path, source: TrackSource):
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.source = source
        self._lock = threading.Lock()
        self._jobs: dict[str, _Entry] = {}
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._stop = threading.Event()
        self._load_persisted()
        self._worker = threading.Thread(target=self._work, name="visionflow-jobs", daemon=True)
        self._worker.start()

    # --- public API -------------------------------------------------------------------------

    def submit(
        self, video_path: Path, display_name: str, line: Line, options: ProcessOptions
    ) -> str:
        video_path = Path(video_path)
        if not video_path.is_file():
            raise FileNotFoundError(f"video not found: {video_path.name}")
        if self._stop.is_set():
            raise RuntimeError("job runner is shut down")
        job_id = uuid.uuid4().hex
        job = VideoJob(job_id, safe_stem(display_name), line, JobStatus.QUEUED, 0.0, None)
        entry = _Entry(job, time.time(), video_path, options)
        (self.output_root / job_id).mkdir()
        with self._lock:
            self._jobs[job_id] = entry
            self._persist(entry)
        self._queue.put(job_id)
        return job_id

    def get(self, job_id: str) -> VideoJob:
        with self._lock:
            return dataclasses.replace(self._entry(job_id).job)

    def list_jobs(self) -> list[VideoJob]:
        with self._lock:
            entries = sorted(self._jobs.values(), key=lambda e: e.created)
            return [dataclasses.replace(e.job) for e in entries]

    def result(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            entry = self._entry(job_id)
            if entry.job.status != JobStatus.DONE:
                raise ValueError(f"job {job_id} has no result (status {entry.job.status})")
            path = self.output_root / job_id / RESULT_FILE
            return json.loads(path.read_text(encoding="utf-8"))

    def delete(self, job_id: str) -> None:
        """Remove a job and its folder. Raises OSError (the job stays listed and can be
        deleted later) if a file is still locked, e.g. the video is open in a player."""
        with self._lock:
            entry = self._entry(job_id)
            if entry.job.status == JobStatus.RUNNING:
                raise ValueError(f"job {job_id} is running and cannot be deleted")
            # Under the lock, so a QUEUED job cannot start while its folder is removed.
            _remove_job_folder(self.output_root / job_id)
            del self._jobs[job_id]
            # A QUEUED job is skipped by the worker; drop its source video here.
            if entry.video_path is not None:
                _remove_source(entry.video_path)

    def sweep_temp_files(self) -> list[Path]:
        """Delete `*.tmp*` leftovers (e.g. from a crash) in the folders of jobs that are not
        RUNNING. Best effort; returns the removed paths."""
        removed: list[Path] = []
        with self._lock:  # a job cannot become RUNNING (and start writing) meanwhile
            for job_id, entry in self._jobs.items():
                if entry.job.status == JobStatus.RUNNING:
                    continue
                folder = self.output_root / job_id
                if not folder.is_dir():
                    continue
                for path in folder.glob("*.tmp*"):
                    try:
                        _delete_path(path)
                        removed.append(path)
                    except OSError:
                        continue
        return removed

    def shutdown(self, timeout: float = SHUTDOWN_TIMEOUT_S) -> None:
        """Stop the worker: the running job and all queued jobs end as ERROR "interrupted"."""
        self._stop.set()
        self._queue.put(None)
        self._worker.join(timeout)

    # --- internals --------------------------------------------------------------------------

    def _entry(self, job_id: str) -> _Entry:
        if not isinstance(job_id, str) or not _JOB_ID.fullmatch(job_id):
            raise KeyError(job_id)
        entry = self._jobs.get(job_id)
        if entry is None:
            raise KeyError(job_id)
        return entry

    def _persist(self, entry: _Entry) -> None:
        """Write job.json; caller holds the lock. Skipped if the job folder was deleted."""
        folder = self.output_root / entry.job.job_id
        if entry.job.job_id not in self._jobs or not folder.is_dir():
            return
        job = entry.job
        record: dict[str, Any] = {
            "job_id": job.job_id,
            "video_name": job.video_name,
            "line": _line_list(job.line),
            "status": job.status.value,
            "error": job.error,
            "created": entry.created,
        }
        if entry.video_path is not None and job.status in (JobStatus.QUEUED, JobStatus.RUNNING):
            # Lets a later start-up delete the source if this process dies before it does.
            record["source"] = str(entry.video_path)
        _write_json(folder / JOB_FILE, record)

    def _persist_quietly(self, entry: _Entry) -> None:
        """Like `_persist`, but a failed write (e.g. job.json locked by OneDrive or an
        antivirus scanner on Windows) never raises; the in-memory state stays authoritative."""
        try:
            self._persist(entry)
        except OSError:
            pass

    def _load_persisted(self) -> None:
        for folder in self.output_root.iterdir():
            if not (folder.is_dir() and _JOB_ID.fullmatch(folder.name)):
                continue
            try:
                data = json.loads((folder / JOB_FILE).read_text(encoding="utf-8"))
                x1, y1, x2, y2 = (float(v) for v in data["line"])
                job = VideoJob(
                    folder.name,
                    safe_stem(str(data["video_name"])),
                    Line((x1, y1), (x2, y2)),
                    JobStatus(data["status"]),
                    0.0,
                    data.get("error"),
                )
                created = float(data.get("created", 0.0))
                source = data.get("source")
            except (OSError, ValueError, KeyError, TypeError):
                continue  # not a readable job folder; leave it alone
            stale = job.status in (JobStatus.QUEUED, JobStatus.RUNNING)
            if stale:
                job.status, job.error = JobStatus.ERROR, INTERRUPTED
                if isinstance(source, str) and source:
                    orphan = Path(source)
                    if orphan.is_file():  # best effort: the old process never got to it
                        _remove_source(orphan)
                    else:
                        remove_empty_upload_folder(orphan)
            job.progress = 1.0 if job.status == JobStatus.DONE else 0.0
            entry = _Entry(job, created)
            self._jobs[job.job_id] = entry
            if stale:  # only rewrite records that changed; never let a write stop startup
                self._persist_quietly(entry)

    def _set_status(self, job_id: str, status: JobStatus, error: str | None = None) -> None:
        with self._lock:
            entry = self._jobs.get(job_id)
            if entry is None:
                return
            entry.job.status = status
            entry.job.error = error
            entry.job.phase = None
            if status == JobStatus.DONE:
                entry.job.progress = 1.0
            self._persist_quietly(entry)

    def _set_phase(self, job_id: str, phase: str) -> None:
        with self._lock:
            entry = self._jobs.get(job_id)
            if entry is not None and entry.job.status == JobStatus.RUNNING:
                entry.job.phase = phase

    def _on_progress(self, job_id: str, value: float) -> None:
        now = time.monotonic()
        with self._lock:
            entry = self._jobs.get(job_id)
            if entry is None:
                return
            job = entry.job
            due = now - entry.last_progress_at >= PROGRESS_INTERVAL_S
            if due or value - job.progress >= PROGRESS_STEP or value >= 1.0:
                job.progress = value
                entry.last_progress_at = now

    def _work(self) -> None:
        while True:
            job_id = self._queue.get()
            if job_id is None:
                return
            with self._lock:
                entry = self._jobs.get(job_id)
            if entry is None:
                continue  # deleted while queued
            # Everything per job is guarded, including the RUNNING persist: a failure ends
            # this job as ERROR (source deleted) and the worker moves on to the next job.
            try:
                if self._stop.is_set():
                    self._finish(entry, None, _Interrupted(INTERRUPTED))
                elif self._mark_running(entry):
                    self._run(entry)
            except Exception as exc:
                try:
                    self._finish(entry, self.output_root / job_id, exc)
                except Exception:
                    with self._lock:
                        entry.job.status, entry.job.error = JobStatus.ERROR, _short_error(exc)

    def _mark_running(self, entry: _Entry) -> bool:
        """QUEUED -> RUNNING, persisted (may raise). False if the job was deleted meanwhile."""
        with self._lock:
            if entry.job.job_id not in self._jobs:
                return False
            entry.job.status = JobStatus.RUNNING
            entry.job.phase = PHASE_LOADING
            self._persist(entry)
            return True

    def _run(self, entry: _Entry) -> None:
        job = entry.job
        folder = self.output_root / job.job_id
        assert entry.video_path is not None and entry.options is not None
        try:
            info = probe_video(entry.video_path)
            with AnnotatedWriter(
                folder / VIDEO_FILE, info.width, info.height, info.fps, job.line
            ) as writer:
                processing = False

                def on_frame(frame: FrameResult, events: list[CrossingEvent]) -> None:
                    nonlocal processing
                    if self._stop.is_set():
                        raise _Interrupted(INTERRUPTED)
                    if not processing:  # first frame: the model is loaded
                        processing = True
                        self._set_phase(job.job_id, PHASE_PROCESSING)
                    writer.write(frame, events)

                result = process_video(
                    entry.video_path,
                    job.line,
                    entry.options,
                    self.source,
                    on_frame=on_frame,
                    on_progress=lambda p: self._on_progress(job.job_id, p),
                )
                # Leaving the `with` transcodes the annotated video (can take a while).
                self._set_phase(job.job_id, PHASE_ENCODING)
            write_csv(result, job.video_name, folder / CSV_FILE)
            write_summary_csv(
                result, job.video_name, entry.options.model_name, folder / SUMMARY_FILE
            )
            _write_json(folder / RESULT_FILE, build_result(job, entry.options, result))
        except Exception as exc:  # any failure ends this job only; the worker keeps going
            self._finish(entry, folder, exc)
            return
        finally:
            _remove_source(entry.video_path)
        self._set_status(job.job_id, JobStatus.DONE)

    def _finish(self, entry: _Entry, folder: Path | None, exc: Exception) -> None:
        """Mark a job as ERROR, remove partial outputs and its source video."""
        if folder is not None:
            for name in (VIDEO_FILE, CSV_FILE, SUMMARY_FILE, RESULT_FILE):
                _remove_quietly(folder / name)
        if entry.video_path is not None:
            _remove_source(entry.video_path)
        error = INTERRUPTED if isinstance(exc, _Interrupted) else _short_error(exc)
        self._set_status(entry.job.job_id, JobStatus.ERROR, error)
