"""FastAPI app: upload, job and download endpoints (see the design spec, sections 3-5).

Security defaults: no CORS middleware; ids from the URL are validated as 32-char hex before any
file-system access; stored and downloaded names are always the sanitized `safe_stem`; only a
fixed allow-list of model names is accepted. A Host allow-list and a same-origin check on
state-changing requests keep other web sites from driving the local server. The web UI
(`web/`) is served at `/` and `/static`.
"""

import functools
import re
import shutil
import threading
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

import cv2
from fastapi import FastAPI, Request, Response, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from . import paths
from .geometry import Line
from .jobs import CSV_FILE, SUMMARY_FILE, VIDEO_FILE, JobRunner, JobStatus, safe_stem
from .pipeline import ProcessOptions
from .tracking import VideoReadError, probe_video

ALLOWED_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv")
ALLOWED_MODELS = ("yolo11n.pt", "yolo11s.pt", "yolo11m.pt")
MAX_FILES = 5
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
CHUNK_BYTES = 1024 * 1024
MIN_CONFIDENCE, MAX_CONFIDENCE = 0.05, 0.95
DEFAULT_MODEL = "yolo11s.pt"  # POST /api/jobs default when model_name is missing
CPU_DEFAULT_MODEL = "yolo11n.pt"  # suggested by GET /api/system when CUDA is unavailable
REPO_ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = REPO_ROOT / "web"
DEFAULT_ALLOWED_HOSTS = frozenset({"127.0.0.1", "localhost", "[::1]"})

_HEX_ID = re.compile(r"[0-9a-f]{32}")

MESSAGES: dict[str, dict[str, str]] = {
    "no_files": {"tr": "En az bir video dosyası yükleyin.", "en": "Upload at least one video."},
    "too_many_files": {
        "tr": "Bir seferde en çok {max} video yüklenebilir.",
        "en": "At most {max} videos can be uploaded at once.",
    },
    "bad_ext": {
        "tr": "Desteklenmeyen dosya türü: {name}. İzin verilenler: {allowed}.",
        "en": "Unsupported file type: {name}. Allowed: {allowed}.",
    },
    "too_large": {
        "tr": "Dosya çok büyük (en çok {limit_mb} MB).",
        "en": "File is too large (at most {limit_mb} MB).",
    },
    "unreadable": {
        "tr": "Video okunamadı veya bozuk: {name}. Hiçbir dosya kaydedilmedi.",
        "en": "Video could not be read or is corrupt: {name}. No file was kept.",
    },
    "video_used": {
        "tr": "Bu video zaten bir işe gönderildi.",
        "en": "This video was already submitted.",
    },
    "invalid_request": {"tr": "Geçersiz istek.", "en": "Invalid request."},
    "invalid_id": {"tr": "Geçersiz kimlik.", "en": "Invalid id."},
    "video_not_found": {"tr": "Video bulunamadı.", "en": "Video not found."},
    "job_not_found": {"tr": "İş bulunamadı.", "en": "Job not found."},
    "job_not_done": {"tr": "İş henüz tamamlanmadı.", "en": "The job is not finished."},
    "job_running": {
        "tr": "İş çalışıyor, şimdi silinemez.",
        "en": "The job is running and cannot be deleted now.",
    },
    "result_missing": {
        "tr": "Sonuç dosyası bulunamadı.",
        "en": "The result file is missing.",
    },
    "result_corrupt": {"tr": "Sonuç dosyası okunamadı.", "en": "The result file is unreadable."},
    "file_in_use": {
        "tr": "Bir dosya kullanımda. Videoyu kapatıp tekrar deneyin.",
        "en": "A file is in use. Close the video and try again.",
    },
    "bad_video_count": {
        "tr": "Bir iş için 1-{max} video seçilmelidir.",
        "en": "A job needs between 1 and {max} videos.",
    },
    "duplicate_video": {
        "tr": "Aynı video birden fazla kez seçildi.",
        "en": "The same video was selected more than once.",
    },
    "bad_line": {
        "tr": "Çizgi dört sayıdan oluşmalı, her biri 0-1 aralığında ve uçlar farklı olmalı.",
        "en": "A line needs four numbers in the range 0-1 and two distinct endpoints.",
    },
    "bad_model": {
        "tr": "Geçersiz model. İzin verilenler: {allowed}.",
        "en": "Invalid model. Allowed: {allowed}.",
    },
    "bad_host": {"tr": "Bu ana makine adına izin verilmiyor.", "en": "This host is not allowed."},
    "bad_origin": {
        "tr": "Farklı kaynaktan gelen istek reddedildi.",
        "en": "Cross-origin request rejected.",
    },
    "bad_confidence": {
        "tr": "Güven eşiği {lo} ile {hi} arasında olmalı.",
        "en": "Confidence must be between {lo} and {hi}.",
    },
}


class ApiError(Exception):
    def __init__(self, status: int, key: str, **params: object):
        self.status, self.key, self.params = status, key, params


def _language(request: Request) -> str:
    lang = request.query_params.get("lang", "").lower()
    if lang in ("tr", "en"):
        return lang
    header = request.headers.get("accept-language", "").strip().lower()
    return "en" if header.startswith("en") else "tr"


def _message(request: Request, key: str, **params: object) -> str:
    return MESSAGES[key][_language(request)].format(**params)


class VideoLine(BaseModel):
    video_id: str
    line: list[float]


class JobRequest(BaseModel):
    videos: list[VideoLine]
    model_name: str = DEFAULT_MODEL
    confidence: float = 0.25


def _default_runner() -> JobRunner:
    from .tracking import UltralyticsTrackSource  # lazy: keeps torch/ultralytics off import

    return JobRunner(paths.output_dir(), UltralyticsTrackSource())


@functools.cache
def _cuda_available() -> bool:
    """Whether torch sees a CUDA GPU; torch is imported only here (first call is slow)."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # no torch, a broken install or a driver problem: treat as no GPU
        return False


def _check_id(value: str) -> str:
    if not _HEX_ID.fullmatch(value):
        raise ApiError(400, "invalid_id")
    return value


def _extension(filename: str | None) -> tuple[str, str]:
    """(sanitized stem, lower-case extension) of an uploaded name; raises on a bad type."""
    name = re.split(r"[\\/]", filename or "")[-1]
    ext = Path(name).suffix.lower()
    stem = safe_stem(name)
    if ext not in ALLOWED_EXTENSIONS:
        raise ApiError(400, "bad_ext", name=stem + ext, allowed=" ".join(ALLOWED_EXTENSIONS))
    return stem, ext


def _stream_to_disk(upload: UploadFile, dest: Path, limit: int) -> None:
    written = 0
    with dest.open("wb") as out:
        while chunk := upload.file.read(CHUNK_BYTES):
            written += len(chunk)
            if written > limit:
                raise ApiError(413, "too_large", limit_mb=limit // (1024 * 1024))
            out.write(chunk)


def _parse_line(values: list[float]) -> Line:
    if len(values) != 4 or not all(0.0 <= v <= 1.0 for v in values):
        raise ApiError(400, "bad_line")
    x1, y1, x2, y2 = values
    if (x1, y1) == (x2, y2):
        raise ApiError(400, "bad_line")
    return Line((x1, y1), (x2, y2))


class _UploadBodyCap:
    """Pure-ASGI guard for `POST /api/videos`: caps the whole request body.

    A declared Content-Length above the cap is rejected before any body is read; otherwise the
    bytes actually received are counted (also for chunked bodies) and the request is aborted
    with 413 once the cap is passed, so at most about `cap` bytes reach the multipart spool.
    """

    def __init__(self, app: ASGIApp, cap: int, limit_mb: int):
        self.app, self.cap, self.limit_mb = app, cap, limit_mb

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] != "POST" or scope["path"] != "/api/videos":
            await self.app(scope, receive, send)
            return
        reply = JSONResponse(
            {"detail": _message(Request(scope), "too_large", limit_mb=self.limit_mb)},
            status_code=413,
        )
        declared = Request(scope).headers.get("content-length", "")
        if declared.isdigit() and int(declared) > self.cap:
            await reply(scope, receive, send)
            return
        received = 0
        exceeded = False

        async def counting_receive() -> Message:
            nonlocal received, exceeded
            message = await receive()
            if message["type"] == "http.request" and not exceeded:
                received += len(message.get("body", b""))
                if received > self.cap:
                    exceeded = True
                    await reply(scope, receive, send)
            if exceeded:
                return {"type": "http.disconnect"}
            return message

        async def guarded_send(message: Message) -> None:
            if not exceeded:
                await send(message)

        try:
            await self.app(scope, counting_receive, guarded_send)
        except Exception:
            if not exceeded:
                raise


def _host_without_port(host: str) -> str:
    host = host.strip().lower()
    if host.startswith("["):
        return host[: host.find("]") + 1] if "]" in host else host
    return host.rsplit(":", 1)[0] if ":" in host else host


class _HostOriginGuard:
    """Pure-ASGI guard: unknown Host -> 403; cross-origin state-changing request -> 403."""

    def __init__(self, app: ASGIApp, allowed_hosts: frozenset[str]):
        self.app, self.allowed_hosts = app, allowed_hosts

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request = Request(scope)
        host = request.headers.get("host", "")
        key = None
        if _host_without_port(host) not in self.allowed_hosts:
            key = "bad_host"
        elif scope["method"] not in ("GET", "HEAD", "OPTIONS"):
            origin = request.headers.get("origin")
            if origin is not None and urlsplit(origin).netloc.lower() != host.strip().lower():
                key = "bad_origin"
        if key is not None:
            reply = JSONResponse({"detail": _message(request, key)}, status_code=403)
            await reply(scope, receive, send)
            return
        await self.app(scope, receive, send)


def create_app(
    runner: JobRunner | None = None,
    upload_dir: Path | None = None,
    max_upload_bytes: int = MAX_UPLOAD_BYTES,
    allowed_hosts: set[str] | None = None,
) -> FastAPI:
    """Build the app. Defaults keep data outside the repository (see `paths.data_dir`)."""
    owns_runner = runner is None
    job_runner = runner if runner is not None else _default_runner()
    uploads = Path(upload_dir) if upload_dir is not None else paths.upload_dir()
    # Start-up cleanup (NFR-7): stale/empty upload folders and temp files of crashed jobs.
    paths.sweep_uploads(uploads)
    job_runner.sweep_temp_files()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        yield
        if owns_runner:
            job_runner.shutdown()

    app = FastAPI(title="visionflow", lifespan=lifespan)
    app.add_middleware(
        _UploadBodyCap,
        cap=max_upload_bytes * MAX_FILES + CHUNK_BYTES,
        limit_mb=max_upload_bytes // (1024 * 1024),
    )
    hosts = frozenset(
        h.lower() for h in (DEFAULT_ALLOWED_HOSTS if allowed_hosts is None else allowed_hosts)
    )
    app.add_middleware(_HostOriginGuard, allowed_hosts=hosts)  # added last = outermost
    submitted: set[str] = set()  # video ids already turned into jobs (the runner eats the file)
    submit_lock = threading.Lock()

    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(
            {"detail": _message(request, exc.key, **exc.params)}, status_code=exc.status
        )

    @app.exception_handler(RequestValidationError)
    async def _invalid(request: Request, _exc: RequestValidationError) -> JSONResponse:
        return JSONResponse({"detail": _message(request, "invalid_request")}, status_code=400)

    def find_video(video_id: str) -> Path:
        folder = uploads / _check_id(video_id)
        files = [p for p in folder.iterdir() if p.is_file()] if folder.is_dir() else []
        if not files:
            if folder.is_dir():  # the runner already consumed the source; drop the empty folder
                shutil.rmtree(folder, ignore_errors=True)
            raise ApiError(404, "video_not_found")
        return files[0]

    def finished_job(job_id: str) -> Path:
        """Output folder of a DONE job; 404 unknown, 409 not DONE."""
        _check_id(job_id)
        try:
            job = job_runner.get(job_id)
        except KeyError:
            raise ApiError(404, "job_not_found") from None
        if job.status != JobStatus.DONE:
            raise ApiError(409, "job_not_done")
        return job_runner.output_root / job_id

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/system")
    def system() -> dict[str, object]:
        cuda = _cuda_available()
        return {"cuda": cuda, "default_model": DEFAULT_MODEL if cuda else CPU_DEFAULT_MODEL}

    @app.post("/api/videos")
    def upload_videos(files: list[UploadFile]) -> list[dict[str, object]]:
        if not files:
            raise ApiError(400, "no_files")
        if len(files) > MAX_FILES:
            raise ApiError(400, "too_many_files", max=MAX_FILES)
        names = [_extension(f.filename) for f in files]  # reject all before writing any

        created: list[Path] = []
        items: list[dict[str, object]] = []
        try:
            for upload, (stem, ext) in zip(files, names, strict=True):
                video_id = uuid.uuid4().hex
                folder = uploads / video_id
                folder.mkdir(parents=True)
                created.append(folder)
                dest = folder / f"{stem}{ext}"
                _stream_to_disk(upload, dest, max_upload_bytes)
                try:
                    info = probe_video(dest)
                except VideoReadError:
                    raise ApiError(400, "unreadable", name=stem) from None
                items.append(
                    {"video_id": video_id, "name": stem, "width": info.width, "height": info.height}
                )
        except BaseException:
            for folder in created:
                shutil.rmtree(folder, ignore_errors=True)
            raise
        return items

    @app.get("/api/videos/{video_id}/preview")
    def preview(video_id: str) -> Response:
        path = find_video(video_id)
        cap = cv2.VideoCapture(str(path))
        try:
            ok, image = cap.read()
        finally:
            cap.release()
        if not ok or image is None:
            raise ApiError(404, "video_not_found")
        ok, buffer = cv2.imencode(".jpg", image)
        if not ok:
            raise ApiError(404, "video_not_found")
        return Response(buffer.tobytes(), media_type="image/jpeg")

    @app.delete("/api/videos/{video_id}", status_code=204)
    def delete_video(video_id: str) -> Response:
        """Drop an uploaded video that was not (and now cannot be) turned into a job."""
        with submit_lock:  # never races with POST /api/jobs taking the same video
            if _check_id(video_id) in submitted:
                raise ApiError(409, "video_used")
            folder = find_video(video_id).parent
            try:
                shutil.rmtree(folder)
            except OSError:
                raise ApiError(409, "file_in_use") from None
        return Response(status_code=204)

    @app.post("/api/jobs")
    def create_jobs(body: JobRequest) -> list[dict[str, str]]:
        if not 1 <= len(body.videos) <= MAX_FILES:
            raise ApiError(400, "bad_video_count", max=MAX_FILES)
        ids = [v.video_id for v in body.videos]
        if len(set(ids)) != len(ids):
            raise ApiError(400, "duplicate_video")
        if body.model_name not in ALLOWED_MODELS:
            raise ApiError(400, "bad_model", allowed=" ".join(ALLOWED_MODELS))
        if not MIN_CONFIDENCE <= body.confidence <= MAX_CONFIDENCE:
            raise ApiError(400, "bad_confidence", lo=MIN_CONFIDENCE, hi=MAX_CONFIDENCE)
        lines = [_parse_line(v.line) for v in body.videos]
        options = ProcessOptions(model_name=body.model_name, confidence=body.confidence)
        created = []
        with submit_lock:  # check all, mark all, then submit: atomic for the whole request
            if any(i in submitted for i in ids):
                raise ApiError(409, "video_used")
            try:  # every video must exist before any is submitted
                paths = [find_video(i) for i in ids]
            except ApiError as exc:
                raise ApiError(400, exc.key) from None
            submitted.update(ids)
            for video_id, path, line in zip(ids, paths, lines, strict=True):
                try:
                    job_id = job_runner.submit(path, path.stem, line, options)
                except FileNotFoundError:
                    submitted.discard(video_id)
                    raise ApiError(400, "video_not_found") from None
                created.append({"job_id": job_id, "video_id": video_id})
        return created

    def status_of(job_id: str) -> dict[str, object]:
        _check_id(job_id)
        try:
            job = job_runner.get(job_id)
        except KeyError:
            raise ApiError(404, "job_not_found") from None
        return {
            "job_id": job.job_id,
            "video_name": job.video_name,
            "status": job.status.value,
            "progress": job.progress,
            "phase": job.phase,
            "error": job.error,
        }

    @app.get("/api/jobs")
    def list_jobs() -> list[dict[str, object]]:
        return [status_of(j.job_id) for j in job_runner.list_jobs()]

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str) -> dict[str, object]:
        return status_of(job_id)

    @app.get("/api/jobs/{job_id}/result")
    def job_result(job_id: str) -> dict[str, object]:
        finished_job(job_id)
        try:
            return job_runner.result(job_id)
        except KeyError:
            raise ApiError(404, "job_not_found") from None
        except FileNotFoundError:
            raise ApiError(410, "result_missing") from None
        except ValueError:  # includes json.JSONDecodeError
            raise ApiError(500, "result_corrupt") from None

    def download(job_id: str, file_name: str, media_type: str, suffix: str) -> FileResponse:
        folder = finished_job(job_id)
        path = folder / file_name
        if not path.is_file():
            raise ApiError(410, "result_missing")
        name = f"{job_runner.get(job_id).video_name}{suffix}"
        return FileResponse(path, media_type=media_type, filename=name)

    @app.get("/api/jobs/{job_id}/video")
    def job_video(job_id: str) -> FileResponse:
        return download(job_id, VIDEO_FILE, "video/mp4", "_annotated.mp4")

    @app.get("/api/jobs/{job_id}/csv")
    def job_csv(job_id: str) -> FileResponse:
        return download(job_id, CSV_FILE, "text/csv", ".csv")

    @app.get("/api/jobs/{job_id}/summary")
    def job_summary(job_id: str) -> FileResponse:
        return download(job_id, SUMMARY_FILE, "text/csv", "_summary.csv")

    @app.delete("/api/jobs/{job_id}", status_code=204)
    def delete_job(job_id: str) -> Response:
        _check_id(job_id)
        try:
            job_runner.delete(job_id)
        except KeyError:
            raise ApiError(404, "job_not_found") from None
        except ValueError:
            raise ApiError(409, "job_running") from None
        except OSError:  # a file is still locked (e.g. open in a player); the job stays
            raise ApiError(409, "file_in_use") from None
        return Response(status_code=204)

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html", media_type="text/html")

    if WEB_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
    return app

