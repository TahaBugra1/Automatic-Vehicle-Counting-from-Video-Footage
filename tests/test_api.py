import json
import os
import sys
import threading
import time
import types

import pytest
from fakes import TEST_HOSTS, TIMEOUT_S, FakeSource, make_bad_video, make_video
from fastapi.testclient import TestClient

from visionflow import api, jobs
from visionflow.api import create_app
from visionflow.jobs import JobRunner


def make_client(runner, upload_dir, **kwargs):
    return TestClient(create_app(runner, upload_dir, allowed_hosts=TEST_HOSTS, **kwargs))


@pytest.fixture
def runner(tmp_path):
    r = JobRunner(tmp_path / "out", FakeSource())
    yield r
    r.shutdown()


@pytest.fixture
def upload_dir(tmp_path):
    return tmp_path / "uploads"


@pytest.fixture
def client(runner, upload_dir):
    return make_client(runner, upload_dir)


@pytest.fixture
def video_bytes(tmp_path):
    return make_video(tmp_path / "src.avi").read_bytes()


def upload(client, data, names=("clip.avi",), **kwargs):
    files = [("files", (n, data, "application/octet-stream")) for n in names]
    return client.post("/api/videos", files=files, **kwargs)


def job_body(video_ids, line=(0.0, 0.5, 1.0, 0.5), **extra):
    return {"videos": [{"video_id": v, "line": list(line)} for v in video_ids], **extra}


def wait_done(client, job_ids):
    deadline = time.monotonic() + TIMEOUT_S
    while time.monotonic() < deadline:
        states = [client.get(f"/api/jobs/{j}").json()["status"] for j in job_ids]
        if all(s in ("DONE", "ERROR") for s in states):
            return states
        time.sleep(0.02)
    raise AssertionError("jobs did not finish")


def run_job(client, video_bytes, **kwargs):
    vid = upload(client, video_bytes).json()[0]["video_id"]
    created = client.post("/api/jobs", json=job_body([vid], **kwargs))
    assert created.status_code == 200, created.text
    job_id = created.json()[0]["job_id"]
    assert wait_done(client, [job_id]) == ["DONE"]
    return job_id


def _must_not_run(*args, **kwargs):
    raise AssertionError("endpoint reached: the body cap did not stop the request")


def stored_files(upload_dir):
    return [p for p in upload_dir.rglob("*") if p.is_file()] if upload_dir.exists() else []


def test_health(client):
    assert client.get("/api/health").json() == {"status": "ok"}


def test_upload_returns_metadata_and_preview(client, video_bytes):
    resp = upload(client, video_bytes, names=("Yol Görüntüsü ç.avi",))
    assert resp.status_code == 200
    (item,) = resp.json()
    assert item["name"] == "Yol_Goruntusu_c"
    assert (item["width"], item["height"]) == (64, 48)
    preview = client.get(f"/api/videos/{item['video_id']}/preview")
    assert preview.status_code == 200
    assert preview.headers["content-type"] == "image/jpeg"
    assert preview.content[:2] == b"\xff\xd8"


def test_upload_rejects_bad_extension(client, video_bytes, upload_dir):
    resp = upload(client, video_bytes, names=("notes.txt",))
    assert resp.status_code == 400
    assert stored_files(upload_dir) == []


def test_upload_rejects_more_than_five_files(client, video_bytes, upload_dir):
    resp = upload(client, video_bytes, names=[f"v{i}.avi" for i in range(6)])
    assert resp.status_code == 400
    assert stored_files(upload_dir) == []


def test_upload_rejects_oversize(runner, upload_dir, video_bytes):
    client = make_client(runner, upload_dir, max_upload_bytes=len(video_bytes) - 1)
    resp = upload(client, video_bytes)
    assert resp.status_code == 413
    assert stored_files(upload_dir) == []


def test_corrupt_upload_is_rejected_and_not_kept(client, tmp_path, upload_dir, video_bytes):
    bad = make_bad_video(tmp_path / "bad.avi").read_bytes()
    resp = client.post(
        "/api/videos",
        files=[
            ("files", ("good.avi", video_bytes, "video/x-msvideo")),
            ("files", ("bad.avi", bad, "video/x-msvideo")),
        ],
    )
    assert resp.status_code == 400
    assert "bad" in resp.json()["detail"]
    assert stored_files(upload_dir) == []


def test_job_rejects_line_out_of_range(client, video_bytes):
    vid = upload(client, video_bytes).json()[0]["video_id"]
    for line in [(0.0, 0.5, 1.2, 0.5), (-0.1, 0.5, 1.0, 0.5), (0.0, 0.5, 1.0)]:
        resp = client.post("/api/jobs", json=job_body([vid], line=line))
        assert resp.status_code == 400, line
    assert client.get("/api/jobs").json() == []


def test_job_rejects_identical_endpoints(client, video_bytes):
    vid = upload(client, video_bytes).json()[0]["video_id"]
    resp = client.post("/api/jobs", json=job_body([vid], line=(0.3, 0.3, 0.3, 0.3)))
    assert resp.status_code == 400


def test_job_rejects_other_bad_bodies(client, video_bytes):
    vid = upload(client, video_bytes).json()[0]["video_id"]
    bodies = [
        job_body([]),
        job_body([vid, vid]),
        job_body([vid], confidence=0.01),
        job_body([vid], confidence=0.99),
        job_body(["0" * 32]),
        {"nonsense": 1},
    ]
    for body in bodies:
        assert client.post("/api/jobs", json=body).status_code == 400, body


def test_model_name_not_in_allow_list_is_rejected(client, video_bytes):
    vid = upload(client, video_bytes).json()[0]["video_id"]
    for name in ("../../evil.pt", "yolo11x.pt", "yolo11s.pt "):
        resp = client.post("/api/jobs", json=job_body([vid], model_name=name))
        assert resp.status_code == 400, name


def test_each_video_uses_its_own_line(client, video_bytes):
    ids = [d["video_id"] for d in upload(client, video_bytes, names=("a.avi", "b.avi")).json()]
    body = {
        "videos": [
            {"video_id": ids[0], "line": [0.0, 0.5, 1.0, 0.5]},
            {"video_id": ids[1], "line": [1.0, 0.5, 0.0, 0.5]},
        ]
    }
    created = client.post("/api/jobs", json=body).json()
    assert [c["video_id"] for c in created] == ids
    job_ids = [c["job_id"] for c in created]
    assert wait_done(client, job_ids) == ["DONE", "DONE"]
    r0, r1 = (client.get(f"/api/jobs/{j}/result").json() for j in job_ids)
    assert r0["line"] == [0.0, 0.5, 1.0, 0.5]
    assert r1["line"] == [1.0, 0.5, 0.0, 0.5]
    assert r0["total_count"] == r1["total_count"] == 1
    assert r0["totals"] != r1["totals"]


def test_result_contains_total_text_and_flags(client, video_bytes):
    job_id = run_job(client, video_bytes)
    data = client.get(f"/api/jobs/{job_id}/result").json()
    assert data["job_id"] == job_id
    assert set(data["total_text"]) == {"tr", "en"}
    assert isinstance(data["flags"], list)
    status = client.get(f"/api/jobs/{job_id}").json()
    assert status["status"] == "DONE" and status["progress"] == 1.0
    assert [j["job_id"] for j in client.get("/api/jobs").json()] == [job_id]


def test_result_not_ready_is_409(runner, upload_dir, video_bytes):
    gate = threading.Event()
    runner.source.gate = gate
    client = make_client(runner, upload_dir)
    vid = upload(client, video_bytes).json()[0]["video_id"]
    job_id = client.post("/api/jobs", json=job_body([vid])).json()[0]["job_id"]
    try:
        for suffix in ("result", "video", "csv", "summary"):
            assert client.get(f"/api/jobs/{job_id}/{suffix}").status_code == 409
        assert runner.source.started.wait(TIMEOUT_S)
        assert client.delete(f"/api/jobs/{job_id}").status_code == 409
    finally:
        gate.set()
    assert wait_done(client, [job_id]) == ["DONE"]


def test_download_csv_and_video(client, video_bytes):
    job_id = run_job(client, video_bytes)
    csv = client.get(f"/api/jobs/{job_id}/csv")
    assert csv.status_code == 200
    assert csv.headers["content-disposition"].startswith("attachment")
    assert "clip.csv" in csv.headers["content-disposition"]
    video = client.get(f"/api/jobs/{job_id}/video")
    assert video.status_code == 200
    assert video.headers["content-disposition"].startswith("attachment")
    assert "clip_annotated.mp4" in video.headers["content-disposition"]
    assert len(video.content) > 0


def test_download_summary_csv(client, video_bytes):
    job_id = run_job(client, video_bytes)
    resp = client.get(f"/api/jobs/{job_id}/summary")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/csv")
    assert resp.headers["content-disposition"].startswith("attachment")
    assert "clip_summary.csv" in resp.headers["content-disposition"]
    assert resp.content.startswith(b"\xef\xbb\xbfmetrik,deger")
    csv_resp = client.get(f"/api/jobs/{job_id}/csv")
    assert csv_resp.content.startswith("﻿tür".encode())


def test_delete_job(client, video_bytes):
    job_id = run_job(client, video_bytes)
    assert client.delete(f"/api/jobs/{job_id}").status_code == 204
    assert client.get(f"/api/jobs/{job_id}").status_code == 404
    assert client.delete(f"/api/jobs/{job_id}").status_code == 404


def test_delete_job_with_locked_file_is_409_and_job_stays(client, video_bytes, monkeypatch):
    job_id = run_job(client, video_bytes)
    real_delete = jobs._delete_path

    def locked(path):
        if path.name == "annotated.mp4":
            raise PermissionError("in use")
        real_delete(path)

    monkeypatch.setattr(jobs, "DELETE_DELAY_S", 0.0)
    monkeypatch.setattr(jobs, "_delete_path", locked)
    resp = client.delete(f"/api/jobs/{job_id}?lang=en")
    assert resp.status_code == 409
    assert "close the video" in resp.json()["detail"].lower()
    assert client.get(f"/api/jobs/{job_id}").json()["status"] == "DONE"
    monkeypatch.setattr(jobs, "_delete_path", real_delete)
    assert client.delete(f"/api/jobs/{job_id}").status_code == 204


def test_delete_uploaded_video(client, video_bytes, upload_dir):
    vid = upload(client, video_bytes).json()[0]["video_id"]
    assert (upload_dir / vid).is_dir()
    assert client.delete(f"/api/videos/{vid}").status_code == 204
    assert not (upload_dir / vid).exists()
    assert client.get(f"/api/videos/{vid}/preview").status_code == 404
    assert client.delete(f"/api/videos/{vid}").status_code == 404
    assert client.post("/api/jobs", json=job_body([vid])).status_code == 400


def test_delete_submitted_video_is_409(runner, upload_dir, video_bytes):
    gate = threading.Event()
    runner.source.gate = gate
    client = make_client(runner, upload_dir)
    vid = upload(client, video_bytes).json()[0]["video_id"]
    job_id = client.post("/api/jobs", json=job_body([vid])).json()[0]["job_id"]
    try:
        assert client.delete(f"/api/videos/{vid}").status_code == 409
    finally:
        gate.set()
    assert wait_done(client, [job_id]) == ["DONE"]
    assert client.delete(f"/api/videos/{vid}").status_code == 409
    assert not (upload_dir / vid).exists()  # consumed: the empty folder is gone too


def test_startup_sweeps_old_uploads_and_temp_leftovers(tmp_path, upload_dir, video_bytes):
    old = upload_dir / ("a" * 32)
    old.mkdir(parents=True)
    (old / "v.avi").write_bytes(video_bytes)
    past = time.time() - 25 * 3600
    for path in (old / "v.avi", old):
        os.utime(path, (past, past))
    fresh = upload_dir / ("b" * 32)
    fresh.mkdir()
    (fresh / "v.avi").write_bytes(video_bytes)

    out = tmp_path / "out2"
    done_id = "c" * 32
    (out / done_id).mkdir(parents=True)
    record = {"job_id": done_id, "video_name": "x", "line": [0, 0.5, 1, 0.5],
              "status": "DONE", "error": None, "created": 1.0}  # fmt: skip
    (out / done_id / "job.json").write_text(json.dumps(record), encoding="utf-8")
    (out / done_id / "annotated.raw.tmp.mp4").write_bytes(b"x")
    second = JobRunner(out, FakeSource())
    try:
        make_client(second, upload_dir)
    finally:
        second.shutdown()
    assert not old.exists()
    assert fresh.is_dir()
    assert not (out / done_id / "annotated.raw.tmp.mp4").exists()
    assert (out / done_id / "job.json").exists()


def test_job_status_has_phase(client, video_bytes):
    job_id = run_job(client, video_bytes)
    status = client.get(f"/api/jobs/{job_id}").json()
    assert "phase" in status and status["phase"] is None
    assert client.get("/api/jobs").json()[0]["phase"] is None


def test_system_endpoint_reports_default_model(client, monkeypatch):
    monkeypatch.setattr(api, "_cuda_available", lambda: False)
    assert client.get("/api/system").json() == {"cuda": False, "default_model": "yolo11n.pt"}
    monkeypatch.setattr(api, "_cuda_available", lambda: True)
    assert client.get("/api/system").json() == {"cuda": True, "default_model": "yolo11s.pt"}


def test_cuda_check_uses_torch_lazily_and_survives_missing_torch(monkeypatch):
    fake = types.ModuleType("torch")
    fake.cuda = types.SimpleNamespace(is_available=lambda: True)
    monkeypatch.setitem(sys.modules, "torch", fake)
    api._cuda_available.cache_clear()
    try:
        assert api._cuda_available() is True
        api._cuda_available.cache_clear()
        monkeypatch.setitem(sys.modules, "torch", None)  # import torch -> ImportError
        assert api._cuda_available() is False
        api._cuda_available.cache_clear()
        broken = types.ModuleType("torch")
        broken.cuda = types.SimpleNamespace(is_available=lambda: 1 / 0)
        monkeypatch.setitem(sys.modules, "torch", broken)
        assert api._cuda_available() is False
    finally:
        api._cuda_available.cache_clear()


def test_post_jobs_default_model_is_unchanged(client, video_bytes, monkeypatch):
    monkeypatch.setattr(api, "_cuda_available", lambda: False)
    vid = upload(client, video_bytes).json()[0]["video_id"]
    job_id = client.post("/api/jobs", json=job_body([vid])).json()[0]["job_id"]
    assert wait_done(client, [job_id]) == ["DONE"]
    assert client.get(f"/api/jobs/{job_id}/result").json()["model_name"] == "yolo11s.pt"


def test_default_hosts_exclude_testserver(runner, upload_dir):
    assert "testserver" not in api.DEFAULT_ALLOWED_HOSTS
    app = create_app(runner, upload_dir)
    assert TestClient(app).get("/api/health").status_code == 403
    local = TestClient(app, base_url="http://127.0.0.1:8000")
    assert local.get("/api/health").status_code == 200


def test_default_folders_come_from_the_data_dir(runner, tmp_path, video_bytes, monkeypatch):
    monkeypatch.setenv("VISIONFLOW_DATA_DIR", str(tmp_path / "data"))
    client = TestClient(create_app(runner, allowed_hosts=TEST_HOSTS))
    vid = upload(client, video_bytes).json()[0]["video_id"]
    assert (tmp_path / "data" / "uploads" / vid).is_dir()
    default = api._default_runner()
    try:
        assert default.output_root == tmp_path / "data" / "outputs"
        assert default.output_root.is_dir()
    finally:
        default.shutdown()


def test_unknown_or_malformed_ids_are_rejected(client):
    bad = ["xyz", "..", "0" * 31, "G" * 32, "A" * 32, "..%2F..%2Fx"]
    for bid in bad:
        for path in (
            f"/api/videos/{bid}/preview",
            f"/api/jobs/{bid}",
            f"/api/jobs/{bid}/result",
            f"/api/jobs/{bid}/video",
            f"/api/jobs/{bid}/csv",
            f"/api/jobs/{bid}/summary",
        ):
            assert client.get(path).status_code in (400, 404), path
        assert client.delete(f"/api/jobs/{bid}").status_code in (400, 404, 405)
        assert client.delete(f"/api/videos/{bid}").status_code in (400, 404, 405)
    unknown = "0" * 32
    assert client.get(f"/api/videos/{unknown}/preview").status_code == 404
    assert client.delete(f"/api/videos/{unknown}").status_code == 404
    assert client.delete("/api/videos/" + "A" * 32).status_code == 400
    assert client.get(f"/api/jobs/{unknown}").status_code == 404
    assert client.post("/api/jobs", json=job_body(["../x"])).status_code == 400


def test_errors_follow_lang_query_param(client, video_bytes):
    def post(url="/api/videos", **kw):
        files = [("files", ("a.txt", video_bytes, "text/plain"))]
        return client.post(url, files=files, **kw).json()["detail"]

    tr = post()
    en = post("/api/videos?lang=en")
    from_header = post(headers={"Accept-Language": "en-US,en;q=0.9"})
    query_wins = post("/api/videos?lang=tr", headers={"Accept-Language": "en"})
    assert tr != en
    assert en == from_header
    assert query_wins == tr
    assert "nsupported" in en


def test_upload_body_over_total_cap_is_rejected_with_413(runner, upload_dir, monkeypatch):
    monkeypatch.setattr(api, "_stream_to_disk", _must_not_run)
    client = make_client(runner, upload_dir, max_upload_bytes=1024)
    cap = 1024 * 5 + 1024 * 1024
    resp = client.post(
        "/api/videos",
        content=b"x",
        headers={"Content-Length": str(cap + 1), "Content-Type": "multipart/form-data"},
    )
    assert resp.status_code == 413
    assert stored_files(upload_dir) == [] and not upload_dir.exists()


def test_upload_without_content_length_is_capped_while_streaming(
    runner, upload_dir, monkeypatch
):
    monkeypatch.setattr(api, "_stream_to_disk", _must_not_run)
    client = make_client(runner, upload_dir, max_upload_bytes=1024)
    boundary = "bnd"
    head = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="files"; filename="a.avi"\r\n'
        "Content-Type: video/x-msvideo\r\n\r\n"
    ).encode()
    sent = 0

    def body():
        nonlocal sent
        yield head
        for _ in range(5 * 1024 + 8):  # 1 KiB pieces, past 5 KiB + 1 MiB
            sent += 1024
            yield b"y" * 1024
        yield f"\r\n--{boundary}--\r\n".encode()

    resp = client.post(
        "/api/videos",
        content=body(),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    assert resp.status_code == 413
    assert stored_files(upload_dir) == []
    assert "content-length" not in resp.request.headers


def test_same_video_id_cannot_be_submitted_twice(runner, upload_dir, video_bytes):
    gate = threading.Event()
    runner.source.gate = gate
    client = make_client(runner, upload_dir)
    vid = upload(client, video_bytes).json()[0]["video_id"]
    other = upload(client, video_bytes, names=("o.avi",)).json()[0]["video_id"]
    try:
        first = client.post("/api/jobs", json=job_body([vid]))
        assert first.status_code == 200
        job_id = first.json()[0]["job_id"]
        # Mixing a used and a fresh id: rejected whole, the fresh id stays usable.
        assert client.post("/api/jobs", json=job_body([other, vid])).status_code == 409
        assert client.post("/api/jobs", json=job_body([vid])).status_code == 409
        assert [j["job_id"] for j in client.get("/api/jobs").json()] == [job_id]
    finally:
        gate.set()
    assert wait_done(client, [job_id]) == ["DONE"]
    second = client.post("/api/jobs", json=job_body([other]))
    assert second.status_code == 200
    assert wait_done(client, [second.json()[0]["job_id"]]) == ["DONE"]
