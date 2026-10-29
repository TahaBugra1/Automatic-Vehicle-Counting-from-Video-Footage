import json
import threading
import time
import uuid
from pathlib import Path

import pytest
from fakes import (
    LINE,
    OPTIONS,
    TIMEOUT_S,
    FakeSource,
    make_bad_video,
    make_video,
    wait_finished,
)

from visionflow import jobs
from visionflow.jobs import JobRunner, JobStatus, safe_stem


@pytest.fixture
def runners():
    created: list[JobRunner] = []

    def make(root: Path, source) -> JobRunner:
        runner = JobRunner(root, source)
        created.append(runner)
        return runner

    yield make
    for runner in created:
        runner.shutdown()


def test_jobs_run_sequentially_in_submit_order(tmp_path, runners):
    gate = threading.Event()
    source = FakeSource(gate)
    runner = runners(tmp_path / "out", source)
    ids = [
        runner.submit(make_video(tmp_path / f"in{i}.avi"), f"video {i}.mp4", LINE, OPTIONS)
        for i in range(3)
    ]
    assert source.started.wait(TIMEOUT_S)
    assert runner.get(ids[0]).status == JobStatus.RUNNING
    assert runner.get(ids[1]).status == JobStatus.QUEUED
    assert runner.get(ids[2]).status == JobStatus.QUEUED
    with pytest.raises(ValueError):
        runner.delete(ids[0])
    gate.set()

    jobs = wait_finished(runner, ids)
    assert [j.status for j in jobs] == [JobStatus.DONE] * 3
    assert all(j.progress == 1.0 and j.error is None for j in jobs)
    assert source.calls == ["in0.avi", "in1.avi", "in2.avi"]
    assert source.max_active == 1
    assert [j.job_id for j in runner.list_jobs()] == ids
    assert [j.video_name for j in jobs] == ["video_0", "video_1", "video_2"]

    for job_id in ids:
        folder = tmp_path / "out" / job_id
        assert (folder / "annotated.mp4").stat().st_size > 0
        assert (folder / "crossings.csv").exists()
        result = runner.result(job_id)
        assert result["job_id"] == job_id
        assert result["totals"] == {"Yön 1": 1, "Yön 2": 0}
        assert result["total_count"] == 1
        assert result["line"] == [0.0, 0.5, 1.0, 0.5]
        assert result["model_name"] == OPTIONS.model_name
        assert result["total_text"]["tr"].startswith("1 araç")


def test_bad_video_marks_error_and_next_job_still_completes(tmp_path, runners):
    runner = runners(tmp_path / "out", FakeSource())
    bad = runner.submit(make_bad_video(tmp_path / "bad.mp4"), "bad.mp4", LINE, OPTIONS)
    good = runner.submit(make_video(tmp_path / "good.avi"), "good.avi", LINE, OPTIONS)
    bad_job, good_job = wait_finished(runner, [bad, good])
    assert bad_job.status == JobStatus.ERROR
    assert bad_job.error and len(bad_job.error) <= 200
    assert good_job.status == JobStatus.DONE
    assert runner.result(good)["total_count"] == 1
    assert not (tmp_path / "out" / bad / "result.json").exists()


def test_source_video_deleted_after_processing(tmp_path, runners):
    runner = runners(tmp_path / "out", FakeSource())
    good_path = make_video(tmp_path / "good.avi")
    bad_path = make_bad_video(tmp_path / "bad.mp4")
    ids = [
        runner.submit(good_path, "good.avi", LINE, OPTIONS),
        runner.submit(bad_path, "bad.mp4", LINE, OPTIONS),
    ]
    wait_finished(runner, ids)
    assert not good_path.exists()
    assert not bad_path.exists()


def test_result_survives_new_runner_instance(tmp_path, runners):
    root = tmp_path / "out"
    first = runners(root, FakeSource())
    job_id = first.submit(make_video(tmp_path / "in.avi"), "Kavşak 1.mp4", LINE, OPTIONS)
    wait_finished(first, [job_id])
    expected = first.result(job_id)
    first.shutdown()
    # A job that was RUNNING when the old process died.
    stale = "0" * 32
    (root / stale).mkdir()
    stale_record = {
        "job_id": stale,
        "video_name": "x",
        "line": [0, 0.5, 1, 0.5],
        "status": "RUNNING",
        "error": None,
        "created": time.time() + 1,
    }
    (root / stale / "job.json").write_text(json.dumps(stale_record), encoding="utf-8")

    second = runners(root, FakeSource())
    stale_job = second.get(stale)
    assert (stale_job.status, stale_job.error) == (JobStatus.ERROR, "interrupted")
    with pytest.raises(ValueError):
        second.result(stale)
    second.delete(stale)
    assert second.result(job_id) == expected
    assert expected["video_name"] == "Kavsak_1"
    raw = json.loads((root / job_id / "result.json").read_text(encoding="utf-8"))
    assert raw == expected
    jobs = second.list_jobs()
    assert [(j.job_id, j.status, j.video_name) for j in jobs] == [
        (job_id, JobStatus.DONE, "Kavsak_1")
    ]
    assert jobs[0].line == LINE


def test_delete_removes_result_folder(tmp_path, runners):
    runner = runners(tmp_path / "out", FakeSource())
    job_id = runner.submit(make_video(tmp_path / "in.avi"), "in.avi", LINE, OPTIONS)
    wait_finished(runner, [job_id])
    runner.delete(job_id)
    assert not (tmp_path / "out" / job_id).exists()
    assert runner.list_jobs() == []
    with pytest.raises(KeyError):
        runner.get(job_id)
    with pytest.raises(KeyError):
        runner.delete(job_id)
    with pytest.raises(KeyError):
        runner.delete("../" + job_id)


def test_safe_stem_blocks_path_traversal(tmp_path):
    for name in ["../../etc/x y.mp4", "..\\..\\windows\\x y.mp4", "..", "/etc/passwd"]:
        stem = safe_stem(name)
        assert "/" not in stem and "\\" not in stem and ".." not in stem
        assert all(c.isascii() and (c.isalnum() or c in "_-") for c in stem)
        target = (tmp_path / stem).resolve()
        assert target.parent == tmp_path.resolve()
    assert safe_stem("../../etc/x y.mp4") == "x_y"


def test_safe_stem_handles_turkish_and_empty_names():
    assert safe_stem("Çağrı Şehir İçi ığdır öğün.mp4") == "Cagri_Sehir_Ici_igdir_ogun"
    assert safe_stem("") == "video"
    assert safe_stem("   ") == "video"
    assert safe_stem("...") == "video"
    assert safe_stem("日本.mp4") == "video"
    assert len(safe_stem("a" * 300 + ".mp4")) == 80


def test_worker_survives_persist_failure_and_runs_next_job(tmp_path, runners, monkeypatch):
    real_write = jobs._write_json
    failed: list[Path] = []

    def flaky_write(path: Path, data: dict) -> None:
        if path.name == "job.json" and data["status"] == "RUNNING" and not failed:
            failed.append(path)
            raise PermissionError("job.json is locked by another process")
        real_write(path, data)

    monkeypatch.setattr(jobs, "_write_json", flaky_write)
    runner = runners(tmp_path / "out", FakeSource())
    first_path = make_video(tmp_path / "first.avi")
    first = runner.submit(first_path, "first.avi", LINE, OPTIONS)
    second = runner.submit(make_video(tmp_path / "second.avi"), "second.avi", LINE, OPTIONS)
    first_job, second_job = wait_finished(runner, [first, second])
    assert failed == [tmp_path / "out" / first / "job.json"]
    assert first_job.status == JobStatus.ERROR
    assert "PermissionError" in first_job.error
    assert not first_path.exists()
    assert second_job.status == JobStatus.DONE
    assert runner.result(second)["total_count"] == 1
    runner.delete(first)  # no longer stuck as RUNNING


def test_summary_csv_written_next_to_crossings(tmp_path, runners):
    runner = runners(tmp_path / "out", FakeSource())
    job_id = runner.submit(make_video(tmp_path / "in.avi"), "in.avi", LINE, OPTIONS)
    wait_finished(runner, [job_id])
    text = (tmp_path / "out" / job_id / "summary.csv").read_text(encoding="utf-8-sig")
    assert text.splitlines()[0] == "metrik,deger"
    assert "toplam,1" in text.splitlines()
    assert f"model,{OPTIONS.model_name}" in text.splitlines()


def _upload(tmp_path: Path, name: str = "in.avi") -> Path:
    folder = tmp_path / "uploads" / uuid.uuid4().hex
    folder.mkdir(parents=True)
    return make_video(folder / name)


def test_empty_upload_folder_removed_after_processing(tmp_path, runners):
    runner = runners(tmp_path / "out", FakeSource())
    source = _upload(tmp_path)
    job_id = runner.submit(source, "in.avi", LINE, OPTIONS)
    wait_finished(runner, [job_id])
    assert not source.parent.exists()
    assert (tmp_path / "uploads").is_dir()  # only the per-video folder goes


def test_interrupted_job_source_removed_on_reload(tmp_path, runners):
    root = tmp_path / "out"
    gate = threading.Event()
    first = runners(root, FakeSource(gate))
    running_src, queued_src = _upload(tmp_path, "a.avi"), _upload(tmp_path, "b.avi")
    running = first.submit(running_src, "a.avi", LINE, OPTIONS)
    queued = first.submit(queued_src, "b.avi", LINE, OPTIONS)
    assert first.source.started.wait(TIMEOUT_S)
    record = json.loads((root / queued / "job.json").read_text(encoding="utf-8"))
    assert record["source"] == str(queued_src)
    # Simulate a crash: the old process never cleaned up (files and job.json stay as they are).
    first._stop.set()
    second = runners(root, FakeSource())
    for job_id in (running, queued):
        assert (second.get(job_id).status, second.get(job_id).error) == (
            JobStatus.ERROR,
            "interrupted",
        )
    assert not queued_src.exists() and not queued_src.parent.exists()
    reloaded = json.loads((root / queued / "job.json").read_text(encoding="utf-8"))
    assert "source" not in reloaded
    gate.set()


def test_sweep_temp_files_skips_running_jobs(tmp_path, runners):
    gate = threading.Event()
    done_root = tmp_path / "out"
    stale_id = "e" * 32
    (done_root / stale_id).mkdir(parents=True)
    (done_root / stale_id / "job.json").write_text(
        json.dumps({"job_id": stale_id, "video_name": "x", "line": [0, 0.5, 1, 0.5],
                    "status": "DONE", "error": None, "created": 1.0}),
        encoding="utf-8",
    )  # fmt: skip
    runner = runners(done_root, FakeSource(gate))
    leftovers = [done_root / stale_id / n for n in ("annotated.raw.tmp.mp4", "job.json.tmp")]
    for path in leftovers:
        path.write_bytes(b"x")
    running = runner.submit(make_video(tmp_path / "in.avi"), "in.avi", LINE, OPTIONS)
    assert runner.source.started.wait(TIMEOUT_S)
    busy = done_root / running / "annotated.raw.tmp.mp4"
    busy.write_bytes(b"x")
    try:
        removed = runner.sweep_temp_files()
        assert sorted(removed) == sorted(leftovers)
        assert busy.exists()
        assert (done_root / stale_id / "job.json").exists()
    finally:
        gate.set()
    wait_finished(runner, [running])


def test_delete_retries_then_keeps_job_when_file_stays_locked(tmp_path, runners, monkeypatch):
    runner = runners(tmp_path / "out", FakeSource())
    job_id = runner.submit(make_video(tmp_path / "in.avi"), "in.avi", LINE, OPTIONS)
    wait_finished(runner, [job_id])
    folder = tmp_path / "out" / job_id
    real_delete = jobs._delete_path
    attempts: list[str] = []

    def locked(path: Path) -> None:
        if path.name == "annotated.mp4":
            attempts.append(path.name)
            raise PermissionError("in use by the browser")
        real_delete(path)

    monkeypatch.setattr(jobs, "DELETE_DELAY_S", 0.0)
    monkeypatch.setattr(jobs, "_delete_path", locked)
    with pytest.raises(OSError):
        runner.delete(job_id)
    assert len(attempts) == jobs.DELETE_ATTEMPTS
    assert runner.get(job_id).status == JobStatus.DONE  # still listed
    assert (folder / "job.json").exists() and (folder / "annotated.mp4").exists()

    monkeypatch.setattr(jobs, "_delete_path", real_delete)
    runner.delete(job_id)  # deletable later
    assert not folder.exists()
    assert runner.list_jobs() == []


def test_delete_succeeds_after_a_transient_lock(tmp_path, runners, monkeypatch):
    runner = runners(tmp_path / "out", FakeSource())
    job_id = runner.submit(make_video(tmp_path / "in.avi"), "in.avi", LINE, OPTIONS)
    wait_finished(runner, [job_id])
    real_delete = jobs._delete_path
    failures = [PermissionError("locked"), PermissionError("locked")]

    def flaky(path: Path) -> None:
        if path.name == "annotated.mp4" and failures:
            raise failures.pop()
        real_delete(path)

    monkeypatch.setattr(jobs, "DELETE_DELAY_S", 0.0)
    monkeypatch.setattr(jobs, "_delete_path", flaky)
    runner.delete(job_id)
    assert not (tmp_path / "out" / job_id).exists()


class PhaseProbeSource(FakeSource):
    """Records the job's phase before the first frame and while frames are processed."""

    def __init__(self):
        super().__init__()
        self.runner: JobRunner | None = None
        self.seen: list[str | None] = []

    def frames(self, video_path, model_name, confidence):
        (job,) = self.runner.list_jobs()
        self.seen.append(job.phase)  # before the first frame: model loading
        for i, frame in enumerate(super().frames(video_path, model_name, confidence)):
            yield frame
            if i == 0:
                (job,) = self.runner.list_jobs()
                self.seen.append(job.phase)


def test_phase_transitions(tmp_path, runners, monkeypatch):
    source = PhaseProbeSource()
    runner = runners(tmp_path / "out", source)
    source.runner = runner
    real_exit = jobs.AnnotatedWriter.__exit__

    def probing_exit(self, *exc):
        (job,) = runner.list_jobs()
        source.seen.append(job.phase)  # writer still open: encoding the output video
        return real_exit(self, *exc)

    monkeypatch.setattr(jobs.AnnotatedWriter, "__exit__", probing_exit)
    job_id = runner.submit(make_video(tmp_path / "in.avi"), "in.avi", LINE, OPTIONS)
    assert runner.get(job_id).phase is None  # queued: no phase yet
    (job,) = wait_finished(runner, [job_id])
    assert job.status == JobStatus.DONE
    assert source.seen == ["loading", "processing", "encoding"]
    assert job.phase is None


def test_runner_starts_when_persisted_job_json_is_unwritable(tmp_path, runners, monkeypatch):
    root = tmp_path / "out"
    first = runners(root, FakeSource())
    done_id = first.submit(make_video(tmp_path / "in.avi"), "in.avi", LINE, OPTIONS)
    wait_finished(first, [done_id])
    first.shutdown()
    done_file = root / done_id / "job.json"
    done_before = (done_file.read_bytes(), done_file.stat().st_mtime_ns)

    stale = "f" * 32
    (root / stale).mkdir()
    stale_record = {
        "job_id": stale,
        "video_name": "x",
        "line": [0, 0.5, 1, 0.5],
        "status": "QUEUED",
        "error": None,
        "created": time.time() + 1,
    }
    (root / stale / "job.json").write_text(json.dumps(stale_record), encoding="utf-8")

    attempts: list[Path] = []

    def locked_write(path: Path, data: dict) -> None:
        attempts.append(path)
        raise PermissionError("locked")

    monkeypatch.setattr(jobs, "_write_json", locked_write)
    second = runners(root, FakeSource())
    assert attempts == [root / stale / "job.json"]
    listed = {j.job_id: (j.status, j.error) for j in second.list_jobs()}
    assert listed == {done_id: (JobStatus.DONE, None), stale: (JobStatus.ERROR, "interrupted")}
    assert (done_file.read_bytes(), done_file.stat().st_mtime_ns) == done_before
