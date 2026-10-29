import os
import time
from pathlib import Path

from visionflow import paths
from visionflow.paths import data_dir, output_dir, sweep_uploads, upload_dir

REPO_ROOT = Path(__file__).resolve().parents[1]


def _clear_env(monkeypatch):
    for name in ("VISIONFLOW_DATA_DIR", "LOCALAPPDATA", "XDG_DATA_HOME"):
        monkeypatch.delenv(name, raising=False)


def test_env_override_wins_everywhere(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("VISIONFLOW_DATA_DIR", str(tmp_path / "custom"))
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    for platform in ("win32", "linux"):
        monkeypatch.setattr(paths.sys, "platform", platform)
        assert data_dir() == tmp_path / "custom"
    assert upload_dir() == tmp_path / "custom" / "uploads"
    assert output_dir() == tmp_path / "custom" / "outputs"


def test_windows_uses_localappdata(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setattr(paths.sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "AppData" / "Local"))
    assert data_dir() == tmp_path / "AppData" / "Local" / "visionflow"


def test_windows_without_localappdata_falls_back_to_home(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setattr(paths.sys, "platform", "win32")
    monkeypatch.setattr(paths.Path, "home", classmethod(lambda cls: tmp_path))
    assert data_dir() == tmp_path / "AppData" / "Local" / "visionflow"


def test_other_platforms_use_xdg_data_home(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setattr(paths.sys, "platform", "linux")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    assert data_dir() == tmp_path / "xdg" / "visionflow"


def test_other_platforms_default_to_local_share(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setattr(paths.sys, "platform", "darwin")
    monkeypatch.setattr(paths.Path, "home", classmethod(lambda cls: tmp_path))
    assert data_dir() == tmp_path / ".local" / "share" / "visionflow"


def test_empty_env_values_are_ignored(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("VISIONFLOW_DATA_DIR", "  ")
    monkeypatch.setattr(paths.sys, "platform", "linux")
    monkeypatch.setenv("XDG_DATA_HOME", "")
    monkeypatch.setattr(paths.Path, "home", classmethod(lambda cls: tmp_path))
    assert data_dir() == tmp_path / ".local" / "share" / "visionflow"


def test_real_default_is_outside_the_repository(monkeypatch):
    monkeypatch.delenv("VISIONFLOW_DATA_DIR", raising=False)
    resolved = data_dir().resolve()
    assert REPO_ROOT.resolve() not in (resolved, *resolved.parents)


def _age(path: Path, seconds: float) -> None:
    old = time.time() - seconds
    os.utime(path, (old, old))


def test_sweep_uploads_removes_old_and_empty_folders_only(tmp_path):
    root = tmp_path / "uploads"
    old, fresh, empty, other = "a" * 32, "b" * 32, "c" * 32, "not-an-id"
    for name in (old, fresh, empty, other):
        (root / name).mkdir(parents=True)
    for name in (old, fresh, other):
        (root / name / "v.mp4").write_bytes(b"x")
    _age(root / old / "v.mp4", 25 * 3600)
    _age(root / old, 25 * 3600)
    _age(root / other, 25 * 3600)
    removed = sweep_uploads(root)
    assert sorted(removed) == sorted([old, empty])
    assert sorted(p.name for p in root.iterdir()) == sorted([fresh, other])


def test_sweep_uploads_keeps_folder_with_a_recent_file(tmp_path):
    root = tmp_path / "uploads"
    folder = root / ("d" * 32)
    folder.mkdir(parents=True)
    (folder / "v.mp4").write_bytes(b"x")
    _age(folder, 25 * 3600)  # folder mtime is old but the file is new
    assert sweep_uploads(root) == []
    assert folder.is_dir()


def test_sweep_uploads_on_missing_root_is_a_no_op(tmp_path):
    assert sweep_uploads(tmp_path / "missing") == []
