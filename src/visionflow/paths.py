"""Where visionflow keeps its data (uploads and results) and how stale uploads are cleaned up.

The data folder is deliberately outside the repository, so uploaded videos and results are not
picked up by a synced folder (OneDrive, Dropbox, ...) the repository may live in:

1. `VISIONFLOW_DATA_DIR` if set (and not blank);
2. on Windows `%LOCALAPPDATA%\\visionflow` (local, not roamed or synced by default);
3. elsewhere `$XDG_DATA_HOME/visionflow`, or `~/.local/share/visionflow`.

`uploads/` and `outputs/` are subfolders of it.
"""

import os
import re
import shutil
import sys
import time
from pathlib import Path

APP_NAME = "visionflow"
ENV_DATA_DIR = "VISIONFLOW_DATA_DIR"
UPLOAD_MAX_AGE_S = 24 * 3600

_HEX_ID = re.compile(r"[0-9a-f]{32}")


def _env(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or None


def data_dir() -> Path:
    override = _env(ENV_DATA_DIR)
    if override is not None:
        return Path(override)
    if sys.platform == "win32":
        local = _env("LOCALAPPDATA")
        base = Path(local) if local is not None else Path.home() / "AppData" / "Local"
        return base / APP_NAME
    xdg = _env("XDG_DATA_HOME")
    base = Path(xdg) if xdg is not None else Path.home() / ".local" / "share"
    return base / APP_NAME


def upload_dir() -> Path:
    return data_dir() / "uploads"


def output_dir() -> Path:
    return data_dir() / "outputs"


def _newest_mtime(folder: Path) -> float:
    newest = folder.stat().st_mtime
    for path in folder.rglob("*"):
        try:
            newest = max(newest, path.stat().st_mtime)
        except OSError:
            continue
    return newest


def sweep_uploads(root: Path, max_age_s: float = UPLOAD_MAX_AGE_S) -> list[str]:
    """Delete `root/<video_id>/` folders that are empty or older than `max_age_s`.

    Age is the newest mtime of the folder and its files. Only folders named like a video id
    are touched. Best effort: a folder that cannot be removed (e.g. locked) is skipped.
    Returns the ids of the removed folders.
    """
    root = Path(root)
    if not root.is_dir():
        return []
    cutoff = time.time() - max_age_s
    removed: list[str] = []
    for folder in root.iterdir():
        if not (folder.is_dir() and _HEX_ID.fullmatch(folder.name)):
            continue
        try:
            empty = not any(folder.iterdir())
            if empty or _newest_mtime(folder) < cutoff:
                shutil.rmtree(folder)
                removed.append(folder.name)
        except OSError:
            continue
    return removed


def remove_empty_upload_folder(path: Path) -> None:
    """After a source video was deleted: drop its `<video_id>/` folder if it is now empty."""
    folder = Path(path).parent
    if not _HEX_ID.fullmatch(folder.name):
        return
    try:
        folder.rmdir()  # only succeeds for an empty folder
    except OSError:
        pass
