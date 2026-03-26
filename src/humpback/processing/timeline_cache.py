"""Disk-backed tile cache with per-job LRU eviction."""

from __future__ import annotations

import errno
import logging
import os
import shutil
from pathlib import Path
from typing import BinaryIO
from uuid import uuid4

logger = logging.getLogger(__name__)


class TimelinePrepareLock:
    """Advisory file lock that coordinates prepare work across processes."""

    def __init__(self, handle: BinaryIO) -> None:
        self._handle = handle

    def release(self) -> None:
        if self._handle.closed:
            return

        import fcntl

        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        finally:
            self._handle.close()


class TimelineTileCache:
    """Stores timeline spectrogram tiles on disk, evicting whole jobs LRU."""

    def __init__(self, cache_dir: str | Path, max_jobs: int = 15) -> None:
        self.cache_dir = Path(cache_dir)
        self.max_jobs = max_jobs

    # -- public API --

    def get(self, job_id: str, zoom_level: str, tile_index: int) -> bytes | None:
        path = self._tile_path(job_id, zoom_level, tile_index)
        if not path.exists():
            return None
        self._touch_job(job_id)
        return path.read_bytes()

    def put(self, job_id: str, zoom_level: str, tile_index: int, data: bytes) -> None:
        self._touch_job(job_id)
        path = self._tile_path(job_id, zoom_level, tile_index)
        self._write_atomic(path, data)
        self._touch_job(job_id)
        self._evict_lru_jobs()

    def job_count(self) -> int:
        if not self.cache_dir.exists():
            return 0
        return sum(
            1
            for p in self.cache_dir.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        )

    def has(self, job_id: str, zoom_level: str, tile_index: int) -> bool:
        """Check if a tile exists without touching the job sentinel."""
        return self._tile_path(job_id, zoom_level, tile_index).exists()

    def tile_count_for_zoom(self, job_id: str, zoom_level: str) -> int:
        zoom_dir = self.cache_dir / job_id / zoom_level
        if not zoom_dir.exists():
            return 0
        return sum(1 for f in zoom_dir.iterdir() if f.suffix == ".png")

    def get_ref_db(self, job_id: str) -> float | None:
        """Return the cached per-job reference dB level, or None if not yet computed."""
        import json

        path = self.cache_dir / job_id / ".ref_db.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            return float(data["ref_db"])
        except (json.JSONDecodeError, KeyError, ValueError):
            return None

    def put_ref_db(self, job_id: str, ref_db: float) -> None:
        """Store the per-job reference dB level."""
        import json

        self._touch_job(job_id)
        path = self.cache_dir / job_id / ".ref_db.json"
        self._write_atomic(path, json.dumps({"ref_db": ref_db}))

    def try_acquire_prepare_lock(self, job_id: str) -> TimelinePrepareLock | None:
        """Try to claim exclusive prepare ownership for a job."""
        import fcntl

        lock_path = self.cache_dir / job_id / ".prepare.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        handle = lock_path.open("a+b")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            if exc.errno in (errno.EACCES, errno.EAGAIN):
                return None
            raise
        return TimelinePrepareLock(handle)

    # -- internals --

    def _tile_path(self, job_id: str, zoom_level: str, tile_index: int) -> Path:
        return self.cache_dir / job_id / zoom_level / f"tile_{tile_index:04d}.png"

    def _touch_job(self, job_id: str) -> None:
        job_dir = self.cache_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        sentinel = job_dir / ".last_access"
        sentinel.touch()

    def _write_atomic(self, path: Path, data: bytes | str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            if isinstance(data, bytes):
                tmp.write_bytes(data)
            else:
                tmp.write_text(data)
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)

    def _evict_lru_jobs(self) -> None:
        if not self.cache_dir.exists():
            return
        job_dirs = [
            p
            for p in self.cache_dir.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        ]
        if len(job_dirs) <= self.max_jobs:
            return

        # Sort by sentinel mtime (oldest first)
        def _access_time(d: Path) -> float:
            sentinel = d / ".last_access"
            if sentinel.exists():
                return sentinel.stat().st_mtime
            return 0.0

        job_dirs.sort(key=_access_time)
        to_remove = len(job_dirs) - self.max_jobs
        for job_dir in job_dirs[:to_remove]:
            logger.info("Evicting tile cache for job %s", job_dir.name)
            shutil.rmtree(job_dir, ignore_errors=True)
