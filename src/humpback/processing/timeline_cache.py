"""Disk-backed tile cache with per-job LRU eviction."""

from __future__ import annotations

import errno
import json
import logging
import os
import shutil
import threading
from collections import OrderedDict
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

    _memory_lock = threading.Lock()
    _memory_cache: OrderedDict[str, bytes] = OrderedDict()

    def __init__(
        self,
        cache_dir: str | Path,
        max_jobs: int = 15,
        memory_cache_max_items: int = 0,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.max_jobs = max_jobs
        self.memory_cache_max_items = max(0, int(memory_cache_max_items))

    # -- public API --

    def get(self, job_id: str, zoom_level: str, tile_index: int) -> bytes | None:
        path = self._tile_path(job_id, zoom_level, tile_index)
        memory_key = self._memory_key(path)
        cached = self._get_memory(memory_key)
        if cached is not None:
            self._touch_job(job_id)
            return cached
        if not path.exists():
            return None
        self._touch_job(job_id)
        data = path.read_bytes()
        self._put_memory(memory_key, data)
        return data

    def put(self, job_id: str, zoom_level: str, tile_index: int, data: bytes) -> None:
        self._touch_job(job_id)
        path = self._tile_path(job_id, zoom_level, tile_index)
        self._write_atomic(path, data)
        self._put_memory(self._memory_key(path), data)
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
        self._touch_job(job_id)
        path = self.cache_dir / job_id / ".ref_db.json"
        self._write_atomic(path, json.dumps({"ref_db": ref_db}))

    def get_prepare_plan(self, job_id: str) -> dict | None:
        """Return the persisted prepare plan for a job, if present."""
        path = self.cache_dir / job_id / ".prepare_plan.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict):
            return None
        return data

    def put_prepare_plan(self, job_id: str, plan: dict) -> None:
        """Persist the active prepare plan for prepare-status reporting."""
        self._touch_job(job_id)
        path = self.cache_dir / job_id / ".prepare_plan.json"
        self._write_atomic(path, json.dumps(plan))

    def get_audio_manifest(self, job_id: str) -> dict | None:
        """Return a persisted audio manifest for a job, if present."""
        path = self.cache_dir / job_id / ".audio_manifest.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict):
            return None
        return data

    def put_audio_manifest(self, job_id: str, manifest: dict) -> None:
        """Persist a reusable audio manifest for a job."""
        self._touch_job(job_id)
        path = self.cache_dir / job_id / ".audio_manifest.json"
        self._write_atomic(path, json.dumps(manifest))

    def count_cached_tiles(
        self, job_id: str, zoom_level: str, tile_indices: list[int]
    ) -> int:
        """Count how many requested tile indices already exist on disk."""
        return sum(1 for idx in tile_indices if self.has(job_id, zoom_level, idx))

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

    def _memory_key(self, path: Path) -> str:
        return str(path)

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
            self._drop_memory_for_job(job_dir.name)
            shutil.rmtree(job_dir, ignore_errors=True)

    def _get_memory(self, key: str) -> bytes | None:
        if self.memory_cache_max_items <= 0:
            return None
        with self._memory_lock:
            data = self._memory_cache.pop(key, None)
            if data is None:
                return None
            self._memory_cache[key] = data
            return data

    def _put_memory(self, key: str, data: bytes) -> None:
        if self.memory_cache_max_items <= 0:
            return
        with self._memory_lock:
            self._memory_cache.pop(key, None)
            self._memory_cache[key] = data
            while len(self._memory_cache) > self.memory_cache_max_items:
                self._memory_cache.popitem(last=False)

    @classmethod
    def _drop_memory_for_job(cls, job_id: str) -> None:
        job_fragment = f"{os.sep}{job_id}{os.sep}"
        with cls._memory_lock:
            for key in [key for key in cls._memory_cache if job_fragment in key]:
                cls._memory_cache.pop(key, None)
