"""Disk-backed tile cache with per-job LRU eviction."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


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
        path = self._tile_path(job_id, zoom_level, tile_index)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)
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

    def tile_count_for_zoom(self, job_id: str, zoom_level: str) -> int:
        zoom_dir = self.cache_dir / job_id / zoom_level
        if not zoom_dir.exists():
            return 0
        return sum(1 for f in zoom_dir.iterdir() if f.suffix == ".png")

    # -- internals --

    def _tile_path(self, job_id: str, zoom_level: str, tile_index: int) -> Path:
        return self.cache_dir / job_id / zoom_level / f"tile_{tile_index:04d}.png"

    def _touch_job(self, job_id: str) -> None:
        job_dir = self.cache_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        sentinel = job_dir / ".last_access"
        sentinel.touch()

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
