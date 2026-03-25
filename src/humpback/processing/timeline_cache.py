"""Directory-structured disk cache for timeline spectrogram tiles.

Tiles are stored as: {cache_dir}/{job_id}/{zoom_level}/tile_{index:04d}.png
Global FIFO eviction (mtime-based) caps total tile count across all jobs.
"""

import os
from pathlib import Path


class TimelineTileCache:
    """Disk-backed tile cache with per-job/zoom directory structure."""

    def __init__(self, cache_dir: Path, max_items: int = 5000) -> None:
        self.cache_dir = cache_dir
        self.max_items = max_items

    def _tile_path(self, job_id: str, zoom_level: str, tile_index: int) -> Path:
        return self.cache_dir / job_id / zoom_level / f"tile_{tile_index:04d}.png"

    def get(self, job_id: str, zoom_level: str, tile_index: int) -> bytes | None:
        p = self._tile_path(job_id, zoom_level, tile_index)
        if p.is_file():
            return p.read_bytes()
        return None

    def put(self, job_id: str, zoom_level: str, tile_index: int, data: bytes) -> None:
        p = self._tile_path(job_id, zoom_level, tile_index)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, p)
        self._evict()

    def _evict(self) -> None:
        """Remove oldest tiles globally when count exceeds max_items."""
        files = sorted(self.cache_dir.rglob("*.png"), key=lambda f: f.stat().st_mtime)
        excess = len(files) - self.max_items
        if excess > 0:
            for f in files[:excess]:
                f.unlink(missing_ok=True)
