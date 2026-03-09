"""FIFO disk cache for spectrogram PNG images."""

import hashlib
import json
import os
from pathlib import Path


class SpectrogramCache:
    """Simple disk-backed FIFO cache keyed by spectrogram parameters."""

    def __init__(self, cache_dir: Path, max_items: int = 1000) -> None:
        self.cache_dir = cache_dir
        self.max_items = max_items
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _make_key(
        job_id: str,
        filename: str,
        start_sec: float,
        duration_sec: float,
        hop_length: int,
        dynamic_range_db: float,
        n_fft: int,
        width_px: int,
        height_px: int,
    ) -> str:
        blob = json.dumps(
            [job_id, filename, start_sec, duration_sec, hop_length, dynamic_range_db, n_fft, width_px, height_px],
            sort_keys=False,
        ).encode()
        return hashlib.sha256(blob).hexdigest()

    def _path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.png"

    def get(self, key: str) -> bytes | None:
        p = self._path(key)
        if p.is_file():
            return p.read_bytes()
        return None

    def put(self, key: str, data: bytes) -> None:
        p = self._path(key)
        tmp = p.with_suffix(".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, p)
        self._evict()

    def _evict(self) -> None:
        """Remove oldest files when count exceeds max_items."""
        files = sorted(self.cache_dir.glob("*.png"), key=lambda f: f.stat().st_mtime)
        excess = len(files) - self.max_items
        if excess > 0:
            for f in files[:excess]:
                f.unlink(missing_ok=True)
