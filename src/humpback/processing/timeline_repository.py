"""Shared hydrophone-span repository for timeline tile artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4


@dataclass(frozen=True)
class TimelineSourceRef:
    """Stable identity for a hydrophone timeline span."""

    hydrophone_id: str
    source_identity: str
    job_start_timestamp: float
    job_end_timestamp: float

    @classmethod
    def from_job(cls, job, settings) -> "TimelineSourceRef":
        """Create a source ref from a detection-style or region-style job."""
        local_cache_path = (
            getattr(job, "local_cache_path", None) or settings.s3_cache_path
        )
        source_identity = str(local_cache_path or settings.noaa_cache_path or "")
        return cls(
            hydrophone_id=getattr(job, "hydrophone_id", "") or "",
            source_identity=source_identity,
            job_start_timestamp=float(getattr(job, "start_timestamp", None) or 0.0),
            job_end_timestamp=float(getattr(job, "end_timestamp", None) or 0.0),
        )

    @property
    def span_key(self) -> str:
        """Return a deterministic short digest for this hydrophone span."""
        payload = {
            "hydrophone_id": self.hydrophone_id,
            "source_identity": self.source_identity,
            "job_start_timestamp": round(self.job_start_timestamp, 6),
            "job_end_timestamp": round(self.job_end_timestamp, 6),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()[:20]

    def to_manifest(self) -> dict[str, object]:
        """Serialize source identity metadata for on-disk diagnostics."""
        return asdict(self) | {"span_key": self.span_key}


@dataclass(frozen=True)
class TimelineTileRequest:
    """Cache identity for one timeline tile image."""

    zoom_level: str
    tile_index: int
    freq_min: int
    freq_max: int
    width_px: int
    height_px: int

    @property
    def freq_key(self) -> str:
        return f"f{self.freq_min}-{self.freq_max}"

    @property
    def geometry_key(self) -> str:
        return f"w{self.width_px}_h{self.height_px}"


class TimelineTileRepository:
    """Disk repository keyed by hydrophone span and renderer identity."""

    _memory_lock = threading.Lock()
    _memory_cache: OrderedDict[str, bytes] = OrderedDict()

    def __init__(self, cache_dir: str | Path, memory_cache_max_items: int = 0) -> None:
        self.cache_dir = Path(cache_dir)
        self.memory_cache_max_items = max(0, int(memory_cache_max_items))

    def get(
        self,
        source_ref: TimelineSourceRef,
        renderer_id: str,
        renderer_version: int,
        request: TimelineTileRequest,
    ) -> bytes | None:
        path = self.tile_path(source_ref, renderer_id, renderer_version, request)
        cached = self._get_memory(str(path))
        if cached is not None:
            self._touch_span(source_ref)
            return cached
        if not path.exists():
            return None
        data = path.read_bytes()
        self._put_memory(str(path), data)
        self._touch_span(source_ref)
        return data

    def put(
        self,
        source_ref: TimelineSourceRef,
        renderer_id: str,
        renderer_version: int,
        request: TimelineTileRequest,
        data: bytes,
    ) -> None:
        self._write_source_manifest(source_ref)
        path = self.tile_path(source_ref, renderer_id, renderer_version, request)
        self._write_atomic(path, data)
        self._put_memory(str(path), data)
        self._touch_span(source_ref)

    def has(
        self,
        source_ref: TimelineSourceRef,
        renderer_id: str,
        renderer_version: int,
        request: TimelineTileRequest,
    ) -> bool:
        return self.tile_path(
            source_ref, renderer_id, renderer_version, request
        ).exists()

    def tile_count_for_zoom(
        self,
        source_ref: TimelineSourceRef,
        renderer_id: str,
        renderer_version: int,
        zoom_level: str,
        *,
        freq_min: int,
        freq_max: int,
        width_px: int,
        height_px: int,
    ) -> int:
        zoom_dir = (
            self._renderer_dir(source_ref, renderer_id, renderer_version)
            / zoom_level
            / f"f{freq_min}-{freq_max}"
            / f"w{width_px}_h{height_px}"
        )
        if not zoom_dir.exists():
            return 0
        return sum(1 for f in zoom_dir.iterdir() if f.suffix == ".png")

    def count_cached_tiles(
        self,
        source_ref: TimelineSourceRef,
        renderer_id: str,
        renderer_version: int,
        requests: list[TimelineTileRequest],
    ) -> int:
        return sum(
            1
            for request in requests
            if self.has(source_ref, renderer_id, renderer_version, request)
        )

    def get_audio_manifest(self, source_ref: TimelineSourceRef | str) -> dict | None:
        path = self._span_dir(source_ref) / ".audio_manifest.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None

    def put_audio_manifest(
        self, source_ref: TimelineSourceRef | str, manifest: dict
    ) -> None:
        self._write_source_manifest(source_ref)
        self._write_atomic(
            self._span_dir(source_ref) / ".audio_manifest.json", json.dumps(manifest)
        )
        self._touch_span(source_ref)

    def get_prepare_plan(self, source_ref: TimelineSourceRef | str) -> dict | None:
        path = self._span_dir(source_ref) / ".prepare_plan.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None

    def put_prepare_plan(self, source_ref: TimelineSourceRef | str, plan: dict) -> None:
        self._write_source_manifest(source_ref)
        self._write_atomic(
            self._span_dir(source_ref) / ".prepare_plan.json", json.dumps(plan)
        )
        self._touch_span(source_ref)

    def tile_path(
        self,
        source_ref: TimelineSourceRef,
        renderer_id: str,
        renderer_version: int,
        request: TimelineTileRequest,
    ) -> Path:
        return (
            self._renderer_dir(source_ref, renderer_id, renderer_version)
            / request.zoom_level
            / request.freq_key
            / request.geometry_key
            / f"tile_{request.tile_index:04d}.png"
        )

    def _span_dir(self, source_ref: TimelineSourceRef | str) -> Path:
        span_key = (
            source_ref.span_key
            if isinstance(source_ref, TimelineSourceRef)
            else source_ref
        )
        return self.cache_dir / "spans" / span_key

    def _renderer_dir(
        self,
        source_ref: TimelineSourceRef,
        renderer_id: str,
        renderer_version: int,
    ) -> Path:
        return self._span_dir(source_ref) / renderer_id / f"v{renderer_version}"

    def _write_source_manifest(self, source_ref: TimelineSourceRef | str) -> None:
        if isinstance(source_ref, str):
            return
        self._write_atomic(
            self._span_dir(source_ref) / ".source.json",
            json.dumps(source_ref.to_manifest(), sort_keys=True),
        )

    def _touch_span(self, source_ref: TimelineSourceRef | str) -> None:
        span_dir = self._span_dir(source_ref)
        span_dir.mkdir(parents=True, exist_ok=True)
        (span_dir / ".last_access").touch()

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

    def _get_memory(self, key: str) -> bytes | None:
        if self.memory_cache_max_items <= 0:
            return None
        with self._memory_lock:
            cached = self._memory_cache.pop(key, None)
            if cached is not None:
                self._memory_cache[key] = cached
            return cached

    def _put_memory(self, key: str, data: bytes) -> None:
        if self.memory_cache_max_items <= 0:
            return
        with self._memory_lock:
            self._memory_cache.pop(key, None)
            self._memory_cache[key] = data
            while len(self._memory_cache) > self.memory_cache_max_items:
                self._memory_cache.popitem(last=False)
