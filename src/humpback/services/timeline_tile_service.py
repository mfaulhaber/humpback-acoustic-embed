"""Shared timeline tile rendering and caching service."""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from humpback.processing.timeline_renderers import (
    DEFAULT_TIMELINE_RENDERER,
    TimelineTileRenderInput,
    TimelineTileRenderer,
)
from humpback.processing.timeline_repository import (
    TimelineSourceRef,
    TimelineTileRepository,
    TimelineTileRequest,
)
from humpback.processing.timeline_tiles import tile_time_range

logger = logging.getLogger(__name__)

_tile_locks_guard = threading.Lock()
_tile_locks: OrderedDict[str, threading.Lock] = OrderedDict()
_MAX_TILE_LOCKS = 512


@dataclass(frozen=True)
class TimelineTileResult:
    """Rendered or cached tile bytes plus cache hit metadata."""

    data: bytes
    cache_hit: bool


def repository_from_settings(settings) -> TimelineTileRepository:
    """Build the shared tile repository from application settings."""
    return TimelineTileRepository(
        cache_dir=settings.storage_root / "timeline_cache",
        memory_cache_max_items=settings.timeline_tile_memory_cache_items,
    )


def source_ref_from_job(job, settings) -> TimelineSourceRef:
    """Build a shared timeline source identity from a job model."""
    return TimelineSourceRef.from_job(job, settings)


def tile_request_from_settings(
    *,
    zoom_level: str,
    tile_index: int,
    freq_min: int,
    freq_max: int,
    settings,
) -> TimelineTileRequest:
    """Build a tile repository request from endpoint parameters."""
    return TimelineTileRequest(
        zoom_level=zoom_level,
        tile_index=tile_index,
        freq_min=freq_min,
        freq_max=freq_max,
        width_px=settings.timeline_tile_width_px,
        height_px=settings.timeline_tile_height_px,
    )


def get_or_render_tile(
    *,
    job,
    settings,
    zoom_level: str,
    tile_index: int,
    freq_min: int = 0,
    freq_max: int = 3000,
    repository: TimelineTileRepository | None = None,
    renderer: TimelineTileRenderer = DEFAULT_TIMELINE_RENDERER,
) -> TimelineTileResult:
    """Return cached tile bytes or render and store the missing tile."""
    repository = repository or repository_from_settings(settings)
    source_ref = source_ref_from_job(job, settings)
    request = tile_request_from_settings(
        zoom_level=zoom_level,
        tile_index=tile_index,
        freq_min=freq_min,
        freq_max=freq_max,
        settings=settings,
    )

    cached = repository.get(source_ref, renderer.renderer_id, renderer.version, request)
    if cached is not None:
        return TimelineTileResult(data=cached, cache_hit=True)

    tile_path = repository.tile_path(
        source_ref,
        renderer.renderer_id,
        renderer.version,
        request,
    )
    lock = _lock_for_tile(str(tile_path))
    with lock:
        cached = repository.get(
            source_ref, renderer.renderer_id, renderer.version, request
        )
        if cached is not None:
            return TimelineTileResult(data=cached, cache_hit=True)

        tile_bytes = render_tile(
            job=job,
            settings=settings,
            source_ref=source_ref,
            zoom_level=zoom_level,
            tile_index=tile_index,
            freq_min=freq_min,
            freq_max=freq_max,
            renderer=renderer,
            repository=repository,
        )
        repository.put(
            source_ref,
            renderer.renderer_id,
            renderer.version,
            request,
            tile_bytes,
        )
        return TimelineTileResult(data=tile_bytes, cache_hit=False)


def render_tile(
    *,
    job,
    settings,
    source_ref: TimelineSourceRef,
    zoom_level: str,
    tile_index: int,
    freq_min: int,
    freq_max: int,
    renderer: TimelineTileRenderer = DEFAULT_TIMELINE_RENDERER,
    repository: TimelineTileRepository | None = None,
) -> bytes:
    """Render a spectrogram tile without checking or writing the tile cache."""
    from humpback.processing.timeline_audio import resolve_timeline_audio

    job_start = job.start_timestamp or 0.0
    start_epoch, end_epoch = tile_time_range(
        zoom_level,
        tile_index=tile_index,
        job_start_timestamp=job_start,
    )
    duration_sec = end_epoch - start_epoch
    sr = tile_sample_rate(duration_sec, settings.timeline_tile_width_px)

    warmup_sec = max(0.0, min(float(settings.pcen_warmup_sec), start_epoch - job_start))
    fetch_start = start_epoch - warmup_sec
    fetch_duration = duration_sec + warmup_sec

    audio = resolve_timeline_audio(
        hydrophone_id=job.hydrophone_id or "",
        local_cache_path=source_ref.source_identity,
        job_start_timestamp=job.start_timestamp,
        job_end_timestamp=job.end_timestamp,
        start_sec=fetch_start,
        duration_sec=fetch_duration,
        target_sr=sr,
        noaa_cache_path=settings.noaa_cache_path,
        timeline_cache=repository,
        job_id=source_ref.span_key,
        manifest_cache_items=settings.timeline_manifest_memory_cache_items,
        pcm_cache_max_bytes=pcm_cache_bytes_limit(settings),
    )

    warmup_samples = int(round(warmup_sec * sr))
    n_fft = min(2048, len(audio))
    if n_fft < 16:
        n_fft = 16
    hop_length = max(1, n_fft // 8)

    return renderer.render(
        TimelineTileRenderInput(
            audio=np.asarray(audio, dtype=np.float32),
            sample_rate=sr,
            freq_min=freq_min,
            freq_max=freq_max,
            n_fft=n_fft,
            hop_length=hop_length,
            warmup_samples=warmup_samples,
            pcen_params=renderer.pcen_params(settings),
            vmin=settings.pcen_vmin,
            vmax=settings.pcen_vmax,
            width_px=settings.timeline_tile_width_px,
            height_px=settings.timeline_tile_height_px,
        )
    )


def tile_sample_rate(duration_sec: float, width_px: int) -> int:
    """Compute a target sample rate appropriate for tile rendering."""
    desired_samples = width_px * 4 * 256
    if duration_sec <= 0:
        return 32000
    sr = int(desired_samples / duration_sec)
    return max(200, min(sr, 32000))


def pcm_cache_bytes_limit(settings) -> int:
    """Return configured PCM cache size in bytes."""
    return int(settings.timeline_pcm_memory_cache_mb) * 1024 * 1024


def _lock_for_tile(key: str) -> threading.Lock:
    with _tile_locks_guard:
        lock = _tile_locks.pop(key, None)
        if lock is None:
            lock = threading.Lock()
        _tile_locks[key] = lock
        while len(_tile_locks) > _MAX_TILE_LOCKS:
            _tile_locks.popitem(last=False)
        return lock
