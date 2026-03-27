"""API sub-router for timeline spectrogram tiles, confidence, and audio."""

from __future__ import annotations

import asyncio
import io
import logging
import threading
import wave
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import numpy as np
import pyarrow.parquet as pq
from fastapi import APIRouter, Body, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field

from humpback.api.deps import SessionDep, SettingsDep
from humpback.processing.timeline_cache import TimelinePrepareLock, TimelineTileCache
from humpback.processing.timeline_tiles import (
    ZOOM_LEVELS,
    generate_timeline_tile,
    tile_count,
    tile_duration_sec,
    tile_time_range,
)
from humpback.services import classifier_service
from humpback.storage import detection_diagnostics_path

logger = logging.getLogger(__name__)

router = APIRouter()

# Module-level set to track in-progress prepare jobs (prevents duplicate work)
_preparing: set[str] = set()
_preparing_lock = threading.Lock()


# ---- Response models ----


class ConfidenceResponse(BaseModel):
    window_sec: float
    scores: list[float | None]
    start_timestamp: float
    end_timestamp: float


class PrepareResponse(BaseModel):
    status: str
    timeline_tiles_ready: bool


class PrepareRequest(BaseModel):
    scope: Literal["startup", "full"] = "startup"
    zoom_level: str | None = None
    center_timestamp: float | None = None
    radius_tiles: int | None = Field(default=None, ge=0)


# ---- Helpers ----


PrepareTargets = dict[str, list[int] | None]
_ALL_TILES = "all"


async def _get_job_or_404(session, job_id: str):
    job = await classifier_service.get_detection_job(session, job_id)
    if job is None:
        raise HTTPException(404, "Detection job not found")
    return job


def _job_duration(job) -> float:
    if job.start_timestamp is not None and job.end_timestamp is not None:
        return max(0.0, job.end_timestamp - job.start_timestamp)
    return 0.0


def _timeline_cache(settings) -> TimelineTileCache:
    return TimelineTileCache(
        cache_dir=settings.storage_root / "timeline_cache",
        max_jobs=settings.timeline_cache_max_jobs,
        memory_cache_max_items=settings.timeline_tile_memory_cache_items,
    )


def _pcm_cache_bytes_limit(settings) -> int:
    return int(settings.timeline_pcm_memory_cache_mb) * 1024 * 1024


def _tile_sample_rate(duration_sec: float, width_px: int) -> int:
    """Compute a target sample rate appropriate for tile rendering.

    For very long tiles (e.g. 24h = 86400s) we need far fewer samples than
    32 kHz would produce.  We target roughly (width_px * hop_multiplier)
    STFT frames so the rendered image fills the tile width without
    allocating gigabytes of silence.

    Returns a sample rate clamped between 200 Hz and 32000 Hz.
    """
    # We want roughly width_px * 4 STFT columns.  With hop_length=256
    # that means we need width_px * 4 * 256 samples over duration_sec.
    desired_samples = width_px * 4 * 256
    if duration_sec <= 0:
        return 32000
    sr = int(desired_samples / duration_sec)
    return max(200, min(sr, 32000))


def _resolve_tile_audio(
    *,
    job,
    zoom_level: str,
    tile_index: int,
    settings,
    cache: TimelineTileCache,
) -> tuple[np.ndarray, int]:
    """Resolve audio for a tile and return (audio, sample_rate)."""
    from humpback.processing.timeline_audio import resolve_timeline_audio

    start_epoch, end_epoch = tile_time_range(
        zoom_level, tile_index=tile_index, job_start_timestamp=job.start_timestamp
    )
    duration_sec = end_epoch - start_epoch
    sr = _tile_sample_rate(duration_sec, settings.timeline_tile_width_px)

    audio = resolve_timeline_audio(
        hydrophone_id=job.hydrophone_id or "",
        local_cache_path=job.local_cache_path or settings.s3_cache_path or "",
        job_start_timestamp=job.start_timestamp,
        job_end_timestamp=job.end_timestamp,
        start_sec=start_epoch,
        duration_sec=duration_sec,
        target_sr=sr,
        noaa_cache_path=settings.noaa_cache_path,
        timeline_cache=cache,
        job_id=job.id,
        manifest_cache_items=settings.timeline_manifest_memory_cache_items,
        pcm_cache_max_bytes=_pcm_cache_bytes_limit(settings),
    )
    return audio, sr


def _render_tile_sync(
    *,
    job,
    zoom_level: str,
    tile_index: int,
    settings,
    cache: TimelineTileCache,
    ref_db: float | None = None,
) -> bytes:
    """Render a spectrogram tile (CPU-bound, runs in thread)."""
    audio, sr = _resolve_tile_audio(
        job=job,
        zoom_level=zoom_level,
        tile_index=tile_index,
        settings=settings,
        cache=cache,
    )

    # Adapt n_fft so it does not exceed the audio length
    n_fft = min(2048, len(audio))
    if n_fft < 16:
        n_fft = 16
    hop_length = max(1, n_fft // 8)

    kwargs: dict = {
        "sample_rate": sr,
        "n_fft": n_fft,
        "hop_length": hop_length,
        "width_px": settings.timeline_tile_width_px,
        "height_px": settings.timeline_tile_height_px,
        "dynamic_range_db": settings.timeline_dynamic_range_db,
    }
    if ref_db is not None:
        kwargs["ref_db"] = ref_db

    tile_bytes = generate_timeline_tile(audio, **kwargs)
    cache.put(job.id, zoom_level, tile_index, tile_bytes)
    return tile_bytes


_PREPARE_PRIORITY = ["1h", "15m", "5m", "1m", "6h", "24h"]
_STATS_SAMPLE_COUNT = 10
_STATS_ZOOM_PRIORITY = ["15m", "5m", "1h", "1m"]
_SILENCE_FLOOR_DB = -115.0
_DEFAULT_REF_DB = -50.0


def _prepare_target_status_payload(
    scope: Literal["startup", "full"], targets: PrepareTargets
) -> dict[str, object]:
    return {
        "scope": scope,
        "zooms": {
            zoom: _ALL_TILES if indices is None else indices
            for zoom, indices in targets.items()
        },
    }


def _reserve_prepare_slot(job_id: str) -> bool:
    with _preparing_lock:
        if job_id in _preparing:
            return False
        _preparing.add(job_id)
        return True


def _release_prepare_slot(job_id: str) -> None:
    with _preparing_lock:
        _preparing.discard(job_id)


def _tile_indices_around_center(
    *,
    zoom_level: str,
    center_timestamp: float,
    job_start_timestamp: float,
    total_tiles: int,
    radius_tiles: int,
) -> list[int]:
    if total_tiles <= 0:
        return []
    center_idx = int(
        (center_timestamp - job_start_timestamp) // tile_duration_sec(zoom_level)
    )
    center_idx = min(max(center_idx, 0), total_tiles - 1)
    first = max(0, center_idx - radius_tiles)
    last = min(total_tiles - 1, center_idx + radius_tiles)
    return list(range(first, last + 1))


def _build_full_prepare_targets(job) -> PrepareTargets:
    duration = _job_duration(job)
    targets: PrepareTargets = {}
    for zoom in _PREPARE_PRIORITY:
        if tile_count(zoom, job_duration_sec=duration) > 0:
            targets[zoom] = None
    return targets


def _build_startup_prepare_targets(
    *,
    job,
    settings,
    zoom_level: str,
    center_timestamp: float,
    radius_tiles: int,
) -> PrepareTargets:
    duration = _job_duration(job)
    targets: PrepareTargets = {}
    zoom_order = list(ZOOM_LEVELS)
    zoom_index = next(
        i for i, candidate in enumerate(zoom_order) if candidate == zoom_level
    )

    target_center = min(
        max(center_timestamp, job.start_timestamp or center_timestamp),
        job.end_timestamp or center_timestamp,
    )

    base_total = tile_count(zoom_level, job_duration_sec=duration)
    targets[zoom_level] = _tile_indices_around_center(
        zoom_level=zoom_level,
        center_timestamp=target_center,
        job_start_timestamp=job.start_timestamp or target_center,
        total_tiles=base_total,
        radius_tiles=radius_tiles,
    )

    for coarse_offset in range(1, settings.timeline_startup_coarse_levels + 1):
        coarse_index = zoom_index - coarse_offset
        if coarse_index < 0:
            break
        coarse_zoom = zoom_order[coarse_index]
        coarse_total = tile_count(coarse_zoom, job_duration_sec=duration)
        targets[coarse_zoom] = _tile_indices_around_center(
            zoom_level=coarse_zoom,
            center_timestamp=target_center,
            job_start_timestamp=job.start_timestamp or target_center,
            total_tiles=coarse_total,
            radius_tiles=0,
        )

    return targets


def _prepare_targets_from_request(
    *,
    job,
    settings,
    request: PrepareRequest,
) -> tuple[PrepareTargets, dict[str, object]]:
    if request.scope == "full":
        targets = _build_full_prepare_targets(job)
        return targets, _prepare_target_status_payload("full", targets)

    zoom_level = request.zoom_level or "1h"
    if zoom_level not in ZOOM_LEVELS:
        raise HTTPException(
            400, f"Invalid zoom level: {zoom_level}. Must be one of {ZOOM_LEVELS}"
        )

    center_timestamp = request.center_timestamp
    if center_timestamp is None:
        start = job.start_timestamp or 0.0
        end = job.end_timestamp or start
        center_timestamp = start + max(0.0, end - start) / 2.0

    radius_tiles = (
        request.radius_tiles
        if request.radius_tiles is not None
        else settings.timeline_startup_radius_tiles
    )
    targets = _build_startup_prepare_targets(
        job=job,
        settings=settings,
        zoom_level=zoom_level,
        center_timestamp=center_timestamp,
        radius_tiles=radius_tiles,
    )
    return targets, _prepare_target_status_payload("startup", targets)


def _compute_job_ref_db(
    *,
    job,
    settings,
    cache: TimelineTileCache,
) -> float:
    """Sample tiles across the job to compute a per-job reference dB level.

    Returns the cached value if already computed, otherwise samples up to
    _STATS_SAMPLE_COUNT tiles and takes the max of per-sample max_db values
    as ref_db.  Tries multiple zoom levels in priority order, skipping
    silence tiles.
    """
    from humpback.processing.timeline_tiles import compute_power_db_stats

    cached = cache.get_ref_db(job.id)
    if cached is not None:
        return cached

    duration = _job_duration(job)
    if duration <= 0:
        cache.put_ref_db(job.id, _DEFAULT_REF_DB)
        return _DEFAULT_REF_DB

    max_db_values: list[float] = []

    for stats_zoom in _STATS_ZOOM_PRIORITY:
        total = tile_count(stats_zoom, job_duration_sec=duration)
        if total == 0:
            continue

        # Evenly space sample indices across the job
        if total <= _STATS_SAMPLE_COUNT:
            sample_indices = list(range(total))
        else:
            step = total / _STATS_SAMPLE_COUNT
            sample_indices = [int(i * step) for i in range(_STATS_SAMPLE_COUNT)]

        for idx in sample_indices:
            try:
                audio, sr = _resolve_tile_audio(
                    job=job,
                    zoom_level=stats_zoom,
                    tile_index=idx,
                    settings=settings,
                    cache=cache,
                )
                # Skip silence / zero-audio tiles
                if float(np.max(np.abs(audio))) < 1e-10:
                    continue
                n_fft = min(2048, len(audio))
                if n_fft < 16:
                    n_fft = 16
                hop_length = max(1, n_fft // 8)
                stats = compute_power_db_stats(
                    audio, sr, n_fft=n_fft, hop_length=hop_length
                )
                # Skip floor-level results (silence in dB domain)
                if stats["max_db"] <= _SILENCE_FLOOR_DB:
                    continue
                max_db_values.append(stats["max_db"])
            except Exception:
                logger.exception(
                    "Failed to compute stats for tile %s/%d of job %s",
                    stats_zoom,
                    idx,
                    job.id,
                )

        if max_db_values:
            break  # Got enough data from this zoom level

    if max_db_values:
        ref = float(np.max(max_db_values))
    else:
        ref = _DEFAULT_REF_DB

    cache.put_ref_db(job.id, ref)
    logger.info(
        "Computed ref_db=%.1f for job %s from %d samples",
        ref,
        job.id,
        len(max_db_values),
    )
    return ref


def _prepare_tiles_sync(
    *,
    job,
    settings,
    cache: TimelineTileCache,
    targets: PrepareTargets | None = None,
) -> int:
    """Render tiles for requested zoom levels in priority order. Skips cached."""
    duration = _job_duration(job)
    if duration <= 0:
        return 0

    # Pass 1: compute per-job reference dB level
    ref_db = _compute_job_ref_db(job=job, settings=settings, cache=cache)

    prepare_targets = targets or _build_full_prepare_targets(job)

    def _iter_targets():
        for zoom, indices in prepare_targets.items():
            count = tile_count(zoom, job_duration_sec=duration)
            if indices is None:
                for idx in range(count):
                    yield zoom, idx
            else:
                for idx in indices:
                    if 0 <= idx < count:
                        yield zoom, idx

    def _render_target(target: tuple[str, int]) -> int:
        zoom, idx = target
        if cache.has(job.id, zoom, idx):
            return 0
        try:
            _render_tile_sync(
                job=job,
                zoom_level=zoom,
                tile_index=idx,
                settings=settings,
                cache=cache,
                ref_db=ref_db,
            )
            return 1
        except Exception:
            logger.exception(
                "Failed to render tile %s/%d for job %s",
                zoom,
                idx,
                job.id,
            )
            return 0

    worker_count = max(1, settings.timeline_prepare_workers)
    if worker_count == 1:
        return sum(_render_target(target) for target in _iter_targets())

    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="timeline-prepare",
    ) as executor:
        return sum(executor.map(_render_target, _iter_targets()))


def _launch_prepare_thread(
    *,
    job,
    settings,
    cache: TimelineTileCache,
    prepare_lock: TimelinePrepareLock,
    targets: PrepareTargets | None = None,
) -> None:
    def _background() -> None:
        try:
            _prepare_tiles_sync(
                job=job,
                settings=settings,
                cache=cache,
                targets=targets,
            )
        finally:
            prepare_lock.release()
            _release_prepare_slot(job.id)

    try:
        threading.Thread(target=_background, daemon=True).start()
    except Exception:
        prepare_lock.release()
        _release_prepare_slot(job.id)
        raise


def _try_launch_prepare(
    *,
    job,
    settings,
    cache: TimelineTileCache,
    targets: PrepareTargets,
    status_payload: dict[str, object] | None,
) -> bool:
    if not targets:
        return False
    if not _reserve_prepare_slot(job.id):
        return False

    prepare_lock = cache.try_acquire_prepare_lock(job.id)
    if prepare_lock is None:
        _release_prepare_slot(job.id)
        return False

    if status_payload is not None:
        cache.put_prepare_plan(job.id, status_payload)

    _launch_prepare_thread(
        job=job,
        settings=settings,
        cache=cache,
        prepare_lock=prepare_lock,
        targets=targets,
    )
    return True


def _neighbor_prepare_targets(
    *,
    job,
    zoom_level: str,
    tile_index: int,
    settings,
) -> PrepareTargets | None:
    radius = settings.timeline_neighbor_prefetch_radius
    if radius <= 0:
        return None
    duration = _job_duration(job)
    total = tile_count(zoom_level, job_duration_sec=duration)
    if total <= 1:
        return None
    indices = [
        idx
        for idx in range(
            max(0, tile_index - radius), min(total - 1, tile_index + radius) + 1
        )
        if idx != tile_index
    ]
    if not indices:
        return None
    return {zoom_level: indices}


def _encode_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    """Encode float32 audio to 16-bit PCM WAV bytes with peak normalization."""
    # Peak-normalize so raw hydrophone audio is audible
    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio = audio / peak
    audio_clipped = np.clip(audio, -1.0, 1.0)
    pcm = (audio_clipped * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def _encode_mp3(audio: np.ndarray, sample_rate: int) -> bytes:
    """Encode float32 audio to MP3 via ffmpeg subprocess."""
    import subprocess
    import tempfile
    from pathlib import Path

    wav_bytes = _encode_wav(audio, sample_rate)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_f:
        wav_path = wav_f.name
        wav_f.write(wav_bytes)

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as mp3_f:
        mp3_path = mp3_f.name

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                wav_path,
                "-codec:a",
                "libmp3lame",
                "-b:a",
                "128k",
                "-ac",
                "1",
                mp3_path,
            ],
            capture_output=True,
            check=True,
        )
        return Path(mp3_path).read_bytes()
    finally:
        Path(wav_path).unlink(missing_ok=True)
        Path(mp3_path).unlink(missing_ok=True)


# ---- Endpoints ----


@router.get("/tile")
async def get_tile(
    job_id: str,
    session: SessionDep,
    settings: SettingsDep,
    zoom_level: str = Query(
        ..., description="Zoom level (e.g. 24h, 6h, 1h, 15m, 5m, 1m)"
    ),
    tile_index: int = Query(..., ge=0, description="Tile index within the zoom level"),
) -> Response:
    """Return a spectrogram PNG tile for the given zoom level and index."""
    job = await _get_job_or_404(session, job_id)

    if zoom_level not in ZOOM_LEVELS:
        raise HTTPException(
            400, f"Invalid zoom level: {zoom_level}. Must be one of {ZOOM_LEVELS}"
        )

    duration = _job_duration(job)
    max_tiles = tile_count(zoom_level, job_duration_sec=duration)
    if tile_index >= max_tiles:
        raise HTTPException(
            400,
            f"Tile index {tile_index} out of range (max {max_tiles - 1} for {zoom_level})",
        )

    cache = _timeline_cache(settings)

    # Check cache first
    cached = cache.get(job.id, zoom_level, tile_index)
    if cached is not None:
        return Response(content=cached, media_type="image/png")

    # Use per-job ref_db; compute it on-demand if /prepare hasn't run yet
    ref_db = cache.get_ref_db(job.id)
    if ref_db is None:
        ref_db = await asyncio.to_thread(
            _compute_job_ref_db, job=job, settings=settings, cache=cache
        )

    # Render on miss (CPU-bound -> thread)
    tile_bytes = await asyncio.to_thread(
        _render_tile_sync,
        job=job,
        zoom_level=zoom_level,
        tile_index=tile_index,
        settings=settings,
        cache=cache,
        ref_db=ref_db,
    )

    neighbor_targets = _neighbor_prepare_targets(
        job=job,
        zoom_level=zoom_level,
        tile_index=tile_index,
        settings=settings,
    )
    if neighbor_targets is not None:
        _try_launch_prepare(
            job=job,
            settings=settings,
            cache=cache,
            targets=neighbor_targets,
            status_payload=None,
        )
    return Response(content=tile_bytes, media_type="image/png")


def _parse_filename_epoch(filename: str) -> float | None:
    """Parse a UTC timestamp from a detection filename like '20190601T054600Z.wav'.

    Returns epoch seconds, or None if the filename doesn't match the expected format.
    """
    import re
    from datetime import datetime, timezone

    m = re.match(r"(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z", filename)
    if not m:
        return None
    dt = datetime(
        int(m.group(1)),
        int(m.group(2)),
        int(m.group(3)),
        int(m.group(4)),
        int(m.group(5)),
        int(m.group(6)),
        tzinfo=timezone.utc,
    )
    return dt.timestamp()


@router.get("/confidence")
async def get_confidence(
    job_id: str,
    session: SessionDep,
    settings: SettingsDep,
) -> ConfidenceResponse:
    """Return confidence scores as a timeline-ordered JSON array.

    Scores are bucketed into fixed-size windows spanning the full job
    duration so that index ``i`` maps to
    ``job.start_timestamp + i * window_sec``.  Buckets with no data
    are ``null``.
    """
    job = await _get_job_or_404(session, job_id)

    diag_path = detection_diagnostics_path(settings.storage_root, job.id)
    if not diag_path.exists():
        raise HTTPException(404, "No diagnostics data found for this job")

    # Determine window size from the classifier model
    from humpback.models.classifier import ClassifierModel
    from sqlalchemy import select as sa_select

    result = await session.execute(
        sa_select(ClassifierModel).where(ClassifierModel.id == job.classifier_model_id)
    )
    model = result.scalar_one_or_none()
    window_sec = model.window_size_seconds if model else 5.0

    job_start = job.start_timestamp or 0.0
    job_end = job.end_timestamp or 0.0
    job_duration = job_end - job_start

    table = pq.read_table(str(diag_path))

    offset_col = table.column("offset_sec").to_pylist()
    score_col = table.column("confidence").to_pylist()
    filename_col = (
        table.column("filename").to_pylist()
        if "filename" in table.column_names
        else [None] * len(offset_col)
    )

    # Compute timeline-absolute offset for each row
    absolute_offsets: list[float] = []
    for offset, filename in zip(offset_col, filename_col):
        if filename is not None:
            file_epoch = _parse_filename_epoch(filename)
            if file_epoch is not None:
                # offset_sec is relative to the audio file start
                absolute_offsets.append(file_epoch - job_start + offset)
                continue
        # Fallback: offset is already job-relative (local detection jobs)
        absolute_offsets.append(offset)

    # Bucket into fixed-size windows across the full job duration
    n_buckets = max(1, int(job_duration / window_sec))
    bucket_sums: list[float] = [0.0] * n_buckets
    bucket_counts: list[int] = [0] * n_buckets

    for abs_offset, score in zip(absolute_offsets, score_col):
        idx = int(abs_offset / window_sec)
        if 0 <= idx < n_buckets:
            bucket_sums[idx] += score
            bucket_counts[idx] += 1

    scores: list[float | None] = [
        bucket_sums[i] / bucket_counts[i] if bucket_counts[i] > 0 else None
        for i in range(n_buckets)
    ]

    return ConfidenceResponse(
        window_sec=window_sec,
        scores=scores,
        start_timestamp=job_start,
        end_timestamp=job_end,
    )


@router.get("/audio")
async def get_audio(
    job_id: str,
    session: SessionDep,
    settings: SettingsDep,
    start_sec: float = Query(
        ..., description="Timeline-absolute start position (epoch seconds)"
    ),
    duration_sec: float = Query(..., gt=0, description="Duration in seconds"),
    format: str = Query("wav", pattern="^(wav|mp3)$"),
) -> Response:
    """Return audio for an arbitrary timeline position (WAV or MP3)."""
    if duration_sec > 600.0:
        raise HTTPException(400, "Maximum audio duration is 600 seconds")

    job = await _get_job_or_404(session, job_id)

    if not job.hydrophone_id:
        raise HTTPException(
            400, "Audio endpoint is only available for hydrophone detection jobs"
        )

    from humpback.processing.timeline_audio import resolve_timeline_audio

    cache = _timeline_cache(settings)
    audio = await asyncio.to_thread(
        resolve_timeline_audio,
        hydrophone_id=job.hydrophone_id,
        local_cache_path=job.local_cache_path or settings.s3_cache_path or "",
        job_start_timestamp=job.start_timestamp or 0.0,
        job_end_timestamp=job.end_timestamp or 0.0,
        start_sec=start_sec,
        duration_sec=duration_sec,
        target_sr=32000,
        noaa_cache_path=settings.noaa_cache_path,
        timeline_cache=cache,
        job_id=job.id,
        manifest_cache_items=settings.timeline_manifest_memory_cache_items,
        pcm_cache_max_bytes=_pcm_cache_bytes_limit(settings),
    )

    if format == "mp3":
        data = _encode_mp3(audio, 32000)
        return Response(content=data, media_type="audio/mpeg")
    else:
        data = _encode_wav(audio, sample_rate=32000)
        return Response(content=data, media_type="audio/wav")


@router.post("/prepare")
async def prepare_tiles(
    job_id: str,
    session: SessionDep,
    settings: SettingsDep,
    request: PrepareRequest | None = Body(default=None),
) -> PrepareResponse:
    """Launch background rendering of startup or full timeline tile targets."""
    job = await _get_job_or_404(session, job_id)
    prepare_request = request or PrepareRequest()

    cache = _timeline_cache(settings)
    targets, status_payload = _prepare_targets_from_request(
        job=job,
        settings=settings,
        request=prepare_request,
    )

    _try_launch_prepare(
        job=job,
        settings=settings,
        cache=cache,
        targets=targets,
        status_payload=status_payload,
    )

    # Mark timeline_tiles_ready immediately — signals that preparation was
    # initiated.  Use /prepare-status for real per-zoom progress.
    from humpback.models.classifier import DetectionJob
    from sqlalchemy import select as sa_select

    result = await session.execute(
        sa_select(DetectionJob).where(DetectionJob.id == job_id)
    )
    db_job = result.scalar_one_or_none()
    if db_job is not None and not db_job.timeline_tiles_ready:
        db_job.timeline_tiles_ready = True
        await session.commit()

    return PrepareResponse(status="preparing", timeline_tiles_ready=True)


@router.get("/prepare-status")
async def prepare_status(
    job_id: str,
    session: SessionDep,
    settings: SettingsDep,
) -> dict[str, dict[str, int]]:
    """Return per-zoom-level rendering progress for a detection job."""
    job = await _get_job_or_404(session, job_id)
    duration = _job_duration(job)
    cache = _timeline_cache(settings)
    plan = cache.get_prepare_plan(job.id)

    if isinstance(plan, dict):
        zoom_payload = plan.get("zooms")
        if isinstance(zoom_payload, dict):
            status: dict[str, dict[str, int]] = {}
            for zoom, plan_value in zoom_payload.items():
                if zoom not in ZOOM_LEVELS:
                    continue
                if plan_value == _ALL_TILES:
                    total = tile_count(zoom, job_duration_sec=duration)
                    rendered = cache.tile_count_for_zoom(job.id, zoom)
                elif isinstance(plan_value, list):
                    indices = [
                        int(item)
                        for item in plan_value
                        if isinstance(item, int)
                        or (isinstance(item, float) and item.is_integer())
                    ]
                    total = len(indices)
                    rendered = cache.count_cached_tiles(job.id, zoom, indices)
                else:
                    continue
                status[zoom] = {"total": total, "rendered": min(rendered, total)}
            if status:
                return status

    status: dict[str, dict[str, int]] = {}
    for zoom in ZOOM_LEVELS:
        total = tile_count(zoom, job_duration_sec=duration)
        rendered = cache.tile_count_for_zoom(job.id, zoom)
        status[zoom] = {"total": total, "rendered": min(rendered, total)}
    return status
