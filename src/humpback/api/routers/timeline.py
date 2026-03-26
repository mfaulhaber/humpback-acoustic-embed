"""API sub-router for timeline spectrogram tiles, confidence, and audio."""

from __future__ import annotations

import asyncio
import io
import logging
import threading
import wave

import numpy as np
import pyarrow.parquet as pq
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel

from humpback.api.deps import SessionDep, SettingsDep
from humpback.processing.timeline_cache import TimelinePrepareLock, TimelineTileCache
from humpback.processing.timeline_tiles import (
    ZOOM_LEVELS,
    generate_timeline_tile,
    tile_count,
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


# ---- Helpers ----


async def _get_job_or_404(session, job_id: str):
    job = await classifier_service.get_detection_job(session, job_id)
    if job is None:
        raise HTTPException(404, "Detection job not found")
    return job


def _job_duration(job) -> float:
    if job.start_timestamp is not None and job.end_timestamp is not None:
        return max(0.0, job.end_timestamp - job.start_timestamp)
    return 0.0


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
        job=job, zoom_level=zoom_level, tile_index=tile_index, settings=settings
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
                    job=job, zoom_level=stats_zoom, tile_index=idx, settings=settings
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
    zoom_levels: list[str] | None = None,
) -> int:
    """Render tiles for requested zoom levels in priority order. Skips cached."""
    duration = _job_duration(job)
    if duration <= 0:
        return 0

    # Pass 1: compute per-job reference dB level
    ref_db = _compute_job_ref_db(job=job, settings=settings, cache=cache)

    # Pass 2: render tiles with consistent normalization
    levels = zoom_levels or list(_PREPARE_PRIORITY)
    priority = {z: i for i, z in enumerate(_PREPARE_PRIORITY)}
    levels.sort(key=lambda z: priority.get(z, 99))

    rendered = 0
    for zoom in levels:
        count = tile_count(zoom, job_duration_sec=duration)
        for idx in range(count):
            if cache.has(job.id, zoom, idx):
                continue
            try:
                _render_tile_sync(
                    job=job,
                    zoom_level=zoom,
                    tile_index=idx,
                    settings=settings,
                    cache=cache,
                    ref_db=ref_db,
                )
                rendered += 1
            except Exception:
                logger.exception(
                    "Failed to render tile %s/%d for job %s",
                    zoom,
                    idx,
                    job.id,
                )
    return rendered


def _launch_prepare_thread(
    *,
    job,
    settings,
    cache: TimelineTileCache,
    prepare_lock: TimelinePrepareLock,
) -> None:
    def _background() -> None:
        try:
            _prepare_tiles_sync(job=job, settings=settings, cache=cache)
        finally:
            prepare_lock.release()
            with _preparing_lock:
                _preparing.discard(job.id)

    try:
        threading.Thread(target=_background, daemon=True).start()
    except Exception:
        prepare_lock.release()
        with _preparing_lock:
            _preparing.discard(job.id)
        raise


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

    cache = TimelineTileCache(
        cache_dir=settings.storage_root / "timeline_cache",
        max_jobs=settings.timeline_cache_max_jobs,
    )

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
) -> PrepareResponse:
    """Launch background rendering of all zoom-level tiles for the timeline viewer."""
    job = await _get_job_or_404(session, job_id)

    cache = TimelineTileCache(
        cache_dir=settings.storage_root / "timeline_cache",
        max_jobs=settings.timeline_cache_max_jobs,
    )

    reserved_locally = False
    with _preparing_lock:
        if job.id not in _preparing:
            _preparing.add(job.id)
            reserved_locally = True

    if reserved_locally:
        prepare_lock = cache.try_acquire_prepare_lock(job.id)
        if prepare_lock is None:
            with _preparing_lock:
                _preparing.discard(job.id)
        else:
            _launch_prepare_thread(
                job=job,
                settings=settings,
                cache=cache,
                prepare_lock=prepare_lock,
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
    cache = TimelineTileCache(
        cache_dir=settings.storage_root / "timeline_cache",
        max_jobs=settings.timeline_cache_max_jobs,
    )
    status: dict[str, dict[str, int]] = {}
    for zoom in ZOOM_LEVELS:
        total = tile_count(zoom, job_duration_sec=duration)
        rendered = cache.tile_count_for_zoom(job.id, zoom)
        status[zoom] = {"total": total, "rendered": min(rendered, total)}
    return status
