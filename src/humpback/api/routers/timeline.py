"""API sub-router for timeline spectrogram tiles, confidence, and audio."""

from __future__ import annotations

import asyncio
import io
import logging
import wave

import numpy as np
import pyarrow.parquet as pq
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel

from humpback.api.deps import SessionDep, SettingsDep
from humpback.processing.timeline_cache import TimelineTileCache
from humpback.processing.timeline_tiles import (
    ZOOM_LEVELS,
    generate_timeline_tile,
    tile_count,
    tile_time_range,
)
from humpback.services import classifier_service
from humpback.storage import detection_diagnostics_path, timeline_tiles_dir

logger = logging.getLogger(__name__)

router = APIRouter()


# ---- Response models ----


class ConfidenceResponse(BaseModel):
    window_sec: float
    scores: list[float]
    start_timestamp: float
    end_timestamp: float


class PrepareResponse(BaseModel):
    tiles_rendered: int
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


def _render_tile_sync(
    *,
    job,
    zoom_level: str,
    tile_index: int,
    settings,
    cache: TimelineTileCache,
) -> bytes:
    """Render a spectrogram tile (CPU-bound, runs in thread)."""
    from humpback.processing.timeline_audio import resolve_timeline_audio

    start_epoch, end_epoch = tile_time_range(
        zoom_level, tile_index=tile_index, job_start_timestamp=job.start_timestamp
    )
    duration_sec = end_epoch - start_epoch

    sr = _tile_sample_rate(duration_sec, settings.timeline_tile_width_px)

    audio = resolve_timeline_audio(
        hydrophone_id=job.hydrophone_id or "",
        local_cache_path=job.local_cache_path or "",
        job_start_timestamp=job.start_timestamp,
        job_end_timestamp=job.end_timestamp,
        start_sec=start_epoch,
        duration_sec=duration_sec,
        target_sr=sr,
        noaa_cache_path=settings.noaa_cache_path,
    )

    # Adapt n_fft so it does not exceed the audio length
    n_fft = min(2048, len(audio))
    if n_fft < 16:
        n_fft = 16
    hop_length = max(1, n_fft // 8)

    tile_bytes = generate_timeline_tile(
        audio,
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        width_px=settings.timeline_tile_width_px,
        height_px=settings.timeline_tile_height_px,
        dynamic_range_db=settings.timeline_dynamic_range_db,
    )
    cache.put(job.id, zoom_level, tile_index, tile_bytes)
    return tile_bytes


def _prepare_tiles_sync(
    *,
    job,
    settings,
    cache: TimelineTileCache,
) -> int:
    """Pre-render coarse tiles (24h + 6h zoom levels). Returns count rendered."""
    from humpback.processing.timeline_audio import resolve_timeline_audio

    duration = _job_duration(job)
    rendered = 0

    for level in ("24h", "6h"):
        n_tiles = tile_count(level, job_duration_sec=duration)
        for idx in range(n_tiles):
            # Skip if already cached
            if cache.get(job.id, level, idx) is not None:
                rendered += 1
                continue

            start_epoch, end_epoch = tile_time_range(
                level, tile_index=idx, job_start_timestamp=job.start_timestamp
            )
            tile_dur = end_epoch - start_epoch

            sr = _tile_sample_rate(tile_dur, settings.timeline_tile_width_px)

            audio = resolve_timeline_audio(
                hydrophone_id=job.hydrophone_id or "",
                local_cache_path=job.local_cache_path or "",
                job_start_timestamp=job.start_timestamp,
                job_end_timestamp=job.end_timestamp,
                start_sec=start_epoch,
                duration_sec=tile_dur,
                target_sr=sr,
                noaa_cache_path=settings.noaa_cache_path,
            )

            n_fft = min(2048, len(audio))
            if n_fft < 16:
                n_fft = 16
            hop_length = max(1, n_fft // 8)

            tile_bytes = generate_timeline_tile(
                audio,
                sample_rate=sr,
                n_fft=n_fft,
                hop_length=hop_length,
                width_px=settings.timeline_tile_width_px,
                height_px=settings.timeline_tile_height_px,
                dynamic_range_db=settings.timeline_dynamic_range_db,
            )
            cache.put(job.id, level, idx, tile_bytes)
            rendered += 1

    return rendered


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

    tiles_dir = timeline_tiles_dir(settings.storage_root, job.id)
    cache = TimelineTileCache(
        tiles_dir, max_items=settings.timeline_tile_cache_max_items
    )

    # Check cache first
    cached = cache.get(job.id, zoom_level, tile_index)
    if cached is not None:
        return Response(content=cached, media_type="image/png")

    # Render on miss (CPU-bound -> thread)
    tile_bytes = await asyncio.to_thread(
        _render_tile_sync,
        job=job,
        zoom_level=zoom_level,
        tile_index=tile_index,
        settings=settings,
        cache=cache,
    )
    return Response(content=tile_bytes, media_type="image/png")


@router.get("/confidence")
async def get_confidence(
    job_id: str,
    session: SessionDep,
    settings: SettingsDep,
) -> ConfidenceResponse:
    """Return confidence scores as a timeline-ordered JSON array."""
    job = await _get_job_or_404(session, job_id)

    diag_path = detection_diagnostics_path(settings.storage_root, job.id)
    if not diag_path.exists():
        raise HTTPException(404, "No diagnostics data found for this job")

    table = pq.read_table(str(diag_path))

    offset_col = table.column("offset_sec").to_pylist()
    score_col = table.column("confidence").to_pylist()

    # Sort by offset
    pairs = sorted(zip(offset_col, score_col), key=lambda p: p[0])
    sorted_scores = [s for _, s in pairs]

    # Determine window size from the classifier model
    from humpback.models.classifier import ClassifierModel
    from sqlalchemy import select as sa_select

    result = await session.execute(
        sa_select(ClassifierModel).where(ClassifierModel.id == job.classifier_model_id)
    )
    model = result.scalar_one_or_none()
    window_sec = model.window_size_seconds if model else 5.0

    return ConfidenceResponse(
        window_sec=window_sec,
        scores=sorted_scores,
        start_timestamp=job.start_timestamp or 0.0,
        end_timestamp=job.end_timestamp or 0.0,
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
) -> Response:
    """Return WAV audio for an arbitrary timeline position."""
    if duration_sec > 120.0:
        raise HTTPException(400, "Maximum audio duration is 120 seconds")

    job = await _get_job_or_404(session, job_id)

    if not job.hydrophone_id:
        raise HTTPException(
            400, "Audio endpoint is only available for hydrophone detection jobs"
        )

    from humpback.processing.timeline_audio import resolve_timeline_audio

    audio = await asyncio.to_thread(
        resolve_timeline_audio,
        hydrophone_id=job.hydrophone_id,
        local_cache_path=job.local_cache_path or "",
        job_start_timestamp=job.start_timestamp or 0.0,
        job_end_timestamp=job.end_timestamp or 0.0,
        start_sec=start_sec,
        duration_sec=duration_sec,
        target_sr=32000,
        noaa_cache_path=settings.noaa_cache_path,
    )

    wav_bytes = _encode_wav(audio, sample_rate=32000)
    return Response(content=wav_bytes, media_type="audio/wav")


@router.post("/prepare")
async def prepare_tiles(
    job_id: str,
    session: SessionDep,
    settings: SettingsDep,
) -> PrepareResponse:
    """Pre-render coarse tiles (24h + 6h zoom levels) for the timeline viewer."""
    job = await _get_job_or_404(session, job_id)

    tiles_dir = timeline_tiles_dir(settings.storage_root, job.id)
    cache = TimelineTileCache(
        tiles_dir, max_items=settings.timeline_tile_cache_max_items
    )

    rendered = await asyncio.to_thread(
        _prepare_tiles_sync,
        job=job,
        settings=settings,
        cache=cache,
    )

    # Mark job as timeline_tiles_ready
    from humpback.models.classifier import DetectionJob
    from sqlalchemy import select as sa_select

    result = await session.execute(
        sa_select(DetectionJob).where(DetectionJob.id == job_id)
    )
    db_job = result.scalar_one_or_none()
    if db_job is not None:
        db_job.timeline_tiles_ready = True
        await session.commit()

    return PrepareResponse(
        tiles_rendered=rendered,
        timeline_tiles_ready=True,
    )
