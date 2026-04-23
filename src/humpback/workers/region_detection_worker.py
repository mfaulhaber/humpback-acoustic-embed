"""Pass 1 — region detection worker.

Runs dense Perch inference + hysteresis + padded region emission on one
audio source (uploaded file or hydrophone range) and writes
``trace.parquet`` + ``regions.parquet`` to the per-job storage directory.
See ``docs/specs/2026-04-11-call-parsing-pass1-region-detector-design.md``
for the algorithmic contract and ADR-049 for the defaults rationale.
"""

from __future__ import annotations

import logging
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.call_parsing.regions import decode_regions
from humpback.call_parsing.storage import (
    chunk_parquet_path,
    read_all_chunk_embeddings,
    read_all_chunk_traces,
    read_manifest,
    region_job_dir,
    update_manifest_chunk,
    write_chunk_embeddings,
    write_chunk_trace,
    write_embeddings,
    write_manifest,
    write_regions,
    write_trace,
)
from humpback.call_parsing.types import WindowEmbedding, WindowScore
from humpback.classifier.detector import score_audio_windows
from humpback.classifier.detector_utils import merge_detection_events
from humpback.config import Settings
from humpback.models.audio import AudioFile
from humpback.models.call_parsing import RegionDetectionJob
from humpback.models.classifier import ClassifierModel
from humpback.processing.audio_io import decode_audio, resample
from humpback.schemas.call_parsing import RegionDetectionConfig
from humpback.storage import ensure_dir, resolve_audio_path
from humpback.workers.model_cache import get_model_by_version
from humpback.workers.queue import claim_region_detection_job

logger = logging.getLogger(__name__)


def _aligned_chunk_edges(
    start_ts: float,
    end_ts: float,
    chunk_sec: float,
    alignment_sec: float,
) -> list[tuple[float, float]]:
    """Chunk boundaries for the hydrophone streaming loop.

    Every chunk is at most ``chunk_sec`` wide and every boundary offset
    from ``start_ts`` is a whole multiple of ``alignment_sec``, so a Perch
    window that starts inside a chunk cannot have its body extend into
    the next chunk as long as ``hop_seconds == window_size_seconds``.
    With the ADR-049 defaults (``chunk_sec=1800``, ``alignment_sec=5``)
    the normal case is 48 even chunks per 24-hour range.

    Raises ``ValueError`` on malformed input (non-positive chunk/alignment
    or end <= start). The last chunk's right edge is always exactly
    ``end_ts`` even if it is shorter than ``chunk_sec`` or not aligned.
    """
    if chunk_sec <= 0 or alignment_sec <= 0:
        raise ValueError("chunk_sec and alignment_sec must be positive")
    if end_ts <= start_ts:
        raise ValueError("end_ts must be strictly after start_ts")

    # Round chunk width down to the nearest multiple of alignment_sec so
    # every "normal" chunk is a whole-window multiple. If chunk_sec is
    # smaller than one alignment unit we fall back to a single alignment
    # unit so the loop always makes progress.
    units = max(int(math.floor(chunk_sec / alignment_sec)), 1)
    aligned_chunk_sec = units * alignment_sec

    edges: list[tuple[float, float]] = []
    cursor = start_ts
    while cursor < end_ts:
        nxt = min(cursor + aligned_chunk_sec, end_ts)
        edges.append((cursor, nxt))
        cursor = nxt
    return edges


def _detector_config(
    config: RegionDetectionConfig, input_format: str
) -> dict[str, Any]:
    return {
        "window_size_seconds": config.window_size_seconds,
        "hop_seconds": config.hop_seconds,
        "input_format": input_format,
    }


def _score_records_to_window_scores(
    records: list[dict[str, Any]],
) -> list[WindowScore]:
    return [
        WindowScore(time_sec=float(r["offset_sec"]), score=float(r["confidence"]))
        for r in records
    ]


def _cleanup_partial_artifacts(job_dir: Path) -> None:
    """Delete final parquet files and ``.tmp`` sidecars left by a failed run.

    Completed chunk parquets and manifest.json are preserved so a re-queued
    job can resume from the last completed chunk.
    """
    if not job_dir.exists():
        return
    for name in ("trace.parquet", "regions.parquet", "embeddings.parquet"):
        path = job_dir / name
        if path.exists():
            try:
                path.unlink()
            except OSError:
                logger.warning("Failed to delete %s", path, exc_info=True)
    for tmp in job_dir.glob("*.tmp"):
        try:
            tmp.unlink()
        except OSError:
            logger.warning("Failed to delete %s", tmp, exc_info=True)
    for subdir_name in ("chunks", "chunk_embeddings"):
        subdir = job_dir / subdir_name
        if subdir.exists():
            for tmp in subdir.glob("*.tmp"):
                try:
                    tmp.unlink()
                except OSError:
                    logger.warning("Failed to delete %s", tmp, exc_info=True)


async def _load_file_trace(
    audio_file: AudioFile,
    *,
    config: RegionDetectionConfig,
    perch_model: Any,
    classifier: Any,
    input_format: str,
    target_sample_rate: int,
    storage_root: Path,
) -> tuple[list[dict[str, Any]], np.ndarray, float]:
    """Decode the whole audio file and run one ``score_audio_windows`` call."""
    path = resolve_audio_path(audio_file, storage_root)
    audio, sr = decode_audio(path)
    audio = resample(audio, sr, target_sample_rate)
    duration_sec = float(len(audio)) / float(target_sample_rate)
    records, embeddings = score_audio_windows(
        audio=audio,
        sample_rate=target_sample_rate,
        perch_model=perch_model,
        classifier=classifier,
        config=_detector_config(config, input_format),
    )
    del audio
    return records, embeddings, duration_sec


def _build_manifest(
    job_id: str,
    edges: list[tuple[float, float]],
    config: RegionDetectionConfig,
) -> dict[str, Any]:
    return {
        "job_id": job_id,
        "config": {
            "stream_chunk_sec": config.stream_chunk_sec,
            "window_size_seconds": config.window_size_seconds,
            "hop_seconds": config.hop_seconds,
        },
        "chunks": [
            {
                "index": i,
                "start_sec": start,
                "end_sec": end,
                "status": "pending",
                "completed_at": None,
                "trace_rows": None,
                "elapsed_sec": None,
            }
            for i, (start, end) in enumerate(edges)
        ],
    }


def _verify_manifest_chunks(manifest: dict[str, Any], job_dir: Path) -> int:
    """Verify completed chunks have parquet files. Returns verified count."""
    verified = 0
    for chunk in manifest["chunks"]:
        if chunk["status"] == "complete":
            if chunk_parquet_path(job_dir, chunk["index"]).exists():
                verified += 1
            else:
                chunk["status"] = "pending"
                chunk["completed_at"] = None
                chunk["trace_rows"] = None
                chunk["elapsed_sec"] = None
    return verified


async def _load_hydrophone_trace(
    hydrophone_id: str,
    start_ts: float,
    end_ts: float,
    *,
    config: RegionDetectionConfig,
    perch_model: Any,
    classifier: Any,
    input_format: str,
    target_sample_rate: int,
    settings: Settings,
    session: AsyncSession,
    job_id: str,
    job_dir: Path,
) -> tuple[list[WindowScore], list[WindowEmbedding], float]:
    """Stream a hydrophone range with chunk artifacts, resume, and progress.

    Writes per-chunk parquet files under ``job_dir/chunks/`` and maintains
    a ``manifest.json`` for resume support. Updates ``chunks_completed``
    in the DB after each chunk for API polling.
    """
    from humpback.classifier.archive import StreamSegment
    from humpback.classifier.providers import build_archive_detection_provider
    from humpback.classifier.s3_stream import iter_audio_chunks

    provider = build_archive_detection_provider(
        hydrophone_id,
        local_cache_path=None,
        s3_cache_path=settings.s3_cache_path,
        noaa_cache_path=settings.noaa_cache_path,
        force_refresh=False,
    )

    edges = _aligned_chunk_edges(
        start_ts,
        end_ts,
        chunk_sec=config.stream_chunk_sec,
        alignment_sec=config.window_size_seconds,
    )
    total_chunks = len(edges)
    range_sec = end_ts - start_ts

    # Resume or cold start
    manifest = read_manifest(job_dir)
    if manifest is not None:
        verified = _verify_manifest_chunks(manifest, job_dir)
        write_manifest(job_dir, manifest)
        if verified > 0:
            logger.info(
                "region_detection | job=%s | resumed from manifest (%d chunks cached)",
                job_id,
                verified,
            )
    else:
        manifest = _build_manifest(job_id, edges, config)
        write_manifest(job_dir, manifest)
        verified = 0

    # Restore windows_detected from completed chunks in manifest
    restored_windows = sum(
        c.get("windows_above_threshold", 0) or 0
        for c in manifest["chunks"]
        if c["status"] == "complete"
    )

    # Set progress columns in DB
    refreshed = await session.get(RegionDetectionJob, job_id)
    if refreshed is not None:
        refreshed.chunks_total = total_chunks
        refreshed.chunks_completed = verified
        refreshed.windows_detected = restored_windows
        await session.commit()

    logger.info(
        "region_detection | job=%s | start | chunks=%d | range=%.1fs | hydrophone=%s",
        job_id,
        total_chunks,
        range_sec,
        hydrophone_id,
    )

    # Build timeline once for the full range, then slice per chunk.
    try:
        full_timeline: list[StreamSegment] = provider.build_timeline(start_ts, end_ts)
    except FileNotFoundError:
        logger.info(
            "region_detection | job=%s | no audio in full range, skipping all chunks",
            job_id,
        )
        full_timeline = []

    def _filter_timeline(
        tl: list[StreamSegment], cs: float, ce: float
    ) -> list[StreamSegment]:
        return [s for s in tl if s.start_ts < ce and s.end_ts > cs]

    for i, (chunk_start, chunk_end) in enumerate(edges):
        chunk_meta = manifest["chunks"][i]
        if chunk_meta["status"] == "complete":
            continue

        chunk_duration = chunk_end - chunk_start
        chunk_timeline = _filter_timeline(full_timeline, chunk_start, chunk_end)

        t0 = time.monotonic()
        chunk_scores: list[WindowScore] = []
        chunk_embeddings: list[WindowEmbedding] = []

        if not chunk_timeline:
            logger.info(
                "region_detection | job=%s | chunk %d/%d | no audio segments, skipping",
                job_id,
                i + 1,
                total_chunks,
            )
        else:
            logger.info(
                "region_detection | job=%s | chunk %d/%d | %.1fs-%.1fs | fetching audio",
                job_id,
                i + 1,
                total_chunks,
                chunk_start - start_ts,
                chunk_end - start_ts,
            )
            try:
                for (
                    audio_buf,
                    seg_start_utc,
                    _segs_done,
                    _segs_total,
                ) in iter_audio_chunks(
                    provider,
                    chunk_start,
                    chunk_end,
                    chunk_seconds=chunk_end - chunk_start,
                    target_sr=target_sample_rate,
                    timeline=chunk_timeline,
                ):
                    buf_offset = seg_start_utc.timestamp() - start_ts
                    records, emb_arr = score_audio_windows(
                        audio=audio_buf,
                        sample_rate=target_sample_rate,
                        perch_model=perch_model,
                        classifier=classifier,
                        config=_detector_config(config, input_format),
                        time_offset_sec=buf_offset,
                    )
                    chunk_scores.extend(_score_records_to_window_scores(records))
                    for j, r in enumerate(records):
                        chunk_embeddings.append(
                            WindowEmbedding(
                                time_sec=float(r["offset_sec"]),
                                embedding=emb_arr[j].tolist(),
                            )
                        )
                    del audio_buf
            except FileNotFoundError:
                logger.info(
                    "region_detection | job=%s | chunk %d/%d | no audio segments, skipping",
                    job_id,
                    i + 1,
                    total_chunks,
                )

        elapsed = time.monotonic() - t0
        rate = elapsed / (chunk_duration / 60.0) if chunk_duration > 0 else 0.0

        write_chunk_trace(job_dir, i, chunk_scores)
        if chunk_embeddings:
            write_chunk_embeddings(job_dir, i, chunk_embeddings)

        above = sum(1 for ws in chunk_scores if ws.score >= config.high_threshold)
        now_utc = datetime.now(timezone.utc).isoformat()
        update_manifest_chunk(
            job_dir,
            i,
            {
                "status": "complete",
                "completed_at": now_utc,
                "trace_rows": len(chunk_scores),
                "windows_above_threshold": above,
                "elapsed_sec": round(elapsed, 1),
            },
        )

        refreshed = await session.get(RegionDetectionJob, job_id)
        if refreshed is not None:
            refreshed.chunks_completed = (refreshed.chunks_completed or 0) + 1
            refreshed.windows_detected = (refreshed.windows_detected or 0) + above
            refreshed.updated_at = datetime.now(timezone.utc)
            await session.commit()

        logger.info(
            "region_detection | job=%s | chunk %d/%d | scored %d windows (%d above threshold) | %.1fs (%.1fs/min audio)",
            job_id,
            i + 1,
            total_chunks,
            len(chunk_scores),
            above,
            elapsed,
            rate,
        )

    # Merge all chunks and deduplicate by time_sec (keep highest score).
    # Adjacent chunks may legitimately overlap by a few windows at
    # boundaries; dedup ensures a clean 1-row-per-timestamp trace.
    t0 = time.monotonic()
    raw_scores = read_all_chunk_traces(job_dir, total_chunks)
    raw_count = len(raw_scores)

    seen: dict[float, WindowScore] = {}
    for ws in raw_scores:
        existing = seen.get(ws.time_sec)
        if existing is None or ws.score > existing.score:
            seen[ws.time_sec] = ws
    all_scores = sorted(seen.values(), key=lambda ws: ws.time_sec)
    dup_count = raw_count - len(all_scores)

    # Deduplicate embeddings to match the trace survivor set.
    raw_emb = read_all_chunk_embeddings(job_dir, total_chunks)
    emb_lookup: dict[float, WindowEmbedding] = {}
    for we in raw_emb:
        if we.time_sec not in emb_lookup:
            emb_lookup[we.time_sec] = we
    all_embeddings = [
        emb_lookup[ws.time_sec] for ws in all_scores if ws.time_sec in emb_lookup
    ]

    merge_elapsed = time.monotonic() - t0

    if dup_count > 0:
        logger.warning(
            "region_detection | job=%s | merge | %d duplicate trace rows removed",
            job_id,
            dup_count,
        )
    logger.info(
        "region_detection | job=%s | merge | read %d trace rows (%d after dedup) | %.1fs",
        job_id,
        raw_count,
        len(all_scores),
        merge_elapsed,
    )

    return all_scores, all_embeddings, float(end_ts - start_ts)


async def run_region_detection_job(
    session: AsyncSession,
    job: RegionDetectionJob,
    settings: Settings,
) -> None:
    """Execute a Pass 1 region detection job end-to-end."""
    # Capture everything we might need after a rollback into locals, since
    # rollback expires attached instances and accessing the bound columns
    # afterwards triggers lazy refresh SQL that fails in async contexts.
    job_id = job.id
    classifier_model_id = job.classifier_model_id
    model_config_id = job.model_config_id
    config_json = job.config_json
    audio_file_id = job.audio_file_id
    hydrophone_id = job.hydrophone_id
    start_timestamp = job.start_timestamp
    end_timestamp = job.end_timestamp

    job_dir = ensure_dir(region_job_dir(settings.storage_root, job_id))
    try:
        if not classifier_model_id:
            raise ValueError("region detection job missing classifier_model_id")
        if not model_config_id:
            raise ValueError("region detection job missing model_config_id")
        if not config_json:
            raise ValueError("region detection job missing config_json")

        config = RegionDetectionConfig.model_validate_json(config_json)

        cm_result = await session.execute(
            select(ClassifierModel).where(ClassifierModel.id == classifier_model_id)
        )
        cm = cm_result.scalar_one_or_none()
        if cm is None:
            raise ValueError(f"ClassifierModel {classifier_model_id} not found")

        target_sample_rate = cm.target_sample_rate
        pipeline = joblib.load(cm.model_path)
        perch_model, input_format = await get_model_by_version(
            session, cm.model_version, settings
        )

        job.started_at = datetime.now(timezone.utc)
        await session.commit()

        window_embeddings: list[WindowEmbedding] = []

        if audio_file_id:
            af_result = await session.execute(
                select(AudioFile).where(AudioFile.id == audio_file_id)
            )
            audio_file = af_result.scalar_one_or_none()
            if audio_file is None:
                raise ValueError(f"AudioFile {audio_file_id} not found")
            trace_records, emb_arr, audio_duration_sec = await _load_file_trace(
                audio_file,
                config=config,
                perch_model=perch_model,
                classifier=pipeline,
                input_format=input_format,
                target_sample_rate=target_sample_rate,
                storage_root=settings.storage_root,
            )
            window_scores = _score_records_to_window_scores(trace_records)
            for j, r in enumerate(trace_records):
                window_embeddings.append(
                    WindowEmbedding(
                        time_sec=float(r["offset_sec"]),
                        embedding=emb_arr[j].tolist(),
                    )
                )
        else:
            if not hydrophone_id or start_timestamp is None or end_timestamp is None:
                raise ValueError(
                    "region detection job missing hydrophone source fields"
                )
            (
                window_scores,
                window_embeddings,
                audio_duration_sec,
            ) = await _load_hydrophone_trace(
                hydrophone_id,
                float(start_timestamp),
                float(end_timestamp),
                config=config,
                perch_model=perch_model,
                classifier=pipeline,
                input_format=input_format,
                target_sample_rate=target_sample_rate,
                settings=settings,
                session=session,
                job_id=job_id,
                job_dir=job_dir,
            )

        trace_dicts = [
            {
                "offset_sec": ws.time_sec,
                "end_sec": ws.time_sec + config.window_size_seconds,
                "confidence": ws.score,
            }
            for ws in window_scores
        ]
        events = merge_detection_events(
            trace_dicts,
            config.high_threshold,
            config.low_threshold,
        )
        regions = decode_regions(events, audio_duration_sec, config)

        write_trace(job_dir / "trace.parquet", window_scores)
        write_regions(job_dir / "regions.parquet", regions)
        if window_embeddings:
            write_embeddings(job_dir / "embeddings.parquet", window_embeddings)

        logger.info(
            "region_detection | job=%s | merge | %d trace rows -> %d regions",
            job_id,
            len(window_scores),
            len(regions),
        )

        now = datetime.now(timezone.utc)
        refreshed = await session.get(RegionDetectionJob, job_id)
        target = refreshed if refreshed is not None else job
        target.trace_row_count = len(window_scores)
        target.region_count = len(regions)
        target.completed_at = now
        target.updated_at = now
        target.status = "complete"
        await session.commit()

        logger.info(
            "region_detection | job=%s | complete | %d regions",
            job_id,
            len(regions),
        )

    except Exception as exc:
        logger.exception("Region detection job %s failed", job_id)
        _cleanup_partial_artifacts(job_dir)
        try:
            await session.rollback()
        except Exception:
            logger.debug("rollback failed", exc_info=True)
        try:
            refreshed = await session.get(RegionDetectionJob, job_id)
            if refreshed is not None:
                now = datetime.now(timezone.utc)
                refreshed.status = "failed"
                refreshed.error_message = str(exc) or type(exc).__name__
                refreshed.updated_at = now
                refreshed.completed_at = now
                await session.commit()
        except Exception:
            logger.exception("Failed to mark region detection job %s as failed", job_id)


async def run_one_iteration(
    session: AsyncSession, settings: Settings
) -> RegionDetectionJob | None:
    """Claim and process at most one region detection job. Returns it or None."""
    job = await claim_region_detection_job(session)
    if job is None:
        return None
    await run_region_detection_job(session, job, settings)
    return job
