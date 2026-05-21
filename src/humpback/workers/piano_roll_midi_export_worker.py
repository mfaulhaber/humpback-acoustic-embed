"""Worker that bundles a Piano Roll Notes parquet into a windowed MIDI + FLAC pair.

The notes worker writes ``event_notes_v{N}.parquet``. This worker reads that
parquet, filters notes to the requested ``[window_start_utc, window_end_utc)``
range, synthesizes a Standard MIDI File via
``humpback.processing.midi_synthesis`` whose tick-0 origin is the window
start, and resolves the source audio for the same window into a
co-exported ``.flac`` clip. Both artifacts are written atomically and live
under ``<storage_root>/exports/event_encoders/{job_id}/``.

Lifecycle mirrors ``piano_roll_notes_worker``: claimed by the queue in
``queued``, transitioned to ``running`` on entry, ``complete`` on success, or
``failed`` on exception. Partial output is cleaned up on failure.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.config import Settings
from humpback.models.call_parsing import RegionDetectionJob
from humpback.models.piano_roll_midi_export import PianoRollMidiExport
from humpback.models.processing import JobStatus
from humpback.models.sequence_models import EventEncoderJob
from humpback.processing.audio_encoding import write_flac_clip
from humpback.processing.midi_synthesis import notes_table_to_midi_bytes
from humpback.processing.timeline_audio import resolve_timeline_audio
from humpback.storage import (
    event_encoder_audio_export_path,
    event_encoder_midi_export_path,
    event_encoder_notes_path,
)

logger = logging.getLogger(__name__)

_ERROR_MESSAGE_LIMIT = 2048
_MIN_NOTE_DURATION_S = 1e-3
EXPORT_SAMPLE_RATE = 32000


def _atomic_write_bytes(data: bytes, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    try:
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, dst)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise


def _cleanup_temp(path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.unlink(missing_ok=True)
    except OSError:
        logger.warning(
            "piano_roll_midi_export | could not remove partial artifact %s",
            tmp,
            exc_info=True,
        )


def _resolve_params(params_json: str) -> dict[str, Any]:
    if not params_json:
        return {}
    try:
        raw = json.loads(params_json)
    except json.JSONDecodeError:
        return {}
    return raw if isinstance(raw, dict) else {}


def _relative_to_storage(path: Path, storage_root: Path) -> str:
    try:
        return str(path.relative_to(storage_root))
    except ValueError:
        return str(path)


def _truncate(message: str, *, limit: int) -> str:
    if len(message) <= limit:
        return message
    return message[: limit - 3] + "..."


def _iter_windowed_note_rows(
    notes_table: pa.Table, window_start_utc: float, window_end_utc: float
) -> Iterator[dict[str, Any]]:
    """Yield per-note dicts overlapping the requested UTC window.

    Each yielded row is a shallow copy of the source row with ``start_utc``
    and ``duration_s`` clipped to ``[window_start_utc, window_end_utc)``.
    Notes whose clipped duration is below ``_MIN_NOTE_DURATION_S`` are
    dropped so the downstream MIDI synth never emits zero-tick events.
    """
    columns = notes_table.column_names
    pylists = {name: notes_table.column(name).to_pylist() for name in columns}
    row_count = notes_table.num_rows

    for i in range(row_count):
        start_raw = pylists["start_utc"][i]
        duration_raw = pylists["duration_s"][i]
        try:
            start = float(start_raw)
            duration = float(duration_raw)
        except (TypeError, ValueError):
            continue
        end = start + duration
        if start >= window_end_utc or end <= window_start_utc:
            continue
        clipped_start = max(start, window_start_utc)
        clipped_end = min(end, window_end_utc)
        if (clipped_end - clipped_start) < _MIN_NOTE_DURATION_S:
            continue
        row = {name: pylists[name][i] for name in columns}
        row["start_utc"] = clipped_start
        row["duration_s"] = clipped_end - clipped_start
        yield row


def _windowed_notes_table(
    notes_table: pa.Table, window_start_utc: float, window_end_utc: float
) -> pa.Table:
    rows = list(_iter_windowed_note_rows(notes_table, window_start_utc, window_end_utc))
    return pa.Table.from_pylist(rows, schema=notes_table.schema)


async def _resolve_region_job(
    session: AsyncSession, encoder: EventEncoderJob
) -> RegionDetectionJob:
    from humpback.models.call_parsing import EventSegmentationJob

    seg_job = await session.get(EventSegmentationJob, encoder.event_segmentation_job_id)
    if seg_job is None:
        raise ValueError(
            f"event_segmentation_job not found: {encoder.event_segmentation_job_id}"
        )
    region_job = await session.get(RegionDetectionJob, seg_job.region_detection_job_id)
    if region_job is None:
        raise ValueError(
            f"region_detection_job not found: {seg_job.region_detection_job_id}"
        )
    return region_job


def _resolve_window_audio(
    *,
    region_job: RegionDetectionJob,
    settings: Settings,
    window_start_utc: float,
    window_end_utc: float,
) -> np.ndarray:
    hydrophone_id = region_job.hydrophone_id or ""
    job_start = float(region_job.start_timestamp or 0.0)
    job_end = float(region_job.end_timestamp or 0.0)
    duration_sec = max(0.0, window_end_utc - window_start_utc)
    noaa_cache = str(settings.noaa_cache_path) if settings.noaa_cache_path else None
    audio = resolve_timeline_audio(
        hydrophone_id=hydrophone_id,
        local_cache_path=str(settings.s3_cache_path or ""),
        job_start_timestamp=job_start,
        job_end_timestamp=job_end,
        start_sec=window_start_utc,
        duration_sec=duration_sec,
        target_sr=EXPORT_SAMPLE_RATE,
        noaa_cache_path=noaa_cache,
    )
    return audio.astype(np.float32, copy=False)


async def run_piano_roll_midi_export(
    session: AsyncSession,
    job: PianoRollMidiExport,
    settings: Settings,
) -> None:
    """Execute one Piano Roll bundled (MIDI + FLAC) export end-to-end."""
    job_id = job.id
    started_wall = time.monotonic()
    started_dt = datetime.now(timezone.utc)

    midi_path: Optional[Path] = None
    audio_path: Optional[Path] = None
    try:
        job = await session.merge(job)
        job.started_at = started_dt
        await session.commit()

        window_start = float(job.window_start_utc)
        window_end = float(job.window_end_utc)
        if window_end - window_start <= 0.0:
            raise ValueError("piano_roll_midi_export row has non-positive window")

        params = _resolve_params(job.params_json)

        notes_path = event_encoder_notes_path(
            settings.storage_root, job.event_encoder_job_id, job.extractor_version
        )
        if not notes_path.exists():
            raise FileNotFoundError(f"notes parquet not found at {notes_path}")

        notes_table = pq.read_table(notes_path)
        windowed = _windowed_notes_table(notes_table, window_start, window_end)
        midi_bytes = notes_table_to_midi_bytes(windowed, time_origin_utc=window_start)

        encoder = await session.get(EventEncoderJob, job.event_encoder_job_id)
        if encoder is None:
            raise ValueError(f"event_encoder_job not found: {job.event_encoder_job_id}")
        region_job = await _resolve_region_job(session, encoder)
        audio_samples = _resolve_window_audio(
            region_job=region_job,
            settings=settings,
            window_start_utc=window_start,
            window_end_utc=window_end,
        )

        midi_path = event_encoder_midi_export_path(
            settings.storage_root, job.event_encoder_job_id, job.extractor_version
        )
        audio_path = event_encoder_audio_export_path(
            settings.storage_root, job.event_encoder_job_id, job.extractor_version
        )

        _atomic_write_bytes(midi_bytes, midi_path)
        try:
            write_flac_clip(audio_samples, EXPORT_SAMPLE_RATE, audio_path)
        except BaseException:
            # If the FLAC write fails after the MIDI rename succeeded, roll
            # the MIDI back so the on-disk pair stays consistent.
            try:
                midi_path.unlink(missing_ok=True)
            except OSError:
                logger.warning(
                    "piano_roll_midi_export | could not roll back partial MIDI %s",
                    midi_path,
                    exc_info=True,
                )
            raise

        audio_size_bytes = audio_path.stat().st_size
        audio_duration_s = float(audio_samples.size) / float(EXPORT_SAMPLE_RATE)

        compute_seconds = time.monotonic() - started_wall
        refreshed = await session.get(PianoRollMidiExport, job_id)
        target = refreshed if refreshed is not None else job
        target.status = JobStatus.complete.value
        target.finished_at = datetime.now(timezone.utc)
        target.error_message = None
        target.midi_path = _relative_to_storage(midi_path, settings.storage_root)
        target.audio_path = _relative_to_storage(audio_path, settings.storage_root)
        target.audio_size_bytes = int(audio_size_bytes)
        target.audio_sample_rate = EXPORT_SAMPLE_RATE
        target.audio_duration_s = audio_duration_s
        target.n_notes = windowed.num_rows
        target.n_bytes = len(midi_bytes)
        target.compute_seconds = compute_seconds
        target.params_json = json.dumps(params, sort_keys=True, separators=(",", ":"))
        await session.commit()

        logger.info(
            "piano_roll_midi_export | job=%s | complete | notes=%d "
            "midi_bytes=%d audio_bytes=%d audio_secs=%.2f secs=%.2f",
            job_id,
            windowed.num_rows,
            len(midi_bytes),
            audio_size_bytes,
            audio_duration_s,
            compute_seconds,
        )
    except Exception as exc:
        logger.exception("piano_roll_midi_export job %s failed", job_id)
        if midi_path is not None:
            _cleanup_temp(midi_path)
        if audio_path is not None:
            _cleanup_temp(audio_path)
        failed = await session.get(PianoRollMidiExport, job_id)
        target = failed if failed is not None else job
        target.status = JobStatus.failed.value
        target.finished_at = datetime.now(timezone.utc)
        target.error_message = _truncate(str(exc), limit=_ERROR_MESSAGE_LIMIT)
        target.compute_seconds = time.monotonic() - started_wall
        await session.commit()
