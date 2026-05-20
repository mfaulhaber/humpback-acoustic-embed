"""Worker for Piano Roll Notes jobs.

Phase A scaffold: claims a queued job, writes an empty
``event_notes_v{N}.parquet`` sidecar under the Event Encoder job directory,
and marks the row ``complete``. Phase B replaces the body of
``_extract_notes`` with the real CQT + tracker + harmonic-prior pipeline.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.config import Settings
from humpback.models.piano_roll_notes import PianoRollNotesJob
from humpback.models.processing import JobStatus
from humpback.models.sequence_models import EventEncoderJob
from humpback.storage import ensure_dir, event_encoder_dir, event_encoder_notes_path

logger = logging.getLogger(__name__)


NOTES_SCHEMA = pa.schema(
    [
        pa.field("event_id", pa.string(), nullable=False),
        pa.field("event_token", pa.int32(), nullable=False),
        pa.field("partial_index", pa.int32(), nullable=False),
        pa.field("midi_pitch", pa.uint8(), nullable=False),
        pa.field("start_utc", pa.float64(), nullable=False),
        pa.field("start_offset_s", pa.float64(), nullable=False),
        pa.field("duration_s", pa.float64(), nullable=False),
        pa.field("velocity", pa.uint8(), nullable=False),
        pa.field("peak_magnitude", pa.float32(), nullable=False),
        pa.field("track_id", pa.uint32(), nullable=False),
    ]
)


def _atomic_write_parquet(table: pa.Table, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    try:
        pq.write_table(table, tmp)
        os.replace(tmp, dst)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise


async def run_piano_roll_notes_job(
    session: AsyncSession,
    job: PianoRollNotesJob,
    settings: Settings,
) -> None:
    """Execute one Piano Roll Notes job.

    The worker is claimed in ``running`` by the queue helper. This routine
    runs extraction (currently a stub that emits zero rows) and transitions
    the row to ``complete`` or ``failed``.
    """
    job_id = job.id
    started_wall = time.monotonic()
    started_dt = datetime.now(timezone.utc)

    try:
        job = await session.merge(job)
        job.started_at = started_dt
        await session.commit()

        encoder = await session.get(EventEncoderJob, job.event_encoder_job_id)
        if encoder is None:
            raise ValueError(f"event_encoder_job not found: {job.event_encoder_job_id}")
        if encoder.status != JobStatus.complete.value:
            raise ValueError(
                "piano roll notes requires a completed event_encoder_job "
                f"(status={encoder.status!r})"
            )

        params = _resolve_params(job.params_json)
        notes_path = event_encoder_notes_path(
            settings.storage_root, encoder.id, job.extractor_version
        )
        ensure_dir(event_encoder_dir(settings.storage_root, encoder.id))

        rows, n_events = await _extract_notes(session, encoder, params, settings)
        table = pa.Table.from_pylist(rows, schema=NOTES_SCHEMA)
        _atomic_write_parquet(table, notes_path)

        compute_seconds = time.monotonic() - started_wall
        refreshed = await session.get(PianoRollNotesJob, job_id)
        target = refreshed if refreshed is not None else job
        target.status = JobStatus.complete.value
        target.finished_at = datetime.now(timezone.utc)
        target.error_message = None
        target.notes_path = str(notes_path)
        target.n_events = n_events
        target.n_notes = len(rows)
        target.compute_seconds = compute_seconds
        target.params_json = json.dumps(params, sort_keys=True, separators=(",", ":"))
        await session.commit()

        logger.info(
            "piano_roll_notes | job=%s | complete | events=%d notes=%d secs=%.2f",
            job_id,
            n_events,
            len(rows),
            compute_seconds,
        )
    except Exception as exc:
        logger.exception("piano_roll_notes job %s failed", job_id)
        failed = await session.get(PianoRollNotesJob, job_id)
        target = failed if failed is not None else job
        target.status = JobStatus.failed.value
        target.finished_at = datetime.now(timezone.utc)
        target.error_message = _truncate(str(exc), limit=2048)
        target.notes_path = None
        target.compute_seconds = time.monotonic() - started_wall
        await session.commit()


def _resolve_params(params_json: str) -> dict[str, Any]:
    if not params_json:
        return {}
    try:
        loaded = json.loads(params_json)
    except json.JSONDecodeError:
        return {}
    if not isinstance(loaded, dict):
        return {}
    return loaded


def _truncate(text: str, *, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


async def _extract_notes(
    session: AsyncSession,
    encoder: EventEncoderJob,
    params: dict[str, Any],
    settings: Settings,
) -> tuple[list[dict[str, Any]], int]:
    """Phase A stub — yields no notes.

    Phase B replaces this with the CQT + peak-pick + tracker + harmonic
    prior pipeline. The return contract stays ``(rows, n_events_scanned)``.
    """
    del session, encoder, params, settings
    return [], 0


__all__ = ["run_piano_roll_notes_job", "NOTES_SCHEMA"]
