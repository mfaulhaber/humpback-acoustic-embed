"""Service layer for Piano Roll Notes jobs.

These rows track MIDI-style note extraction runs that decorate completed
Event Encoder jobs. Idempotent on ``(event_encoder_job_id,
extractor_version)``: the same key resets a terminal row to ``queued`` and
refuses to disturb an in-flight one.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.models.piano_roll_notes import (
    DEFAULT_EXTRACTOR_VERSION,
    PianoRollNotesJob,
)
from humpback.models.processing import JobStatus
from humpback.models.sequence_models import EventEncoderJob

logger = logging.getLogger(__name__)


_RESETTABLE_STATUSES = {JobStatus.failed.value, JobStatus.canceled.value}
_TERMINAL_COMPLETE = JobStatus.complete.value
_IN_FLIGHT_STATUSES = {JobStatus.queued.value, JobStatus.running.value}


class PianoRollNotesJobConflict(Exception):
    """Raised when an enqueue is attempted against a non-resettable row."""


async def enqueue_piano_roll_notes_job(
    session: AsyncSession,
    *,
    event_encoder_job_id: str,
    extractor_version: Optional[str] = None,
    params: Optional[dict[str, Any]] = None,
) -> tuple[PianoRollNotesJob, bool]:
    """Create or reset a Piano Roll Notes job row.

    Returns ``(job, created)``. Behavior on existing rows for the same
    ``(event_encoder_job_id, extractor_version)``:

    - ``complete`` — returned as-is, ``created=False``.
    - ``queued`` or ``running`` — raises ``PianoRollNotesJobConflict``.
    - ``failed`` or ``canceled`` — reset to ``queued`` with cleared fields,
      ``created=False``.
    """
    encoder = await session.get(EventEncoderJob, event_encoder_job_id)
    if encoder is None:
        raise ValueError(f"event_encoder_job not found: {event_encoder_job_id}")

    version = extractor_version or DEFAULT_EXTRACTOR_VERSION
    params_payload = params or {}
    params_json = json.dumps(params_payload, sort_keys=True, separators=(",", ":"))

    existing = await _get_by_key(
        session,
        event_encoder_job_id=event_encoder_job_id,
        extractor_version=version,
    )

    if existing is not None:
        if existing.status == _TERMINAL_COMPLETE:
            return existing, False
        if existing.status in _IN_FLIGHT_STATUSES:
            raise PianoRollNotesJobConflict(
                f"piano_roll_notes_job already {existing.status} for "
                f"event_encoder_job_id={event_encoder_job_id} "
                f"extractor_version={version}"
            )
        if existing.status in _RESETTABLE_STATUSES:
            now = datetime.now(timezone.utc)
            existing.status = JobStatus.queued.value
            existing.started_at = None
            existing.finished_at = None
            existing.error_message = None
            existing.notes_path = None
            existing.n_events = None
            existing.n_notes = None
            existing.compute_seconds = None
            existing.params_json = params_json
            existing.updated_at = now
            await session.commit()
            await session.refresh(existing)
            return existing, False
        raise PianoRollNotesJobConflict(
            f"piano_roll_notes_job has unexpected status={existing.status!r}"
        )

    job = PianoRollNotesJob(
        event_encoder_job_id=event_encoder_job_id,
        extractor_version=version,
        status=JobStatus.queued.value,
        params_json=params_json,
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)
    return job, True


async def get_piano_roll_notes_job(
    session: AsyncSession, job_id: str
) -> Optional[PianoRollNotesJob]:
    return await session.get(PianoRollNotesJob, job_id)


async def latest_for_encoder_job(
    session: AsyncSession,
    *,
    event_encoder_job_id: str,
) -> Optional[PianoRollNotesJob]:
    """Most-recently-updated Piano Roll Notes job for an Event Encoder job.

    Prefers the most recent ``complete`` row when one exists, falling back to
    the most recent row in any state.
    """
    complete_q = await session.execute(
        select(PianoRollNotesJob)
        .where(
            PianoRollNotesJob.event_encoder_job_id == event_encoder_job_id,
            PianoRollNotesJob.status == _TERMINAL_COMPLETE,
        )
        .order_by(desc(PianoRollNotesJob.finished_at))
        .limit(1)
    )
    completed = complete_q.scalar_one_or_none()
    if completed is not None:
        return completed

    any_q = await session.execute(
        select(PianoRollNotesJob)
        .where(PianoRollNotesJob.event_encoder_job_id == event_encoder_job_id)
        .order_by(desc(PianoRollNotesJob.updated_at))
        .limit(1)
    )
    return any_q.scalar_one_or_none()


async def auto_enqueue_after_encoder_complete(
    session: AsyncSession,
    *,
    event_encoder_job_id: str,
) -> Optional[PianoRollNotesJob]:
    """Enqueue a notes job after an Event Encoder run completes.

    Best-effort: any failure is logged and swallowed so the caller's
    completion path is never blocked.
    """
    try:
        job, created = await enqueue_piano_roll_notes_job(
            session,
            event_encoder_job_id=event_encoder_job_id,
        )
        if created:
            logger.info(
                "piano_roll_notes | auto-enqueued | job=%s encoder=%s",
                job.id,
                event_encoder_job_id,
            )
        return job
    except PianoRollNotesJobConflict:
        # In-flight job already exists; nothing to do.
        return None
    except Exception:
        logger.exception(
            "piano_roll_notes | auto-enqueue failed for encoder=%s",
            event_encoder_job_id,
        )
        return None


async def _get_by_key(
    session: AsyncSession,
    *,
    event_encoder_job_id: str,
    extractor_version: str,
) -> Optional[PianoRollNotesJob]:
    result = await session.execute(
        select(PianoRollNotesJob).where(
            PianoRollNotesJob.event_encoder_job_id == event_encoder_job_id,
            PianoRollNotesJob.extractor_version == extractor_version,
        )
    )
    return result.scalar_one_or_none()


__all__ = [
    "PianoRollNotesJobConflict",
    "enqueue_piano_roll_notes_job",
    "get_piano_roll_notes_job",
    "latest_for_encoder_job",
    "auto_enqueue_after_encoder_complete",
]
