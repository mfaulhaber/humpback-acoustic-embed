"""Service layer for Piano Roll MIDI export jobs.

These rows track user-initiated bundled exports (MIDI + FLAC clip) that derive
from a completed Piano Roll Notes parquet sidecar. Idempotent on
``(event_encoder_job_id, extractor_version)`` with a single rolling window
per pair: matching the persisted window + ``complete`` is a cache hit;
window-mismatch or non-terminal status resets the row to ``queued``.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.models.piano_roll_midi_export import PianoRollMidiExport
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

# Floating-point tolerance for considering two requested windows identical.
_WINDOW_MATCH_TOLERANCE_S = 1e-3


class PianoRollMidiExportConflict(Exception):
    """Raised when an enqueue is attempted against an in-flight row."""


def _windows_match(a_start: float, a_end: float, b_start: float, b_end: float) -> bool:
    return (
        abs(a_start - b_start) <= _WINDOW_MATCH_TOLERANCE_S
        and abs(a_end - b_end) <= _WINDOW_MATCH_TOLERANCE_S
    )


async def enqueue_piano_roll_midi_export(
    session: AsyncSession,
    *,
    event_encoder_job_id: str,
    window_start_utc: float,
    window_end_utc: float,
    extractor_version: Optional[str] = None,
    params: Optional[dict[str, Any]] = None,
    force: bool = False,
) -> tuple[PianoRollMidiExport, bool]:
    """Create or reset a Piano Roll MIDI export row for ``window``.

    Returns ``(row, created)``. Behavior on existing rows for the same
    ``(event_encoder_job_id, extractor_version)``:

    - ``complete`` with a matching window and ``force=False`` — returned
      as-is (cache hit), ``created=False``.
    - ``complete`` with a mismatching window OR ``force=True`` — reset to
      ``queued`` with the new window, ``created=False``.
    - ``queued`` or ``running`` — raises ``PianoRollMidiExportConflict``.
    - ``failed`` or ``canceled`` — reset to ``queued`` with the new window,
      ``created=False``.

    Resolves ``extractor_version=None`` to the latest ``complete`` notes
    job's version. Raises ``ValueError`` if the event encoder does not
    exist, the referenced notes job is not yet ``complete``, or the
    window is non-positive.
    """
    if window_end_utc - window_start_utc <= 0.0:
        raise ValueError(
            "window_end_utc must be strictly greater than window_start_utc"
        )

    encoder = await session.get(EventEncoderJob, event_encoder_job_id)
    if encoder is None:
        raise ValueError(f"event_encoder_job not found: {event_encoder_job_id}")

    version = (
        extractor_version
        if extractor_version
        else await _latest_complete_notes_version(session, event_encoder_job_id)
    )
    if version is None:
        raise ValueError(
            "no complete piano_roll_notes_job exists for "
            f"event_encoder_job_id={event_encoder_job_id}"
        )

    notes = await _complete_notes_job(session, event_encoder_job_id, version)
    if notes is None:
        raise ValueError(
            "piano_roll_notes_job is not complete for "
            f"event_encoder_job_id={event_encoder_job_id} "
            f"extractor_version={version}"
        )

    params_payload = params or {}
    params_json = json.dumps(params_payload, sort_keys=True, separators=(",", ":"))

    existing = await _get_by_key(
        session,
        event_encoder_job_id=event_encoder_job_id,
        extractor_version=version,
    )

    if existing is not None:
        if existing.status == _TERMINAL_COMPLETE:
            window_matches = _windows_match(
                existing.window_start_utc,
                existing.window_end_utc,
                window_start_utc,
                window_end_utc,
            )
            if window_matches and not force:
                return existing, False
            return (
                await _reset_row(
                    session,
                    existing,
                    params_json=params_json,
                    window_start_utc=window_start_utc,
                    window_end_utc=window_end_utc,
                ),
                False,
            )
        if existing.status in _IN_FLIGHT_STATUSES:
            raise PianoRollMidiExportConflict(
                f"piano_roll_midi_export already {existing.status} for "
                f"event_encoder_job_id={event_encoder_job_id} "
                f"extractor_version={version}"
            )
        if existing.status in _RESETTABLE_STATUSES:
            return (
                await _reset_row(
                    session,
                    existing,
                    params_json=params_json,
                    window_start_utc=window_start_utc,
                    window_end_utc=window_end_utc,
                ),
                False,
            )
        raise PianoRollMidiExportConflict(
            f"piano_roll_midi_export has unexpected status={existing.status!r}"
        )

    row = PianoRollMidiExport(
        event_encoder_job_id=event_encoder_job_id,
        extractor_version=version,
        status=JobStatus.queued.value,
        params_json=params_json,
        window_start_utc=window_start_utc,
        window_end_utc=window_end_utc,
        audio_path="",
        audio_size_bytes=0,
        audio_sample_rate=0,
        audio_duration_s=0.0,
    )
    session.add(row)
    await session.commit()
    await session.refresh(row)
    return row, True


async def latest_for_encoder_job(
    session: AsyncSession,
    *,
    event_encoder_job_id: str,
) -> Optional[PianoRollMidiExport]:
    """Most-recent Piano Roll MIDI export row for an Event Encoder job.

    Prefers the most recent ``complete`` row, falling back to the most
    recently updated row in any state.
    """
    complete_q = await session.execute(
        select(PianoRollMidiExport)
        .where(
            PianoRollMidiExport.event_encoder_job_id == event_encoder_job_id,
            PianoRollMidiExport.status == _TERMINAL_COMPLETE,
        )
        .order_by(desc(PianoRollMidiExport.finished_at))
        .limit(1)
    )
    completed = complete_q.scalar_one_or_none()
    if completed is not None:
        return completed

    any_q = await session.execute(
        select(PianoRollMidiExport)
        .where(PianoRollMidiExport.event_encoder_job_id == event_encoder_job_id)
        .order_by(desc(PianoRollMidiExport.updated_at))
        .limit(1)
    )
    return any_q.scalar_one_or_none()


async def complete_for_encoder_job_version(
    session: AsyncSession,
    *,
    event_encoder_job_id: str,
    extractor_version: str,
) -> Optional[PianoRollMidiExport]:
    """Most-recent ``complete`` row pinned to a specific extractor version."""
    result = await session.execute(
        select(PianoRollMidiExport)
        .where(
            PianoRollMidiExport.event_encoder_job_id == event_encoder_job_id,
            PianoRollMidiExport.extractor_version == extractor_version,
            PianoRollMidiExport.status == _TERMINAL_COMPLETE,
        )
        .order_by(desc(PianoRollMidiExport.finished_at), desc(PianoRollMidiExport.id))
        .limit(1)
    )
    return result.scalar_one_or_none()


async def _reset_row(
    session: AsyncSession,
    row: PianoRollMidiExport,
    *,
    params_json: str,
    window_start_utc: float,
    window_end_utc: float,
) -> PianoRollMidiExport:
    now = datetime.now(timezone.utc)
    row.status = JobStatus.queued.value
    row.started_at = None
    row.finished_at = None
    row.error_message = None
    row.midi_path = None
    row.n_notes = None
    row.n_bytes = None
    row.compute_seconds = None
    row.params_json = params_json
    row.window_start_utc = window_start_utc
    row.window_end_utc = window_end_utc
    row.audio_path = ""
    row.audio_size_bytes = 0
    row.audio_sample_rate = 0
    row.audio_duration_s = 0.0
    row.updated_at = now
    await session.commit()
    await session.refresh(row)
    return row


async def _latest_complete_notes_version(
    session: AsyncSession, event_encoder_job_id: str
) -> Optional[str]:
    """Highest ``complete`` notes version for an encoder job (ADR-069 §10).

    Resolved by string-comparison on ``extractor_version`` so a v2 row
    that completed *after* v3 still defers to v3 — users wanting v2 must
    pin via ``extractor_version=`` explicitly. ``finished_at desc`` is a
    deterministic tiebreaker if two rows share the same version string
    (a re-run that lands at the same version). Lexicographic ordering
    assumes single-digit ``vN`` suffixes; a future ``v10`` would sort
    *before* ``v2`` and break this. Switch to integer-suffix parsing or
    zero-padded names (``v01``/``v02``/.../``v10``) before introducing a
    two-digit version.
    """
    result = await session.execute(
        select(PianoRollNotesJob.extractor_version)
        .where(
            PianoRollNotesJob.event_encoder_job_id == event_encoder_job_id,
            PianoRollNotesJob.status == _TERMINAL_COMPLETE,
        )
        .order_by(
            desc(PianoRollNotesJob.extractor_version),
            desc(PianoRollNotesJob.finished_at),
        )
        .limit(1)
    )
    return result.scalar_one_or_none() or None


async def _complete_notes_job(
    session: AsyncSession, event_encoder_job_id: str, extractor_version: str
) -> Optional[PianoRollNotesJob]:
    result = await session.execute(
        select(PianoRollNotesJob).where(
            PianoRollNotesJob.event_encoder_job_id == event_encoder_job_id,
            PianoRollNotesJob.extractor_version == extractor_version,
            PianoRollNotesJob.status == _TERMINAL_COMPLETE,
        )
    )
    return result.scalar_one_or_none()


async def _get_by_key(
    session: AsyncSession,
    *,
    event_encoder_job_id: str,
    extractor_version: str,
) -> Optional[PianoRollMidiExport]:
    result = await session.execute(
        select(PianoRollMidiExport).where(
            PianoRollMidiExport.event_encoder_job_id == event_encoder_job_id,
            PianoRollMidiExport.extractor_version == extractor_version,
        )
    )
    return result.scalar_one_or_none()


# Re-export for callers that want the default constant alongside the service.
__all__ = [
    "PianoRollMidiExportConflict",
    "enqueue_piano_roll_midi_export",
    "latest_for_encoder_job",
    "complete_for_encoder_job_version",
    "DEFAULT_EXTRACTOR_VERSION",
]
