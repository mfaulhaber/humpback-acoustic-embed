"""Worker that synthesizes Piano Roll Notes parquet sidecars into MIDI files.

The notes worker writes ``event_notes_v{N}.parquet``. This worker reads that
parquet, builds a Standard MIDI File via
``humpback.processing.midi_synthesis``, and persists the bytes under
``<storage_root>/exports/event_encoders/{job_id}/notes_v{N}.mid``.

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
from typing import Any, Optional

import pyarrow.parquet as pq
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.config import Settings
from humpback.models.piano_roll_midi_export import PianoRollMidiExport
from humpback.models.processing import JobStatus
from humpback.processing.midi_synthesis import notes_table_to_midi_bytes
from humpback.storage import (
    event_encoder_midi_export_path,
    event_encoder_notes_path,
)

logger = logging.getLogger(__name__)

_ERROR_MESSAGE_LIMIT = 2048


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


def _cleanup_partial_midi(path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    for candidate in (path, tmp):
        try:
            candidate.unlink(missing_ok=True)
        except OSError:
            logger.warning(
                "piano_roll_midi_export | could not remove partial artifact %s",
                candidate,
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


async def run_piano_roll_midi_export(
    session: AsyncSession,
    job: PianoRollMidiExport,
    settings: Settings,
) -> None:
    """Execute one Piano Roll MIDI export end-to-end."""
    job_id = job.id
    started_wall = time.monotonic()
    started_dt = datetime.now(timezone.utc)

    midi_path: Optional[Path] = None
    try:
        job = await session.merge(job)
        job.started_at = started_dt
        await session.commit()

        params = _resolve_params(job.params_json)

        notes_path = event_encoder_notes_path(
            settings.storage_root, job.event_encoder_job_id, job.extractor_version
        )
        if not notes_path.exists():
            raise FileNotFoundError(f"notes parquet not found at {notes_path}")

        notes_table = pq.read_table(notes_path)
        midi_bytes = notes_table_to_midi_bytes(notes_table)

        midi_path = event_encoder_midi_export_path(
            settings.storage_root, job.event_encoder_job_id, job.extractor_version
        )
        _atomic_write_bytes(midi_bytes, midi_path)

        compute_seconds = time.monotonic() - started_wall
        refreshed = await session.get(PianoRollMidiExport, job_id)
        target = refreshed if refreshed is not None else job
        target.status = JobStatus.complete.value
        target.finished_at = datetime.now(timezone.utc)
        target.error_message = None
        target.midi_path = _relative_to_storage(midi_path, settings.storage_root)
        target.n_notes = notes_table.num_rows
        target.n_bytes = len(midi_bytes)
        target.compute_seconds = compute_seconds
        target.params_json = json.dumps(params, sort_keys=True, separators=(",", ":"))
        await session.commit()

        logger.info(
            "piano_roll_midi_export | job=%s | complete | notes=%d bytes=%d secs=%.2f",
            job_id,
            notes_table.num_rows,
            len(midi_bytes),
            compute_seconds,
        )
    except Exception as exc:
        logger.exception("piano_roll_midi_export job %s failed", job_id)
        if midi_path is not None:
            _cleanup_partial_midi(midi_path)
        failed = await session.get(PianoRollMidiExport, job_id)
        target = failed if failed is not None else job
        target.status = JobStatus.failed.value
        target.finished_at = datetime.now(timezone.utc)
        target.error_message = _truncate(str(exc), limit=_ERROR_MESSAGE_LIMIT)
        target.midi_path = None
        target.compute_seconds = time.monotonic() - started_wall
        await session.commit()
