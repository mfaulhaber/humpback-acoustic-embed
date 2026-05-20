"""Pydantic schemas for Piano Roll MIDI export rows."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict


PianoRollMidiExportJobStatus = Literal[
    "queued", "running", "complete", "failed", "canceled"
]


class PianoRollMidiExportRead(BaseModel):
    """One Piano Roll MIDI export row as exposed to API callers."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    event_encoder_job_id: str
    extractor_version: str
    status: PianoRollMidiExportJobStatus
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    error_message: Optional[str] = None
    midi_path: Optional[str] = None
    n_notes: Optional[int] = None
    n_bytes: Optional[int] = None
    compute_seconds: Optional[float] = None
    params_json: str = "{}"
    created_at: datetime
    updated_at: datetime


class PianoRollMidiExportCreateRequest(BaseModel):
    """Request body for enqueueing a Piano Roll MIDI export."""

    extractor_version: Optional[str] = None
    params: Optional[dict[str, Any]] = None
    force: bool = False


class PianoRollMidiExportStatusAbsent(BaseModel):
    """Placeholder when no Piano Roll MIDI export row exists yet."""

    status: Literal["absent"] = "absent"


PianoRollMidiExportStatusResponse = Union[
    PianoRollMidiExportRead, PianoRollMidiExportStatusAbsent
]


__all__ = [
    "PianoRollMidiExportJobStatus",
    "PianoRollMidiExportRead",
    "PianoRollMidiExportCreateRequest",
    "PianoRollMidiExportStatusAbsent",
    "PianoRollMidiExportStatusResponse",
]
