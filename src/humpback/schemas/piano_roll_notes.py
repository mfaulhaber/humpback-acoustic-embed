"""Pydantic schemas for Piano Roll Notes worker rows."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict


PianoRollNotesJobStatus = Literal["queued", "running", "complete", "failed", "canceled"]


class PianoRollNotesJobRead(BaseModel):
    """One Piano Roll Notes job row as exposed to API callers."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    event_encoder_job_id: str
    extractor_version: str
    status: PianoRollNotesJobStatus
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    error_message: Optional[str] = None
    notes_path: Optional[str] = None
    n_events: Optional[int] = None
    n_notes: Optional[int] = None
    compute_seconds: Optional[float] = None
    params_json: str = "{}"
    created_at: datetime
    updated_at: datetime


class PianoRollNotesJobCreateRequest(BaseModel):
    """Request body for enqueueing a Piano Roll Notes job."""

    extractor_version: Optional[str] = None
    params: Optional[dict[str, Any]] = None


class PianoRollNotesStatusAbsent(BaseModel):
    """Placeholder when no Piano Roll Notes job exists for the encoder job."""

    status: Literal["absent"] = "absent"


PianoRollNotesStatusResponse = Union[PianoRollNotesJobRead, PianoRollNotesStatusAbsent]
"""Timeline-piggyback shape exposed at ``GET .../notes-status``."""


__all__ = [
    "PianoRollNotesJobStatus",
    "PianoRollNotesJobRead",
    "PianoRollNotesJobCreateRequest",
    "PianoRollNotesStatusAbsent",
    "PianoRollNotesStatusResponse",
]
