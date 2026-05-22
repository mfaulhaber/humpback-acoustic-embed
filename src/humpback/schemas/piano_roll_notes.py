"""Pydantic schemas for Piano Roll Notes worker rows."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


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


class PianoRollNote(BaseModel):
    """One MIDI note row decoded from ``event_notes_{version}.parquet``.

    The trailing optional fields (``note_uid``, ``f0_track_id``,
    ``contour_frame_count``) are populated by v3 sidecars only; v1/v2
    rows leave them as ``None`` so legacy responses keep deserializing.
    """

    event_id: str
    event_token: int
    partial_index: int
    midi_pitch: int
    start_utc: float
    start_offset_s: float
    duration_s: float
    velocity: int
    peak_magnitude: float
    track_id: int
    note_uid: Optional[str] = None
    f0_track_id: Optional[int] = None
    contour_frame_count: Optional[int] = None


class PianoRollNotesResponse(BaseModel):
    """Filtered notes payload for the piano roll viewer."""

    job_id: str
    extractor_version: str
    n_notes: int
    notes: list[PianoRollNote] = Field(default_factory=list)


class PianoRollNoteContourFrame(BaseModel):
    """One frame row from ``event_note_contours_v3.parquet``."""

    frame_index: int
    time_offset_s: float
    cents_from_pitch: float
    harmonic_strength: float
    subharmonic_octave: int


class PianoRollNoteContourResponse(BaseModel):
    """Per-``note_uid`` contour payload for the v3 ribbon renderer.

    ``contours`` is a mapping from ``note_uid`` to its ascending-frame
    list. Unknown ``note_uid``s in the request are silently dropped from
    the response (partial misses are not errors).
    """

    job_id: str
    extractor_version: str
    n_notes: int
    contours: dict[str, list[PianoRollNoteContourFrame]] = Field(default_factory=dict)


class PianoRollNoteContourRequest(BaseModel):
    """POST body for the contours endpoint when callers prefer a body.

    Mirrors the repeated ``note_uids`` query-string form used elsewhere in
    this router so both forms are interchangeable.
    """

    note_uids: list[str] = Field(default_factory=list)


__all__ = [
    "PianoRollNotesJobStatus",
    "PianoRollNotesJobRead",
    "PianoRollNotesJobCreateRequest",
    "PianoRollNotesStatusAbsent",
    "PianoRollNotesStatusResponse",
    "PianoRollNote",
    "PianoRollNotesResponse",
    "PianoRollNoteContourFrame",
    "PianoRollNoteContourResponse",
    "PianoRollNoteContourRequest",
]
