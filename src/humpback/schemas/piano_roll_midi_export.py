"""Pydantic schemas for Piano Roll MIDI export rows."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, model_validator


PianoRollMidiExportJobStatus = Literal[
    "queued", "running", "complete", "failed", "canceled"
]

MAX_EXPORT_WINDOW_SECONDS: float = 1800.0


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
    window_start_utc: float
    window_end_utc: float
    audio_path: str = ""
    audio_size_bytes: int = 0
    audio_sample_rate: int = 0
    audio_duration_s: float = 0.0
    created_at: datetime
    updated_at: datetime


class PianoRollMidiExportCreateRequest(BaseModel):
    """Request body for enqueueing a Piano Roll MIDI export.

    ``window_start_utc`` and ``window_end_utc`` are required UTC epoch
    seconds delimiting the viewer's exported region. The window must be
    strictly positive and no longer than ``MAX_EXPORT_WINDOW_SECONDS``.
    """

    extractor_version: Optional[str] = None
    params: Optional[dict[str, Any]] = None
    force: bool = False
    window_start_utc: float
    window_end_utc: float

    @model_validator(mode="after")
    def _validate_window(self) -> "PianoRollMidiExportCreateRequest":
        duration = self.window_end_utc - self.window_start_utc
        if duration <= 0.0:
            raise ValueError(
                "window_end_utc must be strictly greater than window_start_utc"
            )
        if duration > MAX_EXPORT_WINDOW_SECONDS:
            raise ValueError(
                "export window duration exceeds the "
                f"{MAX_EXPORT_WINDOW_SECONDS:.0f}-second cap"
            )
        return self


class PianoRollMidiExportStatusAbsent(BaseModel):
    """Placeholder when no Piano Roll MIDI export row exists yet."""

    status: Literal["absent"] = "absent"


PianoRollMidiExportStatusResponse = Union[
    PianoRollMidiExportRead, PianoRollMidiExportStatusAbsent
]


__all__ = [
    "MAX_EXPORT_WINDOW_SECONDS",
    "PianoRollMidiExportJobStatus",
    "PianoRollMidiExportRead",
    "PianoRollMidiExportCreateRequest",
    "PianoRollMidiExportStatusAbsent",
    "PianoRollMidiExportStatusResponse",
]
