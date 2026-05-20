"""SQLAlchemy model for Piano Roll MIDI export rows.

Tracks user-initiated MIDI export runs for Event Encoder Piano Roll Notes.
Idempotent on ``(event_encoder_job_id, extractor_version)``. The exported
``.mid`` artifact lives under
``<storage_root>/exports/event_encoders/{event_encoder_job_id}/notes_v{N}.mid``;
this row tracks the worker lifecycle and the on-disk file location.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import Float, ForeignKey, Integer, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from humpback.database import Base, TimestampMixin, UUIDMixin
from humpback.models.piano_roll_notes import DEFAULT_EXTRACTOR_VERSION
from humpback.models.processing import JobStatus

__all__ = ["PianoRollMidiExport"]


class PianoRollMidiExport(UUIDMixin, TimestampMixin, Base):
    """One MIDI export run for one Event Encoder job."""

    __tablename__ = "piano_roll_midi_exports"
    __table_args__ = (
        UniqueConstraint(
            "event_encoder_job_id",
            "extractor_version",
            name="uq_piano_roll_midi_exports_encoder_version",
        ),
    )

    event_encoder_job_id: Mapped[str] = mapped_column(
        ForeignKey("event_encoder_jobs.id", ondelete="CASCADE")
    )
    extractor_version: Mapped[str] = mapped_column(default=DEFAULT_EXTRACTOR_VERSION)
    status: Mapped[str] = mapped_column(default=JobStatus.queued.value)
    started_at: Mapped[Optional[datetime]] = mapped_column(default=None)
    finished_at: Mapped[Optional[datetime]] = mapped_column(default=None)
    error_message: Mapped[Optional[str]] = mapped_column(Text, default=None)
    midi_path: Mapped[Optional[str]] = mapped_column(Text, default=None)
    n_notes: Mapped[Optional[int]] = mapped_column(Integer, default=None)
    n_bytes: Mapped[Optional[int]] = mapped_column(Integer, default=None)
    compute_seconds: Mapped[Optional[float]] = mapped_column(Float, default=None)
    params_json: Mapped[str] = mapped_column(Text, default="{}")
