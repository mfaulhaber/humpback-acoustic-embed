"""SQLAlchemy model for Piano Roll Notes worker rows.

Tracks MIDI-style note extraction runs that decorate Event Encoder jobs.
Idempotent on ``(event_encoder_job_id, extractor_version)``. The notes
themselves live in ``event_notes_v{N}.parquet`` under the Event Encoder's
job directory; this row tracks the worker lifecycle and produced artifact.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import Float, ForeignKey, Integer, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from humpback.database import Base, TimestampMixin, UUIDMixin
from humpback.models.processing import JobStatus

__all__ = ["PianoRollNotesJob", "DEFAULT_EXTRACTOR_VERSION"]


DEFAULT_EXTRACTOR_VERSION = "v4"


class PianoRollNotesJob(UUIDMixin, TimestampMixin, Base):
    """One MIDI-note extraction run for one Event Encoder job."""

    __tablename__ = "piano_roll_notes_jobs"
    __table_args__ = (
        UniqueConstraint(
            "event_encoder_job_id",
            "extractor_version",
            name="uq_piano_roll_notes_jobs_encoder_version",
        ),
    )

    event_encoder_job_id: Mapped[str] = mapped_column(
        ForeignKey("event_encoder_jobs.id")
    )
    extractor_version: Mapped[str] = mapped_column(default=DEFAULT_EXTRACTOR_VERSION)
    status: Mapped[str] = mapped_column(default=JobStatus.queued.value)
    started_at: Mapped[Optional[datetime]] = mapped_column(default=None)
    finished_at: Mapped[Optional[datetime]] = mapped_column(default=None)
    error_message: Mapped[Optional[str]] = mapped_column(Text, default=None)
    notes_path: Mapped[Optional[str]] = mapped_column(Text, default=None)
    n_events: Mapped[Optional[int]] = mapped_column(Integer, default=None)
    n_notes: Mapped[Optional[int]] = mapped_column(Integer, default=None)
    compute_seconds: Mapped[Optional[float]] = mapped_column(Float, default=None)
    params_json: Mapped[str] = mapped_column(Text, default="{}")
