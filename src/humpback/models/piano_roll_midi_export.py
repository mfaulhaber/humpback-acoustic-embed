"""SQLAlchemy model for Piano Roll MIDI export rows.

Tracks user-initiated MIDI export runs for Event Encoder Piano Roll Notes.
Idempotent on ``(event_encoder_job_id, extractor_version)`` and scoped to a
single rolling window per (job, version): re-exporting overwrites the row's
window bounds and both on-disk artifacts.

Each row owns a bundled pair of artifacts under
``<storage_root>/exports/event_encoders/{event_encoder_job_id}/``:

* ``notes_{extractor_version}.mid`` — Standard MIDI File whose tick-0
  origin equals ``window_start_utc``.
* ``audio_{extractor_version}.flac`` — 32 kHz mono 16-bit PCM clip of the
  same window, not loudness-normalized so it matches what the piano-roll
  player rendered.

The row tracks the worker lifecycle and the on-disk locations.
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
    """One windowed bundled export (MIDI + FLAC) for one Event Encoder job."""

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
    window_start_utc: Mapped[float] = mapped_column(Float)
    window_end_utc: Mapped[float] = mapped_column(Float)
    audio_path: Mapped[str] = mapped_column(Text, default="")
    audio_size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    audio_sample_rate: Mapped[int] = mapped_column(Integer, default=0)
    audio_duration_s: Mapped[float] = mapped_column(Float, default=0.0)
