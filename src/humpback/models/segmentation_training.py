"""SQLAlchemy models for Pass 2 segmentation training.

Three tables that together form the training contract for the Pass 2
framewise segmentation model:

- ``SegmentationTrainingDataset`` — top-level named training set.
- ``SegmentationTrainingSample`` — one audio crop + event bounds per row.
  Writable by both the one-shot bootstrap script and a future
  timeline-viewer event-bound editor.
- ``SegmentationTrainingJob`` — queued trainer runs that read a dataset
  and register a ``SegmentationModel`` row on success.

See migration ``044_segmentation_training_tables.py`` for the on-disk
schema and ``docs/specs/2026-04-11-call-parsing-pass2-segmentation-design.md``
for the design contract.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Float, Text
from sqlalchemy.orm import Mapped, mapped_column

from humpback.database import Base, TimestampMixin, UUIDMixin


class SegmentationTrainingDataset(UUIDMixin, TimestampMixin, Base):
    """Named container for Pass 2 training samples."""

    __tablename__ = "segmentation_training_datasets"

    name: Mapped[str]
    description: Mapped[Optional[str]] = mapped_column(Text, default=None)


class SegmentationTrainingSample(UUIDMixin, TimestampMixin, Base):
    """One audio crop with event-bound annotations for framewise training.

    Source identity mirrors the Pass 1 pattern — ``audio_file_id`` XOR
    the hydrophone triple. ``events_json`` is an audio-relative list of
    ``{"start_sec", "end_sec"}`` dicts; an empty list is a valid
    negative-only sample.
    """

    __tablename__ = "segmentation_training_samples"

    training_dataset_id: Mapped[str]
    audio_file_id: Mapped[Optional[str]] = mapped_column(default=None)
    hydrophone_id: Mapped[Optional[str]] = mapped_column(default=None)
    start_timestamp: Mapped[Optional[float]] = mapped_column(Float, default=None)
    end_timestamp: Mapped[Optional[float]] = mapped_column(Float, default=None)
    crop_start_sec: Mapped[float] = mapped_column(Float)
    crop_end_sec: Mapped[float] = mapped_column(Float)
    events_json: Mapped[str] = mapped_column(Text)
    source: Mapped[str]
    source_ref: Mapped[Optional[str]] = mapped_column(default=None)
    notes: Mapped[Optional[str]] = mapped_column(Text, default=None)


class SegmentationTrainingJob(UUIDMixin, TimestampMixin, Base):
    """Queued Pass 2 trainer job reading one dataset and producing one model."""

    __tablename__ = "segmentation_training_jobs"

    status: Mapped[str] = mapped_column(default="queued")
    training_dataset_id: Mapped[str]
    config_json: Mapped[str] = mapped_column(Text)
    segmentation_model_id: Mapped[Optional[str]] = mapped_column(default=None)
    result_summary: Mapped[Optional[str]] = mapped_column(Text, default=None)
    error_message: Mapped[Optional[str]] = mapped_column(Text, default=None)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, default=None)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, default=None)
