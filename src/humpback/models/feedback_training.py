"""SQLAlchemy models for human-in-the-loop feedback training.

- ``RegionBoundaryCorrection`` — human corrections to Pass 1 region boundaries.
- ``EventClassifierTrainingJob`` — feedback training jobs for Pass 3.

Event boundary corrections and vocalization corrections are now handled by
unified models in ``models/call_parsing.py``.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Float, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from humpback.database import Base, TimestampMixin, UUIDMixin


class RegionBoundaryCorrection(UUIDMixin, TimestampMixin, Base):
    """Human correction to a Pass 1 region detection boundary."""

    __tablename__ = "region_boundary_corrections"
    __table_args__ = (
        UniqueConstraint(
            "region_detection_job_id",
            "region_id",
            name="uq_region_boundary_corrections_job_region",
        ),
    )

    region_detection_job_id: Mapped[str]
    region_id: Mapped[str]
    correction_type: Mapped[str]
    start_sec: Mapped[Optional[float]] = mapped_column(Float, default=None)
    end_sec: Mapped[Optional[float]] = mapped_column(Float, default=None)


class EventClassifierTrainingJob(UUIDMixin, TimestampMixin, Base):
    """Queued Pass 3 feedback training job sourcing from corrected classification output."""

    __tablename__ = "event_classifier_training_jobs"

    status: Mapped[str] = mapped_column(default="queued")
    source_job_ids: Mapped[str] = mapped_column(Text)
    config_json: Mapped[Optional[str]] = mapped_column(Text, default=None)
    vocalization_model_id: Mapped[Optional[str]] = mapped_column(default=None)
    result_summary: Mapped[Optional[str]] = mapped_column(Text, default=None)
    error_message: Mapped[Optional[str]] = mapped_column(Text, default=None)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, default=None)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, default=None)
