"""SQLAlchemy models for the call parsing pipeline (Phase 0 scaffold).

Four-pass humpback call parsing pipeline: detect → segment → classify →
export. See ``docs/specs/2026-04-11-call-parsing-pipeline-phase0-design.md``
for the architecture contract.

Phase 0 ships the table skeleton and worker shells; subsequent passes
each brainstorm and implement their own internal logic against this
contract.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Float, Integer, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from humpback.database import Base, TimestampMixin, UUIDMixin


class CallParsingRun(UUIDMixin, TimestampMixin, Base):
    """Parent row threading one end-to-end pipeline run across the four passes.

    The nullable child job FKs are populated by the orchestration service
    as each pass is queued. Child jobs created as part of a parent run
    carry this row's id in their ``parent_run_id`` column; standalone
    child jobs leave ``parent_run_id`` NULL.

    Source identity is carried as ``audio_file_id`` XOR the hydrophone
    triple (``hydrophone_id`` + ``start_timestamp`` + ``end_timestamp``).
    The exactly-one-of invariant is enforced by the service layer and the
    Pydantic request model, not by a DB CHECK constraint.
    """

    __tablename__ = "call_parsing_runs"

    status: Mapped[str] = mapped_column(default="queued")
    audio_file_id: Mapped[Optional[str]] = mapped_column(default=None)
    hydrophone_id: Mapped[Optional[str]] = mapped_column(default=None)
    start_timestamp: Mapped[Optional[float]] = mapped_column(default=None)
    end_timestamp: Mapped[Optional[float]] = mapped_column(default=None)
    config_snapshot: Mapped[Optional[str]] = mapped_column(Text, default=None)
    region_detection_job_id: Mapped[Optional[str]] = mapped_column(default=None)
    event_segmentation_job_id: Mapped[Optional[str]] = mapped_column(default=None)
    event_classification_job_id: Mapped[Optional[str]] = mapped_column(default=None)
    error_message: Mapped[Optional[str]] = mapped_column(Text, default=None)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, default=None)


class SegmentationModel(UUIDMixin, Base):
    """Pass 2 PyTorch segmentation model checkpoint registry.

    Distinct from ``vocalization_models`` because framewise segmentation is
    a different task type from per-event multi-label classification. Pass 3
    lives under the existing ``vocalization_models`` table (extended with
    ``model_family`` / ``input_mode`` columns in migration 042).
    """

    __tablename__ = "segmentation_models"

    name: Mapped[str]
    model_family: Mapped[str]
    model_path: Mapped[str]
    config_json: Mapped[Optional[str]] = mapped_column(Text, default=None)
    training_job_id: Mapped[Optional[str]] = mapped_column(default=None)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.utcnow()
    )


class RegionDetectionJob(UUIDMixin, TimestampMixin, Base):
    """Pass 1 — dense Perch inference + hysteresis → padded regions.

    The worker writes ``trace.parquet`` (dense per-window scores) and
    ``regions.parquet`` (padded continuous whale-active regions) to the
    per-job storage directory. Phase 0 ships an empty shell that claims
    and fails; Pass 1 implements the actual logic.

    Source identity mirrors ``CallParsingRun``: ``audio_file_id`` XOR the
    hydrophone triple, enforced by the service layer and Pydantic request
    model rather than a DB CHECK constraint.
    """

    __tablename__ = "region_detection_jobs"

    status: Mapped[str] = mapped_column(default="queued")
    parent_run_id: Mapped[Optional[str]] = mapped_column(default=None)
    audio_file_id: Mapped[Optional[str]] = mapped_column(default=None)
    hydrophone_id: Mapped[Optional[str]] = mapped_column(default=None)
    start_timestamp: Mapped[Optional[float]] = mapped_column(default=None)
    end_timestamp: Mapped[Optional[float]] = mapped_column(default=None)
    model_config_id: Mapped[Optional[str]] = mapped_column(default=None)
    classifier_model_id: Mapped[Optional[str]] = mapped_column(default=None)
    config_json: Mapped[Optional[str]] = mapped_column(Text, default=None)
    chunks_total: Mapped[Optional[int]] = mapped_column(Integer, default=None)
    chunks_completed: Mapped[Optional[int]] = mapped_column(Integer, default=None)
    windows_detected: Mapped[Optional[int]] = mapped_column(Integer, default=None)
    trace_row_count: Mapped[Optional[int]] = mapped_column(Integer, default=None)
    region_count: Mapped[Optional[int]] = mapped_column(Integer, default=None)
    error_message: Mapped[Optional[str]] = mapped_column(Text, default=None)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, default=None)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, default=None)


class EventSegmentationJob(UUIDMixin, TimestampMixin, Base):
    """Pass 2 — framewise segmentation over regions produces events.

    Consumes ``regions.parquet`` produced by a completed
    ``RegionDetectionJob`` and writes ``events.parquet``. Phase 0 ships an
    empty shell; Pass 2 introduces the PyTorch CRNN/transformer model and
    event decoding.
    """

    __tablename__ = "event_segmentation_jobs"

    status: Mapped[str] = mapped_column(default="queued")
    parent_run_id: Mapped[Optional[str]] = mapped_column(default=None)
    region_detection_job_id: Mapped[str]
    segmentation_model_id: Mapped[Optional[str]] = mapped_column(default=None)
    config_json: Mapped[Optional[str]] = mapped_column(Text, default=None)
    event_count: Mapped[Optional[int]] = mapped_column(Integer, default=None)
    compute_device: Mapped[Optional[str]] = mapped_column(Text, default=None)
    gpu_fallback_reason: Mapped[Optional[str]] = mapped_column(Text, default=None)
    error_message: Mapped[Optional[str]] = mapped_column(Text, default=None)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, default=None)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, default=None)


class EventClassificationJob(UUIDMixin, TimestampMixin, Base):
    """Pass 3 — per-event multi-label call-type classification.

    Consumes ``events.parquet`` from a completed ``EventSegmentationJob``
    and writes ``typed_events.parquet``. Pass 3 reuses the existing
    ``vocalization_models`` table via the new ``model_family`` column
    (``pytorch_event_cnn``).
    """

    __tablename__ = "event_classification_jobs"

    status: Mapped[str] = mapped_column(default="queued")
    parent_run_id: Mapped[Optional[str]] = mapped_column(default=None)
    event_segmentation_job_id: Mapped[str]
    vocalization_model_id: Mapped[Optional[str]] = mapped_column(default=None)
    config_json: Mapped[Optional[str]] = mapped_column(Text, default=None)
    typed_event_count: Mapped[Optional[int]] = mapped_column(Integer, default=None)
    compute_device: Mapped[Optional[str]] = mapped_column(Text, default=None)
    gpu_fallback_reason: Mapped[Optional[str]] = mapped_column(Text, default=None)
    error_message: Mapped[Optional[str]] = mapped_column(Text, default=None)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, default=None)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, default=None)


class WindowClassificationJob(UUIDMixin, TimestampMixin, Base):
    """Standalone sidecar: score cached Perch embeddings from Pass 1 regions
    through an existing multi-label vocalization classifier.

    Not numbered as a pass — runs independently against a completed
    ``RegionDetectionJob`` and has no FK on ``CallParsingRun``.
    """

    __tablename__ = "window_classification_jobs"

    status: Mapped[str] = mapped_column(default="queued")
    region_detection_job_id: Mapped[str]
    vocalization_model_id: Mapped[str]
    config_json: Mapped[Optional[str]] = mapped_column(Text, default=None)
    window_count: Mapped[Optional[int]] = mapped_column(Integer, default=None)
    vocabulary_snapshot: Mapped[Optional[str]] = mapped_column(Text, default=None)
    error_message: Mapped[Optional[str]] = mapped_column(Text, default=None)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, default=None)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, default=None)


class VocalizationCorrection(UUIDMixin, TimestampMixin, Base):
    """Unified human correction for vocalization presence/absence.

    Keyed by ``(region_detection_job_id, start_sec, end_sec, type_name)``
    — one row per vocalization type per event time range. Used by both
    Window Classify review and Classify review surfaces.
    """

    __tablename__ = "vocalization_corrections"
    __table_args__ = (
        UniqueConstraint(
            "region_detection_job_id",
            "start_sec",
            "end_sec",
            "type_name",
            name="uq_vocalization_corrections_job_time_type",
        ),
    )

    region_detection_job_id: Mapped[str]
    start_sec: Mapped[float] = mapped_column(Float)
    end_sec: Mapped[float] = mapped_column(Float)
    type_name: Mapped[str]
    correction_type: Mapped[str]


class EventBoundaryCorrection(UUIDMixin, TimestampMixin, Base):
    """Unified human correction to event boundaries.

    Anchored to ``region_detection_job_id`` (Pass 1) so corrections are
    shared across Pass 2, Pass 3, and Window Classify review surfaces.
    Uses explicit original/corrected time pairs for time-range identity.
    """

    __tablename__ = "event_boundary_corrections"

    region_detection_job_id: Mapped[str]
    region_id: Mapped[str]
    correction_type: Mapped[str]
    original_start_sec: Mapped[Optional[float]] = mapped_column(Float, default=None)
    original_end_sec: Mapped[Optional[float]] = mapped_column(Float, default=None)
    corrected_start_sec: Mapped[Optional[float]] = mapped_column(Float, default=None)
    corrected_end_sec: Mapped[Optional[float]] = mapped_column(Float, default=None)
