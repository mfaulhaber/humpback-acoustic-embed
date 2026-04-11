from typing import Optional

from sqlalchemy import Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from humpback.database import Base, TimestampMixin, UUIDMixin


class VocalizationType(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "vocalization_types"
    __table_args__ = (UniqueConstraint("name", name="uq_vocalization_types_name"),)

    name: Mapped[str]
    description: Mapped[Optional[str]] = mapped_column(Text, default=None)


class VocalizationClassifierModel(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "vocalization_models"

    name: Mapped[str]
    model_dir_path: Mapped[str]
    vocabulary_snapshot: Mapped[str] = mapped_column(Text)  # JSON array of type names
    per_class_thresholds: Mapped[str] = mapped_column(Text)  # JSON {type: threshold}
    per_class_metrics: Mapped[Optional[str]] = mapped_column(Text, default=None)  # JSON
    training_summary: Mapped[Optional[str]] = mapped_column(Text, default=None)  # JSON
    is_active: Mapped[bool] = mapped_column(default=False)
    training_dataset_id: Mapped[Optional[str]] = mapped_column(default=None)
    # Pass 3 (call parsing pipeline) coexistence axis. "sklearn_perch_embedding"
    # is the legacy multi-label-on-5s-windows family; "pytorch_event_cnn" is the
    # Pass 3 event-crop CNN family. input_mode: "detection_row" (5s window) vs
    # "segmented_event" (variable-length crop from Pass 2).
    model_family: Mapped[str] = mapped_column(default="sklearn_perch_embedding")
    input_mode: Mapped[str] = mapped_column(default="detection_row")


class VocalizationTrainingJob(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "vocalization_training_jobs"

    status: Mapped[str] = mapped_column(default="queued")
    source_config: Mapped[str] = mapped_column(Text)  # JSON
    parameters: Mapped[Optional[str]] = mapped_column(Text, default=None)  # JSON
    vocalization_model_id: Mapped[Optional[str]] = mapped_column(default=None)
    training_dataset_id: Mapped[Optional[str]] = mapped_column(default=None)
    result_summary: Mapped[Optional[str]] = mapped_column(Text, default=None)  # JSON
    error_message: Mapped[Optional[str]] = mapped_column(Text, default=None)
    # Same coexistence axis as VocalizationClassifierModel — a training job's
    # family determines which trainer code path runs when the worker claims it.
    model_family: Mapped[str] = mapped_column(default="sklearn_perch_embedding")
    input_mode: Mapped[str] = mapped_column(default="detection_row")


class VocalizationInferenceJob(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "vocalization_inference_jobs"

    status: Mapped[str] = mapped_column(default="queued")
    vocalization_model_id: Mapped[str]
    source_type: Mapped[str]  # "detection_job", "embedding_set", "rescore"
    source_id: Mapped[str]
    output_path: Mapped[Optional[str]] = mapped_column(default=None)
    result_summary: Mapped[Optional[str]] = mapped_column(Text, default=None)  # JSON
    error_message: Mapped[Optional[str]] = mapped_column(Text, default=None)
