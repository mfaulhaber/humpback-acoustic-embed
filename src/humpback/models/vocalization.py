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


class VocalizationTrainingJob(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "vocalization_training_jobs"

    status: Mapped[str] = mapped_column(default="queued")
    source_config: Mapped[str] = mapped_column(Text)  # JSON
    parameters: Mapped[Optional[str]] = mapped_column(Text, default=None)  # JSON
    vocalization_model_id: Mapped[Optional[str]] = mapped_column(default=None)
    result_summary: Mapped[Optional[str]] = mapped_column(Text, default=None)  # JSON
    error_message: Mapped[Optional[str]] = mapped_column(Text, default=None)


class VocalizationInferenceJob(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "vocalization_inference_jobs"

    status: Mapped[str] = mapped_column(default="queued")
    vocalization_model_id: Mapped[str]
    source_type: Mapped[str]  # "detection_job", "embedding_set", "rescore"
    source_id: Mapped[str]
    output_path: Mapped[Optional[str]] = mapped_column(default=None)
    result_summary: Mapped[Optional[str]] = mapped_column(Text, default=None)  # JSON
    error_message: Mapped[Optional[str]] = mapped_column(Text, default=None)
