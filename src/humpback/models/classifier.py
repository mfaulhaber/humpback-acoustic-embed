from typing import Optional

from sqlalchemy import Text
from sqlalchemy.orm import Mapped, mapped_column

from humpback.database import Base, TimestampMixin, UUIDMixin


class ClassifierModel(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "classifier_models"

    name: Mapped[str]
    model_path: Mapped[str]
    model_version: Mapped[str]
    vector_dim: Mapped[int]
    window_size_seconds: Mapped[float]
    target_sample_rate: Mapped[int]
    feature_config: Mapped[Optional[str]] = mapped_column(Text, default=None)
    training_summary: Mapped[Optional[str]] = mapped_column(Text, default=None)
    training_job_id: Mapped[Optional[str]] = mapped_column(default=None)


class ClassifierTrainingJob(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "classifier_training_jobs"

    status: Mapped[str] = mapped_column(default="queued")
    name: Mapped[str]
    positive_embedding_set_ids: Mapped[str] = mapped_column(Text)  # JSON array
    negative_audio_folder: Mapped[str]
    model_version: Mapped[str]
    window_size_seconds: Mapped[float]
    target_sample_rate: Mapped[int]
    feature_config: Mapped[Optional[str]] = mapped_column(Text, default=None)
    parameters: Mapped[Optional[str]] = mapped_column(Text, default=None)
    classifier_model_id: Mapped[Optional[str]] = mapped_column(default=None)
    error_message: Mapped[Optional[str]] = mapped_column(Text, default=None)


class DetectionJob(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "detection_jobs"

    status: Mapped[str] = mapped_column(default="queued")
    classifier_model_id: Mapped[str]
    audio_folder: Mapped[str]
    confidence_threshold: Mapped[float] = mapped_column(default=0.5)
    output_tsv_path: Mapped[Optional[str]] = mapped_column(default=None)
    result_summary: Mapped[Optional[str]] = mapped_column(Text, default=None)
    error_message: Mapped[Optional[str]] = mapped_column(Text, default=None)
