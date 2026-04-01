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
    classifier_purpose: Mapped[str] = mapped_column(default="detection")


class ClassifierTrainingJob(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "classifier_training_jobs"

    status: Mapped[str] = mapped_column(default="queued")
    name: Mapped[str]
    positive_embedding_set_ids: Mapped[str] = mapped_column(Text)  # JSON array
    negative_embedding_set_ids: Mapped[str] = mapped_column(Text)  # JSON array
    model_version: Mapped[str]
    window_size_seconds: Mapped[float]
    target_sample_rate: Mapped[int]
    feature_config: Mapped[Optional[str]] = mapped_column(Text, default=None)
    parameters: Mapped[Optional[str]] = mapped_column(Text, default=None)
    classifier_model_id: Mapped[Optional[str]] = mapped_column(default=None)
    error_message: Mapped[Optional[str]] = mapped_column(Text, default=None)
    job_purpose: Mapped[str] = mapped_column(default="detection")
    source_detection_job_ids: Mapped[Optional[str]] = mapped_column(Text, default=None)


class DetectionJob(UUIDMixin, TimestampMixin, Base):
    __tablename__ = "detection_jobs"

    status: Mapped[str] = mapped_column(default="queued")
    classifier_model_id: Mapped[str]
    audio_folder: Mapped[Optional[str]] = mapped_column(default=None)
    confidence_threshold: Mapped[float] = mapped_column(default=0.5)
    hop_seconds: Mapped[float] = mapped_column(default=1.0)
    high_threshold: Mapped[float] = mapped_column(default=0.70)
    low_threshold: Mapped[float] = mapped_column(default=0.45)
    detection_mode: Mapped[Optional[str]] = mapped_column(default=None)
    output_row_store_path: Mapped[Optional[str]] = mapped_column(default=None)
    result_summary: Mapped[Optional[str]] = mapped_column(Text, default=None)
    error_message: Mapped[Optional[str]] = mapped_column(Text, default=None)
    files_processed: Mapped[Optional[int]] = mapped_column(default=None)
    files_total: Mapped[Optional[int]] = mapped_column(default=None)
    extract_status: Mapped[Optional[str]] = mapped_column(default=None)
    extract_error: Mapped[Optional[str]] = mapped_column(Text, default=None)
    extract_summary: Mapped[Optional[str]] = mapped_column(Text, default=None)
    extract_config: Mapped[Optional[str]] = mapped_column(Text, default=None)
    # Hydrophone detection fields
    hydrophone_id: Mapped[Optional[str]] = mapped_column(default=None)
    hydrophone_name: Mapped[Optional[str]] = mapped_column(default=None)
    start_timestamp: Mapped[Optional[float]] = mapped_column(default=None)
    end_timestamp: Mapped[Optional[float]] = mapped_column(default=None)
    segments_processed: Mapped[Optional[int]] = mapped_column(default=None)
    segments_total: Mapped[Optional[int]] = mapped_column(default=None)
    time_covered_sec: Mapped[Optional[float]] = mapped_column(default=None)
    alerts: Mapped[Optional[str]] = mapped_column(Text, default=None)
    local_cache_path: Mapped[Optional[str]] = mapped_column(default=None)
    has_positive_labels: Mapped[Optional[bool]] = mapped_column(default=None)
    timeline_tiles_ready: Mapped[bool] = mapped_column(default=False)
    row_store_version: Mapped[int] = mapped_column(default=1)
