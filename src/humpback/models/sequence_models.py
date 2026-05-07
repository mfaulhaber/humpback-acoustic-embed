"""SQLAlchemy models for retained Sequence Models / Continuous Embedding jobs."""

from typing import Optional

from sqlalchemy import Float, Integer, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from humpback.database import Base, TimestampMixin, UUIDMixin
from humpback.models.processing import JobStatus

__all__ = [
    "ContinuousEmbeddingJob",
    "EventEncoderJob",
    "JobStatus",
]


class ContinuousEmbeddingJob(UUIDMixin, TimestampMixin, Base):
    """Continuous embedding producer.

    SurfPerch jobs emit event-scoped, padded windows. CRNN jobs emit
    region-scoped chunks. Both source families are retained under the
    Continuous Embedding track and remain idempotent on ``encoding_signature``.
    """

    __tablename__ = "continuous_embedding_jobs"
    __table_args__ = (UniqueConstraint("encoding_signature"),)

    status: Mapped[str] = mapped_column(default=JobStatus.queued.value)
    event_segmentation_job_id: Mapped[Optional[str]] = mapped_column(default=None)
    event_source_mode: Mapped[str] = mapped_column(default="raw")
    model_version: Mapped[str]
    window_size_seconds: Mapped[Optional[float]] = mapped_column(Float, default=None)
    hop_seconds: Mapped[Optional[float]] = mapped_column(Float, default=None)
    pad_seconds: Mapped[Optional[float]] = mapped_column(Float, default=None)
    target_sample_rate: Mapped[int] = mapped_column(Integer)
    feature_config_json: Mapped[Optional[str]] = mapped_column(Text, default=None)
    encoding_signature: Mapped[str]
    vector_dim: Mapped[Optional[int]] = mapped_column(Integer, default=None)
    total_events: Mapped[Optional[int]] = mapped_column(Integer, default=None)
    merged_spans: Mapped[Optional[int]] = mapped_column(Integer, default=None)
    total_windows: Mapped[Optional[int]] = mapped_column(Integer, default=None)
    parquet_path: Mapped[Optional[str]] = mapped_column(default=None)
    error_message: Mapped[Optional[str]] = mapped_column(Text, default=None)
    # CRNN region-based embedding source columns.
    region_detection_job_id: Mapped[Optional[str]] = mapped_column(default=None)
    chunk_size_seconds: Mapped[Optional[float]] = mapped_column(Float, default=None)
    chunk_hop_seconds: Mapped[Optional[float]] = mapped_column(Float, default=None)
    crnn_checkpoint_sha256: Mapped[Optional[str]] = mapped_column(Text, default=None)
    crnn_segmentation_model_id: Mapped[Optional[str]] = mapped_column(default=None)
    projection_kind: Mapped[Optional[str]] = mapped_column(Text, default=None)
    projection_dim: Mapped[Optional[int]] = mapped_column(Integer, default=None)
    total_regions: Mapped[Optional[int]] = mapped_column(Integer, default=None)
    total_chunks: Mapped[Optional[int]] = mapped_column(Integer, default=None)


class EventEncoderJob(UUIDMixin, TimestampMixin, Base):
    """Event-level CRNN tokenization job.

    Consumes one completed Pass 2 segmentation job and one matching CRNN
    Continuous Embedding job, then writes per-event vectors, k-means token
    assignments, token sequences, and a report. The retained idempotency key is
    ``tokenization_signature``.
    """

    __tablename__ = "event_encoder_jobs"
    __table_args__ = (UniqueConstraint("tokenization_signature"),)

    status: Mapped[str] = mapped_column(default=JobStatus.queued.value)
    event_segmentation_job_id: Mapped[str]
    event_source_mode: Mapped[str] = mapped_column(default="raw")
    continuous_embedding_job_id: Mapped[str]
    continuous_embedding_signature: Mapped[str] = mapped_column(Text)
    tokenizer_version: Mapped[str] = mapped_column(
        Text, default="crnn-event-encoder-v1"
    )
    pooling_config_json: Mapped[str] = mapped_column(Text)
    descriptor_config_json: Mapped[str] = mapped_column(Text)
    preprocessing_config_json: Mapped[str] = mapped_column(Text)
    k_values_json: Mapped[str] = mapped_column(Text)
    random_seed: Mapped[int] = mapped_column(Integer, default=0)
    tokenization_signature: Mapped[str]
    event_vector_dim: Mapped[Optional[int]] = mapped_column(Integer, default=None)
    total_events: Mapped[Optional[int]] = mapped_column(Integer, default=None)
    encoded_events: Mapped[Optional[int]] = mapped_column(Integer, default=None)
    skipped_events: Mapped[Optional[int]] = mapped_column(Integer, default=None)
    event_vectors_path: Mapped[Optional[str]] = mapped_column(Text, default=None)
    event_tokens_path: Mapped[Optional[str]] = mapped_column(Text, default=None)
    token_sequences_path: Mapped[Optional[str]] = mapped_column(Text, default=None)
    manifest_path: Mapped[Optional[str]] = mapped_column(Text, default=None)
    report_path: Mapped[Optional[str]] = mapped_column(Text, default=None)
    error_message: Mapped[Optional[str]] = mapped_column(Text, default=None)
