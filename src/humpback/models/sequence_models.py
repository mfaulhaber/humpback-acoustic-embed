"""SQLAlchemy models for the Sequence Models track.

``ContinuousEmbeddingJob``: an event-scoped, hydrophone-only producer that
emits 1-second-hop SurfPerch embeddings padded around Pass-2 segmentation
events. Idempotent on ``encoding_signature``.
"""

from typing import Optional

from sqlalchemy import Boolean, Float, Integer, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from humpback.database import Base, TimestampMixin, UUIDMixin
from humpback.models.processing import JobStatus

__all__ = ["ContinuousEmbeddingJob", "HMMSequenceJob", "JobStatus"]


class ContinuousEmbeddingJob(UUIDMixin, TimestampMixin, Base):
    """Continuous embedding producer.

    Given a completed ``EventSegmentationJob``, emits 1-second-hop SurfPerch
    embeddings padded around each event. Each event becomes an independent
    span — no merging of overlapping padded events.
    """

    __tablename__ = "continuous_embedding_jobs"
    __table_args__ = (UniqueConstraint("encoding_signature"),)

    status: Mapped[str] = mapped_column(default=JobStatus.queued.value)
    event_segmentation_job_id: Mapped[Optional[str]] = mapped_column(default=None)
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
    # CRNN region-based embedding source columns (nullable; populated only
    # for the ``region_crnn`` source family added in migration 061).
    region_detection_job_id: Mapped[Optional[str]] = mapped_column(default=None)
    chunk_size_seconds: Mapped[Optional[float]] = mapped_column(Float, default=None)
    chunk_hop_seconds: Mapped[Optional[float]] = mapped_column(Float, default=None)
    crnn_checkpoint_sha256: Mapped[Optional[str]] = mapped_column(Text, default=None)
    crnn_segmentation_model_id: Mapped[Optional[str]] = mapped_column(default=None)
    projection_kind: Mapped[Optional[str]] = mapped_column(Text, default=None)
    projection_dim: Mapped[Optional[int]] = mapped_column(Integer, default=None)
    total_regions: Mapped[Optional[int]] = mapped_column(Integer, default=None)
    total_chunks: Mapped[Optional[int]] = mapped_column(Integer, default=None)


class HMMSequenceJob(UUIDMixin, TimestampMixin, Base):
    """Sequence Models PR 2 — HMM training + Viterbi decode.

    Given a completed ``ContinuousEmbeddingJob``, fits PCA + GaussianHMM,
    decodes Viterbi states per window, and persists model artifacts plus
    summary statistics for visualization.
    """

    __tablename__ = "hmm_sequence_jobs"

    status: Mapped[str] = mapped_column(default=JobStatus.queued.value)
    continuous_embedding_job_id: Mapped[str]
    n_states: Mapped[int] = mapped_column(Integer)
    pca_dims: Mapped[int] = mapped_column(Integer)
    pca_whiten: Mapped[bool] = mapped_column(Boolean, default=False)
    l2_normalize: Mapped[bool] = mapped_column(Boolean, default=True)
    covariance_type: Mapped[str] = mapped_column(default="diag")
    n_iter: Mapped[int] = mapped_column(Integer, default=100)
    random_seed: Mapped[int] = mapped_column(Integer, default=42)
    min_sequence_length_frames: Mapped[int] = mapped_column(Integer, default=3)
    tol: Mapped[float] = mapped_column(Float, default=1e-4)
    library: Mapped[str] = mapped_column(default="hmmlearn")
    train_log_likelihood: Mapped[Optional[float]] = mapped_column(Float, default=None)
    n_train_sequences: Mapped[Optional[int]] = mapped_column(Integer, default=None)
    n_train_frames: Mapped[Optional[int]] = mapped_column(Integer, default=None)
    n_decoded_sequences: Mapped[Optional[int]] = mapped_column(Integer, default=None)
    artifact_dir: Mapped[Optional[str]] = mapped_column(default=None)
    error_message: Mapped[Optional[str]] = mapped_column(Text, default=None)
    # CRNN-only training-mode + tier configuration (added in migration
    # 061; required when the upstream embedding job is CRNN-source).
    training_mode: Mapped[Optional[str]] = mapped_column(Text, default=None)
    event_core_overlap_threshold: Mapped[Optional[float]] = mapped_column(
        Float, default=None
    )
    near_event_window_seconds: Mapped[Optional[float]] = mapped_column(
        Float, default=None
    )
    event_balanced_proportions: Mapped[Optional[str]] = mapped_column(
        Text, default=None
    )
    subsequence_length_chunks: Mapped[Optional[int]] = mapped_column(
        Integer, default=None
    )
    subsequence_stride_chunks: Mapped[Optional[int]] = mapped_column(
        Integer, default=None
    )
    target_train_chunks: Mapped[Optional[int]] = mapped_column(Integer, default=None)
    min_region_length_seconds: Mapped[Optional[float]] = mapped_column(
        Float, default=None
    )
