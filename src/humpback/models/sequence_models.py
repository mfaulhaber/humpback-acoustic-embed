"""SQLAlchemy models for the Sequence Models track.

PR 1 introduces ``ContinuousEmbeddingJob``: a region-bounded, hydrophone-only
producer that emits 1-second-hop SurfPerch embeddings padded around Pass-1
region detections. Idempotent on ``encoding_signature``.
"""

from typing import Optional

from sqlalchemy import Boolean, Float, Integer, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from humpback.database import Base, TimestampMixin, UUIDMixin
from humpback.models.processing import JobStatus

__all__ = ["ContinuousEmbeddingJob", "HMMSequenceJob", "JobStatus"]


class ContinuousEmbeddingJob(UUIDMixin, TimestampMixin, Base):
    """Sequence Models PR 1 — continuous embedding producer.

    Given a completed ``RegionDetectionJob``, emits 1-second-hop SurfPerch
    embeddings padded around each region. Padded regions whose extents
    overlap are merged into single contiguous spans before windowing.
    """

    __tablename__ = "continuous_embedding_jobs"
    __table_args__ = (UniqueConstraint("encoding_signature"),)

    status: Mapped[str] = mapped_column(default=JobStatus.queued.value)
    region_detection_job_id: Mapped[str]
    model_version: Mapped[str]
    window_size_seconds: Mapped[float] = mapped_column(Float)
    hop_seconds: Mapped[float] = mapped_column(Float)
    pad_seconds: Mapped[float] = mapped_column(Float)
    target_sample_rate: Mapped[int] = mapped_column(Integer)
    feature_config_json: Mapped[Optional[str]] = mapped_column(Text, default=None)
    encoding_signature: Mapped[str]
    vector_dim: Mapped[Optional[int]] = mapped_column(Integer, default=None)
    total_regions: Mapped[Optional[int]] = mapped_column(Integer, default=None)
    merged_spans: Mapped[Optional[int]] = mapped_column(Integer, default=None)
    total_windows: Mapped[Optional[int]] = mapped_column(Integer, default=None)
    parquet_path: Mapped[Optional[str]] = mapped_column(default=None)
    error_message: Mapped[Optional[str]] = mapped_column(Text, default=None)


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
    min_sequence_length_frames: Mapped[int] = mapped_column(Integer, default=10)
    tol: Mapped[float] = mapped_column(Float, default=1e-4)
    library: Mapped[str] = mapped_column(default="hmmlearn")
    train_log_likelihood: Mapped[Optional[float]] = mapped_column(Float, default=None)
    n_train_sequences: Mapped[Optional[int]] = mapped_column(Integer, default=None)
    n_train_frames: Mapped[Optional[int]] = mapped_column(Integer, default=None)
    n_decoded_sequences: Mapped[Optional[int]] = mapped_column(Integer, default=None)
    artifact_dir: Mapped[Optional[str]] = mapped_column(default=None)
    error_message: Mapped[Optional[str]] = mapped_column(Text, default=None)
