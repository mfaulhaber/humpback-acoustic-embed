"""SQLAlchemy models for the Sequence Models track.

``ContinuousEmbeddingJob``: an event-scoped, hydrophone-only producer that
emits 1-second-hop SurfPerch embeddings padded around Pass-2 segmentation
events. Idempotent on ``encoding_signature``.
"""

from typing import Optional

from sqlalchemy import Boolean, Float, Index, Integer, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from humpback.database import Base, TimestampMixin, UUIDMixin
from humpback.models.processing import JobStatus

__all__ = [
    "ContinuousEmbeddingJob",
    "HMMSequenceJob",
    "JobStatus",
    "MaskedTransformerJob",
    "MotifExtractionJob",
]


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
    # Pass 3 Classify job whose typed_events.parquet (with VocalizationCorrection
    # overlay) feeds label_distribution.json and exemplar annotations. Nullable
    # in storage for the in-transaction window only; the submit endpoint
    # resolves a non-NULL value before the row commits.
    event_classification_job_id: Mapped[Optional[str]] = mapped_column(default=None)
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


class MaskedTransformerJob(UUIDMixin, TimestampMixin, Base):
    """ADR-061 — masked-span transformer training job.

    Trains a context encoder over a completed CRNN region-based
    ``ContinuousEmbeddingJob`` and produces per-k k-means tokenization
    bundles under the job directory. Idempotent on
    ``training_signature``; ``k_values`` may be extended after completion
    via the extend-k-sweep service entry point.
    """

    __tablename__ = "masked_transformer_jobs"
    __table_args__ = (
        Index("ix_masked_transformer_jobs_status", "status"),
        Index(
            "ix_masked_transformer_jobs_continuous_embedding_job_id",
            "continuous_embedding_job_id",
        ),
        Index(
            "ix_masked_transformer_jobs_training_signature",
            "training_signature",
            unique=True,
        ),
    )

    status: Mapped[str] = mapped_column(default=JobStatus.queued.value)
    status_reason: Mapped[Optional[str]] = mapped_column(Text, default=None)
    continuous_embedding_job_id: Mapped[str]
    # Pass 3 Classify job that feeds label_distribution.json + exemplar
    # annotations; nullable in storage for the in-transaction window only.
    event_classification_job_id: Mapped[Optional[str]] = mapped_column(default=None)
    training_signature: Mapped[str]
    # Training config
    preset: Mapped[str] = mapped_column(Text, default="default")
    mask_fraction: Mapped[float] = mapped_column(Float, default=0.20)
    span_length_min: Mapped[int] = mapped_column(Integer, default=2)
    span_length_max: Mapped[int] = mapped_column(Integer, default=6)
    dropout: Mapped[float] = mapped_column(Float, default=0.1)
    mask_weight_bias: Mapped[bool] = mapped_column(Boolean, default=True)
    cosine_loss_weight: Mapped[float] = mapped_column(Float, default=0.0)
    batch_size: Mapped[int] = mapped_column(Integer, default=8)
    retrieval_head_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    retrieval_dim: Mapped[Optional[int]] = mapped_column(Integer, default=None)
    retrieval_hidden_dim: Mapped[Optional[int]] = mapped_column(Integer, default=None)
    retrieval_l2_normalize: Mapped[bool] = mapped_column(Boolean, default=True)
    sequence_construction_mode: Mapped[str] = mapped_column(Text, default="region")
    event_centered_fraction: Mapped[float] = mapped_column(Float, default=0.0)
    pre_event_context_sec: Mapped[Optional[float]] = mapped_column(Float, default=None)
    post_event_context_sec: Mapped[Optional[float]] = mapped_column(Float, default=None)
    contrastive_loss_weight: Mapped[float] = mapped_column(Float, default=0.0)
    contrastive_temperature: Mapped[float] = mapped_column(Float, default=0.07)
    contrastive_label_source: Mapped[str] = mapped_column(Text, default="none")
    contrastive_min_events_per_label: Mapped[int] = mapped_column(Integer, default=4)
    contrastive_min_regions_per_label: Mapped[int] = mapped_column(Integer, default=2)
    require_cross_region_positive: Mapped[bool] = mapped_column(Boolean, default=True)
    related_label_policy_json: Mapped[Optional[str]] = mapped_column(Text, default=None)
    max_epochs: Mapped[int] = mapped_column(Integer, default=30)
    early_stop_patience: Mapped[int] = mapped_column(Integer, default=3)
    val_split: Mapped[float] = mapped_column(Float, default=0.1)
    seed: Mapped[int] = mapped_column(Integer, default=42)
    # Tokenization config — JSON-encoded list of ints to preserve order
    # and allow extension without a schema migration.
    k_values: Mapped[str] = mapped_column(Text)
    # Device + outcomes
    chosen_device: Mapped[Optional[str]] = mapped_column(Text, default=None)
    fallback_reason: Mapped[Optional[str]] = mapped_column(Text, default=None)
    final_train_loss: Mapped[Optional[float]] = mapped_column(Float, default=None)
    final_val_loss: Mapped[Optional[float]] = mapped_column(Float, default=None)
    total_epochs: Mapped[Optional[int]] = mapped_column(Integer, default=None)
    # Storage
    job_dir: Mapped[Optional[str]] = mapped_column(Text, default=None)
    total_sequences: Mapped[Optional[int]] = mapped_column(Integer, default=None)
    total_chunks: Mapped[Optional[int]] = mapped_column(Integer, default=None)
    error_message: Mapped[Optional[str]] = mapped_column(Text, default=None)


class MotifExtractionJob(UUIDMixin, TimestampMixin, Base):
    """First-class motif extraction over a completed sequence-models parent.

    ADR-061 generalized the parent: ``parent_kind`` discriminates between
    HMM and masked-transformer parents, with corresponding nullable FK
    columns. ``k`` is required for masked-transformer parents and null
    otherwise; the SQL CHECK constraint enforces these invariants.
    """

    __tablename__ = "motif_extraction_jobs"
    __table_args__ = (
        Index("ix_motif_extraction_jobs_status", "status"),
        Index("ix_motif_extraction_jobs_hmm_sequence_job_id", "hmm_sequence_job_id"),
        Index("ix_motif_extraction_jobs_config_signature", "config_signature"),
        Index("ix_motif_extraction_jobs_parent_kind", "parent_kind"),
        Index(
            "ix_motif_extraction_jobs_masked_transformer_job_id",
            "masked_transformer_job_id",
        ),
    )

    status: Mapped[str] = mapped_column(default=JobStatus.queued.value)
    parent_kind: Mapped[str] = mapped_column(Text, default="hmm")
    hmm_sequence_job_id: Mapped[Optional[str]] = mapped_column(default=None)
    masked_transformer_job_id: Mapped[Optional[str]] = mapped_column(default=None)
    k: Mapped[Optional[int]] = mapped_column(Integer, default=None)
    source_kind: Mapped[str] = mapped_column(Text)
    min_ngram: Mapped[int] = mapped_column(Integer, default=2)
    max_ngram: Mapped[int] = mapped_column(Integer, default=8)
    minimum_occurrences: Mapped[int] = mapped_column(Integer, default=5)
    minimum_event_sources: Mapped[int] = mapped_column(Integer, default=2)
    frequency_weight: Mapped[float] = mapped_column(Float, default=0.40)
    event_source_weight: Mapped[float] = mapped_column(Float, default=0.30)
    event_core_weight: Mapped[float] = mapped_column(Float, default=0.20)
    low_background_weight: Mapped[float] = mapped_column(Float, default=0.10)
    call_probability_weight: Mapped[Optional[float]] = mapped_column(
        Float, default=None
    )
    config_signature: Mapped[str]
    total_groups: Mapped[Optional[int]] = mapped_column(Integer, default=None)
    total_collapsed_tokens: Mapped[Optional[int]] = mapped_column(Integer, default=None)
    total_candidate_occurrences: Mapped[Optional[int]] = mapped_column(
        Integer, default=None
    )
    total_motifs: Mapped[Optional[int]] = mapped_column(Integer, default=None)
    artifact_dir: Mapped[Optional[str]] = mapped_column(default=None)
    error_message: Mapped[Optional[str]] = mapped_column(Text, default=None)
