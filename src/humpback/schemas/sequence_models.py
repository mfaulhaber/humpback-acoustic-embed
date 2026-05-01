"""Pydantic schemas for the Sequence Models track."""

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class ContinuousEmbeddingJobCreate(BaseModel):
    """Request body for creating a ``ContinuousEmbeddingJob``.

    Carries fields for both source families (ADR-057). The SurfPerch
    event-padded source requires only ``event_segmentation_job_id`` plus
    the existing hop/pad knobs. The CRNN region-based source additionally
    requires ``region_detection_job_id`` (primary), the matching
    ``event_segmentation_job_id`` as a Pass-2 disambiguator,
    ``crnn_segmentation_model_id``, ``chunk_*`` geometry, and projection
    config. Source kind is derived from ``model_version`` family;
    ``window_size_seconds``, ``target_sample_rate``, and
    ``feature_config_json`` come from the model registry.
    """

    event_segmentation_job_id: Optional[str] = None
    model_version: str = "surfperch-tensorflow2"
    hop_seconds: float = 1.0
    pad_seconds: float = 2.0

    # CRNN region-based source fields (populated only when
    # ``model_version`` belongs to the CRNN family).
    region_detection_job_id: Optional[str] = None
    crnn_segmentation_model_id: Optional[str] = None
    chunk_size_seconds: Optional[float] = None
    chunk_hop_seconds: Optional[float] = None
    projection_kind: Optional[Literal["identity", "random", "pca"]] = None
    projection_dim: Optional[int] = None

    @field_validator("hop_seconds")
    @classmethod
    def _validate_hop(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("hop_seconds must be > 0")
        return v

    @field_validator("pad_seconds")
    @classmethod
    def _validate_pad(cls, v: float) -> float:
        if v < 0:
            raise ValueError("pad_seconds must be >= 0")
        return v

    @field_validator("chunk_size_seconds", "chunk_hop_seconds")
    @classmethod
    def _validate_chunk_geometry(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v <= 0:
            raise ValueError("chunk_size_seconds and chunk_hop_seconds must be > 0")
        return v

    @field_validator("projection_dim")
    @classmethod
    def _validate_projection_dim(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and v <= 0:
            raise ValueError("projection_dim must be > 0")
        return v

    @model_validator(mode="after")
    def _validate_source_combination(self) -> "ContinuousEmbeddingJobCreate":
        # XOR rule: ``region_detection_job_id`` flips the source kind
        # to CRNN and *requires* ``event_segmentation_job_id`` as the
        # disambiguator. SurfPerch source omits ``region_detection_job_id``.
        is_crnn = self.region_detection_job_id is not None
        if is_crnn:
            if self.event_segmentation_job_id is None:
                raise ValueError(
                    "event_segmentation_job_id is required as a Pass-2 "
                    "disambiguator when region_detection_job_id is set"
                )
            missing = [
                name
                for name, value in (
                    ("crnn_segmentation_model_id", self.crnn_segmentation_model_id),
                    ("chunk_size_seconds", self.chunk_size_seconds),
                    ("chunk_hop_seconds", self.chunk_hop_seconds),
                    ("projection_kind", self.projection_kind),
                    ("projection_dim", self.projection_dim),
                )
                if value is None
            ]
            if missing:
                raise ValueError(
                    "CRNN region-based source requires: " + ", ".join(missing)
                )
        else:
            if self.event_segmentation_job_id is None:
                raise ValueError(
                    "event_segmentation_job_id is required for the SurfPerch source"
                )
            crnn_only = [
                name
                for name, value in (
                    ("crnn_segmentation_model_id", self.crnn_segmentation_model_id),
                    ("chunk_size_seconds", self.chunk_size_seconds),
                    ("chunk_hop_seconds", self.chunk_hop_seconds),
                    ("projection_kind", self.projection_kind),
                    ("projection_dim", self.projection_dim),
                )
                if value is not None
            ]
            if crnn_only:
                raise ValueError(
                    "CRNN-only fields cannot be set on the SurfPerch source: "
                    + ", ".join(crnn_only)
                )
        return self


class ContinuousEmbeddingJobOut(BaseModel):
    """Continuous embedding job state returned by the API."""

    id: str
    status: str
    event_segmentation_job_id: Optional[str] = None
    model_version: str
    window_size_seconds: Optional[float] = None
    hop_seconds: Optional[float] = None
    pad_seconds: Optional[float] = None
    target_sample_rate: int
    feature_config_json: Optional[str] = None
    encoding_signature: str
    vector_dim: Optional[int] = None
    total_events: Optional[int] = None
    merged_spans: Optional[int] = None
    total_windows: Optional[int] = None
    parquet_path: Optional[str] = None
    error_message: Optional[str] = None
    # CRNN region-based source fields
    region_detection_job_id: Optional[str] = None
    chunk_size_seconds: Optional[float] = None
    chunk_hop_seconds: Optional[float] = None
    crnn_checkpoint_sha256: Optional[str] = None
    crnn_segmentation_model_id: Optional[str] = None
    projection_kind: Optional[str] = None
    projection_dim: Optional[int] = None
    total_regions: Optional[int] = None
    total_chunks: Optional[int] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ContinuousEmbeddingSpanSummary(BaseModel):
    """Summary of a single event-scoped padded span emitted by the worker."""

    merged_span_id: int
    event_id: str
    region_id: str
    start_timestamp: float
    end_timestamp: float
    window_count: int


class ContinuousEmbeddingRegionSummary(BaseModel):
    """Summary of one Pass 1 region emitted by the CRNN producer."""

    region_id: str
    start_timestamp: float
    end_timestamp: float
    chunk_count: int


class ContinuousEmbeddingJobManifest(BaseModel):
    """Schema matching the JSON sidecar produced alongside ``embeddings.parquet``."""

    job_id: str
    model_version: str
    source_kind: str = "surfperch"
    vector_dim: int
    target_sample_rate: int
    # SurfPerch-only counters
    window_size_seconds: Optional[float] = None
    hop_seconds: Optional[float] = None
    pad_seconds: Optional[float] = None
    total_events: Optional[int] = None
    merged_spans: Optional[int] = None
    total_windows: Optional[int] = None
    spans: list[ContinuousEmbeddingSpanSummary] = Field(default_factory=list)
    # CRNN-only counters
    region_detection_job_id: Optional[str] = None
    event_segmentation_job_id: Optional[str] = None
    crnn_checkpoint_sha256: Optional[str] = None
    chunk_size_seconds: Optional[float] = None
    chunk_hop_seconds: Optional[float] = None
    projection_kind: Optional[str] = None
    projection_dim: Optional[int] = None
    total_regions: Optional[int] = None
    total_chunks: Optional[int] = None
    regions: list[ContinuousEmbeddingRegionSummary] = Field(default_factory=list)


class ContinuousEmbeddingJobDetail(BaseModel):
    """Detail response combining the DB row with the manifest sidecar."""

    job: ContinuousEmbeddingJobOut
    manifest: Optional[ContinuousEmbeddingJobManifest] = None
    extra: Optional[dict[str, Any]] = None


# ---------------------------------------------------------------------------
# HMM Sequence Jobs (PR 2)
# ---------------------------------------------------------------------------


_DEFAULT_TIER_PROPORTIONS: dict[str, float] = {
    "event_core": 0.40,
    "near_event": 0.35,
    "background": 0.25,
}


class HMMSequenceJobCreate(BaseModel):
    """Request body for creating an ``HMMSequenceJob``.

    Carries hyperparameters for both source families. CRNN-source jobs
    additionally accept ``training_mode`` plus tier configuration; the
    service rejects those fields when the upstream embedding job is
    SurfPerch-source.
    """

    continuous_embedding_job_id: str
    n_states: int = Field(ge=2)
    pca_dims: int = Field(default=50, ge=1)
    pca_whiten: bool = False
    l2_normalize: bool = True
    covariance_type: Literal["diag", "full"] = "diag"
    n_iter: int = Field(default=100, ge=1)
    random_seed: int = 42
    min_sequence_length_frames: int = Field(default=3, ge=1)
    tol: float = Field(default=1e-4, gt=0)

    # CRNN-only training-mode + tier configuration (validated against
    # the upstream embedding job's source kind in the service layer).
    training_mode: Optional[Literal["full_region", "event_balanced", "event_only"]] = (
        None
    )
    event_core_overlap_threshold: Optional[float] = None
    near_event_window_seconds: Optional[float] = None
    event_balanced_proportions: Optional[dict[str, float]] = None
    subsequence_length_chunks: Optional[int] = None
    subsequence_stride_chunks: Optional[int] = None
    target_train_chunks: Optional[int] = None
    min_region_length_seconds: Optional[float] = None

    @field_validator("event_balanced_proportions")
    @classmethod
    def _validate_proportions(
        cls, v: Optional[dict[str, float]]
    ) -> Optional[dict[str, float]]:
        if v is None:
            return v
        if not v:
            raise ValueError("event_balanced_proportions cannot be empty")
        for key in v:
            if key not in {"event_core", "near_event", "background"}:
                raise ValueError(
                    f"unexpected tier key in event_balanced_proportions: {key!r}"
                )
        total = sum(v.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"event_balanced_proportions must sum to 1.0 ± 1e-6 (got {total})"
            )
        return v


class HMMSequenceJobOut(BaseModel):
    """HMM sequence job state returned by the API."""

    id: str
    status: str
    continuous_embedding_job_id: str
    n_states: int
    pca_dims: int
    pca_whiten: bool
    l2_normalize: bool
    covariance_type: str
    n_iter: int
    random_seed: int
    min_sequence_length_frames: int
    tol: float
    library: str
    train_log_likelihood: Optional[float] = None
    n_train_sequences: Optional[int] = None
    n_train_frames: Optional[int] = None
    n_decoded_sequences: Optional[int] = None
    artifact_dir: Optional[str] = None
    error_message: Optional[str] = None
    # CRNN-only training-mode + tier configuration (null on SurfPerch jobs).
    training_mode: Optional[str] = None
    event_core_overlap_threshold: Optional[float] = None
    near_event_window_seconds: Optional[float] = None
    event_balanced_proportions: Optional[str] = None
    subsequence_length_chunks: Optional[int] = None
    subsequence_stride_chunks: Optional[int] = None
    target_train_chunks: Optional[int] = None
    min_region_length_seconds: Optional[float] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class HMMStateSummary(BaseModel):
    """Per-state summary from ``state_summary.json``."""

    state: int
    occupancy: float
    mean_dwell_frames: float
    dwell_histogram: list[int] = Field(default_factory=list)


class StateTierComposition(BaseModel):
    """Per-state tier-composition for CRNN-source HMM jobs."""

    state: int
    event_core: float
    near_event: float
    background: float


class HMMSequenceJobDetail(BaseModel):
    """Detail response combining the DB row with state summary stats."""

    job: HMMSequenceJobOut
    region_detection_job_id: str
    region_start_timestamp: float | None = None
    region_end_timestamp: float | None = None
    summary: Optional[list[HMMStateSummary]] = None
    tier_composition: Optional[list[StateTierComposition]] = None
    source_kind: str = "surfperch"


class TransitionMatrixResponse(BaseModel):
    """Transition matrix as a nested list for JSON serialization."""

    n_states: int
    matrix: list[list[float]]


class DwellHistogramResponse(BaseModel):
    """Per-state dwell-time histograms."""

    n_states: int
    histograms: dict[str, list[int]]


# ---------------------------------------------------------------------------
# Interpretation visualizations (PR 3)
# ---------------------------------------------------------------------------


class OverlayPoint(BaseModel):
    """Single point in the PCA/UMAP 2-D overlay (ADR-059, source-agnostic)."""

    sequence_id: str
    position_in_sequence: int
    start_timestamp: float
    end_timestamp: float
    pca_x: float
    pca_y: float
    umap_x: float
    umap_y: float
    viterbi_state: int
    max_state_probability: float


class OverlayResponse(BaseModel):
    """Paginated PCA/UMAP overlay points."""

    total: int
    items: list[OverlayPoint]


class LabelDistributionResponse(BaseModel):
    """Per-state label distribution from center-time join.

    Unified nested shape (ADR-060): outer key is state, middle key is the
    tier bucket, inner key is the label. SurfPerch jobs use the synthetic
    ``"all"`` tier; CRNN jobs use real tier keys (``event_core`` /
    ``near_event`` / ``background``).
    """

    n_states: int
    total_windows: int
    states: dict[str, dict[str, dict[str, int]]]


class ExemplarRecord(BaseModel):
    """One exemplar window for a given HMM state (ADR-059, source-agnostic)."""

    sequence_id: str
    position_in_sequence: int
    audio_file_id: int | None
    start_timestamp: float
    end_timestamp: float
    max_state_probability: float
    exemplar_type: str
    extras: dict[str, str | int | float | None] = Field(default_factory=dict)


class ExemplarsResponse(BaseModel):
    """Per-state exemplar selections."""

    n_states: int
    states: dict[str, list[ExemplarRecord]]


# ---------------------------------------------------------------------------
# Motif Extraction Jobs
# ---------------------------------------------------------------------------


class MotifExtractionJobCreate(BaseModel):
    """Request body for creating a first-class motif extraction job."""

    hmm_sequence_job_id: str
    min_ngram: int = Field(default=2, ge=1)
    max_ngram: int = Field(default=8, ge=1, le=16)
    minimum_occurrences: int = Field(default=5, ge=1)
    minimum_event_sources: int = Field(default=2, ge=1)
    frequency_weight: float = Field(default=0.40, ge=0)
    event_source_weight: float = Field(default=0.30, ge=0)
    event_core_weight: float = Field(default=0.20, ge=0)
    low_background_weight: float = Field(default=0.10, ge=0)
    call_probability_weight: Optional[float] = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _validate_ngram_and_weights(self) -> "MotifExtractionJobCreate":
        if self.max_ngram < self.min_ngram:
            raise ValueError("max_ngram must be >= min_ngram")
        weights = [
            self.frequency_weight,
            self.event_source_weight,
            self.event_core_weight,
            self.low_background_weight,
        ]
        if self.call_probability_weight is not None:
            weights.append(self.call_probability_weight)
        if not any(w > 0 for w in weights):
            raise ValueError("at least one rank weight must be > 0")
        return self


class MotifExtractionJobOut(BaseModel):
    """Motif extraction job state returned by the API."""

    id: str
    status: str
    hmm_sequence_job_id: str
    source_kind: str
    min_ngram: int
    max_ngram: int
    minimum_occurrences: int
    minimum_event_sources: int
    frequency_weight: float
    event_source_weight: float
    event_core_weight: float
    low_background_weight: float
    call_probability_weight: Optional[float] = None
    config_signature: str
    total_groups: Optional[int] = None
    total_collapsed_tokens: Optional[int] = None
    total_candidate_occurrences: Optional[int] = None
    total_motifs: Optional[int] = None
    artifact_dir: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class MotifExtractionManifest(BaseModel):
    """Manifest JSON emitted by motif extraction jobs."""

    schema_version: int
    motif_extraction_job_id: str
    hmm_sequence_job_id: str
    continuous_embedding_job_id: str
    source_kind: str
    config: dict[str, Any]
    config_signature: str
    generated_at: str
    total_groups: int
    total_collapsed_tokens: int
    total_candidate_occurrences: int
    total_motifs: int
    event_source_key_strategy: str


class MotifExtractionJobDetail(BaseModel):
    """Motif extraction job detail response."""

    job: MotifExtractionJobOut
    manifest: Optional[MotifExtractionManifest] = None


class MotifSummary(BaseModel):
    """One ranked motif summary row."""

    motif_key: str
    states: list[int]
    length: int
    occurrence_count: int
    event_source_count: int
    audio_source_count: int
    group_count: int
    event_core_fraction: float
    background_fraction: float
    mean_call_probability: Optional[float] = None
    mean_duration_seconds: float
    median_duration_seconds: float
    rank_score: float
    example_occurrence_ids: list[str] = Field(default_factory=list)


class MotifsResponse(BaseModel):
    """Paginated motif summary response."""

    total: int
    offset: int
    limit: int
    items: list[MotifSummary]


class MotifOccurrence(BaseModel):
    """One occurrence row for a motif."""

    occurrence_id: str
    motif_key: str
    states: list[int]
    source_kind: str
    group_key: str
    event_source_key: str
    audio_source_key: Optional[str] = None
    token_start_index: int
    token_end_index: int
    raw_start_index: int
    raw_end_index: int
    start_timestamp: float
    end_timestamp: float
    duration_seconds: float
    event_core_fraction: float
    background_fraction: float
    mean_call_probability: Optional[float] = None
    anchor_event_id: Optional[str] = None
    anchor_timestamp: float
    relative_start_seconds: float
    relative_end_seconds: float
    anchor_strategy: str


class MotifOccurrencesResponse(BaseModel):
    """Paginated motif occurrence response."""

    total: int
    offset: int
    limit: int
    items: list[MotifOccurrence]
