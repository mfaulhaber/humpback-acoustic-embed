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
    event_source_mode: Literal["raw", "effective"] = "raw"
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
    event_source_mode: str = "raw"
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
    event_source_mode: str = "raw"
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
    # Optional explicit Classify binding. When ``None`` the submit
    # endpoint resolves the most recent completed
    # ``EventClassificationJob`` for the upstream segmentation; when
    # provided, the value must be completed and on the same segmentation.
    event_classification_job_id: Optional[str] = None
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
    event_classification_job_id: Optional[str] = None
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
    """Per-state label distribution sourced from Call Parsing Classify.

    Simplified shape (supersedes ADR-060): outer key is state, inner key
    is the label. The ``(background)`` bucket holds windows whose center
    falls outside any effective event (or inside an event whose
    corrections wiped every type). CRNN tier metadata persists on
    ``decoded.parquet`` and exemplars but no longer stratifies the
    label-distribution chart.
    """

    n_states: int
    total_windows: int
    states: dict[str, dict[str, int]]


class ExemplarRecord(BaseModel):
    """One exemplar window for a given HMM state (ADR-059, source-agnostic)."""

    sequence_id: str
    position_in_sequence: int
    audio_file_id: int | None
    start_timestamp: float
    end_timestamp: float
    max_state_probability: float
    exemplar_type: str
    extras: dict[str, Any] = Field(default_factory=dict)


class ExemplarsResponse(BaseModel):
    """Per-state exemplar selections."""

    n_states: int
    states: dict[str, list[ExemplarRecord]]


# ---------------------------------------------------------------------------
# Motif Extraction Jobs
# ---------------------------------------------------------------------------


class MotifExtractionJobCreate(BaseModel):
    """Request body for creating a first-class motif extraction job.

    ADR-061 generalized the parent: ``parent_kind`` discriminates between
    HMM and masked-transformer sources. Exactly one parent FK must be
    set, consistent with ``parent_kind``; ``k`` is required iff
    ``parent_kind == "masked_transformer"``.
    """

    parent_kind: Literal["hmm", "masked_transformer"] = "hmm"
    hmm_sequence_job_id: Optional[str] = None
    masked_transformer_job_id: Optional[str] = None
    k: Optional[int] = Field(default=None, ge=2)
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
    def _validate_parent_xor(self) -> "MotifExtractionJobCreate":
        if self.parent_kind == "hmm":
            if not self.hmm_sequence_job_id:
                raise ValueError(
                    "hmm_sequence_job_id is required for parent_kind='hmm'"
                )
            if self.masked_transformer_job_id is not None:
                raise ValueError(
                    "masked_transformer_job_id must be null for parent_kind='hmm'"
                )
            if self.k is not None:
                raise ValueError("k must be null for parent_kind='hmm'")
        else:
            if not self.masked_transformer_job_id:
                raise ValueError(
                    "masked_transformer_job_id is required for "
                    "parent_kind='masked_transformer'"
                )
            if self.hmm_sequence_job_id is not None:
                raise ValueError(
                    "hmm_sequence_job_id must be null for "
                    "parent_kind='masked_transformer'"
                )
            if self.k is None:
                raise ValueError("k is required for parent_kind='masked_transformer'")
        return self

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
    parent_kind: str = "hmm"
    hmm_sequence_job_id: Optional[str] = None
    masked_transformer_job_id: Optional[str] = None
    k: Optional[int] = None
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
    parent_kind: str = "hmm"
    hmm_sequence_job_id: Optional[str] = None
    masked_transformer_job_id: Optional[str] = None
    k: Optional[int] = None
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


# ---------------------------------------------------------------------------
# Masked Transformer Jobs (ADR-061)
# ---------------------------------------------------------------------------


class MaskedTransformerJobCreate(BaseModel):
    """Request body for creating a ``MaskedTransformerJob``."""

    continuous_embedding_job_id: str
    # Optional explicit Classify binding; same semantics as on
    # :class:`HMMSequenceJobCreate`.
    event_classification_job_id: Optional[str] = None
    preset: Literal["small", "default", "large"] = "default"
    k_values: list[int] = Field(default_factory=lambda: [100])
    mask_fraction: float = Field(default=0.20, ge=0.0, le=1.0)
    span_length_min: int = Field(default=2, ge=1)
    span_length_max: int = Field(default=6, ge=1)
    dropout: float = Field(default=0.1, ge=0.0, le=1.0)
    mask_weight_bias: bool = True
    cosine_loss_weight: float = Field(default=0.0, ge=0.0)
    batch_size: int = Field(default=8, ge=1)
    retrieval_head_enabled: bool = False
    retrieval_dim: Optional[int] = Field(default=None, gt=0)
    retrieval_hidden_dim: Optional[int] = Field(default=None, gt=0)
    retrieval_l2_normalize: bool = True
    sequence_construction_mode: Literal["region", "event_centered", "mixed"] = "region"
    event_centered_fraction: Optional[float] = Field(default=0.0, ge=0.0, le=1.0)
    pre_event_context_sec: Optional[float] = Field(default=None, ge=0.0)
    post_event_context_sec: Optional[float] = Field(default=None, ge=0.0)
    contrastive_loss_weight: float = Field(default=0.0, ge=0.0)
    contrastive_temperature: float = Field(default=0.07, gt=0.0)
    contrastive_label_source: Literal["none", "human_corrections"] = "none"
    contrastive_min_events_per_label: int = Field(default=4, ge=1)
    contrastive_min_regions_per_label: int = Field(default=2, ge=1)
    require_cross_region_positive: bool = True
    related_label_policy_json: Optional[str] = None
    contrastive_sampler_enabled: bool = True
    contrastive_labels_per_batch: int = Field(default=4, ge=1)
    contrastive_events_per_label: int = Field(default=4, ge=1)
    contrastive_max_unlabeled_fraction: float = Field(default=0.25, ge=0.0, lt=1.0)
    contrastive_region_balance: bool = True
    max_epochs: int = Field(default=30, ge=1)
    early_stop_patience: int = Field(default=3, ge=1)
    val_split: float = Field(default=0.1, ge=0.0, lt=1.0)
    seed: int = 42

    @field_validator("k_values")
    @classmethod
    def _validate_k_values(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("k_values must be a non-empty list")
        for item in v:
            if int(item) < 2:
                raise ValueError("each k must be >= 2")
        return [int(x) for x in v]

    @model_validator(mode="after")
    def _validate_span(self) -> "MaskedTransformerJobCreate":
        if self.span_length_max < self.span_length_min:
            raise ValueError("span_length_max must be >= span_length_min")
        if self.retrieval_head_enabled:
            if self.retrieval_dim is None:
                self.retrieval_dim = 128
            if self.retrieval_hidden_dim is None:
                self.retrieval_hidden_dim = 512
        else:
            self.retrieval_dim = None
            self.retrieval_hidden_dim = None
        if self.sequence_construction_mode == "region":
            self.event_centered_fraction = 0.0
            self.pre_event_context_sec = None
            self.post_event_context_sec = None
        elif self.sequence_construction_mode == "event_centered":
            self.event_centered_fraction = 1.0
            if self.pre_event_context_sec is None:
                self.pre_event_context_sec = 2.0
            if self.post_event_context_sec is None:
                self.post_event_context_sec = 2.0
        else:
            if self.event_centered_fraction is None:
                raise ValueError("mixed mode requires event_centered_fraction")
            if not 0.0 < self.event_centered_fraction < 1.0:
                raise ValueError(
                    "mixed mode requires 0.0 < event_centered_fraction < 1.0"
                )
            if self.pre_event_context_sec is None:
                self.pre_event_context_sec = 2.0
            if self.post_event_context_sec is None:
                self.post_event_context_sec = 2.0
        if self.contrastive_loss_weight == 0.0:
            self.contrastive_label_source = "none"
        else:
            if not self.retrieval_head_enabled:
                raise ValueError("contrastive training requires retrieval_head_enabled")
            if self.sequence_construction_mode == "region":
                raise ValueError(
                    "contrastive training requires event-centered or mixed "
                    "sequence construction"
                )
            if self.contrastive_label_source != "human_corrections":
                raise ValueError(
                    "positive contrastive_loss_weight requires "
                    "contrastive_label_source='human_corrections'"
                )
        return self


class MaskedTransformerJobOut(BaseModel):
    """Masked-transformer job state returned by the API."""

    id: str
    status: str
    status_reason: Optional[str] = None
    continuous_embedding_job_id: str
    event_classification_job_id: Optional[str] = None
    training_signature: str
    preset: str
    mask_fraction: float
    span_length_min: int
    span_length_max: int
    dropout: float
    mask_weight_bias: bool
    cosine_loss_weight: float
    batch_size: int = 8
    retrieval_head_enabled: bool = False
    retrieval_dim: Optional[int] = None
    retrieval_hidden_dim: Optional[int] = None
    retrieval_l2_normalize: bool = True
    sequence_construction_mode: str = "region"
    event_centered_fraction: float = 0.0
    pre_event_context_sec: Optional[float] = None
    post_event_context_sec: Optional[float] = None
    contrastive_loss_weight: float = 0.0
    contrastive_temperature: float = 0.07
    contrastive_label_source: str = "none"
    contrastive_min_events_per_label: int = 4
    contrastive_min_regions_per_label: int = 2
    require_cross_region_positive: bool = True
    related_label_policy_json: Optional[str] = None
    contrastive_sampler_enabled: bool = True
    contrastive_labels_per_batch: int = 4
    contrastive_events_per_label: int = 4
    contrastive_max_unlabeled_fraction: float = 0.25
    contrastive_region_balance: bool = True
    max_epochs: int
    early_stop_patience: int
    val_split: float
    seed: int
    k_values: list[int] = Field(default_factory=list)
    chosen_device: Optional[str] = None
    fallback_reason: Optional[str] = None
    final_train_loss: Optional[float] = None
    final_val_loss: Optional[float] = None
    total_epochs: Optional[int] = None
    job_dir: Optional[str] = None
    total_sequences: Optional[int] = None
    total_chunks: Optional[int] = None
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}

    @field_validator("k_values", mode="before")
    @classmethod
    def _parse_k_values(cls, v: Any) -> list[int]:
        if isinstance(v, list):
            return [int(x) for x in v]
        if isinstance(v, str):
            import json as _json

            try:
                parsed = _json.loads(v)
            except _json.JSONDecodeError:
                return []
            if not isinstance(parsed, list):
                return []
            return [int(x) for x in parsed]
        return []


class MaskedTransformerJobDetail(BaseModel):
    """Detail response for a masked-transformer job."""

    job: MaskedTransformerJobOut
    region_detection_job_id: Optional[str] = None
    region_start_timestamp: Optional[float] = None
    region_end_timestamp: Optional[float] = None
    tier_composition: Optional[list[StateTierComposition]] = None
    source_kind: str = "region_crnn"


class ExtendKSweepRequest(BaseModel):
    """Body for the extend-k-sweep endpoint."""

    additional_k: list[int]

    @field_validator("additional_k")
    @classmethod
    def _validate_additional_k(cls, v: list[int]) -> list[int]:
        if not v:
            raise ValueError("additional_k must be a non-empty list")
        for item in v:
            if int(item) < 2:
                raise ValueError("each k must be >= 2")
        return [int(x) for x in v]


class GenerateInterpretationsRequest(BaseModel):
    """Body for the generate-interpretations endpoint."""

    k_values: Optional[list[int]] = None


class RegenerateLabelDistributionRequest(BaseModel):
    """Body for the HMM/MT ``regenerate-label-distribution`` endpoints.

    ``event_classification_job_id`` is optional: when omitted the
    endpoint regenerates against the currently bound Classify job; when
    provided (and validation passes) the FK is atomically re-bound after
    the artifact write succeeds.
    """

    event_classification_job_id: Optional[str] = None


RetrievalEmbeddingSpace = Literal["contextual", "retrieval"]
RetrievalMode = Literal[
    "unrestricted", "exclude_same_event", "exclude_same_event_and_region"
]
RetrievalEmbeddingVariant = Literal[
    "raw_l2",
    "centered_l2",
    "remove_pc1",
    "remove_pc3",
    "remove_pc5",
    "remove_pc10",
    "whiten_pca",
]


class MaskedTransformerNearestNeighborReportRequest(BaseModel):
    """Body for the masked-transformer nearest-neighbor report endpoint."""

    k: Optional[int] = Field(default=None, ge=2)
    embedding_space: RetrievalEmbeddingSpace = "contextual"
    samples: int = Field(default=50, ge=1)
    topn: int = Field(default=10, ge=1)
    seed: int = 20260504
    retrieval_modes: list[RetrievalMode] = Field(
        default_factory=lambda: [
            "unrestricted",
            "exclude_same_event",
            "exclude_same_event_and_region",
        ]
    )
    embedding_variants: list[RetrievalEmbeddingVariant] = Field(
        default_factory=lambda: [
            "raw_l2",
            "centered_l2",
            "remove_pc1",
            "remove_pc3",
            "remove_pc5",
            "remove_pc10",
            "whiten_pca",
        ]
    )
    include_query_rows: bool = False
    include_neighbor_rows: bool = False
    include_event_level: bool = False

    @field_validator("retrieval_modes")
    @classmethod
    def _validate_retrieval_modes(cls, v: list[RetrievalMode]) -> list[RetrievalMode]:
        if not v:
            raise ValueError("retrieval_modes must be non-empty")
        return v

    @field_validator("embedding_variants")
    @classmethod
    def _validate_embedding_variants(
        cls, v: list[RetrievalEmbeddingVariant]
    ) -> list[RetrievalEmbeddingVariant]:
        if not v:
            raise ValueError("embedding_variants must be non-empty")
        return v


class RetrievalDiagnosticsJobMetadata(BaseModel):
    """Job metadata included in a nearest-neighbor report."""

    job_id: str
    status: str
    continuous_embedding_job_id: str
    event_classification_job_id: Optional[str] = None
    region_detection_job_id: Optional[str] = None
    k_values: list[int]
    k: int
    total_sequences: Optional[int] = None
    total_chunks: Optional[int] = None
    final_train_loss: Optional[float] = None
    final_val_loss: Optional[float] = None
    total_epochs: Optional[int] = None


class RetrievalDiagnosticsOptionsOut(BaseModel):
    """Resolved options echoed by the diagnostics endpoint."""

    embedding_space: RetrievalEmbeddingSpace
    samples: int
    topn: int
    seed: int
    retrieval_modes: list[RetrievalMode]
    embedding_variants: list[RetrievalEmbeddingVariant]
    include_query_rows: bool
    include_neighbor_rows: bool
    include_event_level: bool = False


class RetrievalDiagnosticsLabelMetric(BaseModel):
    """Per-label retrieval metric."""

    query_count: int
    neighbor_count: int
    same_human_label: float


class RetrievalDiagnosticsMetrics(BaseModel):
    """Aggregate nearest-neighbor metrics for one mode/variant."""

    same_human_label: float
    exact_human_label_set: float
    same_event: float
    same_region: float
    adjacent_1s: float
    nearby_5s: float
    same_token: float
    similar_duration: float
    without_human_label: float
    low_event_overlap: float
    avg_cosine: float
    median_cosine: float
    random_pair_percentiles: dict[str, float] = Field(default_factory=dict)
    verdicts: dict[str, int] = Field(default_factory=dict)
    label_specific_same_human_label: dict[str, RetrievalDiagnosticsLabelMetric] = Field(
        default_factory=dict
    )


class RetrievalDiagnosticsQuerySummary(BaseModel):
    """Query-level summary row from retrieval diagnostics."""

    query_order: int
    query_idx: int
    query_region: str
    query_chunk: int
    query_start_timestamp: float
    query_human_types: str
    query_event_id: str
    query_duration: Optional[float] = None
    query_token: int
    neighbor_count: int
    same_human_label_rate: float
    exact_human_label_set_rate: float
    same_event_rate: float
    same_region_rate: float
    adjacent_1s_rate: float
    nearby_5s_rate: float
    same_token_rate: float
    similar_duration_rate: float
    neighbor_without_human_label_rate: float
    neighbor_low_event_overlap_rate: float
    avg_cosine: float
    verdict: str


class RetrievalDiagnosticsNeighborRow(BaseModel):
    """One nearest-neighbor row for an optional detailed response."""

    query_order: int
    query_idx: int
    rank: int
    neighbor_idx: int
    cosine: float
    query_region: str
    neighbor_region: str
    query_chunk: int
    neighbor_chunk: int
    center_delta_sec: float
    same_region: bool
    adjacent_1s: bool
    nearby_5s: bool
    query_human_types: str
    neighbor_human_types: str
    same_human_label: bool
    exact_human_label_set: bool
    query_event_id: str
    neighbor_event_id: str
    same_event: bool
    query_duration: Optional[float] = None
    neighbor_duration: Optional[float] = None
    similar_duration: bool
    query_token: int
    neighbor_token: int
    same_token: bool
    query_tier: str
    neighbor_tier: str
    query_overlap: float
    neighbor_overlap: float
    query_call_probability: float
    neighbor_call_probability: float
    query_start_timestamp: float
    neighbor_start_timestamp: float


class RetrievalDiagnosticsLabelCoverage(BaseModel):
    """Label and artifact coverage for a diagnostics response."""

    embedding_rows: int
    sampled_queries: int
    human_labeled_query_pool_rows: int
    human_labeled_effective_events: int
    vocalization_correction_rows: int
    human_label_chunk_counts: dict[str, int] = Field(default_factory=dict)
    human_label_event_counts: dict[str, int] = Field(default_factory=dict)
    corrections_by_type: dict[str, int] = Field(default_factory=dict)


class MaskedTransformerNearestNeighborReportResponse(BaseModel):
    """Structured nearest-neighbor diagnostics for one MT job."""

    job: RetrievalDiagnosticsJobMetadata
    options: RetrievalDiagnosticsOptionsOut
    artifacts: dict[str, str] = Field(default_factory=dict)
    label_coverage: RetrievalDiagnosticsLabelCoverage
    results: dict[str, dict[str, RetrievalDiagnosticsMetrics]]
    event_level_results: Optional[dict[str, dict[str, RetrievalDiagnosticsMetrics]]] = (
        None
    )
    representative_good_queries: list[RetrievalDiagnosticsQuerySummary] = Field(
        default_factory=list
    )
    representative_risky_queries: list[RetrievalDiagnosticsQuerySummary] = Field(
        default_factory=list
    )
    query_rows: list[RetrievalDiagnosticsQuerySummary] = Field(default_factory=list)
    neighbor_rows: list[RetrievalDiagnosticsNeighborRow] = Field(default_factory=list)


class LossCurveResponse(BaseModel):
    """Loss-curve payload from ``loss_curve.json``."""

    epochs: list[int] = Field(default_factory=list)
    train_loss: list[float] = Field(default_factory=list)
    val_loss: list[Optional[float]] = Field(default_factory=list)
    val_metrics: dict[str, Any] = Field(default_factory=dict)


class ReconstructionErrorRow(BaseModel):
    """One row from ``reconstruction_error.parquet``."""

    sequence_id: str
    position: int
    score: float
    start_timestamp: float
    end_timestamp: float


class ReconstructionErrorResponse(BaseModel):
    """Paginated reconstruction-error timeline strip."""

    total: int
    offset: int
    limit: int
    items: list[ReconstructionErrorRow]


class TokenRow(BaseModel):
    """One decoded-token row from ``k<N>/decoded.parquet``."""

    sequence_id: str
    position: int
    label: int
    confidence: float
    start_timestamp: float
    end_timestamp: float
    tier: Optional[str] = None
    audio_file_id: Optional[int] = None


class TokensResponse(BaseModel):
    """Paginated decoded-token strip."""

    total: int
    offset: int
    limit: int
    items: list[TokenRow]


class RunLengthsResponse(BaseModel):
    """Per-token run-length histogram payload."""

    k: int
    tau: float
    run_lengths: dict[str, list[int]] = Field(default_factory=dict)
