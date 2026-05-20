"""Pydantic schemas for retained Sequence Models APIs."""

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from humpback.schemas.piano_roll_notes import PianoRollNotesStatusResponse


class ContinuousEmbeddingJobCreate(BaseModel):
    """Request body for creating a ``ContinuousEmbeddingJob``.

    Carries fields for both retained source families. The SurfPerch event-padded
    source requires ``event_segmentation_job_id`` plus hop/pad knobs. The CRNN
    region-based source additionally requires ``region_detection_job_id``, a
    matching ``event_segmentation_job_id`` disambiguator, CRNN model metadata,
    chunk geometry, and projection config.
    """

    event_segmentation_job_id: Optional[str] = None
    event_source_mode: Literal["raw", "effective"] = "raw"
    model_version: str = "surfperch-tensorflow2"
    hop_seconds: float = 1.0
    pad_seconds: float = 2.0

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
    window_size_seconds: Optional[float] = None
    hop_seconds: Optional[float] = None
    pad_seconds: Optional[float] = None
    total_events: Optional[int] = None
    merged_spans: Optional[int] = None
    total_windows: Optional[int] = None
    spans: list[ContinuousEmbeddingSpanSummary] = Field(default_factory=list)
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


EventEncoderPoolName = Literal[
    "mean_pool",
    "top_k_pool",
    "start_pool",
    "middle_pool",
    "end_pool",
]


def _default_event_encoder_pools() -> list[EventEncoderPoolName]:
    return [
        "mean_pool",
        "top_k_pool",
        "start_pool",
        "middle_pool",
        "end_pool",
    ]


class EventEncoderPoolingConfig(BaseModel):
    """Pooling knobs for event-level CRNN chunk embeddings."""

    enabled_pools: list[EventEncoderPoolName] = Field(
        default_factory=_default_event_encoder_pools
    )
    top_k_fraction: float = 0.25
    min_overlap_fraction: float = 0.25
    min_chunks_per_event: int = 1

    @field_validator("enabled_pools")
    @classmethod
    def _validate_enabled_pools(
        cls, value: list[EventEncoderPoolName]
    ) -> list[EventEncoderPoolName]:
        if not value:
            raise ValueError("enabled_pools must not be empty")
        if len(set(value)) != len(value):
            raise ValueError("enabled_pools must not contain duplicates")
        return value

    @field_validator("top_k_fraction")
    @classmethod
    def _validate_top_k_fraction(cls, value: float) -> float:
        if value <= 0 or value > 1:
            raise ValueError("top_k_fraction must be > 0 and <= 1")
        return value

    @field_validator("min_overlap_fraction")
    @classmethod
    def _validate_min_overlap_fraction(cls, value: float) -> float:
        if value < 0 or value > 1:
            raise ValueError("min_overlap_fraction must be between 0 and 1")
        return value

    @field_validator("min_chunks_per_event")
    @classmethod
    def _validate_min_chunks(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("min_chunks_per_event must be > 0")
        return value


class EventEncoderDescriptorConfig(BaseModel):
    """Acoustic descriptor extraction config for event crops."""

    target_sample_rate: int = 16000
    n_fft: int = 1024
    hop_length: int = 512
    eps: float = 1e-12
    ridge_min_frequency_hz: float = 100.0
    ridge_max_frequency_hz: float = 6000.0
    ridge_candidate_count: int = 5
    ridge_smoothness_penalty: float = 8.0
    ridge_peak_prominence_ratio: float = 0.0
    ridge_summary_low_percentile: float = 10.0
    ridge_summary_high_percentile: float = 90.0
    band_peak_min_frequency_hz: float = 100.0
    band_peak_max_frequency_hz: Optional[float] = None
    high_band_min_frequency_hz: float = 1000.0
    f0_fmin: float = 70.0
    f0_fmax: float = 1200.0
    pulse_min_rate_hz: float = 2.0
    pulse_max_rate_hz: float = 200.0
    pulse_confidence_threshold: float = 0.3
    pulse_envelope_smooth_ms: float = 5.0

    @field_validator(
        "target_sample_rate",
        "n_fft",
        "hop_length",
        "ridge_candidate_count",
    )
    @classmethod
    def _validate_positive_int(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("descriptor integer settings must be > 0")
        return value

    @field_validator("eps")
    @classmethod
    def _validate_eps(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("eps must be > 0")
        return value

    @field_validator(
        "ridge_min_frequency_hz",
        "ridge_max_frequency_hz",
        "ridge_smoothness_penalty",
        "ridge_peak_prominence_ratio",
    )
    @classmethod
    def _validate_non_negative_ridge_float(cls, value: float) -> float:
        if value < 0:
            raise ValueError("ridge descriptor settings must be >= 0")
        return value

    @field_validator(
        "f0_fmin",
        "f0_fmax",
        "band_peak_min_frequency_hz",
        "high_band_min_frequency_hz",
        "pulse_min_rate_hz",
        "pulse_max_rate_hz",
        "pulse_envelope_smooth_ms",
    )
    @classmethod
    def _validate_positive_descriptor_float(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("tonal and pulse descriptor settings must be > 0")
        return value

    @field_validator("band_peak_max_frequency_hz")
    @classmethod
    def _validate_optional_positive_descriptor_float(
        cls, value: Optional[float]
    ) -> Optional[float]:
        if value is not None and value <= 0:
            raise ValueError("band_peak_max_frequency_hz must be > 0")
        return value

    @field_validator("pulse_confidence_threshold")
    @classmethod
    def _validate_pulse_confidence(cls, value: float) -> float:
        if value < 0 or value > 1:
            raise ValueError("pulse_confidence_threshold must be between 0 and 1")
        return value

    @model_validator(mode="after")
    def _validate_descriptor_bands(self) -> "EventEncoderDescriptorConfig":
        if self.ridge_min_frequency_hz <= 0:
            raise ValueError("ridge_min_frequency_hz must be > 0")
        if self.ridge_max_frequency_hz <= self.ridge_min_frequency_hz:
            raise ValueError(
                "ridge_max_frequency_hz must be greater than ridge_min_frequency_hz"
            )
        if self.ridge_peak_prominence_ratio > 1:
            raise ValueError("ridge_peak_prominence_ratio must be <= 1")
        if not (
            0
            <= self.ridge_summary_low_percentile
            < self.ridge_summary_high_percentile
            <= 100
        ):
            raise ValueError(
                "ridge summary percentiles must satisfy 0 <= low < high <= 100"
            )
        if (
            self.band_peak_max_frequency_hz is not None
            and self.band_peak_max_frequency_hz <= self.band_peak_min_frequency_hz
        ):
            raise ValueError(
                "band_peak_max_frequency_hz must be greater than "
                "band_peak_min_frequency_hz"
            )
        if self.f0_fmax <= self.f0_fmin:
            raise ValueError("f0_fmax must be greater than f0_fmin")
        if self.pulse_max_rate_hz <= self.pulse_min_rate_hz:
            raise ValueError("pulse_max_rate_hz must be greater than pulse_min_rate_hz")
        return self


class EventEncoderPreprocessingConfig(BaseModel):
    """Preprocessing config for event vectors before k-means tokenization."""

    l2_normalize_pools: bool = True
    pca_dim: Literal[64, 128] = 128
    embedding_weight: float = 1.0
    descriptor_weight: float = 0.364
    descriptor_clip_value: Optional[float] = 3.0

    @field_validator("embedding_weight", "descriptor_weight")
    @classmethod
    def _validate_weight(cls, value: float) -> float:
        if value < 0:
            raise ValueError("feature weights must be >= 0")
        return value

    @field_validator("descriptor_clip_value")
    @classmethod
    def _validate_descriptor_clip(cls, value: Optional[float]) -> Optional[float]:
        if value is not None and value < 0:
            raise ValueError("descriptor_clip_value must be >= 0")
        return value

    @model_validator(mode="after")
    def _validate_nonzero_weight(self) -> "EventEncoderPreprocessingConfig":
        if self.embedding_weight == 0 and self.descriptor_weight == 0:
            raise ValueError("at least one feature weight must be > 0")
        return self


class EventEncoderJobCreate(BaseModel):
    """Request body for creating an Event Encoder tokenization job."""

    event_segmentation_job_id: str
    event_source_mode: Literal["raw", "effective"] = "raw"
    continuous_embedding_job_id: str
    tokenizer_version: str = "crnn-event-encoder-v3"
    pooling: EventEncoderPoolingConfig = Field(
        default_factory=EventEncoderPoolingConfig
    )
    descriptor: EventEncoderDescriptorConfig = Field(
        default_factory=EventEncoderDescriptorConfig
    )
    preprocessing: EventEncoderPreprocessingConfig = Field(
        default_factory=EventEncoderPreprocessingConfig
    )
    k_values: list[int] = Field(default_factory=lambda: [50, 100, 200])
    random_seed: int = 0

    @field_validator("event_segmentation_job_id", "continuous_embedding_job_id")
    @classmethod
    def _validate_non_empty_id(cls, value: str) -> str:
        if not value:
            raise ValueError("source job ids must not be empty")
        return value

    @field_validator("tokenizer_version")
    @classmethod
    def _validate_tokenizer_version(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("tokenizer_version must not be empty")
        return value

    @field_validator("k_values")
    @classmethod
    def _validate_k_values(cls, value: list[int]) -> list[int]:
        if not value:
            raise ValueError("k_values must not be empty")
        if any(k <= 0 for k in value):
            raise ValueError("k_values must all be > 0")
        if len(set(value)) != len(value):
            raise ValueError("k_values must be unique")
        return sorted(value)


class EventEncoderJobOut(BaseModel):
    """Event Encoder job state returned by the API."""

    id: str
    status: str
    event_segmentation_job_id: str
    event_source_mode: str
    continuous_embedding_job_id: str
    continuous_embedding_signature: str
    tokenizer_version: str
    pooling_config_json: str
    descriptor_config_json: str
    preprocessing_config_json: str
    k_values_json: str
    random_seed: int
    tokenization_signature: str
    event_vector_dim: Optional[int] = None
    total_events: Optional[int] = None
    encoded_events: Optional[int] = None
    skipped_events: Optional[int] = None
    event_vectors_path: Optional[str] = None
    event_tokens_path: Optional[str] = None
    token_sequences_path: Optional[str] = None
    manifest_path: Optional[str] = None
    report_path: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class EventEncoderJobDetail(BaseModel):
    """Detail response combining the DB row with JSON sidecars."""

    job: EventEncoderJobOut
    manifest: Optional[dict[str, Any]] = None
    report: Optional[dict[str, Any]] = None


class EventEncoderTimelineEvent(BaseModel):
    """One tokenized event row for the Event Encoder timeline viewer."""

    event_id: str
    region_id: str
    source_sequence_key: str
    sequence_index: int
    start_timestamp: float
    end_timestamp: float
    token_id: int
    token_label: str
    token_confidence: float
    distance_to_centroid: float
    second_centroid_distance: Optional[float] = None
    descriptor_values: dict[str, float] = Field(default_factory=dict)
    descriptor_vector_values: dict[str, float] = Field(default_factory=dict)


class EventEncoderTimelineResponse(BaseModel):
    """Timeline-ready token assignments for a completed Event Encoder job."""

    job_id: str
    event_segmentation_job_id: str
    event_source_mode: Literal["raw", "effective"]
    continuous_embedding_job_id: str
    region_detection_job_id: str
    selected_k: int
    valid_k_values: list[int]
    descriptor_feature_names: list[str] = Field(default_factory=list)
    descriptor_feature_units: dict[str, str] = Field(default_factory=dict)
    job_start_timestamp: float
    job_end_timestamp: float
    events: list[EventEncoderTimelineEvent]
    notes_status: PianoRollNotesStatusResponse


EventEncoderProjectionMethod = Literal["umap", "pca"]


class EventEncoderProjectionPoint(BaseModel):
    """One projected Event Encoder event vector for a selected tokenization."""

    event_id: str
    region_id: str
    source_sequence_key: str
    sequence_index: int
    start_timestamp: float
    end_timestamp: float
    token_id: int
    token_label: str
    token_confidence: float
    distance_to_centroid: float
    second_centroid_distance: Optional[float] = None
    x: float
    y: float


class EventEncoderProjectionResponse(BaseModel):
    """Two-dimensional Event Encoder vector projection for a completed job."""

    job_id: str
    selected_k: int
    valid_k_values: list[int]
    method: EventEncoderProjectionMethod
    x_axis_label: str
    y_axis_label: str
    points: list[EventEncoderProjectionPoint]
