"""Pydantic schemas for the Sequence Models track."""

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class ContinuousEmbeddingJobCreate(BaseModel):
    """Request body for creating a ``ContinuousEmbeddingJob``.

    ``window_size_seconds``, ``target_sample_rate``, and ``feature_config_json``
    are derived from the registered ``ModelConfig`` for ``model_version`` at
    submission time and are not user-supplied.
    """

    region_detection_job_id: str
    model_version: str = "surfperch-tensorflow2"
    hop_seconds: float = 1.0
    pad_seconds: float = 10.0

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


class ContinuousEmbeddingJobOut(BaseModel):
    """Continuous embedding job state returned by the API."""

    id: str
    status: str
    region_detection_job_id: str
    model_version: str
    window_size_seconds: float
    hop_seconds: float
    pad_seconds: float
    target_sample_rate: int
    feature_config_json: Optional[str] = None
    encoding_signature: str
    vector_dim: Optional[int] = None
    total_regions: Optional[int] = None
    merged_spans: Optional[int] = None
    total_windows: Optional[int] = None
    parquet_path: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ContinuousEmbeddingSpanSummary(BaseModel):
    """Summary of a single merged padded span emitted by the worker."""

    merged_span_id: int
    start_time_sec: float
    end_time_sec: float
    window_count: int
    source_region_ids: list[str] = Field(default_factory=list)


class ContinuousEmbeddingJobManifest(BaseModel):
    """Schema matching the JSON sidecar produced alongside ``embeddings.parquet``."""

    job_id: str
    model_version: str
    vector_dim: int
    window_size_seconds: float
    hop_seconds: float
    pad_seconds: float
    target_sample_rate: int
    total_regions: int
    merged_spans: int
    total_windows: int
    spans: list[ContinuousEmbeddingSpanSummary] = Field(default_factory=list)


class ContinuousEmbeddingJobDetail(BaseModel):
    """Detail response combining the DB row with the manifest sidecar."""

    job: ContinuousEmbeddingJobOut
    manifest: Optional[ContinuousEmbeddingJobManifest] = None
    extra: Optional[dict[str, Any]] = None


# ---------------------------------------------------------------------------
# HMM Sequence Jobs (PR 2)
# ---------------------------------------------------------------------------


class HMMSequenceJobCreate(BaseModel):
    """Request body for creating an ``HMMSequenceJob``."""

    continuous_embedding_job_id: str
    n_states: int = Field(ge=2)
    pca_dims: int = Field(default=50, ge=1)
    pca_whiten: bool = False
    l2_normalize: bool = True
    covariance_type: Literal["diag", "full"] = "diag"
    n_iter: int = Field(default=100, ge=1)
    random_seed: int = 42
    min_sequence_length_frames: int = Field(default=10, ge=1)
    tol: float = Field(default=1e-4, gt=0)


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
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class HMMStateSummary(BaseModel):
    """Per-state summary from ``state_summary.json``."""

    state: int
    occupancy: float
    mean_dwell_frames: float
    dwell_histogram: list[int] = Field(default_factory=list)


class HMMSequenceJobDetail(BaseModel):
    """Detail response combining the DB row with state summary stats."""

    job: HMMSequenceJobOut
    summary: Optional[list[HMMStateSummary]] = None


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
    """Single point in the PCA/UMAP 2-D overlay."""

    merged_span_id: int
    window_index_in_span: int
    start_time_sec: float
    end_time_sec: float
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
    """Per-state label distribution from center-time join."""

    n_states: int
    total_windows: int
    states: dict[str, dict[str, int]]


class ExemplarRecord(BaseModel):
    """One exemplar window for a given HMM state."""

    merged_span_id: int
    window_index_in_span: int
    audio_file_id: int
    start_time_sec: float
    end_time_sec: float
    max_state_probability: float
    exemplar_type: str


class ExemplarsResponse(BaseModel):
    """Per-state exemplar selections."""

    n_states: int
    states: dict[str, list[ExemplarRecord]]
