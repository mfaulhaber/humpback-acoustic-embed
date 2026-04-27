"""Pydantic schemas for the Sequence Models track (PR 1)."""

from datetime import datetime
from typing import Any, Optional

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
