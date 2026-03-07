from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, model_validator


class ClassifierTrainingJobCreate(BaseModel):
    name: str
    positive_embedding_set_ids: list[str]
    negative_embedding_set_ids: list[str]
    parameters: Optional[dict[str, Any]] = None


class ClassifierTrainingJobOut(BaseModel):
    id: str
    status: str
    name: str
    positive_embedding_set_ids: list[str]
    negative_embedding_set_ids: list[str]
    model_version: str
    window_size_seconds: float
    target_sample_rate: int
    feature_config: Optional[dict[str, Any]] = None
    parameters: Optional[dict[str, Any]] = None
    classifier_model_id: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ClassifierModelOut(BaseModel):
    id: str
    name: str
    model_path: str
    model_version: str
    vector_dim: int
    window_size_seconds: float
    target_sample_rate: int
    feature_config: Optional[dict[str, Any]] = None
    training_summary: Optional[dict[str, Any]] = None
    training_job_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class DetectionJobCreate(BaseModel):
    classifier_model_id: str
    audio_folder: str
    confidence_threshold: float = 0.5
    hop_seconds: float = 1.0
    high_threshold: float = 0.70
    low_threshold: float = 0.45

    @model_validator(mode="after")
    def _validate_thresholds(self):
        if self.high_threshold < self.low_threshold:
            raise ValueError("high_threshold must be >= low_threshold")
        if self.hop_seconds <= 0:
            raise ValueError("hop_seconds must be positive")
        return self


class DetectionJobOut(BaseModel):
    id: str
    status: str
    classifier_model_id: str
    audio_folder: str
    confidence_threshold: float
    hop_seconds: float
    high_threshold: float
    low_threshold: float
    output_tsv_path: Optional[str] = None
    result_summary: Optional[dict[str, Any]] = None
    error_message: Optional[str] = None
    files_processed: Optional[int] = None
    files_total: Optional[int] = None
    extract_status: Optional[str] = None
    extract_error: Optional[str] = None
    extract_summary: Optional[dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# ---- Diagnostics ----


class WindowDiagnosticRecord(BaseModel):
    filename: str
    window_index: int
    offset_sec: float
    end_sec: float
    confidence: float
    is_overlapped: bool
    overlap_sec: float


class DiagnosticsResponse(BaseModel):
    records: list[WindowDiagnosticRecord]
    total: int
    filenames: list[str]


class PerFileDiagnosticSummary(BaseModel):
    filename: str
    n_windows: int
    n_overlapped: int
    mean_confidence: float
    mean_confidence_overlapped: Optional[float] = None
    mean_confidence_normal: Optional[float] = None


class DiagnosticsSummaryResponse(BaseModel):
    total_windows: int
    total_overlapped: int
    overlapped_ratio: float
    confidence_histogram: list[dict[str, Any]]
    overlapped_mean_confidence: Optional[float] = None
    normal_mean_confidence: Optional[float] = None
    per_file: list[PerFileDiagnosticSummary]


# ---- Training Data Summary ----


class TrainingSourceInfo(BaseModel):
    embedding_set_id: str
    audio_file_id: Optional[str] = None
    filename: Optional[str] = None
    n_vectors: int
    duration_represented_sec: Optional[float] = None


class TrainingDataSummaryResponse(BaseModel):
    model_id: str
    model_name: str
    positive_sources: list[TrainingSourceInfo]
    negative_sources: list[TrainingSourceInfo]
    total_positive: int
    total_negative: int
    balance_ratio: float
    window_size_seconds: float
    positive_duration_sec: Optional[float] = None
    negative_duration_sec: Optional[float] = None
