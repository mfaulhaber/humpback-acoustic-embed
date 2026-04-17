from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator

DETECTION_MODE_CREATE_ERROR = (
    "detection_mode is no longer accepted; new jobs are always created in windowed mode"
)


# ---- Batch Label Edit ----


class LabelEditItem(BaseModel):
    action: Literal["add", "move", "delete", "change_type"]
    row_id: Optional[str] = None
    start_utc: Optional[float] = None
    end_utc: Optional[float] = None
    label: Optional[Literal["humpback", "orca", "ship", "background"]] = None


class LabelEditRequest(BaseModel):
    edits: list[LabelEditItem]


class ClassifierTrainingJobCreate(BaseModel):
    name: str
    positive_embedding_set_ids: list[str] = Field(default_factory=list)
    negative_embedding_set_ids: list[str] = Field(default_factory=list)
    detection_job_ids: list[str] = Field(default_factory=list)
    embedding_model_version: Optional[str] = None
    parameters: Optional[dict[str, Any]] = None

    @model_validator(mode="after")
    def _validate_source_mode(self):
        has_embedding_sets = bool(
            self.positive_embedding_set_ids or self.negative_embedding_set_ids
        )
        has_detection_jobs = bool(self.detection_job_ids)

        if has_embedding_sets and has_detection_jobs:
            raise ValueError(
                "Cannot mix embedding_set sources with detection_job sources"
            )
        if not has_embedding_sets and not has_detection_jobs:
            raise ValueError(
                "Must provide either embedding set IDs or detection_job_ids"
            )
        if has_embedding_sets:
            if not self.positive_embedding_set_ids:
                raise ValueError("At least one positive embedding set is required")
            if not self.negative_embedding_set_ids:
                raise ValueError("At least one negative embedding set is required")
        else:
            if not self.embedding_model_version:
                raise ValueError(
                    "embedding_model_version is required when submitting "
                    "detection_job_ids"
                )
        return self


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
    source_mode: Literal[
        "embedding_sets", "autoresearch_candidate", "detection_manifest"
    ] = "embedding_sets"
    source_candidate_id: Optional[str] = None
    source_model_id: Optional[str] = None
    source_detection_job_ids: Optional[list[str]] = None
    manifest_path: Optional[str] = None
    training_split_name: Optional[str] = None
    promoted_config: Optional[dict[str, Any]] = None
    source_comparison_context: Optional[dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


AutoresearchCandidateStatus = Literal[
    "imported",
    "promotable",
    "blocked",
    "training",
    "complete",
    "failed",
]


class AutoresearchCandidateImport(BaseModel):
    name: Optional[str] = None
    manifest_path: str
    best_run_path: str
    comparison_path: Optional[str] = None
    top_false_positives_path: Optional[str] = None
    source_model_id_override: Optional[str] = None
    source_model_name_override: Optional[str] = None


class AutoresearchCandidateArtifactPaths(BaseModel):
    manifest_path: str
    best_run_path: str
    comparison_path: Optional[str] = None
    top_false_positives_path: Optional[str] = None


class AutoresearchCandidateSummaryOut(BaseModel):
    id: str
    name: str
    status: AutoresearchCandidateStatus
    phase: Optional[str] = None
    objective_name: Optional[str] = None
    threshold: Optional[float] = None
    comparison_target: Optional[str] = None
    source_model_id: Optional[str] = None
    source_model_name: Optional[str] = None
    is_reproducible_exact: bool
    promoted_config: dict[str, Any]
    best_run_metrics: Optional[dict[str, Any]] = None
    split_metrics: Optional[dict[str, Any]] = None
    metric_deltas: Optional[dict[str, Any]] = None
    replay_summary: Optional[dict[str, Any]] = None
    source_counts: Optional[dict[str, Any]] = None
    warnings: list[str] = Field(default_factory=list)
    training_job_id: Optional[str] = None
    new_model_id: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class AutoresearchCandidateDetailOut(AutoresearchCandidateSummaryOut):
    artifact_paths: AutoresearchCandidateArtifactPaths
    source_model_metadata: Optional[dict[str, Any]] = None
    top_false_positives_preview: Optional[dict[str, Any]] = None
    prediction_disagreements_preview: Optional[dict[str, Any]] = None
    replay_verification: Optional[dict[str, Any]] = None


class AutoresearchCandidateTrainingJobCreate(BaseModel):
    new_model_name: str
    notes: Optional[str] = None


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
    training_source_mode: Literal[
        "embedding_sets", "autoresearch_candidate", "detection_manifest"
    ] = "embedding_sets"
    source_candidate_id: Optional[str] = None
    source_model_id: Optional[str] = None
    promotion_provenance: Optional[dict[str, Any]] = None
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
    window_selection: Optional[Literal["nms", "prominence", "tiling"]] = None
    min_prominence: Optional[float] = None
    max_logit_drop: Optional[float] = None

    @model_validator(mode="before")
    @classmethod
    def _reject_detection_mode(cls, data: Any) -> Any:
        if isinstance(data, dict) and "detection_mode" in data:
            raise ValueError(DETECTION_MODE_CREATE_ERROR)
        return data

    @model_validator(mode="after")
    def _validate_thresholds(self):
        if self.high_threshold < self.low_threshold:
            raise ValueError("high_threshold must be >= low_threshold")
        if self.hop_seconds <= 0:
            raise ValueError("hop_seconds must be positive")
        if self.min_prominence is not None and self.min_prominence <= 0:
            raise ValueError("min_prominence must be > 0")
        if self.max_logit_drop is not None and self.max_logit_drop <= 0:
            raise ValueError("max_logit_drop must be > 0")
        return self


class DetectionJobOut(BaseModel):
    id: str
    status: str
    classifier_model_id: str
    audio_folder: Optional[str] = None
    confidence_threshold: float
    hop_seconds: float
    high_threshold: float
    low_threshold: float
    detection_mode: Optional[str] = None
    window_selection: Optional[str] = None
    min_prominence: Optional[float] = None
    max_logit_drop: Optional[float] = None
    output_row_store_path: Optional[str] = None
    result_summary: Optional[dict[str, Any]] = None
    error_message: Optional[str] = None
    files_processed: Optional[int] = None
    files_total: Optional[int] = None
    extract_status: Optional[str] = None
    extract_error: Optional[str] = None
    extract_summary: Optional[dict[str, Any]] = None
    # Hydrophone fields
    hydrophone_id: Optional[str] = None
    hydrophone_name: Optional[str] = None
    start_timestamp: Optional[float] = None
    end_timestamp: Optional[float] = None
    segments_processed: Optional[int] = None
    segments_total: Optional[int] = None
    time_covered_sec: Optional[float] = None
    alerts: Optional[list[dict[str, Any]]] = None
    local_cache_path: Optional[str] = None
    has_positive_labels: Optional[bool] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# ---- Hydrophone ----


class HydrophoneInfo(BaseModel):
    id: str
    name: str
    location: str
    provider_kind: str


class HydrophoneDetectionJobCreate(BaseModel):
    classifier_model_id: str
    hydrophone_id: str
    start_timestamp: float
    end_timestamp: float
    confidence_threshold: float = 0.5
    hop_seconds: float = 1.0
    high_threshold: float = 0.70
    low_threshold: float = 0.45
    local_cache_path: Optional[str] = None
    window_selection: Optional[Literal["nms", "prominence", "tiling"]] = None
    min_prominence: Optional[float] = None
    max_logit_drop: Optional[float] = None

    @model_validator(mode="before")
    @classmethod
    def _reject_detection_mode(cls, data: Any) -> Any:
        if isinstance(data, dict) and "detection_mode" in data:
            raise ValueError(DETECTION_MODE_CREATE_ERROR)
        return data

    @model_validator(mode="after")
    def _validate(self):
        if self.high_threshold < self.low_threshold:
            raise ValueError("high_threshold must be >= low_threshold")
        if self.hop_seconds <= 0:
            raise ValueError("hop_seconds must be positive")
        if self.end_timestamp <= self.start_timestamp:
            raise ValueError("end_timestamp must be > start_timestamp")
        max_range = 7 * 24 * 3600  # 7 days
        if self.end_timestamp - self.start_timestamp > max_range:
            raise ValueError("Time range must be <= 7 days")
        if self.min_prominence is not None and self.min_prominence <= 0:
            raise ValueError("min_prominence must be > 0")
        if self.max_logit_drop is not None and self.max_logit_drop <= 0:
            raise ValueError("max_logit_drop must be > 0")
        return self


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
    folder_path: Optional[str] = None
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


# ---- Retrain Workflows ----


# ---- Detection Embeddings ----


class EmbeddingStatusResponse(BaseModel):
    has_embeddings: bool
    count: int | None = None
    sync_needed: bool | None = None


class DetectionEmbeddingJobStatus(BaseModel):
    """Status row for a ``(detection_job_id, model_version)`` pair.

    ``status == "not_started"`` indicates no row exists yet.
    """

    detection_job_id: str
    model_version: str
    status: str
    rows_processed: int = 0
    rows_total: int | None = None
    error_message: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class DetectionEmbeddingJobOut(BaseModel):
    id: str
    status: str
    detection_job_id: str
    model_version: str
    mode: str | None = None
    progress_current: int | None = None
    progress_total: int | None = None
    rows_processed: int = 0
    rows_total: int | None = None
    error_message: str | None = None
    result_summary: str | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class EmbeddingJobListItem(BaseModel):
    id: str
    status: str
    detection_job_id: str
    model_version: str
    mode: str | None = None
    progress_current: int | None = None
    progress_total: int | None = None
    rows_processed: int = 0
    rows_total: int | None = None
    error_message: str | None = None
    result_summary: str | None = None
    created_at: datetime
    updated_at: datetime
    # Detection job context
    hydrophone_name: str | None = None
    audio_folder: str | None = None


class RetrainFolderInfo(BaseModel):
    model_id: str
    model_name: str
    model_version: str
    window_size_seconds: float
    target_sample_rate: int
    feature_config: Optional[dict[str, Any]] = None
    positive_folder_roots: list[str]
    negative_folder_roots: list[str]
    parameters: dict[str, Any]


class RetrainWorkflowCreate(BaseModel):
    source_model_id: str
    new_model_name: str
    parameters: Optional[dict[str, Any]] = None


class RetrainWorkflowOut(BaseModel):
    id: str
    status: str
    source_model_id: str
    new_model_name: str
    model_version: str
    window_size_seconds: float
    target_sample_rate: int
    feature_config: Optional[dict[str, Any]] = None
    parameters: Optional[dict[str, Any]] = None
    positive_folder_roots: list[str]
    negative_folder_roots: list[str]
    import_summary: Optional[dict[str, Any]] = None
    processing_job_ids: Optional[list[str]] = None
    processing_total: Optional[int] = None
    processing_complete: Optional[int] = None
    training_job_id: Optional[str] = None
    new_model_id: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
