from datetime import datetime

from pydantic import BaseModel, Field


class VocalizationLabelCreate(BaseModel):
    label: str
    confidence: float | None = None
    source: str = "manual"
    notes: str | None = None


class VocalizationLabelUpdate(BaseModel):
    label: str | None = None
    confidence: float | None = None
    notes: str | None = None


class VocalizationLabelOut(BaseModel):
    id: str
    detection_job_id: str
    row_id: str
    label: str
    confidence: float | None = None
    source: str
    notes: str | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class NeighborHit(BaseModel):
    score: float
    embedding_set_id: str
    row_index: int
    audio_file_id: str
    audio_filename: str
    audio_folder_path: str | None = None
    window_offset_seconds: float
    inferred_label: str | None = None


class DetectionNeighborsResponse(BaseModel):
    hits: list[NeighborHit]
    total_candidates: int


class LabelingSummary(BaseModel):
    total_rows: int
    labeled_rows: int
    unlabeled_rows: int
    label_distribution: dict[str, int]


class TrainingSummary(BaseModel):
    """Aggregate label stats across all detection jobs for training readiness."""

    labeled_job_ids: list[str]
    labeled_rows: int
    label_distribution: dict[str, int]


class DetectionNeighborsRequest(BaseModel):
    filename: str
    start_sec: float = Field(ge=0)
    end_sec: float = Field(gt=0)
    top_k: int = Field(default=10, ge=1, le=100)
    metric: str = Field(default="cosine", pattern="^(cosine|euclidean)$")
    embedding_set_ids: list[str] | None = None


# ---- Vocalization Classifier Training ----


class VocalizationTrainingJobCreate(BaseModel):
    name: str
    source_detection_job_ids: list[str]
    parameters: dict[str, object] | None = None


class VocalizationTrainingJobOut(BaseModel):
    id: str
    status: str
    name: str
    job_purpose: str
    source_detection_job_ids: list[str]
    classifier_model_id: str | None = None
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime


class VocalizationModelOut(BaseModel):
    id: str
    name: str
    model_version: str
    vector_dim: int
    classifier_purpose: str
    training_summary: dict[str, object] | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class PredictRequest(BaseModel):
    vocalization_model_id: str


class PredictionRow(BaseModel):
    row_id: str
    predicted_label: str
    confidence: float
    probabilities: dict[str, float]


# ---- Sub-Window Annotations ----


class AnnotationCreate(BaseModel):
    start_offset_sec: float = Field(ge=0)
    end_offset_sec: float = Field(gt=0)
    label: str
    notes: str | None = None


class AnnotationUpdate(BaseModel):
    start_offset_sec: float | None = Field(default=None, ge=0)
    end_offset_sec: float | None = Field(default=None, gt=0)
    label: str | None = None
    notes: str | None = None


class AnnotationOut(BaseModel):
    id: str
    detection_job_id: str
    row_id: str
    start_offset_sec: float
    end_offset_sec: float
    label: str
    notes: str | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# ---- Active Learning ----


class ActiveLearningCycleRequest(BaseModel):
    vocalization_model_id: str
    detection_job_ids: list[str]
    name: str


class ActiveLearningCycleResponse(BaseModel):
    training_job_id: str
    status: str


class UncertaintyQueueRow(BaseModel):
    row_id: str
    filename: str
    start_sec: float
    end_sec: float
    avg_confidence: float
    peak_confidence: float
    predicted_label: str | None = None
    prediction_confidence: float | None = None
    probabilities: dict[str, float] | None = None


class ConvergenceMetrics(BaseModel):
    cycles_completed: int
    label_distribution: dict[str, int]
    accuracy_trend: list[float]
    uncertainty_histogram: list[dict[str, object]]
