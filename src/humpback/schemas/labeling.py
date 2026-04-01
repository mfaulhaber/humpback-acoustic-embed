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
    start_utc: float
    end_utc: float
    label: str
    confidence: float | None = None
    source: str
    notes: str | None = None
    row_store_version_at_import: int | None = None
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


class OrphanedLabelDetail(BaseModel):
    id: str
    start_utc: float
    end_utc: float
    label: str


class RefreshPreviewResponse(BaseModel):
    matched_count: int
    orphaned_count: int
    orphaned_labels: list[OrphanedLabelDetail]
    current_version: int


class RefreshApplyResponse(BaseModel):
    deleted_count: int
    surviving_count: int
    current_version: int


class DetectionNeighborsRequest(BaseModel):
    start_utc: float
    end_utc: float
    top_k: int = Field(default=10, ge=1, le=100)
    metric: str = Field(default="cosine", pattern="^(cosine|euclidean)$")
    embedding_set_ids: list[str] | None = None
