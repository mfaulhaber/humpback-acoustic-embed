from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---- Vocabulary ----


class VocalizationTypeCreate(BaseModel):
    name: str
    description: str | None = None


class VocalizationTypeUpdate(BaseModel):
    name: str | None = None
    description: str | None = None


class VocalizationTypeOut(BaseModel):
    id: str
    name: str
    description: str | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class VocalizationTypeImportRequest(BaseModel):
    embedding_set_ids: list[str]


class VocalizationTypeImportResponse(BaseModel):
    added: list[str]
    skipped: list[str]


# ---- Training ----


class VocalizationTrainingSourceConfig(BaseModel):
    embedding_set_ids: list[str] = Field(default_factory=list)
    detection_job_ids: list[str] = Field(default_factory=list)


class VocalizationTrainingJobCreate(BaseModel):
    source_config: VocalizationTrainingSourceConfig
    parameters: Optional[dict[str, Any]] = None


class VocalizationTrainingJobOut(BaseModel):
    id: str
    status: str
    source_config: dict[str, Any]
    parameters: Optional[dict[str, Any]] = None
    vocalization_model_id: str | None = None
    result_summary: Optional[dict[str, Any]] = None
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# ---- Models ----


class VocalizationModelOut(BaseModel):
    id: str
    name: str
    model_dir_path: str
    vocabulary_snapshot: list[str]
    per_class_thresholds: dict[str, float]
    per_class_metrics: Optional[dict[str, Any]] = None
    training_summary: Optional[dict[str, Any]] = None
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


# ---- Inference ----


class VocalizationInferenceJobCreate(BaseModel):
    vocalization_model_id: str
    source_type: str = Field(pattern="^(detection_job|embedding_set|rescore)$")
    source_id: str


class VocalizationInferenceJobOut(BaseModel):
    id: str
    status: str
    vocalization_model_id: str
    source_type: str
    source_id: str
    output_path: str | None = None
    result_summary: Optional[dict[str, Any]] = None
    error_message: str | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class VocalizationTrainingSourceOut(BaseModel):
    source_config: dict[str, Any] | None = None
    parameters: dict[str, Any] | None = None


class VocalizationPredictionRow(BaseModel):
    filename: str
    start_sec: float
    end_sec: float
    start_utc: float | None = None
    end_utc: float | None = None
    scores: dict[str, float]
    tags: list[str]
