from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel


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


class DetectionJobOut(BaseModel):
    id: str
    status: str
    classifier_model_id: str
    audio_folder: str
    confidence_threshold: float
    output_tsv_path: Optional[str] = None
    result_summary: Optional[dict[str, Any]] = None
    error_message: Optional[str] = None
    extract_status: Optional[str] = None
    extract_error: Optional[str] = None
    extract_summary: Optional[dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}
