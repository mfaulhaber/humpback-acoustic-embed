from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, model_validator


class ClusteringJobCreate(BaseModel):
    embedding_set_ids: list[str]
    parameters: Optional[dict[str, Any]] = None
    refined_from_job_id: Optional[str] = None

    @model_validator(mode="after")
    def _validate_embedding_sets(self):
        if not self.embedding_set_ids:
            raise ValueError("At least one embedding set is required")
        return self


class VocalizationClusteringJobCreate(BaseModel):
    detection_job_ids: list[str]
    parameters: Optional[dict[str, Any]] = None

    @model_validator(mode="after")
    def _validate_detection_jobs(self):
        if not self.detection_job_ids:
            raise ValueError("At least one detection job is required")
        return self


class ClusteringEligibleDetectionJobOut(BaseModel):
    id: str
    hydrophone_name: Optional[str] = None
    start_timestamp: Optional[float] = None
    end_timestamp: Optional[float] = None
    detection_count: Optional[int] = None


class ClusteringJobOut(BaseModel):
    id: str
    status: str
    detection_job_ids: list[str]
    parameters: Optional[dict[str, Any]] = None
    error_message: Optional[str] = None
    metrics: Optional[dict[str, Any]] = None
    refined_from_job_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ClusterOut(BaseModel):
    id: str
    clustering_job_id: str
    cluster_label: int
    size: int
    metadata_summary: Optional[dict[str, Any]] = None

    model_config = {"from_attributes": True}


class ClusterAssignmentOut(BaseModel):
    id: str
    cluster_id: str
    source_id: str
    embedding_row_index: int

    model_config = {"from_attributes": True}


class VocalizationClusteringVisualizationOut(BaseModel):
    x: list[float]
    y: list[float]
    cluster_label: list[int]
    detection_job_id: list[str]
    embedding_row_index: list[int]
    audio_filename: list[str]
    audio_file_id: list[str]
    window_size_seconds: list[float]
    category: list[str]
    start_utc: Optional[list[float | None]] = None
