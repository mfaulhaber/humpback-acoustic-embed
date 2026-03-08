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


class ClusteringJobOut(BaseModel):
    id: str
    status: str
    embedding_set_ids: list[str]
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
    embedding_set_id: str
    embedding_row_index: int

    model_config = {"from_attributes": True}
