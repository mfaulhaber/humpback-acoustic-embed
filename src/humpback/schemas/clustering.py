from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel


class ClusteringJobCreate(BaseModel):
    embedding_set_ids: list[str]
    parameters: Optional[dict[str, Any]] = None


class ClusteringJobOut(BaseModel):
    id: str
    status: str
    embedding_set_ids: list[str]
    parameters: Optional[dict[str, Any]] = None
    error_message: Optional[str] = None
    metrics: Optional[dict[str, Any]] = None
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
