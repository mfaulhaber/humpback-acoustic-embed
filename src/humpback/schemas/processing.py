from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel


class ProcessingJobCreate(BaseModel):
    audio_file_id: str
    model_version: Optional[str] = None
    window_size_seconds: float = 5.0
    target_sample_rate: int = 32000
    feature_config: Optional[dict[str, Any]] = None


class ProcessingJobOut(BaseModel):
    id: str
    audio_file_id: str
    status: str
    encoding_signature: str
    model_version: str
    window_size_seconds: float
    target_sample_rate: int
    feature_config: Optional[dict[str, Any]] = None
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    skipped: bool = False

    model_config = {"from_attributes": True}


class EmbeddingSetOut(BaseModel):
    id: str
    audio_file_id: str
    encoding_signature: str
    model_version: str
    window_size_seconds: float
    target_sample_rate: int
    vector_dim: int
    parquet_path: str
    created_at: datetime

    model_config = {"from_attributes": True}
