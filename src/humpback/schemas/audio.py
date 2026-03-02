from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel


class AudioMetadataIn(BaseModel):
    tag_data: Optional[dict[str, Any]] = None
    visual_observations: Optional[dict[str, Any]] = None
    group_composition: Optional[dict[str, Any]] = None
    prey_density_proxy: Optional[dict[str, Any]] = None


class AudioMetadataOut(BaseModel):
    id: str
    audio_file_id: str
    tag_data: Optional[dict[str, Any]] = None
    visual_observations: Optional[dict[str, Any]] = None
    group_composition: Optional[dict[str, Any]] = None
    prey_density_proxy: Optional[dict[str, Any]] = None

    model_config = {"from_attributes": True}


class SpectrogramOut(BaseModel):
    window_index: int
    sample_rate: int
    window_size_seconds: float
    shape: list[int]
    data: list[list[float]]
    total_windows: int
    min_db: float
    max_db: float
    y_axis_hz: list[float] = []
    x_axis_seconds: list[float] = []


class EmbeddingSimilarityOut(BaseModel):
    embedding_set_id: str
    vector_dim: int
    num_windows: int
    row_indices: list[int]
    similarity_matrix: list[list[float]]


class AffectedClusteringJob(BaseModel):
    id: str
    status: str
    overlapping_embedding_set_ids: list[str]


class FolderDeletePreview(BaseModel):
    folder_path: str
    audio_file_count: int
    embedding_set_count: int
    processing_job_count: int
    affected_clustering_jobs: list[AffectedClusteringJob]
    has_clustering_conflicts: bool


class FolderDeleteResult(BaseModel):
    folder_path: str
    deleted_audio_files: int
    deleted_embedding_sets: int
    deleted_processing_jobs: int
    deleted_clustering_jobs: int


class AudioFileOut(BaseModel):
    id: str
    filename: str
    folder_path: str = ""
    checksum_sha256: str
    duration_seconds: Optional[float] = None
    sample_rate_original: Optional[int] = None
    created_at: datetime
    metadata: Optional[AudioMetadataOut] = None

    model_config = {"from_attributes": True}
