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
