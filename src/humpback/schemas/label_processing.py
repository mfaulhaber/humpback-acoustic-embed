from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel


class LabelProcessingJobCreate(BaseModel):
    workflow: str = "score_based"
    classifier_model_id: Optional[str] = None
    annotation_folder: str
    audio_folder: str
    output_root: str
    parameters: Optional[dict[str, Any]] = None


class LabelProcessingJobOut(BaseModel):
    id: str
    status: str
    workflow: str = "score_based"
    classifier_model_id: Optional[str] = None
    annotation_folder: str
    audio_folder: str
    output_root: str
    parameters: Optional[dict[str, Any]] = None
    files_processed: Optional[int] = None
    files_total: Optional[int] = None
    annotations_total: Optional[int] = None
    result_summary: Optional[dict[str, Any]] = None
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class PairedFile(BaseModel):
    annotation_file: str
    audio_file: str
    annotation_count: int


class LabelProcessingPreview(BaseModel):
    paired_files: list[PairedFile]
    total_annotations: int
    call_type_distribution: dict[str, int]
    unpaired_annotations: list[str]
    unpaired_audio: list[str]
