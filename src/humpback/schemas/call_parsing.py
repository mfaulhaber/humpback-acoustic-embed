"""Pydantic schemas for the call parsing pipeline API (Phase 0)."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class CallParsingRunCreate(BaseModel):
    """Request body for ``POST /call-parsing/runs``.

    Phase 0 only records the source id and an optional config snapshot
    — downstream passes will add model ids and per-pass parameters.
    """

    audio_source_id: str
    config_snapshot: Optional[str] = None


class _JobSummary(BaseModel):
    id: str
    status: str
    parent_run_id: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class RegionDetectionJobSummary(_JobSummary):
    audio_source_id: str
    model_config_id: Optional[str] = None
    classifier_model_id: Optional[str] = None
    trace_row_count: Optional[int] = None
    region_count: Optional[int] = None

    model_config = {"from_attributes": True}


class EventSegmentationJobSummary(_JobSummary):
    region_detection_job_id: str
    segmentation_model_id: Optional[str] = None
    event_count: Optional[int] = None

    model_config = {"from_attributes": True}


class EventClassificationJobSummary(_JobSummary):
    event_segmentation_job_id: str
    vocalization_model_id: Optional[str] = None
    typed_event_count: Optional[int] = None

    model_config = {"from_attributes": True}


class CallParsingRunResponse(BaseModel):
    id: str
    audio_source_id: str
    status: str
    config_snapshot: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    completed_at: Optional[datetime] = None
    region_detection_job: Optional[RegionDetectionJobSummary] = None
    event_segmentation_job: Optional[EventSegmentationJobSummary] = None
    event_classification_job: Optional[EventClassificationJobSummary] = None

    model_config = {"from_attributes": True}
