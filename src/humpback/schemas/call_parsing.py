"""Pydantic schemas for the call parsing pipeline API."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class RegionDetectionConfig(BaseModel):
    """Tuning knobs for a Pass 1 region detection job.

    Defaults mirror the existing detector's hysteresis behavior
    (``window_size_seconds=5.0``, ``hop_seconds=1.0``,
    ``high_threshold=0.70``, ``low_threshold=0.45``) plus the Pass 1
    algorithmic defaults from ADR-049: symmetric 1.0 s padding, no
    minimum-duration filter, and 30-minute hydrophone streaming chunks.
    """

    # Detector knobs — passed through to ``score_audio_windows`` and
    # ``merge_detection_events``.
    window_size_seconds: float = 5.0
    hop_seconds: float = 1.0
    high_threshold: float = 0.70
    low_threshold: float = 0.45

    # Region-shaping knobs — consumed by ``decode_regions``.
    padding_sec: float = 1.0
    min_region_duration_sec: float = 0.0

    # Streaming control — only used when the job's source is a hydrophone
    # range. Audio-file jobs load the entire buffer in one shot.
    stream_chunk_sec: float = 1800.0


class CreateRegionJobRequest(BaseModel):
    """Request body for ``POST /call-parsing/region-jobs`` and for
    ``POST /call-parsing/runs`` (same shape; the run POST creates a
    parent row and forwards the source + config to the Pass 1 child).

    Source identity is encoded as ``audio_file_id`` XOR the hydrophone
    triple (``hydrophone_id`` + ``start_timestamp`` + ``end_timestamp``).
    The exactly-one-of invariant is enforced by ``_exactly_one_source``;
    this is the same pattern used by the existing ``DetectionJob`` API.
    """

    audio_file_id: Optional[str] = None
    hydrophone_id: Optional[str] = None
    start_timestamp: Optional[float] = None
    end_timestamp: Optional[float] = None

    model_config_id: str
    classifier_model_id: str
    parent_run_id: Optional[str] = None

    config: RegionDetectionConfig = Field(default_factory=RegionDetectionConfig)

    @model_validator(mode="after")
    def _exactly_one_source(self) -> "CreateRegionJobRequest":
        has_file = self.audio_file_id is not None
        has_hydro = all(
            v is not None
            for v in (
                self.hydrophone_id,
                self.start_timestamp,
                self.end_timestamp,
            )
        )
        any_hydro_field = any(
            v is not None
            for v in (
                self.hydrophone_id,
                self.start_timestamp,
                self.end_timestamp,
            )
        )
        if has_file and any_hydro_field:
            raise ValueError(
                "Provide either audio_file_id or the hydrophone triple, not both"
            )
        if not has_file and not has_hydro:
            raise ValueError(
                "Provide exactly one of audio_file_id or "
                "(hydrophone_id, start_timestamp, end_timestamp)"
            )
        if (
            has_hydro
            and self.start_timestamp is not None
            and self.end_timestamp is not None
            and self.end_timestamp <= self.start_timestamp
        ):
            raise ValueError("end_timestamp must be strictly after start_timestamp")
        return self


# POST /call-parsing/runs reuses the Pass 1 request shape. ``parent_run_id``
# is ignored at the run-creation endpoint (the service always populates it
# from the newly-created parent row), but leaving it on the model keeps the
# single-shape contract clean.
CallParsingRunCreate = CreateRegionJobRequest


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
    audio_file_id: Optional[str] = None
    hydrophone_id: Optional[str] = None
    start_timestamp: Optional[float] = None
    end_timestamp: Optional[float] = None
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
    audio_file_id: Optional[str] = None
    hydrophone_id: Optional[str] = None
    start_timestamp: Optional[float] = None
    end_timestamp: Optional[float] = None
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
