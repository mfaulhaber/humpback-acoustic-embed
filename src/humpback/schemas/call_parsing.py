"""Pydantic schemas for the call parsing pipeline API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


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
    config_json: Optional[str] = None
    chunks_total: Optional[int] = None
    chunks_completed: Optional[int] = None
    windows_detected: Optional[int] = None
    trace_row_count: Optional[int] = None
    region_count: Optional[int] = None

    model_config = {"from_attributes": True}


class EventSegmentationJobSummary(_JobSummary):
    region_detection_job_id: str
    segmentation_model_id: Optional[str] = None
    config_json: Optional[str] = None
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


# ---- Pass 2: segmentation schemas ---------------------------------------


class SegmentationFeatureConfig(BaseModel):
    """Frozen feature-extractor parameters for the Pass 2 segmentation model.

    Matches the parameters documented in the Pass 2 design spec. The
    trainer and the event segmentation worker both instantiate a
    ``SegmentationFeatureConfig`` at the defaults; overriding is possible
    but not exposed on the public API surface.
    """

    sample_rate: int = 16000
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 64
    fmin: float = 20.0
    fmax: float = 4000.0
    normalize: str = "per_region_zscore"


class SegmentationTrainingConfig(BaseModel):
    """Hyperparameters for a segmentation training job.

    Defaults track ADR-050 — conservative CRNN at ~300k parameters,
    30 epochs, early stop patience 5, Adam 1e-3, weight decay 1e-4.
    """

    epochs: int = 30
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 5
    grad_clip: float = 1.0
    seed: int = 42
    val_fraction: float = 0.2
    # Model knobs (forwarded into SegmentationCRNN).
    n_mels: int = 64
    conv_channels: list[int] = Field(default_factory=lambda: [32, 64, 96, 128])
    gru_hidden: int = 64
    gru_layers: int = 2

    @field_validator("val_fraction")
    @classmethod
    def _val_fraction_range(cls, v: float) -> float:
        if not 0.0 <= v < 1.0:
            raise ValueError("val_fraction must satisfy 0.0 <= val_fraction < 1.0")
        return v


class SegmentationDecoderConfig(BaseModel):
    """Hysteresis decoder thresholds for the event segmentation worker."""

    high_threshold: float = 0.5
    low_threshold: float = 0.3
    min_event_sec: float = 0.2
    merge_gap_sec: float = 0.1

    @model_validator(mode="after")
    def _thresholds_valid(self) -> "SegmentationDecoderConfig":
        if not 0.0 <= self.low_threshold <= 1.0:
            raise ValueError("low_threshold must be in [0.0, 1.0]")
        if not 0.0 <= self.high_threshold <= 1.0:
            raise ValueError("high_threshold must be in [0.0, 1.0]")
        if self.low_threshold >= self.high_threshold:
            raise ValueError("low_threshold must be strictly less than high_threshold")
        if self.min_event_sec < 0:
            raise ValueError("min_event_sec must be >= 0")
        if self.merge_gap_sec < 0:
            raise ValueError("merge_gap_sec must be >= 0")
        return self


class CreateSegmentationTrainingJobRequest(BaseModel):
    """Request body for ``POST /call-parsing/segmentation-training-jobs``."""

    training_dataset_id: str
    config: SegmentationTrainingConfig = Field(
        default_factory=SegmentationTrainingConfig
    )


class CreateSegmentationJobRequest(BaseModel):
    """Request body for ``POST /call-parsing/segmentation-jobs``."""

    region_detection_job_id: str
    segmentation_model_id: str
    parent_run_id: Optional[str] = None
    config: SegmentationDecoderConfig = Field(default_factory=SegmentationDecoderConfig)


class CreateEventClassificationJobRequest(BaseModel):
    """Request body for ``POST /call-parsing/classification-jobs``."""

    event_segmentation_job_id: str
    vocalization_model_id: str
    parent_run_id: Optional[str] = None
    config: Optional[dict[str, Any]] = None


class SegmentationTrainingJobResponse(BaseModel):
    id: str
    status: str
    training_dataset_id: str
    config_json: str
    segmentation_model_id: Optional[str] = None
    result_summary: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class SegmentationModelResponse(BaseModel):
    id: str
    name: str
    model_family: str
    model_path: str
    config_json: Optional[str] = None
    training_job_id: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class SegmentationTrainingDatasetResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class SegmentationTrainingDatasetSummary(BaseModel):
    id: str
    name: str
    sample_count: int
    created_at: datetime


class CreateDatasetFromCorrectionsRequest(BaseModel):
    """Request body for creating a training dataset from boundary corrections."""

    segmentation_job_id: str
    name: Optional[str] = None
    description: Optional[str] = None


class CreateDatasetFromCorrectionsResponse(BaseModel):
    """Response for dataset-from-corrections creation."""

    id: str
    name: str
    sample_count: int
    created_at: datetime


# ---- Feedback training: correction schemas ---------------------------------


class BoundaryCorrection(BaseModel):
    """A single boundary correction for a Pass 2 segmentation event."""

    event_id: str
    region_id: str
    correction_type: str = Field(pattern=r"^(adjust|add|delete)$")
    start_sec: Optional[float] = None
    end_sec: Optional[float] = None

    @model_validator(mode="after")
    def _validate_fields(self) -> "BoundaryCorrection":
        if self.correction_type == "add":
            if self.start_sec is None or self.end_sec is None:
                raise ValueError("'add' corrections require start_sec and end_sec")
        if self.correction_type == "adjust":
            if self.start_sec is None or self.end_sec is None:
                raise ValueError("'adjust' corrections require start_sec and end_sec")
        if self.correction_type == "delete":
            if self.start_sec is not None or self.end_sec is not None:
                raise ValueError("'delete' corrections must not set start_sec/end_sec")
        if (
            self.start_sec is not None
            and self.end_sec is not None
            and self.end_sec <= self.start_sec
        ):
            raise ValueError("end_sec must be strictly after start_sec")
        return self


class BoundaryCorrectionRequest(BaseModel):
    """Batch upsert request for Pass 2 boundary corrections."""

    corrections: list[BoundaryCorrection]


class BoundaryCorrectionResponse(BaseModel):
    """A stored boundary correction row."""

    id: str
    event_segmentation_job_id: str
    event_id: str
    region_id: str
    correction_type: str
    start_sec: Optional[float] = None
    end_sec: Optional[float] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class TypeCorrection(BaseModel):
    """A single type correction for a Pass 3 classification event."""

    event_id: str
    type_name: Optional[str] = None


class TypeCorrectionRequest(BaseModel):
    """Batch upsert request for Pass 3 type corrections."""

    corrections: list[TypeCorrection]


class TypeCorrectionResponse(BaseModel):
    """A stored type correction row."""

    id: str
    event_classification_job_id: str
    event_id: str
    type_name: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# ---- Feedback training: training job schemas -------------------------------


class EventClassifierTrainingConfig(BaseModel):
    """Hyperparameters for a Pass 3 event classifier feedback training job.

    Defaults match ``event_classifier.trainer.EventClassifierTrainingConfig``.
    """

    epochs: int = 30
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 5
    grad_clip: float = 1.0
    seed: int = 42
    val_fraction: float = 0.2
    min_examples_per_type: int = 10

    @field_validator("val_fraction")
    @classmethod
    def _val_fraction_range(cls, v: float) -> float:
        if not 0.0 <= v < 1.0:
            raise ValueError("val_fraction must satisfy 0.0 <= val_fraction < 1.0")
        return v


class CreateClassifierTrainingJobRequest(BaseModel):
    """Request body for ``POST /call-parsing/classifier-training-jobs``."""

    source_job_ids: list[str] = Field(min_length=1)
    config: EventClassifierTrainingConfig = Field(
        default_factory=EventClassifierTrainingConfig
    )


class ClassifierTrainingJobResponse(BaseModel):
    """Response model for Pass 3 feedback training jobs."""

    id: str
    status: str
    source_job_ids: str
    config_json: Optional[str] = None
    vocalization_model_id: Optional[str] = None
    result_summary: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class ClassifierModelResponse(BaseModel):
    """Response model for ``GET /call-parsing/classifier-models``."""

    id: str
    name: str
    model_family: str
    input_mode: Optional[str] = None
    model_dir_path: Optional[str] = None
    vocabulary_snapshot: Optional[str] = None
    per_class_thresholds: Optional[str] = None
    per_class_metrics: Optional[str] = None
    training_summary: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}
