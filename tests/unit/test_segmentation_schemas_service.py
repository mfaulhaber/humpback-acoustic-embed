"""Tests for Pass 2 Pydantic schemas and service layer."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from humpback.models.call_parsing import (
    RegionDetectionJob,
    SegmentationModel,
)
from humpback.schemas.call_parsing import (
    CreateSegmentationJobRequest,
    SegmentationDecoderConfig,
    SegmentationTrainingConfig,
)
from humpback.services.call_parsing import (
    CallParsingFKError,
    CallParsingStateError,
    create_segmentation_job,
)


def test_training_config_defaults_roundtrip() -> None:
    cfg = SegmentationTrainingConfig()
    assert cfg.epochs == 30
    assert cfg.batch_size == 16
    assert cfg.learning_rate == pytest.approx(1e-3)
    assert cfg.weight_decay == pytest.approx(1e-4)
    assert cfg.early_stopping_patience == 5
    assert cfg.grad_clip == pytest.approx(1.0)
    assert cfg.seed == 42
    assert cfg.n_mels == 64
    assert cfg.conv_channels == [32, 64, 96, 128]
    assert cfg.gru_hidden == 64
    assert cfg.gru_layers == 2

    payload = cfg.model_dump_json()
    roundtripped = SegmentationTrainingConfig.model_validate_json(payload)
    assert roundtripped == cfg


def test_training_config_val_fraction_bounds() -> None:
    with pytest.raises(ValidationError):
        SegmentationTrainingConfig(val_fraction=-0.1)
    with pytest.raises(ValidationError):
        SegmentationTrainingConfig(val_fraction=1.0)
    SegmentationTrainingConfig(val_fraction=0.0)
    SegmentationTrainingConfig(val_fraction=0.5)


def test_decoder_config_defaults() -> None:
    cfg = SegmentationDecoderConfig()
    assert cfg.high_threshold == 0.5
    assert cfg.low_threshold == 0.3
    assert cfg.min_event_sec == 0.2
    assert cfg.merge_gap_sec == 0.1


def test_decoder_config_rejects_low_ge_high() -> None:
    with pytest.raises(ValidationError):
        SegmentationDecoderConfig(high_threshold=0.3, low_threshold=0.3)
    with pytest.raises(ValidationError):
        SegmentationDecoderConfig(high_threshold=0.2, low_threshold=0.4)


def test_decoder_config_rejects_out_of_range_thresholds() -> None:
    with pytest.raises(ValidationError):
        SegmentationDecoderConfig(high_threshold=1.5, low_threshold=0.3)
    with pytest.raises(ValidationError):
        SegmentationDecoderConfig(high_threshold=0.5, low_threshold=-0.1)


def test_decoder_config_rejects_negative_durations() -> None:
    with pytest.raises(ValidationError):
        SegmentationDecoderConfig(min_event_sec=-0.1)
    with pytest.raises(ValidationError):
        SegmentationDecoderConfig(merge_gap_sec=-0.1)


def test_create_segmentation_job_request_defaults() -> None:
    req = CreateSegmentationJobRequest(
        region_detection_job_id="rd-1",
        segmentation_model_id="sm-1",
    )
    assert req.region_detection_job_id == "rd-1"
    assert req.segmentation_model_id == "sm-1"
    assert req.parent_run_id is None
    assert req.config.high_threshold == 0.5


def test_create_segmentation_job_request_rejects_malformed_config() -> None:
    with pytest.raises(ValidationError):
        CreateSegmentationJobRequest.model_validate(
            {
                "region_detection_job_id": "rd-1",
                "segmentation_model_id": "sm-1",
                "config": {"high_threshold": 0.2, "low_threshold": 0.4},
            }
        )


# ---- Service-layer tests ------------------------------------------------


async def test_create_segmentation_job_happy_path(session) -> None:
    rd = RegionDetectionJob(id="rd-1", status="complete")
    sm = SegmentationModel(
        id="sm-1",
        name="tiny",
        model_family="pytorch_crnn",
        model_path="/tmp/nope.pt",
    )
    session.add_all([rd, sm])
    await session.commit()

    req = CreateSegmentationJobRequest(
        region_detection_job_id="rd-1",
        segmentation_model_id="sm-1",
    )
    job = await create_segmentation_job(session, req)
    await session.commit()

    assert job.region_detection_job_id == "rd-1"
    assert job.segmentation_model_id == "sm-1"
    assert job.status == "queued"
    assert job.config_json
    cfg = json.loads(job.config_json)
    assert cfg["high_threshold"] == 0.5


async def test_create_segmentation_job_missing_region_job(session) -> None:
    sm = SegmentationModel(
        id="sm-1",
        name="tiny",
        model_family="pytorch_crnn",
        model_path="/tmp/nope.pt",
    )
    session.add(sm)
    await session.commit()

    req = CreateSegmentationJobRequest(
        region_detection_job_id="missing", segmentation_model_id="sm-1"
    )
    with pytest.raises(CallParsingFKError) as exc:
        await create_segmentation_job(session, req)
    assert exc.value.field == "region_detection_job_id"


async def test_create_segmentation_job_missing_model(session) -> None:
    rd = RegionDetectionJob(id="rd-1", status="complete")
    session.add(rd)
    await session.commit()

    req = CreateSegmentationJobRequest(
        region_detection_job_id="rd-1", segmentation_model_id="missing"
    )
    with pytest.raises(CallParsingFKError) as exc:
        await create_segmentation_job(session, req)
    assert exc.value.field == "segmentation_model_id"


async def test_create_segmentation_job_upstream_not_complete(session) -> None:
    rd = RegionDetectionJob(id="rd-1", status="queued")
    sm = SegmentationModel(
        id="sm-1",
        name="tiny",
        model_family="pytorch_crnn",
        model_path="/tmp/nope.pt",
    )
    session.add_all([rd, sm])
    await session.commit()

    req = CreateSegmentationJobRequest(
        region_detection_job_id="rd-1", segmentation_model_id="sm-1"
    )
    with pytest.raises(CallParsingStateError):
        await create_segmentation_job(session, req)
