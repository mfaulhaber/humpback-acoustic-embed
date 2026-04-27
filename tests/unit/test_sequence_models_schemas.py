"""Tests for Sequence Models PR 1 Pydantic schemas."""

import pytest
from pydantic import ValidationError

from humpback.schemas.sequence_models import (
    ContinuousEmbeddingJobCreate,
    ContinuousEmbeddingJobManifest,
    ContinuousEmbeddingSpanSummary,
)


def test_continuous_embedding_job_create_defaults():
    payload = ContinuousEmbeddingJobCreate(region_detection_job_id="rj-1")
    assert payload.model_version == "surfperch-tensorflow2"
    assert payload.hop_seconds == 1.0
    assert payload.pad_seconds == 10.0


def test_continuous_embedding_job_create_rejects_non_positive_hop():
    with pytest.raises(ValidationError) as excinfo:
        ContinuousEmbeddingJobCreate(region_detection_job_id="rj-1", hop_seconds=0)
    assert "hop_seconds" in str(excinfo.value)

    with pytest.raises(ValidationError):
        ContinuousEmbeddingJobCreate(region_detection_job_id="rj-1", hop_seconds=-0.5)


def test_continuous_embedding_job_create_rejects_negative_pad():
    with pytest.raises(ValidationError) as excinfo:
        ContinuousEmbeddingJobCreate(region_detection_job_id="rj-1", pad_seconds=-1.0)
    assert "pad_seconds" in str(excinfo.value)


def test_continuous_embedding_job_create_accepts_zero_pad():
    payload = ContinuousEmbeddingJobCreate(
        region_detection_job_id="rj-1", pad_seconds=0.0
    )
    assert payload.pad_seconds == 0.0


def test_continuous_embedding_job_manifest_round_trip():
    manifest = ContinuousEmbeddingJobManifest(
        job_id="cej-1",
        model_version="surfperch-tensorflow2",
        vector_dim=1280,
        window_size_seconds=5.0,
        hop_seconds=1.0,
        pad_seconds=10.0,
        target_sample_rate=32000,
        total_regions=3,
        merged_spans=2,
        total_windows=42,
        spans=[
            ContinuousEmbeddingSpanSummary(
                merged_span_id=0,
                start_time_sec=100.0,
                end_time_sec=120.0,
                window_count=20,
                source_region_ids=["r1"],
            )
        ],
    )
    data = manifest.model_dump()
    restored = ContinuousEmbeddingJobManifest.model_validate(data)
    assert restored.vector_dim == 1280
    assert restored.spans[0].window_count == 20
