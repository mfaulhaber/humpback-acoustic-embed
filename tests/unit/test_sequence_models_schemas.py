"""Tests for Sequence Models PR 1 Pydantic schemas."""

import pytest
from pydantic import ValidationError

from humpback.schemas.sequence_models import (
    ContinuousEmbeddingJobCreate,
    ContinuousEmbeddingJobManifest,
    ContinuousEmbeddingSpanSummary,
    ExemplarRecord,
    OverlayPoint,
)


def test_continuous_embedding_job_create_defaults():
    payload = ContinuousEmbeddingJobCreate(event_segmentation_job_id="sj-1")
    assert payload.model_version == "surfperch-tensorflow2"
    assert payload.hop_seconds == 1.0
    assert payload.pad_seconds == 2.0


def test_continuous_embedding_job_create_rejects_non_positive_hop():
    with pytest.raises(ValidationError) as excinfo:
        ContinuousEmbeddingJobCreate(event_segmentation_job_id="sj-1", hop_seconds=0)
    assert "hop_seconds" in str(excinfo.value)

    with pytest.raises(ValidationError):
        ContinuousEmbeddingJobCreate(event_segmentation_job_id="sj-1", hop_seconds=-0.5)


def test_continuous_embedding_job_create_rejects_negative_pad():
    with pytest.raises(ValidationError) as excinfo:
        ContinuousEmbeddingJobCreate(event_segmentation_job_id="sj-1", pad_seconds=-1.0)
    assert "pad_seconds" in str(excinfo.value)


def test_continuous_embedding_job_create_accepts_zero_pad():
    payload = ContinuousEmbeddingJobCreate(
        event_segmentation_job_id="sj-1", pad_seconds=0.0
    )
    assert payload.pad_seconds == 0.0


def test_continuous_embedding_job_manifest_round_trip():
    manifest = ContinuousEmbeddingJobManifest(
        job_id="cej-1",
        model_version="surfperch-tensorflow2",
        vector_dim=1280,
        window_size_seconds=5.0,
        hop_seconds=1.0,
        pad_seconds=2.0,
        target_sample_rate=32000,
        total_events=3,
        merged_spans=2,
        total_windows=42,
        spans=[
            ContinuousEmbeddingSpanSummary(
                merged_span_id=0,
                event_id="evt-1",
                region_id="r1",
                start_timestamp=100.0,
                end_timestamp=120.0,
                window_count=20,
            )
        ],
    )
    data = manifest.model_dump()
    restored = ContinuousEmbeddingJobManifest.model_validate(data)
    assert restored.vector_dim == 1280
    assert restored.spans[0].window_count == 20


def test_exemplar_record_accepts_null_audio_file_id_for_hydrophone_jobs():
    record = ExemplarRecord(
        sequence_id="0",
        position_in_sequence=3,
        audio_file_id=None,
        start_timestamp=10.0,
        end_timestamp=15.0,
        max_state_probability=0.93,
        exemplar_type="high_confidence",
    )

    assert record.audio_file_id is None
    assert record.extras == {}


def test_exemplar_record_carries_extras_for_crnn_tier():
    record = ExemplarRecord(
        sequence_id="region-uuid-1",
        position_in_sequence=17,
        audio_file_id=42,
        start_timestamp=10.0,
        end_timestamp=10.25,
        max_state_probability=0.93,
        exemplar_type="high_confidence",
        extras={"tier": "event_core"},
    )
    assert record.extras["tier"] == "event_core"


def test_overlay_point_validates_unified_shape_and_rejects_missing_sequence_id():
    OverlayPoint(
        sequence_id="0",
        position_in_sequence=1,
        start_timestamp=10.0,
        end_timestamp=15.0,
        pca_x=0.0,
        pca_y=0.0,
        umap_x=0.0,
        umap_y=0.0,
        viterbi_state=2,
        max_state_probability=0.5,
    )
    with pytest.raises(ValidationError):
        OverlayPoint.model_validate(
            {
                "position_in_sequence": 1,
                "start_timestamp": 10.0,
                "end_timestamp": 15.0,
                "pca_x": 0.0,
                "pca_y": 0.0,
                "umap_x": 0.0,
                "umap_y": 0.0,
                "viterbi_state": 2,
                "max_state_probability": 0.5,
            }
        )
