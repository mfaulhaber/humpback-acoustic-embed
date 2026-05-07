"""Tests for retained Sequence Models / Continuous Embedding schemas."""

import pytest
from pydantic import ValidationError

from humpback.schemas.sequence_models import (
    ContinuousEmbeddingJobCreate,
    ContinuousEmbeddingJobManifest,
    ContinuousEmbeddingRegionSummary,
    ContinuousEmbeddingSpanSummary,
    EventEncoderJobCreate,
    EventEncoderPoolingConfig,
    EventEncoderPreprocessingConfig,
)


def test_continuous_embedding_job_create_defaults():
    payload = ContinuousEmbeddingJobCreate(event_segmentation_job_id="sj-1")
    assert payload.model_version == "surfperch-tensorflow2"
    assert payload.hop_seconds == 1.0
    assert payload.pad_seconds == 2.0
    assert payload.event_source_mode == "raw"


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


def test_continuous_embedding_job_create_requires_source_id():
    with pytest.raises(ValidationError) as excinfo:
        ContinuousEmbeddingJobCreate()
    assert "event_segmentation_job_id is required" in str(excinfo.value)


def test_continuous_embedding_job_create_rejects_crnn_fields_on_surfperch():
    with pytest.raises(ValidationError) as excinfo:
        ContinuousEmbeddingJobCreate(
            event_segmentation_job_id="sj-1",
            chunk_size_seconds=0.25,
        )
    assert "CRNN-only fields cannot be set" in str(excinfo.value)


def test_continuous_embedding_job_create_requires_crnn_fields():
    with pytest.raises(ValidationError) as excinfo:
        ContinuousEmbeddingJobCreate(
            event_segmentation_job_id="sj-1",
            region_detection_job_id="rd-1",
        )
    assert "CRNN region-based source requires" in str(excinfo.value)


def test_continuous_embedding_job_create_accepts_crnn_source():
    payload = ContinuousEmbeddingJobCreate(
        event_segmentation_job_id="sj-1",
        region_detection_job_id="rd-1",
        crnn_segmentation_model_id="seg-model-1",
        chunk_size_seconds=0.25,
        chunk_hop_seconds=0.125,
        projection_kind="pca",
        projection_dim=64,
    )

    assert payload.region_detection_job_id == "rd-1"
    assert payload.projection_kind == "pca"
    assert payload.projection_dim == 64


def test_continuous_embedding_job_manifest_round_trip_surfperch():
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


def test_continuous_embedding_job_manifest_round_trip_crnn():
    manifest = ContinuousEmbeddingJobManifest(
        job_id="cej-crnn",
        model_version="crnn-call-parsing-pytorch",
        source_kind="region_crnn",
        vector_dim=64,
        target_sample_rate=32000,
        event_segmentation_job_id="sj-1",
        region_detection_job_id="rd-1",
        crnn_checkpoint_sha256="abc123",
        chunk_size_seconds=0.25,
        chunk_hop_seconds=0.125,
        projection_kind="pca",
        projection_dim=64,
        total_regions=2,
        total_chunks=10,
        regions=[
            ContinuousEmbeddingRegionSummary(
                region_id="r1",
                start_timestamp=10.0,
                end_timestamp=12.0,
                chunk_count=10,
            )
        ],
    )

    restored = ContinuousEmbeddingJobManifest.model_validate(manifest.model_dump())
    assert restored.source_kind == "region_crnn"
    assert restored.regions[0].region_id == "r1"
    assert restored.total_chunks == 10


def test_event_encoder_job_create_defaults():
    payload = EventEncoderJobCreate(
        event_segmentation_job_id="seg-1",
        continuous_embedding_job_id="cej-1",
    )

    assert payload.event_source_mode == "raw"
    assert payload.tokenizer_version == "crnn-event-encoder-v1"
    assert payload.pooling.enabled_pools == [
        "mean_pool",
        "top_k_pool",
        "start_pool",
        "middle_pool",
        "end_pool",
    ]
    assert payload.preprocessing.pca_dim == 128
    assert payload.k_values == [50, 100, 200]
    assert payload.random_seed == 0


def test_event_encoder_job_create_sorts_k_values():
    payload = EventEncoderJobCreate(
        event_segmentation_job_id="seg-1",
        continuous_embedding_job_id="cej-1",
        k_values=[200, 50, 100],
    )
    assert payload.k_values == [50, 100, 200]


def test_event_encoder_job_create_rejects_invalid_k_values():
    with pytest.raises(ValidationError, match="k_values"):
        EventEncoderJobCreate(
            event_segmentation_job_id="seg-1",
            continuous_embedding_job_id="cej-1",
            k_values=[50, 50],
        )
    with pytest.raises(ValidationError, match="k_values"):
        EventEncoderJobCreate(
            event_segmentation_job_id="seg-1",
            continuous_embedding_job_id="cej-1",
            k_values=[0],
        )


def test_event_encoder_pooling_config_rejects_invalid_values():
    with pytest.raises(ValidationError, match="enabled_pools"):
        EventEncoderPoolingConfig(enabled_pools=[])
    with pytest.raises(ValidationError, match="enabled_pools"):
        EventEncoderPoolingConfig(enabled_pools=["mean_pool", "mean_pool"])
    with pytest.raises(ValidationError, match="top_k_fraction"):
        EventEncoderPoolingConfig(top_k_fraction=0)
    with pytest.raises(ValidationError, match="min_overlap_fraction"):
        EventEncoderPoolingConfig(min_overlap_fraction=1.5)
    with pytest.raises(ValidationError, match="min_chunks_per_event"):
        EventEncoderPoolingConfig(min_chunks_per_event=0)


def test_event_encoder_preprocessing_restricts_pca_dim_and_weights():
    with pytest.raises(ValidationError, match="pca_dim"):
        EventEncoderPreprocessingConfig.model_validate({"pca_dim": 32})
    with pytest.raises(ValidationError, match="feature weights"):
        EventEncoderPreprocessingConfig(embedding_weight=-1.0)
    with pytest.raises(ValidationError, match="at least one feature weight"):
        EventEncoderPreprocessingConfig(
            embedding_weight=0.0,
            descriptor_weight=0.0,
        )


def test_event_encoder_job_create_accepts_effective_mode():
    payload = EventEncoderJobCreate(
        event_segmentation_job_id="seg-1",
        event_source_mode="effective",
        continuous_embedding_job_id="cej-1",
        preprocessing=EventEncoderPreprocessingConfig(pca_dim=64),
        k_values=[50],
    )

    assert payload.event_source_mode == "effective"
    assert payload.preprocessing.pca_dim == 64
    assert payload.k_values == [50]
