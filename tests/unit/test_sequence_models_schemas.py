"""Tests for Sequence Models PR 1 Pydantic schemas."""

import pytest
from pydantic import ValidationError

from humpback.schemas.sequence_models import (
    ContinuousEmbeddingJobCreate,
    ContinuousEmbeddingJobManifest,
    ContinuousEmbeddingSpanSummary,
    ExemplarRecord,
    MaskedTransformerNearestNeighborReportRequest,
    MaskedTransformerNearestNeighborReportResponse,
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


def test_exemplar_record_accepts_classify_event_extras():
    """Spec ADR-063: extras carries event_types (list) and event_confidence (dict)."""
    record = ExemplarRecord(
        sequence_id="0",
        position_in_sequence=3,
        audio_file_id=None,
        start_timestamp=10.0,
        end_timestamp=15.0,
        max_state_probability=0.93,
        exemplar_type="high_confidence",
        extras={
            "event_id": 7,
            "event_types": ["moan", "song"],
            "event_confidence": {"moan": 0.9, "song": 0.7},
        },
    )
    assert record.extras["event_types"] == ["moan", "song"]
    assert record.extras["event_confidence"] == {"moan": 0.9, "song": 0.7}

    background = ExemplarRecord(
        sequence_id="0",
        position_in_sequence=4,
        audio_file_id=None,
        start_timestamp=15.0,
        end_timestamp=20.0,
        max_state_probability=0.5,
        exemplar_type="boundary",
        extras={"event_id": None, "event_types": [], "event_confidence": {}},
    )
    assert background.extras["event_types"] == []
    assert background.extras["event_confidence"] == {}


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


def test_nearest_neighbor_report_request_defaults():
    payload = MaskedTransformerNearestNeighborReportRequest()

    assert payload.embedding_space == "contextual"
    assert payload.samples == 50
    assert payload.topn == 10
    assert payload.retrieval_modes == [
        "unrestricted",
        "exclude_same_event",
        "exclude_same_event_and_region",
    ]
    assert payload.embedding_variants == [
        "raw_l2",
        "centered_l2",
        "remove_pc1",
        "remove_pc3",
        "remove_pc5",
        "remove_pc10",
        "whiten_pca",
    ]
    assert payload.include_query_rows is False
    assert payload.include_neighbor_rows is False


def test_nearest_neighbor_report_request_rejects_invalid_options():
    with pytest.raises(ValidationError):
        MaskedTransformerNearestNeighborReportRequest.model_validate(
            {"embedding_space": "bad"}
        )
    with pytest.raises(ValidationError):
        MaskedTransformerNearestNeighborReportRequest(samples=0)
    with pytest.raises(ValidationError):
        MaskedTransformerNearestNeighborReportRequest(topn=0)
    with pytest.raises(ValidationError):
        MaskedTransformerNearestNeighborReportRequest(retrieval_modes=[])
    with pytest.raises(ValidationError):
        MaskedTransformerNearestNeighborReportRequest(embedding_variants=[])
    with pytest.raises(ValidationError):
        MaskedTransformerNearestNeighborReportRequest.model_validate(
            {"retrieval_modes": ["exclude_same_region"]}
        )
    with pytest.raises(ValidationError):
        MaskedTransformerNearestNeighborReportRequest.model_validate(
            {"embedding_variants": ["raw"]}
        )


def test_nearest_neighbor_report_response_serializes_detail_rows():
    response = MaskedTransformerNearestNeighborReportResponse.model_validate(
        {
            "job": {
                "job_id": "mt-1",
                "status": "complete",
                "continuous_embedding_job_id": "cej-1",
                "event_classification_job_id": None,
                "region_detection_job_id": "rdj-1",
                "k_values": [100],
                "k": 100,
            },
            "options": {
                "embedding_space": "contextual",
                "samples": 1,
                "topn": 1,
                "seed": 1,
                "retrieval_modes": ["unrestricted"],
                "embedding_variants": ["raw_l2"],
                "include_query_rows": True,
                "include_neighbor_rows": True,
            },
            "artifacts": {"embedding_path": "/tmp/contextual_embeddings.parquet"},
            "label_coverage": {
                "embedding_rows": 2,
                "sampled_queries": 1,
                "human_labeled_query_pool_rows": 1,
                "human_labeled_effective_events": 1,
                "vocalization_correction_rows": 1,
                "human_label_chunk_counts": {"moan": 1},
                "human_label_event_counts": {"moan": 1},
                "corrections_by_type": {"add:moan": 1},
            },
            "results": {
                "unrestricted": {
                    "raw_l2": {
                        "same_human_label": 1.0,
                        "exact_human_label_set": 1.0,
                        "same_event": 0.0,
                        "same_region": 0.0,
                        "adjacent_1s": 0.0,
                        "nearby_5s": 0.0,
                        "same_token": 0.0,
                        "similar_duration": 1.0,
                        "without_human_label": 0.0,
                        "low_event_overlap": 0.0,
                        "avg_cosine": 0.9,
                        "median_cosine": 0.9,
                        "random_pair_percentiles": {"50": 0.1},
                        "verdicts": {"good": 1},
                        "label_specific_same_human_label": {
                            "moan": {
                                "query_count": 1,
                                "neighbor_count": 1,
                                "same_human_label": 1.0,
                            }
                        },
                    }
                }
            },
            "representative_good_queries": [
                {
                    "query_order": 1,
                    "query_idx": 0,
                    "query_region": "r1",
                    "query_chunk": 0,
                    "query_start_timestamp": 10.0,
                    "query_human_types": "moan",
                    "query_event_id": "E1",
                    "query_duration": 0.5,
                    "query_token": 3,
                    "neighbor_count": 1,
                    "same_human_label_rate": 1.0,
                    "exact_human_label_set_rate": 1.0,
                    "same_event_rate": 0.0,
                    "same_region_rate": 0.0,
                    "adjacent_1s_rate": 0.0,
                    "nearby_5s_rate": 0.0,
                    "same_token_rate": 0.0,
                    "similar_duration_rate": 1.0,
                    "neighbor_without_human_label_rate": 0.0,
                    "neighbor_low_event_overlap_rate": 0.0,
                    "avg_cosine": 0.9,
                    "verdict": "good",
                }
            ],
            "representative_risky_queries": [],
            "query_rows": [],
            "neighbor_rows": [
                {
                    "query_order": 1,
                    "query_idx": 0,
                    "rank": 1,
                    "neighbor_idx": 1,
                    "cosine": 0.9,
                    "query_region": "r1",
                    "neighbor_region": "r2",
                    "query_chunk": 0,
                    "neighbor_chunk": 0,
                    "center_delta_sec": 2.0,
                    "same_region": False,
                    "adjacent_1s": False,
                    "nearby_5s": False,
                    "query_human_types": "moan",
                    "neighbor_human_types": "moan",
                    "same_human_label": True,
                    "exact_human_label_set": True,
                    "query_event_id": "E1",
                    "neighbor_event_id": "E2",
                    "same_event": False,
                    "query_duration": 0.5,
                    "neighbor_duration": 0.5,
                    "similar_duration": True,
                    "query_token": 3,
                    "neighbor_token": 4,
                    "same_token": False,
                    "query_tier": "event_core",
                    "neighbor_tier": "event_core",
                    "query_overlap": 1.0,
                    "neighbor_overlap": 1.0,
                    "query_call_probability": 0.9,
                    "neighbor_call_probability": 0.8,
                    "query_start_timestamp": 10.0,
                    "neighbor_start_timestamp": 12.0,
                }
            ],
        }
    )

    dumped = response.model_dump()
    assert dumped["job"]["job_id"] == "mt-1"
    assert dumped["neighbor_rows"][0]["neighbor_region"] == "r2"
