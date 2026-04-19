"""Tests for classifier schema models — DetectionSourceInfo and TrainingDataSummaryResponse."""

from humpback.schemas.classifier import (
    DetectionSourceInfo,
    TrainingDataSummaryResponse,
)


def test_detection_source_info_round_trip():
    info = DetectionSourceInfo(
        detection_job_id="abc-123",
        hydrophone_name="Orcasound Lab",
        start_timestamp=1635638400.0,
        end_timestamp=1635724800.0,
        positive_count=248,
        negative_count=216,
    )
    data = info.model_dump()
    restored = DetectionSourceInfo.model_validate(data)
    assert restored.detection_job_id == "abc-123"
    assert restored.hydrophone_name == "Orcasound Lab"
    assert restored.start_timestamp == 1635638400.0
    assert restored.positive_count == 248


def test_detection_source_info_optional_fields():
    info = DetectionSourceInfo(detection_job_id="abc-123")
    assert info.hydrophone_name is None
    assert info.start_timestamp is None
    assert info.positive_count is None


def test_training_data_summary_without_detection_sources():
    resp = TrainingDataSummaryResponse(
        model_id="m1",
        model_name="test",
        positive_sources=[],
        negative_sources=[],
        total_positive=100,
        total_negative=50,
        balance_ratio=2.0,
        window_size_seconds=5.0,
    )
    assert resp.detection_sources is None


def test_training_data_summary_with_detection_sources():
    sources = [
        DetectionSourceInfo(
            detection_job_id="j1",
            hydrophone_name="Orcasound Lab",
            start_timestamp=1635638400.0,
            end_timestamp=1635724800.0,
            positive_count=100,
            negative_count=50,
        ),
    ]
    resp = TrainingDataSummaryResponse(
        model_id="m1",
        model_name="test",
        positive_sources=[],
        negative_sources=[],
        total_positive=100,
        total_negative=50,
        balance_ratio=2.0,
        window_size_seconds=5.0,
        detection_sources=sources,
    )
    assert resp.detection_sources is not None
    assert len(resp.detection_sources) == 1
    assert resp.detection_sources[0].hydrophone_name == "Orcasound Lab"
