"""Unit tests for sample_builder Stage 1 — annotation normalization."""

from humpback.classifier.raven_parser import RavenAnnotation
from humpback.sample_builder.normalize import normalize_annotations
from humpback.sample_builder.types import REASON_INVALID_ANNOTATION


def _ann(begin: float, end: float, call_type: str = "Whup") -> RavenAnnotation:
    return RavenAnnotation(
        selection=1,
        begin_time=begin,
        end_time=end,
        low_freq=200.0,
        high_freq=2000.0,
        call_type=call_type,
    )


class TestNormalizeAnnotations:
    def test_valid_annotation(self) -> None:
        results = normalize_annotations([_ann(1.0, 2.0)])
        assert len(results) == 1
        r = results[0]
        assert r.valid is True
        assert r.rejection_reason is None
        assert r.duration_sec == 1.0
        assert r.midpoint_sec == 1.5

    def test_midpoint_computation(self) -> None:
        results = normalize_annotations([_ann(10.0, 13.0)])
        assert results[0].midpoint_sec == 11.5
        assert results[0].duration_sec == 3.0

    def test_too_short_rejected(self) -> None:
        results = normalize_annotations([_ann(1.0, 1.1)])  # 0.1s < 0.3s default
        assert results[0].valid is False
        assert results[0].rejection_reason == REASON_INVALID_ANNOTATION

    def test_too_long_rejected(self) -> None:
        results = normalize_annotations([_ann(0.0, 5.0)])  # 5.0s > 4.0s default
        assert results[0].valid is False
        assert results[0].rejection_reason == REASON_INVALID_ANNOTATION

    def test_zero_duration_rejected(self) -> None:
        results = normalize_annotations([_ann(2.0, 2.0)])
        assert results[0].valid is False
        assert results[0].rejection_reason == REASON_INVALID_ANNOTATION

    def test_negative_duration_rejected(self) -> None:
        results = normalize_annotations([_ann(3.0, 2.0)])
        assert results[0].valid is False
        assert results[0].rejection_reason == REASON_INVALID_ANNOTATION

    def test_boundary_min_duration_accepted(self) -> None:
        # Exactly min_duration should be accepted
        results = normalize_annotations([_ann(1.0, 1.3)], min_duration=0.3)
        assert results[0].valid is True

    def test_boundary_max_duration_accepted(self) -> None:
        # Exactly max_duration should be accepted
        results = normalize_annotations([_ann(0.0, 4.0)], max_duration=4.0)
        assert results[0].valid is True

    def test_custom_bounds(self) -> None:
        ann = _ann(0.0, 0.5)
        # Default min=0.3 accepts, custom min=1.0 rejects
        assert normalize_annotations([ann])[0].valid is True
        assert normalize_annotations([ann], min_duration=1.0)[0].valid is False

    def test_mixed_valid_invalid(self) -> None:
        anns = [
            _ann(1.0, 2.0),  # valid (1.0s)
            _ann(5.0, 5.1),  # too short (0.1s)
            _ann(10.0, 12.0),  # valid (2.0s)
            _ann(20.0, 25.0),  # too long (5.0s)
        ]
        results = normalize_annotations(anns)
        assert len(results) == 4
        assert [r.valid for r in results] == [True, False, True, False]

    def test_empty_list(self) -> None:
        assert normalize_annotations([]) == []

    def test_original_preserved(self) -> None:
        ann = _ann(3.0, 4.5, call_type="Chirp")
        result = normalize_annotations([ann])[0]
        assert result.original is ann
        assert result.original.call_type == "Chirp"
