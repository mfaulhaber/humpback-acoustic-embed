"""Unit tests for sample_builder Stage 2 — exclusion map."""

from humpback.classifier.raven_parser import RavenAnnotation
from humpback.sample_builder.exclusion import build_exclusion_map
from humpback.sample_builder.types import (
    ExclusionMap,
    NormalizedAnnotation,
    ProtectedInterval,
)


def _ann(begin: float, end: float) -> RavenAnnotation:
    return RavenAnnotation(
        selection=1,
        begin_time=begin,
        end_time=end,
        low_freq=200.0,
        high_freq=2000.0,
        call_type="Whup",
    )


def _valid(begin: float, end: float) -> NormalizedAnnotation:
    ann = _ann(begin, end)
    return NormalizedAnnotation(
        original=ann,
        midpoint_sec=(begin + end) / 2.0,
        duration_sec=end - begin,
        valid=True,
    )


def _invalid(begin: float, end: float) -> NormalizedAnnotation:
    ann = _ann(begin, end)
    return NormalizedAnnotation(
        original=ann,
        midpoint_sec=(begin + end) / 2.0,
        duration_sec=end - begin,
        valid=False,
        rejection_reason="invalid",
    )


class TestBuildExclusionMap:
    def test_single_annotation(self) -> None:
        em = build_exclusion_map([_valid(5.0, 6.0)], guard_band_sec=1.0)
        assert len(em.protected_intervals) == 1
        iv = em.protected_intervals[0]
        assert iv.start_sec == 4.0  # 5.0 - 1.0
        assert iv.end_sec == 7.0  # 6.0 + 1.0
        assert iv.annotation_index == 0

    def test_guard_band_clamps_to_zero(self) -> None:
        em = build_exclusion_map([_valid(0.3, 1.0)], guard_band_sec=1.0)
        assert em.protected_intervals[0].start_sec == 0.0

    def test_non_overlapping_annotations(self) -> None:
        anns = [_valid(2.0, 3.0), _valid(10.0, 11.0)]
        em = build_exclusion_map(anns, guard_band_sec=1.0)
        assert len(em.protected_intervals) == 2
        assert em.protected_intervals[0].start_sec == 1.0
        assert em.protected_intervals[0].end_sec == 4.0
        assert em.protected_intervals[1].start_sec == 9.0
        assert em.protected_intervals[1].end_sec == 12.0

    def test_overlapping_annotations_merged(self) -> None:
        # Annotations 2-3s and 3.5-4.5s with 1s guard → [1,4] and [2.5,5.5] overlap
        anns = [_valid(2.0, 3.0), _valid(3.5, 4.5)]
        em = build_exclusion_map(anns, guard_band_sec=1.0)
        assert len(em.protected_intervals) == 1
        iv = em.protected_intervals[0]
        assert iv.start_sec == 1.0
        assert iv.end_sec == 5.5

    def test_adjacent_annotations_merged(self) -> None:
        # guard bands make them exactly adjacent → merged
        anns = [_valid(2.0, 3.0), _valid(5.0, 6.0)]
        em = build_exclusion_map(anns, guard_band_sec=1.0)
        # [1.0, 4.0] and [4.0, 7.0] → start<=end → merged
        assert len(em.protected_intervals) == 1
        assert em.protected_intervals[0].start_sec == 1.0
        assert em.protected_intervals[0].end_sec == 7.0

    def test_invalid_annotations_excluded(self) -> None:
        anns = [_invalid(2.0, 3.0), _valid(10.0, 11.0)]
        em = build_exclusion_map(anns, guard_band_sec=1.0)
        assert len(em.protected_intervals) == 1
        assert em.protected_intervals[0].start_sec == 9.0

    def test_all_invalid_returns_empty(self) -> None:
        em = build_exclusion_map([_invalid(2.0, 3.0)], guard_band_sec=1.0)
        assert len(em.protected_intervals) == 0

    def test_empty_list(self) -> None:
        em = build_exclusion_map([], guard_band_sec=1.0)
        assert len(em.protected_intervals) == 0

    def test_zero_guard_band(self) -> None:
        em = build_exclusion_map([_valid(5.0, 6.0)], guard_band_sec=0.0)
        assert em.protected_intervals[0].start_sec == 5.0
        assert em.protected_intervals[0].end_sec == 6.0


class TestExclusionMapOverlaps:
    def test_overlaps_inside(self) -> None:
        em = ExclusionMap(
            [
                _make_pi(4.0, 7.0),
            ]
        )
        assert em.overlaps(5.0, 6.0) is True

    def test_overlaps_partial_left(self) -> None:
        em = ExclusionMap([_make_pi(4.0, 7.0)])
        assert em.overlaps(3.0, 5.0) is True

    def test_overlaps_partial_right(self) -> None:
        em = ExclusionMap([_make_pi(4.0, 7.0)])
        assert em.overlaps(6.0, 8.0) is True

    def test_no_overlap_before(self) -> None:
        em = ExclusionMap([_make_pi(4.0, 7.0)])
        assert em.overlaps(1.0, 4.0) is False

    def test_no_overlap_after(self) -> None:
        em = ExclusionMap([_make_pi(4.0, 7.0)])
        assert em.overlaps(7.0, 9.0) is False

    def test_overlap_enclosing(self) -> None:
        em = ExclusionMap([_make_pi(4.0, 7.0)])
        assert em.overlaps(3.0, 8.0) is True

    def test_no_overlap_gap_between(self) -> None:
        em = ExclusionMap([_make_pi(2.0, 4.0), _make_pi(8.0, 10.0)])
        assert em.overlaps(5.0, 7.0) is False

    def test_overlap_second_interval(self) -> None:
        em = ExclusionMap([_make_pi(2.0, 4.0), _make_pi(8.0, 10.0)])
        assert em.overlaps(9.0, 11.0) is True

    def test_empty_map_no_overlap(self) -> None:
        em = ExclusionMap([])
        assert em.overlaps(0.0, 100.0) is False


def _make_pi(start: float, end: float) -> ProtectedInterval:
    return ProtectedInterval(start_sec=start, end_sec=end, annotation_index=0)
