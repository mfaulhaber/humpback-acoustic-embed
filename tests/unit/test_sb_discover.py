"""Unit tests for sample_builder Stage 3 — background fragment discovery."""

from humpback.classifier.raven_parser import RavenAnnotation
from humpback.sample_builder.discover import _find_gaps, discover_background_fragments
from humpback.sample_builder.types import (
    ExclusionMap,
    NormalizedAnnotation,
    ProtectedInterval,
)


def _valid(begin: float, end: float) -> NormalizedAnnotation:
    ann = RavenAnnotation(
        selection=1,
        begin_time=begin,
        end_time=end,
        low_freq=200.0,
        high_freq=2000.0,
        call_type="Whup",
    )
    return NormalizedAnnotation(
        original=ann,
        midpoint_sec=(begin + end) / 2.0,
        duration_sec=end - begin,
        valid=True,
    )


def _pi(start: float, end: float) -> ProtectedInterval:
    return ProtectedInterval(start_sec=start, end_sec=end, annotation_index=0)


class TestFindGaps:
    def test_no_protected_intervals(self) -> None:
        em = ExclusionMap([])
        gaps = _find_gaps(em, 0.0, 30.0)
        assert gaps == [(0.0, 30.0)]

    def test_single_interval_in_middle(self) -> None:
        em = ExclusionMap([_pi(10.0, 15.0)])
        gaps = _find_gaps(em, 0.0, 30.0)
        assert gaps == [(0.0, 10.0), (15.0, 30.0)]

    def test_interval_at_start(self) -> None:
        em = ExclusionMap([_pi(0.0, 5.0)])
        gaps = _find_gaps(em, 0.0, 30.0)
        assert gaps == [(5.0, 30.0)]

    def test_interval_at_end(self) -> None:
        em = ExclusionMap([_pi(25.0, 30.0)])
        gaps = _find_gaps(em, 0.0, 30.0)
        assert gaps == [(0.0, 25.0)]

    def test_fully_covered(self) -> None:
        em = ExclusionMap([_pi(0.0, 30.0)])
        gaps = _find_gaps(em, 0.0, 30.0)
        assert gaps == []

    def test_multiple_intervals(self) -> None:
        em = ExclusionMap([_pi(5.0, 8.0), _pi(12.0, 15.0), _pi(20.0, 22.0)])
        gaps = _find_gaps(em, 0.0, 30.0)
        assert gaps == [(0.0, 5.0), (8.0, 12.0), (15.0, 20.0), (22.0, 30.0)]

    def test_search_window_subset(self) -> None:
        em = ExclusionMap([_pi(5.0, 8.0), _pi(12.0, 15.0)])
        gaps = _find_gaps(em, 6.0, 14.0)
        # 6.0 inside first interval, so first gap starts at 8.0
        assert gaps == [(8.0, 12.0)]

    def test_intervals_outside_window_ignored(self) -> None:
        em = ExclusionMap([_pi(1.0, 3.0), _pi(50.0, 55.0)])
        gaps = _find_gaps(em, 10.0, 40.0)
        assert gaps == [(10.0, 40.0)]


class TestDiscoverBackgroundFragments:
    def test_simple_discovery(self) -> None:
        # Target at 15-16s, protected [14, 17] with 1s guard
        target = _valid(15.0, 16.0)
        em = ExclusionMap([_pi(14.0, 17.0)])
        frags = discover_background_fragments(target, em, 30.0)
        assert len(frags) == 2
        # Closest fragment first — both equidistant from midpoint 15.5
        # [0, 14] has distance 1.5 from midpoint; [17, 30] has distance 1.5
        for f in frags:
            assert f.duration_sec >= 0.5

    def test_fragment_sorted_by_distance(self) -> None:
        # Target at 20-21s (midpoint 20.5), two gaps at different distances
        target = _valid(20.0, 21.0)
        em = ExclusionMap([_pi(5.0, 8.0), _pi(19.0, 22.0)])
        frags = discover_background_fragments(target, em, 30.0)
        # [22, 30] dist=1.5, [8, 19] dist=1.5, [0, 5] dist=15.5
        distances = [f.distance_from_target for f in frags]
        assert distances == sorted(distances)

    def test_min_fragment_filter(self) -> None:
        target = _valid(15.0, 16.0)
        # Small gap of 0.3s between two protected intervals
        em = ExclusionMap([_pi(10.0, 14.7), _pi(15.0, 20.0)])
        frags = discover_background_fragments(
            target,
            em,
            30.0,
            min_fragment_sec=0.5,
        )
        # 0.3s gap at [14.7, 15.0] should be filtered out
        for f in frags:
            assert f.duration_sec >= 0.5

    def test_max_search_radius(self) -> None:
        target = _valid(50.0, 51.0)  # midpoint 50.5
        em = ExclusionMap([_pi(49.0, 52.0)])
        frags = discover_background_fragments(
            target,
            em,
            100.0,
            max_search_radius_sec=5.0,
        )
        # Search window: [45.5, 55.5]
        for f in frags:
            assert f.start_sec >= 45.5
            assert f.end_sec <= 55.5

    def test_no_fragments_when_fully_protected(self) -> None:
        target = _valid(5.0, 6.0)
        em = ExclusionMap([_pi(0.0, 100.0)])
        frags = discover_background_fragments(target, em, 100.0)
        assert frags == []

    def test_empty_exclusion_map(self) -> None:
        target = _valid(15.0, 16.0)
        em = ExclusionMap([])
        frags = discover_background_fragments(target, em, 30.0)
        # Entire search window is one fragment
        assert len(frags) == 1
        assert frags[0].distance_from_target == 0.0

    def test_fragment_distance_when_midpoint_inside_gap(self) -> None:
        # Target at 15-16, gap at 10-20 (no protection nearby)
        target = _valid(15.0, 16.0)
        em = ExclusionMap([_pi(0.0, 5.0), _pi(25.0, 30.0)])
        frags = discover_background_fragments(target, em, 30.0)
        # [5, 25] contains midpoint 15.5 → distance 0
        assert any(f.distance_from_target == 0.0 for f in frags)

    def test_search_clamps_to_audio_bounds(self) -> None:
        target = _valid(2.0, 3.0)  # midpoint 2.5
        em = ExclusionMap([_pi(1.0, 4.0)])
        frags = discover_background_fragments(
            target,
            em,
            10.0,
            max_search_radius_sec=100.0,
        )
        # Search window clamped to [0, 10]
        for f in frags:
            assert f.start_sec >= 0.0
            assert f.end_sec <= 10.0

    def test_dense_annotations_few_fragments(self) -> None:
        # Many annotations close together — limited background available
        target = _valid(15.0, 16.0)
        intervals = [
            _pi(0.0, 5.0),
            _pi(6.0, 9.0),
            _pi(10.0, 14.0),
            _pi(14.5, 17.5),
            _pi(18.0, 22.0),
            _pi(23.0, 28.0),
        ]
        em = ExclusionMap(intervals)
        frags = discover_background_fragments(target, em, 30.0, min_fragment_sec=0.3)
        # Only tiny gaps survive: [5,6]=1s, [9,10]=1s, [14,14.5]=0.5s,
        # [17.5,18]=0.5s, [22,23]=1s, [28,30]=2s
        assert len(frags) > 0
        total_bg = sum(f.duration_sec for f in frags)
        assert total_bg > 0
