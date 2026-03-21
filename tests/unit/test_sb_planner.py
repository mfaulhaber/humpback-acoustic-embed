"""Unit tests for sample_builder Stage 6 — assembly planning."""

from humpback.classifier.raven_parser import RavenAnnotation
from humpback.sample_builder.planner import plan_assembly
from humpback.sample_builder.types import (
    REASON_INSUFFICIENT_BACKGROUND,
    BackgroundFragment,
    NormalizedAnnotation,
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


def _frag(start: float, end: float, distance: float = 0.0) -> BackgroundFragment:
    return BackgroundFragment(
        start_sec=start,
        end_sec=end,
        duration_sec=end - start,
        distance_from_target=distance,
    )


class TestPlanAssembly:
    def test_simple_assembly(self) -> None:
        # 1s call at 10-11s, need 2s left + 2s right for a 5s window
        target = _valid(10.0, 11.0)
        candidates = [
            _frag(7.0, 10.0, distance=1.5),  # 3s left
            _frag(11.0, 14.0, distance=1.5),  # 3s right
        ]
        plan = plan_assembly(target, candidates)
        assert plan.can_assemble is True
        assert plan.rejection_reason is None
        assert len(plan.left_fragments) >= 1
        assert len(plan.right_fragments) >= 1

    def test_left_right_needed_computation(self) -> None:
        # 1s call → need 2s each side for 5s window
        target = _valid(10.0, 11.0)
        candidates = [_frag(7.0, 10.0), _frag(11.0, 14.0)]
        plan = plan_assembly(target, candidates, window_size=5.0)
        assert plan.left_needed == 2.0
        assert plan.right_needed == 2.0

    def test_long_call_zero_background_needed(self) -> None:
        # 5s call fills the entire window
        target = _valid(10.0, 15.0)
        plan = plan_assembly(target, [], window_size=5.0)
        assert plan.can_assemble is True
        assert plan.left_needed == 0.0
        assert plan.right_needed == 0.0

    def test_insufficient_background_rejected(self) -> None:
        # 1s call, need 4s total bg, only 1s available
        target = _valid(10.0, 11.0)
        candidates = [_frag(9.0, 9.5, distance=1.0)]  # only 0.5s left
        plan = plan_assembly(target, candidates, min_fill_fraction=0.9)
        assert plan.can_assemble is False
        assert plan.rejection_reason == REASON_INSUFFICIENT_BACKGROUND

    def test_no_candidates_with_background_needed(self) -> None:
        target = _valid(10.0, 11.0)
        plan = plan_assembly(target, [])
        assert plan.can_assemble is False
        assert plan.rejection_reason == REASON_INSUFFICIENT_BACKGROUND

    def test_fragments_trimmed_to_needed(self) -> None:
        # 1s call, 2s needed each side; 5s fragments available
        target = _valid(10.0, 11.0)
        candidates = [
            _frag(3.0, 10.0, distance=3.5),  # 7s left (only need 2s)
            _frag(11.0, 18.0, distance=3.5),  # 7s right (only need 2s)
        ]
        plan = plan_assembly(target, candidates, window_size=5.0)
        assert plan.can_assemble is True
        left_total = sum(f.duration_sec for f in plan.left_fragments)
        right_total = sum(f.duration_sec for f in plan.right_fragments)
        assert abs(left_total - 2.0) < 0.01
        assert abs(right_total - 2.0) < 0.01

    def test_splice_points_generated(self) -> None:
        target = _valid(10.0, 11.0)
        candidates = [
            _frag(8.0, 10.0, distance=1.0),
            _frag(11.0, 13.0, distance=1.0),
        ]
        plan = plan_assembly(target, candidates, window_size=5.0)
        assert plan.can_assemble is True
        # Should have splice points: after left bg, after target
        assert len(plan.splice_points_sec) >= 1

    def test_custom_window_size(self) -> None:
        target = _valid(10.0, 11.0)  # 1s call
        candidates = [
            _frag(6.0, 10.0, distance=2.0),  # 4s left
            _frag(11.0, 15.0, distance=2.0),  # 4s right
        ]
        plan = plan_assembly(target, candidates, window_size=3.0)
        # 3s window - 1s call = 2s background, 1s each side
        assert plan.can_assemble is True
        assert plan.left_needed == 1.0
        assert plan.right_needed == 1.0

    def test_low_fill_threshold_accepts_partial(self) -> None:
        target = _valid(10.0, 11.0)
        # 1s left, 0s right = 1s / 4s needed = 25% fill
        candidates = [_frag(9.0, 10.0, distance=0.5)]
        plan = plan_assembly(target, candidates, min_fill_fraction=0.2)
        assert plan.can_assemble is True

    def test_multiple_left_fragments(self) -> None:
        target = _valid(10.0, 11.0)
        candidates = [
            _frag(9.0, 10.0, distance=0.5),  # 1s left
            _frag(7.5, 8.5, distance=2.0),  # 1s left
            _frag(11.0, 14.0, distance=1.5),  # 3s right
        ]
        plan = plan_assembly(target, candidates, window_size=5.0)
        assert plan.can_assemble is True
        assert len(plan.left_fragments) == 2
