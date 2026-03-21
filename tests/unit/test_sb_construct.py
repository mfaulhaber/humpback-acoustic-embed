"""Unit tests for sample_builder Stage 7 — sample construction."""

import numpy as np

from humpback.sample_builder.construct import construct_sample
from humpback.sample_builder.planner import AssemblyPlan
from humpback.sample_builder.types import BackgroundFragment

SR = 16000


def _frag(start: float, end: float) -> BackgroundFragment:
    return BackgroundFragment(
        start_sec=start,
        end_sec=end,
        duration_sec=end - start,
        distance_from_target=0.0,
    )


def _make_audio(duration_sec: float) -> np.ndarray:
    """Deterministic audio: value at each sample = sample_index / sr."""
    n = int(duration_sec * SR)
    return np.arange(n, dtype=np.float32) / SR


class TestConstructSample:
    def test_output_exact_length(self) -> None:
        audio = _make_audio(30.0)
        plan = AssemblyPlan(
            left_fragments=[_frag(7.0, 9.0)],
            target_start_sec=10.0,
            target_end_sec=11.0,
            right_fragments=[_frag(12.0, 14.0)],
            can_assemble=True,
            window_size=5.0,
        )
        result, _ = construct_sample(plan, audio, SR)
        assert len(result) == int(5.0 * SR)

    def test_pads_when_short(self) -> None:
        audio = _make_audio(5.0)
        plan = AssemblyPlan(
            left_fragments=[_frag(0.0, 1.0)],
            target_start_sec=1.5,
            target_end_sec=2.5,
            right_fragments=[_frag(3.0, 3.5)],
            can_assemble=True,
            window_size=5.0,
        )
        result, _ = construct_sample(plan, audio, SR)
        assert len(result) == int(5.0 * SR)
        # Trailing samples should be zero (padding)
        tail_start = (
            int(1.0 * SR) + int(1.0 * SR) + int(0.5 * SR)
        )  # left + target + right
        assert np.all(result[tail_start:] == 0.0)

    def test_trims_when_long(self) -> None:
        audio = _make_audio(30.0)
        # Fragments that exceed window size
        plan = AssemblyPlan(
            left_fragments=[_frag(5.0, 8.0)],  # 3s
            target_start_sec=10.0,
            target_end_sec=12.0,  # 2s
            right_fragments=[_frag(13.0, 16.0)],  # 3s → total 8s > 5s
            can_assemble=True,
            window_size=5.0,
        )
        result, _ = construct_sample(plan, audio, SR)
        assert len(result) == int(5.0 * SR)

    def test_splice_points_returned(self) -> None:
        audio = _make_audio(30.0)
        plan = AssemblyPlan(
            left_fragments=[_frag(7.0, 9.0)],  # 2s
            target_start_sec=10.0,
            target_end_sec=11.0,  # 1s
            right_fragments=[_frag(12.0, 14.0)],  # 2s
            can_assemble=True,
            window_size=5.0,
        )
        _, splice_points = construct_sample(plan, audio, SR)
        # Should have splice point after left bg and after target
        assert len(splice_points) == 2
        # First splice at sample = 2s * SR
        assert splice_points[0] == int(2.0 * SR)
        # Second splice at sample = (2s + 1s) * SR
        assert splice_points[1] == int(3.0 * SR)

    def test_no_fragments_target_only(self) -> None:
        audio = _make_audio(30.0)
        plan = AssemblyPlan(
            target_start_sec=10.0,
            target_end_sec=15.0,
            can_assemble=True,
            window_size=5.0,
        )
        result, splice_points = construct_sample(plan, audio, SR)
        assert len(result) == int(5.0 * SR)
        assert len(splice_points) == 0

    def test_left_only_fragments(self) -> None:
        audio = _make_audio(30.0)
        plan = AssemblyPlan(
            left_fragments=[_frag(7.0, 9.0)],
            target_start_sec=10.0,
            target_end_sec=13.0,
            can_assemble=True,
            window_size=5.0,
        )
        result, splice_points = construct_sample(plan, audio, SR)
        assert len(result) == int(5.0 * SR)
        # One splice point after left bg
        assert len(splice_points) == 1

    def test_audio_content_preserved(self) -> None:
        # Audio with distinct values per second
        audio = np.zeros(int(30.0 * SR), dtype=np.float32)
        audio[int(7.0 * SR) : int(9.0 * SR)] = 1.0  # left bg
        audio[int(10.0 * SR) : int(11.0 * SR)] = 2.0  # target
        audio[int(12.0 * SR) : int(14.0 * SR)] = 3.0  # right bg

        plan = AssemblyPlan(
            left_fragments=[_frag(7.0, 9.0)],
            target_start_sec=10.0,
            target_end_sec=11.0,
            right_fragments=[_frag(12.0, 14.0)],
            can_assemble=True,
            window_size=5.0,
        )
        result, _ = construct_sample(plan, audio, SR)
        # First 2s should be 1.0 (left bg)
        assert np.all(result[: int(2.0 * SR)] == 1.0)
        # Next 1s should be 2.0 (target)
        assert np.all(result[int(2.0 * SR) : int(3.0 * SR)] == 2.0)
        # Next 2s should be 3.0 (right bg)
        assert np.all(result[int(3.0 * SR) : int(5.0 * SR)] == 3.0)
