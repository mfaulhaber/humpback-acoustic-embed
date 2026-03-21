"""Unit tests for sample_builder Stage 9 — post-assembly validation."""

import numpy as np

from humpback.sample_builder.planner import AssemblyPlan
from humpback.sample_builder.types import (
    REASON_ACOUSTIC_MISMATCH,
    REASON_VALIDATION_FAILED,
    BackgroundFragment,
)
from humpback.sample_builder.validate import ValidationConfig, validate_sample

SR = 16000


def _frag(start: float, end: float) -> BackgroundFragment:
    return BackgroundFragment(
        start_sec=start,
        end_sec=end,
        duration_sec=end - start,
        distance_from_target=0.0,
    )


class TestValidateSample:
    def test_valid_sample_passes(self) -> None:
        # Use identical noise for left/right bg so they correlate well
        rng = np.random.default_rng(42)
        bg_noise = rng.normal(0, 0.001, 2 * SR).astype(np.float32)
        target = rng.normal(0, 0.001, 1 * SR).astype(np.float32)
        audio = np.concatenate([bg_noise, target, bg_noise])  # 5s total

        plan = AssemblyPlan(
            left_fragments=[_frag(7.0, 9.0)],
            target_start_sec=10.0,
            target_end_sec=11.0,
            right_fragments=[_frag(12.0, 14.0)],
            can_assemble=True,
            window_size=5.0,
        )
        splice_points = [2 * SR, 3 * SR]
        # noise_floor=0 skips contamination re-check
        passed, reason = validate_sample(audio, SR, plan, splice_points, 0.0)
        assert passed is True
        assert reason is None

    def test_edge_placement_rejection(self) -> None:
        audio = np.zeros(5 * SR, dtype=np.float32)
        # Very thin left background (0.1s < 0.3s margin)
        plan = AssemblyPlan(
            left_fragments=[_frag(9.9, 10.0)],
            target_start_sec=10.0,
            target_end_sec=14.0,
            right_fragments=[_frag(14.0, 15.0)],
            can_assemble=True,
            window_size=5.0,
        )
        passed, reason = validate_sample(
            audio,
            SR,
            plan,
            [],
            0.01,
            config=ValidationConfig(edge_margin_sec=0.3),
        )
        assert passed is False
        assert reason == REASON_VALIDATION_FAILED

    def test_splice_artifact_rejection(self) -> None:
        # Audio with huge energy discontinuity at splice point
        audio = np.zeros(5 * SR, dtype=np.float32)
        audio[: 2 * SR] = 0.001  # quiet left
        audio[2 * SR :] = 0.8  # loud right
        plan = AssemblyPlan(
            left_fragments=[_frag(7.0, 9.0)],
            target_start_sec=10.0,
            target_end_sec=11.0,
            right_fragments=[_frag(12.0, 14.0)],
            can_assemble=True,
            window_size=5.0,
        )
        splice_points = [2 * SR]
        passed, reason = validate_sample(
            audio,
            SR,
            plan,
            splice_points,
            0.01,
            config=ValidationConfig(splice_energy_ratio_max=5.0),
        )
        assert passed is False
        assert reason == REASON_VALIDATION_FAILED

    def test_no_splice_points_passes(self) -> None:
        audio = np.random.default_rng(42).normal(0, 0.01, 5 * SR).astype(np.float32)
        plan = AssemblyPlan(
            target_start_sec=10.0,
            target_end_sec=15.0,
            can_assemble=True,
            window_size=5.0,
        )
        passed, _ = validate_sample(audio, SR, plan, [], 0.01)
        assert passed is True

    def test_acoustic_mismatch_rejection(self) -> None:
        rng = np.random.default_rng(42)
        audio = np.zeros(5 * SR, dtype=np.float32)
        # Left bg: low-freq noise
        t = np.arange(2 * SR) / SR
        audio[: 2 * SR] = 0.1 * np.sin(2 * np.pi * 200 * t).astype(np.float32)
        # Target
        audio[2 * SR : 3 * SR] = rng.normal(0, 0.01, SR).astype(np.float32)
        # Right bg: high-freq noise (very different from left)
        audio[3 * SR :] = 0.1 * np.sin(2 * np.pi * 6000 * t).astype(np.float32)

        plan = AssemblyPlan(
            left_fragments=[_frag(7.0, 9.0)],
            target_start_sec=10.0,
            target_end_sec=11.0,
            right_fragments=[_frag(12.0, 14.0)],
            can_assemble=True,
            window_size=5.0,
        )
        # noise_floor=0 skips contamination re-check; relax splice check
        passed, reason = validate_sample(
            audio,
            SR,
            plan,
            [2 * SR, 3 * SR],
            0.0,
            config=ValidationConfig(
                mismatch_correlation_min=0.8,
                splice_energy_ratio_max=1000.0,
            ),
        )
        assert passed is False
        assert reason == REASON_ACOUSTIC_MISMATCH

    def test_no_background_fragments_passes(self) -> None:
        audio = np.random.default_rng(42).normal(0, 0.01, 5 * SR).astype(np.float32)
        plan = AssemblyPlan(
            target_start_sec=10.0,
            target_end_sec=15.0,
            can_assemble=True,
            window_size=5.0,
        )
        passed, reason = validate_sample(audio, SR, plan, [], 0.01)
        assert passed is True
        assert reason is None
