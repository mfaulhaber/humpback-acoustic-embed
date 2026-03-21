"""Unit tests for sample_builder Stage 8 — join smoothing."""

import numpy as np

from humpback.processing.dsp import raised_cosine_fade
from humpback.sample_builder.smooth import smooth_joins

SR = 16000


class TestRaisedCosineFade:
    def test_starts_at_zero(self) -> None:
        ramp = raised_cosine_fade(100)
        assert ramp[0] < 0.01

    def test_ends_at_one(self) -> None:
        ramp = raised_cosine_fade(100)
        assert ramp[-1] > 0.99

    def test_monotonically_increasing(self) -> None:
        ramp = raised_cosine_fade(100)
        assert np.all(np.diff(ramp) >= 0)

    def test_length_matches(self) -> None:
        for length in [1, 10, 100, 1000]:
            assert len(raised_cosine_fade(length)) == length

    def test_zero_length(self) -> None:
        ramp = raised_cosine_fade(0)
        assert len(ramp) == 0

    def test_negative_length(self) -> None:
        ramp = raised_cosine_fade(-5)
        assert len(ramp) == 0

    def test_dtype_float32(self) -> None:
        ramp = raised_cosine_fade(10)
        assert ramp.dtype == np.float32


class TestSmoothJoins:
    def test_no_splice_points_unchanged(self) -> None:
        audio = np.ones(SR, dtype=np.float32)
        result = smooth_joins(audio, [], SR)
        np.testing.assert_array_equal(result, audio)

    def test_does_not_modify_input(self) -> None:
        audio = np.ones(SR, dtype=np.float32)
        original = audio.copy()
        smooth_joins(audio, [SR // 2], SR)
        np.testing.assert_array_equal(audio, original)

    def test_output_length_preserved(self) -> None:
        audio = np.ones(SR, dtype=np.float32)
        result = smooth_joins(audio, [SR // 2], SR)
        assert len(result) == len(audio)

    def test_splice_region_modified(self) -> None:
        audio = np.ones(SR, dtype=np.float32)
        sp = SR // 2
        result = smooth_joins(audio, [sp], SR, crossfade_ms=50.0)
        half_fade = int(0.05 * SR / 2)
        # Region around splice should be modified (faded)
        region = result[sp - half_fade : sp + half_fade]
        # Not all ones anymore (fade was applied)
        assert not np.allclose(region, 1.0)

    def test_far_from_splice_unchanged(self) -> None:
        audio = np.ones(SR, dtype=np.float32) * 0.5
        sp = SR // 2
        result = smooth_joins(audio, [sp], SR, crossfade_ms=10.0)
        half_fade = int(0.01 * SR / 2)
        # Audio far from splice point should be unchanged
        assert np.all(result[: sp - half_fade - 1] == 0.5)
        assert np.all(result[sp + half_fade + 1 :] == 0.5)

    def test_multiple_splice_points(self) -> None:
        audio = np.ones(SR * 5, dtype=np.float32)
        points = [SR, 2 * SR, 3 * SR]
        result = smooth_joins(audio, points, SR, crossfade_ms=20.0)
        # Each splice region should be affected
        for sp in points:
            half_fade = int(0.02 * SR / 2)
            region = result[sp - half_fade : sp + half_fade]
            assert not np.allclose(region, 1.0)

    def test_splice_at_start(self) -> None:
        audio = np.ones(SR, dtype=np.float32)
        # Splice at sample 0 — should not crash
        result = smooth_joins(audio, [0], SR, crossfade_ms=50.0)
        assert len(result) == SR

    def test_splice_at_end(self) -> None:
        audio = np.ones(SR, dtype=np.float32)
        # Splice at last sample — should not crash
        result = smooth_joins(audio, [SR - 1], SR, crossfade_ms=50.0)
        assert len(result) == SR

    def test_zero_crossfade_unchanged(self) -> None:
        audio = np.ones(SR, dtype=np.float32) * 0.7
        result = smooth_joins(audio, [SR // 2], SR, crossfade_ms=0.0)
        np.testing.assert_array_equal(result, audio)
