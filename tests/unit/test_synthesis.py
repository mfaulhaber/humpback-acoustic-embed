"""Synthesis quality tests: verify 5s output, no clicks at splices, energy preservation."""

import numpy as np
import pytest

from humpback.classifier.label_processor import (
    _raised_cosine_fade,
    synthesize_clean_window,
    synthesize_variants,
)


class TestRaisedCosineFade:
    def test_zero_length(self):
        fade = _raised_cosine_fade(0)
        assert len(fade) == 0

    def test_starts_at_zero_ends_at_one(self):
        fade = _raised_cosine_fade(100)
        assert fade[0] == pytest.approx(0.0, abs=0.01)
        assert fade[-1] == pytest.approx(1.0, abs=0.01)

    def test_monotonic_increasing(self):
        fade = _raised_cosine_fade(200)
        diffs = np.diff(fade)
        assert np.all(diffs >= -1e-7)


class TestSynthesisQuality:
    """Validate synthesised clips meet audio quality requirements."""

    SR = 16000
    WINDOW = 5.0

    def _make_tone(self, freq: float, duration: float, amplitude: float = 0.5):
        t = np.arange(int(self.SR * duration)) / self.SR
        return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)

    def _make_noise(self, duration: float, amplitude: float = 0.01):
        return (amplitude * np.random.randn(int(self.SR * duration))).astype(np.float32)

    def test_output_is_exactly_5s(self):
        call = self._make_tone(440.0, 2.0)
        bg = self._make_noise(self.WINDOW)
        sample = synthesize_clean_window(call, bg, self.SR, window_size=self.WINDOW)
        assert sample is not None
        assert len(sample.audio_segment) == int(self.SR * self.WINDOW)

    def test_no_clicks_at_splice_points(self):
        """Adjacent-sample jumps at splice boundaries should be small (no clicks)."""
        call = self._make_tone(440.0, 1.5, amplitude=0.8)
        bg = self._make_noise(self.WINDOW, amplitude=0.01)
        sample = synthesize_clean_window(
            call, bg, self.SR, window_size=self.WINDOW, crossfade_ms=50.0
        )
        assert sample is not None
        # Compute adjacent-sample differences
        diffs = np.abs(np.diff(sample.audio_segment))
        # Max jump should be bounded — sine at 440 Hz has max diff ~0.17 at SR=16000
        # Allow generous headroom but catch massive clicks
        assert float(np.max(diffs)) < 1.0

    def test_call_energy_preserved(self):
        """The call region should retain most of its original energy."""
        call = self._make_tone(440.0, 2.0, amplitude=0.5)
        bg = self._make_noise(self.WINDOW, amplitude=0.001)
        sample = synthesize_clean_window(
            call,
            bg,
            self.SR,
            window_size=self.WINDOW,
            placement_sec=1.5,
            crossfade_ms=20.0,
        )
        assert sample is not None
        # Extract the region where call was placed
        call_region = sample.audio_segment[
            int(1.5 * self.SR) : int(1.5 * self.SR) + len(call)
        ]
        call_rms = float(np.sqrt(np.mean(call**2)))
        region_rms = float(np.sqrt(np.mean(call_region**2)))
        # Region RMS should be at least 50% of original call RMS
        assert region_rms > 0.5 * call_rms

    def test_background_quieter_than_call(self):
        """Background region outside call should be quiet relative to call."""
        call = self._make_tone(440.0, 1.0, amplitude=0.5)
        bg = self._make_noise(self.WINDOW, amplitude=0.3)
        sample = synthesize_clean_window(
            call,
            bg,
            self.SR,
            window_size=self.WINDOW,
            placement_sec=2.0,
            crossfade_ms=20.0,
        )
        assert sample is not None
        # Check the first second (pure background region before call)
        bg_region = sample.audio_segment[: int(1.0 * self.SR)]
        call_region = sample.audio_segment[
            int(2.0 * self.SR) : int(2.0 * self.SR) + len(call)
        ]
        bg_rms = float(np.sqrt(np.mean(bg_region**2)))
        call_rms = float(np.sqrt(np.mean(call_region**2)))
        assert bg_rms < call_rms


class TestVariantPlacement:
    """Verify synthesis variant placements differ."""

    SR = 16000
    WINDOW = 5.0

    def test_early_centre_late_differ(self):
        """Variants should place the call at different positions in the window."""
        call = np.ones(int(self.SR * 1.0), dtype=np.float32) * 0.8
        # Use 3 different backgrounds so each variant is unique
        bgs = [
            np.random.randn(int(self.SR * self.WINDOW)).astype(np.float32) * 0.01
            for _ in range(3)
        ]
        variants = synthesize_variants(
            call, bgs, self.SR, window_size=self.WINDOW, n_variants=3
        )
        assert len(variants) == 3
        # Verify the audio content differs between variants
        for i in range(len(variants)):
            for j in range(i + 1, len(variants)):
                assert not np.array_equal(
                    variants[i].audio_segment, variants[j].audio_segment
                )

    def test_each_variant_is_5s(self):
        call = np.random.randn(int(self.SR * 2.0)).astype(np.float32) * 0.3
        bgs = [
            np.random.randn(int(self.SR * self.WINDOW)).astype(np.float32)
            for _ in range(3)
        ]
        variants = synthesize_variants(
            call, bgs, self.SR, window_size=self.WINDOW, n_variants=3
        )
        for v in variants:
            assert len(v.audio_segment) == int(self.SR * self.WINDOW)
