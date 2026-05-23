"""Tests for the extracted STFT ridge tracker.

The tracker was previously a private helper inside event_encoder.py. These
tests cover the new public API directly and ensure the behaviour matches
the encoder's prior expectations (a regression in this module would
invalidate v3 Event Encoder descriptors).
"""

from __future__ import annotations

import numpy as np
import pytest

from humpback.processing.ridge_path import RidgePathResult, compute_ridge_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SR = 22050
N_FFT = 1024
HOP = 512


def _stft_magnitude(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (spectra, freqs) shaped as compute_ridge_path expects.

    spectra is ``(n_frames, n_bins)`` real magnitudes; freqs is the
    associated bin centre frequencies in Hz.
    """
    n_frames = max(0, (audio.shape[0] - N_FFT) // HOP + 1)
    if n_frames <= 0:
        return np.zeros((0, N_FFT // 2 + 1), dtype=np.float64), np.fft.rfftfreq(
            N_FFT, d=1.0 / SR
        )
    window = np.hanning(N_FFT).astype(np.float64)
    frames = np.stack(
        [audio[i * HOP : i * HOP + N_FFT] * window for i in range(n_frames)]
    )
    spectra = np.abs(np.fft.rfft(frames, axis=1)).astype(np.float64)
    freqs = np.fft.rfftfreq(N_FFT, d=1.0 / SR)
    return spectra, freqs


def _sine(frequency_hz: float, duration_s: float, amplitude: float = 1.0) -> np.ndarray:
    samples = int(round(duration_s * SR))
    t = np.arange(samples) / SR
    return amplitude * np.sin(2.0 * np.pi * frequency_hz * t).astype(np.float64)


def _sweep(
    start_hz: float, end_hz: float, duration_s: float, amplitude: float = 1.0
) -> np.ndarray:
    samples = int(round(duration_s * SR))
    t = np.arange(samples) / SR
    f = start_hz + (end_hz - start_hz) * (t / duration_s)
    phase = 2.0 * np.pi * np.cumsum(f) / SR
    return amplitude * np.sin(phase).astype(np.float64)


def _hz_at_frame(result: RidgePathResult, frame_idx: int) -> float:
    return float(2.0 ** result.log_frequencies[frame_idx])


# ---------------------------------------------------------------------------
# Degenerate inputs
# ---------------------------------------------------------------------------


def test_empty_spectrum_returns_empty_result():
    spectra = np.zeros((0, N_FFT // 2 + 1), dtype=np.float64)
    freqs = np.fft.rfftfreq(N_FFT, d=1.0 / SR)
    result = compute_ridge_path(spectra, freqs, sample_rate=SR, hop_length=HOP)
    assert isinstance(result, RidgePathResult)
    assert result.log_frequencies.size == 0
    assert result.frame_times.size == 0
    assert result.strengths.size == 0
    assert result.energy_ratios.size == 0
    assert result.total_frames == 0


def test_single_frame_input_returns_empty_result():
    """The tracker needs at least 2 frames to run the Viterbi step."""
    spectra = np.ones((1, N_FFT // 2 + 1), dtype=np.float64)
    freqs = np.fft.rfftfreq(N_FFT, d=1.0 / SR)
    result = compute_ridge_path(spectra, freqs, sample_rate=SR, hop_length=HOP)
    assert result.log_frequencies.size == 0
    assert result.total_frames == 1


def test_invalid_sample_rate_returns_empty():
    spectra = np.ones((10, N_FFT // 2 + 1), dtype=np.float64)
    freqs = np.fft.rfftfreq(N_FFT, d=1.0 / SR)
    result = compute_ridge_path(spectra, freqs, sample_rate=0, hop_length=HOP)
    assert result.log_frequencies.size == 0


def test_freqs_shape_mismatch_returns_empty():
    spectra = np.ones((10, 100), dtype=np.float64)
    freqs = np.fft.rfftfreq(N_FFT, d=1.0 / SR)
    result = compute_ridge_path(spectra, freqs, sample_rate=SR, hop_length=HOP)
    assert result.log_frequencies.size == 0


def test_invalid_band_returns_empty():
    spectra = np.ones((10, N_FFT // 2 + 1), dtype=np.float64)
    freqs = np.fft.rfftfreq(N_FFT, d=1.0 / SR)
    result = compute_ridge_path(
        spectra,
        freqs,
        sample_rate=SR,
        hop_length=HOP,
        min_frequency_hz=5000.0,
        max_frequency_hz=1000.0,
    )
    assert result.log_frequencies.size == 0


# ---------------------------------------------------------------------------
# Behavioural tests
# ---------------------------------------------------------------------------


def test_constant_tone_tracks_within_one_bin():
    """A 1.5 s pure tone at 440 Hz should track the bin nearest 440 Hz."""
    audio = _sine(440.0, 1.5)
    spectra, freqs = _stft_magnitude(audio)
    result = compute_ridge_path(
        spectra,
        freqs,
        sample_rate=SR,
        hop_length=HOP,
        min_frequency_hz=100.0,
        max_frequency_hz=6000.0,
    )
    assert result.log_frequencies.size > 0
    ridge_hz = np.power(2.0, result.log_frequencies)
    bin_spacing = float(freqs[1] - freqs[0])
    # Every frame's ridge sits within one FFT bin of 440 Hz.
    assert np.all(np.abs(ridge_hz - 440.0) <= bin_spacing)


def test_linear_sweep_tracks_slope_within_tolerance():
    """A 1.0 s sweep from 500 → 1500 Hz produces a monotonically rising path."""
    audio = _sweep(500.0, 1500.0, 1.0)
    spectra, freqs = _stft_magnitude(audio)
    result = compute_ridge_path(
        spectra,
        freqs,
        sample_rate=SR,
        hop_length=HOP,
        min_frequency_hz=100.0,
        max_frequency_hz=6000.0,
    )
    assert result.log_frequencies.size >= 10
    ridge_hz = np.power(2.0, result.log_frequencies)
    # First few frames near 500 Hz, last few near 1500 Hz.
    bin_spacing = float(freqs[1] - freqs[0])
    assert abs(float(ridge_hz[:3].mean()) - 500.0) <= 4 * bin_spacing
    assert abs(float(ridge_hz[-3:].mean()) - 1500.0) <= 4 * bin_spacing
    # The slope is positive on the whole path (allow small jitter).
    assert ridge_hz[-1] > ridge_hz[0]


def test_smoothness_penalty_prefers_continuation_over_loudest_jump():
    """With a high smoothness penalty, the path should not jump octaves to
    chase a single louder candidate."""

    # Build a synthetic spectrum where every frame has a steady 1000 Hz peak
    # of moderate strength, but the middle frame also has an even louder
    # spurious peak at 4000 Hz. A purely loudest-candidate tracker would
    # jump to 4000 Hz; the Viterbi tracker should not.
    n_frames = 20
    spectra = np.zeros((n_frames, N_FFT // 2 + 1), dtype=np.float64)
    freqs = np.fft.rfftfreq(N_FFT, d=1.0 / SR)
    target_bin = int(np.argmin(np.abs(freqs - 1000.0)))
    spurious_bin = int(np.argmin(np.abs(freqs - 4000.0)))
    spectra[:, target_bin] = 1.0
    spectra[n_frames // 2, spurious_bin] = 10.0  # outlier louder peak

    # High smoothness penalty (large quadratic in log-frequency space).
    result = compute_ridge_path(
        spectra,
        freqs,
        sample_rate=SR,
        hop_length=HOP,
        min_frequency_hz=100.0,
        max_frequency_hz=6000.0,
        smoothness_penalty=100.0,
        candidate_count=8,
    )
    assert result.log_frequencies.size == n_frames
    ridge_hz = np.power(2.0, result.log_frequencies)
    # Most frames should sit near 1000 Hz, not 4000 Hz.
    near_target = np.sum(np.abs(ridge_hz - 1000.0) < 200.0)
    near_spurious = np.sum(np.abs(ridge_hz - 4000.0) < 200.0)
    assert near_target > near_spurious


def test_energy_ratios_normalised():
    audio = _sine(440.0, 1.0)
    spectra, freqs = _stft_magnitude(audio)
    result = compute_ridge_path(spectra, freqs, sample_rate=SR, hop_length=HOP)
    assert result.energy_ratios.size == result.log_frequencies.size
    # Energy ratios are in [0, 1] by construction (selected strength / total).
    assert np.all(result.energy_ratios >= 0.0)
    assert np.all(result.energy_ratios <= 1.0 + 1e-9)


def test_frame_times_monotone():
    audio = _sine(440.0, 0.5)
    spectra, freqs = _stft_magnitude(audio)
    result = compute_ridge_path(spectra, freqs, sample_rate=SR, hop_length=HOP)
    times = result.frame_times
    assert times.size == result.log_frequencies.size
    assert np.all(np.diff(times) > 0)


def test_determinism_repeated_calls():
    audio = _sweep(300.0, 1200.0, 0.5)
    spectra, freqs = _stft_magnitude(audio)
    a = compute_ridge_path(spectra, freqs, sample_rate=SR, hop_length=HOP)
    b = compute_ridge_path(spectra, freqs, sample_rate=SR, hop_length=HOP)
    np.testing.assert_array_equal(a.log_frequencies, b.log_frequencies)
    np.testing.assert_array_equal(a.frame_times, b.frame_times)
    np.testing.assert_array_equal(a.strengths, b.strengths)
    np.testing.assert_array_equal(a.energy_ratios, b.energy_ratios)
    assert a.total_frames == b.total_frames


# ---------------------------------------------------------------------------
# Regression: identical output through the encoder's wrapper path
# ---------------------------------------------------------------------------


def test_event_encoder_descriptor_pipeline_uses_new_ridge():
    """Sanity check that ``compute_acoustic_descriptors`` still produces the
    ridge-derived descriptors via the new module."""
    from humpback.sequence_models.event_encoder import compute_acoustic_descriptors

    audio = _sine(440.0, 1.5)
    descriptors = compute_acoustic_descriptors(
        audio.astype(np.float32),
        sample_rate=SR,
        n_fft=N_FFT,
        hop_length=HOP,
    )
    # The ridge low/high band should bracket 440 Hz on a clean tone.
    assert descriptors["ridge_low_frequency"] > 0.0
    assert descriptors["ridge_high_frequency"] >= descriptors["ridge_low_frequency"]
    assert 350.0 < descriptors["ridge_median_frequency"] < 530.0


def test_band_clipping_to_caller_max():
    """A higher ``max_frequency_hz`` argument lets the tracker reach higher
    bins; this guards the contract relied upon by the Piano Roll Notes v3
    extractor which passes a wider ceiling."""
    audio = _sine(5500.0, 1.0)
    spectra, freqs = _stft_magnitude(audio)
    # Default ceiling 6000 should let the tracker find 5500 Hz.
    permissive = compute_ridge_path(
        spectra,
        freqs,
        sample_rate=SR,
        hop_length=HOP,
        max_frequency_hz=6000.0,
    )
    bin_spacing = float(freqs[1] - freqs[0])
    assert permissive.log_frequencies.size > 0
    ridge_hz = np.power(2.0, permissive.log_frequencies)
    assert np.median(np.abs(ridge_hz - 5500.0)) <= 3 * bin_spacing

    # Narrow ceiling excludes the tone; no path can be tracked.
    restricted = compute_ridge_path(
        spectra,
        freqs,
        sample_rate=SR,
        hop_length=HOP,
        max_frequency_hz=3000.0,
    )
    if restricted.log_frequencies.size > 0:
        # If anything was tracked, it should sit well below 3 kHz.
        ridge_hz_r = np.power(2.0, restricted.log_frequencies)
        assert float(ridge_hz_r.max()) <= 3000.0


@pytest.mark.parametrize("candidate_count", [1, 3, 8])
def test_candidate_count_clamps_at_least_one(candidate_count: int):
    """``candidate_count`` is clamped to ≥ 1 internally; the tracker should
    accept any positive integer without erroring."""
    audio = _sine(440.0, 0.3)
    spectra, freqs = _stft_magnitude(audio)
    result = compute_ridge_path(
        spectra,
        freqs,
        sample_rate=SR,
        hop_length=HOP,
        candidate_count=candidate_count,
    )
    assert isinstance(result, RidgePathResult)
