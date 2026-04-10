"""Tests for PCEN-based timeline tile rendering helper."""

from __future__ import annotations

import numpy as np
import pytest

from humpback.processing.pcen_rendering import PcenParams, render_tile_pcen


def _freq_index(freqs: np.ndarray, target_hz: float) -> int:
    return int(np.argmin(np.abs(freqs - target_hz)))


def test_chirp_in_noise_rises_above_floor():
    """A narrowband tone embedded in broadband noise should show up as a
    PCEN peak clearly above the adjacent noise-floor bins.
    """
    sr = 8000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    rng = np.random.default_rng(42)

    tone_hz = 1200.0
    tone = 0.05 * np.sin(2.0 * np.pi * tone_hz * t)
    noise = 0.01 * rng.standard_normal(t.shape[0]).astype(np.float32)
    audio = (tone + noise).astype(np.float32)

    freqs, pcen = render_tile_pcen(
        audio=audio,
        sample_rate=sr,
        n_fft=1024,
        hop_length=128,
    )

    tone_bin = _freq_index(freqs, tone_hz)
    # Sample noise bins well away from the tone to avoid spectral leakage.
    noise_bins = [
        _freq_index(freqs, 200.0),
        _freq_index(freqs, 2500.0),
        _freq_index(freqs, 3200.0),
    ]

    tone_mean = float(np.mean(pcen[tone_bin, :]))
    noise_mean = float(np.mean([np.mean(pcen[b, :]) for b in noise_bins]))

    assert tone_mean > noise_mean * 3.0, (
        f"Tone bin should dominate ({tone_mean=}, {noise_mean=})"
    )


def test_step_gain_pcen_flattens():
    """PCEN should largely equalize two contiguous regions of the same
    signal recorded at different gain levels.
    """
    sr = 8000
    # Use 5 s per segment so the default 0.5 s time constant has ~10
    # time constants to fully converge in each region before we inspect
    # the tail.
    segment_sec = 5.0
    rng = np.random.default_rng(7)

    def noise_at_rms(n: int, target_rms: float) -> np.ndarray:
        x = rng.standard_normal(n).astype(np.float32)
        current = float(np.sqrt(np.mean(x**2)))
        if current > 0:
            x *= target_rms / current
        return x

    quiet = noise_at_rms(int(sr * segment_sec), 0.01)
    loud = noise_at_rms(int(sr * segment_sec), 0.1)  # +20 dB
    audio = np.concatenate([quiet, loud]).astype(np.float32)

    _freqs, pcen = render_tile_pcen(
        audio=audio,
        sample_rate=sr,
        n_fft=1024,
        hop_length=128,
    )

    # Inspect the tail of each segment so PCEN has settled past the step.
    n_frames = pcen.shape[1]
    half = n_frames // 2
    tail_q = pcen[:, max(0, half - 10) : half]
    tail_l = pcen[:, n_frames - 10 : n_frames]
    mean_q = float(np.mean(tail_q))
    mean_l = float(np.mean(tail_l))
    assert mean_q > 0
    assert mean_l > 0
    # Raw magnitude ratio would be ~10x between these two segments; PCEN
    # should bring the ratio close to 1. Allow a generous window because
    # the step spans the filter's settling region.
    ratio = mean_l / mean_q
    assert 0.25 < ratio < 2.5, f"PCEN did not flatten gain step (ratio={ratio})"


def test_warmup_affects_leading_frames():
    """Warm-up should bring the PCEN filter into its settled state before
    the first rendered frame. With warm-up, the first frames reflect the
    converged filter behavior; without warm-up, they climb from zero.
    """
    sr = 8000
    rng = np.random.default_rng(1)
    # 15 s of stationary noise so both the cold and warm renders have
    # enough frames past the settling region to verify convergence.
    audio = 0.05 * rng.standard_normal(sr * 15).astype(np.float32)

    warmup_samples = 2 * sr
    n_fft = 1024
    hop_length = 128

    _, pcen_cold = render_tile_pcen(
        audio=audio[warmup_samples:],
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        warmup_samples=0,
    )
    _, pcen_warm = render_tile_pcen(
        audio=audio,
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        warmup_samples=warmup_samples,
    )

    # Same number of rendered frames after trimming.
    assert pcen_cold.shape[1] == pcen_warm.shape[1]

    # The first few frames should differ dramatically: cold starts from
    # a filter state of zero, warm starts already settled.
    cold_early = float(np.mean(pcen_cold[:, :4]))
    warm_early = float(np.mean(pcen_warm[:, :4]))
    assert warm_early > cold_early * 10, (
        f"Warm-up should yield substantially higher early-frame values "
        f"({warm_early=}, {cold_early=})"
    )

    # Past the filter's settling region, both should converge to the
    # same steady-state level. With time_constant=0.5 s, hop=128,
    # sr=8000, the filter needs ~5 time constants ≈ 155 frames to
    # reach 99% of the settled level — well within the 500-frame mark.
    cold_tail = float(np.mean(pcen_cold[:, 500:]))
    warm_tail = float(np.mean(pcen_warm[:, 500:]))
    tail_ratio = warm_tail / max(cold_tail, 1e-12)
    assert 0.9 < tail_ratio < 1.1, (
        f"Trailing frames should converge ({warm_tail=}, {cold_tail=}, "
        f"ratio={tail_ratio})"
    )


def test_tile_at_start_with_less_warmup_than_requested():
    """Supplying fewer pre-roll samples than requested should not error,
    and the returned frame count should match the short pre-roll.
    """
    sr = 8000
    rng = np.random.default_rng(2)
    audio = 0.03 * rng.standard_normal(sr * 3).astype(np.float32)

    # Ask for 5 seconds of warm-up but provide only 3 seconds of audio.
    warmup_samples = 5 * sr
    _, pcen = render_tile_pcen(
        audio=audio,
        sample_rate=sr,
        n_fft=1024,
        hop_length=128,
        warmup_samples=warmup_samples,
    )

    # Everything should have been trimmed; we end up with zero frames
    # rather than an error.
    assert pcen.ndim == 2
    assert pcen.shape[1] == 0


def test_empty_audio_returns_empty_frames():
    freqs, pcen = render_tile_pcen(
        audio=np.zeros(0, dtype=np.float32),
        sample_rate=8000,
        n_fft=1024,
        hop_length=128,
    )
    assert freqs.shape[0] == pcen.shape[0]
    assert pcen.shape[1] == 0


def test_nan_input_raises_value_error():
    audio = np.array([0.0, 0.1, np.nan, 0.2], dtype=np.float32)
    with pytest.raises(ValueError, match="non-finite"):
        render_tile_pcen(
            audio=audio,
            sample_rate=8000,
            n_fft=1024,
            hop_length=128,
        )


def test_custom_params_affect_output():
    """Different PcenParams should produce distinct outputs for the same
    audio — sanity check that parameters propagate."""
    sr = 8000
    rng = np.random.default_rng(3)
    audio = 0.05 * rng.standard_normal(sr * 2).astype(np.float32)

    _, pcen_default = render_tile_pcen(
        audio=audio,
        sample_rate=sr,
        n_fft=1024,
        hop_length=128,
        params=PcenParams(),
    )
    _, pcen_aggressive = render_tile_pcen(
        audio=audio,
        sample_rate=sr,
        n_fft=1024,
        hop_length=128,
        params=PcenParams(gain=0.98, bias=2.0, power=0.5, time_constant=0.4),
    )
    assert not np.allclose(pcen_default, pcen_aggressive)
