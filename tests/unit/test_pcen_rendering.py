"""Tests for PCEN-based timeline tile rendering helper."""

from __future__ import annotations

import numpy as np
import pytest

from humpback.processing.pcen_rendering import PcenParams, render_tile_pcen


def _freq_index(freqs: np.ndarray, target_hz: float) -> int:
    return int(np.argmin(np.abs(freqs - target_hz)))


def test_transient_tone_burst_rises_above_floor():
    """A brief tone burst embedded in steady broadband noise should show
    up as a PCEN peak at the tone frequency during the burst, above the
    steady-state level of adjacent noise bins.

    PCEN normalizes each frequency bin against its own running envelope,
    so a *stationary* tone would be flattened toward the noise bins. A
    transient burst is the right probe for the AGC's transient response.
    """
    sr = 8000
    duration = 8.0
    n_samples = int(sr * duration)
    rng = np.random.default_rng(42)

    noise = 0.01 * rng.standard_normal(n_samples).astype(np.float32)
    audio = noise.copy()

    tone_hz = 1200.0
    burst_start_sec = 4.0
    burst_dur_sec = 0.5
    bs = int(burst_start_sec * sr)
    be = bs + int(burst_dur_sec * sr)
    t_burst = np.arange(be - bs) / sr
    audio[bs:be] += 0.2 * np.sin(2.0 * np.pi * tone_hz * t_burst).astype(np.float32)

    freqs, pcen = render_tile_pcen(
        audio=audio,
        sample_rate=sr,
        n_fft=1024,
        hop_length=128,
    )

    tone_bin = _freq_index(freqs, tone_hz)
    noise_bins = [
        _freq_index(freqs, 200.0),
        _freq_index(freqs, 2500.0),
        _freq_index(freqs, 3200.0),
    ]

    # Frame index of burst start/end (accounting for STFT hop).
    burst_start_frame = int(bs / 128)
    burst_end_frame = int(be / 128)
    tone_burst_peak = float(np.max(pcen[tone_bin, burst_start_frame:burst_end_frame]))
    # Noise floor: steady-state level well away from the burst.
    noise_floor = float(np.mean(pcen[noise_bins, : burst_start_frame - 20]))

    assert tone_burst_peak > noise_floor * 3.0, (
        f"Tone burst should dominate noise floor ({tone_burst_peak=}, {noise_floor=})"
    )


def test_step_gain_pcen_flattens():
    """PCEN should largely equalize two contiguous regions of the same
    signal recorded at different gain levels.
    """
    sr = 8000
    # Use 20 s per segment so the 2.0 s time constant has ~10 time
    # constants to fully converge in each region before we inspect the
    # tail.
    segment_sec = 20.0
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


def test_no_cold_start_transient_at_first_frame():
    """The first rendered frames must already reflect the signal's
    steady-state PCEN level — no near-zero dark strip at the left edge.

    Before the warm-zi fix, librosa.pcen's default ``lfilter_zi``
    initialized the per-bin low-pass filter as if the signal had been at
    unit amplitude forever. Real hydrophone magnitudes are orders of
    magnitude smaller, so the filter spent many frames decaying and the
    PCEN output was crushed to near zero at the start of every tile,
    producing a dark vertical strip at the junction between tiles.
    """
    sr = 8000
    rng = np.random.default_rng(1)
    audio = (0.01 * rng.standard_normal(sr * 5).astype(np.float32)).astype(np.float32)

    _, pcen = render_tile_pcen(
        audio=audio,
        sample_rate=sr,
        n_fft=1024,
        hop_length=128,
        warmup_samples=0,
    )

    # Compare the first handful of frames to the stable mid-signal region.
    head_mean = float(np.mean(pcen[:, :8]))
    mid_mean = float(np.mean(pcen[:, 150:250]))

    assert mid_mean > 0
    ratio = head_mean / mid_mean
    assert 0.5 < ratio < 2.0, (
        f"First frames should already be at the steady-state level, not "
        f"decaying from a cold-start (head={head_mean}, mid={mid_mean}, "
        f"ratio={ratio})"
    )


def test_warmup_matches_no_warmup_for_stationary_signal():
    """With warm zi, a render over [−warm, end] should converge to the
    same body as a render over [0, end] beyond a few PCEN time constants.
    """
    sr = 8000
    rng = np.random.default_rng(1)
    audio = 0.05 * rng.standard_normal(sr * 15).astype(np.float32)

    warmup_samples = 2 * sr
    n_fft = 1024
    hop_length = 128

    _, pcen_no_warm = render_tile_pcen(
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

    assert pcen_no_warm.shape[1] == pcen_warm.shape[1]
    # Past the filter's settling region both should converge to the
    # same steady state for a stationary input.
    a = float(np.mean(pcen_no_warm[:, 500:]))
    b = float(np.mean(pcen_warm[:, 500:]))
    ratio = b / max(a, 1e-12)
    assert 0.9 < ratio < 1.1, f"Tails should converge ({a=}, {b=}, {ratio=})"


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
