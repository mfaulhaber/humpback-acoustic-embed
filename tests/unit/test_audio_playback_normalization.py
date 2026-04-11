"""Tests for the playback audio normalization helper."""

from __future__ import annotations

import numpy as np

from humpback.processing.audio_encoding import normalize_for_playback


def _sine(n: int, sr: int, freq: float, amplitude: float) -> np.ndarray:
    t = np.arange(n, dtype=np.float32) / sr
    return (amplitude * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


def _rms_dbfs(audio: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    if rms <= 0:
        return -np.inf
    return 20.0 * np.log10(rms)


def test_rms_target_reached_from_quiet_input():
    """A sine well below the target should be scaled up to the target RMS."""
    sr = 16000
    audio = _sine(sr, sr, 440.0, 0.01)  # ~-43 dBFS RMS
    out = normalize_for_playback(audio, target_rms_dbfs=-20.0, ceiling=0.95)
    assert abs(_rms_dbfs(out) - (-20.0)) < 0.5


def test_rms_target_reached_from_loud_input():
    """A sine well above the target should be scaled down to the target RMS."""
    sr = 16000
    audio = _sine(sr, sr, 440.0, 0.7)  # ~-6 dBFS RMS
    out = normalize_for_playback(audio, target_rms_dbfs=-20.0, ceiling=0.95)
    assert abs(_rms_dbfs(out) - (-20.0)) < 0.5


def test_soft_clip_holds_ceiling():
    """Peaky input that would exceed the ceiling after RMS scaling should
    stay under the ceiling via the ``tanh`` soft clip."""
    rng = np.random.default_rng(0)
    audio = 0.0001 * rng.standard_normal(10000).astype(np.float32)
    # Inject a large spike so RMS scaling pushes a single sample far
    # above unity before the soft clip catches it.
    audio[5000] = 50.0
    out = normalize_for_playback(audio, target_rms_dbfs=-20.0, ceiling=0.95)
    assert float(np.max(np.abs(out))) <= 0.95 + 1e-6


def test_silent_input_stays_silent():
    out = normalize_for_playback(np.zeros(1000, dtype=np.float32))
    assert np.all(out == 0.0)
    assert out.dtype == np.float32


def test_near_silent_input_does_not_blow_up():
    """RMS far below the silence floor should not be divided-by."""
    audio = np.full(1000, 1e-12, dtype=np.float32)
    out = normalize_for_playback(audio)
    assert np.all(np.isfinite(out))
    assert float(np.max(np.abs(out))) < 1e-6


def test_empty_input_returns_empty_array():
    out = normalize_for_playback(np.zeros(0, dtype=np.float32))
    assert out.shape == (0,)
    assert out.dtype == np.float32


def test_does_not_mutate_input():
    sr = 16000
    audio = _sine(sr, sr, 440.0, 0.5)
    original = audio.copy()
    normalize_for_playback(audio)
    np.testing.assert_array_equal(audio, original)


def test_output_dtype_is_float32():
    audio = np.ones(1000, dtype=np.float64) * 0.2
    out = normalize_for_playback(audio)
    assert out.dtype == np.float32
