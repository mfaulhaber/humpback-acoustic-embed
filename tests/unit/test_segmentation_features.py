"""Tests for the Pass 2 segmentation feature extractor."""

from __future__ import annotations

import numpy as np
import pytest

from humpback.call_parsing.segmentation.features import (
    audio_sec_to_frame_index,
    extract_logmel,
    frame_index_to_audio_sec,
    normalize_per_region_zscore,
)
from humpback.schemas.call_parsing import SegmentationFeatureConfig


def _silence(duration_sec: float, sr: int = 16000) -> np.ndarray:
    return np.zeros(int(duration_sec * sr), dtype=np.float32)


def _sine(freq: float, duration_sec: float, sr: int = 16000) -> np.ndarray:
    t = np.arange(int(duration_sec * sr)) / sr
    return np.sin(2 * np.pi * freq * t).astype(np.float32)


def test_extract_logmel_shape() -> None:
    cfg = SegmentationFeatureConfig()
    audio = _sine(440.0, 2.0)
    logmel = extract_logmel(audio, cfg)
    assert logmel.ndim == 2
    assert logmel.shape[0] == cfg.n_mels
    expected_T = 1 + (audio.shape[0] // cfg.hop_length)
    assert abs(logmel.shape[1] - expected_T) <= 1


def test_extract_logmel_rejects_stereo() -> None:
    cfg = SegmentationFeatureConfig()
    stereo = np.zeros((2, 16000), dtype=np.float32)
    with pytest.raises(ValueError):
        extract_logmel(stereo, cfg)


def test_normalize_zscore_zero_mean_unit_std() -> None:
    rng = np.random.default_rng(42)
    x = rng.normal(size=(64, 100)).astype(np.float32) * 3.0 + 7.0
    out = normalize_per_region_zscore(x)
    assert abs(float(out.mean())) < 1e-5
    assert abs(float(out.std()) - 1.0) < 1e-4


def test_normalize_zscore_silence_safe() -> None:
    x = np.zeros((64, 100), dtype=np.float32)
    out = normalize_per_region_zscore(x)
    assert out.shape == x.shape
    assert np.all(np.isfinite(out))


def test_frame_audio_round_trip() -> None:
    cfg = SegmentationFeatureConfig()
    for frame_idx in (0, 1, 17, 128, 1000):
        t = frame_index_to_audio_sec(frame_idx, cfg)
        assert audio_sec_to_frame_index(t, cfg) == frame_idx


def test_audio_sec_to_frame_index_clamps_negative() -> None:
    cfg = SegmentationFeatureConfig()
    assert audio_sec_to_frame_index(-1.0, cfg) == 0


def test_extract_logmel_in_band_peak_sits_in_lower_mel_bins() -> None:
    """A 1 kHz tone (well inside fmin=20 / fmax=4000) should peak in the
    lower half of the mel axis. This directly exercises the fmin/fmax
    configuration without relying on absolute dB magnitudes (which are
    flattened by ``power_to_db(..., ref=np.max)``).
    """
    cfg = SegmentationFeatureConfig()
    logmel = extract_logmel(_sine(1000.0, 2.0), cfg)
    # Collapse time axis to frequency energy profile.
    per_bin = logmel.mean(axis=1)
    peak_bin = int(np.argmax(per_bin))
    assert peak_bin < cfg.n_mels // 2, (
        f"1 kHz peak fell in bin {peak_bin} — expected below {cfg.n_mels // 2}"
    )


def test_extract_logmel_pure_function() -> None:
    cfg = SegmentationFeatureConfig()
    audio = _sine(500.0, 1.5)
    a = extract_logmel(audio, cfg)
    b = extract_logmel(audio, cfg)
    assert np.allclose(a, b)


def test_normalize_handles_empty() -> None:
    out = normalize_per_region_zscore(np.zeros((0, 0), dtype=np.float32))
    assert out.size == 0
