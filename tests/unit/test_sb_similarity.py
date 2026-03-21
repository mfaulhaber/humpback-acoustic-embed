"""Unit tests for sample_builder Stage 5 — acoustic similarity scoring."""

import numpy as np

from humpback.sample_builder.similarity import (
    SimilarityConfig,
    _spectral_flatness_value,
    _spectral_tilt,
    _stationarity,
    score_similarity,
)

SR = 16000


def _noise(
    duration_sec: float = 1.0, amplitude: float = 0.01, seed: int = 42
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(SR * duration_sec)
    return rng.normal(0, amplitude, n).astype(np.float32)


def _tone(
    freq_hz: float = 1000.0, duration_sec: float = 1.0, amplitude: float = 0.5
) -> np.ndarray:
    t = np.linspace(0, duration_sec, int(SR * duration_sec), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


class TestScoreSimilarity:
    def test_identical_signals_high_score(self) -> None:
        audio = _noise(amplitude=0.05, seed=1)
        result = score_similarity(audio, audio, SR)
        assert result.score > 0.8
        assert result.band_energy > 0.8

    def test_similar_noise_high_score(self) -> None:
        a = _noise(amplitude=0.05, seed=1)
        b = _noise(amplitude=0.05, seed=2)
        result = score_similarity(a, b, SR)
        # Same distribution, different realizations — should be fairly similar
        assert result.score > 0.5

    def test_noise_vs_tone_lower_score(self) -> None:
        noise = _noise(amplitude=0.05, seed=1)
        tone = _tone(freq_hz=1000.0, amplitude=0.05)
        result_same = score_similarity(noise, noise, SR)
        result_diff = score_similarity(noise, tone, SR)
        # Noise vs noise should score higher than noise vs tone
        assert result_same.score > result_diff.score

    def test_score_in_range(self) -> None:
        a = _noise(amplitude=0.05, seed=1)
        b = _tone(freq_hz=500.0, amplitude=0.1)
        result = score_similarity(a, b, SR)
        assert 0.0 <= result.score <= 1.0
        assert 0.0 <= result.band_energy <= 1.0
        assert 0.0 <= result.spectral_tilt <= 1.0
        assert 0.0 <= result.spectral_flatness <= 1.0
        assert 0.0 <= result.stationarity <= 1.0

    def test_custom_weights(self) -> None:
        a = _noise(amplitude=0.05, seed=1)
        b = _noise(amplitude=0.05, seed=2)
        # All weight on stationarity
        config = SimilarityConfig(
            weight_band_energy=0.0,
            weight_spectral_tilt=0.0,
            weight_spectral_flatness=0.0,
            weight_stationarity=1.0,
        )
        result = score_similarity(a, b, SR, config)
        assert 0.0 <= result.score <= 1.0
        # Score should equal stationarity (only weighted feature)
        assert abs(result.score - result.stationarity) < 0.01

    def test_zero_weights_returns_zero(self) -> None:
        a = _noise(amplitude=0.05, seed=1)
        config = SimilarityConfig(
            weight_band_energy=0.0,
            weight_spectral_tilt=0.0,
            weight_spectral_flatness=0.0,
            weight_stationarity=0.0,
        )
        result = score_similarity(a, a, SR, config)
        assert result.score == 0.0


class TestSpectralTilt:
    def test_white_noise_near_zero_tilt(self) -> None:
        audio = _noise(amplitude=0.1, seed=10)
        tilt = _spectral_tilt(audio, SR, n_fft=1024)
        # White noise has approximately flat spectrum
        assert abs(tilt) < 5.0

    def test_tone_has_nonzero_tilt(self) -> None:
        audio = _tone(freq_hz=500.0, amplitude=0.5)
        tilt = _spectral_tilt(audio, SR, n_fft=1024)
        # A single tone creates a peaked spectrum — tilt value should be nonzero
        assert tilt != 0.0


class TestSpectralFlatness:
    def test_noise_high_flatness(self) -> None:
        audio = _noise(amplitude=0.1, seed=20)
        sf = _spectral_flatness_value(audio, SR, n_fft=1024)
        # White noise has high spectral flatness (close to 1)
        assert sf > 0.3

    def test_tone_low_flatness(self) -> None:
        audio = _tone(freq_hz=1000.0, amplitude=0.5)
        sf = _spectral_flatness_value(audio, SR, n_fft=1024)
        # Pure tone has very low spectral flatness
        assert sf < 0.1

    def test_silence_returns_zero(self) -> None:
        audio = np.zeros(SR, dtype=np.float32)
        sf = _spectral_flatness_value(audio, SR, n_fft=1024)
        assert sf == 0.0


class TestStationarity:
    def test_stationary_noise_high_score(self) -> None:
        audio = _noise(amplitude=0.05, seed=30)
        score = _stationarity(audio, SR, frame_length=1024)
        assert score > 0.7

    def test_nonstationary_low_score(self) -> None:
        # First half silence, second half loud
        n = SR
        audio = np.zeros(n, dtype=np.float32)
        audio[n // 2 :] = 0.5
        score = _stationarity(audio, SR, frame_length=1024)
        assert score < 0.5

    def test_short_audio_returns_one(self) -> None:
        audio = np.zeros(500, dtype=np.float32)
        score = _stationarity(audio, SR, frame_length=1024)
        assert score == 1.0
