"""Unit tests for sample_builder Stage 4 — contamination screening."""

import numpy as np

from humpback.sample_builder.contamination import (
    ContaminationConfig,
    _band_limited_rms,
    _spectral_occupancy,
    _tonal_persistence,
    _transient_energy,
    screen_fragment,
)

SR = 16000


def _noise(duration_sec: float = 1.0, amplitude: float = 0.01) -> np.ndarray:
    """Low-amplitude white noise (clean background)."""
    rng = np.random.default_rng(42)
    n = int(SR * duration_sec)
    return rng.normal(0, amplitude, n).astype(np.float32)


def _tone(
    freq_hz: float = 1000.0, duration_sec: float = 1.0, amplitude: float = 0.5
) -> np.ndarray:
    """Pure sine tone (contamination: tonal)."""
    t = np.linspace(0, duration_sec, int(SR * duration_sec), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def _transient(duration_sec: float = 1.0, onset_sec: float = 0.5) -> np.ndarray:
    """Silence followed by a sharp onset (contamination: transient)."""
    n = int(SR * duration_sec)
    audio = np.zeros(n, dtype=np.float32)
    onset_sample = int(onset_sec * SR)
    # Sharp burst of energy
    audio[onset_sample : onset_sample + 512] = 0.8
    return audio


class TestBandLimitedRms:
    def test_noise_low_rms(self) -> None:
        audio = _noise(amplitude=0.001)
        rms = _band_limited_rms(audio, SR)
        assert rms < 0.01

    def test_tone_in_band_higher_rms(self) -> None:
        audio = _tone(freq_hz=1000.0, amplitude=0.5)
        rms = _band_limited_rms(audio, SR)
        assert rms > 0.1

    def test_tone_out_of_band_low_rms(self) -> None:
        # Tone at 100 Hz is below the 200-4000 Hz bandpass
        audio = _tone(freq_hz=100.0, amplitude=0.5)
        rms = _band_limited_rms(audio, SR, low_hz=200.0, high_hz=4000.0)
        # Should be heavily attenuated
        assert rms < 0.1


class TestSpectralOccupancy:
    def test_noise_moderate_occupancy(self) -> None:
        audio = _noise(amplitude=0.1)
        occ = _spectral_occupancy(audio, SR, n_fft=1024, noise_floor_db=-60.0)
        # White noise has energy across bins but noise floor threshold limits it
        assert 0.0 <= occ <= 1.0

    def test_tone_low_occupancy(self) -> None:
        # Pure tone occupies very few bins
        audio = _tone(freq_hz=1000.0, amplitude=0.5)
        occ = _spectral_occupancy(audio, SR, n_fft=1024, noise_floor_db=-20.0)
        assert occ < 0.3

    def test_short_audio_returns_zero(self) -> None:
        audio = np.zeros(100, dtype=np.float32)
        assert _spectral_occupancy(audio, SR, n_fft=1024) == 0.0


class TestTonalPersistence:
    def test_intermittent_tone_high_persistence(self) -> None:
        # Tone present in first 40% of frames — per-bin median detects the
        # intermittent contamination since the median reflects silence-only frames.
        rng = np.random.default_rng(99)
        n = int(SR * 1.0)
        audio = rng.normal(0, 0.001, n).astype(np.float32)
        tone_end = int(0.4 * n)
        t = np.arange(tone_end) / SR
        audio[:tone_end] += (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
        persistence = _tonal_persistence(audio, SR, n_fft=1024)
        assert persistence > 0.3

    def test_constant_tone_low_persistence(self) -> None:
        # A constant tone has per-bin median equal to its own level — no frame
        # exceeds median + margin, so persistence is ~0.  This is the expected
        # trade-off: constant background tones are treated as baseline.
        audio = _tone(freq_hz=1000.0, duration_sec=1.0, amplitude=0.5)
        persistence = _tonal_persistence(audio, SR, n_fft=1024)
        assert persistence < 0.3

    def test_noise_low_persistence(self) -> None:
        audio = _noise(duration_sec=1.0, amplitude=0.01)
        persistence = _tonal_persistence(audio, SR, n_fft=1024)
        # Random noise — no bin should persist strongly
        assert persistence < 0.5

    def test_short_audio_returns_zero(self) -> None:
        audio = np.zeros(100, dtype=np.float32)
        assert _tonal_persistence(audio, SR, n_fft=1024) == 0.0


class TestTransientEnergy:
    def test_transient_detected(self) -> None:
        audio = _transient(duration_sec=1.0, onset_sec=0.5)
        te = _transient_energy(audio, SR, frame_length=1024)
        assert te > 5.0

    def test_stationary_noise_low_transient(self) -> None:
        audio = _noise(duration_sec=1.0, amplitude=0.01)
        te = _transient_energy(audio, SR, frame_length=1024)
        assert te < 5.0

    def test_short_audio_returns_zero(self) -> None:
        audio = np.zeros(500, dtype=np.float32)
        assert _transient_energy(audio, SR, frame_length=1024) == 0.0


class TestScreenFragment:
    def test_clean_noise_passes(self) -> None:
        audio = _noise(amplitude=0.0001)
        noise_floor = float(np.sqrt(np.mean(audio**2)))
        result = screen_fragment(audio, SR, noise_floor)
        assert result.passed is True
        assert result.reason is None

    def test_loud_tone_fails(self) -> None:
        audio = _tone(freq_hz=1000.0, amplitude=0.5)
        noise_floor = 0.005  # much lower than tone RMS
        result = screen_fragment(audio, SR, noise_floor)
        assert result.passed is False
        assert result.reason is not None
        assert "rms_ratio" in result.reason

    def test_transient_fails(self) -> None:
        audio = _transient()
        noise_floor = 0.01
        config = ContaminationConfig(
            rms_threshold_factor=100.0,  # don't fail on RMS
            occupancy_threshold=1.0,  # don't fail on occupancy
            persistence_threshold=1.0,  # don't fail on persistence
            transient_threshold=5.0,  # strict transient threshold
        )
        result = screen_fragment(audio, SR, noise_floor, config)
        assert result.passed is False
        assert result.reason is not None
        assert "transient_energy" in result.reason

    def test_tonal_fails_persistence(self) -> None:
        # Intermittent tone in first 40% — persistence ~0.4 exceeds 0.3 threshold
        rng = np.random.default_rng(99)
        n = int(SR * 1.0)
        audio = rng.normal(0, 0.001, n).astype(np.float32)
        tone_end = int(0.4 * n)
        t = np.arange(tone_end) / SR
        audio[:tone_end] += (0.01 * np.sin(2 * np.pi * 1500 * t)).astype(np.float32)
        noise_floor = 0.01
        config = ContaminationConfig(
            rms_threshold_factor=100.0,  # don't fail on RMS
            occupancy_threshold=1.0,  # don't fail on occupancy
            persistence_threshold=0.3,  # strict persistence check
            transient_threshold=100.0,  # don't fail on transient
        )
        result = screen_fragment(audio, SR, noise_floor, config)
        assert result.passed is False
        assert "tonal_persistence" in (result.reason or "")

    def test_default_config_used(self) -> None:
        audio = _noise(amplitude=0.005)
        noise_floor = 0.005
        result = screen_fragment(audio, SR, noise_floor)
        assert isinstance(result.feature_scores, dict)
        assert "rms_ratio" in result.feature_scores

    def test_feature_scores_populated(self) -> None:
        audio = _noise(amplitude=0.0001)
        noise_floor = 0.0001
        result = screen_fragment(audio, SR, noise_floor)
        # All 4 features should have scores when passing
        assert result.passed is True
        assert "rms_ratio" in result.feature_scores
        assert "spectral_occupancy" in result.feature_scores
        assert "tonal_persistence" in result.feature_scores
        assert "transient_energy" in result.feature_scores

    def test_zero_noise_floor_handled(self) -> None:
        audio = _noise(amplitude=0.001)
        result = screen_fragment(audio, SR, 0.0)
        # Should not crash; rms_ratio should be 0
        assert result.feature_scores["rms_ratio"] == 0.0


# ---------------------------------------------------------------------------
# Marine acoustic profile tests
# ---------------------------------------------------------------------------


def _pink_noise(
    duration_sec: float = 1.0, amplitude: float = 0.01, seed: int = 42
) -> np.ndarray:
    """Generate 1/f (pink) noise — typical ocean ambient spectral shape."""
    rng = np.random.default_rng(seed)
    n = int(SR * duration_sec)
    white = rng.normal(0, 1, n).astype(np.float64)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, 1.0 / SR)
    freqs[0] = 1.0  # avoid division by zero at DC
    spectrum /= np.sqrt(freqs)
    pink = np.fft.irfft(spectrum, n=n)
    pink_rms = float(np.sqrt(np.mean(pink**2)))
    if pink_rms > 0:
        pink *= amplitude / pink_rms
    return pink.astype(np.float32)


class TestMarineAcousticProfiles:
    """Tests verifying contamination screening works with realistic marine noise."""

    def test_pink_noise_passes_contamination_screen(self) -> None:
        """Pink noise at marine-typical amplitude should pass all 4 features."""
        audio = _pink_noise(duration_sec=2.0, amplitude=0.02, seed=1)
        noise_floor = float(np.sqrt(np.mean(audio**2)))
        result = screen_fragment(audio, SR, noise_floor)
        assert result.passed is True, (
            f"Pink noise rejected: {result.reason} (scores: {result.feature_scores})"
        )

    def test_pink_noise_with_embedded_call_fails(self) -> None:
        """Pink noise + loud whale-like tone should fail RMS check."""
        audio = _pink_noise(duration_sec=2.0, amplitude=0.02, seed=2)
        # Embed a 500 Hz tone at 10x background amplitude (simulates whale call)
        t = np.arange(len(audio)) / SR
        audio = audio + (0.2 * np.sin(2 * np.pi * 500 * t)).astype(np.float32)
        noise_floor = 0.02  # reference is the quiet background level
        result = screen_fragment(audio, SR, noise_floor)
        assert result.passed is False
        assert "rms_ratio" in (result.reason or "")

    def test_pink_noise_with_ship_tone_fails(self) -> None:
        """Pink noise + loud in-band ship tone should fail RMS check."""
        audio = _pink_noise(duration_sec=2.0, amplitude=0.02, seed=3)
        t = np.arange(len(audio)) / SR
        # Ship engine harmonic at 300 Hz (inside 200-4000 Hz RMS bandpass)
        audio = audio + (0.15 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
        noise_floor = 0.02
        result = screen_fragment(audio, SR, noise_floor)
        assert result.passed is False
        assert "rms_ratio" in (result.reason or "")

    def test_per_bin_persistence_low_for_pink_noise(self) -> None:
        """Pink noise should have low tonal persistence with per-bin median."""
        audio = _pink_noise(duration_sec=2.0, amplitude=0.02, seed=4)
        persistence = _tonal_persistence(audio, SR, n_fft=1024)
        assert persistence < 0.3, f"Pink noise persistence too high: {persistence:.3f}"

    def test_intermittent_tone_detected_by_persistence(self) -> None:
        """Tone in first 40% of pink noise should be detected by persistence."""
        audio = _pink_noise(duration_sec=2.0, amplitude=0.02, seed=5)
        # Add loud tone to first 40% only
        tone_end = int(0.4 * len(audio))
        t = np.arange(tone_end) / SR
        audio[:tone_end] += (0.5 * np.sin(2 * np.pi * 800 * t)).astype(np.float32)
        persistence = _tonal_persistence(audio, SR, n_fft=1024)
        assert persistence > 0.3

    def test_spectral_occupancy_marine_noise_floor(self) -> None:
        """Pink noise at quiet-to-moderate marine amplitudes passes at -10 dB / 0.8."""
        for amp in [0.005, 0.01, 0.02]:
            audio = _pink_noise(duration_sec=1.0, amplitude=amp, seed=6)
            occ = _spectral_occupancy(audio, SR, n_fft=1024, noise_floor_db=-10.0)
            assert occ < 0.8, (
                f"Pink noise (amp={amp}) occupancy {occ:.3f} >= 0.8 at -10 dB floor"
            )
