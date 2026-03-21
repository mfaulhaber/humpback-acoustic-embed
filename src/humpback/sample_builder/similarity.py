"""Stage 5: Rank background fragments by acoustic similarity to reference.

The reference is the local background audio immediately around the target
annotation.  Fragments that sound similar to the reference are preferred for
sample assembly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# Config & result types
# ---------------------------------------------------------------------------


@dataclass
class SimilarityConfig:
    """Feature weights and parameters for similarity scoring."""

    # Mel-band energy similarity
    n_mels: int = 64
    n_fft: int = 1024
    weight_band_energy: float = 0.4

    # Spectral tilt
    weight_spectral_tilt: float = 0.2

    # Spectral flatness
    weight_spectral_flatness: float = 0.2

    # Stationarity
    weight_stationarity: float = 0.2
    stationarity_frame_length: int = 1024


@dataclass
class SimilarityScore:
    """Composite acoustic similarity score for a fragment."""

    score: float  # weighted composite in [0, 1]
    band_energy: float
    spectral_tilt: float
    spectral_flatness: float
    stationarity: float


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def score_similarity(
    fragment_audio: NDArray[np.floating],
    reference_audio: NDArray[np.floating],
    sr: int,
    config: SimilarityConfig | None = None,
) -> SimilarityScore:
    """Score how acoustically similar a fragment is to a reference.

    Parameters
    ----------
    fragment_audio:
        Candidate background fragment (1-D, mono, float).
    reference_audio:
        Reference background near the target annotation (1-D, mono, float).
    sr:
        Sample rate in Hz.
    config:
        Feature weights and parameters.  Uses defaults when ``None``.

    Returns
    -------
    SimilarityScore with per-feature and weighted composite scores in [0, 1].
    """
    if config is None:
        config = SimilarityConfig()

    be = _band_energy_similarity(
        fragment_audio, reference_audio, sr, config.n_mels, config.n_fft
    )
    st = _spectral_tilt_similarity(fragment_audio, reference_audio, sr, config.n_fft)
    sf = _spectral_flatness_similarity(
        fragment_audio, reference_audio, sr, config.n_fft
    )
    sn = _stationarity(fragment_audio, sr, config.stationarity_frame_length)

    total_weight = (
        config.weight_band_energy
        + config.weight_spectral_tilt
        + config.weight_spectral_flatness
        + config.weight_stationarity
    )
    if total_weight <= 0:
        composite = 0.0
    else:
        composite = (
            config.weight_band_energy * be
            + config.weight_spectral_tilt * st
            + config.weight_spectral_flatness * sf
            + config.weight_stationarity * sn
        ) / total_weight

    return SimilarityScore(
        score=float(np.clip(composite, 0.0, 1.0)),
        band_energy=be,
        spectral_tilt=st,
        spectral_flatness=sf,
        stationarity=sn,
    )


# ---------------------------------------------------------------------------
# Feature implementations
# ---------------------------------------------------------------------------


def _mel_energy_vector(
    audio: NDArray[np.floating], sr: int, n_mels: int, n_fft: int
) -> NDArray[np.floating]:
    """Compute mean mel-band energy vector for the signal."""
    from humpback.processing.features import extract_logmel_batch

    specs = extract_logmel_batch(
        [audio],
        sample_rate=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        normalization="none",
    )
    if not specs:
        return np.zeros(n_mels, dtype=np.float64)
    # specs[0] shape: (n_mels, time_frames) — mean across time
    return np.mean(specs[0], axis=1).astype(np.float64)


def _band_energy_similarity(
    a: NDArray[np.floating],
    b: NDArray[np.floating],
    sr: int,
    n_mels: int,
    n_fft: int,
) -> float:
    """Cosine similarity of mel-band energy vectors."""
    va = _mel_energy_vector(a, sr, n_mels, n_fft)
    vb = _mel_energy_vector(b, sr, n_mels, n_fft)
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    cosine = float(np.dot(va, vb) / (norm_a * norm_b))
    # Map from [-1, 1] to [0, 1]
    return float(np.clip((cosine + 1.0) / 2.0, 0.0, 1.0))


def _spectral_tilt(audio: NDArray[np.floating], sr: int, n_fft: int) -> float:
    """Linear regression slope of log-energy vs log-frequency."""
    spectrum = np.abs(np.fft.rfft(audio, n=n_fft))
    power = spectrum**2
    # Skip DC bin
    power = power[1:]
    if len(power) == 0:
        return 0.0

    freqs = np.arange(1, len(power) + 1) * (sr / n_fft)
    log_freq = np.log10(freqs + 1e-10)
    log_power = np.log10(power + 1e-10)

    # Linear regression: slope
    mean_x = np.mean(log_freq)
    mean_y = np.mean(log_power)
    dx = log_freq - mean_x
    dy = log_power - mean_y
    denom = np.sum(dx**2)
    if denom < 1e-10:
        return 0.0
    return float(np.sum(dx * dy) / denom)


def _spectral_tilt_similarity(
    a: NDArray[np.floating],
    b: NDArray[np.floating],
    sr: int,
    n_fft: int,
) -> float:
    """Similarity based on closeness of spectral tilt (slope).

    Returns 1.0 when tilts are identical, decaying toward 0 as they diverge.
    """
    tilt_a = _spectral_tilt(a, sr, n_fft)
    tilt_b = _spectral_tilt(b, sr, n_fft)
    diff = abs(tilt_a - tilt_b)
    # Exponential decay: sigma controls how quickly similarity drops
    sigma = 2.0
    return float(np.exp(-(diff**2) / (2 * sigma**2)))


def _spectral_flatness_value(audio: NDArray[np.floating], sr: int, n_fft: int) -> float:
    """Geometric / arithmetic mean ratio of power spectrum (Wiener entropy)."""
    spectrum = np.abs(np.fft.rfft(audio, n=n_fft))
    power = spectrum[1:] ** 2  # skip DC
    if len(power) == 0 or np.all(power < 1e-20):
        return 0.0
    log_mean = np.mean(np.log(power + 1e-20))
    geo_mean = np.exp(log_mean)
    arith_mean = np.mean(power)
    if arith_mean < 1e-20:
        return 0.0
    return float(geo_mean / arith_mean)


def _spectral_flatness_similarity(
    a: NDArray[np.floating],
    b: NDArray[np.floating],
    sr: int,
    n_fft: int,
) -> float:
    """Similarity based on closeness of spectral flatness values.

    Returns 1.0 when flatness values are identical, decaying toward 0.
    """
    sf_a = _spectral_flatness_value(a, sr, n_fft)
    sf_b = _spectral_flatness_value(b, sr, n_fft)
    diff = abs(sf_a - sf_b)
    # Flatness is in [0, 1], so use a tighter sigma
    sigma = 0.3
    return float(np.exp(-(diff**2) / (2 * sigma**2)))


def _stationarity(
    audio: NDArray[np.floating],
    sr: int,
    frame_length: int = 1024,
) -> float:
    """Score stationarity as inverse of frame-RMS variance.

    Lower variance means more stationary — better background.
    Returns a value in [0, 1] where 1 is perfectly stationary.
    """
    n_samples = len(audio)
    if n_samples < frame_length:
        return 1.0  # too short to measure; assume stationary

    n_frames = n_samples // frame_length
    if n_frames < 2:
        return 1.0

    rms_values = np.zeros(n_frames)
    for i in range(n_frames):
        frame = audio[i * frame_length : (i + 1) * frame_length]
        rms_values[i] = np.sqrt(np.mean(frame**2))

    mean_rms = np.mean(rms_values)
    if mean_rms < 1e-10:
        return 1.0

    # Coefficient of variation (std / mean)
    cv = float(np.std(rms_values) / mean_rms)
    # Map CV to [0, 1]: cv=0 → 1.0, cv→∞ → 0.0
    return float(np.exp(-(cv**2)))
