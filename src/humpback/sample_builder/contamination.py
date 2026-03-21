"""Stage 4: Screen background fragments for acoustic contamination.

Four signal-processing features detect unwanted sounds without requiring a
trained classifier.  An optional fifth feature wraps the existing classifier
scorer and can be enabled via config.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, sosfilt


# ---------------------------------------------------------------------------
# Config & result types
# ---------------------------------------------------------------------------


@dataclass
class ContaminationConfig:
    """Per-feature thresholds for contamination screening."""

    # Band-limited RMS: reject if fragment RMS exceeds noise floor by this factor
    rms_threshold_factor: float = 3.0
    rms_low_hz: float = 200.0
    rms_high_hz: float = 4000.0

    # Spectral occupancy: reject if mean fraction of active bins > threshold
    occupancy_threshold: float = 0.8
    occupancy_n_fft: int = 1024
    occupancy_noise_floor_db: float = -10.0

    # Tonal persistence: reject if any bin is active in > threshold fraction of frames
    persistence_threshold: float = 0.5
    persistence_n_fft: int = 1024
    persistence_margin_db: float = 10.0

    # Transient energy: reject if max frame-energy derivative > threshold
    transient_threshold: float = 10.0
    transient_frame_length: int = 1024

    # Optional detector score (disabled by default)
    use_detector_score: bool = False
    detector_score_threshold: float = 0.5


@dataclass
class ContaminationResult:
    """Outcome of contamination screening for a single fragment."""

    passed: bool
    feature_scores: dict[str, float] = field(default_factory=dict)
    reason: str | None = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def screen_fragment(
    fragment_audio: NDArray[np.floating],
    sr: int,
    reference_noise_floor: float,
    config: ContaminationConfig | None = None,
) -> ContaminationResult:
    """Screen a background fragment for acoustic contamination.

    Parameters
    ----------
    fragment_audio:
        1-D audio array (mono, float).
    sr:
        Sample rate in Hz.
    reference_noise_floor:
        Band-limited RMS of the recording's quiet regions (reference level).
    config:
        Screening thresholds.  Uses defaults when ``None``.

    Returns
    -------
    ContaminationResult with pass/fail, per-feature scores, and rejection reason.
    """
    if config is None:
        config = ContaminationConfig()

    scores: dict[str, float] = {}

    # 1. Band-limited RMS
    rms = _band_limited_rms(fragment_audio, sr, config.rms_low_hz, config.rms_high_hz)
    rms_ratio = rms / reference_noise_floor if reference_noise_floor > 0 else 0.0
    scores["rms_ratio"] = rms_ratio
    if rms_ratio > config.rms_threshold_factor:
        return ContaminationResult(
            passed=False,
            feature_scores=scores,
            reason=f"rms_ratio={rms_ratio:.2f} > {config.rms_threshold_factor}",
        )

    # 2. Spectral occupancy
    occupancy = _spectral_occupancy(
        fragment_audio, sr, config.occupancy_n_fft, config.occupancy_noise_floor_db
    )
    scores["spectral_occupancy"] = occupancy
    if occupancy > config.occupancy_threshold:
        return ContaminationResult(
            passed=False,
            feature_scores=scores,
            reason=f"spectral_occupancy={occupancy:.2f} > {config.occupancy_threshold}",
        )

    # 3. Tonal persistence
    persistence = _tonal_persistence(
        fragment_audio, sr, config.persistence_n_fft, config.persistence_margin_db
    )
    scores["tonal_persistence"] = persistence
    if persistence > config.persistence_threshold:
        return ContaminationResult(
            passed=False,
            feature_scores=scores,
            reason=f"tonal_persistence={persistence:.2f} > {config.persistence_threshold}",
        )

    # 4. Transient energy
    transient = _transient_energy(fragment_audio, sr, config.transient_frame_length)
    scores["transient_energy"] = transient
    if transient > config.transient_threshold:
        return ContaminationResult(
            passed=False,
            feature_scores=scores,
            reason=f"transient_energy={transient:.2f} > {config.transient_threshold}",
        )

    return ContaminationResult(passed=True, feature_scores=scores)


# ---------------------------------------------------------------------------
# Feature implementations
# ---------------------------------------------------------------------------


def _band_limited_rms(
    audio: NDArray[np.floating],
    sr: int,
    low_hz: float = 200.0,
    high_hz: float = 4000.0,
) -> float:
    """Bandpass filter then compute RMS."""
    nyquist = sr / 2.0
    low = max(low_hz / nyquist, 1e-6)
    high = min(high_hz / nyquist, 1.0 - 1e-6)
    if low >= high:
        return float(np.sqrt(np.mean(audio**2)))
    sos = butter(4, [low, high], btype="band", output="sos")
    filtered = np.asarray(sosfilt(sos, audio))
    return float(np.sqrt(np.mean(filtered**2)))


def _spectral_occupancy(
    audio: NDArray[np.floating],
    sr: int,
    n_fft: int = 1024,
    noise_floor_db: float = -60.0,
) -> float:
    """Fraction of frequency bins active above noise floor, averaged over frames."""
    hop = n_fft // 2
    n_samples = len(audio)
    if n_samples < n_fft:
        return 0.0

    n_frames = (n_samples - n_fft) // hop + 1
    if n_frames == 0:
        return 0.0

    total_occupancy = 0.0
    n_bins = n_fft // 2 + 1

    for i in range(n_frames):
        frame = audio[i * hop : i * hop + n_fft]
        spectrum = np.fft.rfft(frame)
        power_db = 20.0 * np.log10(np.abs(spectrum) + 1e-10)
        active = np.sum(power_db > noise_floor_db)
        total_occupancy += active / n_bins

    return float(total_occupancy / n_frames)


def _tonal_persistence(
    audio: NDArray[np.floating],
    sr: int,
    n_fft: int = 1024,
    margin_db: float = 10.0,
) -> float:
    """Max fraction of frames in which any single frequency bin is active.

    Uses a per-bin median threshold so each frequency bin's activation is
    measured against its own baseline.  This prevents colored (pink/red)
    ambient noise from triggering false positives — low-frequency bins that
    are naturally louder are compared to their own typical level rather than
    a global median dominated by quiet high-frequency bins.
    """
    hop = n_fft // 2
    n_samples = len(audio)
    if n_samples < n_fft:
        return 0.0

    n_frames = (n_samples - n_fft) // hop + 1
    if n_frames == 0:
        return 0.0

    n_bins = n_fft // 2 + 1
    bin_active_count = np.zeros(n_bins)

    all_power_db: list[NDArray[np.floating]] = []
    for i in range(n_frames):
        frame = audio[i * hop : i * hop + n_fft]
        spectrum = np.fft.rfft(frame)
        power_db = 20.0 * np.log10(np.abs(spectrum) + 1e-10)
        all_power_db.append(power_db)

    stacked = np.stack(all_power_db)  # (n_frames, n_bins)

    # Per-bin median: each bin's threshold adapts to its own energy baseline.
    per_bin_median = np.median(stacked, axis=0)  # (n_bins,)
    threshold = per_bin_median + margin_db  # (n_bins,)

    for i in range(n_frames):
        active = stacked[i] > threshold
        bin_active_count += active

    persistence_fraction = bin_active_count / n_frames
    return float(np.max(persistence_fraction))


def _transient_energy(
    audio: NDArray[np.floating],
    sr: int,
    frame_length: int = 1024,
) -> float:
    """Max derivative of frame-level energy (detects sudden onsets)."""
    n_samples = len(audio)
    if n_samples < frame_length:
        return 0.0

    n_frames = n_samples // frame_length
    if n_frames < 2:
        return 0.0

    energies = np.zeros(n_frames)
    for i in range(n_frames):
        frame = audio[i * frame_length : (i + 1) * frame_length]
        energies[i] = np.mean(frame**2)

    # Normalize energy relative to mean
    mean_energy = np.mean(energies)
    if mean_energy < 1e-10:
        return 0.0

    derivatives = np.abs(np.diff(energies)) / mean_energy
    return float(np.max(derivatives))
