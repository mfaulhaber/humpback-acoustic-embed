"""Stage 9: Validate an assembled sample before accepting it."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from humpback.sample_builder.contamination import (
    ContaminationConfig,
    screen_fragment,
)
from humpback.sample_builder.planner import AssemblyPlan
from humpback.sample_builder.types import (
    REASON_ACOUSTIC_MISMATCH,
    REASON_CONTAMINATION_DETECTED,
    REASON_VALIDATION_FAILED,
)


@dataclass
class ValidationConfig:
    """Thresholds for post-assembly validation."""

    # Splice artifact: max acceptable frame-energy ratio at splice points.
    # Set high because bg-to-call transitions naturally have large energy
    # ratios, and the crossfade in smooth.py already prevents audible clicks.
    splice_energy_ratio_max: float = 1000.0
    splice_frame_length: int = 512

    # Edge placement: minimum distance from sample edges for the vocalization
    edge_margin_sec: float = 0.3

    # Contamination re-check config (uses same ContaminationConfig)
    contamination_config: ContaminationConfig | None = None

    # Acoustic mismatch: min spectral correlation between left/right bg
    mismatch_correlation_min: float = 0.3


def validate_sample(
    audio: NDArray[np.floating],
    sr: int,
    plan: AssemblyPlan,
    splice_points: list[int],
    reference_noise_floor: float,
    config: ValidationConfig | None = None,
) -> tuple[bool, str | None]:
    """Validate an assembled sample.

    Parameters
    ----------
    audio:
        The assembled sample (1-D, mono).
    sr:
        Sample rate in Hz.
    plan:
        The assembly plan used to construct the sample.
    splice_points:
        Sample indices where fragments were joined.
    reference_noise_floor:
        Band-limited RMS of the recording's quiet regions.
    config:
        Validation thresholds.

    Returns
    -------
    ``(True, None)`` if valid, or ``(False, rejection_reason)`` if invalid.
    """
    if config is None:
        config = ValidationConfig()

    # 1. Edge placement: target not too close to sample edges
    target_duration = plan.target_end_sec - plan.target_start_sec
    left_bg_duration = sum(f.duration_sec for f in plan.left_fragments)
    edge_margin_samples = int(config.edge_margin_sec * sr)

    if left_bg_duration > 0 and int(left_bg_duration * sr) < edge_margin_samples:
        return False, REASON_VALIDATION_FAILED

    right_bg_duration = sum(f.duration_sec for f in plan.right_fragments)
    if right_bg_duration > 0 and int(right_bg_duration * sr) < edge_margin_samples:
        return False, REASON_VALIDATION_FAILED

    # 2. Splice artifacts: check energy discontinuity at splice points
    frame_len = config.splice_frame_length
    for sp in splice_points:
        if sp < frame_len or sp + frame_len > len(audio):
            continue
        pre_energy = float(np.mean(audio[sp - frame_len : sp] ** 2))
        post_energy = float(np.mean(audio[sp : sp + frame_len] ** 2))
        min_energy = min(pre_energy, post_energy)
        max_energy = max(pre_energy, post_energy)
        if min_energy > 1e-10:
            ratio = max_energy / min_energy
            if ratio > config.splice_energy_ratio_max:
                return False, REASON_VALIDATION_FAILED

    # 3. Contamination re-check on background regions
    if reference_noise_floor > 0 and plan.left_fragments:
        left_end = int(left_bg_duration * sr)
        if left_end > 0:
            left_bg = audio[:left_end]
            result = screen_fragment(
                left_bg, sr, reference_noise_floor, config.contamination_config
            )
            if not result.passed:
                return False, REASON_CONTAMINATION_DETECTED

    if reference_noise_floor > 0 and plan.right_fragments:
        right_start = int((left_bg_duration + target_duration) * sr)
        if right_start < len(audio):
            right_bg = audio[right_start:]
            if len(right_bg) > 0:
                result = screen_fragment(
                    right_bg, sr, reference_noise_floor, config.contamination_config
                )
                if not result.passed:
                    return False, REASON_CONTAMINATION_DETECTED

    # 4. Acoustic mismatch: spectral correlation between left and right bg
    if plan.left_fragments and plan.right_fragments:
        left_end = int(left_bg_duration * sr)
        right_start = int((left_bg_duration + target_duration) * sr)
        left_bg = audio[:left_end]
        right_bg = audio[right_start:]
        if len(left_bg) > 0 and len(right_bg) > 0:
            corr = _spectral_correlation(left_bg, right_bg, sr)
            if corr < config.mismatch_correlation_min:
                return False, REASON_ACOUSTIC_MISMATCH

    return True, None


def _spectral_correlation(
    a: NDArray[np.floating],
    b: NDArray[np.floating],
    sr: int,
    n_fft: int = 1024,
) -> float:
    """Pearson correlation of averaged power spectra.

    Uses frame-averaged spectra so that spectral *shape* (e.g. 1/f for ocean
    ambient) is compared rather than random per-frame fluctuations.
    """
    spec_a = _averaged_power_spectrum(a, n_fft)
    spec_b = _averaged_power_spectrum(b, n_fft)

    # Normalize
    a_norm = spec_a - np.mean(spec_a)
    b_norm = spec_b - np.mean(spec_b)

    denom = np.sqrt(np.sum(a_norm**2) * np.sum(b_norm**2))
    if denom < 1e-10:
        return 1.0  # both silent — consider matched
    return float(np.sum(a_norm * b_norm) / denom)


def _averaged_power_spectrum(
    audio: NDArray[np.floating],
    n_fft: int = 1024,
) -> NDArray[np.floating]:
    """Compute frame-averaged power spectrum (Welch-style)."""
    hop = n_fft // 2
    n_samples = len(audio)
    if n_samples < n_fft:
        return np.asarray(np.abs(np.fft.rfft(audio, n=n_fft)) ** 2, dtype=np.float64)

    n_frames = (n_samples - n_fft) // hop + 1
    n_bins = n_fft // 2 + 1
    total = np.zeros(n_bins, dtype=np.float64)
    for i in range(n_frames):
        frame = audio[i * hop : i * hop + n_fft]
        total += np.abs(np.fft.rfft(frame)) ** 2
    return total / n_frames
