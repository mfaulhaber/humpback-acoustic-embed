"""CQT computation and per-frame spectral peak picking for Piano Roll Notes.

Pure functions only — no I/O. The worker is responsible for loading audio
and writing parquet. These helpers operate on numpy arrays so they can be
unit-tested with synthetic signals.

The default parameter values target Basic Pitch-style geometry: 27.5 Hz
(A0) up through 88 semitones × 3 bins per semitone, with a ~11.6 ms hop at
22050 Hz. See ``docs/specs/2026-05-20-piano-roll-midi-notes-design.md``
§6 for the design rationale.
"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

__all__ = [
    "CQTParams",
    "PeakParams",
    "compute_event_cqt",
    "pick_peaks_per_frame",
    "bin_frequency_hz",
    "midi_pitch_for_bin",
]


@dataclass(frozen=True, slots=True)
class CQTParams:
    """Inputs to ``librosa.cqt`` plus the canonical sample rate."""

    target_sample_rate: int = 22050
    hop_length: int = 256
    fmin: float = 27.5
    n_bins: int = 264
    bins_per_octave: int = 36
    filter_scale: float = 1.0


@dataclass(frozen=True, slots=True)
class PeakParams:
    """Per-frame peak picking parameters."""

    k_noise: float = 3.0
    top_k: int = 8


def compute_event_cqt(
    audio: np.ndarray,
    sr: int,
    *,
    params: CQTParams = CQTParams(),
    eps: float = 1e-6,
) -> np.ndarray:
    """Return log-magnitude CQT of an event audio slice.

    Resamples to ``params.target_sample_rate`` if ``sr`` differs, downmixes
    multi-channel input, and returns a ``(n_bins, n_frames)`` array of
    ``log(|CQT| + eps)``. Zero-length input returns a zero-frame matrix.
    """
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim > 1:
        samples = samples.mean(axis=tuple(range(1, samples.ndim)))
    if sr != params.target_sample_rate:
        samples = librosa.resample(
            samples,
            orig_sr=sr,
            target_sr=params.target_sample_rate,
            res_type="polyphase",
        )
    if samples.size == 0:
        return np.zeros((params.n_bins, 0), dtype=np.float32)

    magnitude = np.abs(
        librosa.cqt(
            samples,
            sr=params.target_sample_rate,
            hop_length=params.hop_length,
            fmin=params.fmin,
            n_bins=params.n_bins,
            bins_per_octave=params.bins_per_octave,
            filter_scale=params.filter_scale,
        )
    ).astype(np.float32)
    return np.log(magnitude + eps).astype(np.float32)


def pick_peaks_per_frame(
    log_magnitude: np.ndarray,
    *,
    params: PeakParams = PeakParams(),
) -> list[list[tuple[int, float]]]:
    """Find local-maxima peaks per frame above a per-frame noise floor.

    Returns ``frames`` where each entry is a list of ``(bin, log_magnitude)``
    tuples sorted from strongest to weakest. Frames with no qualifying
    peaks return an empty list. ``log_magnitude`` is expected to have shape
    ``(n_bins, n_frames)``.
    """
    if log_magnitude.size == 0:
        return []
    n_bins, n_frames = log_magnitude.shape
    if n_frames == 0:
        return []

    # Per-frame noise floor: median + k_noise * MAD, estimated over the
    # bottom half of each frame's bins. Restricting to the lower half
    # keeps strong signal peaks from inflating MAD and pushing the floor
    # above the harmonics — which happens routinely with clean,
    # sustained tones where the signal IS the variance.
    sorted_log_mag = np.sort(log_magnitude, axis=0)
    bottom_half = sorted_log_mag[: n_bins // 2, :]
    medians = np.median(bottom_half, axis=0)
    deviations = np.abs(bottom_half - medians[np.newaxis, :])
    mads = np.median(deviations, axis=0)
    noise_floor = medians + params.k_noise * mads

    # 3-bin neighborhood local maxima (strictly greater than both neighbors,
    # with edges treated as -inf to allow endpoint peaks).
    padded = np.empty((n_bins + 2, n_frames), dtype=log_magnitude.dtype)
    padded[0, :] = -np.inf
    padded[-1, :] = -np.inf
    padded[1:-1, :] = log_magnitude
    is_peak = (padded[1:-1, :] > padded[:-2, :]) & (padded[1:-1, :] > padded[2:, :])

    out: list[list[tuple[int, float]]] = []
    for t in range(n_frames):
        peak_bins = np.flatnonzero(is_peak[:, t])
        if peak_bins.size == 0:
            out.append([])
            continue
        magnitudes = log_magnitude[peak_bins, t]
        keep = magnitudes >= noise_floor[t]
        peak_bins = peak_bins[keep]
        magnitudes = magnitudes[keep]
        if peak_bins.size == 0:
            out.append([])
            continue
        # Sort strongest-first, keep top-K.
        order = np.argsort(-magnitudes, kind="stable")
        if order.size > params.top_k:
            order = order[: params.top_k]
        out.append([(int(peak_bins[idx]), float(magnitudes[idx])) for idx in order])
    return out


def bin_frequency_hz(bin_index: float, params: CQTParams) -> float:
    """Return the CQT centre frequency in Hz for a (fractional) bin."""
    return float(params.fmin * (2.0 ** (bin_index / params.bins_per_octave)))


def midi_pitch_for_bin(bin_index: float, params: CQTParams) -> int:
    """Round a CQT bin to the nearest MIDI pitch given the params.

    With ``fmin = 27.5`` Hz (A0 = MIDI 21) and 3 bins/semitone, this is the
    direct mapping ``21 + round(bin / 3)``. The implementation derives the
    pitch from frequency so non-default ``fmin`` / ``bins_per_octave`` still
    work correctly.
    """
    freq = bin_frequency_hz(bin_index, params)
    midi = 69.0 + 12.0 * np.log2(freq / 440.0)
    return int(round(midi))
