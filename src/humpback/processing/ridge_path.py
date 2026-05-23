"""STFT ridge path tracking.

Extracted from ``humpback.sequence_models.event_encoder`` so Sequence Models
subsystems can consume the same dominant-energy path tracker. The Event
Encoder uses it to compute descriptor-level summaries (ADR-063); the Piano
Roll Notes v3 extractor uses the per-frame trajectory to drive coherent F0
notes and per-voice MPE pitch bend (ADR-069 / docs/specs/2026-05-22-
piano-roll-mpe-ridge-aligned-design.md §5.1).

Pure functions only — no I/O. The helpers operate on numpy arrays so they
can be exercised with synthetic spectra.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "RidgePathResult",
    "compute_ridge_path",
]


@dataclass(frozen=True)
class RidgePathResult:
    """Outcome of a single ridge-tracking pass over an event spectrum.

    Fields are aligned by frame index over the contributing frames (i.e.,
    frames where at least one ridge candidate cleared the prominence floor).
    ``total_frames`` is the original frame count of the input spectrum,
    independent of how many frames produced ridge candidates — callers use
    the difference to compute coverage ratios.
    """

    log_frequencies: np.ndarray
    frame_times: np.ndarray
    strengths: np.ndarray
    energy_ratios: np.ndarray
    total_frames: int


def compute_ridge_path(
    spectra: np.ndarray,
    freqs: np.ndarray,
    *,
    sample_rate: int,
    hop_length: int,
    eps: float = 1e-12,
    min_frequency_hz: float = 100.0,
    max_frequency_hz: float = 6000.0,
    candidate_count: int = 5,
    smoothness_penalty: float = 8.0,
    peak_prominence_ratio: float = 0.0,
) -> RidgePathResult:
    """Track a smooth dominant-energy ridge across a magnitude spectrum.

    Parameters mirror the v3 Event Encoder defaults; downstream callers
    (e.g., the Piano Roll Notes v3 extractor) pass a wider
    ``max_frequency_hz`` to cover the full humpback band.

    Args:
        spectra: 2-D ``(n_frames, n_bins)`` magnitude spectrum.
        freqs: 1-D array of bin centre frequencies in Hz, length
            ``n_bins``.
        sample_rate: Sample rate of the source audio in Hz.
        hop_length: STFT hop in samples.
        eps: Small additive guard for log/division operations.
        min_frequency_hz / max_frequency_hz: Inclusive frequency band the
            tracker may pick from.
        candidate_count: Maximum number of per-frame candidate peaks to
            forward to the Viterbi step.
        smoothness_penalty: Quadratic penalty on log-frequency jumps
            between adjacent frames; larger values favour smoother paths.
        peak_prominence_ratio: Per-frame relative-to-max threshold for
            candidate inclusion. ``0.0`` admits every local maximum above
            ``eps``.

    Returns a ``RidgePathResult``. Degenerate inputs (empty spectrum,
    mismatched shapes, invalid frequency band) yield an empty result so
    callers can branch on ``log_frequencies.size``.
    """
    total_frames = int(spectra.shape[0]) if spectra.ndim == 2 else 0
    empty = _empty_ridge_path_result(total_frames)
    if sample_rate <= 0 or hop_length <= 0 or spectra.ndim != 2 or spectra.shape[0] < 2:
        return empty
    freqs = np.asarray(freqs, dtype=np.float64)
    spectra = np.asarray(spectra, dtype=np.float64)
    if freqs.ndim != 1 or freqs.shape[0] != spectra.shape[1]:
        return empty
    if min_frequency_hz <= 0 or max_frequency_hz <= min_frequency_hz:
        return empty

    band_max = min(float(max_frequency_hz), float(freqs[-1]))
    band_mask = (
        np.isfinite(freqs)
        & (freqs >= float(min_frequency_hz))
        & (freqs <= band_max)
        & (freqs > 0)
    )
    band_indices = np.flatnonzero(band_mask)
    if band_indices.size == 0:
        return empty

    frame_candidates: list[tuple[np.ndarray, np.ndarray]] = []
    frame_times: list[float] = []
    frame_energy_totals: list[float] = []
    for frame_idx, spectrum in enumerate(spectra[:, band_indices]):
        spectrum = np.asarray(spectrum, dtype=np.float64)
        spectrum = np.where(np.isfinite(spectrum), spectrum, 0.0)
        spectrum = np.maximum(spectrum, 0.0)
        candidate_bins = _ridge_candidate_bins(
            spectrum,
            candidate_count=max(1, int(candidate_count)),
            peak_prominence_ratio=max(0.0, float(peak_prominence_ratio)),
            eps=eps,
        )
        if candidate_bins.size == 0:
            continue
        strengths = spectrum[candidate_bins].astype(np.float64)
        if not np.any(np.isfinite(strengths)) or float(np.max(strengths)) <= eps:
            continue
        candidate_freqs = freqs[band_indices[candidate_bins]]
        frame_candidates.append((np.log2(candidate_freqs), strengths))
        frame_times.append(float(frame_idx * hop_length / sample_rate))
        frame_energy_totals.append(float(np.sum(spectrum)))

    if len(frame_candidates) < 2:
        return empty

    path, path_indices = _track_log_frequency_path(
        frame_candidates,
        smoothness_penalty=max(0.0, float(smoothness_penalty)),
        eps=eps,
    )
    selected_strengths = np.asarray(
        [
            frame_candidates[frame_idx][1][candidate_idx]
            for frame_idx, candidate_idx in enumerate(path_indices)
        ],
        dtype=np.float64,
    )
    energy_ratios = np.asarray(
        [
            0.0
            if total <= eps
            else float(selected_strengths[frame_idx] / max(total, eps))
            for frame_idx, total in enumerate(frame_energy_totals)
        ],
        dtype=np.float64,
    )
    return RidgePathResult(
        log_frequencies=path,
        frame_times=np.asarray(frame_times, dtype=np.float64),
        strengths=selected_strengths,
        energy_ratios=energy_ratios,
        total_frames=total_frames,
    )


def _empty_ridge_path_result(total_frames: int = 0) -> RidgePathResult:
    empty = np.asarray([], dtype=np.float64)
    return RidgePathResult(
        log_frequencies=empty,
        frame_times=empty,
        strengths=empty,
        energy_ratios=empty,
        total_frames=max(0, int(total_frames)),
    )


def _ridge_candidate_bins(
    spectrum: np.ndarray,
    *,
    candidate_count: int,
    peak_prominence_ratio: float,
    eps: float,
) -> np.ndarray:
    if spectrum.size == 0:
        return np.asarray([], dtype=np.int64)
    frame_peak = float(np.nanmax(spectrum))
    if not np.isfinite(frame_peak) or frame_peak <= eps:
        return np.asarray([], dtype=np.int64)

    if spectrum.size == 1:
        local_maxima = np.asarray([0], dtype=np.int64)
    else:
        middle = (
            np.flatnonzero(
                (spectrum[1:-1] >= spectrum[:-2]) & (spectrum[1:-1] >= spectrum[2:])
            )
            + 1
        )
        endpoints = []
        if spectrum[0] >= spectrum[1]:
            endpoints.append(0)
        if spectrum[-1] >= spectrum[-2]:
            endpoints.append(spectrum.size - 1)
        local_maxima = np.asarray([*endpoints, *middle.tolist()], dtype=np.int64)

    threshold = frame_peak * peak_prominence_ratio
    local_maxima = local_maxima[spectrum[local_maxima] >= threshold]
    if local_maxima.size == 0:
        local_maxima = np.flatnonzero(spectrum >= threshold)
    if local_maxima.size == 0:
        local_maxima = np.arange(spectrum.size, dtype=np.int64)

    order = np.argsort(spectrum[local_maxima])[::-1]
    return local_maxima[order[:candidate_count]]


def _track_log_frequency_path(
    frame_candidates: list[tuple[np.ndarray, np.ndarray]],
    *,
    smoothness_penalty: float,
    eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    previous_logs, previous_strengths = frame_candidates[0]
    previous_cost = _ridge_emission_cost(previous_strengths, eps=eps)
    backpointers: list[np.ndarray] = []
    candidate_logs = [previous_logs]

    for current_logs, current_strengths in frame_candidates[1:]:
        emission = _ridge_emission_cost(current_strengths, eps=eps)
        transition = smoothness_penalty * np.square(
            previous_logs[:, None] - current_logs[None, :]
        )
        total = previous_cost[:, None] + transition + emission[None, :]
        backpointer = np.argmin(total, axis=0)
        previous_cost = total[backpointer, np.arange(current_logs.shape[0])]
        previous_logs = current_logs
        backpointers.append(backpointer.astype(np.int64))
        candidate_logs.append(current_logs)

    index = int(np.argmin(previous_cost))
    path_indices = [index]
    for frame_idx in range(len(backpointers) - 1, -1, -1):
        index = int(backpointers[frame_idx][index])
        path_indices.append(index)
    path_indices.reverse()
    path = [
        float(candidate_logs[frame_idx][candidate_idx])
        for frame_idx, candidate_idx in enumerate(path_indices)
    ]
    return (
        np.asarray(path, dtype=np.float64),
        np.asarray(path_indices, dtype=np.int64),
    )


def _ridge_emission_cost(strengths: np.ndarray, *, eps: float) -> np.ndarray:
    strengths = np.asarray(strengths, dtype=np.float64)
    frame_peak = float(np.max(strengths))
    if frame_peak <= eps:
        return np.zeros_like(strengths, dtype=np.float64)
    normalized = np.clip(strengths / frame_peak, eps, None)
    return -np.log(normalized)
