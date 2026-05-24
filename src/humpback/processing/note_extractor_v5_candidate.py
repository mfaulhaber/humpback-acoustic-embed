"""Harmonic-Viterbi F0 + harmonics note extractor (v5 candidate).

Phase-2 candidate for Piano Roll Notes v5. Replaces v4's ridge-locked
HPS divisor selection with direct harmonic-sum F0 estimation over the
CQT plus log-frequency Viterbi smoothing. Temporal smoothness is part
of the cost function rather than a post-filter, so frame-to-frame F0
hopping (the v4 failure mode on token #47 of job 690580c5) is penalised
during decoding instead of being amplified by independent per-frame
divisor selection.

See ``docs/specs/2026-05-24-piano-roll-notes-v5-test-bed-design.md``
§4.3 for the algorithm specification and the design rationale.

The starting candidate algorithm in this module is replaceable during
Phase 2 iteration — Phase 3 promotes whatever the iteration converges
on to ``note_extractor_v5.py`` under the same public interface.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Optional, Sequence

import librosa
import numpy as np

from humpback.processing.note_extractor_v3 import (
    ContourFrame,
    HarmonicSearchParams,
    MidiRangeParams,
    NotesV3Result,
    NoteV3,
    SegmentationParams,
    STFTParams,
    _build_f0_note,
    _build_harmonic_notes,
    _cqt_bin_frequencies,
    _F0Note,
    _frame_noise_floor,
    _RefinedFrame,
    _split_runs,
)
from humpback.processing.piano_roll_cqt import CQTParams, compute_event_cqt

__all__ = [
    "HarmonicViterbiParams",
    "STFTParams",
    "SegmentationParams",
    "HarmonicSearchParams",
    "MidiRangeParams",
    "ExtractNotesV5Params",
    "ContourFrame",
    "NoteV3",
    "NotesV3Result",
    "extract_notes_v5_candidate",
]


_CENTS_PER_OCTAVE = 1200.0
# Mirror of v4's noise-floor MAD clamp. Pure-tone columns collapse MAD
# to zero and the above-floor threshold becomes trivially clearable;
# the clamp keeps the gate meaningful.
_MIN_MAD: float = 0.3


@dataclass(frozen=True, slots=True)
class HarmonicViterbiParams:
    """Per-frame harmonic-sum + temporal Viterbi parameters (spec §4.3).

    The voicing flag is hard: when ``H_t.max() − H_t.median() <=
    tau_voicing`` the frame is forced into the rest state. Combined with
    ``voicing_transition_cost`` this provides hysteresis on note
    boundaries without admitting low-confidence voiced candidates inside
    a note.

    ``transition_lambda`` scales squared log-frequency steps to Viterbi
    cost. Larger values smooth more aggressively. A jump of one octave
    costs exactly ``transition_lambda``; the cost grows with the square
    of the step so multi-octave jumps are heavily disfavoured relative
    to gentle glides.
    """

    n_harmonics: int = 4
    harmonic_weight: Literal["uniform", "inv_k", "inv_sqrt_k"] = "inv_sqrt_k"
    f0_min_hz: float = 30.0
    f0_max_hz: float = 600.0
    cents_tolerance: float = 50.0
    k_noise: float = 2.0
    # Voicing oracle in CQT log-magnitude units: a frame is voiced when
    # ``cqt_log[:, t].max() − noise_floor_t > tau_voicing``. Scale-
    # invariant (CQT log-magnitudes are relative); empirically real
    # tonal signals sit at peakedness 5–12 and broadband noise at 1.5–2.
    tau_voicing: float = 3.0
    transition_lambda: float = 2.0
    voicing_transition_cost: float = 1.0
    # Candidates with fewer than this many harmonics clearing the
    # noise-floor + local-peak gates score 0. Matches v4's
    # ``high_band_min_harmonics = 2`` noise-rejection role: broadband
    # noise can clear single-bin energy gates by luck, but rarely aligns
    # at integer-ratio harmonic positions in two bins simultaneously.
    min_harmonics_present: int = 2


@dataclass(frozen=True, slots=True)
class ExtractNotesV5Params:
    """All settings + per-event identity used by the v5 candidate.

    Mirrors :class:`ExtractNotesV4Params` field-for-field with
    ``harmonic_viterbi`` replacing ``hps``. STFT, CQT, segmentation,
    harmonic, and MIDI sub-params carry the v4 defaults unchanged
    (including the 30 Hz STFT band floor).
    """

    job_id: str
    event_id: str
    event_start_utc: float
    pad_seconds: float = 0.05
    cqt: CQTParams = field(default_factory=CQTParams)
    stft: STFTParams = field(default_factory=lambda: STFTParams(min_frequency_hz=30.0))
    harmonic_viterbi: HarmonicViterbiParams = field(
        default_factory=HarmonicViterbiParams
    )
    segmentation: SegmentationParams = field(default_factory=SegmentationParams)
    harmonic: HarmonicSearchParams = field(default_factory=HarmonicSearchParams)
    midi: MidiRangeParams = field(default_factory=MidiRangeParams)


def extract_notes_v5_candidate(
    audio: np.ndarray,
    sample_rate: int,
    *,
    params: ExtractNotesV5Params,
    ridge_sidecar_rows: Optional[Sequence[Mapping[str, Any]]] = None,
) -> NotesV3Result:
    """Extract harmonic-Viterbi F0 + harmonics notes for one event.

    Same input/output contract as :func:`extract_notes_v3` and
    :func:`extract_notes_v4`. The ``ridge_sidecar_rows`` argument is
    accepted for signature parity with v3/v4 and is unused — v5 derives
    F0 directly from the CQT and does not consume the ridge.

    Returns ``NotesV3Result`` so worker code and downstream parquet
    schemas stay unchanged across versions. ``subharmonic_octave`` is
    populated as ``0`` for every contour frame (v5 has no divisor
    concept; the column is reserved / unused in v5+).
    """
    del ridge_sidecar_rows  # accepted for signature parity; unused in v5
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim > 1:
        samples = samples.mean(axis=tuple(range(1, samples.ndim)))

    target_sr = params.cqt.target_sample_rate
    if sample_rate != target_sr and samples.size > 0:
        samples = librosa.resample(
            samples,
            orig_sr=sample_rate,
            target_sr=target_sr,
            res_type="polyphase",
        ).astype(np.float32)

    if samples.size == 0:
        return NotesV3Result(notes=[], contours=[])

    cqt_log = compute_event_cqt(samples, target_sr, params=params.cqt)
    if cqt_log.size == 0:
        return NotesV3Result(notes=[], contours=[])
    n_cqt_frames = int(cqt_log.shape[1])
    if n_cqt_frames < params.segmentation.min_note_frames:
        return NotesV3Result(notes=[], contours=[])
    cqt_seconds_per_frame = float(params.cqt.hop_length) / float(target_sr)
    cqt_bin_freqs = _cqt_bin_frequencies(params.cqt)
    cqt_bin_log_freqs = np.log2(cqt_bin_freqs)

    f0_segments = _decode_f0(
        cqt_log=cqt_log,
        cqt_bin_log_freqs=cqt_bin_log_freqs,
        cqt_seconds_per_frame=cqt_seconds_per_frame,
        params=params.harmonic_viterbi,
        segmentation=params.segmentation,
    )
    if not f0_segments:
        return NotesV3Result(notes=[], contours=[])

    notes: list[NoteV3] = []
    contours: list[ContourFrame] = []
    next_track_id = 0
    pad = float(params.pad_seconds)
    f0_built: list[_F0Note] = []
    v3_params = _adapt_to_v3_params(params)
    for segment in f0_segments:
        note, contour_rows = _build_f0_note(
            segment,
            track_id=next_track_id,
            params=v3_params,
            pad_seconds=pad,
        )
        next_track_id += 1
        notes.append(note)
        contours.extend(contour_rows)
        f0_built.append(
            _F0Note(
                note=note,
                contour=contour_rows,
                segment_frames=segment,
            )
        )

    for f0 in f0_built:
        harmonics, harmonic_contours, next_track_id = _build_harmonic_notes(
            f0,
            params=v3_params,
            cqt_log=cqt_log,
            cqt_bin_freqs=cqt_bin_freqs,
            cqt_bin_log_freqs=cqt_bin_log_freqs,
            cqt_seconds_per_frame=cqt_seconds_per_frame,
            n_cqt_frames=n_cqt_frames,
            pad_seconds=pad,
            next_track_id=next_track_id,
        )
        notes.extend(harmonics)
        contours.extend(harmonic_contours)

    return NotesV3Result(notes=notes, contours=contours)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _decode_f0(
    *,
    cqt_log: np.ndarray,
    cqt_bin_log_freqs: np.ndarray,
    cqt_seconds_per_frame: float,
    params: HarmonicViterbiParams,
    segmentation: SegmentationParams,
) -> list[list[_RefinedFrame]]:
    """Run harmonic-sum emission + log-frequency Viterbi smoothing.

    Returns a list of contiguous voiced runs as ``_RefinedFrame``
    segments. Each ``_RefinedFrame`` carries ``subharmonic_octave = 0``
    (the column is reserved / unused in v5).
    """
    _, candidate_log_freqs = _build_candidate_grid(
        cqt_bin_log_freqs=cqt_bin_log_freqs,
        params=params,
    )
    if candidate_log_freqs.size == 0:
        return []

    emission = _compute_emission(
        cqt_log=cqt_log,
        cqt_bin_log_freqs=cqt_bin_log_freqs,
        candidate_log_freqs=candidate_log_freqs,
        params=params,
    )
    if emission.size == 0:
        return []

    voiced_frame = _voicing_mask_from_cqt(cqt_log, params=params)

    voiced_mask, f0_indices = _viterbi_decode(
        emission=emission,
        candidate_log_freqs=candidate_log_freqs,
        voiced_frame=voiced_frame,
        params=params,
    )

    refined: list[_RefinedFrame] = []
    for t in range(emission.shape[0]):
        if not bool(voiced_mask[t]):
            continue
        idx = int(f0_indices[t])
        refined.append(
            _RefinedFrame(
                frame_index=t,
                time_offset_s=float(t * cqt_seconds_per_frame),
                log_frequency=float(candidate_log_freqs[idx]),
                strength=float(emission[t, idx]),
                subharmonic_octave=0,
            )
        )

    if len(refined) < segmentation.min_note_frames:
        return []

    runs = _split_runs(refined, min_break_frames=int(segmentation.min_break_frames))
    return [run for run in runs if len(run) >= int(segmentation.min_note_frames)]


def _build_candidate_grid(
    *,
    cqt_bin_log_freqs: np.ndarray,
    params: HarmonicViterbiParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(candidate_cqt_bin_indices, candidate_log_freqs)``.

    Candidates are the subset of CQT bins between
    ``f0_min_hz`` and ``f0_max_hz``. Operating on existing CQT bins keeps
    the harmonic-bin lookup pure-integer (``c + bpo · log2(k)``) and lets
    later iterations swap in a finer/coarser grid by adjusting the CQT.
    """
    lo_log = math.log2(max(float(params.f0_min_hz), 1e-9))
    hi_log = math.log2(max(float(params.f0_max_hz), 1e-9))
    mask = (cqt_bin_log_freqs >= lo_log) & (cqt_bin_log_freqs <= hi_log)
    indices = np.flatnonzero(mask).astype(np.int64)
    return indices, cqt_bin_log_freqs[indices].astype(np.float64)


def _harmonic_weights(weight_kind: str, n_harmonics: int) -> np.ndarray:
    """Per-harmonic weight vector (length ``n_harmonics``)."""
    n = max(int(n_harmonics), 1)
    ks = np.arange(1, n + 1, dtype=np.float64)
    if weight_kind == "uniform":
        return np.ones(n, dtype=np.float64)
    if weight_kind == "inv_k":
        return 1.0 / ks
    if weight_kind == "inv_sqrt_k":
        return 1.0 / np.sqrt(ks)
    raise ValueError(
        f"unknown harmonic_weight: {weight_kind!r}; expected "
        "'uniform', 'inv_k', or 'inv_sqrt_k'"
    )


def _compute_emission(
    *,
    cqt_log: np.ndarray,
    cqt_bin_log_freqs: np.ndarray,
    candidate_log_freqs: np.ndarray,
    params: HarmonicViterbiParams,
) -> np.ndarray:
    """Per-frame, per-candidate harmonic-sum emission score.

    Shape: ``(n_frames, n_candidates)``. Higher score = better F0
    explanation. Bins that fail the noise-floor + local-peak gates
    contribute 0, so this score is non-negative.
    """
    n_bins, n_frames = cqt_log.shape
    n_candidates = candidate_log_freqs.size
    if n_candidates == 0 or n_frames == 0:
        return np.zeros((n_frames, 0), dtype=np.float64)

    weights = _harmonic_weights(params.harmonic_weight, params.n_harmonics)
    cents_window_log = float(params.cents_tolerance) / _CENTS_PER_OCTAVE
    max_cqt_log_freq = float(cqt_bin_log_freqs[-1])

    # Precompute strict 3-bin local-peak mask per frame (matches v4).
    is_peak = np.zeros_like(cqt_log, dtype=bool)
    if n_bins >= 3:
        middle = cqt_log[1:-1, :]
        is_peak[1:-1, :] = (middle > cqt_log[:-2, :]) & (middle > cqt_log[2:, :])
        is_peak[0, :] = cqt_log[0, :] > cqt_log[1, :]
        is_peak[-1, :] = cqt_log[-1, :] > cqt_log[-2, :]

    emission = np.zeros((n_frames, n_candidates), dtype=np.float64)
    for t in range(n_frames):
        column = cqt_log[:, t]
        peaks_column = is_peak[:, t]
        floor, mad = _frame_noise_floor(column)
        mad = max(mad, _MIN_MAD)
        threshold = floor + float(params.k_noise) * mad
        for c, base_log in enumerate(candidate_log_freqs):
            total = 0.0
            count = 0
            for k_idx, k in enumerate(range(1, weights.size + 1)):
                target_log = float(base_log) + math.log2(float(k))
                if target_log > max_cqt_log_freq:
                    break
                lo_log = target_log - cents_window_log
                hi_log = target_log + cents_window_log
                window_mask = (cqt_bin_log_freqs >= lo_log) & (
                    cqt_bin_log_freqs <= hi_log
                )
                window_bins = np.flatnonzero(window_mask)
                if window_bins.size == 0:
                    continue
                window_magnitudes = column[window_bins]
                best_offset = int(np.argmax(window_magnitudes))
                bin_idx = int(window_bins[best_offset])
                m_n = float(column[bin_idx])
                if not math.isfinite(m_n):
                    continue
                if m_n < threshold:
                    continue
                if not bool(peaks_column[bin_idx]):
                    continue
                contribution = max(0.0, m_n - floor)
                total += float(weights[k_idx]) * contribution
                count += 1
            if count < int(params.min_harmonics_present):
                # Insufficient harmonic alignment for a real F0; the
                # contribution we found is more likely noise that happens
                # to clear the per-bin gates.
                continue
            emission[t, c] = total
    return emission


def _voicing_mask_from_cqt(
    cqt_log: np.ndarray, *, params: HarmonicViterbiParams
) -> np.ndarray:
    """Per-frame voiced/unvoiced flag from CQT peakedness.

    Compares each column's max to the column's noise floor (median of
    the bottom half). Scale-invariant by construction because CQT log
    magnitudes are relative. A frame is voiced when at least one bin
    rises ``tau_voicing`` log-units above its column floor.
    """
    n_bins, n_frames = cqt_log.shape
    if n_frames == 0:
        return np.zeros(0, dtype=bool)
    voiced = np.zeros(n_frames, dtype=bool)
    tau = float(params.tau_voicing)
    for t in range(n_frames):
        column = cqt_log[:, t]
        floor, _ = _frame_noise_floor(column)
        voiced[t] = float(column.max()) - float(floor) > tau
    return voiced


def _viterbi_decode(
    *,
    emission: np.ndarray,
    candidate_log_freqs: np.ndarray,
    voiced_frame: np.ndarray,
    params: HarmonicViterbiParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode a smooth F0 sequence with hard voicing gating.

    Returns ``(voiced_mask, f0_indices)``. ``voiced_mask`` is a bool
    array of length ``n_frames``; ``f0_indices`` holds the candidate
    index for voiced frames (undefined and ignored at unvoiced frames).
    """
    n_frames, n_candidates = emission.shape
    if n_candidates == 0 or n_frames == 0:
        return (
            np.zeros(n_frames, dtype=bool),
            np.zeros(n_frames, dtype=np.int64),
        )

    n_states = n_candidates + 1  # candidates 0..N-1, rest = N
    rest_idx = n_candidates

    emission_cost = np.full((n_frames, n_states), np.inf, dtype=np.float64)
    for t in range(n_frames):
        if bool(voiced_frame[t]):
            voiced_emission = emission[t, :]
            if float(voiced_emission.max()) > 0.0:
                emission_cost[t, :n_candidates] = -voiced_emission
            # Rest is also reachable on a voiced frame so the Viterbi
            # can choose to end/begin a note here; cost = 0 mirrors a
            # neutral baseline relative to negative voiced costs.
            emission_cost[t, rest_idx] = 0.0
        else:
            # Forced rest: only the rest state has finite cost.
            emission_cost[t, rest_idx] = 0.0

    # Transition cost matrix. Shape (n_states, n_states); row = prev,
    # col = curr. Voiced→voiced uses squared-log-octave distance; the
    # rest row/column uses ``voicing_transition_cost``.
    log_diffs = candidate_log_freqs[:, np.newaxis] - candidate_log_freqs[np.newaxis, :]
    voiced_to_voiced = float(params.transition_lambda) * (log_diffs**2)
    trans = np.zeros((n_states, n_states), dtype=np.float64)
    trans[:n_candidates, :n_candidates] = voiced_to_voiced
    voicing_cost = float(params.voicing_transition_cost)
    trans[:n_candidates, rest_idx] = voicing_cost
    trans[rest_idx, :n_candidates] = voicing_cost
    trans[rest_idx, rest_idx] = 0.0

    dp = np.full((n_frames, n_states), np.inf, dtype=np.float64)
    backptr = np.zeros((n_frames, n_states), dtype=np.int64)
    dp[0, :] = emission_cost[0, :]
    for t in range(1, n_frames):
        candidates = dp[t - 1, :, np.newaxis] + trans  # (n_states, n_states)
        prev_best = np.argmin(candidates, axis=0)
        best_costs = candidates[prev_best, np.arange(n_states)]
        dp[t, :] = emission_cost[t, :] + best_costs
        backptr[t, :] = prev_best

    states = np.zeros(n_frames, dtype=np.int64)
    states[-1] = int(np.argmin(dp[-1, :]))
    for t in range(n_frames - 1, 0, -1):
        states[t - 1] = int(backptr[t, states[t]])

    voiced_mask = states != rest_idx
    f0_indices = np.where(voiced_mask, states, 0).astype(np.int64)
    return voiced_mask, f0_indices


def _adapt_to_v3_params(params: ExtractNotesV5Params) -> Any:
    """Wrap v5 params in a v3-shaped namespace for the reused builders.

    ``_build_f0_note`` and ``_build_harmonic_notes`` only read the fields
    they need; they never touch ``subharmonic`` or ``hps`` or
    ``harmonic_viterbi``. A simple namespace keeps the call sites
    byte-identical to v3 / v4.
    """

    class _Namespace:
        job_id = params.job_id
        event_id = params.event_id
        event_start_utc = params.event_start_utc
        pad_seconds = params.pad_seconds
        cqt = params.cqt
        stft = params.stft
        segmentation = params.segmentation
        harmonic = params.harmonic
        midi = params.midi

    return _Namespace()
