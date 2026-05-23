"""HPS-based F0 + harmonics note extractor (Piano Roll Notes v4).

Replaces the v3 per-frame octave-halving subharmonic refinement with an
HPS-style (Harmonic Product / Sum Spectrum) F0 estimator that scores
candidate fundamentals ``ridge / d`` for ``d ∈ {1..6}`` by total
harmonic-stack support in the CQT column. The STFT ridge tracker still
seeds frame presence and a frequency anchor; HPS chooses which
sub-divisor of the ridge represents the true F0 each frame.

See ``docs/specs/2026-05-23-piano-roll-notes-v4-hps-f0-design.md`` and
ADR-070 for the design rationale. The pipeline shape (ridge resolution →
F0 selection → coherent-contour segmentation → harmonic siblings) is
unchanged from v3; only the F0 selection stage differs.

Pure functions: the worker is responsible for loading audio, reading the
encoder's ridge sidecar, and writing parquet.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

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
    _align_cqt_frame,
    _build_f0_note,
    _build_harmonic_notes,
    _cqt_bin_frequencies,
    _F0Note,
    _frame_noise_floor,
    _majority_smooth,
    _RefinedFrame,
    _resolve_ridge_contour,
    _segment_f0_runs,
)
from humpback.processing.piano_roll_cqt import CQTParams, compute_event_cqt

__all__ = [
    "HPSParams",
    "STFTParams",
    "SegmentationParams",
    "HarmonicSearchParams",
    "MidiRangeParams",
    "ExtractNotesV4Params",
    "ContourFrame",
    "NoteV3",
    "NotesV3Result",
    "extract_notes_v4",
]


_CENTS_PER_OCTAVE = 1200.0
# Floor for the MAD term in the per-frame noise-floor estimate. When the
# CQT column has a sparse signal in a few bins and silence elsewhere
# (pure tones), the bottom-half MAD collapses to zero and the
# "above floor" test trivially admits every bin. The clamp keeps the
# threshold meaningful in that edge case.
_MIN_MAD: float = 0.3


@dataclass(frozen=True, slots=True)
class HPSParams:
    """HPS F0 selection (spec §4.3).

    ``candidate_divisors`` enumerates which integer divisors of the ridge
    frequency are tried as candidate fundamentals; ``d=1`` keeps the ridge
    as F0 while ``d=k`` interprets the ridge as the ``k``-th harmonic of
    a true F0 at ``ridge / k``. The ridge is biased toward high-energy
    upper harmonics by the Viterbi tracker; HPS reclaims the true F0 by
    asking which candidate's harmonic stack best explains the CQT column.

    ``low_band_min_harmonics`` and ``low_band_penalty`` jointly suppress
    sub-100 Hz false positives. Broadband infrasonic noise (hydrophone
    self-noise, distant ship traffic) can clear a single-bin energy
    threshold on its own; requiring multiple harmonics in the expected
    integer-ratio pattern filters those out, since random noise does not
    align across many bins.
    """

    n_harmonics: int = 8
    cents_tolerance: float = 50.0
    k_noise: float = 2.0
    candidate_divisors: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    smoothing_frames: int = 5
    low_band_penalty: float = 0.5
    low_band_threshold_hz: float = 100.0
    low_band_min_harmonics: int = 3
    high_band_min_harmonics: int = 2
    # Minimum log-magnitude above the per-frame noise floor required for
    # a harmonic bin to count toward ``count_present``. Default 1.0
    # (~8.7 dB). Independently from the noise-floor threshold, this gate
    # prevents pure-tone columns (where the noise floor is at the
    # CQT epsilon) from declaring every bin "present" with zero real
    # contribution.
    min_above_floor: float = 1.0
    # A real harmonic stack has all partials within a bounded dynamic
    # range of each other. CQT filter leakage from a strong bin produces
    # weak shoulder peaks at the bin's subharmonic positions; those are
    # 30+ dB below the parent peak. Requiring each "present" harmonic to
    # be within ``max_harmonic_dynamic_range_log`` of the candidate's
    # strongest harmonic rejects those artifacts.
    max_harmonic_dynamic_range_log: float = 3.0


@dataclass(frozen=True, slots=True)
class ExtractNotesV4Params:
    """All settings + per-event identity used by ``extract_notes_v4``.

    Mirrors ``ExtractNotesV3Params`` field-for-field with ``hps``
    replacing ``subharmonic``. The default ``STFTParams`` here drops the
    band floor from 100 Hz to 30 Hz; v3 retains its own 100 Hz default
    via its own dataclass instance.
    """

    job_id: str
    event_id: str
    event_start_utc: float
    pad_seconds: float = 0.05
    cqt: CQTParams = field(default_factory=CQTParams)
    stft: STFTParams = field(default_factory=lambda: STFTParams(min_frequency_hz=30.0))
    hps: HPSParams = field(default_factory=HPSParams)
    segmentation: SegmentationParams = field(default_factory=SegmentationParams)
    harmonic: HarmonicSearchParams = field(default_factory=HarmonicSearchParams)
    midi: MidiRangeParams = field(default_factory=MidiRangeParams)


def extract_notes_v4(
    audio: np.ndarray,
    sample_rate: int,
    *,
    params: ExtractNotesV4Params,
    ridge_sidecar_rows: Optional[Sequence[Mapping[str, Any]]] = None,
) -> NotesV3Result:
    """Extract HPS-aligned F0 + harmonics notes for one event.

    Same input/output contract as :func:`extract_notes_v3`. Returns
    ``NotesV3Result`` so worker code and downstream parquet schemas stay
    unchanged across versions; only the populated ``log_frequency`` per
    frame and the semantic of the ``subharmonic_octave`` column shift
    (v4 stores ``chosen_divisor − 1``, 0..5, while v3 stored an octave
    halving count, 0..3).
    """
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
    cqt_seconds_per_frame = float(params.cqt.hop_length) / float(target_sr)
    cqt_bin_freqs = _cqt_bin_frequencies(params.cqt)
    cqt_bin_log_freqs = np.log2(cqt_bin_freqs)

    ridge_frames = _resolve_ridge_contour(
        samples=samples,
        sample_rate=target_sr,
        stft=params.stft,
        sidecar=ridge_sidecar_rows,
    )
    if len(ridge_frames) < params.segmentation.min_note_frames:
        return NotesV3Result(notes=[], contours=[])

    refined = _score_f0_candidates(
        ridge_frames,
        cqt_log=cqt_log,
        cqt_bin_freqs=cqt_bin_freqs,
        cqt_bin_log_freqs=cqt_bin_log_freqs,
        cqt_seconds_per_frame=cqt_seconds_per_frame,
        n_cqt_frames=n_cqt_frames,
        params=params.hps,
    )

    f0_segments = _segment_f0_runs(refined, params=params.segmentation)
    if not f0_segments:
        return NotesV3Result(notes=[], contours=[])

    notes: list[NoteV3] = []
    contours: list[ContourFrame] = []
    next_track_id = 0
    pad = float(params.pad_seconds)

    f0_built: list[_F0Note] = []
    # ``_build_f0_note`` / ``_build_harmonic_notes`` are version-agnostic:
    # they read ``log_frequency`` and ``subharmonic_octave`` off
    # ``_RefinedFrame`` without interpreting the latter, so they slot in
    # unchanged for v4.
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


def _score_f0_candidates(
    ridge: list[Any],
    *,
    cqt_log: np.ndarray,
    cqt_bin_freqs: np.ndarray,
    cqt_bin_log_freqs: np.ndarray,
    cqt_seconds_per_frame: float,
    n_cqt_frames: int,
    params: HPSParams,
) -> list[_RefinedFrame]:
    """Per-frame HPS-style F0 selection over the ridge (spec §4.2)."""
    if not ridge:
        return []

    cqt_max_log_freq = float(cqt_bin_log_freqs[-1])
    cents_window_log = float(params.cents_tolerance) / _CENTS_PER_OCTAVE
    divisors = tuple(int(d) for d in params.candidate_divisors)
    if not divisors:
        divisors = (1,)
    low_threshold_log = math.log2(max(float(params.low_band_threshold_hz), 1e-9))

    # Precompute strict 3-bin local-maxima per frame. Filter skirts from
    # a CQT bin loaded with signal create broad magnitude shoulders at
    # neighboring frequencies — those bins are above the noise floor but
    # are not real harmonics. Requiring each "present" harmonic to be a
    # local maximum rejects those shoulders structurally.
    n_bins = cqt_log.shape[0]
    is_peak = np.zeros_like(cqt_log, dtype=bool)
    if n_bins >= 3:
        middle = cqt_log[1:-1, :]
        is_peak[1:-1, :] = (middle > cqt_log[:-2, :]) & (middle > cqt_log[2:, :])
        is_peak[0, :] = cqt_log[0, :] > cqt_log[1, :]
        is_peak[-1, :] = cqt_log[-1, :] > cqt_log[-2, :]

    raw_divisors = np.ones(len(ridge), dtype=np.int64)
    for idx, ridge_frame in enumerate(ridge):
        cqt_frame_idx = _align_cqt_frame(
            ridge_frame.time_offset_s, cqt_seconds_per_frame, n_cqt_frames
        )
        column = cqt_log[:, cqt_frame_idx]
        peaks_column = is_peak[:, cqt_frame_idx]
        floor, mad = _frame_noise_floor(column)
        mad = max(mad, _MIN_MAD)
        threshold = floor + float(params.k_noise) * mad
        min_above_floor = float(params.min_above_floor)
        max_dynamic_range = float(params.max_harmonic_dynamic_range_log)
        best_d = 1
        best_score = -math.inf
        ridge_log = float(ridge_frame.log_frequency)
        for d in divisors:
            candidate_log = ridge_log - math.log2(float(d))
            is_low = candidate_log < low_threshold_log
            k_min = (
                int(params.low_band_min_harmonics)
                if is_low
                else int(params.high_band_min_harmonics)
            )
            # First pass: collect every harmonic that clears the noise,
            # peak, and min_above_floor gates.
            survivors: list[float] = []
            for n in range(1, int(params.n_harmonics) + 1):
                target_log = candidate_log + math.log2(float(n))
                if target_log > cqt_max_log_freq:
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
                above_floor = m_n - floor
                is_real_peak = bool(peaks_column[bin_idx])
                if is_real_peak and m_n >= threshold and above_floor >= min_above_floor:
                    survivors.append(m_n)
            if not survivors:
                continue
            # Second pass: drop survivors more than ``max_dynamic_range``
            # below the strongest one. Real harmonic stacks span a
            # bounded dynamic range; CQT leakage shoulders at subharmonic
            # frequencies of a strong tone sit 30+ dB below the parent
            # peak and get dropped here.
            max_m = max(survivors)
            kept = [m for m in survivors if max_m - m <= max_dynamic_range]
            count_present = len(kept)
            if count_present < k_min:
                continue
            raw_score = float(sum(max(0.0, m - floor) for m in kept))
            penalty = float(params.low_band_penalty) if is_low else 0.0
            score = raw_score - penalty
            # Tie-break toward the smallest divisor (Occam's razor: when
            # two candidates score identically, the one explaining the
            # ridge as F0 itself wins over an artifact divisor that gets
            # the same numerical support from incidental harmonic
            # alignments.
            if score > best_score or (score == best_score and d < best_d):
                best_score = score
                best_d = d
        # Frames where no candidate cleared the harmonic-count gate fall
        # back to the ridge itself (``d=1``). The majority-smoothing pass
        # below removes any orphan flips before they reach segmentation.
        raw_divisors[idx] = best_d

    smoothed = _majority_smooth(raw_divisors, window=int(params.smoothing_frames))
    refined: list[_RefinedFrame] = []
    for idx, ridge_frame in enumerate(ridge):
        d = int(smoothed[idx])
        log_freq = float(ridge_frame.log_frequency) - math.log2(float(d))
        refined.append(
            _RefinedFrame(
                frame_index=int(ridge_frame.frame_index),
                time_offset_s=float(ridge_frame.time_offset_s),
                log_frequency=log_freq,
                strength=float(ridge_frame.strength),
                subharmonic_octave=d - 1,
            )
        )
    return refined


def _adapt_to_v3_params(params: ExtractNotesV4Params) -> Any:
    """Wrap v4 params in a v3-shaped namespace for the reused builders.

    ``_build_f0_note`` and ``_build_harmonic_notes`` only read the fields
    they need (``job_id``, ``event_id``, ``event_start_utc``, ``midi``,
    ``harmonic``); they never touch ``subharmonic`` or ``hps``. A simple
    namespace keeps the call sites byte-identical to v3.
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
