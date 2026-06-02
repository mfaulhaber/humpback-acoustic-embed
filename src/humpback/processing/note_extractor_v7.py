"""Residual-discontinuity note extractor (Piano Roll Notes v7).

v7 keeps the v6 harmonic-Viterbi decode and slope de-spike, then adds two
small post-decode passes before note construction:

* split any remaining high-slope F0 discontinuity into separate notes, so
  a branch jump is not rendered as a continuous MPE pitch bend;
* rescue flat decoded F0 segments when the persisted Event Encoder STFT
  ridge carries a smooth upsweep/downsweep that the harmonic-Viterbi state
  path missed.

The parquet/MIDI contract remains the v3+ contract: one F0 note plus
harmonic siblings with per-frame contour rows.
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
    _build_f0_note,
    _build_harmonic_notes,
    _cqt_bin_frequencies,
    _F0Note,
    _RefinedFrame,
)
from humpback.processing.note_extractor_v5 import (
    ExtractNotesV5Params,
    HarmonicViterbiParams,
    _adapt_to_v3_params,
    _decode_f0,
)
from humpback.processing.note_extractor_v6 import (
    DespikeParams,
    despike_f0_segments,
)
from humpback.processing.piano_roll_cqt import CQTParams, compute_event_cqt

__all__ = [
    "DespikeParams",
    "DiscontinuityParams",
    "RidgeRescueParams",
    "HarmonicViterbiParams",
    "STFTParams",
    "SegmentationParams",
    "HarmonicSearchParams",
    "MidiRangeParams",
    "ExtractNotesV7Params",
    "ContourFrame",
    "NoteV3",
    "NotesV3Result",
    "rescue_flat_segments_from_ridge",
    "split_residual_discontinuities",
    "extract_notes_v7",
]

_SLOPE_EPS = 1e-9


@dataclass(frozen=True, slots=True)
class DiscontinuityParams:
    """Split residual branch jumps that v6 intentionally leaves intact."""

    enabled: bool = True
    max_continuous_slope_oct_per_s: float = 6.0


@dataclass(frozen=True, slots=True)
class RidgeRescueParams:
    """Use a smooth persisted STFT ridge when decoded F0 is suspiciously flat."""

    enabled: bool = True
    max_decoded_span_semitones: float = 2.0
    min_ridge_span_semitones: float = 5.0
    min_overlap_frames: int = 8
    max_ratio_mad_semitones: float = 2.0
    min_carrier_harmonic: int = 1
    max_carrier_harmonic: int = 32


@dataclass(frozen=True, slots=True)
class ExtractNotesV7Params:
    """All settings + per-event identity used by the v7 extractor."""

    job_id: str
    event_id: str
    event_start_utc: float
    pad_seconds: float = 0.05
    cqt: CQTParams = field(default_factory=CQTParams)
    stft: STFTParams = field(default_factory=lambda: STFTParams(min_frequency_hz=30.0))
    harmonic_viterbi: HarmonicViterbiParams = field(
        default_factory=HarmonicViterbiParams
    )
    segmentation: SegmentationParams = field(
        default_factory=lambda: SegmentationParams(min_break_frames=6)
    )
    harmonic: HarmonicSearchParams = field(default_factory=HarmonicSearchParams)
    midi: MidiRangeParams = field(default_factory=MidiRangeParams)
    despike: DespikeParams = field(default_factory=DespikeParams)
    discontinuity: DiscontinuityParams = field(default_factory=DiscontinuityParams)
    ridge_rescue: RidgeRescueParams = field(default_factory=RidgeRescueParams)


def extract_notes_v7(
    audio: np.ndarray,
    sample_rate: int,
    *,
    params: ExtractNotesV7Params,
    ridge_sidecar_rows: Optional[Sequence[Mapping[str, Any]]] = None,
) -> NotesV3Result:
    """Extract v7 F0 + harmonic notes for one event."""

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
        pad_seconds=params.pad_seconds,
        params=params.harmonic_viterbi,
        segmentation=params.segmentation,
    )
    if not f0_segments:
        return NotesV3Result(notes=[], contours=[])

    f0_segments = despike_f0_segments(
        f0_segments,
        dt=cqt_seconds_per_frame,
        params=params.despike,
    )
    f0_segments = rescue_flat_segments_from_ridge(
        f0_segments,
        ridge_sidecar_rows=ridge_sidecar_rows,
        pad_seconds=params.pad_seconds,
        params=params.ridge_rescue,
    )
    f0_segments = split_residual_discontinuities(
        f0_segments,
        params=params.discontinuity,
        min_note_frames=params.segmentation.min_note_frames,
    )
    if not f0_segments:
        return NotesV3Result(notes=[], contours=[])

    notes: list[NoteV3] = []
    contours: list[ContourFrame] = []
    next_track_id = 0
    pad = float(params.pad_seconds)
    f0_built: list[_F0Note] = []
    v3_params = _adapt_to_v3_params(_to_v5_params(params))
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


def split_residual_discontinuities(
    segments: list[list[_RefinedFrame]],
    *,
    params: DiscontinuityParams,
    min_note_frames: int,
) -> list[list[_RefinedFrame]]:
    """Split segments where adjacent decoded F0 frames exceed the slope budget."""

    if not params.enabled:
        return segments
    max_slope = float(params.max_continuous_slope_oct_per_s)
    if max_slope <= 0.0:
        return segments

    min_len = max(int(min_note_frames), 1)
    split_segments: list[list[_RefinedFrame]] = []
    for segment in segments:
        if len(segment) < 2:
            if len(segment) >= min_len:
                split_segments.append(segment)
            continue

        start = 0
        for idx in range(1, len(segment)):
            prev = segment[idx - 1]
            cur = segment[idx]
            dt = float(cur.time_offset_s) - float(prev.time_offset_s)
            if dt <= 0.0:
                continue
            slope = abs(float(cur.log_frequency) - float(prev.log_frequency)) / dt
            if slope > max_slope + _SLOPE_EPS:
                if idx - start >= min_len:
                    split_segments.append(segment[start:idx])
                start = idx
        if len(segment) - start >= min_len:
            split_segments.append(segment[start:])
    return split_segments


def rescue_flat_segments_from_ridge(
    segments: list[list[_RefinedFrame]],
    *,
    ridge_sidecar_rows: Optional[Sequence[Mapping[str, Any]]],
    pad_seconds: float,
    params: RidgeRescueParams,
) -> list[list[_RefinedFrame]]:
    """Replace flat decoded F0 with a smooth aligned ridge-derived F0."""

    if not params.enabled or not ridge_sidecar_rows:
        return segments
    ridge = _ridge_arrays(ridge_sidecar_rows)
    if ridge is None:
        return segments
    ridge_times, ridge_log = ridge

    rescued: list[list[_RefinedFrame]] = []
    for segment in segments:
        candidate = _rescue_one_segment(
            segment,
            ridge_times=ridge_times,
            ridge_log=ridge_log,
            pad_seconds=pad_seconds,
            params=params,
        )
        rescued.append(candidate if candidate is not None else segment)
    return rescued


def _rescue_one_segment(
    segment: list[_RefinedFrame],
    *,
    ridge_times: np.ndarray,
    ridge_log: np.ndarray,
    pad_seconds: float,
    params: RidgeRescueParams,
) -> Optional[list[_RefinedFrame]]:
    if len(segment) < max(int(params.min_overlap_frames), 1):
        return None

    decoded_log = np.asarray([f.log_frequency for f in segment], dtype=np.float64)
    decoded_span = _span_semitones(decoded_log)
    if decoded_span > float(params.max_decoded_span_semitones):
        return None

    event_times = np.asarray(
        [float(f.time_offset_s) - float(pad_seconds) for f in segment],
        dtype=np.float64,
    )
    overlap = (event_times >= ridge_times[0]) & (event_times <= ridge_times[-1])
    if int(np.count_nonzero(overlap)) < int(params.min_overlap_frames):
        return None

    interp_log = np.interp(event_times, ridge_times, ridge_log)
    ridge_overlap = interp_log[overlap]
    if _span_semitones(ridge_overlap) < float(params.min_ridge_span_semitones):
        return None

    if _trend_residual_mad_semitones(event_times[overlap], ridge_overlap) > float(
        params.max_ratio_mad_semitones
    ):
        return None

    carrier = _carrier_harmonic(
        ridge_overlap=ridge_overlap,
        decoded_overlap=decoded_log[overlap],
        params=params,
    )
    if carrier is None:
        return None
    carrier_log = math.log2(float(carrier))
    rescued_log = interp_log - carrier_log

    return [
        _RefinedFrame(
            frame_index=frame.frame_index,
            time_offset_s=frame.time_offset_s,
            log_frequency=float(rescued_log[idx]),
            strength=frame.strength,
            subharmonic_octave=frame.subharmonic_octave,
        )
        for idx, frame in enumerate(segment)
    ]


def _ridge_arrays(
    rows: Sequence[Mapping[str, Any]],
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    pairs: list[tuple[float, float, int]] = []
    for row in rows:
        time_s = _finite_float(row.get("frame_time_offset_s"))
        log_frequency = _finite_float(row.get("log_frequency"))
        if time_s is None or log_frequency is None:
            continue
        pairs.append((time_s, log_frequency, int(row.get("frame_index", 0))))
    if len(pairs) < 2:
        return None
    pairs.sort(key=lambda p: (p[0], p[2]))

    times: list[float] = []
    logs: list[float] = []
    last_time: Optional[float] = None
    for time_s, log_frequency, _frame_index in pairs:
        if last_time is not None and abs(time_s - last_time) <= 1e-9:
            logs[-1] = log_frequency
            continue
        times.append(time_s)
        logs.append(log_frequency)
        last_time = time_s
    if len(times) < 2:
        return None
    return np.asarray(times, dtype=np.float64), np.asarray(logs, dtype=np.float64)


def _finite_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _span_semitones(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float((np.max(values) - np.min(values)) * 12.0)


def _trend_residual_mad_semitones(times: np.ndarray, values: np.ndarray) -> float:
    if times.size < 3:
        return 0.0
    centered_t = times - float(np.mean(times))
    design = np.column_stack([centered_t, np.ones_like(centered_t)])
    slope, intercept = np.linalg.lstsq(design, values, rcond=None)[0]
    residual = values - (slope * centered_t + intercept)
    median = float(np.median(residual))
    return float(np.median(np.abs(residual - median)) * 12.0)


def _carrier_harmonic(
    *,
    ridge_overlap: np.ndarray,
    decoded_overlap: np.ndarray,
    params: RidgeRescueParams,
) -> Optional[int]:
    spacing = float(np.median(ridge_overlap - decoded_overlap))
    if not math.isfinite(spacing):
        return None
    carrier = int(round(2.0**spacing))
    lo = max(int(params.min_carrier_harmonic), 1)
    hi = max(int(params.max_carrier_harmonic), lo)
    if carrier < lo or carrier > hi:
        return None
    return carrier


def _to_v5_params(params: ExtractNotesV7Params) -> ExtractNotesV5Params:
    return ExtractNotesV5Params(
        job_id=params.job_id,
        event_id=params.event_id,
        event_start_utc=params.event_start_utc,
        pad_seconds=params.pad_seconds,
        cqt=params.cqt,
        stft=params.stft,
        harmonic_viterbi=params.harmonic_viterbi,
        segmentation=params.segmentation,
        harmonic=params.harmonic,
        midi=params.midi,
    )
