"""Slope-based F0 de-spike note extractor (Piano Roll Notes v6).

v6 is structurally :func:`extract_notes_v5` with one inserted step: a
slope-based de-spike pass over the decoded F0 segments, applied *before*
note building. v5's harmonic-Viterbi decoder usually produces smooth
contours, but a strong-enough wrong-octave / wrong-harmonic emission over
a short run of frames can still leave a surviving spike — the F0 jumps
far off the local trajectory and immediately returns. Because harmonic
siblings inherit the F0 bend by cents conservation, every harmonic ribbon
mirrors the same excursion.

The de-spike pass is a slew-rate anchor walk (spec §4): walk frames
left → right holding a trusted anchor, accept frames within a per-frame
log-frequency budget of the anchor, and excise + linearly bridge the
log-frequency of any frames that fall outside the budget. Detection is a
pure slope threshold ("steep is always an error" — no return-to-baseline
guard). The note stays one continuous contour: spike frames are retained
(timing, ``frame_index``, ``strength``, ``subharmonic_octave`` unchanged)
and only their log-frequency is rewritten.

See ``docs/specs/2026-05-29-piano-roll-notes-v6-f0-despike-design.md`` for
the algorithm specification and ADR-072 for the design rationale.

Pure functions: the worker is responsible for loading audio and writing
parquet. The ``ridge_sidecar_rows`` parameter is accepted for signature
parity with v3/v4/v5 and is unused (v6, like v5, derives F0 from the CQT).
"""

from __future__ import annotations

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
from humpback.processing.piano_roll_cqt import CQTParams, compute_event_cqt

__all__ = [
    "DespikeParams",
    "HarmonicViterbiParams",
    "STFTParams",
    "SegmentationParams",
    "HarmonicSearchParams",
    "MidiRangeParams",
    "ExtractNotesV6Params",
    "ContourFrame",
    "NoteV3",
    "NotesV3Result",
    "despike_f0_segments",
    "extract_notes_v6",
]

# Floating-point slack for the slope envelope comparison so a frame
# sitting exactly on the budget is accepted rather than flagged.
_SLOPE_EPS = 1e-9


@dataclass(frozen=True, slots=True)
class DespikeParams:
    """Slew-rate F0 de-spike parameters (spec §5).

    ``enabled=False`` skips the pass entirely, making v6 byte-identical to
    v5 for the same audio/params. ``max_slope_oct_per_s`` is the slope
    threshold in octaves per second; a frame-to-frame motion steeper than
    this is treated as an artifact. ``max_spike_frames`` bounds the
    worst-case excursion width: an excursion that never returns within
    this many frames is accepted as a genuine level change rather than
    excised indefinitely.
    """

    enabled: bool = True
    max_slope_oct_per_s: float = 6.0
    max_spike_frames: int = 12


@dataclass(frozen=True, slots=True)
class ExtractNotesV6Params:
    """All settings + per-event identity used by the v6 extractor.

    Mirrors :class:`ExtractNotesV5Params` field-for-field plus a
    ``despike`` sub-param. STFT, CQT, segmentation, harmonic, and MIDI
    sub-params carry the v5 defaults unchanged (including the 30 Hz STFT
    band floor and the v5 ``min_break_frames``).
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
    segmentation: SegmentationParams = field(
        default_factory=lambda: SegmentationParams(min_break_frames=6)
    )
    harmonic: HarmonicSearchParams = field(default_factory=HarmonicSearchParams)
    midi: MidiRangeParams = field(default_factory=MidiRangeParams)
    despike: DespikeParams = field(default_factory=DespikeParams)


def extract_notes_v6(
    audio: np.ndarray,
    sample_rate: int,
    *,
    params: ExtractNotesV6Params,
    ridge_sidecar_rows: Optional[Sequence[Mapping[str, Any]]] = None,
) -> NotesV3Result:
    """Extract de-spiked harmonic-Viterbi F0 + harmonics notes for one event.

    Same input/output contract as :func:`extract_notes_v5`. The
    ``ridge_sidecar_rows`` argument is accepted for signature parity and
    unused. Returns ``NotesV3Result``; ``subharmonic_octave`` is 0 for
    every contour frame (v5/v6 have no divisor concept).
    """
    del ridge_sidecar_rows  # accepted for signature parity; unused in v6
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


# ---------------------------------------------------------------------------
# De-spike pass
# ---------------------------------------------------------------------------


def despike_f0_segments(
    segments: list[list[_RefinedFrame]],
    *,
    dt: float,
    params: DespikeParams,
) -> list[list[_RefinedFrame]]:
    """Apply the slew-rate de-spike to every decoded F0 segment.

    Returns segments with the same length and frame timing; only the
    ``log_frequency`` of excised frames is rewritten. A no-op when
    ``params.enabled`` is ``False`` or ``dt`` is non-positive.
    """
    if not params.enabled or dt <= 0.0:
        return segments
    return [_despike_segment(seg, dt=dt, params=params) for seg in segments]


def _despike_segment(
    frames: list[_RefinedFrame],
    *,
    dt: float,
    params: DespikeParams,
) -> list[_RefinedFrame]:
    n = len(frames)
    if n < 3:
        return frames
    lf = np.asarray([f.log_frequency for f in frames], dtype=np.float64)
    new_lf = lf.copy()
    max_step_log = float(params.max_slope_oct_per_s) * float(dt)
    if max_step_log <= 0.0:
        return frames
    max_spike = max(int(params.max_spike_frames), 1)

    # Leading-spike re-seed: a steep first transition means frame 0 may be
    # an isolated leading excursion. Scan to the first frame with an
    # in-envelope right neighbour and hold the leading frames at that
    # value (constant extrapolation), then start the walk there.
    start = 0
    if abs(lf[1] - lf[0]) > max_step_log + _SLOPE_EPS:
        s = 1
        while s < n - 1 and abs(lf[s + 1] - lf[s]) > max_step_log + _SLOPE_EPS:
            s += 1
        if s > 0:
            new_lf[:s] = lf[s]
            start = s

    anchor = start
    i = anchor + 1
    while i < n:
        reach = float(i - anchor) * max_step_log + _SLOPE_EPS
        in_envelope = abs(lf[i] - lf[anchor]) <= reach
        force_anchor = (i - anchor) >= max_spike
        if in_envelope or force_anchor:
            _bridge(new_lf, anchor, i, lf)
            anchor = i
        i += 1

    # Trailing spike: frames after the last anchor never re-entered the
    # envelope; hold the last good value (constant extrapolation).
    if anchor < n - 1:
        new_lf[anchor + 1 :] = lf[anchor]

    if np.array_equal(new_lf, lf):
        return frames
    return [
        _RefinedFrame(
            frame_index=frame.frame_index,
            time_offset_s=frame.time_offset_s,
            log_frequency=float(new_lf[idx]),
            strength=frame.strength,
            subharmonic_octave=frame.subharmonic_octave,
        )
        for idx, frame in enumerate(frames)
    ]


def _bridge(new_lf: np.ndarray, anchor: int, target: int, lf: np.ndarray) -> None:
    """Linearly interpolate log-frequency across excised interior frames.

    Frames strictly between ``anchor`` and ``target`` are rewritten on the
    straight line from ``lf[anchor]`` to ``lf[target]``. Anchor and target
    frames keep their original values.
    """
    span = target - anchor
    if span <= 1:
        return
    lo = float(lf[anchor])
    hi = float(lf[target])
    for k in range(anchor + 1, target):
        new_lf[k] = lo + (hi - lo) * float(k - anchor) / float(span)


def _to_v5_params(params: ExtractNotesV6Params) -> ExtractNotesV5Params:
    """Project v6 params onto the v5 shape for the shared note builders.

    The v3 builders reached through :func:`_adapt_to_v3_params` only read
    the STFT/CQT/segmentation/harmonic/MIDI fields, which v6 mirrors from
    v5 one-to-one; the ``despike`` sub-param has no role past de-spiking.
    """
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
