"""Ridge-aware F0 + harmonics note extractor (Piano Roll Notes v3).

Replaces the independent CQT peak-tracker in the v2 pipeline with a single
canonical path: STFT ridge → subharmonic refinement → coherent-contour
note segmentation → CQT harmonic siblings. The extractor produces one
``NoteV3`` per coherent F0 contour plus a per-frame ``ContourFrame``
sidecar that downstream consumers (renderer, MPE MIDI synthesizer,
analysis) read for sub-semitone pitch trajectories.

See ``docs/specs/2026-05-22-piano-roll-mpe-ridge-aligned-design.md`` §5
for the algorithmic specification and ADR-069 for the design rationale.

Pure functions: the worker is responsible for loading audio, reading the
encoder's ridge sidecar, and writing parquet.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

import librosa
import numpy as np

from humpback.processing.piano_roll_cqt import CQTParams, compute_event_cqt
from humpback.processing.ridge_path import RidgePathResult, compute_ridge_path

__all__ = [
    "STFTParams",
    "SubharmonicParams",
    "SegmentationParams",
    "HarmonicSearchParams",
    "MidiRangeParams",
    "ExtractNotesV3Params",
    "ContourFrame",
    "NoteV3",
    "NotesV3Result",
    "extract_notes_v3",
]


_MIDI_A4 = 69.0
_HZ_A4 = 440.0
_CENTS_PER_OCTAVE = 1200.0
_NOTE_UID_NAMESPACE = uuid.UUID("4ce4dc63-a07f-4f3e-a0e0-1b6f1bc6f7e8")


@dataclass(frozen=True, slots=True)
class STFTParams:
    """STFT settings for in-process ridge recomputation.

    Used only when the encoder ridge sidecar is unavailable. The defaults
    mirror the encoder's descriptor STFT (``n_fft=1024``, ``hop=512``)
    but the notes extractor resamples to its own ``sample_rate`` (the
    CQT's ``target_sample_rate``) so STFT and CQT frame grids share an
    integer alignment.
    """

    n_fft: int = 1024
    hop_length: int = 512
    min_frequency_hz: float = 100.0
    max_frequency_hz: float = 8500.0
    candidate_count: int = 5
    smoothness_penalty: float = 8.0
    peak_prominence_ratio: float = 0.0


@dataclass(frozen=True, slots=True)
class SubharmonicParams:
    """Subharmonic refinement (spec §5.2).

    ``min_relative_log_magnitude`` requires the candidate sub-octave bin's
    CQT log magnitude to sit within this many natural-log units of the
    current ridge magnitude. Without it, spectral leakage at ``f₀/2`` can
    pass the bare noise-floor test for pure tones with no real
    sub-fundamental energy. The constant-Q filter bank rejects an octave
    away by ~4 log units even for clean sines, so -2.5 (≈22 dB below the
    ridge) is a permissive threshold that still discriminates leakage
    from a real weak fundamental whose CQT magnitude is within ~15 dB.
    """

    k_sub: float = 2.0
    max_halvings: int = 3
    smoothing_frames: int = 5
    min_relative_log_magnitude: float = -2.5


@dataclass(frozen=True, slots=True)
class SegmentationParams:
    """Coherent-contour F0 segmentation (spec §5.3)."""

    amplitude_floor_percentile: float = 5.0
    min_break_frames: int = 3
    min_note_frames: int = 3


@dataclass(frozen=True, slots=True)
class HarmonicSearchParams:
    """Harmonic sibling extraction (spec §5.4)."""

    min_harmonic: int = 2
    max_harmonic: int = 16
    cents_tolerance: float = 75.0
    min_break_frames: int = 3
    min_note_frames: int = 3


@dataclass(frozen=True, slots=True)
class MidiRangeParams:
    """MIDI pitch + cents clamping (spec §5.3)."""

    min_pitch: int = 12
    max_pitch: int = 120
    cents_clip: float = 9600.0


@dataclass(frozen=True, slots=True)
class ExtractNotesV3Params:
    """All settings + per-event identity used by ``extract_notes_v3``."""

    job_id: str
    event_id: str
    event_start_utc: float
    pad_seconds: float = 0.05
    cqt: CQTParams = field(default_factory=CQTParams)
    stft: STFTParams = field(default_factory=STFTParams)
    subharmonic: SubharmonicParams = field(default_factory=SubharmonicParams)
    segmentation: SegmentationParams = field(default_factory=SegmentationParams)
    harmonic: HarmonicSearchParams = field(default_factory=HarmonicSearchParams)
    midi: MidiRangeParams = field(default_factory=MidiRangeParams)


@dataclass(frozen=True, slots=True)
class ContourFrame:
    """One per-frame row in ``event_note_contours_v3.parquet``."""

    note_uid: str
    frame_index: int
    time_offset_s: float
    cents_from_pitch: float
    harmonic_strength: float
    subharmonic_octave: int


@dataclass(frozen=True, slots=True)
class NoteV3:
    """One note row in ``event_notes_v3.parquet`` (pre-velocity)."""

    note_uid: str
    track_id: int
    f0_track_id: int
    partial_index: int
    midi_pitch: int
    start_utc: float
    start_offset_s: float
    duration_s: float
    peak_magnitude: float
    contour_frame_count: int


@dataclass(frozen=True, slots=True)
class NotesV3Result:
    """Aligned outputs of one ``extract_notes_v3`` call."""

    notes: list[NoteV3]
    contours: list[ContourFrame]


def extract_notes_v3(
    audio: np.ndarray,
    sample_rate: int,
    *,
    params: ExtractNotesV3Params,
    ridge_sidecar_rows: Optional[Sequence[Mapping[str, Any]]] = None,
) -> NotesV3Result:
    """Extract ridge-aligned F0 + harmonics notes for one event.

    Args:
        audio: 1-D event-padded audio buffer in float32.
        sample_rate: Sample rate of ``audio`` in Hz. Resampled internally
            to ``params.cqt.target_sample_rate`` for CQT and STFT work.
        params: Per-event identity and DSP knobs.
        ridge_sidecar_rows: When provided, pre-computed ridge contour rows
            from the encoder ``event_ridges_*.parquet`` sidecar filtered
            to this event (one mapping per frame with the keys
            ``frame_time_offset_s``, ``log_frequency``, ``strength``,
            ``energy_ratio``). When ``None`` the extractor recomputes the
            ridge in-process via :func:`compute_ridge_path` with the
            spec's wider ``max_frequency_hz`` ceiling.

    Returns:
        ``NotesV3Result`` with notes and contour rows aligned by
        ``note_uid``. Empty when the audio is too short or the ridge
        carried fewer than ``segmentation.min_note_frames`` frames.
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

    # ---- CQT for subharmonic + harmonic search ------------------------
    cqt_log = compute_event_cqt(samples, target_sr, params=params.cqt)
    if cqt_log.size == 0:
        return NotesV3Result(notes=[], contours=[])
    n_cqt_frames = int(cqt_log.shape[1])
    cqt_seconds_per_frame = float(params.cqt.hop_length) / float(target_sr)
    cqt_bin_freqs = _cqt_bin_frequencies(params.cqt)
    cqt_bin_log_freqs = np.log2(cqt_bin_freqs)

    # ---- Ridge contour (sidecar or recomputed) ------------------------
    ridge_frames = _resolve_ridge_contour(
        samples=samples,
        sample_rate=target_sr,
        stft=params.stft,
        sidecar=ridge_sidecar_rows,
    )
    if len(ridge_frames) < params.segmentation.min_note_frames:
        return NotesV3Result(notes=[], contours=[])

    # ---- Subharmonic refinement --------------------------------------
    refined = _refine_subharmonic(
        ridge_frames,
        cqt_log=cqt_log,
        cqt_bin_freqs=cqt_bin_freqs,
        cqt_seconds_per_frame=cqt_seconds_per_frame,
        n_cqt_frames=n_cqt_frames,
        params=params.subharmonic,
    )

    # ---- F0 segmentation ---------------------------------------------
    f0_segments = _segment_f0_runs(refined, params=params.segmentation)
    if not f0_segments:
        return NotesV3Result(notes=[], contours=[])

    # ---- Build F0 notes ----------------------------------------------
    notes: list[NoteV3] = []
    contours: list[ContourFrame] = []
    next_track_id = 0
    pad = float(params.pad_seconds)

    f0_built: list[_F0Note] = []
    for segment in f0_segments:
        note, contour_rows = _build_f0_note(
            segment,
            track_id=next_track_id,
            params=params,
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

    # ---- Harmonic siblings -------------------------------------------
    for f0 in f0_built:
        harmonics, harmonic_contours, next_track_id = _build_harmonic_notes(
            f0,
            params=params,
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


@dataclass(frozen=True, slots=True)
class _RidgeFrame:
    """One frame of the F0 ridge prior to subharmonic refinement."""

    frame_index: int
    time_offset_s: float
    log_frequency: float
    strength: float


@dataclass(frozen=True, slots=True)
class _RefinedFrame:
    """One frame after subharmonic refinement."""

    frame_index: int
    time_offset_s: float
    log_frequency: float
    strength: float
    subharmonic_octave: int


@dataclass(frozen=True, slots=True)
class _F0Note:
    note: NoteV3
    contour: list[ContourFrame]
    segment_frames: list[_RefinedFrame]


def _cqt_bin_frequencies(params: CQTParams) -> np.ndarray:
    return np.asarray(
        [
            params.fmin * (2.0 ** (i / params.bins_per_octave))
            for i in range(params.n_bins)
        ],
        dtype=np.float64,
    )


def _resolve_ridge_contour(
    *,
    samples: np.ndarray,
    sample_rate: int,
    stft: STFTParams,
    sidecar: Optional[Sequence[Mapping[str, Any]]],
) -> list[_RidgeFrame]:
    if sidecar is not None:
        frames: list[_RidgeFrame] = []
        for row in sidecar:
            frames.append(
                _RidgeFrame(
                    frame_index=int(row.get("frame_index", 0)),
                    time_offset_s=float(row.get("frame_time_offset_s", 0.0)),
                    log_frequency=float(row.get("log_frequency", 0.0)),
                    strength=float(row.get("strength", 0.0)),
                )
            )
        frames.sort(key=lambda f: (f.time_offset_s, f.frame_index))
        return frames

    if samples.size < stft.n_fft:
        return []

    n_fft = int(stft.n_fft)
    hop = int(stft.hop_length)
    n_frames = (samples.size - n_fft) // hop + 1
    if n_frames <= 1:
        return []
    window = np.hanning(n_fft).astype(np.float64)
    spectra = np.empty((n_frames, n_fft // 2 + 1), dtype=np.float64)
    for i in range(n_frames):
        slice_ = samples[i * hop : i * hop + n_fft].astype(np.float64)
        spectra[i, :] = np.abs(np.fft.rfft(slice_ * window))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    result: RidgePathResult = compute_ridge_path(
        spectra,
        freqs,
        sample_rate=sample_rate,
        hop_length=hop,
        min_frequency_hz=stft.min_frequency_hz,
        max_frequency_hz=stft.max_frequency_hz,
        candidate_count=stft.candidate_count,
        smoothness_penalty=stft.smoothness_penalty,
        peak_prominence_ratio=stft.peak_prominence_ratio,
    )
    if result.log_frequencies.size == 0:
        return []
    seconds_per_frame = float(hop) / float(sample_rate)
    return [
        _RidgeFrame(
            frame_index=int(round(float(result.frame_times[i]) / seconds_per_frame)),
            time_offset_s=float(result.frame_times[i]),
            log_frequency=float(result.log_frequencies[i]),
            strength=float(result.strengths[i]),
        )
        for i in range(int(result.log_frequencies.shape[0]))
    ]


def _refine_subharmonic(
    ridge: list[_RidgeFrame],
    *,
    cqt_log: np.ndarray,
    cqt_bin_freqs: np.ndarray,
    cqt_seconds_per_frame: float,
    n_cqt_frames: int,
    params: SubharmonicParams,
) -> list[_RefinedFrame]:
    """Detect a stronger sub-octave fundamental per frame and smooth (spec §5.2).

    For each ridge frame we ask whether the CQT shows energy at ``f₀/2``
    that exceeds the per-frame noise floor by ``k_sub · MAD``. If yes,
    we halve ``f₀`` and recurse (up to ``max_halvings``). Per-frame
    decisions are then smoothed with a centred ``smoothing_frames``
    majority vote so single-frame flips don't fragment the contour.
    """
    if not ridge:
        return []
    raw_offsets = np.zeros(len(ridge), dtype=np.int64)
    cqt_frame_indices = np.asarray(
        [
            _align_cqt_frame(r.time_offset_s, cqt_seconds_per_frame, n_cqt_frames)
            for r in ridge
        ],
        dtype=np.int64,
    )
    for idx, ridge_frame in enumerate(ridge):
        cqt_frame_idx = int(cqt_frame_indices[idx])
        column = cqt_log[:, cqt_frame_idx]
        floor, mad = _frame_noise_floor(column)
        log_freq = ridge_frame.log_frequency
        # Track current-ridge magnitude in the CQT so we can require the
        # candidate sub-octave bin to be comparable, not merely above the
        # noise floor. Pure tones have leakage at f0/2 that clears the
        # bare floor test on its own.
        current_bin_idx = _nearest_bin(float(2.0**log_freq), cqt_bin_freqs)
        current_magnitude = float(column[current_bin_idx])
        offset = 0
        while offset < int(params.max_halvings):
            candidate_log_freq = log_freq - 1.0
            candidate_hz = float(2.0**candidate_log_freq)
            if candidate_hz <= float(cqt_bin_freqs[0]):
                break
            bin_idx = _nearest_bin(candidate_hz, cqt_bin_freqs)
            magnitude = float(column[bin_idx])
            if magnitude - floor < float(params.k_sub) * mad:
                break
            if magnitude - current_magnitude < float(params.min_relative_log_magnitude):
                break
            offset += 1
            log_freq = candidate_log_freq
            current_magnitude = magnitude
        raw_offsets[idx] = offset

    smoothed = _majority_smooth(raw_offsets, window=int(params.smoothing_frames))
    refined: list[_RefinedFrame] = []
    for idx, ridge_frame in enumerate(ridge):
        offset = int(smoothed[idx])
        refined.append(
            _RefinedFrame(
                frame_index=ridge_frame.frame_index,
                time_offset_s=ridge_frame.time_offset_s,
                log_frequency=ridge_frame.log_frequency - float(offset),
                strength=ridge_frame.strength,
                subharmonic_octave=offset,
            )
        )
    return refined


def _segment_f0_runs(
    refined: list[_RefinedFrame],
    *,
    params: SegmentationParams,
) -> list[list[_RefinedFrame]]:
    """Split the F0 contour on energy gaps and surviving octave jumps.

    The amplitude floor is derived from the per-event strength
    distribution at ``amplitude_floor_percentile``. A run of at least
    ``min_break_frames`` frames below that floor closes the current note.
    A change in ``subharmonic_octave`` between adjacent frames is treated
    as a structural register jump and also closes the current note.
    """
    if len(refined) < params.min_note_frames:
        return []
    strengths = np.asarray([f.strength for f in refined], dtype=np.float64)
    floor = float(np.percentile(strengths, params.amplitude_floor_percentile))
    below_floor = strengths < floor

    segments: list[list[_RefinedFrame]] = []
    current: list[_RefinedFrame] = []
    gap_count = 0
    last_octave: Optional[int] = None
    last_frame_index: Optional[int] = None
    for idx, frame in enumerate(refined):
        # Frames skipped by the ridge tracker (no candidate cleared the
        # prominence floor) appear as a jump in ``frame_index``. Treat
        # the skipped frames as implicit below-floor evidence.
        if last_frame_index is not None:
            skipped = max(0, frame.frame_index - last_frame_index - 1)
            if skipped > 0:
                gap_count += skipped
        if gap_count >= params.min_break_frames:
            if len(current) >= params.min_note_frames:
                segments.append(current)
            current = []
            gap_count = 0
            last_octave = None
        if last_octave is not None and frame.subharmonic_octave != last_octave:
            if len(current) >= params.min_note_frames:
                segments.append(current)
            current = []
            gap_count = 0
        if bool(below_floor[idx]):
            gap_count += 1
            if gap_count >= params.min_break_frames:
                if len(current) >= params.min_note_frames:
                    segments.append(current)
                current = []
                gap_count = 0
                last_octave = None
                last_frame_index = frame.frame_index
                continue
        else:
            gap_count = 0
        current.append(frame)
        last_octave = frame.subharmonic_octave
        last_frame_index = frame.frame_index

    if len(current) >= params.min_note_frames:
        segments.append(current)
    return segments


def _build_f0_note(
    segment: list[_RefinedFrame],
    *,
    track_id: int,
    params: ExtractNotesV3Params,
    pad_seconds: float,
) -> tuple[NoteV3, list[ContourFrame]]:
    log_freqs = np.asarray([f.log_frequency for f in segment], dtype=np.float64)
    midi_continuous = _MIDI_A4 + 12.0 * (log_freqs - math.log2(_HZ_A4))
    raw_midi = int(round(float(np.median(midi_continuous))))
    midi_pitch = int(max(params.midi.min_pitch, min(params.midi.max_pitch, raw_midi)))
    nominal_log = math.log2(_HZ_A4) + (midi_pitch - _MIDI_A4) / 12.0
    cents = _CENTS_PER_OCTAVE * (log_freqs - nominal_log)
    cents = np.clip(cents, -params.midi.cents_clip, params.midi.cents_clip)

    start_offset_s = float(segment[0].time_offset_s) - pad_seconds
    end_offset_s = float(segment[-1].time_offset_s) - pad_seconds
    duration_s = max(end_offset_s - start_offset_s, 0.0)
    start_utc = float(params.event_start_utc) + start_offset_s
    peak_magnitude = float(max((f.strength for f in segment), default=0.0))
    note_uid = _build_note_uid(
        job_id=params.job_id,
        event_id=params.event_id,
        partial_index=0,
        track_id=track_id,
        start_utc=start_utc,
    )
    contour_rows = [
        ContourFrame(
            note_uid=note_uid,
            frame_index=i,
            time_offset_s=float(segment[i].time_offset_s)
            - pad_seconds
            - start_offset_s,
            cents_from_pitch=float(cents[i]),
            harmonic_strength=float(segment[i].strength),
            subharmonic_octave=int(segment[i].subharmonic_octave),
        )
        for i in range(len(segment))
    ]
    note = NoteV3(
        note_uid=note_uid,
        track_id=track_id,
        f0_track_id=track_id,
        partial_index=0,
        midi_pitch=midi_pitch,
        start_utc=start_utc,
        start_offset_s=start_offset_s,
        duration_s=duration_s,
        peak_magnitude=peak_magnitude,
        contour_frame_count=len(contour_rows),
    )
    return note, contour_rows


def _build_harmonic_notes(
    f0: _F0Note,
    *,
    params: ExtractNotesV3Params,
    cqt_log: np.ndarray,
    cqt_bin_freqs: np.ndarray,
    cqt_bin_log_freqs: np.ndarray,
    cqt_seconds_per_frame: float,
    n_cqt_frames: int,
    pad_seconds: float,
    next_track_id: int,
) -> tuple[list[NoteV3], list[ContourFrame], int]:
    """Emit harmonic siblings at integer multiples of the F0 contour.

    Per spec §5.4 a harmonic at multiplier ``n`` is present at frame
    ``t`` when the CQT peak nearest ``n·f₀(t)`` is within
    ``cents_tolerance`` and above the per-frame noise floor. The
    harmonic's bend stream reuses the parent F0's cents — measured peaks
    only gate presence.
    """
    harmonic_notes: list[NoteV3] = []
    harmonic_contours: list[ContourFrame] = []
    f0_segment = f0.segment_frames
    if not f0_segment:
        return harmonic_notes, harmonic_contours, next_track_id

    cents_by_frame = {row.frame_index: row.cents_from_pitch for row in f0.contour}

    cents_tolerance = float(params.harmonic.cents_tolerance)
    cents_window_log = cents_tolerance / _CENTS_PER_OCTAVE

    for n in range(
        int(params.harmonic.min_harmonic),
        int(params.harmonic.max_harmonic) + 1,
    ):
        partial_index = n - 1
        present: list[tuple[_RefinedFrame, float]] = []
        for frame in f0_segment:
            cqt_frame_idx = _align_cqt_frame(
                frame.time_offset_s, cqt_seconds_per_frame, n_cqt_frames
            )
            column = cqt_log[:, cqt_frame_idx]
            floor, _mad = _frame_noise_floor(column)
            target_log_freq = frame.log_frequency + math.log2(float(n))
            if target_log_freq > float(cqt_bin_log_freqs[-1]):
                continue
            lo_log = target_log_freq - cents_window_log
            hi_log = target_log_freq + cents_window_log
            window_mask = (cqt_bin_log_freqs >= lo_log) & (cqt_bin_log_freqs <= hi_log)
            window_bins = np.flatnonzero(window_mask)
            if window_bins.size == 0:
                continue
            window_magnitudes = column[window_bins]
            best_offset = int(np.argmax(window_magnitudes))
            bin_idx = int(window_bins[best_offset])
            magnitude = float(column[bin_idx])
            if not math.isfinite(magnitude):
                continue
            if magnitude - floor < 0.0:
                continue
            bin_log_freq = float(cqt_bin_log_freqs[bin_idx])
            cents_dev = abs(_CENTS_PER_OCTAVE * (bin_log_freq - target_log_freq))
            if cents_dev > cents_tolerance:
                continue
            present.append((frame, magnitude))

        if not present:
            continue

        runs = _split_runs(
            [pair[0] for pair in present],
            min_break_frames=int(params.harmonic.min_break_frames),
        )
        if not runs:
            continue
        magnitudes_by_frame_index = {pair[0].frame_index: pair[1] for pair in present}

        for run in runs:
            if len(run) < params.harmonic.min_note_frames:
                continue
            harmonic_midi_pitch = _harmonic_midi_pitch(
                run, partial_index=partial_index, params=params.midi
            )
            start_offset_s = float(run[0].time_offset_s) - pad_seconds
            end_offset_s = float(run[-1].time_offset_s) - pad_seconds
            duration_s = max(end_offset_s - start_offset_s, 0.0)
            start_utc = float(params.event_start_utc) + start_offset_s
            note_uid = _build_note_uid(
                job_id=params.job_id,
                event_id=params.event_id,
                partial_index=partial_index,
                track_id=next_track_id,
                start_utc=start_utc,
            )
            peak_magnitude = float(
                max(
                    (
                        magnitudes_by_frame_index.get(frame.frame_index, 0.0)
                        for frame in run
                    ),
                    default=0.0,
                )
            )
            contour_rows: list[ContourFrame] = []
            for i, frame in enumerate(run):
                base_cents = float(cents_by_frame.get(frame.frame_index, 0.0))
                cents = max(
                    -float(params.midi.cents_clip),
                    min(float(params.midi.cents_clip), base_cents),
                )
                contour_rows.append(
                    ContourFrame(
                        note_uid=note_uid,
                        frame_index=i,
                        time_offset_s=float(frame.time_offset_s)
                        - pad_seconds
                        - start_offset_s,
                        cents_from_pitch=cents,
                        harmonic_strength=float(
                            magnitudes_by_frame_index.get(frame.frame_index, 0.0)
                        ),
                        subharmonic_octave=int(frame.subharmonic_octave),
                    )
                )
            harmonic_notes.append(
                NoteV3(
                    note_uid=note_uid,
                    track_id=next_track_id,
                    f0_track_id=f0.note.track_id,
                    partial_index=partial_index,
                    midi_pitch=harmonic_midi_pitch,
                    start_utc=start_utc,
                    start_offset_s=start_offset_s,
                    duration_s=duration_s,
                    peak_magnitude=peak_magnitude,
                    contour_frame_count=len(contour_rows),
                )
            )
            harmonic_contours.extend(contour_rows)
            next_track_id += 1
    return harmonic_notes, harmonic_contours, next_track_id


def _harmonic_midi_pitch(
    run: list[_RefinedFrame],
    *,
    partial_index: int,
    params: MidiRangeParams,
) -> int:
    """MIDI pitch for a harmonic = F0 pitch + 12·log2(n) rounded.

    Computed from the run's median log_frequency rather than the F0
    median so glissandi at the harmonic stay tightly bound to the
    measured ridge. ``partial_index`` is one less than the integer
    multiple per spec §5.4 (H2 → ``partial_index = 1``).
    """
    log_freqs = np.asarray([f.log_frequency for f in run], dtype=np.float64)
    midi_continuous = (
        _MIDI_A4
        + 12.0 * (log_freqs - math.log2(_HZ_A4))
        + 12.0 * math.log2(float(partial_index + 1))
    )
    raw = int(round(float(np.median(midi_continuous))))
    return int(max(params.min_pitch, min(params.max_pitch, raw)))


def _split_runs(
    frames: list[_RefinedFrame],
    *,
    min_break_frames: int,
) -> list[list[_RefinedFrame]]:
    """Group frames into contiguous runs, splitting on gaps.

    The notion of "contiguous" here is over the F0 segment's frames, so
    a gap of one F0 frame is one missing harmonic frame even if the
    underlying STFT indices skip wider.
    """
    if not frames:
        return []
    frames_sorted = sorted(frames, key=lambda f: f.frame_index)
    runs: list[list[_RefinedFrame]] = []
    current = [frames_sorted[0]]
    last_idx = frames_sorted[0].frame_index
    for frame in frames_sorted[1:]:
        if frame.frame_index - last_idx >= min_break_frames + 1:
            runs.append(current)
            current = [frame]
        else:
            current.append(frame)
        last_idx = frame.frame_index
    runs.append(current)
    return runs


def _align_cqt_frame(
    time_offset_s: float, seconds_per_frame: float, n_frames: int
) -> int:
    if seconds_per_frame <= 0.0 or n_frames <= 0:
        return 0
    idx = int(round(float(time_offset_s) / float(seconds_per_frame)))
    return int(max(0, min(n_frames - 1, idx)))


def _nearest_bin(target_hz: float, bin_freqs: np.ndarray) -> int:
    if bin_freqs.size == 0:
        return 0
    log_target = math.log2(max(target_hz, 1e-9))
    log_bins = np.log2(bin_freqs)
    return int(np.argmin(np.abs(log_bins - log_target)))


def _frame_noise_floor(column: np.ndarray) -> tuple[float, float]:
    """Per-frame ``(noise_floor, MAD)`` over the bottom half of the column."""
    if column.size == 0:
        return 0.0, 0.0
    sorted_vals = np.sort(column)
    bottom = sorted_vals[: max(1, column.size // 2)]
    median = float(np.median(bottom))
    mad = float(np.median(np.abs(bottom - median)))
    return median, mad


def _majority_smooth(values: np.ndarray, *, window: int) -> np.ndarray:
    if window <= 1 or values.size == 0:
        return values.astype(np.int64)
    half = window // 2
    n = values.size
    out = np.empty(n, dtype=np.int64)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        chunk = values[lo:hi]
        counts: dict[int, int] = {}
        for v in chunk.tolist():
            counts[int(v)] = counts.get(int(v), 0) + 1
        out[i] = max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0]
    return out


def _build_note_uid(
    *,
    job_id: str,
    event_id: str,
    partial_index: int,
    track_id: int,
    start_utc: float,
) -> str:
    """Deterministic UUID v5 over (job, event, partial, track, t_ms)."""
    start_ms = int(round(float(start_utc) * 1000.0))
    payload = f"{job_id}|{event_id}|{int(partial_index)}|{int(track_id)}|{start_ms}"
    return str(uuid.uuid5(_NOTE_UID_NAMESPACE, payload))
