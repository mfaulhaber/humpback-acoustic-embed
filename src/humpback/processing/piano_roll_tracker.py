"""Greedy cross-frame partial tracker, harmonic prior, MIDI quantizer.

Pure functions and small dataclasses. Consumes the output of
``pick_peaks_per_frame`` from ``piano_roll_cqt`` and emits Track objects
that the worker quantizes into MIDI note records.

See ``docs/specs/2026-05-20-piano-roll-midi-notes-design.md`` §6.4–§6.6.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from humpback.processing.piano_roll_cqt import (
    CQTParams,
    bin_frequency_hz,
    midi_pitch_for_bin,
)

__all__ = [
    "TrackerParams",
    "HarmonicParams",
    "MIDIQuantizeParams",
    "Track",
    "MidiNote",
    "build_tracks",
    "label_harmonics",
    "quantize_to_midi",
]


@dataclass(frozen=True, slots=True)
class TrackerParams:
    """Cross-frame tracker tolerances and survival floor."""

    bin_tolerance: int = 3  # ±1 semitone at 3 bins/semitone
    miss_tolerance_frames: int = 2
    min_duration_s: float = 0.05
    amplitude_floor_percentile: float = 5.0


@dataclass(frozen=True, slots=True)
class HarmonicParams:
    """Harmonic-prior labeling pass (does not filter tracks)."""

    enabled: bool = True
    max_harmonic: int = 8
    cents_tolerance: float = 50.0


@dataclass(frozen=True, slots=True)
class MIDIQuantizeParams:
    """MIDI pitch clamping."""

    min_pitch: int = 21
    max_pitch: int = 108


@dataclass(slots=True)
class Track:
    """One spectral peak followed across frames."""

    track_id: int
    start_frame: int
    end_frame: int
    bins: list[float] = field(default_factory=list)
    log_magnitudes: list[float] = field(default_factory=list)
    partial_index: int = -1

    @property
    def median_bin(self) -> float:
        return float(np.median(self.bins)) if self.bins else 0.0

    @property
    def median_log_magnitude(self) -> float:
        return float(np.median(self.log_magnitudes)) if self.log_magnitudes else 0.0


@dataclass(frozen=True, slots=True)
class MidiNote:
    """One note as it lands in the parquet sidecar (pre-velocity)."""

    track_id: int
    midi_pitch: int
    start_offset_s: float
    duration_s: float
    peak_magnitude: float
    partial_index: int


def build_tracks(
    per_frame_peaks: list[list[tuple[int, float]]],
    *,
    cqt_params: CQTParams,
    params: TrackerParams = TrackerParams(),
) -> list[Track]:
    """Greedy nearest-neighbor partial tracker.

    Tracks open when a peak has no match in the previous frame, extend
    when a peak lies within ``bin_tolerance`` of an open track's last
    bin, and close after ``miss_tolerance_frames`` consecutive frames
    without an extension. After the sweep, tracks shorter than
    ``min_duration_s`` (in seconds, given the CQT hop) are dropped, then
    a per-event amplitude floor at ``amplitude_floor_percentile`` is
    applied to the surviving tracks' median log-magnitudes.
    """

    @dataclass(slots=True)
    class _Open:
        track: Track
        last_bin: int
        misses: int

    if not per_frame_peaks:
        return []

    sr = cqt_params.target_sample_rate
    frame_seconds = cqt_params.hop_length / sr
    min_frames = max(1, int(round(params.min_duration_s / frame_seconds)))

    open_tracks: list[_Open] = []
    finished: list[Track] = []
    next_track_id = 0

    for t, peaks in enumerate(per_frame_peaks):
        matched_track_indices: set[int] = set()
        matched_peak_indices: set[int] = set()

        # Match peaks to open tracks: strongest peaks first, each track
        # accepts at most one extension per frame.
        peaks_by_strength = sorted(
            enumerate(peaks), key=lambda kv: -kv[1][1] if kv[1] else 0.0
        )
        for peak_idx, (bin_idx, magnitude) in peaks_by_strength:
            best_track_index = -1
            best_distance = params.bin_tolerance + 1
            for track_index, entry in enumerate(open_tracks):
                if track_index in matched_track_indices:
                    continue
                distance = abs(entry.last_bin - bin_idx)
                if distance <= params.bin_tolerance and distance < best_distance:
                    best_track_index = track_index
                    best_distance = distance
            if best_track_index >= 0:
                entry = open_tracks[best_track_index]
                entry.track.bins.append(float(bin_idx))
                entry.track.log_magnitudes.append(float(magnitude))
                entry.track.end_frame = t
                entry.last_bin = bin_idx
                entry.misses = 0
                matched_track_indices.add(best_track_index)
                matched_peak_indices.add(peak_idx)

        # Increment miss counters for unmatched tracks; close any past
        # the tolerance.
        survivors: list[_Open] = []
        for track_index, entry in enumerate(open_tracks):
            if track_index in matched_track_indices:
                survivors.append(entry)
                continue
            entry.misses += 1
            if entry.misses > params.miss_tolerance_frames:
                finished.append(entry.track)
            else:
                survivors.append(entry)
        open_tracks = survivors

        # Unmatched peaks open new tracks.
        for peak_idx, (bin_idx, magnitude) in enumerate(peaks):
            if peak_idx in matched_peak_indices:
                continue
            new_track = Track(
                track_id=next_track_id,
                start_frame=t,
                end_frame=t,
                bins=[float(bin_idx)],
                log_magnitudes=[float(magnitude)],
            )
            next_track_id += 1
            open_tracks.append(_Open(track=new_track, last_bin=bin_idx, misses=0))

    for entry in open_tracks:
        finished.append(entry.track)

    long_enough = [
        track
        for track in finished
        if (track.end_frame - track.start_frame + 1) >= min_frames
    ]

    if not long_enough:
        return []

    magnitudes = np.asarray([track.median_log_magnitude for track in long_enough])
    floor = float(np.percentile(magnitudes, params.amplitude_floor_percentile))
    return [
        track
        for track in long_enough
        if track.median_log_magnitude >= floor or len(long_enough) == 1
    ]


def label_harmonics(
    tracks: list[Track],
    *,
    cqt_params: CQTParams,
    params: HarmonicParams = HarmonicParams(),
) -> list[Track]:
    """Set ``track.partial_index`` per the harmonic-prior rules.

    The lowest-bin track in each overlapping cluster is treated as the F0
    candidate. Other tracks that overlap in time AND whose median
    frequency is within ``cents_tolerance`` cents of an integer multiple
    (``2×``…``max_harmonic×``) of the F0 candidate's median frequency
    receive ``partial_index = harmonic_number - 1``. Tracks not matched
    keep ``partial_index = -1``. The harmonic prior never drops a track.

    When ``params.enabled`` is False this is a no-op (all tracks keep
    ``partial_index = -1``). Returns the same list for fluent chaining.
    """
    if not params.enabled or not tracks:
        return tracks

    sorted_tracks = sorted(tracks, key=lambda t: (t.start_frame, t.median_bin))
    used: set[int] = set()
    for i, candidate in enumerate(sorted_tracks):
        if id(candidate) in used:
            continue
        f0_hz = bin_frequency_hz(candidate.median_bin, cqt_params)
        if f0_hz <= 0.0:
            continue
        candidate.partial_index = 0
        used.add(id(candidate))
        # Sweep all later tracks that share time with this F0. Tracks in
        # the cluster get a harmonic label when their median frequency
        # falls within tolerance of an integer multiple; otherwise they
        # keep partial_index = -1. Either way they are considered
        # "processed" and won't be promoted to F0 on a later iteration.
        for other in sorted_tracks[i + 1 :]:
            if id(other) in used:
                continue
            if not _overlaps(candidate, other):
                continue
            used.add(id(other))
            other_hz = bin_frequency_hz(other.median_bin, cqt_params)
            if other_hz <= 0.0:
                continue
            ratio = other_hz / f0_hz
            harmonic = int(round(ratio))
            if harmonic < 2 or harmonic > params.max_harmonic:
                continue
            cents = 1200.0 * abs(np.log2(other_hz / (harmonic * f0_hz)))
            if cents <= params.cents_tolerance:
                other.partial_index = harmonic - 1
    return tracks


def quantize_to_midi(
    track: Track,
    *,
    cqt_params: CQTParams,
    midi_params: MIDIQuantizeParams = MIDIQuantizeParams(),
) -> Optional[MidiNote]:
    """Convert a track to a ``MidiNote``.

    Returns ``None`` if the quantized pitch falls outside
    ``[min_pitch, max_pitch]`` after clamping is rejected — i.e., the raw
    bin sits more than one semitone outside the configured MIDI range. A
    bin that rounds to within ±1 semitone of the boundary is clamped
    silently.
    """
    sr = cqt_params.target_sample_rate
    frame_seconds = cqt_params.hop_length / sr
    raw_midi = midi_pitch_for_bin(track.median_bin, cqt_params)
    if raw_midi < midi_params.min_pitch - 1 or raw_midi > midi_params.max_pitch + 1:
        return None
    midi_pitch = max(midi_params.min_pitch, min(midi_params.max_pitch, raw_midi))
    duration_s = (track.end_frame - track.start_frame + 1) * frame_seconds
    return MidiNote(
        track_id=track.track_id,
        midi_pitch=midi_pitch,
        start_offset_s=track.start_frame * frame_seconds,
        duration_s=duration_s,
        peak_magnitude=track.median_log_magnitude,
        partial_index=track.partial_index,
    )


def _overlaps(a: Track, b: Track) -> bool:
    return not (a.end_frame < b.start_frame or b.end_frame < a.start_frame)
