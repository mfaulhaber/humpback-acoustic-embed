"""Greedy cross-frame partial tracker, harmonic prior, MIDI quantizer.

Pure functions and small dataclasses. Consumes the output of
``pick_peaks_per_frame`` from ``piano_roll_cqt`` and emits Track objects
that the worker quantizes into MIDI note records.

The harmonic prior labels tracks with their integer-multiple relationship
to the lowest-bin track in each overlap cluster. See
``docs/specs/2026-05-20-event-encoder-midi-channelized-design.md`` §5 for
the per-frame ratio algorithm. The original v1 algorithm is documented in
``docs/specs/2026-05-20-piano-roll-midi-notes-design.md`` §6.4–§6.6.
"""

from __future__ import annotations

import math
import statistics
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
    max_harmonic: int = 16
    cents_tolerance: float = 75.0
    min_overlap_frames: int = 3


@dataclass(frozen=True, slots=True)
class MIDIQuantizeParams:
    """MIDI pitch clamping."""

    min_pitch: int = 21
    max_pitch: int = 108


@dataclass(slots=True)
class Track:
    """One spectral peak followed across frames.

    ``bins``, ``log_magnitudes``, and ``frames`` are kept in lock-step:
    one entry per frame in which the track was extended. The ``frames``
    list records the absolute frame indices of those extensions so the
    per-frame harmonic labeler can align two tracks even when one of
    them has missed frames in the middle of its lifetime.
    """

    track_id: int
    start_frame: int
    end_frame: int
    bins: list[float] = field(default_factory=list)
    log_magnitudes: list[float] = field(default_factory=list)
    frames: list[int] = field(default_factory=list)
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
                entry.track.frames.append(t)
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
                frames=[t],
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
    """Set ``track.partial_index`` using per-frame harmonic ratios.

    Iterates ``tracks`` by ``median_bin`` ascending. Each unprocessed
    track becomes the F0 anchor of its cluster
    (``partial_index = 0``). For every other unprocessed track that
    overlaps the anchor for at least ``params.min_overlap_frames``
    frames, per-frame frequency ratios are computed at every shared
    frame. The track is labeled as the Nth harmonic when:

    - The median of the per-frame ``round(ratio)`` values
      (``median_harmonic``) lies in ``[2, params.max_harmonic]``, AND
    - The median absolute cents deviation against ``median_harmonic``
      across overlapping frames is ≤ ``params.cents_tolerance``.

    Tracks that fail the check are **left unprocessed** so they remain
    eligible to anchor their own clusters on later iterations. Tracks
    never matched as either anchor or harmonic keep
    ``partial_index = -1``. The harmonic prior never drops a track.

    When ``params.enabled`` is False this is a no-op (all tracks keep
    ``partial_index = -1``). Returns the same list for fluent chaining.
    """
    if not params.enabled or not tracks:
        return tracks

    # Sort lowest-bin first so F0 anchors are selected by frequency rank
    # rather than start-time. Break ties by track_id for determinism.
    sorted_tracks = sorted(tracks, key=lambda t: (t.median_bin, t.track_id))
    processed: set[int] = set()

    for candidate in sorted_tracks:
        if candidate.track_id in processed:
            continue
        if bin_frequency_hz(candidate.median_bin, cqt_params) <= 0.0:
            continue
        candidate.partial_index = 0
        processed.add(candidate.track_id)

        for other in sorted_tracks:
            if other.track_id in processed:
                continue
            ratios = _per_frame_ratios(candidate, other, cqt_params)
            if len(ratios) < params.min_overlap_frames:
                continue
            per_frame_harmonics = [round(r) for r in ratios]
            median_harmonic = statistics.median_low(per_frame_harmonics)
            if not (2 <= median_harmonic <= params.max_harmonic):
                continue
            cents_per_frame = [
                abs(1200.0 * math.log2(r / median_harmonic)) for r in ratios
            ]
            if statistics.median(cents_per_frame) > params.cents_tolerance:
                continue
            other.partial_index = median_harmonic - 1
            processed.add(other.track_id)

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


def _per_frame_ratios(
    anchor: Track, other: Track, cqt_params: CQTParams
) -> list[float]:
    """Return ``freq(other) / freq(anchor)`` at every shared frame index.

    Only frames present in both tracks' ``frames`` lists contribute. A
    legacy/test Track with empty ``frames`` is synthesized as
    contiguous from ``start_frame`` so existing fixtures keep working.
    """
    anchor_by_frame = _frame_to_bin(anchor)
    other_by_frame = _frame_to_bin(other)
    shared = anchor_by_frame.keys() & other_by_frame.keys()
    ratios: list[float] = []
    for frame in shared:
        anchor_hz = bin_frequency_hz(anchor_by_frame[frame], cqt_params)
        if anchor_hz <= 0.0:
            continue
        other_hz = bin_frequency_hz(other_by_frame[frame], cqt_params)
        ratios.append(other_hz / anchor_hz)
    return ratios


def _frame_to_bin(track: Track) -> dict[int, float]:
    """Map absolute frame index → bin for a track.

    When ``track.frames`` is empty (typical of hand-built test fixtures
    that pre-date the per-frame labeler), assume the track was extended
    contiguously starting at ``start_frame``. When ``frames`` is
    populated by ``build_tracks`` it is used verbatim.
    """
    if track.frames:
        return dict(zip(track.frames, track.bins))
    if not track.bins:
        return {}
    return {track.start_frame + i: bin_idx for i, bin_idx in enumerate(track.bins)}
