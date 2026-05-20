"""Tests for the Piano Roll Notes tracker, harmonic prior, and quantizer."""

from __future__ import annotations

import math

import pytest

from humpback.processing.piano_roll_cqt import CQTParams
from humpback.processing.piano_roll_tracker import (
    HarmonicParams,
    MIDIQuantizeParams,
    Track,
    TrackerParams,
    build_tracks,
    label_harmonics,
    quantize_to_midi,
)


def _peaks_constant_bin(bin_index: int, frames: int, magnitude: float = 1.0):
    return [[(bin_index, magnitude)] for _ in range(frames)]


def _frames_for_seconds(seconds: float, params: CQTParams) -> int:
    return int(math.ceil(seconds * params.target_sample_rate / params.hop_length))


def test_build_tracks_short_track_dropped() -> None:
    params = CQTParams()
    tracker = TrackerParams(min_duration_s=0.05)
    # 2 frames = ~23 ms at default params → below 50 ms floor.
    short = _peaks_constant_bin(144, frames=2)
    assert build_tracks(short, cqt_params=params, params=tracker) == []


def test_build_tracks_long_track_survives() -> None:
    params = CQTParams()
    tracker = TrackerParams(min_duration_s=0.05)
    frames = _frames_for_seconds(0.10, params)
    peaks = _peaks_constant_bin(144, frames=frames)
    tracks = build_tracks(peaks, cqt_params=params, params=tracker)
    assert len(tracks) == 1
    assert tracks[0].start_frame == 0
    assert tracks[0].end_frame == frames - 1


def test_build_tracks_small_gap_keeps_one_track() -> None:
    params = CQTParams()
    tracker = TrackerParams(min_duration_s=0.05, miss_tolerance_frames=2)
    # Pattern: 6 frames present, 2 missed, 6 frames present → ~70 ms each.
    peaks = (
        _peaks_constant_bin(144, frames=6)
        + [[] for _ in range(2)]
        + _peaks_constant_bin(144, frames=6)
    )
    tracks = build_tracks(peaks, cqt_params=params, params=tracker)
    assert len(tracks) == 1
    assert tracks[0].start_frame == 0
    assert tracks[0].end_frame == 13


def test_build_tracks_large_gap_splits_track() -> None:
    params = CQTParams()
    tracker = TrackerParams(min_duration_s=0.05, miss_tolerance_frames=2)
    long_a = _frames_for_seconds(0.08, params)
    long_b = _frames_for_seconds(0.08, params)
    peaks = (
        _peaks_constant_bin(144, frames=long_a)
        + [[] for _ in range(4)]
        + _peaks_constant_bin(144, frames=long_b)
    )
    tracks = build_tracks(peaks, cqt_params=params, params=tracker)
    assert len(tracks) == 2


def test_label_harmonics_exact_ratios() -> None:
    params = CQTParams()
    # F0 at bin 108 (A3=220 Hz). 2x = 440 Hz = bin 144. 3x = 660 Hz ≈ bin 165.
    tracks = [
        Track(
            track_id=0, start_frame=0, end_frame=10, bins=[108.0], log_magnitudes=[1.0]
        ),
        Track(
            track_id=1, start_frame=0, end_frame=10, bins=[144.0], log_magnitudes=[1.0]
        ),
        Track(
            track_id=2, start_frame=0, end_frame=10, bins=[165.0], log_magnitudes=[1.0]
        ),
    ]
    label_harmonics(tracks, cqt_params=params)
    partial_by_id = {t.track_id: t.partial_index for t in tracks}
    assert partial_by_id == {0: 0, 1: 1, 2: 2}


def test_label_harmonics_near_miss_within_50_cents() -> None:
    params = CQTParams()
    # F0 = bin 108 (220 Hz). 2x at bin 144 = 440 Hz, exact. Try a slight
    # offset that should still be within ±50 cents (50 cents ≈ 1.5 bins
    # at 3 bins/semitone, so ±1 bin is well inside tolerance).
    tracks = [
        Track(
            track_id=0, start_frame=0, end_frame=10, bins=[108.0], log_magnitudes=[1.0]
        ),
        Track(
            track_id=1, start_frame=0, end_frame=10, bins=[145.0], log_magnitudes=[1.0]
        ),
    ]
    label_harmonics(tracks, cqt_params=params)
    assert tracks[0].partial_index == 0
    assert tracks[1].partial_index == 1


def test_label_harmonics_out_of_range_keeps_minus_one() -> None:
    params = CQTParams()
    # F0 = 108, then a track at bin 130 — that's between 2x and 3x ratio,
    # not close to an integer multiple → no harmonic label.
    tracks = [
        Track(
            track_id=0, start_frame=0, end_frame=10, bins=[108.0], log_magnitudes=[1.0]
        ),
        Track(
            track_id=1, start_frame=0, end_frame=10, bins=[130.0], log_magnitudes=[1.0]
        ),
    ]
    label_harmonics(tracks, cqt_params=params)
    assert tracks[0].partial_index == 0
    assert tracks[1].partial_index == -1


def test_label_harmonics_disabled_keeps_minus_one() -> None:
    params = CQTParams()
    tracks = [
        Track(
            track_id=0, start_frame=0, end_frame=10, bins=[108.0], log_magnitudes=[1.0]
        ),
        Track(
            track_id=1, start_frame=0, end_frame=10, bins=[144.0], log_magnitudes=[1.0]
        ),
    ]
    label_harmonics(tracks, cqt_params=params, params=HarmonicParams(enabled=False))
    assert all(t.partial_index == -1 for t in tracks)


def test_label_harmonics_does_not_drop_tracks() -> None:
    params = CQTParams()
    tracks = [
        Track(
            track_id=0, start_frame=0, end_frame=10, bins=[108.0], log_magnitudes=[1.0]
        ),
        Track(
            track_id=1, start_frame=0, end_frame=10, bins=[130.0], log_magnitudes=[1.0]
        ),
    ]
    returned = label_harmonics(tracks, cqt_params=params)
    assert returned is tracks
    assert len(returned) == 2


def test_quantize_to_midi_a4() -> None:
    params = CQTParams()
    track = Track(
        track_id=7,
        start_frame=10,
        end_frame=20,
        bins=[144.0],
        log_magnitudes=[2.5],
        partial_index=0,
    )
    note = quantize_to_midi(track, cqt_params=params)
    assert note is not None
    assert note.midi_pitch == 69
    assert note.track_id == 7
    assert note.partial_index == 0
    assert note.peak_magnitude == pytest.approx(2.5)
    frame_seconds = params.hop_length / params.target_sample_rate
    assert note.start_offset_s == pytest.approx(10 * frame_seconds)
    assert note.duration_s == pytest.approx(11 * frame_seconds)


def test_quantize_to_midi_clamps_near_boundary() -> None:
    params = CQTParams()
    midi_params = MIDIQuantizeParams(min_pitch=21, max_pitch=108)
    # bin 0 → MIDI 21 (A0) — should pass through unchanged.
    track = Track(
        track_id=0,
        start_frame=0,
        end_frame=10,
        bins=[0.0],
        log_magnitudes=[1.0],
    )
    note = quantize_to_midi(track, cqt_params=params, midi_params=midi_params)
    assert note is not None
    assert note.midi_pitch == 21


def test_quantize_to_midi_drops_far_out_of_range() -> None:
    params = CQTParams()
    midi_params = MIDIQuantizeParams(min_pitch=60, max_pitch=72)
    track = Track(
        track_id=0,
        start_frame=0,
        end_frame=10,
        bins=[0.0],  # MIDI 21, well below min_pitch
        log_magnitudes=[1.0],
    )
    note = quantize_to_midi(track, cqt_params=params, midi_params=midi_params)
    assert note is None
