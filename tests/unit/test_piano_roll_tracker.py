"""Tests for the Piano Roll Notes tracker, harmonic prior, and quantizer."""

from __future__ import annotations

import math

import pytest

from humpback.processing.piano_roll_cqt import CQTParams, bin_frequency_hz
from humpback.processing.piano_roll_tracker import (
    HarmonicParams,
    MIDIQuantizeParams,
    Track,
    TrackerParams,
    build_tracks,
    label_harmonics,
    quantize_to_midi,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _peaks_constant_bin(bin_index: int, frames: int, magnitude: float = 1.0):
    return [[(bin_index, magnitude)] for _ in range(frames)]


def _frames_for_seconds(seconds: float, params: CQTParams) -> int:
    return int(math.ceil(seconds * params.target_sample_rate / params.hop_length))


def _constant_track(
    *,
    track_id: int,
    bin_idx: float,
    n_frames: int,
    start_frame: int = 0,
    magnitude: float = 1.0,
) -> Track:
    """Construct a Track that sits at one bin for ``n_frames`` consecutive frames."""
    return Track(
        track_id=track_id,
        start_frame=start_frame,
        end_frame=start_frame + n_frames - 1,
        bins=[float(bin_idx)] * n_frames,
        log_magnitudes=[magnitude] * n_frames,
        frames=list(range(start_frame, start_frame + n_frames)),
    )


def _per_frame_track(
    *,
    track_id: int,
    start_frame: int,
    bin_per_frame: list[float],
    magnitude: float = 1.0,
) -> Track:
    """Construct a Track from an explicit per-frame bin list."""
    n = len(bin_per_frame)
    return Track(
        track_id=track_id,
        start_frame=start_frame,
        end_frame=start_frame + n - 1,
        bins=[float(b) for b in bin_per_frame],
        log_magnitudes=[magnitude] * n,
        frames=list(range(start_frame, start_frame + n)),
    )


def _bin_for_ratio(anchor_bin: float, ratio: float, params: CQTParams) -> float:
    """Inverse of ``bin_frequency_hz`` for synthetic fixtures."""
    return anchor_bin + params.bins_per_octave * math.log2(ratio)


# --------------------------------------------------------------------------- #
# build_tracks
# --------------------------------------------------------------------------- #


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
    # build_tracks populates the per-frame indices list.
    assert tracks[0].frames == list(range(frames))


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
    # The missed frames are absent from the per-frame index list.
    assert tracks[0].frames == list(range(6)) + list(range(8, 14))


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


# --------------------------------------------------------------------------- #
# label_harmonics — basic ratios and defaults
# --------------------------------------------------------------------------- #


def test_label_harmonics_exact_ratios() -> None:
    params = CQTParams()
    # F0 at bin 108 (A3=220 Hz). 2x = 440 Hz = bin 144. 3x = 660 Hz ≈ bin 165.
    tracks = [
        _constant_track(track_id=0, bin_idx=108, n_frames=11),
        _constant_track(track_id=1, bin_idx=144, n_frames=11),
        _constant_track(track_id=2, bin_idx=165, n_frames=11),
    ]
    label_harmonics(tracks, cqt_params=params)
    partial_by_id = {t.track_id: t.partial_index for t in tracks}
    assert partial_by_id == {0: 0, 1: 1, 2: 2}


def test_label_harmonics_near_miss_within_75_cents() -> None:
    params = CQTParams()
    # F0 = 220 Hz, candidate at exactly 2x ratio of 2.04 (~34 cents off
    # the 2x integer — inside the new 75¢ tolerance).
    tracks = [
        _constant_track(track_id=0, bin_idx=108, n_frames=11),
        _constant_track(
            track_id=1, bin_idx=_bin_for_ratio(108.0, 2.04, params), n_frames=11
        ),
    ]
    label_harmonics(tracks, cqt_params=params)
    assert tracks[0].partial_index == 0
    assert tracks[1].partial_index == 1


def test_label_harmonics_tolerance_boundary_inside_new_outside_old() -> None:
    """A 70¢-off candidate is accepted under v2 but would have failed under v1.

    Locks the policy change ``cents_tolerance: 50 → 75`` against regression.
    """
    params = CQTParams()
    # 2x F0 with a 70-cent positive offset on the harmonic.
    deviation_cents = 70.0
    ratio = 2.0 * 2.0 ** (deviation_cents / 1200.0)
    tracks = [
        _constant_track(track_id=0, bin_idx=108, n_frames=11),
        _constant_track(
            track_id=1, bin_idx=_bin_for_ratio(108.0, ratio, params), n_frames=11
        ),
    ]
    label_harmonics(tracks, cqt_params=params)
    assert tracks[1].partial_index == 1

    # Same input, but explicitly held to the historical 50¢ tolerance — the
    # harmonic check fails so the candidate is left unprocessed and ends up
    # anchoring its own (singleton) cluster as F0 rather than being labeled
    # as the 2nd harmonic.
    tight_tracks = [
        _constant_track(track_id=0, bin_idx=108, n_frames=11),
        _constant_track(
            track_id=1, bin_idx=_bin_for_ratio(108.0, ratio, params), n_frames=11
        ),
    ]
    label_harmonics(
        tight_tracks,
        cqt_params=params,
        params=HarmonicParams(cents_tolerance=50.0, min_overlap_frames=3),
    )
    assert tight_tracks[0].partial_index == 0
    assert tight_tracks[1].partial_index != 1
    assert tight_tracks[1].partial_index == 0


def test_label_harmonics_far_off_rejected_at_default_tolerance() -> None:
    """≈84¢ off the nearest integer fails the default 75¢ tolerance.

    Under v2's leave-unprocessed semantic the rejected candidate is not
    labeled as the 2nd harmonic; it instead anchors its own cluster as
    ``partial_index = 0``.
    """
    params = CQTParams()
    deviation_cents = 84.0
    ratio = 2.0 * 2.0 ** (deviation_cents / 1200.0)
    tracks = [
        _constant_track(track_id=0, bin_idx=108, n_frames=11),
        _constant_track(
            track_id=1, bin_idx=_bin_for_ratio(108.0, ratio, params), n_frames=11
        ),
    ]
    label_harmonics(tracks, cqt_params=params)
    assert tracks[0].partial_index == 0
    assert tracks[1].partial_index != 1
    assert tracks[1].partial_index == 0


def test_label_harmonics_out_of_range_keeps_minus_one() -> None:
    """A non-integer-multiple overlap leaves the candidate unlabeled."""
    params = CQTParams()
    # F0 = 108, then a track at bin 130 — that's between 2x and 3x ratio,
    # not close to an integer multiple → no harmonic label.
    tracks = [
        _constant_track(track_id=0, bin_idx=108, n_frames=11),
        _constant_track(track_id=1, bin_idx=130, n_frames=11),
    ]
    label_harmonics(tracks, cqt_params=params)
    assert tracks[0].partial_index == 0
    # Bin 130 is between F0 (108) and 2x F0 (144), so it sorts after F0.
    # Its ratio (~1.5) rounds to 2 → cents deviation ≈ 500¢ → rejected.
    # Since v2 no longer "consumes" non-matching tracks, this track is left
    # available and ends up anchoring its own (singleton) cluster as partial_index=0.
    assert tracks[1].partial_index == 0


def test_label_harmonics_disabled_keeps_minus_one() -> None:
    params = CQTParams()
    tracks = [
        _constant_track(track_id=0, bin_idx=108, n_frames=11),
        _constant_track(track_id=1, bin_idx=144, n_frames=11),
    ]
    label_harmonics(tracks, cqt_params=params, params=HarmonicParams(enabled=False))
    assert all(t.partial_index == -1 for t in tracks)


def test_label_harmonics_does_not_drop_tracks() -> None:
    params = CQTParams()
    tracks = [
        _constant_track(track_id=0, bin_idx=108, n_frames=11),
        _constant_track(track_id=1, bin_idx=130, n_frames=11),
    ]
    returned = label_harmonics(tracks, cqt_params=params)
    assert returned is tracks
    assert len(returned) == 2


def test_label_harmonics_deterministic() -> None:
    """Repeated calls on equivalent inputs produce identical labels."""
    params = CQTParams()

    def fresh() -> list[Track]:
        return [
            _constant_track(track_id=0, bin_idx=108, n_frames=11),
            _constant_track(track_id=1, bin_idx=144, n_frames=11),
            _constant_track(track_id=2, bin_idx=165, n_frames=11),
        ]

    first = fresh()
    label_harmonics(first, cqt_params=params)
    second = fresh()
    label_harmonics(second, cqt_params=params)
    assert [t.partial_index for t in first] == [t.partial_index for t in second]


# --------------------------------------------------------------------------- #
# label_harmonics — v2-specific behavior
# --------------------------------------------------------------------------- #


def test_label_harmonics_f0_sort_fix_lowest_bin_wins() -> None:
    """The lowest-bin track anchors F0 even when a higher-bin track starts first.

    Under v1 the higher-bin earlier-starting track would have been promoted
    to F0 (sort key was ``(start_frame, median_bin)``). Under v2 the
    lower-bin track is F0 and the higher-bin track is its 2nd harmonic.
    """
    params = CQTParams()
    # Higher-bin track starts at frame 0; lower-bin "true F0" starts at
    # frame 5 holding a 0.5x ratio across their shared frames.
    higher = _constant_track(track_id=0, bin_idx=144, n_frames=20, start_frame=0)
    lower = _constant_track(track_id=1, bin_idx=108, n_frames=15, start_frame=5)
    label_harmonics([higher, lower], cqt_params=params)
    assert lower.partial_index == 0  # F0
    assert higher.partial_index == 1  # 2nd harmonic of lower


def test_label_harmonics_non_match_left_available_as_anchor() -> None:
    """A non-harmonic overlap leaves the track unprocessed.

    Under v1 the track would have been consumed by the F0 candidate it
    overlapped and stayed at ``partial_index = -1`` forever. Under v2 it
    is left available and ends up anchoring its own (singleton) cluster.
    """
    params = CQTParams()
    f0 = _constant_track(track_id=0, bin_idx=108, n_frames=20)
    # Bin 130 ≈ ratio 1.5 to F0 — not an integer multiple. Time-overlaps F0.
    non_harmonic = _constant_track(track_id=1, bin_idx=130, n_frames=20)
    label_harmonics([f0, non_harmonic], cqt_params=params)
    assert f0.partial_index == 0
    # The non-matching track now becomes its own cluster anchor.
    assert non_harmonic.partial_index == 0


def test_label_harmonics_per_frame_ratio_handles_sweep() -> None:
    """Two tracks whose medians do not align cleanly but per-frame ratios do.

    F0 has a long, three-step lifetime: spends frames 0–5 at bin 100,
    frames 6–11 at bin 110, frames 12–20 at bin 120 (median bin = 110).
    The harmonic candidate is alive only during the high-bin tail at
    frames 15–20 with bin 156, which is exactly 2× F0's frequency at
    those shared frames. Across the harmonic's lifetime the medians do
    *not* yield a 2× ratio (`med(156)/med(110) ≈ 2.42`), but the per-frame
    shared-frame ratios are exactly 2.0 each. v2 should accept; v1 with
    median-bin ratios would have rejected.
    """
    params = CQTParams()
    f0_bins = [100.0] * 6 + [110.0] * 6 + [120.0] * 9  # 21 frames, median 110
    harmonic_bins = [156.0] * 6  # 6 frames at exact 2× of 120
    f0 = _per_frame_track(track_id=0, start_frame=0, bin_per_frame=f0_bins)
    harmonic = _per_frame_track(track_id=1, start_frame=15, bin_per_frame=harmonic_bins)
    # The fixture really must exhibit the median-bin mismatch that v1
    # would have tripped on — otherwise this test is vacuous.
    median_ratio = bin_frequency_hz(harmonic.median_bin, params) / bin_frequency_hz(
        f0.median_bin, params
    )
    assert abs(median_ratio - 2.0) > 0.05, (
        f"fixture's median-bin ratio {median_ratio:.4f} would let v1 succeed"
    )
    label_harmonics([f0, harmonic], cqt_params=params)
    assert f0.partial_index == 0
    assert harmonic.partial_index == 1


def test_label_harmonics_min_overlap_frames_rejects_brief_overlap() -> None:
    """A clean 2x candidate that overlaps F0 for only two frames is rejected."""
    params = CQTParams()
    f0 = _constant_track(track_id=0, bin_idx=108, n_frames=20, start_frame=0)
    # Harmonic at exact 2x but only present at frames 0–1 → 2-frame overlap.
    harmonic = _constant_track(track_id=1, bin_idx=144, n_frames=2, start_frame=0)
    label_harmonics(
        [f0, harmonic],
        cqt_params=params,
        params=HarmonicParams(min_overlap_frames=3),
    )
    assert f0.partial_index == 0
    # Below the 3-frame floor, the candidate is left unprocessed and
    # eventually anchors its own (singleton) cluster.
    assert harmonic.partial_index == 0


def test_label_harmonics_max_harmonic_16_reaches_high_harmonics() -> None:
    """A 10× harmonic gets ``partial_index = 9`` under the v2 default cap."""
    params = CQTParams()
    f0 = _constant_track(track_id=0, bin_idx=18, n_frames=11)  # low F0
    tenth = _constant_track(
        track_id=1, bin_idx=_bin_for_ratio(18.0, 10.0, params), n_frames=11
    )
    label_harmonics([f0, tenth], cqt_params=params)
    assert f0.partial_index == 0
    assert tenth.partial_index == 9


def test_label_harmonics_max_harmonic_16_excludes_17th() -> None:
    """Past the cap, even an exact integer multiple is left unlabeled."""
    params = CQTParams()
    f0 = _constant_track(track_id=0, bin_idx=18, n_frames=11)
    too_high = _constant_track(
        track_id=1, bin_idx=_bin_for_ratio(18.0, 17.0, params), n_frames=11
    )
    label_harmonics([f0, too_high], cqt_params=params)
    assert f0.partial_index == 0
    # Past the cap, the harmonic check fails. v2 does not consume the
    # track, so it becomes its own F0 anchor (partial_index = 0). The
    # important assertion is that it is NOT labeled as harmonic 16.
    assert too_high.partial_index != 15


# --------------------------------------------------------------------------- #
# quantize_to_midi
# --------------------------------------------------------------------------- #


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
