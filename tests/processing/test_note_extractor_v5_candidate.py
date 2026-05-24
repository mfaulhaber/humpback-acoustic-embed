"""Tests for ``humpback.processing.note_extractor_v5_candidate``.

Exercises the harmonic-Viterbi F0 candidate against the same synthetic
fixtures used for v4 plus a "no harmonic relationship between competing
tones" case that the per-frame independent decoder in v4 cannot
disambiguate without smoothing.

The candidate algorithm may be replaced in Phase 2; tests assert
fixture-level behaviour (F0 location, voicing, no spurious flapping)
rather than internal algorithm state so they remain meaningful across
parameter or algorithm revisions.
"""

from __future__ import annotations

import math

import numpy as np

from humpback.processing.note_extractor_v3 import STFTParams
from humpback.processing.note_extractor_v5_candidate import (
    ExtractNotesV5Params,
    HarmonicViterbiParams,
    extract_notes_v5_candidate,
)
from humpback.processing.piano_roll_cqt import CQTParams


SAMPLE_RATE = 22050


def _params(
    *,
    harmonic_viterbi: HarmonicViterbiParams | None = None,
    stft: STFTParams | None = None,
    cqt: CQTParams | None = None,
) -> ExtractNotesV5Params:
    return ExtractNotesV5Params(
        job_id="job-v5-test",
        event_id="ev-1",
        event_start_utc=1000.0,
        pad_seconds=0.0,
        cqt=cqt if cqt is not None else CQTParams(),
        stft=stft if stft is not None else STFTParams(min_frequency_hz=30.0),
        harmonic_viterbi=harmonic_viterbi
        if harmonic_viterbi is not None
        else HarmonicViterbiParams(),
    )


def _sine(
    frequency_hz: float, duration_s: float, *, amplitude: float = 0.5
) -> np.ndarray:
    samples = int(round(duration_s * SAMPLE_RATE))
    t = np.arange(samples) / SAMPLE_RATE
    return (amplitude * np.sin(2.0 * np.pi * frequency_hz * t)).astype(np.float32)


def _harmonic_stack(
    fundamental_hz: float,
    duration_s: float,
    *,
    harmonics: list[int],
    amplitudes: list[float] | None = None,
) -> np.ndarray:
    samples = int(round(duration_s * SAMPLE_RATE))
    t = np.arange(samples) / SAMPLE_RATE
    audio = np.zeros_like(t)
    if amplitudes is None:
        amplitudes = [0.4 / n for n in harmonics]
    for n, amp in zip(harmonics, amplitudes):
        audio += amp * np.sin(2.0 * np.pi * fundamental_hz * n * t)
    return audio.astype(np.float32)


# ---------------------------------------------------------------------------
# Basic F0 selection
# ---------------------------------------------------------------------------


def test_pure_tone_200hz_with_harmonics_picks_f0() -> None:
    audio = _harmonic_stack(
        200.0, duration_s=0.30, harmonics=[1, 2, 3, 4], amplitudes=[0.4, 0.3, 0.2, 0.15]
    )
    result = extract_notes_v5_candidate(audio, SAMPLE_RATE, params=_params())
    f0_notes = [n for n in result.notes if n.partial_index == 0]
    assert len(f0_notes) >= 1
    # 200 Hz ≈ MIDI 55 (G3); allow ±1 for CQT bin quantisation.
    assert all(abs(n.midi_pitch - 55) <= 1 for n in f0_notes)
    f0_contours = [c for c in result.contours if c.note_uid == f0_notes[0].note_uid]
    assert f0_contours
    # v5 reserves subharmonic_octave; every contour row carries 0.
    assert all(c.subharmonic_octave == 0 for c in f0_contours)


# ---------------------------------------------------------------------------
# H2 lock recovery (Viterbi should pick the F0 that explains the most
# partials, not the loudest single bin)
# ---------------------------------------------------------------------------


def test_strong_h2_does_not_lock_f0_to_h2() -> None:
    # F0 200 Hz with H2..H4 layered on top; H2 is the loudest single
    # partial. A frame-by-frame max picker would call F0 = 400 Hz; the
    # harmonic-sum at 200 Hz includes all four partials and outscores
    # the harmonic-sum at 400 Hz (which only sees H2/H4).
    audio = _harmonic_stack(
        200.0,
        duration_s=0.40,
        harmonics=[1, 2, 3, 4],
        amplitudes=[0.10, 0.40, 0.20, 0.20],
    )
    result = extract_notes_v5_candidate(audio, SAMPLE_RATE, params=_params())
    f0_notes = [n for n in result.notes if n.partial_index == 0]
    assert len(f0_notes) >= 1
    # 200 Hz → MIDI 55; absolutely should not land at MIDI 67 (400 Hz).
    assert all(n.midi_pitch <= 60 for n in f0_notes)


# ---------------------------------------------------------------------------
# Sweep tracking (Viterbi smoothness should not flatten the sweep)
# ---------------------------------------------------------------------------


def test_linear_sweep_50_to_80_hz_tracks_endpoints() -> None:
    duration_s = 1.0
    samples = int(round(duration_s * SAMPLE_RATE))
    t = np.arange(samples) / SAMPLE_RATE
    log_start = math.log2(50.0)
    log_end = math.log2(80.0)
    inst_log = log_start + (log_end - log_start) * t / duration_s
    inst_hz = np.power(2.0, inst_log)
    phase_f0 = 2.0 * np.pi * np.cumsum(inst_hz) / SAMPLE_RATE
    audio = (
        0.10 * np.sin(phase_f0)
        + 0.30 * np.sin(2.0 * phase_f0)
        + 0.20 * np.sin(3.0 * phase_f0)
        + 0.15 * np.sin(4.0 * phase_f0)
    ).astype(np.float32)
    result = extract_notes_v5_candidate(audio, SAMPLE_RATE, params=_params())
    f0_notes = [n for n in result.notes if n.partial_index == 0]
    assert f0_notes
    # 50 Hz ≈ MIDI 32; 80 Hz ≈ MIDI 40. Collect contour frames across
    # all F0 notes (sweeps can produce one note or several depending on
    # segmentation gaps). MIDI range should span the sweep within a
    # couple of semitones at each end.
    contour_by_uid = {
        n.note_uid: [c for c in result.contours if c.note_uid == n.note_uid]
        for n in f0_notes
    }
    instantaneous_midi: list[float] = []
    for n in f0_notes:
        for c in contour_by_uid[n.note_uid]:
            instantaneous_midi.append(n.midi_pitch + c.cents_from_pitch / 100.0)
    assert instantaneous_midi
    assert min(instantaneous_midi) <= 35  # near 50 Hz (MIDI 32)
    assert max(instantaneous_midi) >= 37  # near 80 Hz (MIDI 40)


# ---------------------------------------------------------------------------
# Voicing gating
# ---------------------------------------------------------------------------


def test_pure_broadband_noise_emits_no_notes() -> None:
    rng = np.random.default_rng(seed=0)
    duration_s = 0.50
    samples = int(round(duration_s * SAMPLE_RATE))
    audio = (rng.standard_normal(samples) * 0.1).astype(np.float32)
    result = extract_notes_v5_candidate(audio, SAMPLE_RATE, params=_params())
    # No coherent harmonic content → voicing inactive → no notes.
    assert not [n for n in result.notes if n.partial_index == 0]


# ---------------------------------------------------------------------------
# Competing tones with no harmonic relationship
# ---------------------------------------------------------------------------


def test_two_pure_tones_no_harmonic_relation_does_not_flap() -> None:
    # 200 Hz and 470 Hz (not an integer multiple); the louder of the two
    # should win and the Viterbi should not bounce between them
    # frame-to-frame.
    duration_s = 0.40
    samples = int(round(duration_s * SAMPLE_RATE))
    t = np.arange(samples) / SAMPLE_RATE
    audio = (
        0.50 * np.sin(2.0 * np.pi * 200.0 * t) + 0.20 * np.sin(2.0 * np.pi * 470.0 * t)
    ).astype(np.float32)
    result = extract_notes_v5_candidate(audio, SAMPLE_RATE, params=_params())
    f0_notes = [n for n in result.notes if n.partial_index == 0]
    assert f0_notes
    # Median pitch should sit at 200 Hz (MIDI 55), not 470 Hz (MIDI ~71).
    pitches = [n.midi_pitch for n in f0_notes]
    assert max(pitches) <= 60


# ---------------------------------------------------------------------------
# Degenerate inputs
# ---------------------------------------------------------------------------


def test_empty_audio_returns_empty_result() -> None:
    result = extract_notes_v5_candidate(
        np.zeros(0, dtype=np.float32), SAMPLE_RATE, params=_params()
    )
    assert result.notes == []
    assert result.contours == []


def test_very_short_audio_returns_empty_result() -> None:
    # ~5 ms → fewer than min_note_frames after CQT.
    samples = int(round(0.005 * SAMPLE_RATE))
    result = extract_notes_v5_candidate(
        _sine(200.0, duration_s=0.005)[:samples], SAMPLE_RATE, params=_params()
    )
    assert result.notes == []


# ---------------------------------------------------------------------------
# Signature parity
# ---------------------------------------------------------------------------


def test_ridge_sidecar_argument_is_accepted_and_ignored() -> None:
    audio = _harmonic_stack(
        200.0, duration_s=0.30, harmonics=[1, 2, 3], amplitudes=[0.4, 0.3, 0.2]
    )
    fake_sidecar = [
        {
            "frame_index": 0,
            "frame_time_offset_s": 0.0,
            "log_frequency": 999.0,  # nonsense; v5 should ignore
            "strength": 0.0,
            "energy_ratio": 0.0,
        }
    ]
    result_with = extract_notes_v5_candidate(
        audio, SAMPLE_RATE, params=_params(), ridge_sidecar_rows=fake_sidecar
    )
    result_without = extract_notes_v5_candidate(audio, SAMPLE_RATE, params=_params())
    pitches_with = sorted(
        n.midi_pitch for n in result_with.notes if n.partial_index == 0
    )
    pitches_without = sorted(
        n.midi_pitch for n in result_without.notes if n.partial_index == 0
    )
    assert pitches_with == pitches_without
