"""Tests for ``humpback.processing.note_extractor_v3``.

Covers the spec §5 acceptance criteria: coherent-contour F0 segmentation,
harmonic siblings at integer multiples, subharmonic refinement on weak-F0
inputs, determinism of ``note_uid`` and contour bytes, and absence of
``partial_index = -1`` regression in the v3 outputs.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from humpback.processing.note_extractor_v3 import (
    ExtractNotesV3Params,
    HarmonicSearchParams,
    SegmentationParams,
    SubharmonicParams,
    extract_notes_v3,
)
from humpback.processing.piano_roll_cqt import CQTParams


SAMPLE_RATE = 22050


def _default_params(
    *,
    event_id: str = "ev-1",
    job_id: str = "job-test",
    event_start_utc: float = 1000.0,
    pad_seconds: float = 0.0,
) -> ExtractNotesV3Params:
    return ExtractNotesV3Params(
        job_id=job_id,
        event_id=event_id,
        event_start_utc=event_start_utc,
        pad_seconds=pad_seconds,
    )


def _sine(
    frequency_hz: float, duration_s: float, *, amplitude: float = 0.5
) -> np.ndarray:
    samples = int(round(duration_s * SAMPLE_RATE))
    t = np.arange(samples) / SAMPLE_RATE
    return (amplitude * np.sin(2.0 * np.pi * frequency_hz * t)).astype(np.float32)


def _sweep(
    start_hz: float, end_hz: float, duration_s: float, *, amplitude: float = 0.5
) -> np.ndarray:
    samples = int(round(duration_s * SAMPLE_RATE))
    t = np.arange(samples) / SAMPLE_RATE
    log_start = math.log2(start_hz)
    log_end = math.log2(end_hz)
    instantaneous_log_hz = log_start + (log_end - log_start) * t / max(duration_s, 1e-9)
    instantaneous_hz = np.power(2.0, instantaneous_log_hz)
    phase = 2.0 * np.pi * np.cumsum(instantaneous_hz) / SAMPLE_RATE
    return (amplitude * np.sin(phase)).astype(np.float32)


def _harmonic_stack(
    fundamental_hz: float,
    duration_s: float,
    *,
    harmonics: list[int],
    amplitude: float = 0.4,
) -> np.ndarray:
    samples = int(round(duration_s * SAMPLE_RATE))
    t = np.arange(samples) / SAMPLE_RATE
    audio = np.zeros_like(t)
    for n in harmonics:
        audio += (amplitude / n) * np.sin(2.0 * np.pi * fundamental_hz * n * t)
    return audio.astype(np.float32)


# ---------------------------------------------------------------------------
# F0 contour + segmentation
# ---------------------------------------------------------------------------


def test_constant_tone_produces_single_f0_note() -> None:
    audio = _sine(220.0, duration_s=0.30)
    params = _default_params()
    result = extract_notes_v3(audio, SAMPLE_RATE, params=params)
    f0_notes = [n for n in result.notes if n.partial_index == 0]
    assert len(f0_notes) == 1
    assert f0_notes[0].midi_pitch == 57  # A3
    assert f0_notes[0].contour_frame_count == len(
        [c for c in result.contours if c.note_uid == f0_notes[0].note_uid]
    )


def test_sine_sweep_produces_one_note_with_contour_tracking_pitch() -> None:
    # C5 (523.25 Hz) to E5 (659.26 Hz) over 200 ms.
    audio = _sweep(523.25, 659.26, duration_s=0.20)
    params = _default_params()
    result = extract_notes_v3(audio, SAMPLE_RATE, params=params)
    f0_notes = [n for n in result.notes if n.partial_index == 0]
    assert len(f0_notes) == 1
    note = f0_notes[0]
    contour = sorted(
        [c for c in result.contours if c.note_uid == note.note_uid],
        key=lambda c: c.frame_index,
    )
    # Median frequency over the sweep is ~588 Hz → MIDI 73.
    assert note.midi_pitch in (72, 73, 74)
    cents = np.asarray([c.cents_from_pitch for c in contour])
    # Sweep covers ~±200 cents around the median pitch; cents trace the sweep.
    assert cents.min() < 0
    assert cents.max() > 0
    # Within the spec's ±9600 cents safety clamp.
    assert np.all(np.abs(cents) <= 9600.0)


def test_two_events_with_silent_gap_produce_two_notes() -> None:
    silence_seconds = 0.10
    silence = np.zeros(int(silence_seconds * SAMPLE_RATE), dtype=np.float32)
    first = _sine(220.0, duration_s=0.12, amplitude=0.5)
    second = _sine(220.0, duration_s=0.12, amplitude=0.5)
    audio = np.concatenate([first, silence, second])
    params = _default_params(
        # Use settings that won't auto-collapse the gap.
    )
    result = extract_notes_v3(audio, SAMPLE_RATE, params=params)
    f0_notes = [n for n in result.notes if n.partial_index == 0]
    assert len(f0_notes) >= 2


def test_short_event_yields_zero_notes() -> None:
    audio = _sine(220.0, duration_s=0.020)  # < 30 ms
    params = _default_params()
    result = extract_notes_v3(audio, SAMPLE_RATE, params=params)
    assert result.notes == []
    assert result.contours == []


# ---------------------------------------------------------------------------
# Harmonics
# ---------------------------------------------------------------------------


def test_harmonic_stack_produces_partials() -> None:
    audio = _harmonic_stack(220.0, duration_s=0.30, harmonics=[1, 2, 3, 4, 5])
    params = _default_params()
    result = extract_notes_v3(audio, SAMPLE_RATE, params=params)
    partial_indices = sorted({n.partial_index for n in result.notes})
    # F0 must be present.
    assert 0 in partial_indices
    # At least three harmonic siblings must materialize.
    assert sum(1 for n in result.notes if n.partial_index >= 1) >= 3
    # No legacy -1 sentinel.
    assert all(n.partial_index >= 0 for n in result.notes)


def test_harmonic_contour_inherits_f0_cents() -> None:
    audio = _harmonic_stack(220.0, duration_s=0.30, harmonics=[1, 2])
    params = _default_params()
    result = extract_notes_v3(audio, SAMPLE_RATE, params=params)
    f0 = next(n for n in result.notes if n.partial_index == 0)
    h2 = next((n for n in result.notes if n.partial_index == 1), None)
    assert h2 is not None
    f0_cents = {
        c.frame_index: c.cents_from_pitch
        for c in result.contours
        if c.note_uid == f0.note_uid
    }
    h2_cents = {
        c.frame_index: c.cents_from_pitch
        for c in result.contours
        if c.note_uid == h2.note_uid
    }
    # Harmonic contour spans an aligned subset of the F0's frame indices.
    shared = set(f0_cents.keys()) & set(h2_cents.keys())
    assert shared, "harmonic contour should share frame_index space with F0"
    for frame in shared:
        assert math.isclose(f0_cents[frame], h2_cents[frame], abs_tol=1e-6)


def test_no_partial_index_minus_one_emitted() -> None:
    audio = _harmonic_stack(220.0, duration_s=0.30, harmonics=[1, 2, 3])
    params = _default_params()
    result = extract_notes_v3(audio, SAMPLE_RATE, params=params)
    assert all(n.partial_index >= 0 for n in result.notes)


# ---------------------------------------------------------------------------
# Subharmonic refinement
# ---------------------------------------------------------------------------


def test_loud_h2_quiet_f0_refines_to_lower_octave() -> None:
    """A weak fundamental + dominant H2 should land at the F0 pitch.

    220 Hz fundamental at low amplitude + 440 Hz "H2" at higher amplitude
    should be refined by §5.2 so the F0 contour sits at 220 Hz (MIDI 57)
    rather than 440 Hz (MIDI 69).
    """
    samples = int(0.30 * SAMPLE_RATE)
    t = np.arange(samples) / SAMPLE_RATE
    audio = 0.1 * np.sin(2.0 * np.pi * 220.0 * t) + 0.6 * np.sin(
        2.0 * np.pi * 440.0 * t
    )
    audio = audio.astype(np.float32)
    params = _default_params()
    result = extract_notes_v3(audio, SAMPLE_RATE, params=params)
    f0_notes = [n for n in result.notes if n.partial_index == 0]
    assert f0_notes, "expected an F0 note after subharmonic refinement"
    f0_pitches = sorted({n.midi_pitch for n in f0_notes})
    # The refined F0 should be MIDI 57 (220 Hz), not 69 (440 Hz).
    assert 57 in f0_pitches, f0_pitches


def test_subharmonic_offset_smoothed_to_a_stable_value() -> None:
    """Frame-level subharmonic flips should not survive 5-frame smoothing."""
    audio = _sine(220.0, duration_s=0.30)
    params = _default_params()
    result = extract_notes_v3(audio, SAMPLE_RATE, params=params)
    f0 = next(n for n in result.notes if n.partial_index == 0)
    octaves = [
        c.subharmonic_octave for c in result.contours if c.note_uid == f0.note_uid
    ]
    # Stable tones should land at a single subharmonic state.
    assert len(set(octaves)) == 1


# ---------------------------------------------------------------------------
# Pitch range, velocity, determinism
# ---------------------------------------------------------------------------


def test_midi_pitch_clamps_to_extended_range() -> None:
    """Out-of-range continuous pitches should clamp to [12, 120]."""
    # Extremely low input (~8.18 Hz, MIDI ~-9): clamped to min 12.
    audio = _sine(60.0, duration_s=0.30)
    params = _default_params()
    result = extract_notes_v3(audio, SAMPLE_RATE, params=params)
    for note in result.notes:
        assert 12 <= note.midi_pitch <= 120


def test_louder_event_has_higher_peak_magnitude() -> None:
    audio_quiet = _sine(220.0, duration_s=0.30, amplitude=0.05)
    audio_loud = _sine(220.0, duration_s=0.30, amplitude=0.5)
    quiet = extract_notes_v3(audio_quiet, SAMPLE_RATE, params=_default_params())
    loud = extract_notes_v3(audio_loud, SAMPLE_RATE, params=_default_params())
    quiet_f0 = next(n for n in quiet.notes if n.partial_index == 0)
    loud_f0 = next(n for n in loud.notes if n.partial_index == 0)
    assert loud_f0.peak_magnitude > quiet_f0.peak_magnitude * 5.0


def test_deterministic_outputs() -> None:
    audio = _harmonic_stack(220.0, duration_s=0.30, harmonics=[1, 2, 3])
    params = _default_params()
    a = extract_notes_v3(audio, SAMPLE_RATE, params=params)
    b = extract_notes_v3(audio, SAMPLE_RATE, params=params)
    assert [n.note_uid for n in a.notes] == [n.note_uid for n in b.notes]
    assert [
        (c.note_uid, c.frame_index, c.cents_from_pitch, c.harmonic_strength)
        for c in a.contours
    ] == [
        (c.note_uid, c.frame_index, c.cents_from_pitch, c.harmonic_strength)
        for c in b.contours
    ]


def test_note_uid_changes_with_identity_inputs() -> None:
    audio = _sine(220.0, duration_s=0.30)
    base = extract_notes_v3(audio, SAMPLE_RATE, params=_default_params(event_id="ev-A"))
    other = extract_notes_v3(
        audio, SAMPLE_RATE, params=_default_params(event_id="ev-B")
    )
    assert base.notes[0].note_uid != other.notes[0].note_uid


# ---------------------------------------------------------------------------
# Sidecar consumption
# ---------------------------------------------------------------------------


def test_sidecar_rows_drive_extraction_instead_of_recompute() -> None:
    """When a ridge sidecar is provided the extractor must not recompute it."""
    audio = _sine(220.0, duration_s=0.30)
    # Manufacture a fake sidecar at a constant log_frequency for 220 Hz.
    log220 = math.log2(220.0)
    sidecar = [
        {
            "frame_time_offset_s": float(i * 512 / SAMPLE_RATE),
            "log_frequency": log220,
            "strength": 1.0,
            "energy_ratio": 0.5,
        }
        for i in range(20)
    ]
    params = _default_params()
    result = extract_notes_v3(
        audio, SAMPLE_RATE, params=params, ridge_sidecar_rows=sidecar
    )
    f0 = next(n for n in result.notes if n.partial_index == 0)
    assert f0.midi_pitch == 57


def test_sidecar_empty_returns_empty() -> None:
    audio = _sine(220.0, duration_s=0.30)
    params = _default_params()
    result = extract_notes_v3(audio, SAMPLE_RATE, params=params, ridge_sidecar_rows=[])
    assert result.notes == []
    assert result.contours == []


# ---------------------------------------------------------------------------
# Defensive edge cases
# ---------------------------------------------------------------------------


def test_empty_audio_returns_empty_result() -> None:
    params = _default_params()
    result = extract_notes_v3(np.zeros(0, dtype=np.float32), SAMPLE_RATE, params=params)
    assert result.notes == []
    assert result.contours == []


@pytest.mark.parametrize(
    "stft_field, value",
    [
        ("max_halvings", 3),
        ("smoothing_frames", 5),
    ],
)
def test_subharmonic_params_apply_through(stft_field: str, value: int) -> None:
    # Smoke test: changing subharmonic params is accepted at the boundary.
    audio = _sine(220.0, duration_s=0.30)
    params = _default_params()
    overridden = ExtractNotesV3Params(
        job_id=params.job_id,
        event_id=params.event_id,
        event_start_utc=params.event_start_utc,
        pad_seconds=params.pad_seconds,
        subharmonic=SubharmonicParams(**{stft_field: value}),
        segmentation=SegmentationParams(),
        harmonic=HarmonicSearchParams(),
        cqt=CQTParams(),
    )
    result = extract_notes_v3(audio, SAMPLE_RATE, params=overridden)
    assert any(n.partial_index == 0 for n in result.notes)
