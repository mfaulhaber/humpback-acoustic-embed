"""Tests for ``humpback.processing.note_extractor_v6``.

Two layers:

1. The pure ``despike_f0_segments`` slew-rate anchor walk against hand-
   built ``_RefinedFrame`` contours (spec §4): out-and-back spikes are
   bridged, sustained legal glides are untouched, leading/trailing spikes
   are held, and over-wide excursions are accepted as level changes.
2. ``extract_notes_v6`` end-to-end against synthetic audio: the v5
   ``NotesV3Result`` contract is preserved, and ``despike.enabled=False``
   reproduces v5 output exactly.
"""

from __future__ import annotations

import numpy as np

from humpback.processing.note_extractor_v3 import STFTParams, _RefinedFrame
from humpback.processing.note_extractor_v5 import (
    ExtractNotesV5Params,
    HarmonicViterbiParams,
    extract_notes_v5,
)
from humpback.processing.note_extractor_v6 import (
    ContourFrame,
    DespikeParams,
    ExtractNotesV6Params,
    despike_f0_segments,
    extract_notes_v6,
)
from humpback.processing.piano_roll_cqt import CQTParams

SAMPLE_RATE = 22050
# dt=0.1 with max_slope=1.0 oct/s gives a clean 0.1-octave per-frame
# budget, so the test contours read directly in octaves.
_DT = 0.1


def _mk(log_freqs: list[float]) -> list[_RefinedFrame]:
    return [
        _RefinedFrame(
            frame_index=i,
            time_offset_s=i * _DT,
            log_frequency=float(lf),
            strength=1.0,
            subharmonic_octave=0,
        )
        for i, lf in enumerate(log_freqs)
    ]


def _despike(
    log_freqs: list[float],
    *,
    max_slope_oct_per_s: float = 1.0,
    max_spike_frames: int = 12,
) -> list[float]:
    out = despike_f0_segments(
        [_mk(log_freqs)],
        dt=_DT,
        params=DespikeParams(
            enabled=True,
            max_slope_oct_per_s=max_slope_oct_per_s,
            max_spike_frames=max_spike_frames,
        ),
    )
    return [f.log_frequency for f in out[0]]


# ---------------------------------------------------------------------------
# Pure de-spike: anchor walk
# ---------------------------------------------------------------------------


def test_out_and_back_spike_is_bridged() -> None:
    result = _despike([6.0, 6.0, 6.0, 6.0, 6.0, 6.5, 6.0, 6.0, 6.0, 6.0])
    # The single steep frame is excised and bridged to the flat line.
    assert result[5] == 6.0
    assert all(abs(v - 6.0) < 1e-9 for v in result)


def test_sustained_legal_glide_is_unchanged() -> None:
    glide = [6.0 + 0.05 * i for i in range(10)]  # 0.05 oct/frame < 0.1 budget
    out = despike_f0_segments(
        [_mk(glide)],
        dt=_DT,
        params=DespikeParams(enabled=True, max_slope_oct_per_s=1.0),
    )
    assert [f.log_frequency for f in out[0]] == glide


def test_trailing_excursion_left_untouched() -> None:
    # A trailing excursion that never returns to the trajectory is a
    # level change, not an out-and-back spike: leave it untouched rather
    # than flattening (joining) it back to the prior level.
    result = _despike([6.0] * 8 + [7.0, 7.0])
    assert result[8] == 7.0
    assert result[9] == 7.0
    assert all(abs(v - 6.0) < 1e-9 for v in result[:8])


def test_leading_excursion_left_untouched() -> None:
    # Frame 0 sits a register away and the contour never returns to it:
    # it is not a confirmed out-and-back spike, so the real low contour
    # is preserved and the lone leading frame is left as-is.
    result = _despike([7.0] + [6.0] * 9)
    assert result[0] == 7.0
    assert all(abs(v - 6.0) < 1e-9 for v in result[1:])


def test_register_jump_is_not_bridged() -> None:
    # Regression for event cb23dfcd: a low anchor (left), a sustained high
    # region, and a high right anchor. The earlier "steep-is-always-error"
    # guard ramped the high region down from the low anchor. It must now be
    # left intact — a non-returning excursion is a level change, not a spike.
    contour = [3.0, 3.0, 3.0] + [9.0] * 11 + [8.9, 8.9, 8.9]
    result = _despike(contour, max_spike_frames=12)
    # The high region is preserved (NOT ramped down toward the low anchor).
    assert all(abs(v - 9.0) < 1e-9 for v in result[3:14])
    # The low lead-in and high tail are preserved too.
    assert all(abs(v - 3.0) < 1e-9 for v in result[:3])
    assert all(abs(v - 8.9) < 1e-9 for v in result[14:])


def test_sustained_step_is_left_unchanged() -> None:
    # A clean step up that never returns is a level change: despike is a
    # no-op (no ramp, no flattening).
    contour = [6.0, 6.0, 6.0] + [7.0] * 7
    result = _despike(contour, max_spike_frames=3)
    assert result == contour


def test_multiple_spikes_all_bridged() -> None:
    result = _despike([6.0, 6.0, 6.5, 6.0, 6.0, 5.5, 6.0, 6.0, 6.0, 6.0])
    assert result[2] == 6.0  # up spike bridged
    assert result[5] == 6.0  # down spike bridged
    assert all(abs(v - 6.0) < 1e-9 for v in result)


def test_disabled_is_noop() -> None:
    spiky = [6.0, 6.0, 6.5, 6.0, 6.0]
    out = despike_f0_segments([_mk(spiky)], dt=_DT, params=DespikeParams(enabled=False))
    assert [f.log_frequency for f in out[0]] == spiky


def test_short_segment_is_untouched() -> None:
    short = [6.0, 7.5]  # fewer than 3 frames; no anchor walk
    out = despike_f0_segments([_mk(short)], dt=_DT, params=DespikeParams(enabled=True))
    assert [f.log_frequency for f in out[0]] == short


def test_strength_and_subharmonic_carried_through() -> None:
    frames = [
        _RefinedFrame(
            frame_index=i,
            time_offset_s=i * _DT,
            log_frequency=lf,
            strength=float(i),
            subharmonic_octave=0,
        )
        for i, lf in enumerate([6.0, 6.0, 6.5, 6.0, 6.0])
    ]
    out = despike_f0_segments(
        [frames], dt=_DT, params=DespikeParams(enabled=True, max_slope_oct_per_s=1.0)
    )[0]
    assert [f.strength for f in out] == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert all(f.subharmonic_octave == 0 for f in out)


# ---------------------------------------------------------------------------
# extract_notes_v6 end-to-end
# ---------------------------------------------------------------------------


def _harmonic_stack(
    fundamental_hz: float,
    duration_s: float,
    *,
    harmonics: list[int],
    amplitudes: list[float],
) -> np.ndarray:
    samples = int(round(duration_s * SAMPLE_RATE))
    t = np.arange(samples) / SAMPLE_RATE
    audio = np.zeros_like(t)
    for n, amp in zip(harmonics, amplitudes):
        audio += amp * np.sin(2.0 * np.pi * fundamental_hz * n * t)
    return audio.astype(np.float32)


def _v6_params(*, despike: DespikeParams) -> ExtractNotesV6Params:
    return ExtractNotesV6Params(
        job_id="job-v6-test",
        event_id="ev-1",
        event_start_utc=1000.0,
        pad_seconds=0.0,
        cqt=CQTParams(),
        stft=STFTParams(min_frequency_hz=30.0),
        harmonic_viterbi=HarmonicViterbiParams(),
        despike=despike,
    )


def _v5_params() -> ExtractNotesV5Params:
    return ExtractNotesV5Params(
        job_id="job-v6-test",
        event_id="ev-1",
        event_start_utc=1000.0,
        pad_seconds=0.0,
        cqt=CQTParams(),
        stft=STFTParams(min_frequency_hz=30.0),
        harmonic_viterbi=HarmonicViterbiParams(),
    )


def test_extract_v6_picks_f0_and_reserves_subharmonic() -> None:
    audio = _harmonic_stack(
        200.0, duration_s=0.30, harmonics=[1, 2, 3, 4], amplitudes=[0.4, 0.3, 0.2, 0.15]
    )
    result = extract_notes_v6(
        audio, SAMPLE_RATE, params=_v6_params(despike=DespikeParams())
    )
    f0_notes = [n for n in result.notes if n.partial_index == 0]
    assert len(f0_notes) >= 1
    assert all(abs(n.midi_pitch - 55) <= 1 for n in f0_notes)  # 200 Hz ≈ MIDI 55
    assert all(c.subharmonic_octave == 0 for c in result.contours)


def test_disabled_despike_reproduces_v5() -> None:
    audio = _harmonic_stack(
        200.0,
        duration_s=0.40,
        harmonics=[1, 2, 3, 4],
        amplitudes=[0.10, 0.40, 0.20, 0.20],
    )
    v6 = extract_notes_v6(
        audio, SAMPLE_RATE, params=_v6_params(despike=DespikeParams(enabled=False))
    )
    v5 = extract_notes_v5(audio, SAMPLE_RATE, params=_v5_params())
    assert v6.notes == v5.notes
    assert v6.contours == v5.contours


def test_enabled_despike_is_noop_on_clean_tone() -> None:
    # A clean, smooth contour has no steep frames, so enabling de-spike
    # changes nothing relative to the disabled run.
    audio = _harmonic_stack(
        200.0, duration_s=0.40, harmonics=[1, 2, 3], amplitudes=[0.4, 0.3, 0.2]
    )
    enabled = extract_notes_v6(
        audio, SAMPLE_RATE, params=_v6_params(despike=DespikeParams(enabled=True))
    )
    disabled = extract_notes_v6(
        audio, SAMPLE_RATE, params=_v6_params(despike=DespikeParams(enabled=False))
    )
    assert enabled.notes == disabled.notes
    assert enabled.contours == disabled.contours


def test_empty_audio_returns_empty_result() -> None:
    result = extract_notes_v6(
        np.zeros(0, dtype=np.float32),
        SAMPLE_RATE,
        params=_v6_params(despike=DespikeParams()),
    )
    assert result.notes == []
    assert result.contours == []


def test_harmonic_cents_track_f0_when_segment_starts_after_silence() -> None:
    """Harmonic contour cents must equal the F0 cents at the same frame.

    Regression for the upper-harmonic "slope spike" ladder: leading
    silence forces the F0 segment to begin at a CQT frame_index > 0, and a
    pitch glide makes the cents time-varying. Harmonic notes inherit the
    F0 bend by cents conservation, so at each shared event-time the
    harmonic's ``cents_from_pitch`` must match the F0's. A frame-index
    key mismatch in the harmonic contour lookup borrows F0 cents from a
    time-shifted frame, which this test detects. De-spike is disabled so
    the F0 cents are the raw glide.
    """
    sr = SAMPLE_RATE
    silence = np.zeros(int(0.30 * sr), dtype=np.float32)
    duration_s = 0.60
    t = np.arange(int(duration_s * sr)) / sr
    f_start, f_end = 200.0, 250.0
    inst_hz = f_start * (f_end / f_start) ** (t / duration_s)  # log-linear glide
    phase = 2.0 * np.pi * np.cumsum(inst_hz) / sr
    tone = (
        0.40 * np.sin(phase)
        + 0.30 * np.sin(2.0 * phase)
        + 0.20 * np.sin(3.0 * phase)
        + 0.15 * np.sin(4.0 * phase)
    ).astype(np.float32)
    audio = np.concatenate([silence, tone, silence])

    result = extract_notes_v6(
        audio, sr, params=_v6_params(despike=DespikeParams(enabled=False))
    )
    f0_notes = [n for n in result.notes if n.partial_index == 0]
    assert f0_notes
    f0 = max(f0_notes, key=lambda n: n.duration_s)
    h_notes = [
        n
        for n in result.notes
        if n.partial_index == 1 and abs(n.start_offset_s - f0.start_offset_s) < 0.05
    ]
    assert h_notes, "no H2 note overlapping the F0 onset"
    h2 = max(h_notes, key=lambda n: n.duration_s)

    by_uid: dict[str, list[ContourFrame]] = {}
    for c in result.contours:
        by_uid.setdefault(c.note_uid, []).append(c)
    f0_by_t = {
        round(f0.start_offset_s + c.time_offset_s, 4): c.cents_from_pitch
        for c in by_uid[f0.note_uid]
    }
    h2_by_t = {
        round(h2.start_offset_s + c.time_offset_s, 4): c.cents_from_pitch
        for c in by_uid[h2.note_uid]
    }
    shared = sorted(set(f0_by_t) & set(h2_by_t))
    assert len(shared) > 10
    max_diff = max(abs(f0_by_t[t] - h2_by_t[t]) for t in shared)
    assert max_diff < 5.0, (
        "harmonic cents diverge from F0 cents (key mismatch); "
        f"max |Δcents| = {max_diff:.1f}"
    )
