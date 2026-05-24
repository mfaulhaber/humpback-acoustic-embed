"""Tests for ``humpback.processing.note_extractor_v4``.

Cover the spec acceptance criteria for the HPS F0 selection stage and
the 30 Hz ridge band floor. Reuses synthetic signals like the v3 tests
but focuses on cases where v3's octave-halving refinement was known to
fail: H3 lock, sub-100 Hz fundamentals, and broadband sub-100 Hz noise
without harmonic structure.
"""

from __future__ import annotations

import math

import numpy as np

from humpback.processing.note_extractor_v4 import (
    ExtractNotesV4Params,
    HPSParams,
    STFTParams,
    extract_notes_v4,
)
from humpback.processing.piano_roll_cqt import CQTParams


SAMPLE_RATE = 22050


def _params(
    *,
    hps: HPSParams | None = None,
    stft: STFTParams | None = None,
    cqt: CQTParams | None = None,
) -> ExtractNotesV4Params:
    return ExtractNotesV4Params(
        job_id="job-v4-test",
        event_id="ev-1",
        event_start_utc=1000.0,
        pad_seconds=0.0,
        cqt=cqt if cqt is not None else CQTParams(),
        stft=stft if stft is not None else STFTParams(min_frequency_hz=30.0),
        hps=hps if hps is not None else HPSParams(),
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


def test_pure_tone_picks_ridge_as_f0() -> None:
    audio = _sine(220.0, duration_s=0.30)
    result = extract_notes_v4(audio, SAMPLE_RATE, params=_params())
    f0_notes = [n for n in result.notes if n.partial_index == 0]
    assert len(f0_notes) == 1
    # A3 = MIDI 57; HPS picks d=1 so no shift applied.
    assert f0_notes[0].midi_pitch == 57
    # All F0 contour frames should record divisor=1 (subharmonic_octave=0).
    f0_contours = [c for c in result.contours if c.note_uid == f0_notes[0].note_uid]
    assert f0_contours
    assert all(c.subharmonic_octave == 0 for c in f0_contours)


def test_pure_tone_1khz_no_harmonics_keeps_d1() -> None:
    audio = _sine(1000.0, duration_s=0.30)
    result = extract_notes_v4(audio, SAMPLE_RATE, params=_params())
    f0_notes = [n for n in result.notes if n.partial_index == 0]
    assert len(f0_notes) == 1
    # 1 kHz ≈ MIDI 83. Allow ±1 for nearest-bin quantization.
    assert abs(f0_notes[0].midi_pitch - 83) <= 1


# ---------------------------------------------------------------------------
# H2 / H3 lock recovery
# ---------------------------------------------------------------------------


def test_strong_h2_ridge_picks_d2_for_true_f0() -> None:
    # F0 at 200 Hz, H2 at 400 Hz +12 dB louder so the ridge Viterbi locks
    # on H2. HPS should pick d=2 and emit F0 at MIDI 55 (G3 ≈ 196 Hz).
    audio = _harmonic_stack(
        200.0, duration_s=0.30, harmonics=[1, 2], amplitudes=[0.10, 0.40]
    )
    result = extract_notes_v4(audio, SAMPLE_RATE, params=_params())
    f0_notes = [n for n in result.notes if n.partial_index == 0]
    assert len(f0_notes) == 1
    # 200 Hz → MIDI 55 (G3); allow ±1.
    assert abs(f0_notes[0].midi_pitch - 55) <= 1
    f0_contours = [c for c in result.contours if c.note_uid == f0_notes[0].note_uid]
    # Majority of frames should record divisor=2 (subharmonic_octave=1).
    div_two = sum(1 for c in f0_contours if c.subharmonic_octave == 1)
    assert div_two >= int(0.5 * len(f0_contours))


def test_strong_h3_ridge_picks_d3_for_true_f0() -> None:
    # F0 at 150 Hz, H3 at 450 Hz dominates; H2 also present so HPS has
    # support for d=3 (450/150 = 3, 300 Hz also visible at d=2).
    audio = _harmonic_stack(
        150.0,
        duration_s=0.40,
        harmonics=[1, 2, 3],
        amplitudes=[0.05, 0.10, 0.40],
    )
    result = extract_notes_v4(audio, SAMPLE_RATE, params=_params())
    f0_notes = [n for n in result.notes if n.partial_index == 0]
    assert len(f0_notes) == 1
    # 150 Hz → MIDI 50 (D3); allow ±2 to absorb CQT bin coarseness.
    assert abs(f0_notes[0].midi_pitch - 50) <= 2


# ---------------------------------------------------------------------------
# Low-band coverage and noise rejection
# ---------------------------------------------------------------------------


def test_sub_100hz_fundamental_with_harmonic_support_recovered() -> None:
    # 50 Hz F0 with H2..H6. Without the 30 Hz floor and HPS, v3 would
    # cap at 100 Hz and emit nothing below it.
    audio = _harmonic_stack(
        50.0,
        duration_s=0.5,
        harmonics=[1, 2, 3, 4, 5, 6],
        amplitudes=[0.05, 0.10, 0.10, 0.10, 0.10, 0.10],
    )
    result = extract_notes_v4(audio, SAMPLE_RATE, params=_params())
    f0_notes = [n for n in result.notes if n.partial_index == 0]
    assert len(f0_notes) >= 1
    lowest = min(n.midi_pitch for n in f0_notes)
    # 50 Hz ≈ MIDI 32 (G♯1/A♭1); accept anything ≤ MIDI 40.
    assert lowest <= 40


def test_pure_sub100_noise_does_not_pull_f0_down() -> None:
    rng = np.random.default_rng(seed=0)
    duration_s = 0.30
    # Mid-band 600 Hz tone, plus broadband infrasonic noise below ~80 Hz.
    samples = int(round(duration_s * SAMPLE_RATE))
    t = np.arange(samples) / SAMPLE_RATE
    tone = 0.4 * np.sin(2.0 * np.pi * 600.0 * t)
    raw_noise = rng.standard_normal(samples) * 0.3
    # Cheap low-pass: cumulative-mean averaging compresses high frequencies.
    kernel_size = 220  # ~10 ms at 22050 Hz → roughly fc ≈ 100 Hz
    kernel = np.ones(kernel_size) / kernel_size
    infrasonic = np.convolve(raw_noise, kernel, mode="same")
    audio = (tone + infrasonic).astype(np.float32)
    result = extract_notes_v4(audio, SAMPLE_RATE, params=_params())
    f0_notes = [n for n in result.notes if n.partial_index == 0]
    assert f0_notes
    # No F0 should land below MIDI 50; the noise has no coherent
    # harmonics, so HPS cannot find sufficient support sub-100 Hz.
    assert min(n.midi_pitch for n in f0_notes) >= 50


# ---------------------------------------------------------------------------
# FM sweeps and segmentation
# ---------------------------------------------------------------------------


def test_fm_sweep_with_harmonic_support_tracks_f0() -> None:
    # Sweep F0 50 → 80 Hz with H2..H4 layered on top so the ridge tracker
    # may lock on H2 (100 → 160 Hz) yet HPS pulls back to the true F0.
    duration_s = 0.6
    samples = int(round(duration_s * SAMPLE_RATE))
    t = np.arange(samples) / SAMPLE_RATE
    log_start = math.log2(50.0)
    log_end = math.log2(80.0)
    inst_log = log_start + (log_end - log_start) * t / duration_s
    inst_hz = np.power(2.0, inst_log)
    phase_f0 = 2.0 * np.pi * np.cumsum(inst_hz) / SAMPLE_RATE
    audio = (
        0.05 * np.sin(phase_f0)
        + 0.20 * np.sin(2.0 * phase_f0)
        + 0.20 * np.sin(3.0 * phase_f0)
        + 0.20 * np.sin(4.0 * phase_f0)
    ).astype(np.float32)
    result = extract_notes_v4(audio, SAMPLE_RATE, params=_params())
    f0_notes = [n for n in result.notes if n.partial_index == 0]
    assert f0_notes
    # 50 Hz ≈ MIDI 32; 80 Hz ≈ MIDI 40. F0 should land somewhere in
    # that range (medians fall mid-sweep).
    pitches = [n.midi_pitch for n in f0_notes]
    assert max(pitches) <= 45


# ---------------------------------------------------------------------------
# Degenerate inputs
# ---------------------------------------------------------------------------


def test_empty_audio_returns_empty_result() -> None:
    result = extract_notes_v4(
        np.zeros(0, dtype=np.float32), SAMPLE_RATE, params=_params()
    )
    assert result.notes == []
    assert result.contours == []


def test_audio_too_short_for_stft_returns_empty() -> None:
    # 256 samples is well below n_fft=1024.
    result = extract_notes_v4(
        np.zeros(256, dtype=np.float32), SAMPLE_RATE, params=_params()
    )
    assert result.notes == []
    assert result.contours == []


# ---------------------------------------------------------------------------
# Sidecar ridge consumption
# ---------------------------------------------------------------------------


def test_sidecar_rows_are_consumed_without_recompute() -> None:
    """Passing ridge sidecar rows must drive HPS without re-running the
    STFT ridge tracker. Construct a synthetic ridge sitting on the 2nd
    harmonic of a 220 Hz tone; HPS should pick d=2 and emit F0 ≈ MIDI 57.
    """
    audio = _harmonic_stack(
        220.0, duration_s=0.30, harmonics=[1, 2], amplitudes=[0.10, 0.40]
    )
    hop = 512
    sr = 22050
    n_frames = int(audio.size / hop)
    # Synthetic ridge: every frame parked at 2*220 = 440 Hz exactly.
    sidecar = []
    for i in range(n_frames):
        sidecar.append(
            {
                "frame_index": i,
                "frame_time_offset_s": float(i * hop / sr),
                "log_frequency": float(math.log2(440.0)),
                "strength": 0.4,
                "energy_ratio": 0.3,
            }
        )
    result = extract_notes_v4(
        audio, SAMPLE_RATE, params=_params(), ridge_sidecar_rows=sidecar
    )
    f0_notes = [n for n in result.notes if n.partial_index == 0]
    assert f0_notes
    assert abs(f0_notes[0].midi_pitch - 57) <= 1


# ---------------------------------------------------------------------------
# Defaults and parameter contract
# ---------------------------------------------------------------------------


def test_divisor_oscillation_does_not_shred_long_contour() -> None:
    """ADR-070 regression: shipping v4 split F0 contours on every change in
    the chosen HPS divisor, fragmenting one coherent sweep into many short
    notes (96% of v4 fragmentation came from divisor-change splits on a
    production job). v4 must split only on amplitude gaps; divisor swaps
    within a continuous trajectory must stay one note.
    """
    # 0.8 s sweep with strong, sustained H2 + H3 + H4 layered on a weak F0.
    # The ridge tracker locks on H4 for the loudest stretch and on H2/H3
    # for the rest, so HPS legitimately picks different divisors across
    # frames. v3 would have shredded this; v4 must not.
    duration_s = 0.8
    samples = int(round(duration_s * SAMPLE_RATE))
    t = np.arange(samples) / SAMPLE_RATE
    log_start = math.log2(120.0)
    log_end = math.log2(150.0)
    inst_log = log_start + (log_end - log_start) * t / duration_s
    inst_hz = np.power(2.0, inst_log)
    phase_f0 = 2.0 * np.pi * np.cumsum(inst_hz) / SAMPLE_RATE
    # Time-varying harmonic amplitudes so the ridge wanders across H2/H3/H4.
    envelope = 0.5 + 0.5 * np.sin(2.0 * np.pi * 3.0 * t)
    audio = (
        0.05 * np.sin(phase_f0)
        + (0.30 * envelope) * np.sin(2.0 * phase_f0)
        + (0.30 * (1.0 - envelope)) * np.sin(3.0 * phase_f0)
        + 0.10 * np.sin(4.0 * phase_f0)
    ).astype(np.float32)

    result = extract_notes_v4(audio, SAMPLE_RATE, params=_params())
    f0_notes = [n for n in result.notes if n.partial_index == 0]
    # A single coherent sweep with no amplitude gaps must collapse into a
    # very small number of F0 notes — one in the ideal case, but allow up
    # to 3 to accommodate amplitude-floor edge effects on the envelope.
    assert len(f0_notes) <= 3, (
        f"v4 fragmented a continuous sweep into {len(f0_notes)} F0 notes "
        "(divisor-change splits should have been removed in segmentation)"
    )


def test_isolated_divisor_spikes_get_killed_by_median_pass() -> None:
    """ADR-070 regression: a 5-frame majority-smoothed divisor stream like
    [4, 6, 1, 6, 1] resolves the tie between 6 and 1 toward 1 (smallest),
    surfacing as a one-frame upward pitch spike in the contour. The
    3-point median pass added after the majority pass must collapse any
    single-frame divisor outlier between two matching neighbors.
    """
    from humpback.processing.note_extractor_v4 import _median3_smooth

    arr = np.asarray([6, 6, 1, 6, 6], dtype=np.int64)
    out = _median3_smooth(arr)
    # Center spike at index 2 gets replaced with its surrounding value.
    assert out.tolist() == [6, 6, 6, 6, 6]

    # Real multi-frame transitions are preserved.
    arr2 = np.asarray([6, 6, 1, 1, 1], dtype=np.int64)
    out2 = _median3_smooth(arr2)
    # Index 2 is the start of a real run, surrounded by 6 and 1 → median
    # is 1, which collapses one frame of the transition (acceptable
    # cost; alternative is to keep the spike-killing benefit).
    assert out2.tolist()[0] == 6 and out2.tolist()[-1] == 1
    assert sum(v == 1 for v in out2.tolist()) >= 2

    # Edge frames replicate (never lose them to median).
    arr3 = np.asarray([1, 6, 6, 6, 6], dtype=np.int64)
    out3 = _median3_smooth(arr3)
    assert out3.tolist() == [1, 6, 6, 6, 6]


def test_default_params_match_spec() -> None:
    p = ExtractNotesV4Params(job_id="j", event_id="e", event_start_utc=0.0)
    assert p.stft.min_frequency_hz == 30.0
    assert p.stft.max_frequency_hz == 6000.0
    assert p.hps.n_harmonics == 8
    assert p.hps.cents_tolerance == 50.0
    assert p.hps.k_noise == 2.0
    assert p.hps.candidate_divisors == (1, 2, 3, 4, 5, 6)
    assert p.hps.smoothing_frames == 5
    assert p.hps.low_band_penalty == 0.5
    assert p.hps.low_band_threshold_hz == 100.0
    assert p.hps.low_band_min_harmonics == 3
    assert p.hps.high_band_min_harmonics == 2
    assert p.hps.min_above_floor == 1.0
    assert p.hps.max_harmonic_dynamic_range_log == 3.0
