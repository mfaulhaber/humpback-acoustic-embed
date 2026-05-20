"""Tests for Piano Roll Notes CQT and peak-pick helpers."""

from __future__ import annotations

import numpy as np
import pytest

from humpback.processing.piano_roll_cqt import (
    CQTParams,
    PeakParams,
    bin_frequency_hz,
    compute_event_cqt,
    midi_pitch_for_bin,
    pick_peaks_per_frame,
)


def _sine(freq_hz: float, duration_s: float, sr: int) -> np.ndarray:
    t = np.arange(int(duration_s * sr)) / sr
    return (0.5 * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def test_cqt_resamples_to_target_sample_rate() -> None:
    sr_in = 16000
    audio = _sine(440.0, 1.0, sr_in)
    log_mag = compute_event_cqt(audio, sr_in)
    assert log_mag.dtype == np.float32
    assert log_mag.shape[0] == 264
    assert log_mag.shape[1] > 0


def test_cqt_a4_lands_in_expected_bin() -> None:
    sr = 22050
    audio = _sine(440.0, 1.0, sr)
    log_mag = compute_event_cqt(audio, sr)
    # A4 = MIDI 69; with fmin=A0 (MIDI 21) and 3 bins/semitone,
    # A4 -> bin (69 - 21) * 3 = 144.
    midframe = log_mag[:, log_mag.shape[1] // 2]
    peak_bin = int(np.argmax(midframe))
    assert abs(peak_bin - 144) <= 1


def test_cqt_zero_length_returns_zero_frames() -> None:
    log_mag = compute_event_cqt(np.zeros(0, dtype=np.float32), 22050)
    assert log_mag.shape == (264, 0)


def test_cqt_downmixes_multichannel() -> None:
    sr = 22050
    mono = _sine(440.0, 0.5, sr)
    stereo = np.stack([mono, mono], axis=1)  # shape (N, 2)
    log_mag = compute_event_cqt(stereo, sr)
    assert log_mag.shape[0] == 264


def test_pick_peaks_finds_sinusoid_peak() -> None:
    sr = 22050
    audio = _sine(440.0, 0.5, sr)
    log_mag = compute_event_cqt(audio, sr)
    peaks = pick_peaks_per_frame(log_mag)
    non_empty = [frame for frame in peaks if frame]
    assert non_empty, "expected at least one frame with peaks for a 440 Hz tone"
    sample = non_empty[len(non_empty) // 2]
    # Strongest peak in this frame should be around MIDI 69 (bin 144).
    top_bin, _ = sample[0]
    assert abs(top_bin - 144) <= 1


def test_pick_peaks_harmonic_stack_finds_all_partials() -> None:
    sr = 22050
    fundamental = 220.0  # A3 = MIDI 57 -> bin (57-21)*3 = 108
    t = np.arange(int(0.5 * sr)) / sr
    audio = np.zeros_like(t, dtype=np.float32)
    for k in (1, 2, 3, 4):
        audio += (1.0 / k) * np.sin(2 * np.pi * fundamental * k * t)
    log_mag = compute_event_cqt(audio, sr)
    peaks = pick_peaks_per_frame(log_mag, params=PeakParams(top_k=8))
    midframe = peaks[len(peaks) // 2]
    bins = sorted(b for b, _ in midframe)
    expected_bins = {108, 144, 165, 180}
    found_close = sum(
        1 for expected in expected_bins if any(abs(b - expected) <= 1 for b in bins)
    )
    assert found_close == len(expected_bins), (
        f"expected partials near bins {expected_bins}, got {bins}"
    )


def test_pick_peaks_returns_strongest_peak_in_clean_signal() -> None:
    """End-to-end: a pure 440 Hz tone produces one consistent peak bin."""
    sr = 22050
    audio = _sine(440.0, 0.5, sr)
    log_mag = compute_event_cqt(audio, sr)
    peaks = pick_peaks_per_frame(log_mag)
    # In the middle 60% of frames, the strongest peak should be at A4.
    n_frames = log_mag.shape[1]
    head_skip = n_frames // 5
    tail_skip = n_frames // 5
    samples = peaks[head_skip : n_frames - tail_skip]
    a4_count = 0
    for frame in samples:
        if frame and abs(frame[0][0] - 144) <= 1:
            a4_count += 1
    assert a4_count >= len(samples) // 2, (
        f"expected A4 (bin 144) to dominate; got {a4_count}/{len(samples)}"
    )


def test_midi_pitch_for_bin_default_grid() -> None:
    params = CQTParams()
    # bin 0 -> A0 -> MIDI 21
    assert midi_pitch_for_bin(0, params) == 21
    # bin 3 -> A#0 -> MIDI 22
    assert midi_pitch_for_bin(3, params) == 22
    # bin 144 -> A4 -> MIDI 69
    assert midi_pitch_for_bin(144, params) == 69


def test_bin_frequency_hz_matches_a4() -> None:
    params = CQTParams()
    freq = bin_frequency_hz(144, params)
    assert freq == pytest.approx(440.0, rel=1e-3)
