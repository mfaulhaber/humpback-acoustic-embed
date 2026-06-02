"""Tests for ``humpback.processing.note_extractor_v7``."""

from __future__ import annotations

import math

import numpy as np

from humpback.processing.note_extractor_v3 import STFTParams, _RefinedFrame
from humpback.processing.note_extractor_v6 import (
    DespikeParams,
    ExtractNotesV6Params,
    extract_notes_v6,
)
from humpback.processing.note_extractor_v7 import (
    DiscontinuityParams,
    ExtractNotesV7Params,
    RidgeRescueParams,
    extract_notes_v7,
    rescue_flat_segments_from_ridge,
    split_residual_discontinuities,
)
from humpback.processing.piano_roll_cqt import CQTParams

SAMPLE_RATE = 22050
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


def test_residual_discontinuity_splits_on_adjacent_slope() -> None:
    segments = [_mk([6.0, 6.0, 6.0, 7.2, 7.2, 7.2])]
    out = split_residual_discontinuities(
        segments,
        params=DiscontinuityParams(enabled=True, max_continuous_slope_oct_per_s=6.0),
        min_note_frames=3,
    )
    assert [[f.log_frequency for f in segment] for segment in out] == [
        [6.0, 6.0, 6.0],
        [7.2, 7.2, 7.2],
    ]


def test_residual_discontinuity_disabled_is_noop() -> None:
    segments = [_mk([6.0, 6.0, 8.0, 8.0])]
    out = split_residual_discontinuities(
        segments,
        params=DiscontinuityParams(enabled=False),
        min_note_frames=3,
    )
    assert out == segments


def test_ridge_rescue_replaces_flat_decode_with_smooth_ridge() -> None:
    base = math.log2(200.0)
    segment = _mk([base] * 10)
    times = np.arange(10, dtype=np.float64) * _DT
    ridge_hz = np.geomspace(800.0, 1200.0, num=10)
    rows = [
        {
            "frame_index": i,
            "frame_time_offset_s": float(t),
            "log_frequency": float(math.log2(hz)),
            "strength": 1.0,
            "energy_ratio": 1.0,
        }
        for i, (t, hz) in enumerate(zip(times, ridge_hz))
    ]

    out = rescue_flat_segments_from_ridge(
        [segment],
        ridge_sidecar_rows=rows,
        pad_seconds=0.0,
        params=RidgeRescueParams(
            enabled=True,
            min_overlap_frames=8,
            max_decoded_span_semitones=2.0,
            min_ridge_span_semitones=5.0,
            max_ratio_mad_semitones=0.25,
        ),
    )

    rescued = out[0]
    rescued_hz = np.asarray([2.0**f.log_frequency for f in rescued])
    assert _span_semitones([f.log_frequency for f in rescued]) > 5.0
    assert math.isclose(float(np.median(rescued_hz)), 200.0, rel_tol=0.05)
    assert rescued_hz[-1] > rescued_hz[0]


def test_ridge_rescue_rejects_jagged_ridge() -> None:
    base = math.log2(200.0)
    segment = _mk([base] * 10)
    rows = [
        {
            "frame_index": i,
            "frame_time_offset_s": i * _DT,
            "log_frequency": math.log2(800.0) + (0.0 if i % 2 == 0 else 0.5),
            "strength": 1.0,
            "energy_ratio": 1.0,
        }
        for i in range(10)
    ]

    out = rescue_flat_segments_from_ridge(
        [segment],
        ridge_sidecar_rows=rows,
        pad_seconds=0.0,
        params=RidgeRescueParams(
            enabled=True,
            min_overlap_frames=8,
            max_decoded_span_semitones=2.0,
            min_ridge_span_semitones=5.0,
            max_ratio_mad_semitones=1.0,
        ),
    )

    assert out[0] == segment


def test_disabled_v7_passes_reproduce_v6() -> None:
    audio = _harmonic_stack(
        200.0, duration_s=0.40, harmonics=[1, 2, 3, 4], amplitudes=[0.4, 0.3, 0.2, 0.1]
    )
    v6 = extract_notes_v6(
        audio,
        SAMPLE_RATE,
        params=ExtractNotesV6Params(
            job_id="job-v7-test",
            event_id="ev-1",
            event_start_utc=1000.0,
            pad_seconds=0.0,
            cqt=CQTParams(),
            stft=STFTParams(min_frequency_hz=30.0),
            despike=DespikeParams(enabled=False),
        ),
    )
    v7 = extract_notes_v7(
        audio,
        SAMPLE_RATE,
        params=ExtractNotesV7Params(
            job_id="job-v7-test",
            event_id="ev-1",
            event_start_utc=1000.0,
            pad_seconds=0.0,
            cqt=CQTParams(),
            stft=STFTParams(min_frequency_hz=30.0),
            despike=DespikeParams(enabled=False),
            discontinuity=DiscontinuityParams(enabled=False),
            ridge_rescue=RidgeRescueParams(enabled=False),
        ),
    )
    assert v7.notes == v6.notes
    assert v7.contours == v6.contours


def _span_semitones(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float((np.max(arr) - np.min(arr)) * 12.0)


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
