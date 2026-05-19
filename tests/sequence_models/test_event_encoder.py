"""Tests for CRNN event vector construction utilities."""

import numpy as np
import pytest

from humpback.sequence_models.event_encoder import (
    ChunkEmbedding,
    DESCRIPTOR_ORDER,
    EventInterval,
    _RidgePathResult,
    _compute_ridge_summary_descriptors,
    build_event_embedding,
    compute_acoustic_descriptors,
    compute_gap_to_previous,
    descriptor_vector,
    interval_overlap,
    select_event_chunks,
)


def _chunk(
    start: float,
    end: float,
    value: float,
    *,
    region_id: str = "r1",
    call_probability: float = 0.5,
) -> ChunkEmbedding:
    return ChunkEmbedding(
        region_id=region_id,
        start_timestamp=start,
        end_timestamp=end,
        call_probability=call_probability,
        embedding=np.asarray([value, value + 1.0], dtype=np.float32),
    )


def test_interval_overlap_uses_half_open_intervals():
    assert interval_overlap(0.0, 1.0, 0.5, 1.5) == 0.5
    assert interval_overlap(0.0, 1.0, 1.0, 2.0) == 0.0
    assert interval_overlap(0.0, 2.0, 0.5, 1.5) == 1.0


def test_select_event_chunks_recomputes_positive_overlap_and_coverage():
    event = EventInterval("e1", "r1", 0.0, 1.0)
    chunks = [
        _chunk(-0.25, 0.25, 1.0),
        _chunk(0.25, 0.75, 2.0),
        _chunk(1.0, 1.25, 3.0),
        _chunk(0.0, 1.0, 99.0, region_id="other"),
    ]

    selected, coverage, skip_reason = select_event_chunks(
        event,
        chunks,
        min_overlap_fraction=0.5,
    )

    assert skip_reason is None
    assert len(selected) == 2
    assert coverage == pytest.approx(0.75)


def test_select_event_chunks_reports_skip_reason_for_low_coverage():
    event = EventInterval("e1", "r1", 0.0, 10.0)
    selected, coverage, skip_reason = select_event_chunks(
        event,
        [_chunk(0.0, 1.0, 1.0)],
        min_overlap_fraction=0.5,
        min_chunks_per_event=2,
    )

    assert len(selected) == 1
    assert coverage == pytest.approx(0.1)
    assert skip_reason == "insufficient_chunk_coverage"


def test_build_event_embedding_emits_all_pools_with_short_event_fallback():
    event = EventInterval("e1", "r1", 0.0, 0.5)
    chunks = [_chunk(0.0, 0.5, 1.0, call_probability=0.2)]

    result = build_event_embedding(event, chunks)

    assert result.chunk_count == 1
    assert result.coverage_fraction == pytest.approx(1.0)
    assert result.pool_vector.shape == (10,)
    for pool in result.pools.values():
        assert np.allclose(pool, [1.0, 2.0])


def test_build_event_embedding_top_k_uses_high_call_probability():
    event = EventInterval("e1", "r1", 0.0, 1.0)
    chunks = [
        _chunk(0.0, 0.5, 1.0, call_probability=0.1),
        _chunk(0.5, 1.0, 9.0, call_probability=0.9),
    ]

    result = build_event_embedding(event, chunks, top_k_fraction=0.5)

    assert np.allclose(result.pools["mean_pool"], [5.0, 6.0])
    assert np.allclose(result.pools["top_k_pool"], [9.0, 10.0])


def test_compute_gap_to_previous_is_per_source_sequence():
    events = [
        EventInterval("e2", "r1", 4.0, 5.0, source_sequence_key="a"),
        EventInterval("e1", "r1", 1.0, 2.0, source_sequence_key="a"),
        EventInterval("e3", "r1", 1.5, 2.0, source_sequence_key="b"),
    ]

    gaps = compute_gap_to_previous(events)

    assert gaps == {"e1": 0.0, "e2": 2.0, "e3": 0.0}


def test_acoustic_descriptors_identify_sine_peak_frequency():
    sample_rate = 16000
    t = np.arange(sample_rate, dtype=np.float32) / sample_rate
    audio = np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)

    descriptors = compute_acoustic_descriptors(
        audio,
        sample_rate=sample_rate,
        gap_to_previous=1.25,
        n_fft=1024,
        hop_length=512,
    )
    vec = descriptor_vector(descriptors)

    assert descriptors["duration"] == pytest.approx(1.0)
    assert descriptors["peak_frequency"] == pytest.approx(437.5, abs=16.0)
    assert descriptors["spectral_centroid"] > 0
    assert descriptors["bandwidth"] > 0
    assert 0 <= descriptors["spectral_entropy"] <= 1
    assert descriptors["ridge_log_frequency_slope"] == pytest.approx(0.0, abs=0.1)
    assert descriptors["ridge_median_frequency"] == pytest.approx(437.5, abs=16.0)
    assert descriptors["ridge_low_frequency"] == pytest.approx(437.5, abs=16.0)
    assert descriptors["ridge_high_frequency"] == pytest.approx(437.5, abs=16.0)
    assert descriptors["ridge_frequency_span"] == pytest.approx(0.0, abs=16.0)
    assert descriptors["ridge_coverage"] > 0.9
    assert descriptors["ridge_energy_ratio"] > 0.4
    assert descriptors["band_limited_peak_frequency"] == pytest.approx(437.5, abs=16.0)
    assert descriptors["high_band_energy_ratio"] < 0.1
    assert descriptors["gap_to_previous"] == pytest.approx(1.25)
    assert vec.shape == (22,)
    assert "frequency_slope" not in descriptors


def test_f0_descriptors_track_steady_sine():
    sample_rate = 16000
    t = np.arange(sample_rate, dtype=np.float32) / sample_rate
    audio = np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)

    descriptors = compute_acoustic_descriptors(
        audio,
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=512,
    )

    assert descriptors["median_f0"] == pytest.approx(440.0, abs=5.0)
    assert descriptors["f0_range"] == pytest.approx(0.0, abs=20.0)
    assert descriptors["voicing_fraction"] == pytest.approx(1.0, abs=0.1)


def test_f0_descriptors_track_log_chirp_range():
    sample_rate = 16000
    audio = _log_chirp(
        sample_rate=sample_rate,
        duration=1.0,
        start_hz=300.0,
        end_hz=1200.0,
    )

    descriptors = compute_acoustic_descriptors(
        audio,
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=256,
    )

    assert 300.0 <= descriptors["median_f0"] <= 1200.0
    assert descriptors["f0_range"] > 100.0
    assert descriptors["voicing_fraction"] > 0.5


def test_f0_descriptors_return_zero_for_empty_and_silent_audio():
    empty = compute_acoustic_descriptors(
        np.asarray([], dtype=np.float32), sample_rate=16000
    )
    silent = compute_acoustic_descriptors(
        np.zeros(16000, dtype=np.float32), sample_rate=16000
    )

    for descriptors in (empty, silent):
        assert descriptors["median_f0"] == 0.0
        assert descriptors["f0_range"] == 0.0
        assert descriptors["voicing_fraction"] == 0.0


def test_pulse_descriptors_track_amplitude_modulation_rate():
    sample_rate = 16000
    audio = _am_tone(
        sample_rate=sample_rate,
        duration=1.0,
        carrier_hz=500.0,
        pulse_rate_hz=20.0,
    )

    descriptors = compute_acoustic_descriptors(
        audio,
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=512,
    )

    assert descriptors["pulse_rate"] == pytest.approx(20.0, abs=1.0)
    assert abs(descriptors["pulse_rate_slope"]) < 5.0


def test_pulse_descriptors_return_zero_for_smooth_tone():
    sample_rate = 16000
    t = np.arange(sample_rate, dtype=np.float32) / sample_rate
    audio = np.sin(2.0 * np.pi * 440.0 * t).astype(np.float32)

    descriptors = compute_acoustic_descriptors(
        audio,
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=512,
    )

    assert descriptors["pulse_rate"] == 0.0
    assert descriptors["pulse_rate_slope"] == 0.0


def test_pulse_rate_slope_tracks_decaying_modulation_rate():
    sample_rate = 16000
    duration = 2.0
    t = np.arange(int(sample_rate * duration), dtype=np.float64) / sample_rate
    start_rate = 40.0
    rate_slope = -15.0
    modulation_phase = 2.0 * np.pi * (start_rate * t + 0.5 * rate_slope * np.square(t))
    envelope = 1.0 + 0.75 * np.sin(modulation_phase)
    audio = (envelope * np.sin(2.0 * np.pi * 500.0 * t)).astype(np.float32)

    descriptors = compute_acoustic_descriptors(
        audio,
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=512,
        pulse_min_rate_hz=5.0,
        pulse_max_rate_hz=80.0,
        pulse_confidence_threshold=0.1,
    )

    assert descriptors["pulse_rate"] > 0.0
    assert descriptors["pulse_rate_slope"] < 0.0


def test_pulse_descriptors_return_zero_for_degenerate_audio():
    sample_rate = 16000
    empty = compute_acoustic_descriptors(
        np.asarray([], dtype=np.float32), sample_rate=sample_rate
    )
    silent = compute_acoustic_descriptors(
        np.zeros(sample_rate, dtype=np.float32), sample_rate=sample_rate
    )
    very_short = compute_acoustic_descriptors(
        _am_tone(
            sample_rate=sample_rate,
            duration=0.01,
            carrier_hz=500.0,
            pulse_rate_hz=20.0,
        ),
        sample_rate=sample_rate,
    )

    for descriptors in (empty, silent, very_short):
        assert descriptors["pulse_rate"] == 0.0
        assert descriptors["pulse_rate_slope"] == 0.0


def test_ridge_log_frequency_slope_tracks_log_chirp():
    sample_rate = 16000
    duration = 1.0
    audio = _log_chirp(
        sample_rate=sample_rate,
        duration=duration,
        start_hz=300.0,
        end_hz=1200.0,
    )

    descriptors = compute_acoustic_descriptors(
        audio,
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=256,
        ridge_min_frequency_hz=100.0,
        ridge_max_frequency_hz=3000.0,
    )

    expected = np.log2(1200.0 / 300.0) / duration
    assert descriptors["ridge_log_frequency_slope"] == pytest.approx(
        expected,
        abs=0.5,
    )
    assert 300.0 <= descriptors["ridge_median_frequency"] <= 1200.0
    assert (
        250.0
        <= descriptors["ridge_low_frequency"]
        <= descriptors["ridge_median_frequency"]
    )
    assert descriptors["ridge_high_frequency"] >= descriptors["ridge_median_frequency"]
    assert descriptors["ridge_frequency_span"] > 200.0


def test_ridge_summary_tracks_high_frequency_chirp():
    sample_rate = 16000
    audio = _log_chirp(
        sample_rate=sample_rate,
        duration=1.0,
        start_hz=2200.0,
        end_hz=4200.0,
    )

    descriptors = compute_acoustic_descriptors(
        audio,
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=256,
        ridge_max_frequency_hz=6000.0,
        band_peak_min_frequency_hz=1000.0,
        high_band_min_frequency_hz=1000.0,
    )

    assert 2200.0 <= descriptors["ridge_median_frequency"] <= 4200.0
    assert (
        1800.0
        <= descriptors["ridge_low_frequency"]
        <= descriptors["ridge_median_frequency"]
    )
    assert descriptors["ridge_high_frequency"] >= descriptors["ridge_median_frequency"]
    assert descriptors["ridge_frequency_span"] > 500.0
    assert descriptors["ridge_coverage"] > 0.8
    assert descriptors["ridge_energy_ratio"] > 0.2
    assert descriptors["band_limited_peak_frequency"] >= 2000.0
    assert descriptors["high_band_energy_ratio"] > 0.9


def test_band_limited_peak_resists_low_frequency_rumble():
    sample_rate = 16000
    t = np.arange(sample_rate, dtype=np.float32) / sample_rate
    rumble = np.sin(2.0 * np.pi * 62.5 * t)
    whistle = 0.35 * np.sin(2.0 * np.pi * 2600.0 * t)
    audio = (rumble + whistle).astype(np.float32)

    descriptors = compute_acoustic_descriptors(
        audio,
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=512,
        ridge_min_frequency_hz=500.0,
        ridge_max_frequency_hz=6000.0,
        band_peak_min_frequency_hz=500.0,
    )

    assert descriptors["peak_frequency"] < 100.0
    assert descriptors["band_limited_peak_frequency"] == pytest.approx(
        2600.0,
        abs=20.0,
    )
    assert descriptors["ridge_median_frequency"] == pytest.approx(
        2600.0,
        abs=30.0,
    )


def test_ridge_summary_trimmed_bounds_resist_single_outlier_frame():
    result = _RidgePathResult(
        log_frequencies=np.log2(np.asarray([1000.0] * 9 + [6000.0], dtype=np.float64)),
        frame_times=np.arange(10, dtype=np.float64),
        strengths=np.ones(10, dtype=np.float64),
        energy_ratios=np.ones(10, dtype=np.float64),
        total_frames=10,
    )

    descriptors = _compute_ridge_summary_descriptors(
        result,
        low_percentile=10.0,
        high_percentile=90.0,
    )

    assert descriptors["ridge_median_frequency"] == pytest.approx(1000.0)
    assert descriptors["ridge_low_frequency"] == pytest.approx(1000.0)
    assert descriptors["ridge_high_frequency"] < 6000.0
    assert descriptors["ridge_frequency_span"] < 1000.0


def test_ridge_log_frequency_slope_is_stable_across_harmonics():
    sample_rate = 16000
    fundamental = _log_chirp(
        sample_rate=sample_rate,
        duration=1.0,
        start_hz=250.0,
        end_hz=900.0,
    )
    harmonic = _log_chirp(
        sample_rate=sample_rate,
        duration=1.0,
        start_hz=500.0,
        end_hz=1800.0,
    )

    fundamental_descriptors = compute_acoustic_descriptors(
        fundamental,
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=256,
    )
    harmonic_descriptors = compute_acoustic_descriptors(
        harmonic,
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=256,
    )

    assert harmonic_descriptors["ridge_log_frequency_slope"] == pytest.approx(
        fundamental_descriptors["ridge_log_frequency_slope"],
        abs=0.35,
    )


def test_inflection_count_is_zero_for_ascending_log_chirp():
    sample_rate = 16000
    audio = _log_chirp(
        sample_rate=sample_rate,
        duration=1.0,
        start_hz=300.0,
        end_hz=1200.0,
    )

    descriptors = compute_acoustic_descriptors(
        audio,
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=256,
    )

    assert descriptors["inflection_count"] == 0.0


def test_inflection_count_counts_up_then_down_chirp():
    sample_rate = 16000
    audio = np.concatenate(
        [
            _log_chirp(
                sample_rate=sample_rate,
                duration=0.5,
                start_hz=300.0,
                end_hz=1200.0,
            ),
            _log_chirp(
                sample_rate=sample_rate,
                duration=0.5,
                start_hz=1200.0,
                end_hz=300.0,
            ),
        ]
    )

    descriptors = compute_acoustic_descriptors(
        audio,
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=256,
    )

    assert descriptors["inflection_count"] == pytest.approx(np.log(2.0))


def test_ridge_log_frequency_slope_returns_finite_zero_for_degenerate_inputs():
    empty = compute_acoustic_descriptors(
        np.asarray([], dtype=np.float32), sample_rate=16000
    )
    silent = compute_acoustic_descriptors(
        np.zeros(128, dtype=np.float32), sample_rate=16000
    )
    invalid_band = compute_acoustic_descriptors(
        np.ones(2048, dtype=np.float32),
        sample_rate=16000,
        ridge_min_frequency_hz=9000.0,
        ridge_max_frequency_hz=10000.0,
    )

    for descriptors in (empty, silent, invalid_band):
        assert np.isfinite(descriptors["ridge_log_frequency_slope"])
        assert descriptors["ridge_log_frequency_slope"] == 0.0
        assert np.isfinite(descriptors["inflection_count"])
        assert descriptors["inflection_count"] == 0.0
        assert descriptors["ridge_median_frequency"] == 0.0
        assert descriptors["ridge_low_frequency"] == 0.0
        assert descriptors["ridge_high_frequency"] == 0.0
        assert descriptors["ridge_frequency_span"] == 0.0
        assert descriptors["ridge_coverage"] == 0.0
        assert descriptors["ridge_energy_ratio"] == 0.0
        assert descriptor_vector(descriptors).shape == (len(DESCRIPTOR_ORDER),)
    for descriptors in (empty, silent):
        assert descriptors["band_limited_peak_frequency"] == 0.0
        assert descriptors["high_band_energy_ratio"] == 0.0


def _log_chirp(
    *,
    sample_rate: int,
    duration: float,
    start_hz: float,
    end_hz: float,
) -> np.ndarray:
    t = np.arange(int(sample_rate * duration), dtype=np.float64) / sample_rate
    slope = np.log(end_hz / start_hz) / duration
    phase = 2.0 * np.pi * start_hz * (np.exp(slope * t) - 1.0) / slope
    return np.sin(phase).astype(np.float32)


def _am_tone(
    *,
    sample_rate: int,
    duration: float,
    carrier_hz: float,
    pulse_rate_hz: float,
) -> np.ndarray:
    t = np.arange(int(sample_rate * duration), dtype=np.float64) / sample_rate
    envelope = 1.0 + 0.75 * np.sin(2.0 * np.pi * pulse_rate_hz * t)
    carrier = np.sin(2.0 * np.pi * carrier_hz * t)
    return (envelope * carrier).astype(np.float32)
