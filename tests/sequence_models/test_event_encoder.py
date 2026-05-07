"""Tests for CRNN event vector construction utilities."""

import numpy as np
import pytest

from humpback.sequence_models.event_encoder import (
    ChunkEmbedding,
    EventInterval,
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
    assert descriptors["gap_to_previous"] == pytest.approx(1.25)
    assert vec.shape == (8,)
