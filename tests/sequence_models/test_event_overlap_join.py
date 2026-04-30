"""Tests for ``compute_chunk_event_metadata``."""

from __future__ import annotations

import numpy as np

from humpback.sequence_models.event_overlap_join import (
    BACKGROUND,
    EVENT_CORE,
    NEAR_EVENT,
    ChunkBounds,
    EventBounds,
    compute_chunk_event_metadata,
)


def _chunks(*intervals: tuple[float, float]) -> ChunkBounds:
    starts = np.asarray([s for s, _ in intervals], dtype=np.float64)
    ends = np.asarray([e for _, e in intervals], dtype=np.float64)
    return ChunkBounds(starts=starts, ends=ends)


def _events(intervals: list[tuple[str, float, float]]) -> EventBounds:
    if not intervals:
        return EventBounds(
            ids=np.empty(0, dtype=object),
            starts=np.empty(0, dtype=np.float64),
            ends=np.empty(0, dtype=np.float64),
        )
    ids = np.asarray([eid for eid, _, _ in intervals], dtype=object)
    starts = np.asarray([s for _, s, _ in intervals], dtype=np.float64)
    ends = np.asarray([e for _, _, e in intervals], dtype=np.float64)
    return EventBounds(ids=ids, starts=starts, ends=ends)


def test_chunk_fully_inside_event_is_event_core():
    chunks = _chunks((10.0, 10.25))
    events = _events([("e1", 5.0, 15.0)])
    out = compute_chunk_event_metadata(chunks, events)

    assert out.tier[0] == EVENT_CORE
    assert out.event_overlap_fraction[0] == 1.0
    assert out.distance_to_nearest_event_seconds[0] == 0.0
    assert out.nearest_event_id[0] == "e1"


def test_chunk_far_from_event_is_background():
    chunks = _chunks((100.0, 100.25))
    events = _events([("e1", 0.0, 10.0)])
    out = compute_chunk_event_metadata(chunks, events, near_event_window_seconds=5.0)

    assert out.tier[0] == BACKGROUND
    assert out.nearest_event_id[0] is None
    assert out.distance_to_nearest_event_seconds[0] is None
    assert out.event_overlap_fraction[0] == 0.0


def test_chunk_a_few_seconds_from_event_is_near_event():
    chunks = _chunks((14.0, 14.25))  # 4 s after event end
    events = _events([("e1", 0.0, 10.0)])
    out = compute_chunk_event_metadata(chunks, events, near_event_window_seconds=5.0)

    assert out.tier[0] == NEAR_EVENT
    assert out.nearest_event_id[0] == "e1"
    assert out.distance_to_nearest_event_seconds[0] is not None
    assert abs(out.distance_to_nearest_event_seconds[0] - 4.0) < 1e-9


def test_no_events_in_region_yields_all_background():
    chunks = _chunks((0.0, 0.25), (1.0, 1.25), (2.0, 2.25))
    events = _events([])
    out = compute_chunk_event_metadata(chunks, events)

    assert all(t == BACKGROUND for t in out.tier)
    assert all(eid is None for eid in out.nearest_event_id)
    assert all(d is None for d in out.distance_to_nearest_event_seconds)
    np.testing.assert_array_equal(
        out.event_overlap_fraction, np.zeros(3, dtype=np.float32)
    )


def test_partial_overlap_below_threshold_is_near_event():
    """Chunk overlaps 30% with the event (below 0.5 threshold) → near_event."""
    chunks = _chunks((9.0, 10.0))  # 1 s wide; only [9.0, 9.3] overlaps
    events = _events([("e1", 5.0, 9.3)])
    out = compute_chunk_event_metadata(chunks, events)

    assert out.tier[0] == NEAR_EVENT
    assert abs(out.event_overlap_fraction[0] - 0.3) < 1e-6
    assert out.nearest_event_id[0] == "e1"


def test_partial_overlap_above_threshold_is_event_core():
    chunks = _chunks((5.0, 6.0))  # 1 s wide; [5.0, 5.7] overlaps
    events = _events([("e1", 0.0, 5.7)])
    out = compute_chunk_event_metadata(chunks, events)

    assert out.tier[0] == EVENT_CORE
    assert abs(out.event_overlap_fraction[0] - 0.7) < 1e-6


def test_signed_distance_is_negative_when_chunk_precedes_event():
    chunks = _chunks((0.0, 0.25))
    events = _events([("e1", 3.0, 5.0)])
    out = compute_chunk_event_metadata(chunks, events, near_event_window_seconds=5.0)

    assert out.tier[0] == NEAR_EVENT
    assert out.distance_to_nearest_event_seconds[0] < 0
    assert abs(out.distance_to_nearest_event_seconds[0] - (-2.75)) < 1e-9


def test_overlap_with_two_overlapping_events_uses_union():
    """Two events ``[5,10]`` and ``[8,12]`` cover ``[5,12]`` as a union;
    a chunk ``[6,11]`` overlaps the union by 5 s out of 5 s = 1.0."""
    chunks = _chunks((6.0, 11.0))
    events = _events([("e1", 5.0, 10.0), ("e2", 8.0, 12.0)])
    out = compute_chunk_event_metadata(chunks, events)

    assert out.tier[0] == EVENT_CORE
    assert abs(out.event_overlap_fraction[0] - 1.0) < 1e-6


def test_picks_closest_event_when_multiple_in_region():
    chunks = _chunks((20.0, 20.25))
    events = _events([("e1", 0.0, 5.0), ("e2", 17.0, 19.0), ("e3", 50.0, 60.0)])
    out = compute_chunk_event_metadata(chunks, events, near_event_window_seconds=5.0)

    assert out.nearest_event_id[0] == "e2"
    assert abs(out.distance_to_nearest_event_seconds[0] - 1.0) < 1e-9


def test_vectorization_returns_per_chunk_arrays():
    chunks = _chunks((0.0, 0.25), (5.0, 5.25), (50.0, 50.25))
    events = _events([("e1", 4.0, 6.0)])
    out = compute_chunk_event_metadata(chunks, events, near_event_window_seconds=5.0)

    assert out.tier[0] == NEAR_EVENT  # chunk 4 s before event
    assert out.tier[1] == EVENT_CORE  # chunk overlaps event
    assert out.tier[2] == BACKGROUND  # chunk far from event
    assert out.event_overlap_fraction.shape == (3,)
    assert out.nearest_event_id.shape == (3,)
