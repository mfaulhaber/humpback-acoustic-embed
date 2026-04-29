"""Vectorized join from chunk timestamps to Pass 2 events.

Given a region's chunk timestamps and the events Pass 2 produced for
that same region, computes per-chunk:

- ``event_overlap_fraction`` — overlap with the union of all events in
  this region, divided by chunk duration.
- ``nearest_event_id`` — id of the event closest in time, or ``None``
  when no event lies within ``near_event_window_seconds``.
- ``distance_to_nearest_event_seconds`` — signed seconds (positive if
  the chunk is after the event, negative if before, 0.0 inside),
  ``None`` for background tier rows.
- ``tier`` ∈ ``{"event_core", "near_event", "background"}`` — derived
  from the configured thresholds.

This is the producer-time join the spec calls out in §5; downstream
consumers read ``tier`` rather than re-running overlap math.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EVENT_CORE = "event_core"
NEAR_EVENT = "near_event"
BACKGROUND = "background"


@dataclass(frozen=True)
class ChunkBounds:
    """Per-chunk start/end timestamps (any consistent timeline)."""

    starts: np.ndarray  # (T_chunks,) float64
    ends: np.ndarray  # (T_chunks,) float64


@dataclass(frozen=True)
class EventBounds:
    """Per-event timestamps and ids for the same region as ``ChunkBounds``."""

    ids: np.ndarray  # (N_events,) object (string ids); empty array OK
    starts: np.ndarray  # (N_events,) float64
    ends: np.ndarray  # (N_events,) float64


@dataclass(frozen=True)
class ChunkEventMetadata:
    """Output of ``compute_chunk_event_metadata``.

    ``nearest_event_id`` and ``distance_to_nearest_event_seconds`` use
    ``None`` (object dtype) for background-tier rows so parquet writers
    can serialize them as nullable columns.
    """

    event_overlap_fraction: np.ndarray  # (T_chunks,) float32
    nearest_event_id: np.ndarray  # (T_chunks,) object, may contain None
    distance_to_nearest_event_seconds: np.ndarray  # (T_chunks,) object
    tier: np.ndarray  # (T_chunks,) object


def _union_overlap_per_chunk(chunks: ChunkBounds, events: EventBounds) -> np.ndarray:
    """Length of ``chunk_i ∩ union(events)`` for each chunk.

    Uses the standard ``[start, end]`` interval-union projection:
    sort merged events, sum overlap with each chunk against the merged
    intervals. Vectorized via numpy broadcasting.
    """
    n_chunks = chunks.starts.shape[0]
    if events.starts.size == 0:
        return np.zeros(n_chunks, dtype=np.float64)

    # Merge overlapping events into disjoint intervals so the per-chunk
    # overlap sum doesn't double-count adjacent events.
    order = np.argsort(events.starts, kind="stable")
    starts_sorted = events.starts[order]
    ends_sorted = events.ends[order]

    merged_starts = [float(starts_sorted[0])]
    merged_ends = [float(ends_sorted[0])]
    for s, e in zip(starts_sorted[1:], ends_sorted[1:]):
        if s <= merged_ends[-1]:
            if e > merged_ends[-1]:
                merged_ends[-1] = float(e)
        else:
            merged_starts.append(float(s))
            merged_ends.append(float(e))

    ms = np.asarray(merged_starts, dtype=np.float64)[None, :]
    me = np.asarray(merged_ends, dtype=np.float64)[None, :]
    cs = chunks.starts[:, None]
    ce = chunks.ends[:, None]

    overlap = np.minimum(ce, me) - np.maximum(cs, ms)
    overlap = np.clip(overlap, 0.0, None)
    return overlap.sum(axis=1)


def _nearest_event(
    chunks: ChunkBounds, events: EventBounds
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(event_index, signed_distance_seconds)`` for each chunk.

    Distance is 0 if the chunk overlaps the event; otherwise it is the
    minimum gap between intervals (signed: positive when the chunk lies
    after the event, negative when before).
    """
    n_chunks = chunks.starts.shape[0]
    if events.starts.size == 0:
        return (
            np.full(n_chunks, -1, dtype=np.int64),
            np.full(n_chunks, np.inf, dtype=np.float64),
        )

    cs = chunks.starts[:, None]
    ce = chunks.ends[:, None]
    es = events.starts[None, :]
    ee = events.ends[None, :]

    # Negative gap when the chunk is before the event start, positive
    # gap when the chunk is after the event end, zero when overlapping.
    before = es - ce  # > 0 when chunk lies fully before event
    after = cs - ee  # > 0 when chunk lies fully after event
    overlap_mask = (before <= 0) & (after <= 0)

    abs_distance = np.maximum(before, after)
    abs_distance = np.where(overlap_mask, 0.0, abs_distance)

    signed_distance = np.where(after > 0, after, -np.maximum(before, 0.0))
    signed_distance = np.where(overlap_mask, 0.0, signed_distance)

    nearest_idx = np.argmin(abs_distance, axis=1)
    chosen_signed = np.take_along_axis(
        signed_distance, nearest_idx[:, None], axis=1
    ).squeeze(1)
    chosen_abs = np.take_along_axis(abs_distance, nearest_idx[:, None], axis=1).squeeze(
        1
    )

    # Replace ``inf``-only rows (no events at all) with ``-1`` so callers
    # know nothing was matched.
    no_event_row = ~np.isfinite(chosen_abs)
    nearest_idx = np.where(no_event_row, -1, nearest_idx)
    chosen_signed = np.where(no_event_row, np.inf, chosen_signed)

    return nearest_idx.astype(np.int64), chosen_signed.astype(np.float64)


def compute_chunk_event_metadata(
    chunks: ChunkBounds,
    events: EventBounds,
    *,
    event_core_overlap_threshold: float = 0.5,
    near_event_window_seconds: float = 5.0,
) -> ChunkEventMetadata:
    """Vectorized chunk × events metadata join (no Python row loop)."""
    if chunks.starts.shape != chunks.ends.shape:
        raise ValueError("ChunkBounds.starts and .ends must have the same shape")
    if not (events.starts.shape == events.ends.shape == events.ids.shape):
        raise ValueError("EventBounds arrays must all share the same shape")

    n_chunks = chunks.starts.shape[0]
    chunk_durations = chunks.ends - chunks.starts
    if np.any(chunk_durations <= 0):
        raise ValueError("All chunks must have positive duration")

    union_overlap = _union_overlap_per_chunk(chunks, events)
    overlap_fraction = (union_overlap / chunk_durations).astype(np.float32)
    overlap_fraction = np.clip(overlap_fraction, 0.0, 1.0)

    nearest_idx, signed_distance = _nearest_event(chunks, events)

    nearest_event_id = np.empty(n_chunks, dtype=object)
    distance_field = np.empty(n_chunks, dtype=object)
    tier = np.empty(n_chunks, dtype=object)

    abs_distance = np.abs(signed_distance)
    for i in range(n_chunks):
        if overlap_fraction[i] >= event_core_overlap_threshold:
            tier[i] = EVENT_CORE
            nearest_event_id[i] = (
                str(events.ids[nearest_idx[i]]) if nearest_idx[i] >= 0 else None
            )
            distance_field[i] = (
                float(signed_distance[i]) if nearest_idx[i] >= 0 else None
            )
        elif nearest_idx[i] >= 0 and abs_distance[i] <= near_event_window_seconds:
            tier[i] = NEAR_EVENT
            nearest_event_id[i] = str(events.ids[nearest_idx[i]])
            distance_field[i] = float(signed_distance[i])
        else:
            tier[i] = BACKGROUND
            nearest_event_id[i] = None
            distance_field[i] = None

    return ChunkEventMetadata(
        event_overlap_fraction=overlap_fraction,
        nearest_event_id=nearest_event_id,
        distance_to_nearest_event_seconds=distance_field,
        tier=tier,
    )
