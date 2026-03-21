"""Stage 2: Build guard-band exclusion map from annotations."""

from __future__ import annotations

from humpback.sample_builder.types import (
    ExclusionMap,
    NormalizedAnnotation,
    ProtectedInterval,
)


def build_exclusion_map(
    annotations: list[NormalizedAnnotation],
    *,
    guard_band_sec: float = 1.0,
) -> ExclusionMap:
    """Build an exclusion map by expanding valid annotations with guard bands.

    Each valid annotation is expanded by ``guard_band_sec`` on both sides.
    Overlapping intervals are merged so downstream code can do simple
    gap-based searches.

    Parameters
    ----------
    annotations:
        Normalized annotations (only valid ones are included).
    guard_band_sec:
        Seconds to add before and after each annotation boundary.

    Returns
    -------
    An ExclusionMap with merged, non-overlapping protected intervals.
    """
    # Collect raw intervals from valid annotations only
    raw: list[tuple[float, float, int]] = []
    for idx, ann in enumerate(annotations):
        if not ann.valid:
            continue
        start = max(0.0, ann.original.begin_time - guard_band_sec)
        end = ann.original.end_time + guard_band_sec
        raw.append((start, end, idx))

    if not raw:
        return ExclusionMap(protected_intervals=[])

    # Sort by start time, then merge overlapping intervals
    raw.sort(key=lambda x: x[0])
    merged: list[ProtectedInterval] = []
    cur_start, cur_end, cur_idx = raw[0]

    for start, end, idx in raw[1:]:
        if start <= cur_end:
            # Overlapping — extend; keep the first annotation index
            cur_end = max(cur_end, end)
        else:
            merged.append(ProtectedInterval(cur_start, cur_end, cur_idx))
            cur_start, cur_end, cur_idx = start, end, idx

    merged.append(ProtectedInterval(cur_start, cur_end, cur_idx))
    return ExclusionMap(protected_intervals=merged)
