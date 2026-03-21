"""Stage 3: Discover candidate background fragments outside protected intervals."""

from __future__ import annotations

from humpback.sample_builder.types import (
    BackgroundFragment,
    ExclusionMap,
    NormalizedAnnotation,
)


def discover_background_fragments(
    target: NormalizedAnnotation,
    exclusion_map: ExclusionMap,
    audio_duration_sec: float,
    *,
    min_fragment_sec: float = 0.5,
    max_search_radius_sec: float = 60.0,
) -> list[BackgroundFragment]:
    """Find candidate background fragments near a target annotation.

    Searches outward from the target midpoint, splitting the gaps between
    protected intervals into fragments. Fragments shorter than
    ``min_fragment_sec`` are discarded.

    Parameters
    ----------
    target:
        The annotation to build a sample around.
    exclusion_map:
        Merged protected intervals covering all annotations + guard bands.
    audio_duration_sec:
        Total duration of the recording in seconds.
    min_fragment_sec:
        Minimum usable fragment duration in seconds.
    max_search_radius_sec:
        Maximum distance from target midpoint to search in either direction.

    Returns
    -------
    List of BackgroundFragment sorted by distance from target midpoint
    (closest first).
    """
    midpoint = target.midpoint_sec
    search_start = max(0.0, midpoint - max_search_radius_sec)
    search_end = min(audio_duration_sec, midpoint + max_search_radius_sec)

    # Build list of gaps (unprotected intervals) within the search window
    gaps = _find_gaps(exclusion_map, search_start, search_end)

    # Convert gaps into fragments, computing distance from target
    fragments: list[BackgroundFragment] = []
    for gap_start, gap_end in gaps:
        duration = gap_end - gap_start
        if duration < min_fragment_sec:
            continue

        # Distance = minimum distance from any point in the gap to the midpoint
        if midpoint < gap_start:
            distance = gap_start - midpoint
        elif midpoint > gap_end:
            distance = midpoint - gap_end
        else:
            distance = 0.0

        fragments.append(
            BackgroundFragment(
                start_sec=gap_start,
                end_sec=gap_end,
                duration_sec=duration,
                distance_from_target=distance,
            )
        )

    # Sort by distance from target (closest first)
    fragments.sort(key=lambda f: f.distance_from_target)
    return fragments


def _find_gaps(
    exclusion_map: ExclusionMap,
    search_start: float,
    search_end: float,
) -> list[tuple[float, float]]:
    """Return unprotected intervals within [search_start, search_end].

    The protected intervals in the exclusion map are assumed to be sorted
    and non-overlapping (guaranteed by ``build_exclusion_map``).
    """
    gaps: list[tuple[float, float]] = []
    cursor = search_start

    for iv in exclusion_map.protected_intervals:
        # Skip intervals entirely before our search window
        if iv.end_sec <= search_start:
            continue
        # Stop if we've passed the search window
        if iv.start_sec >= search_end:
            break

        # Gap between cursor and start of this protected interval
        gap_start = cursor
        gap_end = min(iv.start_sec, search_end)
        if gap_end > gap_start:
            gaps.append((gap_start, gap_end))

        cursor = max(cursor, iv.end_sec)

    # Trailing gap after last protected interval
    if cursor < search_end:
        gaps.append((cursor, search_end))

    return gaps
