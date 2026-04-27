"""Pure helpers for the Sequence Models continuous embedding producer.

``merge_padded_regions`` expands each region by ``±pad_seconds``, merges
padded spans whose extents overlap, and clips merged spans to the audio
envelope. ``iter_windows`` walks each merged span at a configured hop /
window-size, classifying windows as in-region or in-pad based on whether
their center timestamp lies inside any of the un-padded source regions.

These helpers are deterministic, side-effect-free, and have no I/O or
model-runner dependencies — they are the geometry layer the worker calls.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Region:
    """An un-padded source region (a Pass-1 detection's region row)."""

    region_id: str
    start_time_sec: float
    end_time_sec: float


@dataclass(frozen=True, slots=True)
class AudioEnvelope:
    """Hard bounds (in epoch seconds) the merged spans must be clipped to."""

    start_time_sec: float
    end_time_sec: float


@dataclass(slots=True)
class MergedSpan:
    """A contiguous padded span produced by merging overlapping regions.

    ``source_regions`` lists the un-padded regions whose padded extent
    contributed to this span (in start-time order). ``start_time_sec`` and
    ``end_time_sec`` are the padded, envelope-clipped bounds.
    """

    merged_span_id: int
    start_time_sec: float
    end_time_sec: float
    source_regions: list[Region] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class WindowRecord:
    """One step of ``iter_windows``: position within a merged span and its
    relationship to the original un-padded regions."""

    window_index_in_span: int
    start_time_sec: float
    end_time_sec: float
    is_in_pad: bool
    source_region_ids: list[str]


def merge_padded_regions(
    regions: list[Region],
    pad_seconds: float,
    audio_envelope: AudioEnvelope,
) -> list[MergedSpan]:
    """Expand each region by ``±pad_seconds``, merge overlapping padded
    spans, and clip merged spans to the audio envelope.

    Returns merged spans in chronological order. Each span tracks the
    original (un-padded) source regions whose padded extents contributed
    to it.

    Notes:
        - Padded spans whose bounds touch (``a.end == b.start``) are
          treated as overlapping and merged.
        - A merged span's ``[start_time_sec, end_time_sec]`` is the
          envelope-clipped padded extent; the un-padded source regions
          themselves are not clipped (the worker uses the un-padded
          extents to decide ``is_in_pad`` for windows inside the span).
        - If the envelope clip makes a span empty (start >= end), the
          span is dropped.
    """
    if pad_seconds < 0:
        raise ValueError("pad_seconds must be >= 0")
    if not regions:
        return []

    sorted_regions = sorted(regions, key=lambda r: (r.start_time_sec, r.end_time_sec))

    merged: list[MergedSpan] = []
    next_id = 0
    for region in sorted_regions:
        padded_start = region.start_time_sec - pad_seconds
        padded_end = region.end_time_sec + pad_seconds

        if merged and padded_start <= merged[-1].end_time_sec:
            current = merged[-1]
            current.end_time_sec = max(current.end_time_sec, padded_end)
            current.source_regions.append(region)
        else:
            merged.append(
                MergedSpan(
                    merged_span_id=next_id,
                    start_time_sec=padded_start,
                    end_time_sec=padded_end,
                    source_regions=[region],
                )
            )
            next_id += 1

    clipped: list[MergedSpan] = []
    next_id = 0
    for span in merged:
        clipped_start = max(span.start_time_sec, audio_envelope.start_time_sec)
        clipped_end = min(span.end_time_sec, audio_envelope.end_time_sec)
        if clipped_end <= clipped_start:
            continue
        clipped.append(
            MergedSpan(
                merged_span_id=next_id,
                start_time_sec=clipped_start,
                end_time_sec=clipped_end,
                source_regions=span.source_regions,
            )
        )
        next_id += 1

    return clipped


def iter_windows(
    span: MergedSpan,
    hop_seconds: float,
    window_size_seconds: float,
) -> Iterator[WindowRecord]:
    """Yield ``WindowRecord`` entries for ``span`` at the given hop and
    window size.

    Window centers determine in-region membership: a window is treated
    as in-region (``is_in_pad=False``) when its center timestamp falls
    inside any source region's un-padded extent. The boundary is
    inclusive on both sides — a center at ``region.start_time_sec`` or
    ``region.end_time_sec`` counts as in-region.

    Yields windows whose end timestamp does not exceed
    ``span.end_time_sec``. The number of windows in a span equals
    ``floor((span_duration - window_size) / hop) + 1`` when
    ``span_duration >= window_size``, else 0.
    """
    if hop_seconds <= 0:
        raise ValueError("hop_seconds must be > 0")
    if window_size_seconds <= 0:
        raise ValueError("window_size_seconds must be > 0")

    span_duration = span.end_time_sec - span.start_time_sec
    if span_duration < window_size_seconds:
        return

    last_start_offset = span_duration - window_size_seconds
    n_windows = int(last_start_offset / hop_seconds) + 1

    for idx in range(n_windows):
        start = span.start_time_sec + idx * hop_seconds
        end = start + window_size_seconds
        center = start + window_size_seconds / 2.0

        source_region_ids: list[str] = []
        for region in span.source_regions:
            if region.start_time_sec <= center <= region.end_time_sec:
                source_region_ids.append(region.region_id)

        yield WindowRecord(
            window_index_in_span=idx,
            start_time_sec=start,
            end_time_sec=end,
            is_in_pad=not source_region_ids,
            source_region_ids=source_region_ids,
        )
