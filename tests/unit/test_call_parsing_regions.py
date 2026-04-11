"""Unit tests for the Pass 1 region decoder."""

from __future__ import annotations

import re
from typing import Any

import pytest

from humpback.call_parsing.regions import decode_regions
from humpback.call_parsing.types import Region
from humpback.schemas.call_parsing import RegionDetectionConfig


UUID4_HEX_RE = re.compile(r"^[0-9a-f]{32}$")


def _event(
    start_sec: float,
    end_sec: float,
    *,
    avg: float = 0.8,
    peak: float = 0.9,
    n: int = 1,
) -> dict[str, Any]:
    return {
        "start_sec": start_sec,
        "end_sec": end_sec,
        "avg_confidence": avg,
        "peak_confidence": peak,
        "n_windows": n,
    }


def _cfg(**overrides: float) -> RegionDetectionConfig:
    return RegionDetectionConfig(**overrides)


def test_empty_input_returns_empty_list() -> None:
    assert decode_regions([], audio_duration_sec=60.0, config=_cfg()) == []


def test_single_event_returns_single_region_with_clamped_padding() -> None:
    events = [_event(10.0, 15.0, avg=0.7, peak=0.95, n=6)]
    regions = decode_regions(
        events, audio_duration_sec=100.0, config=_cfg(padding_sec=1.0)
    )

    assert len(regions) == 1
    region = regions[0]
    assert isinstance(region, Region)
    assert region.start_sec == 10.0
    assert region.end_sec == 15.0
    assert region.padded_start_sec == 9.0
    assert region.padded_end_sec == 16.0
    assert region.max_score == 0.95
    assert region.mean_score == 0.7
    assert region.n_windows == 6


def test_adjacent_padded_bounds_touching_fuse_inclusively() -> None:
    # Event 1: padded bounds [4.0, 11.0]. Event 2: padded bounds [11.0, 18.0].
    # Touching at 11.0 must merge per the inclusive (<=) rule.
    events = [
        _event(5.0, 10.0, avg=0.6, peak=0.85, n=3),
        _event(12.0, 17.0, avg=0.7, peak=0.95, n=4),
    ]
    regions = decode_regions(
        events, audio_duration_sec=100.0, config=_cfg(padding_sec=1.0)
    )

    assert len(regions) == 1
    region = regions[0]
    assert region.start_sec == 5.0
    assert region.end_sec == 17.0
    assert region.padded_start_sec == 4.0
    assert region.padded_end_sec == 18.0
    assert region.max_score == 0.95
    assert region.n_windows == 7


def test_non_overlapping_padded_bounds_stay_separate() -> None:
    events = [
        _event(5.0, 10.0, avg=0.6, peak=0.85, n=3),
        _event(20.0, 25.0, avg=0.7, peak=0.95, n=4),
    ]
    regions = decode_regions(
        events, audio_duration_sec=100.0, config=_cfg(padding_sec=1.0)
    )

    assert len(regions) == 2
    assert regions[0].start_sec == 5.0
    assert regions[1].start_sec == 20.0
    # No fusing — n_windows stays per-event.
    assert regions[0].n_windows == 3
    assert regions[1].n_windows == 4


def test_event_at_zero_clamps_padded_start_to_zero() -> None:
    events = [_event(0.0, 5.0, avg=0.75, peak=0.9, n=2)]
    regions = decode_regions(
        events, audio_duration_sec=60.0, config=_cfg(padding_sec=1.0)
    )

    assert len(regions) == 1
    assert regions[0].padded_start_sec == 0.0
    assert regions[0].start_sec == 0.0


def test_event_at_audio_end_clamps_padded_end_to_duration() -> None:
    duration = 60.0
    events = [_event(55.0, duration, avg=0.7, peak=0.9, n=2)]
    regions = decode_regions(
        events, audio_duration_sec=duration, config=_cfg(padding_sec=1.0)
    )

    assert len(regions) == 1
    assert regions[0].padded_end_sec == duration
    assert regions[0].end_sec == duration


def test_min_region_duration_drops_short_regions() -> None:
    events = [
        _event(5.0, 5.5, avg=0.8, peak=0.9, n=1),  # 0.5 s — dropped
        _event(20.0, 25.0, avg=0.7, peak=0.95, n=5),  # 5.0 s — kept
    ]
    regions = decode_regions(
        events,
        audio_duration_sec=100.0,
        config=_cfg(padding_sec=1.0, min_region_duration_sec=2.0),
    )

    assert len(regions) == 1
    assert regions[0].start_sec == 20.0
    assert regions[0].end_sec == 25.0


def test_three_events_partial_merge_cascades_correctly() -> None:
    # Events 1 and 2 share a padded overlap; event 3 is isolated.
    # Padded bounds @ padding=1.0:
    #   ev1 [4, 11], ev2 [9, 16]  -> merge
    #   ev3 [29, 36]               -> standalone
    events = [
        _event(5.0, 10.0, avg=0.6, peak=0.8, n=3),
        _event(10.0, 15.0, avg=0.7, peak=0.92, n=4),
        _event(30.0, 35.0, avg=0.8, peak=0.95, n=5),
    ]
    regions = decode_regions(
        events, audio_duration_sec=100.0, config=_cfg(padding_sec=1.0)
    )

    assert len(regions) == 2
    merged = regions[0]
    assert merged.start_sec == 5.0
    assert merged.end_sec == 15.0
    assert merged.padded_start_sec == 4.0
    assert merged.padded_end_sec == 16.0
    assert merged.max_score == 0.92
    assert merged.n_windows == 7

    standalone = regions[1]
    assert standalone.start_sec == 30.0
    assert standalone.end_sec == 35.0
    assert standalone.n_windows == 5


def test_weighted_mean_score_uses_window_weights() -> None:
    # Merged mean = (0.6 * 3 + 0.9 * 7) / 10 = 0.81
    events = [
        _event(5.0, 10.0, avg=0.6, peak=0.8, n=3),
        _event(11.0, 16.0, avg=0.9, peak=0.95, n=7),
    ]
    regions = decode_regions(
        events, audio_duration_sec=100.0, config=_cfg(padding_sec=1.0)
    )

    assert len(regions) == 1
    assert regions[0].mean_score == pytest.approx(0.81)
    assert regions[0].n_windows == 10


def test_region_ids_are_unique_uuid4_hex_strings() -> None:
    events = [
        _event(10.0, 12.0, n=1),
        _event(30.0, 32.0, n=1),
        _event(50.0, 52.0, n=1),
    ]
    regions = decode_regions(events, audio_duration_sec=100.0, config=_cfg())

    ids = [r.region_id for r in regions]
    assert len(ids) == 3
    assert len(set(ids)) == 3
    for rid in ids:
        assert UUID4_HEX_RE.match(rid), f"region_id {rid!r} is not a UUID4 hex string"


def test_unsorted_input_is_handled() -> None:
    events = [
        _event(30.0, 35.0, avg=0.8, peak=0.95, n=5),
        _event(5.0, 10.0, avg=0.6, peak=0.8, n=3),
        _event(50.0, 52.0, avg=0.7, peak=0.85, n=2),
    ]
    regions = decode_regions(
        events, audio_duration_sec=100.0, config=_cfg(padding_sec=1.0)
    )

    assert [r.start_sec for r in regions] == [5.0, 30.0, 50.0]


def test_output_is_sorted_by_start_sec() -> None:
    events = [
        _event(5.0, 7.0, n=1),
        _event(10.0, 12.0, n=1),
        _event(20.0, 22.0, n=1),
    ]
    regions = decode_regions(
        events, audio_duration_sec=100.0, config=_cfg(padding_sec=0.5)
    )

    starts = [r.start_sec for r in regions]
    assert starts == sorted(starts)
