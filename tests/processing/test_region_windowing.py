"""Tests for the pure region-windowing helpers."""

import math

import pytest

from humpback.processing.region_windowing import (
    AudioEnvelope,
    MergedSpan,
    Region,
    iter_windows,
    merge_padded_regions,
)

ENVELOPE = AudioEnvelope(start_time_sec=0.0, end_time_sec=10_000.0)


def _ids(span: MergedSpan) -> list[str]:
    return [r.region_id for r in span.source_regions]


def test_merge_padded_regions_empty_input():
    assert merge_padded_regions([], pad_seconds=10.0, audio_envelope=ENVELOPE) == []


def test_merge_padded_regions_non_overlapping():
    regions = [
        Region("r1", 100.0, 110.0),
        Region("r2", 200.0, 220.0),
        Region("r3", 500.0, 505.0),
    ]
    spans = merge_padded_regions(regions, pad_seconds=5.0, audio_envelope=ENVELOPE)
    assert len(spans) == 3
    assert _ids(spans[0]) == ["r1"]
    assert _ids(spans[1]) == ["r2"]
    assert _ids(spans[2]) == ["r3"]
    assert spans[0].start_time_sec == 95.0
    assert spans[0].end_time_sec == 115.0


def test_merge_padded_regions_two_padded_overlap():
    # r1 padded extent [90, 130]; r2 padded extent [120, 160] → overlap.
    regions = [
        Region("r1", 100.0, 120.0),
        Region("r2", 130.0, 150.0),
    ]
    spans = merge_padded_regions(regions, pad_seconds=10.0, audio_envelope=ENVELOPE)
    assert len(spans) == 1
    assert _ids(spans[0]) == ["r1", "r2"]
    assert spans[0].start_time_sec == 90.0
    assert spans[0].end_time_sec == 160.0


def test_merge_padded_regions_three_chained():
    regions = [
        Region("r1", 100.0, 110.0),
        Region("r2", 115.0, 125.0),
        Region("r3", 130.0, 140.0),
    ]
    spans = merge_padded_regions(regions, pad_seconds=10.0, audio_envelope=ENVELOPE)
    assert len(spans) == 1
    assert _ids(spans[0]) == ["r1", "r2", "r3"]
    assert spans[0].start_time_sec == 90.0
    assert spans[0].end_time_sec == 150.0


def test_merge_padded_regions_clip_at_start():
    envelope = AudioEnvelope(start_time_sec=100.0, end_time_sec=10_000.0)
    regions = [Region("r1", 105.0, 120.0)]
    spans = merge_padded_regions(regions, pad_seconds=10.0, audio_envelope=envelope)
    assert len(spans) == 1
    assert spans[0].start_time_sec == 100.0
    assert spans[0].end_time_sec == 130.0
    assert spans[0].source_regions[0].start_time_sec == 105.0


def test_merge_padded_regions_clip_at_end():
    envelope = AudioEnvelope(start_time_sec=0.0, end_time_sec=200.0)
    regions = [Region("r1", 180.0, 195.0)]
    spans = merge_padded_regions(regions, pad_seconds=10.0, audio_envelope=envelope)
    assert len(spans) == 1
    assert spans[0].start_time_sec == 170.0
    assert spans[0].end_time_sec == 200.0


def test_merge_padded_regions_drops_fully_clipped_span():
    envelope = AudioEnvelope(start_time_sec=0.0, end_time_sec=50.0)
    regions = [Region("r1", 100.0, 110.0)]
    spans = merge_padded_regions(regions, pad_seconds=5.0, audio_envelope=envelope)
    assert spans == []


def test_merge_padded_regions_rejects_negative_pad():
    with pytest.raises(ValueError):
        merge_padded_regions(
            [Region("r1", 0.0, 1.0)], pad_seconds=-1.0, audio_envelope=ENVELOPE
        )


def test_iter_windows_count_matches_floor_formula():
    span = MergedSpan(
        merged_span_id=0,
        start_time_sec=0.0,
        end_time_sec=20.0,
        source_regions=[Region("r1", 0.0, 20.0)],
    )
    windows = list(iter_windows(span, hop_seconds=1.0, window_size_seconds=5.0))
    expected = math.floor((20.0 - 5.0) / 1.0) + 1
    assert len(windows) == expected


def test_iter_windows_too_short_yields_nothing():
    span = MergedSpan(
        merged_span_id=0,
        start_time_sec=0.0,
        end_time_sec=4.0,
        source_regions=[Region("r1", 0.0, 4.0)],
    )
    windows = list(iter_windows(span, hop_seconds=1.0, window_size_seconds=5.0))
    assert windows == []


def test_iter_windows_is_in_pad_classification():
    # Span: [0, 20]; region [10, 14]; window size 5, hop 1.
    span = MergedSpan(
        merged_span_id=0,
        start_time_sec=0.0,
        end_time_sec=20.0,
        source_regions=[Region("r1", 10.0, 14.0)],
    )
    windows = list(iter_windows(span, hop_seconds=1.0, window_size_seconds=5.0))
    in_region = [w for w in windows if not w.is_in_pad]
    assert in_region, "expected at least one in-region window"
    for w in windows:
        center = w.start_time_sec + 2.5
        in_region_expected = 10.0 <= center <= 14.0
        assert w.is_in_pad == (not in_region_expected)
        if not w.is_in_pad:
            assert w.source_region_ids == ["r1"]
        else:
            assert w.source_region_ids == []


def test_iter_windows_boundary_is_inclusive():
    # window center at exactly region start counts as in-region.
    span = MergedSpan(
        merged_span_id=0,
        start_time_sec=0.0,
        end_time_sec=20.0,
        source_regions=[Region("r1", 7.5, 10.0)],
    )
    # window [5, 10] centers on 7.5 — exactly the region start.
    windows = list(iter_windows(span, hop_seconds=1.0, window_size_seconds=5.0))
    boundary = [w for w in windows if math.isclose(w.start_time_sec, 5.0)]
    assert boundary and not boundary[0].is_in_pad
    assert boundary[0].source_region_ids == ["r1"]


def test_iter_windows_multi_region_overlap():
    # Two touching regions inside a merged span — center inside both.
    r1 = Region("r1", 5.0, 10.0)
    r2 = Region("r2", 10.0, 15.0)
    span = MergedSpan(
        merged_span_id=0,
        start_time_sec=0.0,
        end_time_sec=20.0,
        source_regions=[r1, r2],
    )
    windows = list(iter_windows(span, hop_seconds=0.5, window_size_seconds=5.0))
    centered_at_10 = [w for w in windows if math.isclose(w.start_time_sec + 2.5, 10.0)]
    assert centered_at_10
    assert centered_at_10[0].source_region_ids == ["r1", "r2"]
    assert not centered_at_10[0].is_in_pad


def test_iter_windows_validates_inputs():
    span = MergedSpan(0, 0.0, 10.0, [Region("r", 0.0, 10.0)])
    with pytest.raises(ValueError):
        list(iter_windows(span, hop_seconds=0.0, window_size_seconds=5.0))
    with pytest.raises(ValueError):
        list(iter_windows(span, hop_seconds=1.0, window_size_seconds=0.0))
