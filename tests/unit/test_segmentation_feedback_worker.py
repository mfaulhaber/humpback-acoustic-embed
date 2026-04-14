"""Unit tests for the Pass 2 feedback training worker helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from humpback.call_parsing.segmentation.dataset import (
    build_framewise_target,
    compute_pos_weight,
    SegmentationSampleDataset,
)
from humpback.schemas.call_parsing import SegmentationFeatureConfig
from humpback.workers.event_segmentation_feedback_worker import (
    _apply_corrections,
    _subdivide_region,
)
from humpback.call_parsing.types import Event


@dataclass
class _FakeSample:
    events_json: str
    crop_start_sec: float
    crop_end_sec: float
    hydrophone_id: str = "h1"
    start_timestamp: float = 0.0
    end_timestamp: float = 0.0


def test_apply_corrections_adjust() -> None:
    original = [
        Event(
            event_id="e1",
            region_id="r1",
            start_sec=10.0,
            end_sec=12.0,
            center_sec=11.0,
            segmentation_confidence=0.9,
        ),
    ]

    @dataclass
    class _Corr:
        event_id: str
        correction_type: str
        start_sec: float | None
        end_sec: float | None

    corr = [
        _Corr(event_id="e1", correction_type="adjust", start_sec=10.5, end_sec=11.5)
    ]
    result = _apply_corrections(original, corr)  # type: ignore[arg-type]
    assert len(result) == 1
    assert result[0]["start_sec"] == 10.5
    assert result[0]["end_sec"] == 11.5


def test_apply_corrections_add() -> None:
    original: list[Event] = []

    @dataclass
    class _Corr:
        event_id: str
        correction_type: str
        start_sec: float | None
        end_sec: float | None

    corr = [_Corr(event_id="new1", correction_type="add", start_sec=5.0, end_sec=7.0)]
    result = _apply_corrections(original, corr)  # type: ignore[arg-type]
    assert len(result) == 1
    assert result[0]["start_sec"] == 5.0


def test_apply_corrections_delete() -> None:
    original = [
        Event(
            event_id="e1",
            region_id="r1",
            start_sec=10.0,
            end_sec=12.0,
            center_sec=11.0,
            segmentation_confidence=0.9,
        ),
    ]

    @dataclass
    class _Corr:
        event_id: str
        correction_type: str
        start_sec: float | None
        end_sec: float | None

    corr = [
        _Corr(event_id="e1", correction_type="delete", start_sec=None, end_sec=None)
    ]
    result = _apply_corrections(original, corr)  # type: ignore[arg-type]
    assert len(result) == 0


def test_subdivide_short_region_returns_single_sample() -> None:
    events = [{"start_sec": 5.0, "end_sec": 7.0}]
    samples = _subdivide_region(
        crop_start=0.0,
        crop_end=20.0,
        corrected_events=events,
        hydrophone_id="h1",
        start_timestamp=0.0,
        end_timestamp=100.0,
        max_crop_sec=30.0,
    )
    assert len(samples) == 1
    assert samples[0].crop_start_sec == 0.0
    assert samples[0].crop_end_sec == 20.0


def test_subdivide_long_region_creates_multiple_crops() -> None:
    events = [
        {"start_sec": 10.0, "end_sec": 12.0},
        {"start_sec": 40.0, "end_sec": 42.0},
        {"start_sec": 70.0, "end_sec": 72.0},
    ]
    samples = _subdivide_region(
        crop_start=0.0,
        crop_end=90.0,
        corrected_events=events,
        hydrophone_id="h1",
        start_timestamp=0.0,
        end_timestamp=100.0,
        max_crop_sec=30.0,
        crop_hop_sec=15.0,
    )
    # 90s with 30s windows hopping by 15s → multiple crops
    assert len(samples) > 1
    # Each crop should be <= 30s
    for s in samples:
        assert s.crop_end_sec - s.crop_start_sec <= 30.0 + 0.01
    # All crops should cover the region
    assert samples[0].crop_start_sec == 0.0
    assert samples[-1].crop_end_sec == 90.0


def test_subdivide_filters_events_to_window() -> None:
    events = [
        {"start_sec": 5.0, "end_sec": 7.0},
        {"start_sec": 50.0, "end_sec": 52.0},
    ]
    samples = _subdivide_region(
        crop_start=0.0,
        crop_end=60.0,
        corrected_events=events,
        hydrophone_id="h1",
        start_timestamp=0.0,
        end_timestamp=100.0,
        max_crop_sec=30.0,
        crop_hop_sec=30.0,
    )
    # First crop [0,30] should have event at 5-7, not 50-52
    first_events = json.loads(samples[0].events_json)
    assert len(first_events) == 1
    assert first_events[0]["start_sec"] == 5.0
    # Last crop should have event at 50-52
    last_events = json.loads(samples[-1].events_json)
    assert any(e["start_sec"] == 50.0 for e in last_events)


def test_feedback_sample_events_not_double_subtracted() -> None:
    """Regression: events must stay in absolute coords so build_framewise_target
    subtracts crop_start_sec exactly once.

    Before the fix, _collect_samples pre-subtracted crop_start from events AND
    build_framewise_target subtracted it again, producing zero positive frames.
    """
    cfg = SegmentationFeatureConfig(sample_rate=1000, hop_length=100)

    # Simulate what the worker builds: region at 100-110s, event at 103-105s
    crop_start = 100.0
    crop_end = 110.0
    events = [{"start_sec": 103.0, "end_sec": 105.0}]

    # Correct approach: events in absolute coords, crop_start_sec = region start
    sample = _FakeSample(
        events_json=json.dumps(events),
        crop_start_sec=crop_start,
        crop_end_sec=crop_end,
    )

    target = build_framewise_target(
        sample.events_json,
        sample.crop_start_sec,
        sample.crop_end_sec,
        cfg,
    )
    assert target.sum() > 0, (
        "Events in absolute coords with non-zero crop_start_sec must produce "
        "positive frames — if zero, events were likely double-subtracted"
    )

    # Verify compute_pos_weight also sees positives
    dataset = SegmentationSampleDataset(
        samples=[sample],
        feature_config=cfg,
        audio_loader=lambda _s: np.zeros(0, dtype=np.float32),
    )
    weight = compute_pos_weight(dataset)
    assert weight > 1.0, (
        "pos_weight should be > 1 when positives exist but are minority"
    )


def test_feedback_sample_double_subtracted_gives_zero() -> None:
    """Demonstrates the bug: if events are pre-converted to crop-relative but
    crop_start_sec is still non-zero, build_framewise_target double-subtracts
    and produces zero positive frames."""
    cfg = SegmentationFeatureConfig(sample_rate=1000, hop_length=100)

    crop_start = 100.0
    crop_end = 110.0
    # BUG path: events already made relative (3.0 = 103 - 100)
    events = [{"start_sec": 3.0, "end_sec": 5.0}]

    target = build_framewise_target(
        json.dumps(events),
        crop_start_sec=crop_start,  # non-zero — will subtract again
        crop_end_sec=crop_end,
        feature_config=cfg,
    )
    # This produces event_start = 3.0 - 100.0 = -97.0 — all negative, zero frames
    assert target.sum() == 0, (
        "Double-subtracted events should produce zero positives (demonstrating the bug)"
    )
