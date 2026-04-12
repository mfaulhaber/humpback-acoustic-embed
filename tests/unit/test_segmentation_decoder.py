"""Tests for the Pass 2 segmentation hysteresis decoder."""

from __future__ import annotations

import uuid

import numpy as np
import pytest

from humpback.call_parsing.segmentation.decoder import decode_events
from humpback.schemas.call_parsing import SegmentationDecoderConfig


def _config(
    high: float = 0.5,
    low: float = 0.3,
    min_event_sec: float = 0.0,
    merge_gap_sec: float = 0.0,
) -> SegmentationDecoderConfig:
    return SegmentationDecoderConfig(
        high_threshold=high,
        low_threshold=low,
        min_event_sec=min_event_sec,
        merge_gap_sec=merge_gap_sec,
    )


def test_empty_input_returns_empty_list() -> None:
    events = decode_events(
        np.zeros(0, dtype=np.float32),
        region_id="r1",
        region_start_sec=0.0,
        hop_sec=0.1,
        config=_config(),
    )
    assert events == []


def test_all_zeros_returns_empty_list() -> None:
    events = decode_events(
        np.zeros(10, dtype=np.float32),
        region_id="r1",
        region_start_sec=0.0,
        hop_sec=0.1,
        config=_config(),
    )
    assert events == []


def test_single_peak_above_high_produces_one_event() -> None:
    # Frames [2, 4] inclusive above high=0.5 → event (2, 4).
    probs = np.array([0.0, 0.0, 0.8, 0.8, 0.8, 0.0, 0.0], dtype=np.float32)
    events = decode_events(
        probs,
        region_id="r1",
        region_start_sec=0.0,
        hop_sec=0.1,
        config=_config(),
    )
    assert len(events) == 1
    assert events[0].start_sec == pytest.approx(0.2)
    assert events[0].end_sec == pytest.approx(0.5)


def test_peak_only_crossing_low_produces_no_event() -> None:
    probs = np.array([0.0, 0.35, 0.4, 0.35, 0.0], dtype=np.float32)
    events = decode_events(
        probs,
        region_id="r1",
        region_start_sec=0.0,
        hop_sec=0.1,
        config=_config(),
    )
    assert events == []


def test_two_peaks_with_large_gap_produce_two_events() -> None:
    # merge_gap_sec=0.1, hop_sec=0.1 → merge_gap_frames=1.
    # Events (1,2) and (5,6): gap=2 frames → 2 < 1 is False → no merge.
    probs = np.zeros(10, dtype=np.float32)
    probs[1:3] = 0.8
    probs[5:7] = 0.8
    events = decode_events(
        probs,
        region_id="r1",
        region_start_sec=0.0,
        hop_sec=0.1,
        config=_config(merge_gap_sec=0.1),
    )
    assert len(events) == 2


def test_two_peaks_with_small_gap_merge() -> None:
    # merge_gap_sec=0.5, hop_sec=0.1 → merge_gap_frames=5.
    # Events (1,2) and (4,5): gap=1 frame → 1 < 5 → merge into (1,5).
    probs = np.zeros(10, dtype=np.float32)
    probs[1:3] = 0.8
    probs[4:6] = 0.8
    events = decode_events(
        probs,
        region_id="r1",
        region_start_sec=0.0,
        hop_sec=0.1,
        config=_config(merge_gap_sec=0.5),
    )
    assert len(events) == 1
    assert events[0].start_sec == pytest.approx(0.1)
    assert events[0].end_sec == pytest.approx(0.6)


def test_hysteresis_dip_below_high_but_above_low_stays_one_event() -> None:
    # Peak at 0.8 → dip to 0.4 (above low=0.3) → back to 0.8 → zero.
    # Without hysteresis this would be two events. With hysteresis it's one.
    probs = np.array([0.0, 0.8, 0.4, 0.8, 0.0], dtype=np.float32)
    events = decode_events(
        probs,
        region_id="r1",
        region_start_sec=0.0,
        hop_sec=0.1,
        config=_config(),
    )
    assert len(events) == 1
    assert events[0].start_sec == pytest.approx(0.1)
    assert events[0].end_sec == pytest.approx(0.4)


def test_event_shorter_than_min_is_dropped() -> None:
    probs = np.array([0.0, 0.8, 0.0, 0.0], dtype=np.float32)
    events = decode_events(
        probs,
        region_id="r1",
        region_start_sec=0.0,
        hop_sec=0.1,
        config=_config(min_event_sec=0.15),
    )
    assert events == []


def test_starts_above_threshold_produces_event_from_frame_zero() -> None:
    probs = np.array([0.8, 0.8, 0.0, 0.0], dtype=np.float32)
    events = decode_events(
        probs,
        region_id="r1",
        region_start_sec=0.0,
        hop_sec=0.1,
        config=_config(),
    )
    assert len(events) == 1
    assert events[0].start_sec == pytest.approx(0.0)
    assert events[0].end_sec == pytest.approx(0.2)


def test_ends_above_threshold_produces_event_through_last_frame() -> None:
    probs = np.array([0.0, 0.0, 0.8, 0.8], dtype=np.float32)
    events = decode_events(
        probs,
        region_id="r1",
        region_start_sec=0.0,
        hop_sec=0.1,
        config=_config(),
    )
    assert len(events) == 1
    assert events[0].start_sec == pytest.approx(0.2)
    assert events[0].end_sec == pytest.approx(0.4)


def test_max_confidence_uses_max_of_frame_range() -> None:
    probs = np.array([0.8, 0.9, 0.7, 0.0], dtype=np.float32)
    events = decode_events(
        probs,
        region_id="r1",
        region_start_sec=0.0,
        hop_sec=0.1,
        config=_config(),
    )
    assert len(events) == 1
    assert events[0].segmentation_confidence == pytest.approx(0.9)


def test_absolute_timestamp_computation() -> None:
    # region_start_sec=100.0, hop_sec=0.032, first_frame_idx=10.
    probs = np.zeros(30, dtype=np.float32)
    probs[10:20] = 0.8
    events = decode_events(
        probs,
        region_id="r1",
        region_start_sec=100.0,
        hop_sec=0.032,
        config=_config(),
    )
    assert len(events) == 1
    assert events[0].start_sec == pytest.approx(100.32)
    assert events[0].end_sec == pytest.approx(100.0 + 20 * 0.032)
    assert events[0].center_sec == pytest.approx(
        (events[0].start_sec + events[0].end_sec) / 2.0
    )


def test_event_ids_are_unique_uuid4_hex_strings() -> None:
    probs = np.zeros(20, dtype=np.float32)
    probs[1:4] = 0.8
    probs[7:10] = 0.8
    probs[13:16] = 0.8
    events = decode_events(
        probs,
        region_id="r1",
        region_start_sec=0.0,
        hop_sec=0.1,
        config=_config(merge_gap_sec=0.0),
    )
    assert len(events) == 3
    ids = [e.event_id for e in events]
    assert len(set(ids)) == 3
    for event_id in ids:
        parsed = uuid.UUID(hex=event_id)
        assert parsed.version == 4
        assert event_id == parsed.hex


def test_region_id_is_carried_to_every_event() -> None:
    probs = np.zeros(20, dtype=np.float32)
    probs[1:4] = 0.8
    probs[7:10] = 0.8
    events = decode_events(
        probs,
        region_id="region-xyz",
        region_start_sec=0.0,
        hop_sec=0.1,
        config=_config(merge_gap_sec=0.0),
    )
    assert len(events) == 2
    for ev in events:
        assert ev.region_id == "region-xyz"


def test_single_frame_input_does_not_crash() -> None:
    events = decode_events(
        np.array([0.8], dtype=np.float32),
        region_id="r1",
        region_start_sec=0.0,
        hop_sec=0.1,
        config=_config(min_event_sec=0.0),
    )
    assert len(events) == 1
    assert events[0].start_sec == pytest.approx(0.0)
    assert events[0].end_sec == pytest.approx(0.1)


def test_rejects_non_1d_input() -> None:
    with pytest.raises(ValueError):
        decode_events(
            np.zeros((2, 5), dtype=np.float32),
            region_id="r1",
            region_start_sec=0.0,
            hop_sec=0.1,
            config=_config(),
        )


def test_rejects_non_positive_hop_sec() -> None:
    with pytest.raises(ValueError):
        decode_events(
            np.zeros(5, dtype=np.float32),
            region_id="r1",
            region_start_sec=0.0,
            hop_sec=0.0,
            config=_config(),
        )
