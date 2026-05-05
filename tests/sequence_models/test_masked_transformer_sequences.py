"""Tests for masked-transformer sequence construction."""

from __future__ import annotations

import numpy as np
import pytest

from humpback.sequence_models.masked_transformer_sequences import (
    EffectiveEventInterval,
    build_masked_transformer_training_sequences,
)


def _sequence(length: int, offset: float = 0.0) -> np.ndarray:
    return (np.arange(length * 3, dtype=np.float32).reshape(length, 3) + offset).astype(
        np.float32
    )


def test_region_mode_returns_original_full_region_sequences_and_tiers() -> None:
    sequences = [_sequence(4), _sequence(3, offset=100.0)]
    tiers = [["background"] * 4, ["event_core"] * 3]

    result = build_masked_transformer_training_sequences(
        region_ids=["r1", "r2"],
        sequences=sequences,
        tier_lists=tiers,
        starts=[[0.0, 1.0, 2.0, 3.0], [10.0, 11.0, 12.0]],
        ends=[[1.0, 2.0, 3.0, 4.0], [11.0, 12.0, 13.0]],
        effective_events=[
            EffectiveEventInterval("r1", 1.2, 1.8),
            EffectiveEventInterval("r2", 10.2, 10.8),
        ],
        mode="region",
    )

    assert result.sequences == sequences
    assert result.sequences[0] is sequences[0]
    assert result.tier_lists == tiers
    assert result.source_kinds == ["region", "region"]
    assert all(candidate.event_id is None for candidate in result.candidates)
    assert all(candidate.human_types == () for candidate in result.candidates)


def test_event_centered_windows_include_context_and_clamp_edges() -> None:
    sequences = [_sequence(6)]
    starts = [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]]
    ends = [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]
    tiers = [
        [
            "background",
            "near_event",
            "event_core",
            "event_core",
            "near_event",
            "background",
        ]
    ]

    result = build_masked_transformer_training_sequences(
        region_ids=["r1"],
        sequences=sequences,
        tier_lists=tiers,
        starts=starts,
        ends=ends,
        effective_events=[
            EffectiveEventInterval("r1", 2.2, 2.8, "event-mid", ("Moan",)),
            EffectiveEventInterval("r1", 0.1, 0.4, "event-start", ()),
            EffectiveEventInterval("r1", 5.6, 5.9, "event-end", ("Whup",)),
        ],
        mode="event_centered",
        pre_event_context_sec=1.0,
        post_event_context_sec=1.0,
    )

    assert [(c.start_index, c.end_index) for c in result.candidates] == [
        (0, 2),
        (1, 4),
        (4, 6),
    ]
    np.testing.assert_array_equal(result.sequences[1], sequences[0][1:4])
    assert result.tier_lists[1] == tiers[0][1:4]
    assert result.candidates[1].event_id == "event-mid"
    assert result.candidates[1].event_start_timestamp == 2.2
    assert result.candidates[1].event_end_timestamp == 2.8
    assert result.candidates[1].event_start_index == 1
    assert result.candidates[1].event_end_index == 2
    assert result.candidates[1].human_types == ("Moan",)
    assert all(kind == "event_centered" for kind in result.source_kinds)


def test_event_centered_skips_events_without_overlapping_chunks() -> None:
    result = build_masked_transformer_training_sequences(
        region_ids=["r1"],
        sequences=[_sequence(2)],
        tier_lists=[["background", "background"]],
        starts=[[0.0, 1.0]],
        ends=[[1.0, 2.0]],
        effective_events=[
            EffectiveEventInterval("r1", 10.0, 11.0),
            EffectiveEventInterval("unknown", 0.0, 1.0),
        ],
        mode="event_centered",
    )

    assert result.sequences == []
    assert result.tier_lists == []


def test_mixed_mode_is_deterministic_and_seed_sensitive() -> None:
    sequences = [_sequence(8), _sequence(8, offset=100.0)]
    starts = [list(np.arange(8, dtype=float)), list(np.arange(20, 28, dtype=float))]
    ends = [[v + 1.0 for v in seq] for seq in starts]
    events = [
        EffectiveEventInterval("r1", 0.2, 0.8),
        EffectiveEventInterval("r1", 2.2, 2.8),
        EffectiveEventInterval("r1", 4.2, 4.8),
        EffectiveEventInterval("r2", 20.2, 20.8),
        EffectiveEventInterval("r2", 22.2, 22.8),
        EffectiveEventInterval("r2", 24.2, 24.8),
    ]

    first = build_masked_transformer_training_sequences(
        region_ids=["r1", "r2"],
        sequences=sequences,
        tier_lists=[[""] * 8, [""] * 8],
        starts=starts,
        ends=ends,
        effective_events=events,
        mode="mixed",
        event_centered_fraction=0.5,
        seed=7,
    )
    second = build_masked_transformer_training_sequences(
        region_ids=["r1", "r2"],
        sequences=sequences,
        tier_lists=[[""] * 8, [""] * 8],
        starts=starts,
        ends=ends,
        effective_events=events,
        mode="mixed",
        event_centered_fraction=0.5,
        seed=7,
    )
    changed = build_masked_transformer_training_sequences(
        region_ids=["r1", "r2"],
        sequences=sequences,
        tier_lists=[[""] * 8, [""] * 8],
        starts=starts,
        ends=ends,
        effective_events=events,
        mode="mixed",
        event_centered_fraction=0.5,
        seed=8,
    )

    first_shape = [
        (c.source_kind, c.region_id, c.start_index, c.end_index)
        for c in first.candidates
    ]
    second_shape = [
        (c.source_kind, c.region_id, c.start_index, c.end_index)
        for c in second.candidates
    ]
    changed_shape = [
        (c.source_kind, c.region_id, c.start_index, c.end_index)
        for c in changed.candidates
    ]
    assert first_shape == second_shape
    assert first_shape != changed_shape
    assert "region" in first.source_kinds
    assert "event_centered" in first.source_kinds


def test_builder_rejects_misaligned_metadata() -> None:
    with pytest.raises(ValueError, match="start/end metadata"):
        build_masked_transformer_training_sequences(
            region_ids=["r1"],
            sequences=[_sequence(2)],
            tier_lists=[["background", "background"]],
            starts=[[0.0]],
            ends=[[1.0, 2.0]],
            effective_events=[],
        )
