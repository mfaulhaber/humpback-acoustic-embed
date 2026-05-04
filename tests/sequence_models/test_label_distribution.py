"""Tests for event-scoped state-to-label distribution.

Covers the pure functions ``assign_labels_to_windows`` and
``compute_label_distribution``. The async ``load_effective_event_labels``
loader is exercised end-to-end in the HMM and Masked Transformer service
tests, which set up the full DB + parquet fixtures.
"""

from __future__ import annotations

import pytest

from humpback.sequence_models.label_distribution import (
    BACKGROUND_LABEL,
    EffectiveEventLabels,
    WindowAnnotation,
    assign_labels_to_windows,
    compute_label_distribution,
)


def _row(start: float, end: float, state: int) -> dict:
    return {
        "start_timestamp": start,
        "end_timestamp": end,
        "viterbi_state": state,
    }


def _event(
    event_id: str,
    start_utc: float,
    end_utc: float,
    types: tuple[str, ...] = (),
    confidences: dict[str, float] | None = None,
) -> EffectiveEventLabels:
    return EffectiveEventLabels(
        event_id=event_id,
        start_utc=start_utc,
        end_utc=end_utc,
        types=frozenset(types),
        confidences=dict(confidences or {}),
    )


class TestAssignLabelsToWindows:
    def test_inverts_events_to_windows(self):
        """Three disjoint events, windows in/out, multi-label union."""
        rows = [
            _row(100.0, 101.0, 0),  # center 100.5 → in event A
            _row(101.0, 102.0, 1),  # center 101.5 → in event A
            _row(150.0, 151.0, 2),  # center 150.5 → outside (background)
            _row(200.0, 201.0, 0),  # center 200.5 → in event B
            _row(300.0, 301.0, 1),  # center 300.5 → in event C (empty types)
        ]
        events = [
            _event("A", 100.0, 105.0, types=("moan", "whup")),
            _event("B", 200.0, 205.0, types=("song",)),
            _event("C", 300.0, 305.0, types=()),  # empty -> background
        ]

        annotations = assign_labels_to_windows(rows, events)

        assert len(annotations) == 5
        assert annotations[0].event_id == "A"
        assert annotations[0].event_types == ("moan", "whup")
        assert annotations[1].event_id == "A"
        assert annotations[1].event_types == ("moan", "whup")
        # Outside any event.
        assert annotations[2].event_id is None
        assert annotations[2].event_types == ()
        # In event B.
        assert annotations[3].event_id == "B"
        assert annotations[3].event_types == ("song",)
        # In event C but C has empty types -> treat as background.
        assert annotations[4].event_id is None
        assert annotations[4].event_types == ()

    def test_empty_event_types_become_background(self):
        """Event with all types stripped by corrections -> background bucket."""
        rows = [_row(100.0, 101.0, 0)]
        events = [_event("A", 100.0, 105.0, types=())]

        annotations = assign_labels_to_windows(rows, events)

        assert annotations[0].event_id is None
        assert annotations[0].event_types == ()
        assert annotations[0].event_confidence == {}

    def test_half_open_interval_at_event_end(self):
        """Window center exactly at event.end_utc is excluded."""
        rows = [_row(104.5, 105.5, 0)]  # center 105.0 → on boundary
        events = [_event("A", 100.0, 105.0, types=("moan",))]

        annotations = assign_labels_to_windows(rows, events)
        # 105.0 < 105.0 is False → background.
        assert annotations[0].event_id is None

    def test_no_events_all_background(self):
        rows = [_row(100.0, 101.0, 0), _row(200.0, 201.0, 1)]
        annotations = assign_labels_to_windows(rows, [])

        assert all(a.event_id is None for a in annotations)
        assert all(a.event_types == () for a in annotations)

    def test_no_rows(self):
        events = [_event("A", 100.0, 105.0, types=("moan",))]
        assert assign_labels_to_windows([], events) == []

    def test_two_pointer_handles_unsorted_rows(self):
        """Rows in arbitrary order must still tag correctly."""
        rows = [
            _row(200.0, 201.0, 1),  # in B
            _row(100.0, 101.0, 0),  # in A
            _row(150.0, 151.0, 2),  # background
        ]
        events = [
            _event("A", 100.0, 105.0, types=("moan",)),
            _event("B", 200.0, 205.0, types=("song",)),
        ]

        annotations = assign_labels_to_windows(rows, events)
        assert annotations[0].event_id == "B"
        assert annotations[1].event_id == "A"
        assert annotations[2].event_id is None

    def test_two_pointer_o_n_large_input(self):
        """10k windows + 1k events runs to completion with correct counts."""
        rows: list[dict] = []
        for i in range(10_000):
            t = 100.0 + i * 0.5
            rows.append(_row(t, t + 1.0, i % 4))

        events: list[EffectiveEventLabels] = []
        # 1k disjoint events, each 2.0 s wide, every 5.0 s.
        for j in range(1_000):
            t0 = 100.0 + j * 5.0
            events.append(_event(f"E{j}", t0, t0 + 2.0, types=("moan",)))

        annotations = assign_labels_to_windows(rows, events)
        assert len(annotations) == len(rows)
        # Spot check: at minimum, several thousand rows fall inside events.
        labeled = sum(1 for a in annotations if a.event_id is not None)
        assert 0 < labeled < len(rows)

    def test_confidence_propagates(self):
        rows = [_row(100.0, 101.0, 0)]
        events = [
            _event(
                "A",
                100.0,
                105.0,
                types=("moan",),
                confidences={"moan": 0.92},
            )
        ]

        annotations = assign_labels_to_windows(rows, events)
        assert annotations[0].event_confidence == {"moan": 0.92}


class TestComputeLabelDistribution:
    def test_basic_buckets(self):
        rows = [
            _row(100.0, 101.0, 0),
            _row(101.0, 102.0, 0),
            _row(200.0, 201.0, 1),
        ]
        annotations = [
            WindowAnnotation(
                event_id="A", event_types=("moan",), event_confidence={"moan": 0.9}
            ),
            WindowAnnotation(
                event_id="A", event_types=("moan",), event_confidence={"moan": 0.9}
            ),
            WindowAnnotation(event_id=None, event_types=(), event_confidence={}),
        ]

        result = compute_label_distribution(rows, annotations, n_states=2)

        assert result["n_states"] == 2
        assert result["total_windows"] == 3
        assert result["states"]["0"] == {"moan": 2}
        assert result["states"]["1"] == {BACKGROUND_LABEL: 1}

    def test_multi_label_union(self):
        """Multi-label event contributes +1 to each label per window."""
        rows = [_row(100.0, 101.0, 0), _row(101.0, 102.0, 0)]
        annotations = [
            WindowAnnotation(
                event_id="A",
                event_types=("moan", "whup"),
                event_confidence={"moan": 0.9, "whup": 0.7},
            ),
            WindowAnnotation(
                event_id="A",
                event_types=("moan", "whup"),
                event_confidence={"moan": 0.9, "whup": 0.7},
            ),
        ]

        result = compute_label_distribution(rows, annotations, n_states=1)

        assert result["states"]["0"] == {"moan": 2, "whup": 2}
        # Per-state total exceeds total_windows for multi-label events.
        assert result["total_windows"] == 2

    def test_background_bucket(self):
        rows = [_row(100.0, 101.0, 0), _row(200.0, 201.0, 0)]
        annotations = [
            WindowAnnotation(event_id=None, event_types=(), event_confidence={}),
            WindowAnnotation(event_id=None, event_types=(), event_confidence={}),
        ]

        result = compute_label_distribution(rows, annotations, n_states=1)

        assert result["states"]["0"] == {BACKGROUND_LABEL: 2}

    def test_no_tier_dimension(self):
        """Output shape: states[i] is dict[label, int], not a nested tier dict."""
        rows = [_row(100.0, 101.0, 0)]
        annotations = [
            WindowAnnotation(
                event_id="A",
                event_types=("moan",),
                event_confidence={"moan": 0.9},
            )
        ]

        result = compute_label_distribution(rows, annotations, n_states=1)

        # The bucket value is an int count, not a nested dict.
        assert isinstance(result["states"]["0"]["moan"], int)
        assert "all" not in result["states"]["0"]
        assert "event_core" not in result["states"]["0"]

    def test_empty_states_dict_per_unused_state(self):
        rows = [_row(100.0, 101.0, 0)]
        annotations = [
            WindowAnnotation(event_id=None, event_types=(), event_confidence={})
        ]

        result = compute_label_distribution(rows, annotations, n_states=3)

        assert result["states"]["0"] == {BACKGROUND_LABEL: 1}
        assert result["states"]["1"] == {}
        assert result["states"]["2"] == {}

    def test_length_mismatch_raises(self):
        rows = [_row(100.0, 101.0, 0), _row(101.0, 102.0, 0)]
        annotations = [
            WindowAnnotation(event_id=None, event_types=(), event_confidence={})
        ]

        with pytest.raises(ValueError, match="annotations length"):
            compute_label_distribution(rows, annotations, n_states=1)
