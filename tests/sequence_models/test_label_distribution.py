"""Tests for state-to-label distribution computation."""

from __future__ import annotations

import pytest

from humpback.sequence_models.label_distribution import (
    DetectionWindow,
    LabelRecord,
    compute_label_distribution,
)


def _window(start: float, end: float, state: int) -> dict:
    return {"start_timestamp": start, "end_timestamp": end, "viterbi_state": state}


class TestComputeLabelDistribution:
    def test_single_label_inheritance(self):
        states = [_window(100.0, 105.0, 0)]
        dw = [DetectionWindow(row_id="r1", start_utc=100.0, end_utc=105.0)]
        labels = [LabelRecord(row_id="r1", label="song")]
        result = compute_label_distribution(states, dw, labels, n_states=2)

        assert result["n_states"] == 2
        assert result["total_windows"] == 1
        assert result["states"]["0"]["all"]["song"] == 1
        assert result["states"]["1"] == {}

    def test_multi_label_window(self):
        states = [_window(100.0, 105.0, 0)]
        dw = [DetectionWindow(row_id="r1", start_utc=100.0, end_utc=105.0)]
        labels = [
            LabelRecord(row_id="r1", label="song"),
            LabelRecord(row_id="r1", label="call"),
        ]
        result = compute_label_distribution(states, dw, labels, n_states=1)

        assert result["states"]["0"]["all"]["song"] == 1
        assert result["states"]["0"]["all"]["call"] == 1

    def test_unlabeled_windows(self):
        states = [_window(100.0, 105.0, 0), _window(200.0, 205.0, 1)]
        dw = [DetectionWindow(row_id="r1", start_utc=100.0, end_utc=105.0)]
        labels = [LabelRecord(row_id="r1", label="song")]
        result = compute_label_distribution(states, dw, labels, n_states=2)

        assert result["states"]["0"]["all"]["song"] == 1
        assert result["states"]["1"]["all"]["unlabeled"] == 1

    def test_no_detection_windows_all_unlabeled(self):
        states = [_window(100.0, 105.0, 0), _window(101.0, 106.0, 0)]
        result = compute_label_distribution(states, [], [], n_states=1)

        assert result["states"]["0"]["all"]["unlabeled"] == 2

    def test_center_at_detection_end_is_excluded(self):
        """HMM window center at exactly the detection window end is excluded (half-open)."""
        states = [_window(100.0, 110.0, 0)]
        dw = [DetectionWindow(row_id="r1", start_utc=100.0, end_utc=105.0)]
        labels = [LabelRecord(row_id="r1", label="song")]
        result = compute_label_distribution(states, dw, labels, n_states=1)

        assert result["states"]["0"]["all"]["unlabeled"] == 1

    def test_center_inside_detection_window_inherits(self):
        """HMM window center just inside the detection window inherits labels."""
        states = [_window(100.0, 109.0, 0)]
        dw = [DetectionWindow(row_id="r1", start_utc=100.0, end_utc=105.0)]
        labels = [LabelRecord(row_id="r1", label="song")]
        result = compute_label_distribution(states, dw, labels, n_states=1)

        assert result["states"]["0"]["all"]["song"] == 1

    def test_center_outside_detection_window(self):
        """HMM window whose center falls outside detection range gets unlabeled."""
        states = [_window(110.0, 115.0, 0)]
        dw = [DetectionWindow(row_id="r1", start_utc=100.0, end_utc=105.0)]
        labels = [LabelRecord(row_id="r1", label="song")]
        result = compute_label_distribution(states, dw, labels, n_states=1)

        assert result["states"]["0"]["all"]["unlabeled"] == 1

    def test_empty_states(self):
        result = compute_label_distribution([], [], [], n_states=3)

        assert result["total_windows"] == 0
        assert result["n_states"] == 3
        for s in range(3):
            assert result["states"][str(s)] == {}

    def test_multiple_detection_windows_contribute_labels(self):
        states = [_window(102.0, 107.0, 0)]
        dw = [
            DetectionWindow(row_id="r1", start_utc=100.0, end_utc=105.0),
            DetectionWindow(row_id="r2", start_utc=103.0, end_utc=108.0),
        ]
        labels = [
            LabelRecord(row_id="r1", label="song"),
            LabelRecord(row_id="r2", label="call"),
        ]
        result = compute_label_distribution(states, dw, labels, n_states=1)

        assert result["states"]["0"]["all"]["song"] == 1
        assert result["states"]["0"]["all"]["call"] == 1


class TestTierAwareLabelDistribution:
    def test_tier_per_row_buckets_independently(self):
        """Each row buckets to its own tier value when tier_per_row is provided."""
        states = [
            _window(100.0, 105.0, 0),
            _window(200.0, 205.0, 0),
        ]
        dw = [
            DetectionWindow(row_id="r1", start_utc=100.0, end_utc=105.0),
            DetectionWindow(row_id="r2", start_utc=200.0, end_utc=205.0),
        ]
        labels = [
            LabelRecord(row_id="r1", label="song"),
            LabelRecord(row_id="r2", label="call"),
        ]
        result = compute_label_distribution(
            states,
            dw,
            labels,
            n_states=1,
            tier_per_row=["event_core", "near_event"],
        )

        assert result["states"]["0"]["event_core"]["song"] == 1
        assert result["states"]["0"]["near_event"]["call"] == 1
        assert "all" not in result["states"]["0"]

    def test_state_with_multiple_tiers(self):
        """A state with rows in multiple tiers produces multiple inner tier keys."""
        states = [
            _window(100.0, 105.0, 0),
            _window(200.0, 205.0, 0),
            _window(300.0, 305.0, 0),
        ]
        result = compute_label_distribution(
            states,
            [],
            [],
            n_states=1,
            tier_per_row=["event_core", "event_core", "background"],
        )

        assert result["states"]["0"]["event_core"]["unlabeled"] == 2
        assert result["states"]["0"]["background"]["unlabeled"] == 1

    def test_none_tier_per_row_uses_all_bucket(self):
        states = [_window(100.0, 105.0, 0)]
        result = compute_label_distribution(states, [], [], n_states=1)

        assert "all" in result["states"]["0"]
        assert result["states"]["0"]["all"]["unlabeled"] == 1

    def test_length_mismatch_raises(self):
        states = [_window(100.0, 105.0, 0), _window(200.0, 205.0, 0)]
        with pytest.raises(ValueError, match="tier_per_row length"):
            compute_label_distribution(
                states, [], [], n_states=1, tier_per_row=["event_core"]
            )

    def test_unlabeled_per_state_tier(self):
        states = [_window(100.0, 105.0, 0)]
        result = compute_label_distribution(
            states, [], [], n_states=1, tier_per_row=["background"]
        )

        assert result["states"]["0"]["background"]["unlabeled"] == 1
