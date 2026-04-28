"""Tests for state-to-label distribution computation."""

from __future__ import annotations

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
        assert result["states"]["0"]["song"] == 1
        assert result["states"]["1"] == {}

    def test_multi_label_window(self):
        states = [_window(100.0, 105.0, 0)]
        dw = [DetectionWindow(row_id="r1", start_utc=100.0, end_utc=105.0)]
        labels = [
            LabelRecord(row_id="r1", label="song"),
            LabelRecord(row_id="r1", label="call"),
        ]
        result = compute_label_distribution(states, dw, labels, n_states=1)

        assert result["states"]["0"]["song"] == 1
        assert result["states"]["0"]["call"] == 1

    def test_unlabeled_windows(self):
        states = [_window(100.0, 105.0, 0), _window(200.0, 205.0, 1)]
        dw = [DetectionWindow(row_id="r1", start_utc=100.0, end_utc=105.0)]
        labels = [LabelRecord(row_id="r1", label="song")]
        result = compute_label_distribution(states, dw, labels, n_states=2)

        assert result["states"]["0"]["song"] == 1
        assert result["states"]["1"]["unlabeled"] == 1

    def test_no_detection_windows_all_unlabeled(self):
        states = [_window(100.0, 105.0, 0), _window(101.0, 106.0, 0)]
        result = compute_label_distribution(states, [], [], n_states=1)

        assert result["states"]["0"]["unlabeled"] == 2

    def test_center_at_detection_end_is_excluded(self):
        """HMM window center at exactly the detection window end is excluded (half-open)."""
        states = [_window(100.0, 110.0, 0)]
        dw = [DetectionWindow(row_id="r1", start_utc=100.0, end_utc=105.0)]
        labels = [LabelRecord(row_id="r1", label="song")]
        result = compute_label_distribution(states, dw, labels, n_states=1)

        assert result["states"]["0"]["unlabeled"] == 1

    def test_center_inside_detection_window_inherits(self):
        """HMM window center just inside the detection window inherits labels."""
        states = [_window(100.0, 109.0, 0)]
        dw = [DetectionWindow(row_id="r1", start_utc=100.0, end_utc=105.0)]
        labels = [LabelRecord(row_id="r1", label="song")]
        result = compute_label_distribution(states, dw, labels, n_states=1)

        assert result["states"]["0"]["song"] == 1

    def test_center_outside_detection_window(self):
        """HMM window whose center falls outside detection range gets unlabeled."""
        states = [_window(110.0, 115.0, 0)]
        dw = [DetectionWindow(row_id="r1", start_utc=100.0, end_utc=105.0)]
        labels = [LabelRecord(row_id="r1", label="song")]
        result = compute_label_distribution(states, dw, labels, n_states=1)

        assert result["states"]["0"]["unlabeled"] == 1

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

        assert result["states"]["0"]["song"] == 1
        assert result["states"]["0"]["call"] == 1
