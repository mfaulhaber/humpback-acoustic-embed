"""Exhaustive tests for merge_detection_spans."""

from humpback.classifier.detector import merge_detection_spans


def test_all_negative():
    """No windows above threshold → no spans."""
    result = merge_detection_spans([0.1, 0.2, 0.3], 0.5, 5.0)
    assert result == []


def test_all_positive():
    """All windows above threshold → single span covering everything."""
    result = merge_detection_spans([0.8, 0.9, 0.7], 0.5, 5.0)
    assert len(result) == 1
    assert result[0]["start_sec"] == 0.0
    assert result[0]["end_sec"] == 15.0
    assert abs(result[0]["avg_confidence"] - 0.8) < 0.01
    assert result[0]["peak_confidence"] == 0.9


def test_alternating():
    """Alternating positive/negative → multiple single-window spans."""
    result = merge_detection_spans([0.8, 0.2, 0.9, 0.1], 0.5, 5.0)
    assert len(result) == 2
    assert result[0]["start_sec"] == 0.0
    assert result[0]["end_sec"] == 5.0
    assert result[1]["start_sec"] == 10.0
    assert result[1]["end_sec"] == 15.0


def test_single_positive_window():
    """Single positive window in the middle."""
    result = merge_detection_spans([0.1, 0.8, 0.1], 0.5, 5.0)
    assert len(result) == 1
    assert result[0]["start_sec"] == 5.0
    assert result[0]["end_sec"] == 10.0
    assert result[0]["avg_confidence"] == 0.8
    assert result[0]["peak_confidence"] == 0.8


def test_consecutive_merge():
    """Two consecutive positive windows merge into one span."""
    result = merge_detection_spans([0.1, 0.7, 0.9, 0.1], 0.5, 5.0)
    assert len(result) == 1
    assert result[0]["start_sec"] == 5.0
    assert result[0]["end_sec"] == 15.0
    assert abs(result[0]["avg_confidence"] - 0.8) < 0.01
    assert result[0]["peak_confidence"] == 0.9


def test_empty_input():
    """Empty list → no spans."""
    result = merge_detection_spans([], 0.5, 5.0)
    assert result == []


def test_single_window_positive():
    """Single window that is positive."""
    result = merge_detection_spans([0.9], 0.5, 5.0)
    assert len(result) == 1
    assert result[0]["start_sec"] == 0.0
    assert result[0]["end_sec"] == 5.0


def test_single_window_negative():
    """Single window that is negative."""
    result = merge_detection_spans([0.3], 0.5, 5.0)
    assert result == []


def test_threshold_boundary():
    """Window exactly at threshold is positive."""
    result = merge_detection_spans([0.5], 0.5, 5.0)
    assert len(result) == 1


def test_just_below_threshold():
    """Window just below threshold is negative."""
    result = merge_detection_spans([0.499], 0.5, 5.0)
    assert result == []


def test_two_separate_spans():
    """Two groups of positives separated by negatives."""
    confs = [0.8, 0.9, 0.1, 0.1, 0.7, 0.8]
    result = merge_detection_spans(confs, 0.5, 5.0)
    assert len(result) == 2
    assert result[0]["start_sec"] == 0.0
    assert result[0]["end_sec"] == 10.0
    assert result[1]["start_sec"] == 20.0
    assert result[1]["end_sec"] == 30.0


def test_different_window_size():
    """Works with non-default window size."""
    result = merge_detection_spans([0.8, 0.9], 0.5, 3.0)
    assert result[0]["start_sec"] == 0.0
    assert result[0]["end_sec"] == 6.0


def test_confidence_calculations():
    """Verify avg and peak confidence are computed correctly."""
    confs = [0.6, 0.8, 0.7]
    result = merge_detection_spans(confs, 0.5, 5.0)
    assert len(result) == 1
    assert abs(result[0]["avg_confidence"] - 0.7) < 0.01
    assert result[0]["peak_confidence"] == 0.8


def test_span_at_end():
    """Positive span at the very end of the list."""
    result = merge_detection_spans([0.1, 0.1, 0.8, 0.9], 0.5, 5.0)
    assert len(result) == 1
    assert result[0]["start_sec"] == 10.0
    assert result[0]["end_sec"] == 20.0
