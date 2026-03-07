"""Exhaustive tests for merge_detection_spans and merge_detection_events."""

from humpback.classifier.detector import merge_detection_events, merge_detection_spans


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


# ---- merge_detection_events (hysteresis) tests ----


def _make_records(confs, window_size=5.0, hop=5.0):
    """Build window_records from a list of confidences with given hop."""
    return [
        {"offset_sec": i * hop, "end_sec": i * hop + window_size, "confidence": c}
        for i, c in enumerate(confs)
    ]


def test_events_hysteresis_basic():
    """High opens event, low sustains, below-low closes."""
    records = _make_records([0.3, 0.8, 0.5, 0.3, 0.2])
    result = merge_detection_events(records, high_threshold=0.7, low_threshold=0.4)
    assert len(result) == 1
    assert result[0]["start_sec"] == 5.0   # window index 1
    assert result[0]["end_sec"] == 15.0    # window index 2 end
    assert result[0]["n_windows"] == 2


def test_events_no_hysteresis():
    """high == low is equivalent to single threshold."""
    confs = [0.8, 0.9, 0.3, 0.7]
    records = _make_records(confs)
    result = merge_detection_events(records, high_threshold=0.5, low_threshold=0.5)
    assert len(result) == 2
    assert result[0]["start_sec"] == 0.0
    assert result[0]["end_sec"] == 10.0
    assert result[1]["start_sec"] == 15.0
    assert result[1]["end_sec"] == 20.0


def test_events_overlapping_windows():
    """Correct start/end with overlapping window times (hop < window)."""
    records = [
        {"offset_sec": 0.0, "end_sec": 5.0, "confidence": 0.3},
        {"offset_sec": 1.0, "end_sec": 6.0, "confidence": 0.8},
        {"offset_sec": 2.0, "end_sec": 7.0, "confidence": 0.6},
        {"offset_sec": 3.0, "end_sec": 8.0, "confidence": 0.3},
    ]
    result = merge_detection_events(records, high_threshold=0.7, low_threshold=0.5)
    assert len(result) == 1
    assert result[0]["start_sec"] == 1.0
    assert result[0]["end_sec"] == 7.0
    assert result[0]["n_windows"] == 2


def test_events_n_windows():
    """Verify n_windows count per event."""
    records = _make_records([0.8, 0.6, 0.6, 0.3, 0.9])
    result = merge_detection_events(records, high_threshold=0.7, low_threshold=0.5)
    assert len(result) == 2
    assert result[0]["n_windows"] == 3
    assert result[1]["n_windows"] == 1


def test_events_empty_input():
    """Empty list returns no events."""
    assert merge_detection_events([], 0.7, 0.4) == []


def test_events_all_positive():
    """All windows above high → single event."""
    records = _make_records([0.8, 0.9, 0.75])
    result = merge_detection_events(records, high_threshold=0.7, low_threshold=0.5)
    assert len(result) == 1
    assert result[0]["start_sec"] == 0.0
    assert result[0]["end_sec"] == 15.0
    assert result[0]["n_windows"] == 3


def test_events_all_negative():
    """All windows below high → no events."""
    records = _make_records([0.3, 0.4, 0.2])
    result = merge_detection_events(records, high_threshold=0.7, low_threshold=0.5)
    assert result == []


def test_write_tsv_preserves_n_windows():
    """write_detections_tsv includes n_windows, readable by DictReader."""
    import csv
    import tempfile
    from pathlib import Path

    from humpback.classifier.detector import write_detections_tsv

    detections = [
        {
            "filename": "a.wav",
            "start_sec": 0.0,
            "end_sec": 5.0,
            "avg_confidence": 0.8,
            "peak_confidence": 0.9,
            "n_windows": 3,
        },
        {
            "filename": "a.wav",
            "start_sec": 5.0,
            "end_sec": 12.0,
            "avg_confidence": 0.7,
            "peak_confidence": 0.85,
            "n_windows": 5,
        },
    ]

    with tempfile.TemporaryDirectory() as tmp:
        tsv_path = Path(tmp) / "detections.tsv"
        write_detections_tsv(detections, tsv_path)

        with open(tsv_path, newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["n_windows"] == "3"
        assert rows[1]["n_windows"] == "5"
        # All expected fieldnames present
        assert set(reader.fieldnames) == {
            "filename", "start_sec", "end_sec",
            "avg_confidence", "peak_confidence", "n_windows",
        }


def test_events_backward_compat_equivalence():
    """When hop == window and high == low == threshold, matches merge_detection_spans."""
    import numpy as np

    confs = [0.8, 0.2, 0.9, 0.7, 0.1, 0.6]
    threshold = 0.5
    window_size = 5.0

    old_result = merge_detection_spans(confs, threshold, window_size)
    records = _make_records(confs, window_size=window_size, hop=window_size)
    new_result = merge_detection_events(records, high_threshold=threshold, low_threshold=threshold)

    assert len(old_result) == len(new_result)
    for old, new in zip(old_result, new_result):
        assert old["start_sec"] == new["start_sec"]
        assert old["end_sec"] == new["end_sec"]
        assert abs(old["avg_confidence"] - new["avg_confidence"]) < 1e-6
        assert old["peak_confidence"] == new["peak_confidence"]
