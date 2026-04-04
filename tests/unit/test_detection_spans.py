"""Exhaustive tests for merge_detection_spans and merge_detection_events."""

from humpback.classifier.detector import (
    merge_detection_events,
    merge_detection_spans,
    snap_and_merge_detection_events,
    snap_event_bounds,
)


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
    assert result[0]["start_sec"] == 5.0  # window index 1
    assert result[0]["end_sec"] == 15.0  # window index 2 end
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
            "start_utc": 1000.0,
            "end_utc": 1005.0,
            "avg_confidence": 0.8,
            "peak_confidence": 0.9,
            "n_windows": 3,
        },
        {
            "start_utc": 1005.0,
            "end_utc": 1012.0,
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
        assert reader.fieldnames is not None
        assert set(reader.fieldnames) == {
            "start_utc",
            "end_utc",
            "avg_confidence",
            "peak_confidence",
            "n_windows",
            "raw_start_utc",
            "raw_end_utc",
            "merged_event_count",
        }


def test_append_detections_tsv_creates_with_header(tmp_path):
    """append_detections_tsv creates file with header on first call."""
    import csv

    from humpback.classifier.detector import append_detections_tsv

    tsv_path = tmp_path / "detections.tsv"
    detections = [
        {
            "start_utc": 1000.0,
            "end_utc": 1005.0,
            "avg_confidence": 0.8,
            "peak_confidence": 0.9,
            "n_windows": 2,
        },
    ]
    append_detections_tsv(detections, tsv_path)

    with open(tsv_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)

    assert len(rows) == 1
    assert rows[0]["start_utc"] == "1000.0"
    assert reader.fieldnames is not None
    assert set(reader.fieldnames) == {
        "start_utc",
        "end_utc",
        "avg_confidence",
        "peak_confidence",
        "n_windows",
        "raw_start_utc",
        "raw_end_utc",
        "merged_event_count",
    }


def test_append_detections_tsv_appends_without_duplicate_header(tmp_path):
    """Second call appends rows without duplicating the header."""
    import csv

    from humpback.classifier.detector import append_detections_tsv

    tsv_path = tmp_path / "detections.tsv"
    det1 = [
        {
            "start_utc": 1000.0,
            "end_utc": 1005.0,
            "avg_confidence": 0.8,
            "peak_confidence": 0.9,
            "n_windows": 2,
        }
    ]
    det2 = [
        {
            "start_utc": 2000.0,
            "end_utc": 2005.0,
            "avg_confidence": 0.7,
            "peak_confidence": 0.85,
            "n_windows": 3,
        }
    ]
    append_detections_tsv(det1, tsv_path)
    append_detections_tsv(det2, tsv_path)

    with open(tsv_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)

    assert len(rows) == 2
    assert rows[0]["start_utc"] == "1000.0"
    assert rows[1]["start_utc"] == "2000.0"


def test_append_detections_tsv_noop_on_empty():
    """append_detections_tsv does nothing for empty list."""
    import tempfile
    from pathlib import Path

    from humpback.classifier.detector import append_detections_tsv

    with tempfile.TemporaryDirectory() as tmp:
        tsv_path = Path(tmp) / "detections.tsv"
        append_detections_tsv([], tsv_path)
        assert not tsv_path.exists()


def test_write_tsv_with_hydrophone_extract_filename_column(tmp_path):
    """write_detections_tsv supports hydrophone-only extract_filename column."""
    import csv

    from humpback.classifier.detector import write_detections_tsv

    tsv_path = tmp_path / "hydrophone_detections.tsv"
    fieldnames = [
        "start_utc",
        "end_utc",
        "avg_confidence",
        "peak_confidence",
        "n_windows",
        "hydrophone_name",
    ]
    detections = [
        {
            "start_utc": 1719907315.0,
            "end_utc": 1719907325.0,
            "avg_confidence": 0.95,
            "peak_confidence": 0.97,
            "n_windows": 4,
            "hydrophone_name": "rpi_north",
        },
    ]

    write_detections_tsv(detections, tsv_path, fieldnames=fieldnames)

    with open(tsv_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)

    assert len(rows) == 1
    assert rows[0]["hydrophone_name"] == "rpi_north"
    assert reader.fieldnames == fieldnames


def test_append_tsv_with_hydrophone_extract_filename_column(tmp_path):
    """append_detections_tsv writes and appends rows with extract_filename."""
    import csv

    from humpback.classifier.detector import append_detections_tsv

    tsv_path = tmp_path / "hydrophone_detections.tsv"
    fieldnames = [
        "start_utc",
        "end_utc",
        "avg_confidence",
        "peak_confidence",
        "n_windows",
        "hydrophone_name",
    ]
    det1 = [
        {
            "start_utc": 1719907315.0,
            "end_utc": 1719907325.0,
            "avg_confidence": 0.95,
            "peak_confidence": 0.97,
            "n_windows": 4,
            "hydrophone_name": "rpi_north",
        },
    ]
    det2 = [
        {
            "start_utc": 1719907400.0,
            "end_utc": 1719907410.0,
            "avg_confidence": 0.91,
            "peak_confidence": 0.93,
            "n_windows": 3,
            "hydrophone_name": "rpi_north",
        },
    ]

    append_detections_tsv(det1, tsv_path, fieldnames=fieldnames)
    append_detections_tsv(det2, tsv_path, fieldnames=fieldnames)

    with open(tsv_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)

    assert len(rows) == 2
    assert rows[0]["start_utc"] == "1719907315.0"
    assert rows[1]["start_utc"] == "1719907400.0"
    assert reader.fieldnames == fieldnames


def test_events_backward_compat_equivalence():
    """When hop == window and high == low == threshold, matches merge_detection_spans."""

    confs = [0.8, 0.2, 0.9, 0.7, 0.1, 0.6]
    threshold = 0.5
    window_size = 5.0

    old_result = merge_detection_spans(confs, threshold, window_size)
    records = _make_records(confs, window_size=window_size, hop=window_size)
    new_result = merge_detection_events(
        records, high_threshold=threshold, low_threshold=threshold
    )

    assert len(old_result) == len(new_result)
    for old, new in zip(old_result, new_result):
        assert old["start_sec"] == new["start_sec"]
        assert old["end_sec"] == new["end_sec"]
        assert abs(old["avg_confidence"] - new["avg_confidence"]) < 1e-6
        assert old["peak_confidence"] == new["peak_confidence"]


def test_snap_event_bounds_outward_to_window():
    """Bounds snap outward to enclosing window-size multiples."""
    start, end = snap_event_bounds(2.0, 11.0, 5.0)
    assert start == 0.0
    assert end == 15.0


def test_snap_and_merge_detection_events_merges_collisions():
    """Events that snap to the same range are merged deterministically."""
    events = [
        {
            "start_sec": 15.0,
            "end_sec": 22.0,
            "avg_confidence": 0.9,
            "peak_confidence": 0.95,
            "n_windows": 3,
        },
        {
            "start_sec": 19.0,
            "end_sec": 25.0,
            "avg_confidence": 0.8,
            "peak_confidence": 0.9,
            "n_windows": 2,
        },
    ]
    result = snap_and_merge_detection_events(events, window_size_seconds=5.0)
    assert len(result) == 1
    merged = result[0]
    assert merged["start_sec"] == 15.0
    assert merged["end_sec"] == 25.0
    # Weighted mean by n_windows: (0.9*3 + 0.8*2)/5
    assert abs(merged["avg_confidence"] - 0.86) < 1e-6
    assert merged["peak_confidence"] == 0.95
    assert merged["n_windows"] == 5
    assert merged["raw_start_sec"] == 15.0
    assert merged["raw_end_sec"] == 25.0
    assert merged["merged_event_count"] == 2


# ---- select_peak_windows_from_events (NMS) ----


class TestSelectPeakWindowsFromEvents:
    """Tests for NMS-based peak window selection within merged events."""

    def _make_window_records(
        self,
        offsets: list[float] | list[int],
        confidences: list[float],
        window_size: float = 5.0,
    ) -> list[dict]:
        return [
            {"offset_sec": o, "end_sec": o + window_size, "confidence": c}
            for o, c in zip(offsets, confidences)
        ]

    def test_empty_events(self):
        from humpback.classifier.detector import select_peak_windows_from_events

        result = select_peak_windows_from_events([], [], 5.0, min_score=0.7)
        assert result == []

    def test_empty_window_records(self):
        from humpback.classifier.detector import select_peak_windows_from_events

        events = [{"start_sec": 0.0, "end_sec": 10.0, "n_windows": 5}]
        result = select_peak_windows_from_events(events, [], 5.0, min_score=0.7)
        assert result == []

    def test_single_event_single_peak(self):
        """One 10-sec event, peak at offset 3 → one 5-sec detection [3, 8]."""
        from humpback.classifier.detector import select_peak_windows_from_events

        window_records = self._make_window_records(
            offsets=[0, 1, 2, 3, 4, 5],
            confidences=[0.5, 0.6, 0.7, 0.9, 0.8, 0.6],
        )
        events = [
            {
                "start_sec": 0.0,
                "end_sec": 10.0,
                "n_windows": 6,
                "raw_start_sec": 0.0,
                "raw_end_sec": 10.0,
                "merged_event_count": 1,
            }
        ]
        result = select_peak_windows_from_events(
            events, window_records, 5.0, min_score=0.7
        )
        assert len(result) == 1
        assert result[0]["start_sec"] == 3.0
        assert result[0]["end_sec"] == 8.0

    def test_single_event_two_peaks_nms(self):
        """One 25-sec event with two distinct peaks → two 5-sec detections."""
        from humpback.classifier.detector import select_peak_windows_from_events

        # Peak at offset 2 and offset 18, with a valley in between
        offsets = list(range(0, 21))  # 0..20
        confidences = [
            0.5,
            0.6,
            0.9,
            0.8,
            0.5,  # peak around 2
            0.3,
            0.2,
            0.2,
            0.2,
            0.2,  # valley
            0.2,
            0.2,
            0.2,
            0.2,
            0.3,  # valley
            0.5,
            0.6,
            0.8,
            0.95,
            0.7,  # peak around 18
            0.4,
        ]
        window_records = self._make_window_records(offsets, confidences)
        events = [
            {
                "start_sec": 0.0,
                "end_sec": 25.0,
                "n_windows": 21,
                "raw_start_sec": 0.0,
                "raw_end_sec": 25.0,
                "merged_event_count": 1,
            }
        ]
        result = select_peak_windows_from_events(
            events, window_records, 5.0, min_score=0.7
        )
        assert len(result) == 2
        # Sorted by offset: first at 2, second at 18
        assert result[0]["start_sec"] == 2.0
        assert result[0]["end_sec"] == 7.0
        assert result[1]["start_sec"] == 18.0
        assert result[1]["end_sec"] == 23.0

    def test_all_below_min_score(self):
        """Event where all windows are below min_score → no output."""
        from humpback.classifier.detector import select_peak_windows_from_events

        window_records = self._make_window_records(
            offsets=[0, 1, 2, 3, 4],
            confidences=[0.3, 0.4, 0.5, 0.4, 0.3],
        )
        events = [
            {
                "start_sec": 0.0,
                "end_sec": 9.0,
                "n_windows": 5,
                "raw_start_sec": 0.0,
                "raw_end_sec": 9.0,
                "merged_event_count": 1,
            }
        ]
        result = select_peak_windows_from_events(
            events, window_records, 5.0, min_score=0.7
        )
        assert result == []

    def test_multiple_events(self):
        """Two separate events → one peak each."""
        from humpback.classifier.detector import select_peak_windows_from_events

        window_records = self._make_window_records(
            offsets=[0, 1, 2, 3, 4, 20, 21, 22, 23, 24],
            confidences=[0.5, 0.8, 0.9, 0.7, 0.5, 0.6, 0.7, 0.95, 0.8, 0.6],
        )
        events = [
            {
                "start_sec": 0.0,
                "end_sec": 9.0,
                "n_windows": 5,
                "raw_start_sec": 0.5,
                "raw_end_sec": 8.5,
                "merged_event_count": 1,
            },
            {
                "start_sec": 20.0,
                "end_sec": 29.0,
                "n_windows": 5,
                "raw_start_sec": 20.5,
                "raw_end_sec": 28.5,
                "merged_event_count": 1,
            },
        ]
        result = select_peak_windows_from_events(
            events, window_records, 5.0, min_score=0.7
        )
        assert len(result) == 2
        assert result[0]["start_sec"] == 2.0
        assert result[1]["start_sec"] == 22.0

    def test_preserves_audit_fields(self):
        """Audit fields (raw bounds, merged_event_count) are preserved."""
        from humpback.classifier.detector import select_peak_windows_from_events

        window_records = self._make_window_records(
            offsets=[0, 1, 2], confidences=[0.5, 0.9, 0.6]
        )
        events = [
            {
                "start_sec": 0.0,
                "end_sec": 7.0,
                "n_windows": 3,
                "raw_start_sec": 0.3,
                "raw_end_sec": 6.8,
                "merged_event_count": 2,
                "filename": "test.wav",
            }
        ]
        result = select_peak_windows_from_events(
            events, window_records, 5.0, min_score=0.7
        )
        assert len(result) == 1
        assert result[0]["raw_start_sec"] == 0.3
        assert result[0]["raw_end_sec"] == 6.8
        assert result[0]["merged_event_count"] == 2
        assert result[0]["filename"] == "test.wav"

    def test_output_exactly_window_size(self):
        """Every output detection spans exactly window_size_seconds."""
        from humpback.classifier.detector import select_peak_windows_from_events

        offsets = list(range(0, 16))
        confidences = [0.9 - 0.02 * abs(i - 5) for i in range(16)]
        window_records = self._make_window_records(offsets, confidences)
        events = [
            {
                "start_sec": 0.0,
                "end_sec": 20.0,
                "n_windows": 16,
                "raw_start_sec": 0.0,
                "raw_end_sec": 20.0,
                "merged_event_count": 1,
            }
        ]
        result = select_peak_windows_from_events(
            events, window_records, 5.0, min_score=0.7
        )
        for det in result:
            assert abs((det["end_sec"] - det["start_sec"]) - 5.0) < 1e-6

    def test_overlapping_events_deduplicate(self):
        """Two events sharing the same peak window produce one detection, not two."""
        from humpback.classifier.detector import select_peak_windows_from_events

        # Both events cover the window at offset 5 (the shared peak).
        window_records = self._make_window_records(
            offsets=[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            confidences=[0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
        )
        events = [
            {
                "start_sec": 0.0,
                "end_sec": 10.0,
                "n_windows": 6,
                "raw_start_sec": 1.0,
                "raw_end_sec": 8.0,
                "merged_event_count": 1,
            },
            {
                "start_sec": 5.0,
                "end_sec": 15.0,
                "n_windows": 6,
                "raw_start_sec": 6.0,
                "raw_end_sec": 14.0,
                "merged_event_count": 1,
            },
        ]
        result = select_peak_windows_from_events(
            events, window_records, 5.0, min_score=0.7
        )
        starts = [d["start_sec"] for d in result]
        # offset 5 should appear only once despite being in both events
        assert starts.count(5.0) == 1


# ---- select_prominent_peaks_from_events (prominence-based) ----


class TestSelectProminentPeaksFromEvents:
    """Tests for prominence-based peak window selection within merged events."""

    def _make_window_records(
        self,
        offsets: list[float] | list[int],
        confidences: list[float],
        window_size: float = 5.0,
    ) -> list[dict]:
        return [
            {"offset_sec": o, "end_sec": o + window_size, "confidence": c}
            for o, c in zip(offsets, confidences)
        ]

    def _make_event(self, start: float, end: float, n_windows: int) -> dict:
        return {
            "start_sec": start,
            "end_sec": end,
            "n_windows": n_windows,
            "raw_start_sec": start,
            "raw_end_sec": end,
            "merged_event_count": 1,
        }

    def test_empty_events(self):
        from humpback.classifier.detector_utils import (
            select_prominent_peaks_from_events,
        )

        result = select_prominent_peaks_from_events([], [], 5.0, min_score=0.7)
        assert result == []

    def test_empty_window_records(self):
        from humpback.classifier.detector_utils import (
            select_prominent_peaks_from_events,
        )

        events = [self._make_event(0.0, 10.0, 5)]
        result = select_prominent_peaks_from_events(events, [], 5.0, min_score=0.7)
        assert result == []

    def test_clear_peaks_separated_by_deep_valley(self):
        """Two peaks separated by a large score drop — both detected."""
        from humpback.classifier.detector_utils import (
            select_prominent_peaks_from_events,
        )

        offsets = list(range(0, 20))
        confidences = [
            0.5,
            0.7,
            0.9,
            0.95,
            0.8,  # peak around 3
            0.4,
            0.2,
            0.1,
            0.2,
            0.3,  # valley
            0.5,
            0.7,
            0.85,
            0.92,
            0.88,  # peak around 13
            0.6,
            0.4,
            0.3,
            0.2,
            0.1,  # tail
        ]
        window_records = self._make_window_records(offsets, confidences)
        events = [self._make_event(0.0, 24.0, 20)]
        result = select_prominent_peaks_from_events(
            events, window_records, 5.0, min_score=0.7, min_prominence=1.0
        )
        assert len(result) == 2
        starts = [d["start_sec"] for d in result]
        assert starts[0] < starts[1]

    def test_subtle_peak_above_threshold(self):
        """A peak with prominence just above min_prominence is detected."""
        from humpback.classifier.detector_utils import (
            select_prominent_peaks_from_events,
        )

        # Dip from 0.95 to 0.91 → logit prominence ~0.63, above 0.5
        offsets = list(range(0, 12))
        confidences = [
            0.8,
            0.9,
            0.95,
            0.93,  # peak at 2
            0.91,  # shallow valley
            0.93,
            0.95,
            0.94,  # peak at 6
            0.9,
            0.85,
            0.8,
            0.7,
        ]
        window_records = self._make_window_records(offsets, confidences)
        events = [self._make_event(0.0, 16.0, 12)]
        result = select_prominent_peaks_from_events(
            events, window_records, 5.0, min_score=0.7, min_prominence=0.5
        )
        # Both peaks detected (logit prominence ~0.63 > 0.5)
        assert len(result) == 2

    def test_subtle_peak_below_threshold(self):
        """A peak with prominence below min_prominence is filtered out."""
        from humpback.classifier.detector_utils import (
            select_prominent_peaks_from_events,
        )

        # Dip from 0.95 to 0.94 → logit prominence ~0.19, below 0.5
        offsets = list(range(0, 10))
        confidences = [
            0.8,
            0.9,
            0.95,
            0.94,
            0.95,
            0.94,
            0.9,
            0.85,
            0.8,
            0.7,
        ]
        window_records = self._make_window_records(offsets, confidences)
        events = [self._make_event(0.0, 14.0, 10)]
        result = select_prominent_peaks_from_events(
            events, window_records, 5.0, min_score=0.7, min_prominence=0.5
        )
        # Only one peak via fallback (logit prominence ~0.19 < 0.5)
        assert len(result) == 1

    def test_plateau_single_peak(self):
        """Constant high scores produce a single peak via fallback."""
        from humpback.classifier.detector_utils import (
            select_prominent_peaks_from_events,
        )

        offsets = list(range(0, 8))
        confidences = [0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95]
        window_records = self._make_window_records(offsets, confidences)
        events = [self._make_event(0.0, 12.0, 8)]
        result = select_prominent_peaks_from_events(
            events, window_records, 5.0, min_score=0.7, min_prominence=1.0
        )
        # All peaks have zero prominence in logit space too — fallback emits one.
        assert len(result) == 1

    def test_single_peak_in_event(self):
        """One clear peak — emits one window."""
        from humpback.classifier.detector_utils import (
            select_prominent_peaks_from_events,
        )

        offsets = list(range(0, 6))
        confidences = [0.5, 0.6, 0.9, 0.8, 0.6, 0.4]
        window_records = self._make_window_records(offsets, confidences)
        events = [self._make_event(0.0, 10.0, 6)]
        result = select_prominent_peaks_from_events(
            events, window_records, 5.0, min_score=0.7, min_prominence=0.5
        )
        assert len(result) == 1
        assert result[0]["start_sec"] == 2.0
        assert result[0]["end_sec"] == 7.0

    def test_no_peaks_above_min_score(self):
        """No windows above min_score — empty result."""
        from humpback.classifier.detector_utils import (
            select_prominent_peaks_from_events,
        )

        offsets = list(range(0, 5))
        confidences = [0.3, 0.4, 0.5, 0.4, 0.3]
        window_records = self._make_window_records(offsets, confidences)
        events = [self._make_event(0.0, 9.0, 5)]
        result = select_prominent_peaks_from_events(
            events, window_records, 5.0, min_score=0.7, min_prominence=1.0
        )
        assert result == []

    def test_overlapping_windows_emitted(self):
        """Peaks < 5 seconds apart produce overlapping 5-sec windows."""
        from humpback.classifier.detector_utils import (
            select_prominent_peaks_from_events,
        )

        offsets = list(range(0, 12))
        confidences = [
            0.5,
            0.7,
            0.95,
            0.7,  # peak at 2
            0.3,  # valley
            0.7,
            0.92,
            0.7,  # peak at 6 — only 4s from peak at 2
            0.3,
            0.2,
            0.1,
            0.1,
        ]
        window_records = self._make_window_records(offsets, confidences)
        events = [self._make_event(0.0, 16.0, 12)]
        result = select_prominent_peaks_from_events(
            events, window_records, 5.0, min_score=0.7, min_prominence=1.0
        )
        assert len(result) == 2
        # Windows overlap: [2,7] and [6,11]
        assert result[0]["start_sec"] == 2.0
        assert result[0]["end_sec"] == 7.0
        assert result[1]["start_sec"] == 6.0
        assert result[1]["end_sec"] == 11.0
        # Confirm they overlap
        assert result[0]["end_sec"] > result[1]["start_sec"]

    def test_edge_peaks(self):
        """Peaks at first and last window are handled correctly."""
        from humpback.classifier.detector_utils import (
            select_prominent_peaks_from_events,
        )

        offsets = list(range(0, 8))
        confidences = [0.95, 0.7, 0.5, 0.3, 0.3, 0.5, 0.7, 0.92]
        window_records = self._make_window_records(offsets, confidences)
        events = [self._make_event(0.0, 12.0, 8)]
        result = select_prominent_peaks_from_events(
            events, window_records, 5.0, min_score=0.7, min_prominence=1.0
        )
        assert len(result) == 2
        assert result[0]["start_sec"] == 0.0
        assert result[1]["start_sec"] == 7.0

    def test_raw_scores_used_for_prominence(self):
        """Prominence uses raw scores in logit space, detecting dips that smoothing would blur."""
        from humpback.classifier.detector_utils import (
            select_prominent_peaks_from_events,
        )

        # Two clear peaks at indices 2 and 7 with a dip at index 4 (0.90).
        # Logit prominence ~0.98, above min_prominence=0.5.
        offsets = list(range(0, 11))
        confidences = [
            0.7,
            0.85,
            0.96,  # peak 1
            0.93,
            0.90,  # dip
            0.92,
            0.94,
            0.97,  # peak 2
            0.91,
            0.8,
            0.6,
        ]
        window_records = self._make_window_records(offsets, confidences)
        events = [self._make_event(0.0, 15.0, 11)]
        result = select_prominent_peaks_from_events(
            events, window_records, 5.0, min_score=0.7, min_prominence=0.5
        )
        # Should detect two peaks (logit prominence ~0.98 > 0.5)
        assert len(result) == 2
        assert result[0]["start_sec"] == 2.0
        assert result[1]["start_sec"] == 7.0

    def test_preserves_audit_fields(self):
        """Audit fields and extra event fields are preserved."""
        from humpback.classifier.detector_utils import (
            select_prominent_peaks_from_events,
        )

        offsets = [0, 1, 2]
        confidences = [0.5, 0.9, 0.6]
        window_records = self._make_window_records(offsets, confidences)
        events = [
            {
                "start_sec": 0.0,
                "end_sec": 7.0,
                "n_windows": 3,
                "raw_start_sec": 0.3,
                "raw_end_sec": 6.8,
                "merged_event_count": 2,
                "filename": "test.wav",
            }
        ]
        result = select_prominent_peaks_from_events(
            events, window_records, 5.0, min_score=0.7, min_prominence=0.5
        )
        assert len(result) == 1
        assert result[0]["raw_start_sec"] == 0.3
        assert result[0]["raw_end_sec"] == 6.8
        assert result[0]["merged_event_count"] == 2
        assert result[0]["filename"] == "test.wav"

    def test_deduplication_across_events(self):
        """Two events sharing the same peak produce one detection."""
        from humpback.classifier.detector_utils import (
            select_prominent_peaks_from_events,
        )

        window_records = self._make_window_records(
            offsets=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            confidences=[0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
        )
        events = [
            {
                "start_sec": 0.0,
                "end_sec": 10.0,
                "n_windows": 6,
                "raw_start_sec": 1.0,
                "raw_end_sec": 8.0,
                "merged_event_count": 1,
            },
            {
                "start_sec": 5.0,
                "end_sec": 15.0,
                "n_windows": 6,
                "raw_start_sec": 6.0,
                "raw_end_sec": 14.0,
                "merged_event_count": 1,
            },
        ]
        result = select_prominent_peaks_from_events(
            events, window_records, 5.0, min_score=0.7, min_prominence=0.5
        )
        starts = [d["start_sec"] for d in result]
        assert starts.count(5.0) == 1

    def test_high_confidence_plateau_detects_dips_in_logit_space(self):
        """Logit transform detects peaks in high-confidence plateaus.

        In probability space the dip from 0.998 to 0.983 has prominence 0.015,
        which would be invisible at any reasonable threshold.  In logit space
        the same dip produces ~2.15 logit units of prominence.
        """
        from humpback.classifier.detector_utils import (
            select_prominent_peaks_from_events,
        )

        offsets = list(range(0, 5))
        confidences = [0.99, 0.998, 0.983, 0.997, 0.999]
        window_records = self._make_window_records(offsets, confidences)
        events = [self._make_event(0.0, 9.0, 5)]
        result = select_prominent_peaks_from_events(
            events, window_records, 5.0, min_score=0.9, min_prominence=1.0
        )
        # Both peaks (0.998 at index 1, 0.999 at index 4) detected —
        # the 0.983 dip at index 2 has ~2.15 logit prominence, well above 1.0.
        assert len(result) == 2
        assert result[0]["start_sec"] == 1.0
        assert result[1]["start_sec"] == 4.0

    def test_noise_level_wobbles_filtered_at_default_threshold(self):
        """Score jitter (0.997–0.998) does not produce spurious peaks."""
        from humpback.classifier.detector_utils import (
            select_prominent_peaks_from_events,
        )

        offsets = list(range(0, 5))
        confidences = [0.997, 0.998, 0.996, 0.998, 0.997]
        window_records = self._make_window_records(offsets, confidences)
        events = [self._make_event(0.0, 9.0, 5)]
        result = select_prominent_peaks_from_events(
            events, window_records, 5.0, min_score=0.9, min_prominence=1.0
        )
        # Logit prominence of these wobbles is ~0.4, below 1.0.
        # Fallback emits one window.
        assert len(result) == 1
