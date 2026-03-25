"""Tests for null confidence normalization in detection row handling."""

from humpback.classifier.detection_rows import normalize_detection_row


def test_normalize_null_confidence_row() -> None:
    """normalize_detection_row returns None for avg/peak confidence when fields are empty."""
    row = {
        "filename": "test.wav",
        "start_sec": "0.0",
        "end_sec": "5.0",
        "avg_confidence": "",
        "peak_confidence": "",
        "humpback": "1",
    }
    result = normalize_detection_row(row, is_hydrophone=False, window_size_seconds=5.0)
    assert result["avg_confidence"] is None
    assert result["peak_confidence"] is None


def test_normalize_confidence_row_with_values() -> None:
    """normalize_detection_row correctly parses numeric confidence values."""
    row = {
        "filename": "test.wav",
        "start_sec": "0.0",
        "end_sec": "5.0",
        "avg_confidence": "0.85",
        "peak_confidence": "0.92",
    }
    result = normalize_detection_row(row, is_hydrophone=False, window_size_seconds=5.0)
    assert result["avg_confidence"] == 0.85
    assert result["peak_confidence"] == 0.92


def test_normalize_confidence_row_with_none_values() -> None:
    """normalize_detection_row returns None when confidence fields are absent."""
    row = {
        "filename": "test.wav",
        "start_sec": "1.0",
        "end_sec": "6.0",
    }
    result = normalize_detection_row(row, is_hydrophone=False, window_size_seconds=5.0)
    assert result["avg_confidence"] is None
    assert result["peak_confidence"] is None
