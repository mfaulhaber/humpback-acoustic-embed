"""Tests for null confidence normalization and batch label edits in detection row handling."""

import pytest

from humpback.classifier.detection_rows import (
    LABEL_FIELDNAMES,
    ROW_STORE_FIELDNAMES,
    apply_label_edits,
    normalize_detection_row,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


_row_counter = 0


def _make_row(
    start_utc: float,
    end_utc: float,
    label: str | None = None,
    row_id: str | None = None,
) -> dict[str, str]:
    global _row_counter  # noqa: PLW0603
    _row_counter += 1
    row = {f: "" for f in ROW_STORE_FIELDNAMES}
    row["row_id"] = row_id or f"test-row-{_row_counter}"
    row["start_utc"] = str(float(start_utc))
    row["end_utc"] = str(float(end_utc))
    if label:
        row[label] = "1"
    return row


# ---------------------------------------------------------------------------
# Existing null-confidence tests
# ---------------------------------------------------------------------------


def test_normalize_null_confidence_row() -> None:
    """normalize_detection_row returns None for avg/peak confidence when fields are empty."""
    row = {
        "start_utc": "1000.0",
        "end_utc": "1005.0",
        "avg_confidence": "",
        "peak_confidence": "",
        "humpback": "1",
    }
    result = normalize_detection_row(row)
    assert result["avg_confidence"] is None
    assert result["peak_confidence"] is None


def test_normalize_confidence_row_with_values() -> None:
    """normalize_detection_row correctly parses numeric confidence values."""
    row = {
        "start_utc": "1000.0",
        "end_utc": "1005.0",
        "avg_confidence": "0.85",
        "peak_confidence": "0.92",
    }
    result = normalize_detection_row(row)
    assert result["avg_confidence"] == 0.85
    assert result["peak_confidence"] == 0.92


def test_normalize_confidence_row_with_none_values() -> None:
    """normalize_detection_row returns None when confidence fields are absent."""
    row = {
        "start_utc": "1001.0",
        "end_utc": "1006.0",
    }
    result = normalize_detection_row(row)
    assert result["avg_confidence"] is None
    assert result["peak_confidence"] is None


# ---------------------------------------------------------------------------
# apply_label_edits tests
# ---------------------------------------------------------------------------

JOB_DURATION = 60.0


def test_apply_label_edits_add() -> None:
    """Adding a new labeled row creates it with the correct label."""
    rows = [_make_row(0, 5)]
    edits = [{"action": "add", "start_utc": 10.0, "end_utc": 15.0, "label": "humpback"}]
    result = apply_label_edits(rows, edits, job_duration=JOB_DURATION)
    added = [r for r in result if r.get("start_utc") == "10.0"]
    assert len(added) == 1
    assert added[0]["start_utc"] == "10.0"
    assert added[0]["end_utc"] == "15.0"
    assert added[0]["humpback"] == "1"
    # Other label fields should be empty
    for lbl in LABEL_FIELDNAMES:
        if lbl != "humpback":
            assert added[0][lbl] == ""


def test_apply_label_edits_move() -> None:
    """Moving a row updates its start_utc and end_utc."""
    rows = [_make_row(5, 10, "humpback", row_id="m1")]
    edits = [
        {
            "action": "move",
            "row_id": "m1",
            "start_utc": 15.0,
            "end_utc": 20.0,
        }
    ]
    result = apply_label_edits(rows, edits, job_duration=JOB_DURATION)
    assert len(result) == 1
    assert result[0]["start_utc"] == "15.0"
    assert result[0]["end_utc"] == "20.0"


def test_apply_label_edits_delete() -> None:
    """Deleting a row removes it from the result."""
    rows = [
        _make_row(0, 5, "humpback", row_id="d1"),
        _make_row(10, 15, "orca", row_id="d2"),
    ]
    edits = [{"action": "delete", "row_id": "d1"}]
    result = apply_label_edits(rows, edits, job_duration=JOB_DURATION)
    assert len(result) == 1
    assert result[0]["start_utc"] == "10.0"


def test_apply_label_edits_change_type() -> None:
    """Changing type sets the target label and clears all others."""
    rows = [_make_row(0, 5, "humpback", row_id="ct1")]
    edits = [{"action": "change_type", "row_id": "ct1", "label": "orca"}]
    result = apply_label_edits(rows, edits, job_duration=JOB_DURATION)
    assert result[0]["orca"] == "1"
    assert result[0]["humpback"] == ""
    assert result[0]["ship"] == ""
    assert result[0]["background"] == ""


def test_apply_label_edits_overlap_rejected() -> None:
    """Two adds in the same batch that overlap are rejected."""
    rows: list[dict[str, str]] = []
    edits = [
        {"action": "add", "start_utc": 5.0, "end_utc": 10.0, "label": "humpback"},
        {"action": "add", "start_utc": 8.0, "end_utc": 13.0, "label": "orca"},
    ]
    with pytest.raises(ValueError, match="overlap"):
        apply_label_edits(rows, edits, job_duration=JOB_DURATION)


def test_apply_label_edits_move_succeeds() -> None:
    """Moving a row updates its UTC coordinates."""
    rows = [_make_row(5, 10, "humpback", row_id="ms1")]
    edits = [
        {
            "action": "move",
            "row_id": "ms1",
            "start_utc": 55.0,
            "end_utc": 60.0,
        }
    ]
    result = apply_label_edits(rows, edits, job_duration=JOB_DURATION)
    assert len(result) == 1
    assert result[0]["start_utc"] == "55.0"
    assert result[0]["end_utc"] == "60.0"


def test_apply_label_edits_unlabeled_replacement() -> None:
    """Adding a labeled row over an unlabeled row replaces the unlabeled row."""
    rows = [_make_row(10, 15)]  # no label = unlabeled
    edits = [{"action": "add", "start_utc": 10.0, "end_utc": 15.0, "label": "humpback"}]
    result = apply_label_edits(rows, edits, job_duration=JOB_DURATION)
    # The unlabeled row should have been removed, replaced by the new one
    assert len(result) == 1
    assert result[0]["humpback"] == "1"


def test_apply_label_edits_single_label_enforcement() -> None:
    """change_type clears all label fields and sets only the target one."""
    row = _make_row(0, 5, row_id="sle1")
    row["humpback"] = "1"
    row["orca"] = "1"
    rows = [row]
    edits = [{"action": "change_type", "row_id": "sle1", "label": "ship"}]
    result = apply_label_edits(rows, edits, job_duration=JOB_DURATION)
    assert result[0]["ship"] == "1"
    for lbl in LABEL_FIELDNAMES:
        if lbl != "ship":
            assert result[0][lbl] == ""


def test_apply_label_edits_missing_row_id() -> None:
    """Referencing a non-existent row_id raises ValueError."""
    rows = [_make_row(0, 5, "humpback")]
    edits = [{"action": "delete", "row_id": "nonexistent"}]
    with pytest.raises(ValueError, match="not found"):
        apply_label_edits(rows, edits, job_duration=JOB_DURATION)


def test_apply_label_edits_preexisting_overlaps_tolerated() -> None:
    """Pre-existing overlapping labeled rows do not block unrelated edits."""
    rows = [
        _make_row(0, 5, "humpback"),
        _make_row(3, 8, "humpback"),  # overlaps first
        _make_row(50, 55),  # unlabeled
    ]
    edits = [{"action": "add", "start_utc": 40.0, "end_utc": 45.0, "label": "orca"}]
    result = apply_label_edits(rows, edits, job_duration=JOB_DURATION)
    labels = [r for r in result if r.get("orca") == "1"]
    assert len(labels) == 1
    assert labels[0]["start_utc"] == "40.0"


def test_apply_label_edits_new_overlap_still_rejected() -> None:
    """Two adds in the same batch that overlap each other are still rejected."""
    rows: list[dict[str, str]] = []
    edits = [
        {"action": "add", "start_utc": 10.0, "end_utc": 15.0, "label": "humpback"},
        {"action": "add", "start_utc": 12.0, "end_utc": 17.0, "label": "orca"},
    ]
    with pytest.raises(ValueError, match="overlap"):
        apply_label_edits(rows, edits, job_duration=JOB_DURATION)


def test_apply_label_edits_add_overlapping_existing_tolerated() -> None:
    """Adding a label that overlaps a pre-existing labeled row is tolerated."""
    rows = [_make_row(10, 15, "humpback")]
    edits = [{"action": "add", "start_utc": 12.0, "end_utc": 17.0, "label": "orca"}]
    result = apply_label_edits(rows, edits, job_duration=JOB_DURATION)
    labels = [r for r in result if r.get("orca") == "1"]
    assert len(labels) == 1
