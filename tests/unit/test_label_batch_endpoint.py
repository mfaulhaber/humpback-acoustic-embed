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


def _make_row(
    row_id: str, start: float, end: float, label: str | None = None
) -> dict[str, str]:
    row = {f: "" for f in ROW_STORE_FIELDNAMES}
    row["row_id"] = row_id
    row["start_sec"] = str(start)
    row["end_sec"] = str(end)
    if label:
        row[label] = "1"
    return row


# ---------------------------------------------------------------------------
# Existing null-confidence tests
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# apply_label_edits tests
# ---------------------------------------------------------------------------

JOB_DURATION = 60.0


def test_apply_label_edits_add() -> None:
    """Adding a new labeled row creates it with the correct label."""
    rows = [_make_row("r1", 0, 5)]
    edits = [{"action": "add", "start_sec": 10.0, "end_sec": 15.0, "label": "humpback"}]
    result = apply_label_edits(rows, edits, job_duration=JOB_DURATION)
    added = [r for r in result if r["row_id"] != "r1"]
    assert len(added) == 1
    assert added[0]["start_sec"] == "10.0"
    assert added[0]["end_sec"] == "15.0"
    assert added[0]["humpback"] == "1"
    # Other label fields should be empty
    for lbl in LABEL_FIELDNAMES:
        if lbl != "humpback":
            assert added[0][lbl] == ""


def test_apply_label_edits_move() -> None:
    """Moving a row updates its start_sec and end_sec."""
    rows = [_make_row("r1", 5, 10, "humpback")]
    edits = [
        {
            "action": "move",
            "row_id": "r1",
            "new_start_sec": 15.0,
            "new_end_sec": 20.0,
        }
    ]
    result = apply_label_edits(rows, edits, job_duration=JOB_DURATION)
    assert len(result) == 1
    assert result[0]["start_sec"] == "15.0"
    assert result[0]["end_sec"] == "20.0"


def test_apply_label_edits_delete() -> None:
    """Deleting a row removes it from the result."""
    rows = [_make_row("r1", 0, 5, "humpback"), _make_row("r2", 10, 15, "orca")]
    edits = [{"action": "delete", "row_id": "r1"}]
    result = apply_label_edits(rows, edits, job_duration=JOB_DURATION)
    assert len(result) == 1
    assert result[0]["row_id"] == "r2"


def test_apply_label_edits_change_type() -> None:
    """Changing type sets the target label and clears all others."""
    rows = [_make_row("r1", 0, 5, "humpback")]
    edits = [{"action": "change_type", "row_id": "r1", "label": "orca"}]
    result = apply_label_edits(rows, edits, job_duration=JOB_DURATION)
    assert result[0]["orca"] == "1"
    assert result[0]["humpback"] == ""
    assert result[0]["ship"] == ""
    assert result[0]["background"] == ""


def test_apply_label_edits_overlap_rejected() -> None:
    """Adding a labeled row that overlaps another labeled row raises ValueError."""
    rows = [_make_row("r1", 5, 10, "humpback")]
    edits = [{"action": "add", "start_sec": 8.0, "end_sec": 13.0, "label": "orca"}]
    with pytest.raises(ValueError, match="overlap"):
        apply_label_edits(rows, edits, job_duration=JOB_DURATION)


def test_apply_label_edits_move_out_of_bounds() -> None:
    """Moving a row out of [0, job_duration] raises ValueError."""
    rows = [_make_row("r1", 5, 10, "humpback")]
    # Move before 0
    edits_before = [
        {"action": "move", "row_id": "r1", "new_start_sec": -1.0, "new_end_sec": 4.0}
    ]
    with pytest.raises(ValueError, match="out of bounds"):
        apply_label_edits(rows, edits_before, job_duration=JOB_DURATION)

    # Move after job_duration
    edits_after = [
        {
            "action": "move",
            "row_id": "r1",
            "new_start_sec": 55.0,
            "new_end_sec": 65.0,
        }
    ]
    with pytest.raises(ValueError, match="out of bounds"):
        apply_label_edits(rows, edits_after, job_duration=JOB_DURATION)


def test_apply_label_edits_unlabeled_replacement() -> None:
    """Adding a labeled row over an unlabeled row replaces the unlabeled row."""
    rows = [_make_row("r1", 10, 15)]  # no label = unlabeled
    edits = [{"action": "add", "start_sec": 10.0, "end_sec": 15.0, "label": "humpback"}]
    result = apply_label_edits(rows, edits, job_duration=JOB_DURATION)
    # The unlabeled row should have been removed, replaced by the new one
    assert len(result) == 1
    assert result[0]["humpback"] == "1"
    assert result[0]["row_id"] != "r1"  # new row gets a generated ID


def test_apply_label_edits_single_label_enforcement() -> None:
    """change_type clears all label fields and sets only the target one."""
    row = _make_row("r1", 0, 5)
    row["humpback"] = "1"
    row["orca"] = "1"  # shouldn't normally happen, but enforce clearing
    rows = [row]
    edits = [{"action": "change_type", "row_id": "r1", "label": "ship"}]
    result = apply_label_edits(rows, edits, job_duration=JOB_DURATION)
    assert result[0]["ship"] == "1"
    for lbl in LABEL_FIELDNAMES:
        if lbl != "ship":
            assert result[0][lbl] == ""


def test_apply_label_edits_missing_row_id() -> None:
    """Referencing a non-existent row_id raises ValueError."""
    rows = [_make_row("r1", 0, 5, "humpback")]
    edits = [{"action": "delete", "row_id": "nonexistent"}]
    with pytest.raises(ValueError, match="not found"):
        apply_label_edits(rows, edits, job_duration=JOB_DURATION)
