"""Unit tests for detection row-store synchronization helpers."""

import csv

from humpback.classifier.detection_rows import (
    append_detection_row_store,
    ensure_detection_row_store,
    read_detection_row_store,
    write_detection_row_store,
)


def _write_detection_tsv(path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "filename",
        "start_sec",
        "end_sec",
        "avg_confidence",
        "peak_confidence",
        "n_windows",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def test_append_detection_row_store(tmp_path) -> None:
    """append_detection_row_store creates a new file and appends incrementally."""
    path = tmp_path / "rows.parquet"
    row1 = {"row_id": "a", "filename": "f1", "start_sec": "0", "end_sec": "5"}
    row2 = {"row_id": "b", "filename": "f2", "start_sec": "5", "end_sec": "10"}
    append_detection_row_store(path, [row1])
    _, rows = read_detection_row_store(path)
    assert len(rows) == 1
    append_detection_row_store(path, [row2])
    _, rows = read_detection_row_store(path)
    assert len(rows) == 2
    assert rows[1]["row_id"] == "b"


def test_append_detection_row_store_empty_list(tmp_path) -> None:
    """Appending an empty list to a non-existent file creates an empty row store."""
    path = tmp_path / "rows.parquet"
    append_detection_row_store(path, [])
    _, rows = read_detection_row_store(path)
    assert len(rows) == 0


def test_refresh_detection_row_store_merges_existing_editable_state(tmp_path) -> None:
    """Refreshing from TSV should preserve prior labels, manual bounds, and extracts."""
    tsv_path = tmp_path / "detections.tsv"
    row_store_path = tmp_path / "detection_rows.parquet"

    _write_detection_tsv(
        tsv_path,
        [
            {
                "filename": "20250701T000000Z.wav",
                "start_sec": "0.0",
                "end_sec": "10.0",
                "avg_confidence": "0.90",
                "peak_confidence": "0.95",
                "n_windows": "3",
            }
        ],
    )

    _fieldnames, rows = ensure_detection_row_store(
        row_store_path=row_store_path,
        diagnostics_path=None,
        is_hydrophone=False,
        window_size_seconds=5.0,
        tsv_path=tsv_path,
    )

    rows[0]["humpback"] = "1"
    rows[0]["manual_positive_selection_start_sec"] = "0.000000"
    rows[0]["manual_positive_selection_end_sec"] = "10.000000"
    rows[0]["positive_extract_filename"] = "20250701T000000Z_20250701T000010Z.flac"
    write_detection_row_store(row_store_path, rows)

    _write_detection_tsv(
        tsv_path,
        [
            {
                "filename": "20250701T000000Z.wav",
                "start_sec": "0.0",
                "end_sec": "10.0",
                "avg_confidence": "0.90",
                "peak_confidence": "0.95",
                "n_windows": "3",
            },
            {
                "filename": "20250701T000000Z.wav",
                "start_sec": "15.0",
                "end_sec": "20.0",
                "avg_confidence": "0.83",
                "peak_confidence": "0.88",
                "n_windows": "1",
            },
        ],
    )

    _fieldnames, refreshed_rows = ensure_detection_row_store(
        row_store_path=row_store_path,
        diagnostics_path=None,
        is_hydrophone=False,
        window_size_seconds=5.0,
        refresh_existing=True,
        tsv_path=tsv_path,
    )

    assert len(refreshed_rows) == 2

    original_row = next(row for row in refreshed_rows if row["start_sec"] == "0.000000")
    assert original_row["humpback"] == "1"
    assert original_row["manual_positive_selection_start_sec"] == "0.000000"
    assert original_row["manual_positive_selection_end_sec"] == "10.000000"
    assert original_row["positive_selection_origin"] == "manual_override"
    assert (
        original_row["positive_extract_filename"]
        == "20250701T000000Z_20250701T000010Z.flac"
    )

    new_row = next(row for row in refreshed_rows if row["start_sec"] == "15.000000")
    assert new_row["row_id"]
    assert new_row["humpback"] == ""
    assert new_row["manual_positive_selection_start_sec"] == ""
    assert new_row["positive_extract_filename"] == ""


def test_ensure_detection_row_store_no_tsv_returns_empty(tmp_path) -> None:
    """Without TSV or row store, ensure returns empty rows."""
    row_store_path = tmp_path / "detection_rows.parquet"
    _fieldnames, rows = ensure_detection_row_store(
        row_store_path=row_store_path,
        diagnostics_path=None,
        is_hydrophone=False,
        window_size_seconds=5.0,
    )
    assert rows == []
    # Row store file should have been created (empty)
    assert row_store_path.is_file()


def test_ensure_detection_row_store_reads_existing_parquet(tmp_path) -> None:
    """If row store already exists, ensure reads it directly (no TSV needed)."""
    row_store_path = tmp_path / "detection_rows.parquet"
    row = {
        "row_id": "abc",
        "filename": "test.wav",
        "start_sec": "0.000000",
        "end_sec": "5.000000",
    }
    write_detection_row_store(row_store_path, [row])

    _fieldnames, rows = ensure_detection_row_store(
        row_store_path=row_store_path,
        diagnostics_path=None,
        is_hydrophone=False,
        window_size_seconds=5.0,
    )
    assert len(rows) == 1
    assert rows[0]["row_id"] == "abc"
