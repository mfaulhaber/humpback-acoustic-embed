"""Unit tests for detection row-store synchronization helpers."""

import uuid

from humpback.classifier.detection_rows import (
    ROW_STORE_FIELDNAMES,
    append_detection_row_store,
    apply_label_edits,
    derive_detection_filename,
    ensure_row_ids,
    normalize_detection_row,
    read_detection_row_store,
    write_detection_row_store,
)


def test_append_detection_row_store(tmp_path) -> None:
    """append_detection_row_store creates a new file and appends incrementally."""
    path = tmp_path / "rows.parquet"
    row1 = {"start_utc": "1000.0", "end_utc": "1005.0"}
    row2 = {"start_utc": "1005.0", "end_utc": "1010.0"}
    append_detection_row_store(path, [row1])
    _, rows = read_detection_row_store(path)
    assert len(rows) == 1
    append_detection_row_store(path, [row2])
    _, rows = read_detection_row_store(path)
    assert len(rows) == 2
    assert rows[1]["start_utc"] == "1005.0"


def test_append_detection_row_store_empty_list(tmp_path) -> None:
    """Appending an empty list to a non-existent file creates an empty row store."""
    path = tmp_path / "rows.parquet"
    append_detection_row_store(path, [])
    _, rows = read_detection_row_store(path)
    assert len(rows) == 0


def test_write_read_roundtrip(tmp_path) -> None:
    """Write and read back a row store, verifying field integrity."""
    path = tmp_path / "rows.parquet"
    row = {
        "start_utc": "1719792000.0",
        "end_utc": "1719792010.0",
        "avg_confidence": "0.90",
        "peak_confidence": "0.95",
        "n_windows": "3",
        "raw_start_utc": "1719792001.0",
        "raw_end_utc": "1719792009.0",
        "merged_event_count": "1",
        "hydrophone_name": "rpi_north_sjc",
        "humpback": "1",
    }
    write_detection_row_store(path, [row])
    fieldnames, rows = read_detection_row_store(path)
    assert fieldnames == ROW_STORE_FIELDNAMES
    assert len(rows) == 1
    assert rows[0]["start_utc"] == "1719792000.0"
    assert rows[0]["end_utc"] == "1719792010.0"
    assert rows[0]["hydrophone_name"] == "rpi_north_sjc"
    assert rows[0]["humpback"] == "1"
    assert rows[0]["orca"] == ""


def test_normalize_detection_row_utc() -> None:
    """normalize_detection_row parses UTC float fields."""
    row = {"start_utc": "1719792000.0", "end_utc": "1719792010.0", "n_windows": "3"}
    result = normalize_detection_row(row)
    assert result["start_utc"] == 1719792000.0
    assert result["end_utc"] == 1719792010.0
    assert result["n_windows"] == 3


def test_normalize_detection_row_includes_row_id() -> None:
    """normalize_detection_row preserves row_id for frontend selection."""
    row = {
        "row_id": "abc-123",
        "start_utc": "100.0",
        "end_utc": "105.0",
    }
    result = normalize_detection_row(row)
    assert result["row_id"] == "abc-123"


def test_normalize_detection_row_missing_row_id() -> None:
    """normalize_detection_row returns empty string when row_id is absent."""
    row = {"start_utc": "100.0", "end_utc": "105.0"}
    result = normalize_detection_row(row)
    assert result["row_id"] == ""


def test_derive_detection_filename() -> None:
    """derive_detection_filename formats UTC epochs to compact filename."""
    # 2024-07-01 00:00:00 UTC = 1719792000.0
    # 2024-07-01 00:00:10 UTC = 1719792010.0
    result = derive_detection_filename(1719792000.0, 1719792010.0)
    assert result == "20240701T000000Z_20240701T000010Z.flac"


def test_derive_detection_filename_invalid() -> None:
    """derive_detection_filename returns None for invalid ranges."""
    assert derive_detection_filename(10.0, 5.0) is None
    assert derive_detection_filename(10.0, 10.0) is None


def test_apply_label_edits_add() -> None:
    """apply_label_edits add action creates a new row with a generated row_id."""
    rows: list[dict[str, str]] = [
        {f: "" for f in ROW_STORE_FIELDNAMES}
        | {"row_id": "r1", "start_utc": "1000.0", "end_utc": "1005.0"}
    ]
    edits = [
        {"action": "add", "start_utc": 1010.0, "end_utc": 1015.0, "label": "humpback"}
    ]
    result = apply_label_edits(rows, edits, job_duration=2000.0)
    assert len(result) == 2
    new_row = result[1]
    assert new_row["start_utc"] == "1010.0"
    assert new_row["end_utc"] == "1015.0"
    assert new_row["humpback"] == "1"
    # Add action gets a new UUID row_id
    assert new_row["row_id"] != ""
    uuid.UUID(new_row["row_id"])


def test_apply_label_edits_delete() -> None:
    """apply_label_edits delete action removes row by row_id."""
    rows: list[dict[str, str]] = [
        {f: "" for f in ROW_STORE_FIELDNAMES}
        | {"row_id": "r1", "start_utc": "1000.0", "end_utc": "1005.0"},
        {f: "" for f in ROW_STORE_FIELDNAMES}
        | {"row_id": "r2", "start_utc": "1010.0", "end_utc": "1015.0"},
    ]
    edits = [{"action": "delete", "row_id": "r1"}]
    result = apply_label_edits(rows, edits, job_duration=2000.0)
    assert len(result) == 1
    assert result[0]["start_utc"] == "1010.0"


def test_apply_label_edits_move() -> None:
    """apply_label_edits move action updates start_utc/end_utc, preserves row_id."""
    rows: list[dict[str, str]] = [
        {f: "" for f in ROW_STORE_FIELDNAMES}
        | {"row_id": "r1", "start_utc": "1000.0", "end_utc": "1005.0", "humpback": "1"}
    ]
    edits = [
        {
            "action": "move",
            "row_id": "r1",
            "start_utc": 1020.0,
            "end_utc": 1025.0,
        }
    ]
    result = apply_label_edits(rows, edits, job_duration=2000.0)
    assert len(result) == 1
    assert result[0]["start_utc"] == "1020.0"
    assert result[0]["end_utc"] == "1025.0"
    assert result[0]["row_id"] == "r1"


def test_apply_label_edits_change_type() -> None:
    """apply_label_edits change_type clears old label, sets new one."""
    rows: list[dict[str, str]] = [
        {f: "" for f in ROW_STORE_FIELDNAMES}
        | {"row_id": "r1", "start_utc": "1000.0", "end_utc": "1005.0", "humpback": "1"}
    ]
    edits = [
        {
            "action": "change_type",
            "row_id": "r1",
            "label": "orca",
        }
    ]
    result = apply_label_edits(rows, edits, job_duration=2000.0)
    assert result[0]["humpback"] == ""
    assert result[0]["orca"] == "1"


def test_apply_label_edits_clear_label() -> None:
    """apply_label_edits clear_label sets all label fields to empty."""
    rows: list[dict[str, str]] = [
        {f: "" for f in ROW_STORE_FIELDNAMES}
        | {"row_id": "r1", "start_utc": "1000.0", "end_utc": "1005.0", "humpback": "1"}
    ]
    edits = [{"action": "clear_label", "row_id": "r1"}]
    result = apply_label_edits(rows, edits, job_duration=2000.0)
    assert result[0]["humpback"] == ""
    assert result[0]["orca"] == ""
    assert result[0]["ship"] == ""
    assert result[0]["background"] == ""


def test_apply_label_edits_clear_label_already_unlabeled() -> None:
    """clear_label on an already-unlabeled row is a no-op."""
    rows: list[dict[str, str]] = [
        {f: "" for f in ROW_STORE_FIELDNAMES}
        | {"row_id": "r1", "start_utc": "1000.0", "end_utc": "1005.0"}
    ]
    edits = [{"action": "clear_label", "row_id": "r1"}]
    result = apply_label_edits(rows, edits, job_duration=2000.0)
    assert result[0]["humpback"] == ""
    assert result[0]["orca"] == ""


def test_apply_label_edits_unknown_row_id() -> None:
    """apply_label_edits raises ValueError for unknown row_id."""
    import pytest

    rows: list[dict[str, str]] = [
        {f: "" for f in ROW_STORE_FIELDNAMES}
        | {"row_id": "r1", "start_utc": "1000.0", "end_utc": "1005.0"}
    ]
    with pytest.raises(ValueError, match="not found"):
        apply_label_edits(
            rows,
            [{"action": "delete", "row_id": "nonexistent"}],
            job_duration=2000.0,
        )


def test_legacy_parquet_migration_hydrophone(tmp_path) -> None:
    """Old-schema Parquet with detection_filename is lazily migrated to UTC."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    old_fieldnames = [
        "row_id",
        "filename",
        "start_sec",
        "end_sec",
        "avg_confidence",
        "peak_confidence",
        "n_windows",
        "raw_start_sec",
        "raw_end_sec",
        "merged_event_count",
        "detection_filename",
        "extract_filename",
        "hydrophone_name",
        "humpback",
        "orca",
        "ship",
        "background",
    ]
    old_schema = pa.schema([(f, pa.string()) for f in old_fieldnames])
    old_row = {
        "row_id": "abc123",
        "filename": "20240701T000000Z.wav",
        "start_sec": "10.0",
        "end_sec": "20.0",
        "avg_confidence": "0.90",
        "peak_confidence": "0.95",
        "n_windows": "3",
        "raw_start_sec": "9.5",
        "raw_end_sec": "20.5",
        "merged_event_count": "1",
        "detection_filename": "20240701T000010Z_20240701T000020Z.flac",
        "extract_filename": "20240701T000010Z_20240701T000020Z.flac",
        "hydrophone_name": "rpi_north",
        "humpback": "1",
        "orca": "",
        "ship": "",
        "background": "",
    }
    table = pa.Table.from_pylist([old_row], schema=old_schema)
    path = tmp_path / "detection_rows.parquet"
    pq.write_table(table, path)

    # Reading should trigger lazy migration
    fieldnames, rows = read_detection_row_store(path)
    assert "start_utc" in fieldnames
    assert "row_id" in fieldnames
    assert "filename" not in fieldnames
    assert len(rows) == 1
    row = rows[0]

    # detection_filename "20240701T000010Z_20240701T000020Z.flac" →
    # start_utc = 2024-07-01T00:00:10Z = 1719792010.0
    # end_utc = 2024-07-01T00:00:20Z = 1719792020.0
    assert abs(float(row["start_utc"]) - 1719792010.0) < 1.0
    assert abs(float(row["end_utc"]) - 1719792020.0) < 1.0
    assert row["hydrophone_name"] == "rpi_north"
    assert row["humpback"] == "1"

    # Verify the file was rewritten in new schema (second read should not re-migrate)
    fieldnames2, rows2 = read_detection_row_store(path)
    assert "start_utc" in fieldnames2
    assert len(rows2) == 1


def test_legacy_parquet_migration_local(tmp_path) -> None:
    """Old-schema Parquet with filename+offsets (no detection_filename) migrates via anchor."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    old_fieldnames = [
        "row_id",
        "filename",
        "start_sec",
        "end_sec",
        "avg_confidence",
        "detection_filename",
    ]
    old_schema = pa.schema([(f, pa.string()) for f in old_fieldnames])
    old_row = {
        "row_id": "def456",
        "filename": "20240701T000000Z.wav",
        "start_sec": "5.0",
        "end_sec": "10.0",
        "avg_confidence": "0.80",
        "detection_filename": "",
    }
    table = pa.Table.from_pylist([old_row], schema=old_schema)
    path = tmp_path / "detection_rows.parquet"
    pq.write_table(table, path)

    fieldnames, rows = read_detection_row_store(path)
    assert "start_utc" in fieldnames
    row = rows[0]
    # filename "20240701T000000Z.wav" + start_sec 5.0 → start_utc = 1719792005.0
    assert abs(float(row["start_utc"]) - 1719792005.0) < 1.0
    assert abs(float(row["end_utc"]) - 1719792010.0) < 1.0


def test_row_id_assigned_on_write(tmp_path) -> None:
    """Rows without row_id get a UUID assigned during write."""
    path = tmp_path / "rows.parquet"
    row = {"start_utc": "1000.0", "end_utc": "1005.0"}
    write_detection_row_store(path, [row])
    _, rows = read_detection_row_store(path)
    assert len(rows) == 1
    rid = rows[0]["row_id"]
    assert rid != ""
    uuid.UUID(rid)  # validates format


def test_row_id_preserved_on_roundtrip(tmp_path) -> None:
    """An explicit row_id survives write/read round-trip."""
    path = tmp_path / "rows.parquet"
    fixed_id = str(uuid.uuid4())
    row = {"row_id": fixed_id, "start_utc": "1000.0", "end_utc": "1005.0"}
    write_detection_row_store(path, [row])
    _, rows = read_detection_row_store(path)
    assert rows[0]["row_id"] == fixed_id


def test_row_ids_unique(tmp_path) -> None:
    """Multiple rows without row_id each get a unique UUID."""
    path = tmp_path / "rows.parquet"
    rows_in = [
        {"start_utc": "1000.0", "end_utc": "1005.0"},
        {"start_utc": "1005.0", "end_utc": "1010.0"},
        {"start_utc": "1010.0", "end_utc": "1015.0"},
    ]
    write_detection_row_store(path, rows_in)
    _, rows_out = read_detection_row_store(path)
    ids = [r["row_id"] for r in rows_out]
    assert len(set(ids)) == 3


def test_row_id_assigned_on_read_legacy(tmp_path) -> None:
    """Reading a parquet without row_id column assigns UUIDs without rewriting."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    # Write a parquet with the old schema (no row_id column)
    old_fields = [f for f in ROW_STORE_FIELDNAMES if f != "row_id"]
    old_schema = pa.schema([(f, pa.string()) for f in old_fields])
    old_row = {f: "" for f in old_fields}
    old_row["start_utc"] = "2000.0"
    old_row["end_utc"] = "2005.0"
    table = pa.Table.from_pylist([old_row], schema=old_schema)
    path = tmp_path / "rows.parquet"
    pq.write_table(table, path)

    fieldnames, rows = read_detection_row_store(path)
    assert "row_id" in fieldnames
    assert len(rows) == 1
    rid = rows[0]["row_id"]
    assert rid != ""
    uuid.UUID(rid)


def test_append_preserves_existing_row_ids(tmp_path) -> None:
    """Appending new rows does not change row_ids of existing rows."""
    path = tmp_path / "rows.parquet"
    row1 = {"start_utc": "1000.0", "end_utc": "1005.0"}
    write_detection_row_store(path, [row1])
    _, rows_before = read_detection_row_store(path)
    original_id = rows_before[0]["row_id"]

    row2 = {"start_utc": "1005.0", "end_utc": "1010.0"}
    append_detection_row_store(path, [row2])
    _, rows_after = read_detection_row_store(path)
    assert len(rows_after) == 2
    assert rows_after[0]["row_id"] == original_id
    assert rows_after[1]["row_id"] != ""
    assert rows_after[1]["row_id"] != original_id


def test_ensure_row_ids_idempotent() -> None:
    """ensure_row_ids does not overwrite existing IDs."""
    fixed_id = str(uuid.uuid4())
    rows = [{"row_id": fixed_id, "start_utc": "1000.0"}]
    ensure_row_ids(rows)
    assert rows[0]["row_id"] == fixed_id


def test_row_id_in_fieldnames() -> None:
    """row_id is part of ROW_STORE_FIELDNAMES."""
    assert "row_id" in ROW_STORE_FIELDNAMES
    assert ROW_STORE_FIELDNAMES[0] == "row_id"


# ---- Overlapping detection window tests ----


def test_overlapping_rows_write_read_roundtrip(tmp_path) -> None:
    """Overlapping detection rows can be written to and read from Parquet row store."""
    path = tmp_path / "rows.parquet"
    # Two rows with overlapping time ranges (3 seconds of overlap)
    rows_in = [
        {"start_utc": "1000.0", "end_utc": "1005.0", "avg_confidence": "0.95"},
        {"start_utc": "1002.0", "end_utc": "1007.0", "avg_confidence": "0.93"},
    ]
    write_detection_row_store(path, rows_in)
    _, rows_out = read_detection_row_store(path)
    assert len(rows_out) == 2
    assert rows_out[0]["start_utc"] == "1000.0"
    assert rows_out[1]["start_utc"] == "1002.0"


def test_overlapping_rows_get_unique_row_ids(tmp_path) -> None:
    """Overlapping detection rows each get a distinct row_id."""
    path = tmp_path / "rows.parquet"
    rows_in = [
        {"start_utc": "1000.0", "end_utc": "1005.0"},
        {"start_utc": "1002.0", "end_utc": "1007.0"},
        {"start_utc": "1004.0", "end_utc": "1009.0"},
    ]
    write_detection_row_store(path, rows_in)
    _, rows_out = read_detection_row_store(path)
    ids = [r["row_id"] for r in rows_out]
    assert len(set(ids)) == 3
    for rid in ids:
        uuid.UUID(rid)  # valid UUIDs


def test_overlapping_rows_label_independently(tmp_path) -> None:
    """Labeling operations work on overlapping rows independently."""
    path = tmp_path / "rows.parquet"
    rows_in = [
        {"row_id": "r1", "start_utc": "1000.0", "end_utc": "1005.0"},
        {"row_id": "r2", "start_utc": "1002.0", "end_utc": "1007.0"},
    ]
    write_detection_row_store(path, rows_in)
    _, rows = read_detection_row_store(path)

    # Label first row as humpback, second as background
    edits = [
        {"action": "change_type", "row_id": "r1", "label": "humpback"},
        {"action": "change_type", "row_id": "r2", "label": "background"},
    ]
    result = apply_label_edits(rows, edits, job_duration=2000.0)
    assert result[0]["humpback"] == "1"
    assert result[0]["background"] == ""
    assert result[1]["humpback"] == ""
    assert result[1]["background"] == "1"


def test_prominence_vs_nms_side_by_side() -> None:
    """Prominence mode finds more peaks than NMS in dense high-scoring regions."""
    from humpback.classifier.detector_utils import (
        select_peak_windows_from_events,
        select_prominent_peaks_from_events,
    )

    # Dense region with two distinct peaks 3 seconds apart (within NMS suppression zone)
    offsets = list(range(0, 12))
    confidences = [
        0.5,
        0.7,
        0.95,  # peak at 2
        0.6,  # valley
        0.3,
        0.7,
        0.93,  # peak at 6 — only 4s from peak at 2 (< 5s NMS zone)
        0.6,
        0.4,
        0.3,
        0.2,
        0.1,
    ]
    window_records = [
        {"offset_sec": o, "end_sec": o + 5.0, "confidence": c}
        for o, c in zip(offsets, confidences)
    ]
    events = [
        {
            "start_sec": 0.0,
            "end_sec": 16.0,
            "n_windows": 12,
            "raw_start_sec": 0.0,
            "raw_end_sec": 16.0,
            "merged_event_count": 1,
        }
    ]

    nms_result = select_peak_windows_from_events(
        events, window_records, 5.0, min_score=0.7
    )
    prominence_result = select_prominent_peaks_from_events(
        events, window_records, 5.0, min_score=0.7, min_prominence=1.0
    )

    # NMS: peak at 2 suppresses peak at 6 (distance 4 < window_size 5)
    assert len(nms_result) == 1

    # Prominence: both peaks have clear prominence, both detected.
    # Gap-filling may add a window between them (offset 5, score 0.7).
    assert len(prominence_result) >= 2
    # Prominence allows overlapping windows
    starts = [d["start_sec"] for d in prominence_result]
    assert 2.0 in starts
    assert 6.0 in starts
