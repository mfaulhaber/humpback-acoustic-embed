"""Unit tests for detection row-store synchronization helpers."""

from humpback.classifier.detection_rows import (
    ROW_STORE_FIELDNAMES,
    append_detection_row_store,
    apply_label_edits,
    derive_detection_filename,
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
    """apply_label_edits add action creates a new row with UTC keys."""
    rows: list[dict[str, str]] = [
        {f: "" for f in ROW_STORE_FIELDNAMES}
        | {"start_utc": "1000.0", "end_utc": "1005.0"}
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


def test_apply_label_edits_delete() -> None:
    """apply_label_edits delete action removes row by UTC key."""
    rows: list[dict[str, str]] = [
        {f: "" for f in ROW_STORE_FIELDNAMES}
        | {"start_utc": "1000.0", "end_utc": "1005.0"},
        {f: "" for f in ROW_STORE_FIELDNAMES}
        | {"start_utc": "1010.0", "end_utc": "1015.0"},
    ]
    edits = [{"action": "delete", "start_utc": "1000.0", "end_utc": "1005.0"}]
    result = apply_label_edits(rows, edits, job_duration=2000.0)
    assert len(result) == 1
    assert result[0]["start_utc"] == "1010.0"


def test_apply_label_edits_move() -> None:
    """apply_label_edits move action updates start_utc/end_utc."""
    rows: list[dict[str, str]] = [
        {f: "" for f in ROW_STORE_FIELDNAMES}
        | {"start_utc": "1000.0", "end_utc": "1005.0", "humpback": "1"}
    ]
    edits = [
        {
            "action": "move",
            "start_utc": "1000.0",
            "end_utc": "1005.0",
            "new_start_utc": 1020.0,
            "new_end_utc": 1025.0,
        }
    ]
    result = apply_label_edits(rows, edits, job_duration=2000.0)
    assert len(result) == 1
    assert result[0]["start_utc"] == "1020.0"
    assert result[0]["end_utc"] == "1025.0"


def test_apply_label_edits_change_type() -> None:
    """apply_label_edits change_type clears old label, sets new one."""
    rows: list[dict[str, str]] = [
        {f: "" for f in ROW_STORE_FIELDNAMES}
        | {"start_utc": "1000.0", "end_utc": "1005.0", "humpback": "1"}
    ]
    edits = [
        {
            "action": "change_type",
            "start_utc": "1000.0",
            "end_utc": "1005.0",
            "label": "orca",
        }
    ]
    result = apply_label_edits(rows, edits, job_duration=2000.0)
    assert result[0]["humpback"] == ""
    assert result[0]["orca"] == "1"


def test_apply_label_edits_change_type_mismatched_precision() -> None:
    """change_type matches rows even when UTC strings have different decimal precision."""
    rows: list[dict[str, str]] = [
        {f: "" for f in ROW_STORE_FIELDNAMES}
        | {"start_utc": "1635756126.000000", "end_utc": "1635756131.000000"}
    ]
    edits = [
        {
            "action": "change_type",
            "start_utc": 1635756126.0,
            "end_utc": 1635756131.0,
            "label": "humpback",
        }
    ]
    result = apply_label_edits(rows, edits, job_duration=999999.0)
    assert result[0]["humpback"] == "1"


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
    assert "row_id" not in fieldnames
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
