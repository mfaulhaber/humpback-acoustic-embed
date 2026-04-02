"""Tests for the migrate_row_ids script."""

import pyarrow as pa
import pyarrow.parquet as pq

from humpback.classifier.detection_rows import (
    ROW_STORE_FIELDNAMES,
    read_detection_row_store,
    write_detection_row_store,
)
from scripts.migrate_row_ids import (
    migrate_embeddings,
    migrate_inference_output,
    migrate_row_store,
    run_migration,
)

# Base epoch: 2024-06-15T08:00:00Z (matches "20240615T080000Z.wav")
BASE_EPOCH = 1718438400.0


def _make_row_store(path, rows):
    """Write a row store parquet using the standard helper (includes row_id)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_detection_row_store(path, rows)


def _make_legacy_row_store(path, rows):
    """Write a legacy row store parquet WITHOUT the row_id column."""
    path.parent.mkdir(parents=True, exist_ok=True)
    legacy_fields = [f for f in ROW_STORE_FIELDNAMES if f != "row_id"]
    schema = pa.schema([(f, pa.string()) for f in legacy_fields])
    normalized = []
    for row in rows:
        normalized.append({f: (row.get(f, "") or None) for f in legacy_fields})
    table = pa.Table.from_pylist(normalized, schema=schema)
    pq.write_table(table, str(path))


def _make_old_embeddings(path, filenames, start_secs, end_secs, dim=4):
    """Write old-schema detection embeddings parquet."""
    import numpy as np

    n = len(filenames)
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
        [
            ("filename", pa.string()),
            ("start_sec", pa.float32()),
            ("end_sec", pa.float32()),
            ("embedding", pa.list_(pa.float32(), dim)),
            ("confidence", pa.float32()),
        ]
    )
    table = pa.table(
        {
            "filename": filenames,
            "start_sec": start_secs,
            "end_sec": end_secs,
            "embedding": [
                np.random.default_rng(i).random(dim).astype(float).tolist()
                for i in range(n)
            ],
            "confidence": [0.9 - i * 0.1 for i in range(n)],
        },
        schema=schema,
    )
    pq.write_table(table, str(path))


def _make_old_inference(path, start_utcs, end_utcs, vocab):
    """Write old-schema inference output parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = {
        "start_utc": start_utcs,
        "end_utc": end_utcs,
    }
    for name in vocab:
        columns[name] = [0.5] * len(start_utcs)
    table = pa.table(columns)
    pq.write_table(table, str(path))


# ---------------------------------------------------------------------------
# Row store migration
# ---------------------------------------------------------------------------


class TestMigrateRowStore:
    def test_assigns_ids_to_rows_without_row_id(self, tmp_path):
        """Rows without row_id get UUIDs assigned."""
        rs_path = tmp_path / "detection_rows.parquet"
        rows = [
            {"start_utc": str(BASE_EPOCH), "end_utc": str(BASE_EPOCH + 5.0)},
            {"start_utc": str(BASE_EPOCH + 5.0), "end_utc": str(BASE_EPOCH + 10.0)},
        ]
        _make_legacy_row_store(rs_path, rows)

        total, assigned, result_rows = migrate_row_store(rs_path)

        assert total == 2
        assert assigned == 2
        for row in result_rows:
            assert row["row_id"]
            assert len(row["row_id"]) > 0

        # Verify written to disk
        _fields, disk_rows = read_detection_row_store(rs_path)
        assert all(r["row_id"] for r in disk_rows)

    def test_preserves_existing_row_ids(self, tmp_path):
        """Rows with existing row_id are not reassigned."""
        rs_path = tmp_path / "detection_rows.parquet"
        # Use standard helper which auto-assigns row_ids, then verify no reassignment
        rows = [
            {
                "row_id": "existing-id-1",
                "start_utc": str(BASE_EPOCH),
                "end_utc": str(BASE_EPOCH + 5.0),
            },
            {
                "row_id": "existing-id-2",
                "start_utc": str(BASE_EPOCH + 5.0),
                "end_utc": str(BASE_EPOCH + 10.0),
            },
        ]
        _make_row_store(rs_path, rows)

        total, assigned, result_rows = migrate_row_store(rs_path)

        assert total == 2
        assert assigned == 0  # Both already have IDs
        assert result_rows[0]["row_id"] == "existing-id-1"
        assert result_rows[1]["row_id"] == "existing-id-2"

    def test_idempotent(self, tmp_path):
        """Running twice produces the same result."""
        rs_path = tmp_path / "detection_rows.parquet"
        rows = [
            {"start_utc": str(BASE_EPOCH), "end_utc": str(BASE_EPOCH + 5.0)},
        ]
        _make_legacy_row_store(rs_path, rows)

        _, assigned1, rows1 = migrate_row_store(rs_path)
        assert assigned1 == 1
        first_id = rows1[0]["row_id"]

        _, assigned2, rows2 = migrate_row_store(rs_path)
        assert assigned2 == 0  # No new assignments
        assert rows2[0]["row_id"] == first_id  # Same ID preserved


# ---------------------------------------------------------------------------
# Embedding migration
# ---------------------------------------------------------------------------


class TestMigrateEmbeddings:
    def test_matches_and_rewrites(self, tmp_path):
        """Old embedding entries are matched to row store and rewritten with row_id."""
        rs_path = tmp_path / "detection_rows.parquet"
        emb_path = tmp_path / "detection_embeddings.parquet"

        rows = [
            {
                "row_id": "row-aaa",
                "start_utc": str(BASE_EPOCH),
                "end_utc": str(BASE_EPOCH + 5.0),
            },
            {
                "row_id": "row-bbb",
                "start_utc": str(BASE_EPOCH + 5.0),
                "end_utc": str(BASE_EPOCH + 10.0),
            },
        ]
        _make_row_store(rs_path, rows)
        _, _, store_rows = migrate_row_store(rs_path)

        fname = "20240615T080000Z.wav"
        _make_old_embeddings(emb_path, [fname, fname], [0.0, 5.0], [5.0, 10.0])

        total, matched, dropped = migrate_embeddings(emb_path, store_rows)

        assert total == 2
        assert matched == 2
        assert dropped == 0

        # Verify new schema
        table = pq.read_table(str(emb_path))
        assert "row_id" in table.column_names
        assert "filename" not in table.column_names
        assert "start_sec" not in table.column_names
        assert set(table.column("row_id").to_pylist()) == {"row-aaa", "row-bbb"}

    def test_drops_unmatched(self, tmp_path):
        """Embedding entries that don't match any row store entry are dropped."""
        rs_path = tmp_path / "detection_rows.parquet"
        emb_path = tmp_path / "detection_embeddings.parquet"

        rows = [
            {
                "row_id": "row-aaa",
                "start_utc": str(BASE_EPOCH),
                "end_utc": str(BASE_EPOCH + 5.0),
            },
        ]
        _make_row_store(rs_path, rows)
        _, _, store_rows = migrate_row_store(rs_path)

        fname = "20240615T080000Z.wav"
        # Second entry at 99s won't match any row
        _make_old_embeddings(emb_path, [fname, fname], [0.0, 99.0], [5.0, 104.0])

        total, matched, dropped = migrate_embeddings(emb_path, store_rows)

        assert total == 2
        assert matched == 1
        assert dropped == 1

        table = pq.read_table(str(emb_path))
        assert table.num_rows == 1
        assert table.column("row_id")[0].as_py() == "row-aaa"

    def test_skips_already_migrated(self, tmp_path):
        """Embeddings already in row_id schema are left alone."""
        emb_path = tmp_path / "detection_embeddings.parquet"
        schema = pa.schema(
            [
                ("row_id", pa.string()),
                ("embedding", pa.list_(pa.float32(), 4)),
                ("confidence", pa.float32()),
            ]
        )
        table = pa.table(
            {
                "row_id": ["r1"],
                "embedding": [[1.0, 2.0, 3.0, 4.0]],
                "confidence": [0.9],
            },
            schema=schema,
        )
        pq.write_table(table, str(emb_path))

        total, matched, dropped = migrate_embeddings(emb_path, [])

        assert total == 1
        assert matched == 1
        assert dropped == 0

    def test_idempotent(self, tmp_path):
        """Running embedding migration twice produces the same result."""
        rs_path = tmp_path / "detection_rows.parquet"
        emb_path = tmp_path / "detection_embeddings.parquet"

        rows = [
            {
                "row_id": "row-aaa",
                "start_utc": str(BASE_EPOCH),
                "end_utc": str(BASE_EPOCH + 5.0),
            },
        ]
        _make_row_store(rs_path, rows)
        _, _, store_rows = migrate_row_store(rs_path)

        fname = "20240615T080000Z.wav"
        _make_old_embeddings(emb_path, [fname], [0.0], [5.0])

        migrate_embeddings(emb_path, store_rows)
        # Second run should detect already-migrated and skip
        total, matched, dropped = migrate_embeddings(emb_path, store_rows)
        assert matched == 1
        assert dropped == 0


# ---------------------------------------------------------------------------
# Inference output migration
# ---------------------------------------------------------------------------


class TestMigrateInferenceOutput:
    def test_matches_by_utc(self, tmp_path):
        """Inference entries with (start_utc, end_utc) are matched to row_id."""
        inf_path = tmp_path / "predictions.parquet"

        rows = [
            {
                "row_id": "row-aaa",
                "start_utc": str(BASE_EPOCH),
                "end_utc": str(BASE_EPOCH + 5.0),
            },
        ]

        _make_old_inference(inf_path, [BASE_EPOCH], [BASE_EPOCH + 5.0], ["Moan"])

        total, matched, dropped = migrate_inference_output(inf_path, rows)

        assert total == 1
        assert matched == 1
        assert dropped == 0

        table = pq.read_table(str(inf_path))
        assert "row_id" in table.column_names
        assert "start_utc" not in table.column_names
        assert table.column("row_id")[0].as_py() == "row-aaa"
        assert "Moan" in table.column_names

    def test_drops_unmatched(self, tmp_path):
        """Inference entries that don't match are dropped."""
        inf_path = tmp_path / "predictions.parquet"

        rows = [
            {
                "row_id": "row-aaa",
                "start_utc": str(BASE_EPOCH),
                "end_utc": str(BASE_EPOCH + 5.0),
            },
        ]

        _make_old_inference(
            inf_path,
            [BASE_EPOCH, 9999999.0],
            [BASE_EPOCH + 5.0, 9999999.0 + 5.0],
            ["Moan"],
        )

        total, matched, dropped = migrate_inference_output(inf_path, rows)

        assert total == 2
        assert matched == 1
        assert dropped == 1


# ---------------------------------------------------------------------------
# Full migration
# ---------------------------------------------------------------------------


class TestRunMigration:
    def test_full_migration(self, tmp_path):
        """End-to-end migration processes row store and embeddings."""
        storage_root = tmp_path / "storage"
        det_dir = storage_root / "detections" / "job1"
        det_dir.mkdir(parents=True)

        rs_path = det_dir / "detection_rows.parquet"
        rows = [
            {"start_utc": str(BASE_EPOCH), "end_utc": str(BASE_EPOCH + 5.0)},
            {"start_utc": str(BASE_EPOCH + 5.0), "end_utc": str(BASE_EPOCH + 10.0)},
        ]
        _make_legacy_row_store(rs_path, rows)

        fname = "20240615T080000Z.wav"
        emb_path = det_dir / "detection_embeddings.parquet"
        _make_old_embeddings(emb_path, [fname, fname], [0.0, 5.0], [5.0, 10.0])

        stats = run_migration(storage_root)

        assert stats["jobs_processed"] == 1
        assert stats["rows_total"] == 2
        assert stats["rows_assigned_ids"] == 2
        assert stats["embeddings_matched"] == 2
        assert stats["embeddings_dropped"] == 0

        # Verify row store has IDs
        _fields, disk_rows = read_detection_row_store(rs_path)
        assert all(r["row_id"] for r in disk_rows)

        # Verify embeddings have new schema
        table = pq.read_table(str(emb_path))
        assert "row_id" in table.column_names
        assert "filename" not in table.column_names

    def test_skips_missing_row_store(self, tmp_path):
        """Jobs without a row store file are skipped."""
        storage_root = tmp_path / "storage"
        det_dir = storage_root / "detections" / "job1"
        det_dir.mkdir(parents=True)
        # No row store file — just an empty dir

        stats = run_migration(storage_root)

        assert stats["jobs_processed"] == 0
        assert stats["jobs_skipped_no_row_store"] == 1

    def test_idempotent_full(self, tmp_path):
        """Running full migration twice produces the same result."""
        storage_root = tmp_path / "storage"
        det_dir = storage_root / "detections" / "job1"
        det_dir.mkdir(parents=True)

        rs_path = det_dir / "detection_rows.parquet"
        rows = [
            {"start_utc": str(BASE_EPOCH), "end_utc": str(BASE_EPOCH + 5.0)},
        ]
        _make_legacy_row_store(rs_path, rows)

        fname = "20240615T080000Z.wav"
        emb_path = det_dir / "detection_embeddings.parquet"
        _make_old_embeddings(emb_path, [fname], [0.0], [5.0])

        stats1 = run_migration(storage_root)
        assert stats1["rows_assigned_ids"] == 1
        assert stats1["embeddings_matched"] == 1

        # Read assigned IDs
        _fields, disk_rows = read_detection_row_store(rs_path)
        first_id = disk_rows[0]["row_id"]

        stats2 = run_migration(storage_root)
        assert stats2["rows_assigned_ids"] == 0  # No new assignments
        assert stats2["embeddings_matched"] == 1  # Already migrated, counted as matched

        # Same ID preserved
        _fields, disk_rows2 = read_detection_row_store(rs_path)
        assert disk_rows2[0]["row_id"] == first_id
