"""Tests for Sequence Models timestamp artifact migration."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "migrate_sequence_model_timestamps.py"
)
spec = importlib.util.spec_from_file_location(
    "migrate_sequence_model_timestamps", SCRIPT_PATH
)
assert spec is not None and spec.loader is not None
migration = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = migration
spec.loader.exec_module(migration)


def test_migrate_parquet_converts_relative_legacy_columns(tmp_path):
    path = tmp_path / "embeddings.parquet"
    pq.write_table(
        pa.table(
            {
                "merged_span_id": pa.array([0], type=pa.int32()),
                "start_time_sec": pa.array([10.0], type=pa.float64()),
                "end_time_sec": pa.array([15.0], type=pa.float64()),
                "embedding": pa.array([[1.0, 2.0]], type=pa.list_(pa.float32())),
            }
        ),
        path,
    )

    summary = migration.Summary()
    migration.migrate_parquet(
        path,
        job_start=1000.0,
        job_end=1300.0,
        apply=True,
        summary=summary,
    )

    table = pq.read_table(path)
    assert summary.migrated == 1
    assert "start_time_sec" not in table.column_names
    assert "end_time_sec" not in table.column_names
    assert table.column("start_timestamp").to_pylist() == [1010.0]
    assert table.column("end_timestamp").to_pylist() == [1015.0]


def test_migrate_parquet_dry_run_does_not_rewrite(tmp_path):
    path = tmp_path / "states.parquet"
    pq.write_table(
        pa.table(
            {
                "start_time_sec": pa.array([10.0], type=pa.float64()),
                "end_time_sec": pa.array([15.0], type=pa.float64()),
            }
        ),
        path,
    )

    summary = migration.Summary()
    migration.migrate_parquet(
        path,
        job_start=1000.0,
        job_end=1300.0,
        apply=False,
        summary=summary,
    )

    table = pq.read_table(path)
    assert summary.would_migrate == 1
    assert table.column_names == ["start_time_sec", "end_time_sec"]


def test_migrate_manifest_and_exemplars_rename_epoch_values(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    exemplars_path = tmp_path / "exemplars.json"
    manifest_path.write_text(
        json.dumps({"spans": [{"start_time_sec": 1010.0, "end_time_sec": 1015.0}]}),
        encoding="utf-8",
    )
    exemplars_path.write_text(
        json.dumps(
            {"states": {"0": [{"start_time_sec": 1020.0, "end_time_sec": 1025.0}]}}
        ),
        encoding="utf-8",
    )

    summary = migration.Summary()
    migration.migrate_manifest(
        manifest_path,
        job_start=1000.0,
        job_end=1300.0,
        apply=True,
        summary=summary,
    )
    migration.migrate_exemplars(
        exemplars_path,
        job_start=1000.0,
        job_end=1300.0,
        apply=True,
        summary=summary,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    exemplars = json.loads(exemplars_path.read_text(encoding="utf-8"))
    assert summary.migrated == 2
    assert manifest["spans"][0] == {
        "start_timestamp": 1010.0,
        "end_timestamp": 1015.0,
    }
    assert exemplars["states"]["0"][0] == {
        "start_timestamp": 1020.0,
        "end_timestamp": 1025.0,
    }


def test_ambiguous_parquet_values_fail_loudly(tmp_path):
    path = tmp_path / "ambiguous.parquet"
    pq.write_table(
        pa.table(
            {
                "start_time_sec": pa.array([500.0], type=pa.float64()),
                "end_time_sec": pa.array([505.0], type=pa.float64()),
            }
        ),
        path,
    )

    summary = migration.Summary()
    migration.migrate_parquet(
        path,
        job_start=1000.0,
        job_end=1300.0,
        apply=True,
        summary=summary,
    )

    assert summary.failed == 1
    assert "Ambiguous" in summary.failures[0]
