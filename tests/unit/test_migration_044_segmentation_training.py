"""Tests for migration 044 (Pass 2 segmentation training tables).

Migration 044 adds three new tables — ``segmentation_training_datasets``,
``segmentation_training_samples``, ``segmentation_training_jobs`` — with
two indexes on the samples table and one on the training-jobs table.

Follows the pattern from ``test_migration_043_pass1_source_columns.py``:
build a DB pinned at revision 043, run Alembic upgrade/downgrade, and
inspect the raw sqlite schema via ``PRAGMA table_info`` + ``PRAGMA
index_list``.
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

from alembic import command
from alembic.config import Config

from humpback.database import Base, create_engine


def _db_url(db_path: Path) -> str:
    return f"sqlite+aiosqlite:///{db_path}"


async def _create_db(db_path: Path) -> None:
    engine = create_engine(_db_url(db_path))
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
    finally:
        await engine.dispose()


def _alembic_config(db_path: Path) -> Config:
    repo_root = Path(__file__).resolve().parents[2]
    config = Config(str(repo_root / "alembic.ini"))
    config.set_main_option("script_location", str(repo_root / "alembic"))
    config.set_main_option("sqlalchemy.url", _db_url(db_path))
    return config


def _columns(db_path: Path, table: str) -> dict[str, dict[str, object]]:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    finally:
        conn.close()
    return {row[1]: {"type": row[2].upper(), "notnull": bool(row[3])} for row in rows}


def _indexes(db_path: Path, table: str) -> set[str]:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(f"PRAGMA index_list({table})").fetchall()
    finally:
        conn.close()
    return {row[1] for row in rows}


def _table_exists(db_path: Path, table: str) -> bool:
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchone()
    finally:
        conn.close()
    return row is not None


def _pre_044_schema(db_path: Path) -> None:
    """Build an ORM DB, then drop the three Pass 2 tables and pin to 043.

    ``Base.metadata.create_all`` reflects the post-044 shape (the three
    new tables are part of the ORM). Dropping them and stamping the
    alembic version back to 043 simulates a DB paused just before
    migration 044.
    """
    asyncio.run(_create_db(db_path))
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("DROP TABLE IF EXISTS segmentation_training_jobs")
        conn.execute("DROP TABLE IF EXISTS segmentation_training_samples")
        conn.execute("DROP TABLE IF EXISTS segmentation_training_datasets")

        conn.execute(
            "CREATE TABLE IF NOT EXISTS alembic_version "
            "(version_num VARCHAR(32) NOT NULL)"
        )
        conn.execute("DELETE FROM alembic_version")
        conn.execute("INSERT INTO alembic_version (version_num) VALUES ('043')")
        conn.commit()
    finally:
        conn.close()


def test_migration_044_upgrade_creates_tables(tmp_path: Path) -> None:
    """Upgrade 043 → 044: all three tables + indexes exist."""
    db_path = tmp_path / "test_044_upgrade.db"
    _pre_044_schema(db_path)

    config = _alembic_config(db_path)
    command.upgrade(config, "044")

    for table in (
        "segmentation_training_datasets",
        "segmentation_training_samples",
        "segmentation_training_jobs",
    ):
        assert _table_exists(db_path, table), f"missing table {table} after upgrade"

    dataset_cols = _columns(db_path, "segmentation_training_datasets")
    assert dataset_cols["id"]["notnull"] is True
    assert dataset_cols["name"]["notnull"] is True
    assert dataset_cols["description"]["notnull"] is False
    assert dataset_cols["created_at"]["notnull"] is True
    assert dataset_cols["updated_at"]["notnull"] is True

    sample_cols = _columns(db_path, "segmentation_training_samples")
    for col in (
        "id",
        "training_dataset_id",
        "crop_start_sec",
        "crop_end_sec",
        "events_json",
        "source",
        "created_at",
        "updated_at",
    ):
        assert sample_cols[col]["notnull"] is True, f"samples.{col} must be NOT NULL"
    for col in (
        "audio_file_id",
        "hydrophone_id",
        "start_timestamp",
        "end_timestamp",
        "source_ref",
        "notes",
    ):
        assert sample_cols[col]["notnull"] is False, f"samples.{col} must be nullable"
    assert sample_cols["crop_start_sec"]["type"] == "FLOAT"
    assert sample_cols["crop_end_sec"]["type"] == "FLOAT"
    assert sample_cols["start_timestamp"]["type"] == "FLOAT"
    assert sample_cols["end_timestamp"]["type"] == "FLOAT"

    sample_indexes = _indexes(db_path, "segmentation_training_samples")
    assert "ix_segmentation_training_samples_training_dataset_id" in sample_indexes
    assert "ix_segmentation_training_samples_dataset_source_ref" in sample_indexes

    job_cols = _columns(db_path, "segmentation_training_jobs")
    for col in (
        "id",
        "status",
        "training_dataset_id",
        "config_json",
        "created_at",
        "updated_at",
    ):
        assert job_cols[col]["notnull"] is True, f"jobs.{col} must be NOT NULL"
    for col in (
        "segmentation_model_id",
        "result_summary",
        "error_message",
        "started_at",
        "completed_at",
    ):
        assert job_cols[col]["notnull"] is False, f"jobs.{col} must be nullable"

    job_indexes = _indexes(db_path, "segmentation_training_jobs")
    assert "ix_segmentation_training_jobs_training_dataset_id" in job_indexes

    conn = sqlite3.connect(db_path)
    try:
        version = conn.execute("SELECT version_num FROM alembic_version").fetchone()
    finally:
        conn.close()
    assert version == ("044",)


def test_migration_044_downgrade_drops_tables(tmp_path: Path) -> None:
    """Downgrade 044 → 043 removes all three tables."""
    db_path = tmp_path / "test_044_downgrade.db"
    _pre_044_schema(db_path)

    config = _alembic_config(db_path)
    command.upgrade(config, "044")
    command.downgrade(config, "043")

    for table in (
        "segmentation_training_datasets",
        "segmentation_training_samples",
        "segmentation_training_jobs",
    ):
        assert not _table_exists(db_path, table), (
            f"{table} still exists after downgrade"
        )


def test_migration_044_roundtrip_is_idempotent(tmp_path: Path) -> None:
    """upgrade → downgrade → upgrade produces the same post-044 schema."""
    db_path = tmp_path / "test_044_roundtrip.db"
    _pre_044_schema(db_path)

    config = _alembic_config(db_path)
    command.upgrade(config, "044")
    first_sample_cols = _columns(db_path, "segmentation_training_samples")
    first_job_cols = _columns(db_path, "segmentation_training_jobs")
    first_sample_idx = _indexes(db_path, "segmentation_training_samples")
    first_job_idx = _indexes(db_path, "segmentation_training_jobs")

    command.downgrade(config, "043")
    command.upgrade(config, "044")

    assert _columns(db_path, "segmentation_training_samples") == first_sample_cols
    assert _columns(db_path, "segmentation_training_jobs") == first_job_cols
    assert _indexes(db_path, "segmentation_training_samples") == first_sample_idx
    assert _indexes(db_path, "segmentation_training_jobs") == first_job_idx
