"""Tests for migration 043 (Pass 1 source columns).

Migration 043 replaces the Phase 0 ``audio_source_id`` placeholder on
``call_parsing_runs`` and ``region_detection_jobs`` with the four real
source columns (``audio_file_id``, ``hydrophone_id``, ``start_timestamp``,
``end_timestamp``).

The tests follow the pattern established by
``test_migration_042_call_parsing.py``: build a DB pinned at revision
042, run Alembic upgrade/downgrade against it, and inspect the raw
sqlite schema via ``PRAGMA table_info``.
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
    """Return a ``{column_name: {type, notnull}}`` map for ``table``."""
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


def _pre_043_schema(db_path: Path) -> None:
    """Build an ORM DB and force it back to revision 042.

    The current ORM reflects the post-043 shape, so after ``create_all``
    we rebuild the two affected tables with the Phase 0 ``audio_source_id``
    placeholder column (and nothing else from 043) to simulate a DB
    paused at revision 042.
    """
    asyncio.run(_create_db(db_path))
    conn = sqlite3.connect(db_path)
    try:
        # Rebuild call_parsing_runs with the Phase 0 column shape (adds
        # audio_source_id, drops the four Task 1 columns) while preserving
        # any already-existing rows. We don't have any rows in these
        # fixture DBs, but the SELECT-copy dance is still the correct
        # shape for a migration pre-state.
        conn.execute("DROP TABLE IF EXISTS call_parsing_runs")
        conn.execute(
            """
            CREATE TABLE call_parsing_runs (
                id TEXT PRIMARY KEY,
                audio_source_id TEXT NOT NULL,
                status TEXT NOT NULL,
                config_snapshot TEXT,
                region_detection_job_id TEXT,
                event_segmentation_job_id TEXT,
                event_classification_job_id TEXT,
                error_message TEXT,
                completed_at TIMESTAMP,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX ix_call_parsing_runs_audio_source_id "
            "ON call_parsing_runs (audio_source_id)"
        )

        conn.execute("DROP TABLE IF EXISTS region_detection_jobs")
        conn.execute(
            """
            CREATE TABLE region_detection_jobs (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                parent_run_id TEXT,
                audio_source_id TEXT NOT NULL,
                model_config_id TEXT,
                classifier_model_id TEXT,
                config_json TEXT,
                trace_row_count INTEGER,
                region_count INTEGER,
                error_message TEXT,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                created_at TIMESTAMP NOT NULL,
                updated_at TIMESTAMP NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX ix_region_detection_jobs_parent_run_id "
            "ON region_detection_jobs (parent_run_id)"
        )

        conn.execute(
            "CREATE TABLE IF NOT EXISTS alembic_version "
            "(version_num VARCHAR(32) NOT NULL)"
        )
        conn.execute("DELETE FROM alembic_version")
        conn.execute("INSERT INTO alembic_version (version_num) VALUES ('042')")
        conn.commit()
    finally:
        conn.close()


def test_migration_043_upgrade_swaps_source_columns(tmp_path: Path) -> None:
    """Upgrade 042 → 043: ``audio_source_id`` gone, four new columns present."""
    db_path = tmp_path / "test_043_upgrade.db"
    _pre_043_schema(db_path)

    config = _alembic_config(db_path)
    command.upgrade(config, "043")

    for table in ("call_parsing_runs", "region_detection_jobs"):
        cols = _columns(db_path, table)
        assert "audio_source_id" not in cols, (
            f"{table} still has audio_source_id after upgrade"
        )
        for new_col in (
            "audio_file_id",
            "hydrophone_id",
            "start_timestamp",
            "end_timestamp",
        ):
            assert new_col in cols, f"{table} missing {new_col} after upgrade"
            assert cols[new_col]["notnull"] is False, (
                f"{table}.{new_col} should be nullable"
            )
        audio_type = str(cols["audio_file_id"]["type"])
        hydro_type = str(cols["hydrophone_id"]["type"])
        assert audio_type.startswith("VARCHAR") or audio_type == "TEXT", audio_type
        assert hydro_type.startswith("VARCHAR") or hydro_type == "TEXT", hydro_type
        assert cols["start_timestamp"]["type"] == "FLOAT"
        assert cols["end_timestamp"]["type"] == "FLOAT"

    # The Phase 0 index on call_parsing_runs.audio_source_id is gone.
    assert "ix_call_parsing_runs_audio_source_id" not in _indexes(
        db_path, "call_parsing_runs"
    )

    conn = sqlite3.connect(db_path)
    try:
        version = conn.execute("SELECT version_num FROM alembic_version").fetchone()
    finally:
        conn.close()
    assert version == ("043",)


def test_migration_043_downgrade_restores_placeholder(tmp_path: Path) -> None:
    """Downgrade 043 → 042 restores ``audio_source_id`` as nullable."""
    db_path = tmp_path / "test_043_downgrade.db"
    _pre_043_schema(db_path)

    config = _alembic_config(db_path)
    command.upgrade(config, "043")
    command.downgrade(config, "042")

    for table in ("call_parsing_runs", "region_detection_jobs"):
        cols = _columns(db_path, table)
        assert "audio_source_id" in cols, (
            f"{table} missing restored audio_source_id after downgrade"
        )
        # Downgrade restores it as nullable so no backfill is required.
        assert cols["audio_source_id"]["notnull"] is False
        for removed in (
            "audio_file_id",
            "hydrophone_id",
            "start_timestamp",
            "end_timestamp",
        ):
            assert removed not in cols, f"{table} still has {removed} after downgrade"

    # The Phase 0 index is restored on downgrade.
    assert "ix_call_parsing_runs_audio_source_id" in _indexes(
        db_path, "call_parsing_runs"
    )


def test_migration_043_roundtrip_is_idempotent(tmp_path: Path) -> None:
    """upgrade → downgrade → upgrade produces the same post-043 schema."""
    db_path = tmp_path / "test_043_roundtrip.db"
    _pre_043_schema(db_path)

    config = _alembic_config(db_path)
    command.upgrade(config, "043")
    after_first = {
        t: _columns(db_path, t) for t in ("call_parsing_runs", "region_detection_jobs")
    }

    command.downgrade(config, "042")
    command.upgrade(config, "043")
    after_second = {
        t: _columns(db_path, t) for t in ("call_parsing_runs", "region_detection_jobs")
    }

    assert after_first == after_second
