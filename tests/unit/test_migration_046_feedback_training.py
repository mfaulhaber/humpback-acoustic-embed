"""Tests for migration 046 (feedback training tables).

Verifies upgrade creates all four tables with correct columns, indexes,
and unique constraints, and that downgrade drops them cleanly.
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
    return {r[1]: {"type": r[2], "notnull": bool(r[3]), "pk": bool(r[5])} for r in rows}


def _indexes(db_path: Path, table: str) -> dict[str, bool]:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(f"PRAGMA index_list({table})").fetchall()
    finally:
        conn.close()
    return {r[1]: bool(r[2]) for r in rows}


def _table_exists(db_path: Path, table: str) -> bool:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        ).fetchall()
    finally:
        conn.close()
    return len(rows) > 0


_NEW_TABLES = [
    "event_boundary_corrections",
    "event_type_corrections",
    "event_segmentation_training_jobs",
    "event_classifier_training_jobs",
]


def _pre_046_schema(db_path: Path) -> None:
    """Build a DB at revision 045 by creating all tables then dropping the 046 ones."""
    asyncio.run(_create_db(db_path))
    conn = sqlite3.connect(db_path)
    try:
        for table in _NEW_TABLES:
            conn.execute(f"DROP TABLE IF EXISTS {table}")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS alembic_version "
            "(version_num VARCHAR(32) NOT NULL)"
        )
        conn.execute("DELETE FROM alembic_version")
        conn.execute("INSERT INTO alembic_version (version_num) VALUES ('045')")
        conn.commit()
    finally:
        conn.close()


def test_upgrade_creates_all_four_tables(tmp_path):
    db_path = tmp_path / "test.db"
    _pre_046_schema(db_path)
    cfg = _alembic_config(db_path)
    command.upgrade(cfg, "046")

    for table in _NEW_TABLES:
        assert _table_exists(db_path, table), f"{table} not created"


def test_boundary_corrections_columns(tmp_path):
    db_path = tmp_path / "test.db"
    _pre_046_schema(db_path)
    cfg = _alembic_config(db_path)
    command.upgrade(cfg, "046")

    cols = _columns(db_path, "event_boundary_corrections")
    assert "id" in cols
    assert "event_segmentation_job_id" in cols
    assert "event_id" in cols
    assert "region_id" in cols
    assert "correction_type" in cols
    assert "start_sec" in cols
    assert "end_sec" in cols
    assert cols["start_sec"]["notnull"] is False
    assert cols["end_sec"]["notnull"] is False
    assert cols["event_segmentation_job_id"]["notnull"] is True


def test_type_corrections_unique_constraint(tmp_path):
    db_path = tmp_path / "test.db"
    _pre_046_schema(db_path)
    cfg = _alembic_config(db_path)
    command.upgrade(cfg, "046")

    conn = sqlite3.connect(db_path)
    try:
        sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' "
            "AND name='event_type_corrections'"
        ).fetchone()
    finally:
        conn.close()
    assert sql is not None
    assert "uq_event_type_corrections_job_event" in sql[0]


def test_boundary_corrections_index(tmp_path):
    db_path = tmp_path / "test.db"
    _pre_046_schema(db_path)
    cfg = _alembic_config(db_path)
    command.upgrade(cfg, "046")

    indexes = _indexes(db_path, "event_boundary_corrections")
    assert "ix_event_boundary_corrections_job_id" in indexes


def test_training_job_tables_columns(tmp_path):
    db_path = tmp_path / "test.db"
    _pre_046_schema(db_path)
    cfg = _alembic_config(db_path)
    command.upgrade(cfg, "046")

    seg_cols = _columns(db_path, "event_segmentation_training_jobs")
    assert "source_job_ids" in seg_cols
    assert "segmentation_model_id" in seg_cols
    assert seg_cols["source_job_ids"]["notnull"] is True
    assert seg_cols["segmentation_model_id"]["notnull"] is False

    cls_cols = _columns(db_path, "event_classifier_training_jobs")
    assert "source_job_ids" in cls_cols
    assert "vocalization_model_id" in cls_cols
    assert cls_cols["source_job_ids"]["notnull"] is True
    assert cls_cols["vocalization_model_id"]["notnull"] is False


def test_downgrade_drops_all_tables(tmp_path):
    db_path = tmp_path / "test.db"
    _pre_046_schema(db_path)
    cfg = _alembic_config(db_path)
    command.upgrade(cfg, "046")
    command.downgrade(cfg, "045")

    for table in _NEW_TABLES:
        assert not _table_exists(db_path, table), f"{table} not dropped"


def test_upgrade_downgrade_roundtrip(tmp_path):
    db_path = tmp_path / "test.db"
    _pre_046_schema(db_path)
    cfg = _alembic_config(db_path)

    command.upgrade(cfg, "046")
    first_cols = _columns(db_path, "event_boundary_corrections")
    first_idx = _indexes(db_path, "event_boundary_corrections")

    command.downgrade(cfg, "045")
    command.upgrade(cfg, "046")

    assert _columns(db_path, "event_boundary_corrections") == first_cols
    assert _indexes(db_path, "event_boundary_corrections") == first_idx
