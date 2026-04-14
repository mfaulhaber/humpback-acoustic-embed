"""Tests for migration 047 (drop event_segmentation_training_jobs).

Verifies upgrade drops the table and downgrade recreates it with the
original schema.
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


def _columns(db_path: Path, table: str) -> dict[str, dict[str, object]]:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    finally:
        conn.close()
    return {r[1]: {"type": r[2], "notnull": bool(r[3]), "pk": bool(r[5])} for r in rows}


TABLE = "event_segmentation_training_jobs"


_CREATE_TABLE_SQL = """\
CREATE TABLE IF NOT EXISTS event_segmentation_training_jobs (
    id VARCHAR NOT NULL,
    status VARCHAR NOT NULL,
    source_job_ids TEXT NOT NULL,
    config_json TEXT,
    segmentation_model_id VARCHAR,
    result_summary TEXT,
    error_message TEXT,
    started_at DATETIME,
    completed_at DATETIME,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,
    PRIMARY KEY (id)
)
"""


def _pre_047_schema(db_path: Path) -> None:
    """Build a DB at revision 046 with the target table present."""
    asyncio.run(_create_db(db_path))
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(_CREATE_TABLE_SQL)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS alembic_version "
            "(version_num VARCHAR(32) NOT NULL)"
        )
        conn.execute("DELETE FROM alembic_version")
        conn.execute("INSERT INTO alembic_version (version_num) VALUES ('046')")
        conn.commit()
    finally:
        conn.close()


def test_upgrade_drops_table(tmp_path):
    db_path = tmp_path / "test.db"
    _pre_047_schema(db_path)
    assert _table_exists(db_path, TABLE)

    cfg = _alembic_config(db_path)
    command.upgrade(cfg, "047")

    assert not _table_exists(db_path, TABLE)


def test_downgrade_recreates_table(tmp_path):
    db_path = tmp_path / "test.db"
    _pre_047_schema(db_path)
    cfg = _alembic_config(db_path)
    command.upgrade(cfg, "047")
    assert not _table_exists(db_path, TABLE)

    command.downgrade(cfg, "046")
    assert _table_exists(db_path, TABLE)

    cols = _columns(db_path, TABLE)
    assert "id" in cols
    assert "status" in cols
    assert "source_job_ids" in cols
    assert "config_json" in cols
    assert "segmentation_model_id" in cols
    assert "result_summary" in cols
    assert "error_message" in cols
    assert "started_at" in cols
    assert "completed_at" in cols
    assert "created_at" in cols
    assert "updated_at" in cols
    assert cols["source_job_ids"]["notnull"] is True
    assert cols["segmentation_model_id"]["notnull"] is False


def test_upgrade_downgrade_roundtrip(tmp_path):
    db_path = tmp_path / "test.db"
    _pre_047_schema(db_path)
    cfg = _alembic_config(db_path)

    command.upgrade(cfg, "047")
    assert not _table_exists(db_path, TABLE)

    command.downgrade(cfg, "046")
    assert _table_exists(db_path, TABLE)
    cols_first = _columns(db_path, TABLE)

    command.upgrade(cfg, "047")
    assert not _table_exists(db_path, TABLE)

    command.downgrade(cfg, "046")
    assert _columns(db_path, TABLE) == cols_first
