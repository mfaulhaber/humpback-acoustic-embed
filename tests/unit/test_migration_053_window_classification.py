"""Tests for migration 053 (window_classification_jobs + window_score_corrections).

Verifies that the upgrade creates both tables with expected columns
and that the downgrade drops them cleanly.
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

from alembic import command
from alembic.config import Config

from humpback.database import Base, create_engine

import humpback.models.call_parsing  # noqa: F401


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


def _tables(db_path: Path) -> set[str]:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    finally:
        conn.close()
    return {r[0] for r in rows}


def _columns(db_path: Path, table: str) -> set[str]:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    finally:
        conn.close()
    return {r[1] for r in rows}


def _stamp_pre_053(db_path: Path) -> None:
    asyncio.run(_create_db(db_path))
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("DROP TABLE IF EXISTS window_classification_jobs")
        conn.execute("DROP TABLE IF EXISTS window_score_corrections")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS alembic_version "
            "(version_num VARCHAR(32) NOT NULL)"
        )
        conn.execute("DELETE FROM alembic_version")
        conn.execute("INSERT INTO alembic_version (version_num) VALUES ('052')")
        conn.commit()
    finally:
        conn.close()


def test_upgrade_creates_both_tables(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    _stamp_pre_053(db_path)

    tables_pre = _tables(db_path)
    assert "window_classification_jobs" not in tables_pre
    assert "window_score_corrections" not in tables_pre

    cfg = _alembic_config(db_path)
    command.upgrade(cfg, "053")

    tables_post = _tables(db_path)
    assert "window_classification_jobs" in tables_post
    assert "window_score_corrections" in tables_post

    job_cols = _columns(db_path, "window_classification_jobs")
    assert {
        "id",
        "status",
        "region_detection_job_id",
        "vocalization_model_id",
        "config_json",
        "window_count",
        "vocabulary_snapshot",
        "error_message",
        "started_at",
        "completed_at",
        "created_at",
        "updated_at",
    } <= job_cols

    corr_cols = _columns(db_path, "window_score_corrections")
    assert {
        "id",
        "window_classification_job_id",
        "time_sec",
        "region_id",
        "correction_type",
        "type_name",
        "created_at",
        "updated_at",
    } <= corr_cols


def test_downgrade_drops_both_tables(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    _stamp_pre_053(db_path)
    cfg = _alembic_config(db_path)

    command.upgrade(cfg, "053")
    assert "window_classification_jobs" in _tables(db_path)

    command.downgrade(cfg, "052")
    tables = _tables(db_path)
    assert "window_classification_jobs" not in tables
    assert "window_score_corrections" not in tables
