"""Tests for migration 051 (perch_v2 ModelConfig seed)."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

from alembic import command
from alembic.config import Config

from humpback.database import Base, create_engine
from humpback.models import *  # noqa: F401,F403


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
    cfg = Config(str(repo_root / "alembic.ini"))
    cfg.set_main_option("script_location", str(repo_root / "alembic"))
    cfg.set_main_option("sqlalchemy.url", _db_url(db_path))
    return cfg


def _stamp_pre_051(db_path: Path) -> None:
    asyncio.run(_create_db(db_path))
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("DELETE FROM model_configs WHERE name = 'perch_v2'")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS alembic_version "
            "(version_num VARCHAR(32) NOT NULL)"
        )
        conn.execute("DELETE FROM alembic_version")
        conn.execute("INSERT INTO alembic_version (version_num) VALUES ('050')")
        conn.commit()
    finally:
        conn.close()


def _perch_rows(db_path: Path) -> list[tuple]:
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(
            "SELECT name, display_name, path, vector_dim, model_type, input_format "
            "FROM model_configs WHERE name = 'perch_v2'"
        ).fetchall()
    finally:
        conn.close()


def test_upgrade_inserts_perch_v2(tmp_path):
    db_path = tmp_path / "t.db"
    _stamp_pre_051(db_path)

    cfg = _alembic_config(db_path)
    command.upgrade(cfg, "051")

    rows = _perch_rows(db_path)
    assert len(rows) == 1
    name, display, path, vdim, mtype, input_fmt = rows[0]
    assert name == "perch_v2"
    assert display == "Perch v2 (TFLite)"
    assert path == "models/perch_v2.tflite"
    assert vdim == 1536
    assert mtype == "tflite"
    assert input_fmt == "waveform"


def test_upgrade_is_idempotent(tmp_path):
    db_path = tmp_path / "t.db"
    _stamp_pre_051(db_path)

    cfg = _alembic_config(db_path)
    command.upgrade(cfg, "051")
    # Rerunning upgrade should be a no-op (simulate via direct call).

    # Re-invoking command.upgrade is a no-op once at head.
    # Re-run via a fresh stamp + upgrade cycle to assert idempotency.
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("DELETE FROM alembic_version")
        conn.execute("INSERT INTO alembic_version (version_num) VALUES ('050')")
        conn.commit()
    finally:
        conn.close()

    command.upgrade(cfg, "051")
    rows = _perch_rows(db_path)
    assert len(rows) == 1


def test_downgrade_removes_row(tmp_path):
    db_path = tmp_path / "t.db"
    _stamp_pre_051(db_path)
    cfg = _alembic_config(db_path)

    command.upgrade(cfg, "051")
    assert len(_perch_rows(db_path)) == 1

    command.downgrade(cfg, "050")
    assert len(_perch_rows(db_path)) == 0
