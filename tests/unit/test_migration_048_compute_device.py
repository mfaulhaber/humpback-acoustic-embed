"""Tests for migration 048 (compute_device + gpu_fallback_reason).

Verifies that the upgrade adds the two columns to both
``event_segmentation_jobs`` and ``event_classification_jobs``, and that
the downgrade drops them cleanly. Roundtrip ensures column metadata is
stable across multiple upgrade/downgrade cycles.
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

from alembic import command
from alembic.config import Config

from humpback.database import Base, create_engine

# Importing the call_parsing models registers their tables with
# ``Base.metadata`` so ``create_all`` builds them. Without this the
# event-segmentation/classification tables are missing from the test DB.
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


def _columns(db_path: Path, table: str) -> set[str]:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    finally:
        conn.close()
    return {r[1] for r in rows}


def _stamp_pre_048(db_path: Path) -> None:
    """Build a DB at revision 047 (no compute_device columns yet).

    ``Base.metadata.create_all`` builds the *current* schema, which
    already includes the new columns. Drop them so the smoke test
    starts from a true pre-048 state.
    """
    asyncio.run(_create_db(db_path))
    conn = sqlite3.connect(db_path)
    try:
        for table in ("event_segmentation_jobs", "event_classification_jobs"):
            for col in ("compute_device", "gpu_fallback_reason"):
                conn.execute(f"ALTER TABLE {table} DROP COLUMN {col}")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS alembic_version "
            "(version_num VARCHAR(32) NOT NULL)"
        )
        conn.execute("DELETE FROM alembic_version")
        conn.execute("INSERT INTO alembic_version (version_num) VALUES ('047')")
        conn.commit()
    finally:
        conn.close()


def test_upgrade_adds_columns_to_both_tables(tmp_path):
    db_path = tmp_path / "test.db"
    _stamp_pre_048(db_path)

    seg_pre = _columns(db_path, "event_segmentation_jobs")
    cls_pre = _columns(db_path, "event_classification_jobs")
    assert "compute_device" not in seg_pre
    assert "gpu_fallback_reason" not in seg_pre
    assert "compute_device" not in cls_pre
    assert "gpu_fallback_reason" not in cls_pre

    cfg = _alembic_config(db_path)
    command.upgrade(cfg, "048")

    seg_post = _columns(db_path, "event_segmentation_jobs")
    cls_post = _columns(db_path, "event_classification_jobs")
    assert "compute_device" in seg_post
    assert "gpu_fallback_reason" in seg_post
    assert "compute_device" in cls_post
    assert "gpu_fallback_reason" in cls_post


def test_downgrade_removes_columns_from_both_tables(tmp_path):
    db_path = tmp_path / "test.db"
    _stamp_pre_048(db_path)
    cfg = _alembic_config(db_path)

    command.upgrade(cfg, "048")
    command.downgrade(cfg, "047")

    seg_cols = _columns(db_path, "event_segmentation_jobs")
    cls_cols = _columns(db_path, "event_classification_jobs")
    assert "compute_device" not in seg_cols
    assert "gpu_fallback_reason" not in seg_cols
    assert "compute_device" not in cls_cols
    assert "gpu_fallback_reason" not in cls_cols


def test_upgrade_downgrade_roundtrip(tmp_path):
    db_path = tmp_path / "test.db"
    _stamp_pre_048(db_path)
    cfg = _alembic_config(db_path)

    command.upgrade(cfg, "048")
    seg_after_first = _columns(db_path, "event_segmentation_jobs")
    cls_after_first = _columns(db_path, "event_classification_jobs")

    command.downgrade(cfg, "047")
    command.upgrade(cfg, "048")

    assert _columns(db_path, "event_segmentation_jobs") == seg_after_first
    assert _columns(db_path, "event_classification_jobs") == cls_after_first
