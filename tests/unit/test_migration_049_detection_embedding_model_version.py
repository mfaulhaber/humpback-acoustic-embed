"""Tests for migration 049 (detection_embedding_jobs model_version + progress)."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

from alembic import command
from alembic.config import Config

from humpback.database import Base, create_engine
from humpback.models import *  # noqa: F401,F403
from humpback.models.classifier import (  # noqa: F401
    ClassifierModel,
    DetectionJob,
)
from humpback.models.detection_embedding_job import DetectionEmbeddingJob  # noqa: F401


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


def _columns(db_path: Path, table: str) -> set[str]:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    finally:
        conn.close()
    return {r[1] for r in rows}


def _stamp_pre_049(db_path: Path) -> None:
    asyncio.run(_create_db(db_path))
    conn = sqlite3.connect(db_path)
    try:
        for col in ("model_version", "rows_processed", "rows_total"):
            try:
                conn.execute(f"ALTER TABLE detection_embedding_jobs DROP COLUMN {col}")
            except sqlite3.OperationalError:
                pass
        conn.execute(
            "CREATE TABLE IF NOT EXISTS alembic_version "
            "(version_num VARCHAR(32) NOT NULL)"
        )
        conn.execute("DELETE FROM alembic_version")
        conn.execute("INSERT INTO alembic_version (version_num) VALUES ('048')")
        conn.commit()
    finally:
        conn.close()


def _seed_for_backfill(db_path: Path, storage_root: Path) -> None:
    """Seed one complete embedding job plus its source classifier/detection job.

    Also drops a legacy parquet at the pre-049 path so the migration can relocate it.
    """
    conn = sqlite3.connect(db_path)
    try:
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT INTO classifier_models "
            "(id, name, model_path, model_version, vector_dim, window_size_seconds, "
            " target_sample_rate, classifier_purpose, training_source_mode, "
            " created_at, updated_at) "
            "VALUES ('cm1', 'n', '/tmp/n', 'perch_v2', 1536, 5.0, 32000, "
            " 'detection', 'embedding_sets', ?, ?)",
            (now, now),
        )
        conn.execute(
            "INSERT INTO detection_jobs "
            "(id, status, classifier_model_id, confidence_threshold, hop_seconds, "
            " high_threshold, low_threshold, timeline_tiles_ready, "
            " created_at, updated_at) "
            "VALUES ('dj1', 'complete', 'cm1', 0.5, 1.0, 0.7, 0.45, 0, ?, ?)",
            (now, now),
        )
        conn.execute(
            "INSERT INTO detection_embedding_jobs "
            "(id, status, detection_job_id, created_at, updated_at) "
            "VALUES ('dej1', 'complete', 'dj1', ?, ?)",
            (now, now),
        )
        conn.commit()
    finally:
        conn.close()

    legacy = storage_root / "detections" / "dj1" / "detection_embeddings.parquet"
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_bytes(b"fake-parquet")


def test_upgrade_adds_columns_and_backfills(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    storage_root = tmp_path / "storage"
    storage_root.mkdir()
    _stamp_pre_049(db_path)
    _seed_for_backfill(db_path, storage_root)

    monkeypatch.setenv("HUMPBACK_STORAGE_ROOT", str(storage_root))
    cfg = _alembic_config(db_path)
    command.upgrade(cfg, "049")

    cols = _columns(db_path, "detection_embedding_jobs")
    assert {"model_version", "rows_processed", "rows_total"}.issubset(cols)

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT model_version, rows_processed FROM detection_embedding_jobs "
            "WHERE id = 'dej1'"
        ).fetchone()
        assert row[0] == "perch_v2"
        assert row[1] == 0
    finally:
        conn.close()

    # Parquet physically relocated
    legacy = storage_root / "detections" / "dj1" / "detection_embeddings.parquet"
    new = (
        storage_root
        / "detections"
        / "dj1"
        / "embeddings"
        / "perch_v2"
        / "detection_embeddings.parquet"
    )
    assert not legacy.exists()
    assert new.exists()


def test_upgrade_enforces_composite_uniqueness(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    storage_root = tmp_path / "storage"
    storage_root.mkdir()
    _stamp_pre_049(db_path)
    _seed_for_backfill(db_path, storage_root)

    monkeypatch.setenv("HUMPBACK_STORAGE_ROOT", str(storage_root))
    cfg = _alembic_config(db_path)
    command.upgrade(cfg, "049")

    conn = sqlite3.connect(db_path)
    try:
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        # Same detection_job_id + model_version should raise
        import pytest

        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO detection_embedding_jobs "
                "(id, status, detection_job_id, model_version, rows_processed, "
                " created_at, updated_at) "
                "VALUES ('dej2', 'complete', 'dj1', 'perch_v2', 0, ?, ?)",
                (now, now),
            )
            conn.commit()
        conn.rollback()

        # Different model_version: should succeed
        conn.execute(
            "INSERT INTO detection_embedding_jobs "
            "(id, status, detection_job_id, model_version, rows_processed, "
            " created_at, updated_at) "
            "VALUES ('dej3', 'complete', 'dj1', 'perch_v1', 0, ?, ?)",
            (now, now),
        )
        conn.commit()
    finally:
        conn.close()


def test_downgrade_removes_columns(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    storage_root = tmp_path / "storage"
    storage_root.mkdir()
    _stamp_pre_049(db_path)
    _seed_for_backfill(db_path, storage_root)

    monkeypatch.setenv("HUMPBACK_STORAGE_ROOT", str(storage_root))
    cfg = _alembic_config(db_path)
    command.upgrade(cfg, "049")
    command.downgrade(cfg, "048")

    cols = _columns(db_path, "detection_embedding_jobs")
    assert "model_version" not in cols
    assert "rows_processed" not in cols
    assert "rows_total" not in cols
