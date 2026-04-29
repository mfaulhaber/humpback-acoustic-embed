"""Tests for migration 050 (hyperparameter_manifests.embedding_model_version)."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from alembic import command
from alembic.config import Config

from humpback.database import Base, create_engine
from humpback.models import *  # noqa: F401,F403
from humpback.models.classifier import (  # noqa: F401
    ClassifierModel,
    ClassifierTrainingJob,
    DetectionJob,
)


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


def _stamp_pre_050(db_path: Path) -> None:
    asyncio.run(_create_db(db_path))
    conn = sqlite3.connect(db_path)
    try:
        try:
            conn.execute(
                "ALTER TABLE hyperparameter_manifests DROP COLUMN "
                "embedding_model_version"
            )
        except sqlite3.OperationalError:
            pass
        conn.execute(
            "CREATE TABLE IF NOT EXISTS alembic_version "
            "(version_num VARCHAR(32) NOT NULL)"
        )
        conn.execute("DELETE FROM alembic_version")
        conn.execute("INSERT INTO alembic_version (version_num) VALUES ('049')")
        conn.commit()
    finally:
        conn.close()


def _seed(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    try:
        now = datetime.now(timezone.utc).isoformat()
        # Training job source
        conn.execute(
            "INSERT INTO classifier_training_jobs "
            "(id, status, name, "
            " model_version, window_size_seconds, target_sample_rate, job_purpose, "
            " source_mode, created_at, updated_at) "
            "VALUES ('tj1', 'complete', 'n', 'tf2_v1', 5.0, 32000, "
            " 'detection', 'embedding_sets', ?, ?)",
            (now, now),
        )
        # Detection job source
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
        # Manifest with training-job source
        conn.execute(
            "INSERT INTO hyperparameter_manifests "
            "(id, name, status, training_job_ids, detection_job_ids, "
            " split_ratio, seed, created_at, updated_at) "
            "VALUES ('m_tj', 'tj-only', 'queued', ?, '[]', '[70,15,15]', 42, ?, ?)",
            (json.dumps(["tj1"]), now, now),
        )
        # Manifest with detection-job source only
        conn.execute(
            "INSERT INTO hyperparameter_manifests "
            "(id, name, status, training_job_ids, detection_job_ids, "
            " split_ratio, seed, created_at, updated_at) "
            "VALUES ('m_dj', 'dj-only', 'queued', '[]', ?, '[70,15,15]', 42, ?, ?)",
            (json.dumps(["dj1"]), now, now),
        )
        # Manifest with unresolvable sources
        conn.execute(
            "INSERT INTO hyperparameter_manifests "
            "(id, name, status, training_job_ids, detection_job_ids, "
            " split_ratio, seed, created_at, updated_at) "
            "VALUES ('m_no', 'orphan', 'queued', '[]', '[]', '[70,15,15]', 42, ?, ?)",
            (now, now),
        )
        conn.commit()
    finally:
        conn.close()


def test_upgrade_backfills_from_training_and_detection_sources(tmp_path):
    db_path = tmp_path / "t.db"
    _stamp_pre_050(db_path)
    _seed(db_path)

    cfg = _alembic_config(db_path)
    command.upgrade(cfg, "050")

    conn = sqlite3.connect(db_path)
    try:
        rows = dict(
            conn.execute(
                "SELECT id, embedding_model_version FROM hyperparameter_manifests"
            ).fetchall()
        )
    finally:
        conn.close()

    assert rows["m_tj"] == "tf2_v1"
    assert rows["m_dj"] == "perch_v2"
    assert rows["m_no"] == "unknown"


def test_downgrade_removes_column(tmp_path):
    db_path = tmp_path / "t.db"
    _stamp_pre_050(db_path)
    _seed(db_path)

    cfg = _alembic_config(db_path)
    command.upgrade(cfg, "050")
    command.downgrade(cfg, "049")

    conn = sqlite3.connect(db_path)
    try:
        cols = {
            r[1]
            for r in conn.execute(
                "PRAGMA table_info(hyperparameter_manifests)"
            ).fetchall()
        }
    finally:
        conn.close()
    assert "embedding_model_version" not in cols
