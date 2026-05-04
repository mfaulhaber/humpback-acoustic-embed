"""Tests for migration 065 (effective event identity)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from alembic import command
from alembic.config import Config


def _db_url(db_path: Path) -> str:
    return f"sqlite+aiosqlite:///{db_path}"


def _alembic_config(db_path: Path) -> Config:
    repo_root = Path(__file__).resolve().parents[2]
    cfg = Config(str(repo_root / "alembic.ini"))
    cfg.set_main_option("script_location", str(repo_root / "alembic"))
    cfg.set_main_option("sqlalchemy.url", _db_url(db_path))
    return cfg


def _columns(db_path: Path, table: str) -> dict[str, dict]:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    finally:
        conn.close()
    return {
        r[1]: {"type": r[2], "notnull": bool(r[3]), "dflt_value": r[4]} for r in rows
    }


def _indexes(db_path: Path, table: str) -> set[str]:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(f"PRAGMA index_list({table})").fetchall()
    finally:
        conn.close()
    return {str(r[1]) for r in rows}


def _create_pre_065_schema(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE event_boundary_corrections (
                id VARCHAR NOT NULL PRIMARY KEY,
                region_detection_job_id VARCHAR NOT NULL,
                region_id VARCHAR NOT NULL,
                correction_type VARCHAR NOT NULL,
                original_start_sec FLOAT,
                original_end_sec FLOAT,
                corrected_start_sec FLOAT,
                corrected_end_sec FLOAT,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL
            );
            CREATE INDEX ix_event_boundary_corrections_detection_job
                ON event_boundary_corrections (region_detection_job_id);

            CREATE TABLE continuous_embedding_jobs (
                id VARCHAR NOT NULL PRIMARY KEY,
                status VARCHAR NOT NULL,
                event_segmentation_job_id VARCHAR,
                model_version VARCHAR NOT NULL,
                target_sample_rate INTEGER NOT NULL,
                encoding_signature VARCHAR NOT NULL,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL
            );

            CREATE TABLE alembic_version (
                version_num VARCHAR(32) NOT NULL
            );
            INSERT INTO alembic_version (version_num) VALUES ('064');
            """
        )
        conn.commit()
    finally:
        conn.close()


def test_upgrade_adds_nullable_correction_identity_without_backfill(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "test.db"
    _create_pre_065_schema(db_path)
    cfg = _alembic_config(db_path)

    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "INSERT INTO event_boundary_corrections "
            "(id, region_detection_job_id, region_id, correction_type, "
            " original_start_sec, original_end_sec, corrected_start_sec, "
            " corrected_end_sec, created_at, updated_at) "
            "VALUES "
            "('c1', 'rd-1', 'region-1', 'adjust', "
            " 10.0, 11.0, 10.1, 11.1, '2026-05-03', '2026-05-03')"
        )
        conn.commit()
    finally:
        conn.close()

    command.upgrade(cfg, "065")

    cols = _columns(db_path, "event_boundary_corrections")
    assert "event_segmentation_job_id" in cols
    assert cols["event_segmentation_job_id"]["notnull"] is False
    assert "source_event_id" in cols
    assert cols["source_event_id"]["notnull"] is False

    indexes = _indexes(db_path, "event_boundary_corrections")
    assert "ix_event_boundary_corrections_region_detection_job" in indexes
    assert "ix_event_boundary_corrections_segmentation_job" in indexes
    assert "ix_event_boundary_corrections_source_event" in indexes

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT id, event_segmentation_job_id, source_event_id "
            "FROM event_boundary_corrections"
        ).fetchall()
    finally:
        conn.close()
    assert rows == [("c1", None, None)]


def test_upgrade_adds_raw_event_source_mode_default(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    _create_pre_065_schema(db_path)
    cfg = _alembic_config(db_path)
    command.upgrade(cfg, "065")

    cols = _columns(db_path, "continuous_embedding_jobs")
    assert "event_source_mode" in cols
    assert cols["event_source_mode"]["notnull"] is True
    assert cols["event_source_mode"]["dflt_value"] == "'raw'"
