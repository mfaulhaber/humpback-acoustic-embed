"""Tests for migration 077 piano_roll_notes_jobs."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from alembic import command
from alembic.config import Config


def _alembic_config(db_path: Path) -> Config:
    repo_root = Path(__file__).resolve().parents[2]
    cfg = Config(str(repo_root / "alembic.ini"))
    cfg.set_main_option("script_location", str(repo_root / "alembic"))
    cfg.set_main_option("sqlalchemy.url", f"sqlite+aiosqlite:///{db_path}")
    return cfg


def _columns(db_path: Path, table: str) -> dict[str, dict]:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    finally:
        conn.close()
    return {
        row[1]: {"type": row[2], "notnull": bool(row[3]), "dflt_value": row[4]}
        for row in rows
    }


def _indexes(db_path: Path, table: str) -> set[str]:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(f"PRAGMA index_list({table})").fetchall()
    finally:
        conn.close()
    return {str(row[1]) for row in rows}


def _create_pre_077_schema(db_path: Path) -> None:
    """Minimum schema needed to satisfy the FK from piano_roll_notes_jobs."""
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE event_encoder_jobs (
                id VARCHAR NOT NULL PRIMARY KEY,
                status VARCHAR NOT NULL,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL
            );
            CREATE TABLE alembic_version (
                version_num VARCHAR(32) NOT NULL
            );
            INSERT INTO alembic_version (version_num) VALUES ('076');
            """
        )
        conn.commit()
    finally:
        conn.close()


def test_upgrade_creates_piano_roll_notes_jobs_table(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    _create_pre_077_schema(db_path)

    command.upgrade(_alembic_config(db_path), "077")

    cols = _columns(db_path, "piano_roll_notes_jobs")
    assert cols["id"]["notnull"] is True
    assert cols["event_encoder_job_id"]["notnull"] is True
    assert cols["extractor_version"]["notnull"] is True
    assert cols["extractor_version"]["dflt_value"] == "'v1'"
    assert cols["status"]["notnull"] is True
    assert cols["status"]["dflt_value"] == "'queued'"
    assert cols["params_json"]["notnull"] is True
    assert cols["started_at"]["notnull"] is False
    assert cols["finished_at"]["notnull"] is False
    assert cols["error_message"]["notnull"] is False
    assert cols["notes_path"]["notnull"] is False
    assert cols["n_events"]["notnull"] is False
    assert cols["n_notes"]["notnull"] is False
    assert cols["compute_seconds"]["notnull"] is False

    indexes = _indexes(db_path, "piano_roll_notes_jobs")
    assert "ix_piano_roll_notes_jobs_event_encoder_job_id" in indexes
    assert "ix_piano_roll_notes_jobs_status" in indexes

    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "INSERT INTO event_encoder_jobs "
            "(id, status, created_at, updated_at) "
            "VALUES ('eej-1', 'complete', '2026-05-20', '2026-05-20')"
        )
        conn.execute(
            "INSERT INTO piano_roll_notes_jobs "
            "(id, event_encoder_job_id, extractor_version, status, "
            " params_json, created_at, updated_at) "
            "VALUES ('prn-1', 'eej-1', 'v1', 'queued', '{}', "
            " '2026-05-20', '2026-05-20')"
        )
        try:
            conn.execute(
                "INSERT INTO piano_roll_notes_jobs "
                "(id, event_encoder_job_id, extractor_version, status, "
                " params_json, created_at, updated_at) "
                "VALUES ('prn-2', 'eej-1', 'v1', 'queued', '{}', "
                " '2026-05-20', '2026-05-20')"
            )
        except sqlite3.IntegrityError:
            duplicate_rejected = True
        else:
            duplicate_rejected = False
    finally:
        conn.close()
    assert duplicate_rejected is True


def test_downgrade_removes_piano_roll_notes_jobs_table(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    _create_pre_077_schema(db_path)
    cfg = _alembic_config(db_path)
    command.upgrade(cfg, "077")

    command.downgrade(cfg, "076")

    conn = sqlite3.connect(db_path)
    try:
        exists = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='piano_roll_notes_jobs'"
        ).fetchone()
    finally:
        conn.close()
    assert exists is None
