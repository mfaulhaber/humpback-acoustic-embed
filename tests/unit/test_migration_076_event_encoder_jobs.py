"""Tests for migration 076 event encoder jobs."""

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


def _create_pre_076_schema(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE region_detection_jobs (
                id VARCHAR NOT NULL PRIMARY KEY,
                status VARCHAR NOT NULL,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL
            );
            CREATE TABLE event_segmentation_jobs (
                id VARCHAR NOT NULL PRIMARY KEY,
                status VARCHAR NOT NULL,
                region_detection_job_id VARCHAR NOT NULL,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL
            );
            CREATE TABLE continuous_embedding_jobs (
                id VARCHAR NOT NULL PRIMARY KEY,
                status VARCHAR NOT NULL,
                event_segmentation_job_id VARCHAR,
                event_source_mode VARCHAR NOT NULL,
                model_version VARCHAR NOT NULL,
                target_sample_rate INTEGER NOT NULL,
                encoding_signature VARCHAR NOT NULL UNIQUE,
                region_detection_job_id VARCHAR,
                chunk_size_seconds FLOAT,
                chunk_hop_seconds FLOAT,
                crnn_checkpoint_sha256 TEXT,
                crnn_segmentation_model_id VARCHAR,
                projection_kind TEXT,
                projection_dim INTEGER,
                total_regions INTEGER,
                total_chunks INTEGER,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL
            );
            CREATE TABLE alembic_version (
                version_num VARCHAR(32) NOT NULL
            );
            INSERT INTO alembic_version (version_num) VALUES ('075');
            """
        )
        conn.commit()
    finally:
        conn.close()


def test_upgrade_creates_event_encoder_jobs_table(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    _create_pre_076_schema(db_path)

    command.upgrade(_alembic_config(db_path), "076")

    cols = _columns(db_path, "event_encoder_jobs")
    assert cols["id"]["notnull"] is True
    assert cols["status"]["dflt_value"] == "'queued'"
    assert cols["event_segmentation_job_id"]["notnull"] is True
    assert cols["continuous_embedding_job_id"]["notnull"] is True
    assert cols["continuous_embedding_signature"]["notnull"] is True
    assert cols["tokenization_signature"]["notnull"] is True
    assert cols["pooling_config_json"]["notnull"] is True
    assert cols["descriptor_config_json"]["notnull"] is True
    assert cols["preprocessing_config_json"]["notnull"] is True
    assert cols["k_values_json"]["notnull"] is True
    assert "event_vectors_path" in cols
    assert "event_tokens_path" in cols
    assert "token_sequences_path" in cols

    indexes = _indexes(db_path, "event_encoder_jobs")
    assert "ix_event_encoder_jobs_status" in indexes

    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "INSERT INTO region_detection_jobs "
            "(id, status, created_at, updated_at) "
            "VALUES ('rd-1', 'complete', '2026-05-07', '2026-05-07')"
        )
        conn.execute(
            "INSERT INTO event_segmentation_jobs "
            "(id, status, region_detection_job_id, created_at, updated_at) "
            "VALUES ('seg-1', 'complete', 'rd-1', '2026-05-07', '2026-05-07')"
        )
        conn.execute(
            "INSERT INTO continuous_embedding_jobs "
            "(id, status, event_source_mode, model_version, target_sample_rate, "
            " encoding_signature, created_at, updated_at) "
            "VALUES ('cej-1', 'complete', 'raw', 'crnn-call-parsing-pytorch', "
            " 16000, 'cej-sig', '2026-05-07', '2026-05-07')"
        )
        base_values = (
            "queued",
            "seg-1",
            "raw",
            "cej-1",
            "cej-sig",
            "crnn-event-encoder-v1",
            "{}",
            "{}",
            "{}",
            "[50]",
            0,
            "tok-sig",
            "2026-05-07",
            "2026-05-07",
        )
        conn.execute(
            "INSERT INTO event_encoder_jobs "
            "(id, status, event_segmentation_job_id, event_source_mode, "
            " continuous_embedding_job_id, continuous_embedding_signature, "
            " tokenizer_version, pooling_config_json, descriptor_config_json, "
            " preprocessing_config_json, k_values_json, random_seed, "
            " tokenization_signature, created_at, updated_at) "
            "VALUES ('eej-1', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            base_values,
        )
        try:
            conn.execute(
                "INSERT INTO event_encoder_jobs "
                "(id, status, event_segmentation_job_id, event_source_mode, "
                " continuous_embedding_job_id, continuous_embedding_signature, "
                " tokenizer_version, pooling_config_json, descriptor_config_json, "
                " preprocessing_config_json, k_values_json, random_seed, "
                " tokenization_signature, created_at, updated_at) "
                "VALUES ('eej-2', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                base_values,
            )
        except sqlite3.IntegrityError:
            duplicate_rejected = True
        else:
            duplicate_rejected = False
    finally:
        conn.close()
    assert duplicate_rejected is True


def test_downgrade_removes_event_encoder_jobs_table(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    _create_pre_076_schema(db_path)
    cfg = _alembic_config(db_path)
    command.upgrade(cfg, "076")

    command.downgrade(cfg, "075")

    conn = sqlite3.connect(db_path)
    try:
        exists = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='event_encoder_jobs'"
        ).fetchone()
    finally:
        conn.close()
    assert exists is None
