"""Tests for migration 075 retired Sequence Models table removal."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config


def _alembic_config(db_path: Path) -> Config:
    repo_root = Path(__file__).resolve().parents[2]
    cfg = Config(str(repo_root / "alembic.ini"))
    cfg.set_main_option("script_location", str(repo_root / "alembic"))
    cfg.set_main_option("sqlalchemy.url", f"sqlite+aiosqlite:///{db_path}")
    return cfg


def _tables(db_path: Path) -> set[str]:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    finally:
        conn.close()
    return {row[0] for row in rows}


def _create_pre_075_schema(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
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
            CREATE TABLE hmm_sequence_jobs (
                id VARCHAR NOT NULL PRIMARY KEY,
                status VARCHAR NOT NULL,
                continuous_embedding_job_id VARCHAR NOT NULL,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL
            );
            CREATE TABLE masked_transformer_jobs (
                id VARCHAR NOT NULL PRIMARY KEY,
                status VARCHAR NOT NULL,
                continuous_embedding_job_id VARCHAR NOT NULL,
                training_signature VARCHAR NOT NULL,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL
            );
            CREATE TABLE masked_transformer_job_sources (
                id VARCHAR NOT NULL PRIMARY KEY,
                masked_transformer_job_id VARCHAR NOT NULL,
                source_order INTEGER NOT NULL,
                continuous_embedding_job_id VARCHAR NOT NULL,
                event_classification_job_id VARCHAR NOT NULL,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL,
                FOREIGN KEY(masked_transformer_job_id)
                    REFERENCES masked_transformer_jobs(id)
            );
            CREATE TABLE motif_extraction_jobs (
                id VARCHAR NOT NULL PRIMARY KEY,
                status VARCHAR NOT NULL,
                parent_kind TEXT NOT NULL,
                hmm_sequence_job_id VARCHAR,
                masked_transformer_job_id VARCHAR,
                source_kind TEXT NOT NULL,
                config_signature VARCHAR NOT NULL,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL
            );
            CREATE TABLE alembic_version (
                version_num VARCHAR(32) NOT NULL
            );
            INSERT INTO continuous_embedding_jobs (
                id, status, event_source_mode, model_version, target_sample_rate,
                encoding_signature, region_detection_job_id, chunk_size_seconds,
                chunk_hop_seconds, crnn_checkpoint_sha256,
                crnn_segmentation_model_id, projection_kind, projection_dim,
                total_regions, total_chunks, created_at, updated_at
            ) VALUES (
                'cej-1', 'complete', 'raw', 'crnn-call-parsing-pytorch', 32000,
                'cej-sig-1', 'rd-1', 0.25,
                0.125, 'sha256',
                'seg-1', 'pca', 64,
                2, 10, '2026-05-06', '2026-05-06'
            );
            INSERT INTO alembic_version (version_num) VALUES ('074');
            """
        )
        conn.commit()
    finally:
        conn.close()


def test_upgrade_removes_retired_tables_and_preserves_continuous_embedding(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "test.db"
    _create_pre_075_schema(db_path)

    command.upgrade(_alembic_config(db_path), "075")

    tables = _tables(db_path)
    assert "continuous_embedding_jobs" in tables
    assert "hmm_sequence_jobs" not in tables
    assert "masked_transformer_jobs" not in tables
    assert "masked_transformer_job_sources" not in tables
    assert "motif_extraction_jobs" not in tables

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT id, encoding_signature, region_detection_job_id, "
            "projection_kind, total_chunks FROM continuous_embedding_jobs"
        ).fetchone()
    finally:
        conn.close()

    assert row == ("cej-1", "cej-sig-1", "rd-1", "pca", 10)


def test_downgrade_is_unsupported(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    _create_pre_075_schema(db_path)
    cfg = _alembic_config(db_path)
    command.upgrade(cfg, "075")

    with pytest.raises(NotImplementedError, match="restore the pre-upgrade"):
        command.downgrade(cfg, "074")
