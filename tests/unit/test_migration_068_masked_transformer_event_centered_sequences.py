"""Tests for migration 068 (masked-transformer sequence construction config)."""

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


def _columns(db_path: Path) -> dict[str, dict[str, object]]:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute("PRAGMA table_info(masked_transformer_jobs)").fetchall()
    finally:
        conn.close()
    return {
        row[1]: {"type": row[2], "notnull": bool(row[3]), "default": row[4]}
        for row in rows
    }


def _create_pre_068_schema(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE masked_transformer_jobs (
                id VARCHAR NOT NULL PRIMARY KEY,
                status VARCHAR NOT NULL,
                status_reason TEXT,
                continuous_embedding_job_id VARCHAR NOT NULL,
                event_classification_job_id VARCHAR,
                training_signature VARCHAR NOT NULL,
                preset TEXT NOT NULL,
                mask_fraction FLOAT NOT NULL,
                span_length_min INTEGER NOT NULL,
                span_length_max INTEGER NOT NULL,
                dropout FLOAT NOT NULL,
                mask_weight_bias BOOLEAN NOT NULL,
                cosine_loss_weight FLOAT NOT NULL,
                retrieval_head_enabled BOOLEAN NOT NULL DEFAULT 0,
                retrieval_dim INTEGER,
                retrieval_hidden_dim INTEGER,
                retrieval_l2_normalize BOOLEAN NOT NULL DEFAULT 1,
                max_epochs INTEGER NOT NULL,
                early_stop_patience INTEGER NOT NULL,
                val_split FLOAT NOT NULL,
                seed INTEGER NOT NULL,
                k_values TEXT NOT NULL,
                chosen_device TEXT,
                fallback_reason TEXT,
                final_train_loss FLOAT,
                final_val_loss FLOAT,
                total_epochs INTEGER,
                job_dir TEXT,
                total_sequences INTEGER,
                total_chunks INTEGER,
                error_message TEXT,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL
            );
            INSERT INTO masked_transformer_jobs (
                id, status, continuous_embedding_job_id, event_classification_job_id,
                training_signature, preset, mask_fraction, span_length_min,
                span_length_max, dropout, mask_weight_bias, cosine_loss_weight,
                retrieval_head_enabled, retrieval_l2_normalize,
                max_epochs, early_stop_patience, val_split, seed, k_values,
                created_at, updated_at
            ) VALUES (
                'mt-1', 'complete', 'cej-1', 'cls-1',
                'sig-1', 'small', 0.2, 2,
                6, 0.1, 1, 0.0,
                0, 1,
                30, 3, 0.1, 42, '[10]',
                '2026-05-05', '2026-05-05'
            );
            CREATE TABLE alembic_version (
                version_num VARCHAR(32) NOT NULL
            );
            INSERT INTO alembic_version (version_num) VALUES ('067');
            """
        )
        conn.commit()
    finally:
        conn.close()


def test_upgrade_adds_sequence_construction_defaults(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    _create_pre_068_schema(db_path)

    command.upgrade(_alembic_config(db_path), "068")

    cols = _columns(db_path)
    assert cols["sequence_construction_mode"]["notnull"] is True
    assert cols["event_centered_fraction"]["notnull"] is True
    assert cols["pre_event_context_sec"]["notnull"] is False
    assert cols["post_event_context_sec"]["notnull"] is False

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT sequence_construction_mode, event_centered_fraction, "
            "pre_event_context_sec, post_event_context_sec "
            "FROM masked_transformer_jobs WHERE id='mt-1'"
        ).fetchone()
    finally:
        conn.close()

    assert row == ("region", 0.0, None, None)


def test_downgrade_removes_sequence_construction_columns(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    _create_pre_068_schema(db_path)
    cfg = _alembic_config(db_path)

    command.upgrade(cfg, "068")
    command.downgrade(cfg, "067")

    cols = _columns(db_path)
    assert "sequence_construction_mode" not in cols
    assert "event_centered_fraction" not in cols
    assert "pre_event_context_sec" not in cols
    assert "post_event_context_sec" not in cols
