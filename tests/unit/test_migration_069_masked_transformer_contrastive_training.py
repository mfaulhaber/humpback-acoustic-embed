"""Tests for migration 069 (masked-transformer contrastive config)."""

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


def _create_pre_069_schema(db_path: Path) -> None:
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
                sequence_construction_mode TEXT NOT NULL DEFAULT 'region',
                event_centered_fraction FLOAT NOT NULL DEFAULT 0.0,
                pre_event_context_sec FLOAT,
                post_event_context_sec FLOAT,
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
                sequence_construction_mode, event_centered_fraction,
                max_epochs, early_stop_patience, val_split, seed, k_values,
                created_at, updated_at
            ) VALUES (
                'mt-1', 'complete', 'cej-1', 'cls-1',
                'sig-1', 'small', 0.2, 2,
                6, 0.1, 1, 0.0,
                0, 1,
                'region', 0.0,
                30, 3, 0.1, 42, '[10]',
                '2026-05-05', '2026-05-05'
            );
            CREATE TABLE alembic_version (
                version_num VARCHAR(32) NOT NULL
            );
            INSERT INTO alembic_version (version_num) VALUES ('068');
            """
        )
        conn.commit()
    finally:
        conn.close()


def test_upgrade_adds_contrastive_defaults(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    _create_pre_069_schema(db_path)

    command.upgrade(_alembic_config(db_path), "069")

    cols = _columns(db_path)
    assert cols["contrastive_loss_weight"]["notnull"] is True
    assert cols["contrastive_temperature"]["notnull"] is True
    assert cols["contrastive_label_source"]["notnull"] is True
    assert cols["contrastive_min_events_per_label"]["notnull"] is True
    assert cols["contrastive_min_regions_per_label"]["notnull"] is True
    assert cols["require_cross_region_positive"]["notnull"] is True
    assert cols["related_label_policy_json"]["notnull"] is False

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT contrastive_loss_weight, contrastive_temperature, "
            "contrastive_label_source, contrastive_min_events_per_label, "
            "contrastive_min_regions_per_label, require_cross_region_positive, "
            "related_label_policy_json "
            "FROM masked_transformer_jobs WHERE id='mt-1'"
        ).fetchone()
    finally:
        conn.close()

    assert row == (0.0, 0.07, "none", 4, 2, 1, None)


def test_downgrade_removes_contrastive_columns(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    _create_pre_069_schema(db_path)
    cfg = _alembic_config(db_path)

    command.upgrade(cfg, "069")
    command.downgrade(cfg, "068")

    cols = _columns(db_path)
    assert "contrastive_loss_weight" not in cols
    assert "contrastive_temperature" not in cols
    assert "contrastive_label_source" not in cols
    assert "contrastive_min_events_per_label" not in cols
    assert "contrastive_min_regions_per_label" not in cols
    assert "require_cross_region_positive" not in cols
    assert "related_label_policy_json" not in cols
