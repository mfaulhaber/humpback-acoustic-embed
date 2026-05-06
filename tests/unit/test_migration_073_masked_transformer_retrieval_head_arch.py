"""Tests for migration 073 (masked-transformer retrieval-head architecture)."""

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


def _create_pre_073_schema(db_path: Path) -> None:
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
                batch_size INTEGER NOT NULL DEFAULT '8',
                retrieval_head_enabled BOOLEAN NOT NULL DEFAULT 0,
                retrieval_dim INTEGER,
                retrieval_hidden_dim INTEGER,
                retrieval_l2_normalize BOOLEAN NOT NULL DEFAULT 1,
                sequence_construction_mode TEXT NOT NULL DEFAULT 'region',
                event_centered_fraction FLOAT NOT NULL DEFAULT 0.0,
                pre_event_context_sec FLOAT,
                post_event_context_sec FLOAT,
                contrastive_loss_weight FLOAT NOT NULL DEFAULT 0.0,
                contrastive_temperature FLOAT NOT NULL DEFAULT 0.07,
                contrastive_label_source TEXT NOT NULL DEFAULT 'none',
                contrastive_min_events_per_label INTEGER NOT NULL DEFAULT 4,
                contrastive_min_regions_per_label INTEGER NOT NULL DEFAULT 2,
                require_cross_region_positive BOOLEAN NOT NULL DEFAULT 1,
                related_label_policy_json TEXT,
                contrastive_sampler_enabled BOOLEAN NOT NULL DEFAULT 1,
                contrastive_labels_per_batch INTEGER NOT NULL DEFAULT 4,
                contrastive_events_per_label INTEGER NOT NULL DEFAULT 4,
                contrastive_max_unlabeled_fraction FLOAT NOT NULL DEFAULT 0.25,
                contrastive_region_balance BOOLEAN NOT NULL DEFAULT 1,
                training_freeze_mode TEXT NOT NULL DEFAULT 'none',
                source_masked_transformer_job_id VARCHAR,
                negative_label_family_policy_json TEXT,
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
                batch_size, retrieval_head_enabled, retrieval_l2_normalize,
                sequence_construction_mode, event_centered_fraction,
                max_epochs, early_stop_patience, val_split, seed, k_values,
                created_at, updated_at
            ) VALUES (
                'mt-1', 'complete', 'cej-1', 'cls-1',
                'sig-1', 'small', 0.2, 2,
                6, 0.1, 1, 0.0,
                8, 0, 1,
                'region', 0.0,
                30, 3, 0.1, 42, '[10]',
                '2026-05-06', '2026-05-06'
            );
            CREATE TABLE alembic_version (
                version_num VARCHAR(32) NOT NULL
            );
            INSERT INTO alembic_version (version_num) VALUES ('072');
            """
        )
        conn.commit()
    finally:
        conn.close()


def test_upgrade_adds_retrieval_head_arch_default(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    _create_pre_073_schema(db_path)

    command.upgrade(_alembic_config(db_path), "073")

    cols = _columns(db_path)
    assert cols["retrieval_head_arch"]["notnull"] is True
    assert cols["retrieval_head_arch"]["default"] == "'mlp'"

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT retrieval_head_arch FROM masked_transformer_jobs WHERE id='mt-1'"
        ).fetchone()
    finally:
        conn.close()

    assert row == ("mlp",)


def test_downgrade_removes_retrieval_head_arch(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    _create_pre_073_schema(db_path)
    cfg = _alembic_config(db_path)

    command.upgrade(cfg, "073")
    command.downgrade(cfg, "072")

    cols = _columns(db_path)
    assert "retrieval_head_arch" not in cols
