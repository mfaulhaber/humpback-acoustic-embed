"""Tests for migration 074 (masked-transformer job source rows)."""

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


def _columns(db_path: Path, table: str) -> dict[str, dict[str, object]]:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    finally:
        conn.close()
    return {
        row[1]: {"type": row[2], "notnull": bool(row[3]), "default": row[4]}
        for row in rows
    }


def _create_pre_074_schema(db_path: Path) -> None:
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
                encoding_signature VARCHAR NOT NULL,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL
            );
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
                retrieval_head_arch TEXT NOT NULL DEFAULT 'mlp',
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
            INSERT INTO continuous_embedding_jobs (
                id, status, event_source_mode, model_version, target_sample_rate,
                encoding_signature, created_at, updated_at
            ) VALUES (
                'cej-1', 'complete', 'raw', 'region-crnn-v1', 32000,
                'cej-sig-1', '2026-05-06', '2026-05-06'
            );
            INSERT INTO masked_transformer_jobs (
                id, status, continuous_embedding_job_id, event_classification_job_id,
                training_signature, preset, mask_fraction, span_length_min,
                span_length_max, dropout, mask_weight_bias, cosine_loss_weight,
                batch_size, retrieval_head_enabled, retrieval_l2_normalize,
                retrieval_head_arch, sequence_construction_mode,
                event_centered_fraction, max_epochs, early_stop_patience, val_split,
                seed, k_values, created_at, updated_at
            ) VALUES (
                'mt-1', 'complete', 'cej-1', 'cls-1',
                'sig-1', 'small', 0.2, 2,
                6, 0.1, 1, 0.0,
                8, 0, 1,
                'mlp', 'region',
                0.0, 30, 3, 0.1,
                42, '[10]', '2026-05-06', '2026-05-06'
            );
            CREATE TABLE alembic_version (
                version_num VARCHAR(32) NOT NULL
            );
            INSERT INTO alembic_version (version_num) VALUES ('073');
            """
        )
        conn.commit()
    finally:
        conn.close()


def test_upgrade_creates_source_table_and_constraints(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    _create_pre_074_schema(db_path)

    command.upgrade(_alembic_config(db_path), "074")

    assert "masked_transformer_job_sources" in _tables(db_path)
    cols = _columns(db_path, "masked_transformer_job_sources")
    assert cols["id"]["notnull"] is True
    assert cols["masked_transformer_job_id"]["notnull"] is True
    assert cols["source_order"]["notnull"] is True
    assert cols["continuous_embedding_job_id"]["notnull"] is True
    assert cols["event_classification_job_id"]["notnull"] is True
    assert cols["source_alias"]["notnull"] is False
    assert cols["created_at"]["notnull"] is True
    assert cols["updated_at"]["notnull"] is True

    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "INSERT INTO masked_transformer_job_sources "
            "(id, masked_transformer_job_id, source_order, "
            " continuous_embedding_job_id, event_classification_job_id, "
            " created_at, updated_at) "
            "VALUES "
            "('src-1', 'mt-1', 0, 'cej-1', 'cls-1', '2026-05-06', '2026-05-06')"
        )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO masked_transformer_job_sources "
                "(id, masked_transformer_job_id, source_order, "
                " continuous_embedding_job_id, event_classification_job_id, "
                " created_at, updated_at) "
                "VALUES "
                "('src-dup-order', 'mt-1', 0, 'cej-1', 'cls-2', "
                " '2026-05-06', '2026-05-06')"
            )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO masked_transformer_job_sources "
                "(id, masked_transformer_job_id, source_order, "
                " continuous_embedding_job_id, event_classification_job_id, "
                " created_at, updated_at) "
                "VALUES "
                "('src-dup-pair', 'mt-1', 1, 'cej-1', 'cls-1', "
                " '2026-05-06', '2026-05-06')"
            )
    finally:
        conn.close()


def test_downgrade_removes_source_table(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    _create_pre_074_schema(db_path)
    cfg = _alembic_config(db_path)

    command.upgrade(cfg, "074")
    assert "masked_transformer_job_sources" in _tables(db_path)

    command.downgrade(cfg, "073")

    assert "masked_transformer_job_sources" not in _tables(db_path)
