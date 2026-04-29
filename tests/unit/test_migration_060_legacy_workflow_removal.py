"""Tests for migration 060 (legacy workflow removal)."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
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


def _create_pre_060_schema(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE audio_files (
                id TEXT PRIMARY KEY NOT NULL,
                filename TEXT NOT NULL,
                folder_path TEXT NOT NULL DEFAULT '',
                source_folder TEXT,
                checksum_sha256 TEXT NOT NULL,
                duration_seconds FLOAT,
                sample_rate_original INTEGER,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE audio_metadata (
                id TEXT PRIMARY KEY NOT NULL,
                audio_file_id TEXT NOT NULL UNIQUE,
                tag_data TEXT,
                visual_observations TEXT,
                group_composition TEXT,
                prey_density_proxy TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(audio_file_id) REFERENCES audio_files(id)
            );
            CREATE TABLE processing_jobs (
                id TEXT PRIMARY KEY NOT NULL,
                audio_file_id TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'queued',
                encoding_signature TEXT NOT NULL,
                model_version TEXT NOT NULL,
                window_size_seconds FLOAT NOT NULL,
                target_sample_rate INTEGER NOT NULL,
                feature_config TEXT,
                error_message TEXT,
                warning_message TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(audio_file_id) REFERENCES audio_files(id)
            );
            CREATE TABLE embedding_sets (
                id TEXT PRIMARY KEY NOT NULL,
                audio_file_id TEXT NOT NULL,
                encoding_signature TEXT NOT NULL,
                model_version TEXT NOT NULL,
                window_size_seconds FLOAT NOT NULL,
                target_sample_rate INTEGER NOT NULL,
                vector_dim INTEGER NOT NULL,
                parquet_path TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(audio_file_id) REFERENCES audio_files(id)
            );
            CREATE TABLE classifier_models (
                id TEXT PRIMARY KEY NOT NULL,
                name TEXT NOT NULL,
                model_path TEXT NOT NULL,
                model_version TEXT NOT NULL,
                vector_dim INTEGER NOT NULL,
                window_size_seconds FLOAT NOT NULL,
                target_sample_rate INTEGER NOT NULL,
                feature_config TEXT,
                training_summary TEXT,
                training_job_id TEXT,
                classifier_purpose TEXT NOT NULL DEFAULT 'detection',
                training_source_mode TEXT NOT NULL DEFAULT 'embedding_sets',
                source_candidate_id TEXT,
                source_model_id TEXT,
                promotion_provenance TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE classifier_training_jobs (
                id TEXT PRIMARY KEY NOT NULL,
                status TEXT NOT NULL DEFAULT 'queued',
                name TEXT NOT NULL,
                positive_embedding_set_ids TEXT NOT NULL,
                negative_embedding_set_ids TEXT NOT NULL,
                model_version TEXT NOT NULL,
                window_size_seconds FLOAT NOT NULL,
                target_sample_rate INTEGER NOT NULL,
                feature_config TEXT,
                parameters TEXT,
                classifier_model_id TEXT,
                error_message TEXT,
                job_purpose TEXT NOT NULL DEFAULT 'detection',
                source_detection_job_ids TEXT,
                source_mode TEXT NOT NULL DEFAULT 'embedding_sets',
                source_candidate_id TEXT,
                source_model_id TEXT,
                manifest_path TEXT,
                training_split_name TEXT,
                promoted_config TEXT,
                source_comparison_context TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE clustering_jobs (
                id TEXT PRIMARY KEY NOT NULL,
                status TEXT NOT NULL DEFAULT 'queued',
                embedding_set_ids TEXT NOT NULL,
                parameters TEXT,
                error_message TEXT,
                metrics_json TEXT,
                refined_from_job_id TEXT,
                detection_job_ids TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE clusters (
                id TEXT PRIMARY KEY NOT NULL,
                clustering_job_id TEXT NOT NULL,
                cluster_label INTEGER NOT NULL,
                size INTEGER NOT NULL,
                metadata_summary TEXT,
                FOREIGN KEY(clustering_job_id) REFERENCES clustering_jobs(id)
            );
            CREATE TABLE cluster_assignments (
                id TEXT PRIMARY KEY NOT NULL,
                cluster_id TEXT NOT NULL,
                embedding_set_id TEXT NOT NULL,
                embedding_row_index INTEGER NOT NULL,
                FOREIGN KEY(cluster_id) REFERENCES clusters(id)
            );
            CREATE TABLE search_jobs (
                id TEXT PRIMARY KEY NOT NULL,
                status TEXT NOT NULL DEFAULT 'queued',
                detection_job_id TEXT NOT NULL,
                start_utc FLOAT NOT NULL,
                end_utc FLOAT NOT NULL,
                top_k INTEGER NOT NULL DEFAULT 20,
                metric TEXT NOT NULL DEFAULT 'cosine',
                embedding_set_ids TEXT,
                model_version TEXT,
                embedding_vector TEXT,
                error_message TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE label_processing_jobs (
                id TEXT PRIMARY KEY NOT NULL,
                status TEXT NOT NULL DEFAULT 'queued',
                workflow TEXT NOT NULL DEFAULT 'score_based',
                classifier_model_id TEXT,
                annotation_folder TEXT NOT NULL,
                audio_folder TEXT NOT NULL,
                output_root TEXT NOT NULL,
                parameters TEXT,
                files_processed INTEGER,
                files_total INTEGER,
                annotations_total INTEGER,
                result_summary TEXT,
                error_message TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE retrain_workflows (
                id TEXT PRIMARY KEY NOT NULL,
                status TEXT NOT NULL DEFAULT 'queued',
                updated_at TEXT NOT NULL
            );
            CREATE TABLE alembic_version (
                version_num VARCHAR(32) NOT NULL
            );
            INSERT INTO alembic_version (version_num) VALUES ('059');
            """
        )
        conn.commit()
    finally:
        conn.close()


def _tables(db_path: Path) -> set[str]:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    finally:
        conn.close()
    return {row[0] for row in rows}


def _columns(db_path: Path, table: str) -> set[str]:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    finally:
        conn.close()
    return {row[1] for row in rows}


def _column_defaults(db_path: Path, table: str) -> dict[str, str | None]:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    finally:
        conn.close()
    return {row[1]: row[4] for row in rows}


def _seed_success_case(db_path: Path) -> None:
    now = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "INSERT INTO audio_files "
            "(id, filename, folder_path, source_folder, checksum_sha256, "
            " duration_seconds, sample_rate_original, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "af1",
                "sample.wav",
                "folder",
                "/tmp/folder",
                "sha",
                12.5,
                32000,
                now,
                now,
            ),
        )
        conn.execute(
            "INSERT INTO audio_metadata "
            "(id, audio_file_id, tag_data, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("am1", "af1", json.dumps({"species": "hb"}), now, now),
        )
        conn.execute(
            "INSERT INTO processing_jobs "
            "(id, audio_file_id, status, encoding_signature, model_version, "
            " window_size_seconds, target_sample_rate, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("pj1", "af1", "complete", "sig", "perch_v2", 5.0, 32000, now, now),
        )
        conn.execute(
            "INSERT INTO embedding_sets "
            "(id, audio_file_id, encoding_signature, model_version, "
            " window_size_seconds, target_sample_rate, vector_dim, parquet_path, "
            " created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "es_pos",
                "af1",
                "sig-pos",
                "perch_v2",
                5.0,
                32000,
                1536,
                "/tmp/es_pos.parquet",
                now,
                now,
            ),
        )
        conn.execute(
            "INSERT INTO embedding_sets "
            "(id, audio_file_id, encoding_signature, model_version, "
            " window_size_seconds, target_sample_rate, vector_dim, parquet_path, "
            " created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "es_neg",
                "af1",
                "sig-neg",
                "perch_v2",
                5.0,
                32000,
                1536,
                "/tmp/es_neg.parquet",
                now,
                now,
            ),
        )
        conn.execute(
            "INSERT INTO classifier_training_jobs "
            "(id, status, name, positive_embedding_set_ids, negative_embedding_set_ids, "
            " model_version, window_size_seconds, target_sample_rate, classifier_model_id, "
            " source_mode, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "tj1",
                "complete",
                "legacy-job",
                json.dumps(["es_pos"]),
                json.dumps(["es_neg"]),
                "perch_v2",
                5.0,
                32000,
                "cm1",
                "embedding_sets",
                now,
                now,
            ),
        )
        conn.execute(
            "INSERT INTO classifier_models "
            "(id, name, model_path, model_version, vector_dim, window_size_seconds, "
            " target_sample_rate, training_summary, training_job_id, classifier_purpose, "
            " training_source_mode, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "cm1",
                "legacy-model",
                "/tmp/model.joblib",
                "perch_v2",
                1536,
                5.0,
                32000,
                json.dumps({"n_positive": 120, "n_negative": 95}),
                "tj1",
                "detection",
                "embedding_sets",
                now,
                now,
            ),
        )
        conn.execute(
            "INSERT INTO clustering_jobs "
            "(id, status, embedding_set_ids, parameters, error_message, metrics_json, "
            " refined_from_job_id, detection_job_ids, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "cj1",
                "complete",
                "[]",
                None,
                None,
                None,
                None,
                json.dumps(["dj1"]),
                now,
                now,
            ),
        )
        conn.execute(
            "INSERT INTO clusters "
            "(id, clustering_job_id, cluster_label, size, metadata_summary) "
            "VALUES (?, ?, ?, ?, ?)",
            ("cl1", "cj1", 0, 1, None),
        )
        conn.execute(
            "INSERT INTO cluster_assignments "
            "(id, cluster_id, embedding_set_id, embedding_row_index) "
            "VALUES (?, ?, ?, ?)",
            ("ca1", "cl1", "dj1", 7),
        )
        conn.execute(
            "INSERT INTO search_jobs "
            "(id, status, detection_job_id, start_utc, end_utc, top_k, metric, "
            " created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("sj1", "complete", "dj1", 1.0, 2.0, 20, "cosine", now, now),
        )
        conn.execute(
            "INSERT INTO label_processing_jobs "
            "(id, status, workflow, annotation_folder, audio_folder, output_root, "
            " created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "lp1",
                "complete",
                "score_based",
                "/tmp/ann",
                "/tmp/audio",
                "/tmp/out",
                now,
                now,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _seed_blocker_case(db_path: Path) -> None:
    now = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "INSERT INTO audio_files "
            "(id, filename, folder_path, source_folder, checksum_sha256, "
            " created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("af1", "sample.wav", "folder", "/tmp/folder", "sha", now, now),
        )
        conn.execute(
            "INSERT INTO processing_jobs "
            "(id, audio_file_id, status, encoding_signature, model_version, "
            " window_size_seconds, target_sample_rate, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("pj1", "af1", "queued", "sig", "perch_v2", 5.0, 32000, now, now),
        )
        conn.execute(
            "INSERT INTO clustering_jobs "
            "(id, status, embedding_set_ids, parameters, error_message, metrics_json, "
            " refined_from_job_id, detection_job_ids, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "cj1",
                "queued",
                json.dumps(["es1"]),
                None,
                None,
                None,
                None,
                None,
                now,
                now,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def test_upgrade_blocks_when_cleanup_blockers_remain(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.db"
    _create_pre_060_schema(db_path)
    _seed_blocker_case(db_path)

    cfg = _alembic_config(db_path)
    try:
        command.upgrade(cfg, "060")
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("migration unexpectedly succeeded")

    assert "processing_jobs_active=1" in message
    assert "legacy_clustering_jobs=1" in message


def test_upgrade_drops_legacy_tables_and_preserves_classifier_provenance(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "legacy.db"
    _create_pre_060_schema(db_path)
    _seed_success_case(db_path)

    cfg = _alembic_config(db_path)
    command.upgrade(cfg, "060")

    tables = _tables(db_path)
    assert "search_jobs" not in tables
    assert "label_processing_jobs" not in tables
    assert "processing_jobs" not in tables
    assert "embedding_sets" not in tables
    assert "audio_metadata" not in tables
    assert "audio_files" in tables

    clustering_job_cols = _columns(db_path, "clustering_jobs")
    assert "embedding_set_ids" not in clustering_job_cols
    assert "detection_job_ids" in clustering_job_cols

    assignment_cols = _columns(db_path, "cluster_assignments")
    assert "source_id" in assignment_cols
    assert "embedding_set_id" not in assignment_cols

    training_job_cols = _columns(db_path, "classifier_training_jobs")
    assert "legacy_source_summary" in training_job_cols
    assert "positive_embedding_set_ids" not in training_job_cols
    assert "negative_embedding_set_ids" not in training_job_cols

    defaults = _column_defaults(db_path, "classifier_training_jobs")
    assert defaults["source_mode"] == "'detection_manifest'"
    model_defaults = _column_defaults(db_path, "classifier_models")
    assert model_defaults["training_source_mode"] == "'detection_manifest'"

    conn = sqlite3.connect(db_path)
    try:
        legacy_summary_raw = conn.execute(
            "SELECT legacy_source_summary FROM classifier_training_jobs WHERE id = 'tj1'"
        ).fetchone()[0]
        source_id = conn.execute(
            "SELECT source_id FROM cluster_assignments WHERE id = 'ca1'"
        ).fetchone()[0]
    finally:
        conn.close()

    legacy_summary = json.loads(legacy_summary_raw)
    assert legacy_summary["positive_embedding_set_ids"] == ["es_pos"]
    assert legacy_summary["negative_embedding_set_ids"] == ["es_neg"]
    assert legacy_summary["total_sources"] == 2
    assert source_id == "dj1"
