"""Round-trip tests for migration 061 (CRNN region embeddings + HMM modes)."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

from alembic import command
from alembic.config import Config

import humpback.models.call_parsing  # noqa: F401  - register tables
import humpback.models.sequence_models  # noqa: F401
from humpback.database import Base, create_engine


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
    config = Config(str(repo_root / "alembic.ini"))
    config.set_main_option("script_location", str(repo_root / "alembic"))
    config.set_main_option("sqlalchemy.url", _db_url(db_path))
    return config


def _columns(db_path: Path, table: str) -> dict[str, dict]:
    """Return ``{name: {nullable, default, type}}`` from ``PRAGMA table_info``."""
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    finally:
        conn.close()
    return {
        r[1]: {"type": r[2], "notnull": bool(r[3]), "default": r[4], "pk": bool(r[5])}
        for r in rows
    }


def _stamp_pre_061(db_path: Path) -> None:
    """Build the schema at revision 060 (pre-061) for the round-trip."""
    asyncio.run(_create_db(db_path))
    cfg = _alembic_config(db_path)
    command.stamp(cfg, "060")
    # Roll the schema back to match 060: drop the columns 061 will add,
    # so the upgrade has work to do. ``Base.metadata.create_all`` already
    # built the post-061 schema since the SQLAlchemy models include the
    # new columns. Reset by dropping and re-creating without those.
    conn = sqlite3.connect(db_path)
    try:
        # Recreate continuous_embedding_jobs without 061's new columns,
        # mirroring the 060 schema.
        conn.executescript(
            """
            DROP TABLE IF EXISTS continuous_embedding_jobs;
            CREATE TABLE continuous_embedding_jobs (
                id VARCHAR NOT NULL,
                status VARCHAR NOT NULL DEFAULT 'queued',
                event_segmentation_job_id VARCHAR,
                model_version VARCHAR NOT NULL,
                window_size_seconds FLOAT NOT NULL,
                hop_seconds FLOAT NOT NULL,
                pad_seconds FLOAT NOT NULL,
                target_sample_rate INTEGER NOT NULL,
                feature_config_json TEXT,
                encoding_signature VARCHAR NOT NULL,
                vector_dim INTEGER,
                total_events INTEGER,
                merged_spans INTEGER,
                total_windows INTEGER,
                parquet_path VARCHAR,
                error_message TEXT,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL,
                PRIMARY KEY (id),
                UNIQUE (encoding_signature)
            );
            DROP TABLE IF EXISTS hmm_sequence_jobs;
            CREATE TABLE hmm_sequence_jobs (
                id VARCHAR NOT NULL,
                status VARCHAR NOT NULL DEFAULT 'queued',
                continuous_embedding_job_id VARCHAR NOT NULL,
                n_states INTEGER NOT NULL,
                pca_dims INTEGER NOT NULL,
                pca_whiten BOOLEAN NOT NULL DEFAULT 0,
                l2_normalize BOOLEAN NOT NULL DEFAULT 1,
                covariance_type VARCHAR NOT NULL DEFAULT 'diag',
                n_iter INTEGER NOT NULL DEFAULT 100,
                random_seed INTEGER NOT NULL DEFAULT 42,
                min_sequence_length_frames INTEGER NOT NULL DEFAULT 3,
                tol FLOAT NOT NULL DEFAULT 0.0001,
                library VARCHAR NOT NULL DEFAULT 'hmmlearn',
                train_log_likelihood FLOAT,
                n_train_sequences INTEGER,
                n_train_frames INTEGER,
                n_decoded_sequences INTEGER,
                artifact_dir VARCHAR,
                error_message TEXT,
                created_at DATETIME NOT NULL,
                updated_at DATETIME NOT NULL,
                PRIMARY KEY (id)
            );
            """
        )
        conn.commit()
    finally:
        conn.close()


def test_upgrade_adds_new_columns(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    _stamp_pre_061(db_path)

    cols_pre = _columns(db_path, "continuous_embedding_jobs")
    assert "region_detection_job_id" not in cols_pre
    assert "chunk_size_seconds" not in cols_pre
    assert cols_pre["window_size_seconds"]["notnull"] is True

    cfg = _alembic_config(db_path)
    command.upgrade(cfg, "061")

    cols_post = _columns(db_path, "continuous_embedding_jobs")
    expected_new = {
        "region_detection_job_id",
        "chunk_size_seconds",
        "chunk_hop_seconds",
        "crnn_checkpoint_sha256",
        "crnn_segmentation_model_id",
        "projection_kind",
        "projection_dim",
        "total_regions",
        "total_chunks",
    }
    assert expected_new.issubset(cols_post.keys())
    for c in expected_new:
        assert cols_post[c]["notnull"] is False, f"{c} should be nullable"

    # Legacy SurfPerch fields are now nullable.
    for c in ("window_size_seconds", "hop_seconds", "pad_seconds"):
        assert cols_post[c]["notnull"] is False, f"{c} should be nullable post-061"

    hmm_cols = _columns(db_path, "hmm_sequence_jobs")
    expected_hmm = {
        "training_mode",
        "event_core_overlap_threshold",
        "near_event_window_seconds",
        "event_balanced_proportions",
        "subsequence_length_chunks",
        "subsequence_stride_chunks",
        "target_train_chunks",
        "min_region_length_seconds",
    }
    assert expected_hmm.issubset(hmm_cols.keys())
    for c in expected_hmm:
        assert hmm_cols[c]["notnull"] is False, f"{c} should be nullable"

    # Server defaults landed for the columns that take them.
    assert hmm_cols["event_core_overlap_threshold"]["default"] is not None
    assert hmm_cols["near_event_window_seconds"]["default"] is not None
    assert hmm_cols["subsequence_length_chunks"]["default"] is not None
    assert hmm_cols["target_train_chunks"]["default"] is not None
    assert hmm_cols["min_region_length_seconds"]["default"] is not None


def test_downgrade_removes_new_columns(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    _stamp_pre_061(db_path)
    cfg = _alembic_config(db_path)
    command.upgrade(cfg, "061")
    command.downgrade(cfg, "060")

    cols_post = _columns(db_path, "continuous_embedding_jobs")
    forbidden = {
        "region_detection_job_id",
        "chunk_size_seconds",
        "chunk_hop_seconds",
        "crnn_checkpoint_sha256",
        "crnn_segmentation_model_id",
        "projection_kind",
        "projection_dim",
        "total_regions",
        "total_chunks",
    }
    assert forbidden.isdisjoint(cols_post.keys())
    for c in ("window_size_seconds", "hop_seconds", "pad_seconds"):
        assert cols_post[c]["notnull"] is True, (
            f"{c} should be NOT NULL after downgrade"
        )

    hmm_cols = _columns(db_path, "hmm_sequence_jobs")
    forbidden_hmm = {
        "training_mode",
        "event_core_overlap_threshold",
        "near_event_window_seconds",
        "event_balanced_proportions",
        "subsequence_length_chunks",
        "subsequence_stride_chunks",
        "target_train_chunks",
        "min_region_length_seconds",
    }
    assert forbidden_hmm.isdisjoint(hmm_cols.keys())
