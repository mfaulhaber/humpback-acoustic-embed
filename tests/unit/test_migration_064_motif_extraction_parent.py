"""Tests for migration 064 (motif_extraction_jobs parent generalization)."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config

from humpback.database import Base, create_engine
import humpback.models.sequence_models  # noqa: F401  (registers tables)


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


def _stamp_pre_064(db_path: Path) -> None:
    """Build a pre-064 schema by running migrations 062 + 063."""
    asyncio.run(_create_db(db_path))
    # Reset the schema to a known prior state by dropping tables we'll
    # recreate via alembic.
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("DROP TABLE IF EXISTS motif_extraction_jobs")
        conn.execute("DROP TABLE IF EXISTS masked_transformer_jobs")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS alembic_version "
            "(version_num VARCHAR(32) NOT NULL)"
        )
        conn.execute("DELETE FROM alembic_version")
        conn.execute("INSERT INTO alembic_version (version_num) VALUES ('061')")
        conn.commit()
    finally:
        conn.close()
    cfg = _alembic_config(db_path)
    command.upgrade(cfg, "063")


def test_upgrade_adds_columns_and_keeps_existing(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    _stamp_pre_064(db_path)

    # Insert a pre-existing HMM-parent row to exercise backfill.
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "INSERT INTO motif_extraction_jobs "
            "(id, status, hmm_sequence_job_id, source_kind, "
            " min_ngram, max_ngram, minimum_occurrences, minimum_event_sources, "
            " frequency_weight, event_source_weight, event_core_weight, "
            " low_background_weight, call_probability_weight, config_signature, "
            " created_at, updated_at) "
            "VALUES "
            "('m1', 'queued', 'hmm-1', 'region_crnn', "
            " 2, 8, 5, 2, "
            " 0.4, 0.3, 0.2, 0.1, NULL, 'sig-x', "
            " '2026-05-01 00:00:00', '2026-05-01 00:00:00')"
        )
        conn.commit()
    finally:
        conn.close()

    cfg = _alembic_config(db_path)
    command.upgrade(cfg, "064")

    cols = _columns(db_path, "motif_extraction_jobs")
    assert "parent_kind" in cols
    assert cols["parent_kind"]["notnull"] is True
    assert "masked_transformer_job_id" in cols
    assert cols["masked_transformer_job_id"]["notnull"] is False
    assert "k" in cols
    assert cols["k"]["notnull"] is False
    # hmm_sequence_job_id is now nullable.
    assert cols["hmm_sequence_job_id"]["notnull"] is False

    # Backfill: existing row sees parent_kind='hmm'.
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT id, parent_kind, hmm_sequence_job_id, "
            "masked_transformer_job_id, k FROM motif_extraction_jobs"
        ).fetchall()
    finally:
        conn.close()
    assert rows == [("m1", "hmm", "hmm-1", None, None)]


def test_upgrade_then_downgrade_round_trip(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    _stamp_pre_064(db_path)
    cfg = _alembic_config(db_path)
    command.upgrade(cfg, "064")
    cols = _columns(db_path, "motif_extraction_jobs")
    assert "parent_kind" in cols

    command.downgrade(cfg, "063")
    cols_post = _columns(db_path, "motif_extraction_jobs")
    assert "parent_kind" not in cols_post
    assert "masked_transformer_job_id" not in cols_post
    assert "k" not in cols_post


def test_check_constraint_rejects_invalid_combinations(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    _stamp_pre_064(db_path)
    cfg = _alembic_config(db_path)
    command.upgrade(cfg, "064")

    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys=OFF")  # FK target rows not seeded

        # Both FKs set: should violate XOR check.
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO motif_extraction_jobs "
                "(id, status, parent_kind, hmm_sequence_job_id, "
                " masked_transformer_job_id, k, source_kind, min_ngram, max_ngram, "
                " minimum_occurrences, minimum_event_sources, frequency_weight, "
                " event_source_weight, event_core_weight, low_background_weight, "
                " call_probability_weight, config_signature, created_at, updated_at) "
                "VALUES "
                "('bad-both', 'queued', 'hmm', 'hmm-1', 'mt-1', 100, "
                " 'region_crnn', 2, 8, 5, 2, 0.4, 0.3, 0.2, 0.1, NULL, 'sig-bb', "
                " '2026-05-01', '2026-05-01')"
            )

        # Neither FK set.
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO motif_extraction_jobs "
                "(id, status, parent_kind, source_kind, min_ngram, max_ngram, "
                " minimum_occurrences, minimum_event_sources, frequency_weight, "
                " event_source_weight, event_core_weight, low_background_weight, "
                " call_probability_weight, config_signature, created_at, updated_at) "
                "VALUES "
                "('bad-neither', 'queued', 'hmm', 'region_crnn', 2, 8, 5, 2, "
                " 0.4, 0.3, 0.2, 0.1, NULL, 'sig-bn', '2026-05-01', '2026-05-01')"
            )

        # k set with parent_kind='hmm' is invalid.
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO motif_extraction_jobs "
                "(id, status, parent_kind, hmm_sequence_job_id, k, source_kind, "
                " min_ngram, max_ngram, minimum_occurrences, "
                " minimum_event_sources, frequency_weight, event_source_weight, "
                " event_core_weight, low_background_weight, "
                " call_probability_weight, config_signature, created_at, "
                " updated_at) "
                "VALUES "
                "('bad-k', 'queued', 'hmm', 'hmm-1', 100, 'region_crnn', 2, 8, 5, "
                " 2, 0.4, 0.3, 0.2, 0.1, NULL, 'sig-bk', '2026-05-01', "
                " '2026-05-01')"
            )

        # Valid HMM-parent insert.
        conn.execute(
            "INSERT INTO motif_extraction_jobs "
            "(id, status, parent_kind, hmm_sequence_job_id, source_kind, "
            " min_ngram, max_ngram, minimum_occurrences, minimum_event_sources, "
            " frequency_weight, event_source_weight, event_core_weight, "
            " low_background_weight, call_probability_weight, config_signature, "
            " created_at, updated_at) "
            "VALUES "
            "('ok-hmm', 'queued', 'hmm', 'hmm-1', 'region_crnn', 2, 8, 5, 2, "
            " 0.4, 0.3, 0.2, 0.1, NULL, 'sig-okh', '2026-05-01', '2026-05-01')"
        )

        # Valid masked-transformer-parent insert.
        conn.execute(
            "INSERT INTO motif_extraction_jobs "
            "(id, status, parent_kind, masked_transformer_job_id, k, "
            " source_kind, min_ngram, max_ngram, minimum_occurrences, "
            " minimum_event_sources, frequency_weight, event_source_weight, "
            " event_core_weight, low_background_weight, "
            " call_probability_weight, config_signature, created_at, "
            " updated_at) "
            "VALUES "
            "('ok-mt', 'queued', 'masked_transformer', 'mt-1', 100, "
            " 'region_crnn', 2, 8, 5, 2, 0.4, 0.3, 0.2, 0.1, NULL, "
            " 'sig-okm', '2026-05-01', '2026-05-01')"
        )
        conn.commit()
    finally:
        conn.close()
