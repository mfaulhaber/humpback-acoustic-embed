"""Round-trip tests for migration 062 (motif extraction jobs)."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

from alembic import command
from alembic.config import Config

import humpback.models.call_parsing  # noqa: F401
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
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    finally:
        conn.close()
    return {
        r[1]: {"type": r[2], "notnull": bool(r[3]), "default": r[4], "pk": bool(r[5])}
        for r in rows
    }


def _indexes(db_path: Path, table: str) -> set[str]:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(f"PRAGMA index_list({table})").fetchall()
    finally:
        conn.close()
    return {r[1] for r in rows}


def _stamp_pre_062(db_path: Path) -> None:
    asyncio.run(_create_db(db_path))
    cfg = _alembic_config(db_path)
    command.stamp(cfg, "061")
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("DROP TABLE IF EXISTS motif_extraction_jobs")
        conn.commit()
    finally:
        conn.close()


def test_upgrade_creates_motif_extraction_jobs(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    _stamp_pre_062(db_path)

    assert _columns(db_path, "motif_extraction_jobs") == {}

    cfg = _alembic_config(db_path)
    command.upgrade(cfg, "062")

    cols = _columns(db_path, "motif_extraction_jobs")
    expected = {
        "id",
        "status",
        "hmm_sequence_job_id",
        "source_kind",
        "min_ngram",
        "max_ngram",
        "minimum_occurrences",
        "minimum_event_sources",
        "frequency_weight",
        "event_source_weight",
        "event_core_weight",
        "low_background_weight",
        "call_probability_weight",
        "config_signature",
        "total_groups",
        "total_collapsed_tokens",
        "total_candidate_occurrences",
        "total_motifs",
        "artifact_dir",
        "error_message",
        "created_at",
        "updated_at",
    }
    assert expected.issubset(cols.keys())
    assert cols["id"]["pk"] is True
    assert cols["status"]["notnull"] is True
    assert cols["min_ngram"]["default"] is not None
    assert cols["call_probability_weight"]["notnull"] is False

    indexes = _indexes(db_path, "motif_extraction_jobs")
    assert "ix_motif_extraction_jobs_status" in indexes
    assert "ix_motif_extraction_jobs_hmm_sequence_job_id" in indexes
    assert "ix_motif_extraction_jobs_config_signature" in indexes


def test_downgrade_drops_motif_extraction_jobs(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    _stamp_pre_062(db_path)
    cfg = _alembic_config(db_path)

    command.upgrade(cfg, "062")
    assert _columns(db_path, "motif_extraction_jobs")

    command.downgrade(cfg, "061")
    assert _columns(db_path, "motif_extraction_jobs") == {}
