"""Tests for migration 063 (masked_transformer_jobs)."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy.ext.asyncio import async_sessionmaker

from humpback.database import Base, create_engine
from humpback.models.processing import JobStatus
from humpback.models.sequence_models import (  # noqa: F401  (registers tables)
    MaskedTransformerJob,
)


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


def _tables(db_path: Path) -> set[str]:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    finally:
        conn.close()
    return {r[0] for r in rows}


def _columns(db_path: Path, table: str) -> set[str]:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    finally:
        conn.close()
    return {r[1] for r in rows}


def _stamp_pre_063(db_path: Path) -> None:
    asyncio.run(_create_db(db_path))
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("DROP TABLE IF EXISTS masked_transformer_jobs")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS alembic_version "
            "(version_num VARCHAR(32) NOT NULL)"
        )
        conn.execute("DELETE FROM alembic_version")
        conn.execute("INSERT INTO alembic_version (version_num) VALUES ('062')")
        conn.commit()
    finally:
        conn.close()


def test_upgrade_creates_table(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    _stamp_pre_063(db_path)
    assert "masked_transformer_jobs" not in _tables(db_path)

    cfg = _alembic_config(db_path)
    command.upgrade(cfg, "063")

    tables = _tables(db_path)
    assert "masked_transformer_jobs" in tables
    cols = _columns(db_path, "masked_transformer_jobs")
    assert {
        "id",
        "status",
        "status_reason",
        "continuous_embedding_job_id",
        "training_signature",
        "preset",
        "mask_fraction",
        "span_length_min",
        "span_length_max",
        "dropout",
        "mask_weight_bias",
        "cosine_loss_weight",
        "max_epochs",
        "early_stop_patience",
        "val_split",
        "seed",
        "k_values",
        "chosen_device",
        "fallback_reason",
        "final_train_loss",
        "final_val_loss",
        "total_epochs",
        "job_dir",
        "total_sequences",
        "total_chunks",
        "error_message",
        "created_at",
        "updated_at",
    } <= cols

    # training_signature is unique-indexed.
    conn = sqlite3.connect(db_path)
    try:
        idx_rows = conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='index' "
            "AND tbl_name='masked_transformer_jobs'"
        ).fetchall()
    finally:
        conn.close()
    sig_idx = [r for r in idx_rows if "training_signature" in (r[1] or "")]
    assert sig_idx, "training_signature index missing"
    assert any("UNIQUE" in (r[1] or "").upper() for r in sig_idx)


def test_downgrade_drops_table(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    _stamp_pre_063(db_path)
    cfg = _alembic_config(db_path)

    command.upgrade(cfg, "063")
    assert "masked_transformer_jobs" in _tables(db_path)

    command.downgrade(cfg, "062")
    assert "masked_transformer_jobs" not in _tables(db_path)


@pytest.mark.asyncio
async def test_round_trip_masked_transformer_job_row(tmp_path: Path) -> None:
    """Round-trip a row through the SQLAlchemy session."""
    db_path = tmp_path / "test.db"
    await _create_db(db_path)
    engine = create_engine(_db_url(db_path))
    Session = async_sessionmaker(engine, expire_on_commit=False)
    try:
        async with Session() as session:
            job = MaskedTransformerJob(
                continuous_embedding_job_id="cej-1",
                training_signature="sig-abc",
                preset="default",
                k_values="[100]",
                status=JobStatus.queued.value,
            )
            session.add(job)
            await session.commit()
            await session.refresh(job)

        async with Session() as session:
            fetched = await session.get(MaskedTransformerJob, job.id)
            assert fetched is not None
            assert fetched.training_signature == "sig-abc"
            assert fetched.preset == "default"
            assert fetched.mask_fraction == pytest.approx(0.20)
            assert fetched.span_length_min == 2
            assert fetched.span_length_max == 6
            assert fetched.mask_weight_bias is True
            assert fetched.k_values == "[100]"
            assert fetched.created_at is not None
            assert fetched.updated_at is not None
    finally:
        await engine.dispose()
