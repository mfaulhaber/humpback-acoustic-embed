"""Tests for migration 042 and the call parsing models it backs.

The project's migration chain assumes ORM ``create_all`` has run first
(``model_configs`` is defined only via ORM, not Alembic), so we follow
the same pattern used by ``test_alembic_025_sanctsound_source_ids.py``:
build the DB via the ORM, then probe the parts of the migration we
actually care about.

Coverage:
- ORM round-trip on the five new call-parsing tables (including FK-like
  linkage via ``parent_run_id`` / upstream-pass ids).
- ``VocalizationClassifierModel`` / ``VocalizationTrainingJob`` default
  values for the new ``model_family`` / ``input_mode`` columns — the
  runtime equivalent of migration 042's backfill.
- Direct SQL exercise of migration 042's backfill statements against a
  bare fixture so the actual UPDATE logic is covered without running the
  whole Alembic chain.
- End-to-end downgrade → upgrade round-trip through Alembic itself so
  the migration's upgrade/downgrade pair is exercised against a live
  SQLite DB.
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

from alembic import command
from alembic.config import Config

from humpback.database import Base, create_engine, create_session_factory
from humpback.models.call_parsing import (
    CallParsingRun,
    EventClassificationJob,
    EventSegmentationJob,
    RegionDetectionJob,
    SegmentationModel,
)
from humpback.models.vocalization import (
    VocalizationClassifierModel,
    VocalizationTrainingJob,
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


def test_call_parsing_orm_roundtrip(tmp_path: Path) -> None:
    """Five new tables accept inserts and round-trip their FK linkage."""
    db_path = tmp_path / "test_042_orm.db"

    async def _run() -> None:
        await _create_db(db_path)
        engine = create_engine(_db_url(db_path))
        session_factory = create_session_factory(engine)
        try:
            async with session_factory() as session:
                run = CallParsingRun(
                    audio_file_id="audio-1",
                    status="queued",
                )
                session.add(run)
                await session.flush()

                region_job = RegionDetectionJob(
                    audio_file_id="audio-1",
                    parent_run_id=run.id,
                    status="queued",
                )
                session.add(region_job)
                await session.flush()

                seg_job = EventSegmentationJob(
                    region_detection_job_id=region_job.id,
                    parent_run_id=run.id,
                    status="queued",
                )
                session.add(seg_job)
                await session.flush()

                cls_job = EventClassificationJob(
                    event_segmentation_job_id=seg_job.id,
                    parent_run_id=run.id,
                    status="queued",
                )
                session.add(cls_job)
                await session.flush()

                seg_model = SegmentationModel(
                    name="toy-crnn",
                    model_family="pytorch_crnn",
                    model_path="/tmp/toy.pt",
                )
                session.add(seg_model)
                await session.commit()

                assert run.id
                assert region_job.parent_run_id == run.id
                assert seg_job.region_detection_job_id == region_job.id
                assert cls_job.event_segmentation_job_id == seg_job.id
                assert seg_model.id
        finally:
            await engine.dispose()

    asyncio.run(_run())


def test_vocalization_model_defaults_sklearn_family(tmp_path: Path) -> None:
    """New rows omit model_family / input_mode and fall to sklearn defaults.

    This is the runtime equivalent of migration 042's backfill — existing
    rows carry the same defaults as newly inserted ones.
    """
    db_path = tmp_path / "test_042_defaults.db"

    async def _run() -> None:
        await _create_db(db_path)
        engine = create_engine(_db_url(db_path))
        session_factory = create_session_factory(engine)
        try:
            async with session_factory() as session:
                voc_model = VocalizationClassifierModel(
                    name="legacy-model",
                    model_dir_path="/tmp/legacy",
                    vocabulary_snapshot='["whup"]',
                    per_class_thresholds='{"whup": 0.5}',
                )
                session.add(voc_model)

                voc_job = VocalizationTrainingJob(
                    status="complete",
                    source_config="{}",
                )
                session.add(voc_job)

                await session.commit()

                assert voc_model.model_family == "sklearn_perch_embedding"
                assert voc_model.input_mode == "detection_row"
                assert voc_job.model_family == "sklearn_perch_embedding"
                assert voc_job.input_mode == "detection_row"
        finally:
            await engine.dispose()

    asyncio.run(_run())


def test_migration_042_backfill_sql_updates_null_rows(tmp_path: Path) -> None:
    """Directly exercise the migration's backfill UPDATE statements.

    Constructs a minimal fixture table without the new columns, adds
    them as nullable (mimicking step 1 of the migration), inserts a row
    with NULL values, runs the migration's backfill UPDATE statements,
    and asserts the row is populated with the expected defaults.
    """
    db_path = tmp_path / "test_042_backfill.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE vocalization_models (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                model_family TEXT,
                input_mode TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO vocalization_models (id, name, model_family, input_mode) "
            "VALUES ('legacy-1', 'legacy', NULL, NULL)"
        )
        conn.commit()

        # These are the exact UPDATE statements emitted by migration 042's
        # upgrade(). Any drift in the migration's SQL must be reflected
        # here too — intentional coupling.
        conn.execute(
            "UPDATE vocalization_models "
            "SET model_family = 'sklearn_perch_embedding' "
            "WHERE model_family IS NULL"
        )
        conn.execute(
            "UPDATE vocalization_models "
            "SET input_mode = 'detection_row' "
            "WHERE input_mode IS NULL"
        )
        conn.commit()

        row = conn.execute(
            "SELECT model_family, input_mode FROM vocalization_models "
            "WHERE id='legacy-1'"
        ).fetchone()
        assert row == ("sklearn_perch_embedding", "detection_row")
    finally:
        conn.close()


def _alembic_config(db_path: Path) -> Config:
    repo_root = Path(__file__).resolve().parents[2]
    config = Config(str(repo_root / "alembic.ini"))
    config.set_main_option("script_location", str(repo_root / "alembic"))
    config.set_main_option("sqlalchemy.url", _db_url(db_path))
    return config


def _pre_042_schema(db_path: Path) -> None:
    """Build an ORM DB then strip it to pre-042 state for migration exercise.

    The ORM reflects post-042 schema, so we drop the new tables and the
    two new columns on ``vocalization_models`` / ``vocalization_training_jobs``
    to simulate a DB paused at revision 041.
    """
    asyncio.run(_create_db(db_path))
    conn = sqlite3.connect(db_path)
    try:
        for table in (
            "event_classification_jobs",
            "event_segmentation_jobs",
            "region_detection_jobs",
            "segmentation_models",
            "call_parsing_runs",
        ):
            conn.execute(f"DROP TABLE IF EXISTS {table}")
        for table in ("vocalization_models", "vocalization_training_jobs"):
            cols = [
                row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
            ]
            keep = [c for c in cols if c not in ("model_family", "input_mode")]
            col_list = ", ".join(keep)
            conn.execute(f"CREATE TABLE {table}__tmp AS SELECT {col_list} FROM {table}")
            conn.execute(f"DROP TABLE {table}")
            conn.execute(f"ALTER TABLE {table}__tmp RENAME TO {table}")
        conn.execute("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)")
        conn.execute("INSERT INTO alembic_version (version_num) VALUES ('041')")
        conn.commit()
    finally:
        conn.close()


def _inspect_schema(db_path: Path) -> dict[str, list[str]]:
    conn = sqlite3.connect(db_path)
    try:
        out: dict[str, list[str]] = {}
        tables = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' AND name != 'alembic_version'"
            ).fetchall()
        ]
        for t in tables:
            cols = [
                row[1] for row in conn.execute(f"PRAGMA table_info({t})").fetchall()
            ]
            out[t] = sorted(cols)
        return out
    finally:
        conn.close()


def test_migration_042_upgrade_creates_new_tables_and_columns(tmp_path: Path) -> None:
    """Alembic upgrade 042 adds the five tables and the two new columns."""
    db_path = tmp_path / "test_042_upgrade.db"
    _pre_042_schema(db_path)

    config = _alembic_config(db_path)
    command.upgrade(config, "042")

    schema = _inspect_schema(db_path)
    for expected in (
        "call_parsing_runs",
        "segmentation_models",
        "region_detection_jobs",
        "event_segmentation_jobs",
        "event_classification_jobs",
    ):
        assert expected in schema, f"missing table {expected}"

    for t in ("vocalization_models", "vocalization_training_jobs"):
        assert "model_family" in schema[t]
        assert "input_mode" in schema[t]

    conn = sqlite3.connect(db_path)
    try:
        version = conn.execute("SELECT version_num FROM alembic_version").fetchone()
    finally:
        conn.close()
    assert version == ("042",)


def test_migration_042_backfill_populates_existing_rows(tmp_path: Path) -> None:
    """A row that predates the new columns is backfilled to the sklearn defaults."""
    db_path = tmp_path / "test_042_backfill_alembic.db"
    _pre_042_schema(db_path)

    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "INSERT INTO vocalization_models "
            "(id, name, model_dir_path, vocabulary_snapshot, per_class_thresholds, "
            " is_active, created_at, updated_at) "
            "VALUES ('legacy-row', 'legacy', '/tmp/legacy', '[\"whup\"]', "
            " '{\"whup\": 0.5}', 0, '2026-01-01', '2026-01-01')"
        )
        conn.commit()
    finally:
        conn.close()

    command.upgrade(_alembic_config(db_path), "042")

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT model_family, input_mode FROM vocalization_models "
            "WHERE id='legacy-row'"
        ).fetchone()
    finally:
        conn.close()
    assert row == ("sklearn_perch_embedding", "detection_row")


def test_migration_042_downgrade_roundtrip(tmp_path: Path) -> None:
    """upgrade → downgrade → upgrade produces the same schema."""
    db_path = tmp_path / "test_042_roundtrip.db"
    _pre_042_schema(db_path)

    config = _alembic_config(db_path)
    command.upgrade(config, "042")
    after_first_upgrade = _inspect_schema(db_path)

    command.downgrade(config, "041")
    after_downgrade = _inspect_schema(db_path)
    for dropped in (
        "call_parsing_runs",
        "segmentation_models",
        "region_detection_jobs",
        "event_segmentation_jobs",
        "event_classification_jobs",
    ):
        assert dropped not in after_downgrade, f"downgrade left {dropped}"
    for t in ("vocalization_models", "vocalization_training_jobs"):
        assert "model_family" not in after_downgrade[t]
        assert "input_mode" not in after_downgrade[t]

    command.upgrade(config, "042")
    after_second_upgrade = _inspect_schema(db_path)
    assert after_first_upgrade == after_second_upgrade
