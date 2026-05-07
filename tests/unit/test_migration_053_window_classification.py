"""Tests for migration 053 (window_classification_jobs + window_score_corrections).

Verifies that the upgrade creates both tables with expected columns
and that the downgrade drops them cleanly.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from alembic import command

from tests.helpers.migrations import (
    alembic_config,
    create_current_schema_db_sync,
    sqlite_column_names,
    sqlite_tables,
    stamp_revision_row,
)

import humpback.models.call_parsing  # noqa: F401


def _stamp_pre_053(db_path: Path) -> None:
    create_current_schema_db_sync(db_path)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("DROP TABLE IF EXISTS window_classification_jobs")
        conn.execute("DROP TABLE IF EXISTS window_score_corrections")
        conn.commit()
    finally:
        conn.close()
    stamp_revision_row(db_path, "052")


def test_upgrade_creates_both_tables(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    _stamp_pre_053(db_path)

    tables_pre = sqlite_tables(db_path)
    assert "window_classification_jobs" not in tables_pre
    assert "window_score_corrections" not in tables_pre

    cfg = alembic_config(db_path)
    command.upgrade(cfg, "053")

    tables_post = sqlite_tables(db_path)
    assert "window_classification_jobs" in tables_post
    assert "window_score_corrections" in tables_post

    job_cols = sqlite_column_names(db_path, "window_classification_jobs")
    assert {
        "id",
        "status",
        "region_detection_job_id",
        "vocalization_model_id",
        "config_json",
        "window_count",
        "vocabulary_snapshot",
        "error_message",
        "started_at",
        "completed_at",
        "created_at",
        "updated_at",
    } <= job_cols

    corr_cols = sqlite_column_names(db_path, "window_score_corrections")
    assert {
        "id",
        "window_classification_job_id",
        "time_sec",
        "region_id",
        "correction_type",
        "type_name",
        "created_at",
        "updated_at",
    } <= corr_cols


def test_downgrade_drops_both_tables(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    _stamp_pre_053(db_path)
    cfg = alembic_config(db_path)

    command.upgrade(cfg, "053")
    assert "window_classification_jobs" in sqlite_tables(db_path)

    command.downgrade(cfg, "052")
    tables = sqlite_tables(db_path)
    assert "window_classification_jobs" not in tables
    assert "window_score_corrections" not in tables
