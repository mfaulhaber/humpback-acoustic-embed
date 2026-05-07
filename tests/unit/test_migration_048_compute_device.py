"""Tests for migration 048 (compute_device + gpu_fallback_reason).

Verifies that the upgrade adds the two columns to both
``event_segmentation_jobs`` and ``event_classification_jobs``, and that
the downgrade drops them cleanly. Roundtrip ensures column metadata is
stable across multiple upgrade/downgrade cycles.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from alembic import command

from tests.helpers.migrations import (
    alembic_config,
    create_current_schema_db_sync,
    sqlite_column_names,
    stamp_revision_row,
)

# Importing the call_parsing models registers their tables with
# ``Base.metadata`` so ``create_all`` builds them. Without this the
# event-segmentation/classification tables are missing from the test DB.
import humpback.models.call_parsing  # noqa: F401


def _stamp_pre_048(db_path: Path) -> None:
    """Build a DB at revision 047 (no compute_device columns yet).

    ``Base.metadata.create_all`` builds the *current* schema, which
    already includes the new columns. Drop them so the smoke test
    starts from a true pre-048 state.
    """
    create_current_schema_db_sync(db_path)
    conn = sqlite3.connect(db_path)
    try:
        for table in ("event_segmentation_jobs", "event_classification_jobs"):
            for col in ("compute_device", "gpu_fallback_reason"):
                conn.execute(f"ALTER TABLE {table} DROP COLUMN {col}")
        conn.commit()
    finally:
        conn.close()
    stamp_revision_row(db_path, "047")


def test_upgrade_adds_columns_to_both_tables(tmp_path):
    db_path = tmp_path / "test.db"
    _stamp_pre_048(db_path)

    seg_pre = sqlite_column_names(db_path, "event_segmentation_jobs")
    cls_pre = sqlite_column_names(db_path, "event_classification_jobs")
    assert "compute_device" not in seg_pre
    assert "gpu_fallback_reason" not in seg_pre
    assert "compute_device" not in cls_pre
    assert "gpu_fallback_reason" not in cls_pre

    cfg = alembic_config(db_path)
    command.upgrade(cfg, "048")

    seg_post = sqlite_column_names(db_path, "event_segmentation_jobs")
    cls_post = sqlite_column_names(db_path, "event_classification_jobs")
    assert "compute_device" in seg_post
    assert "gpu_fallback_reason" in seg_post
    assert "compute_device" in cls_post
    assert "gpu_fallback_reason" in cls_post


def test_downgrade_removes_columns_from_both_tables(tmp_path):
    db_path = tmp_path / "test.db"
    _stamp_pre_048(db_path)
    cfg = alembic_config(db_path)

    command.upgrade(cfg, "048")
    command.downgrade(cfg, "047")

    seg_cols = sqlite_column_names(db_path, "event_segmentation_jobs")
    cls_cols = sqlite_column_names(db_path, "event_classification_jobs")
    assert "compute_device" not in seg_cols
    assert "gpu_fallback_reason" not in seg_cols
    assert "compute_device" not in cls_cols
    assert "gpu_fallback_reason" not in cls_cols


def test_upgrade_downgrade_roundtrip(tmp_path):
    db_path = tmp_path / "test.db"
    _stamp_pre_048(db_path)
    cfg = alembic_config(db_path)

    command.upgrade(cfg, "048")
    seg_after_first = sqlite_column_names(db_path, "event_segmentation_jobs")
    cls_after_first = sqlite_column_names(db_path, "event_classification_jobs")

    command.downgrade(cfg, "047")
    command.upgrade(cfg, "048")

    assert sqlite_column_names(db_path, "event_segmentation_jobs") == seg_after_first
    assert sqlite_column_names(db_path, "event_classification_jobs") == cls_after_first
