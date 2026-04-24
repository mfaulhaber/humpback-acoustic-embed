from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path

from humpback.call_parsing.storage import segmentation_job_dir, write_events
from humpback.call_parsing.types import Event


def _load_script_module():
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "recover_event_boundary_corrections.py"
    )
    spec = importlib.util.spec_from_file_location(
        "recover_event_boundary_corrections", script_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _create_legacy_backup_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.executescript(
            """
            CREATE TABLE event_segmentation_jobs (
                id TEXT PRIMARY KEY,
                region_detection_job_id TEXT NOT NULL
            );

            CREATE TABLE event_boundary_corrections (
                id TEXT PRIMARY KEY,
                event_segmentation_job_id TEXT NOT NULL,
                event_id TEXT NOT NULL,
                region_id TEXT NOT NULL,
                correction_type TEXT NOT NULL,
                start_sec FLOAT,
                end_sec FLOAT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        conn.execute(
            """
            INSERT INTO event_segmentation_jobs (id, region_detection_job_id)
            VALUES (?, ?)
            """,
            ("seg-1", "rd-1"),
        )
        rows = [
            (
                "legacy-add",
                "seg-1",
                "add-1",
                "region-1",
                "add",
                20.0,
                21.0,
                "2026-04-13 10:00:00",
                "2026-04-13 10:00:00",
            ),
            (
                "legacy-adjust",
                "seg-1",
                "event-1",
                "region-1",
                "adjust",
                10.5,
                11.5,
                "2026-04-13 10:01:00",
                "2026-04-13 10:01:00",
            ),
            (
                "legacy-delete",
                "seg-1",
                "event-2",
                "region-1",
                "delete",
                None,
                None,
                "2026-04-13 10:02:00",
                "2026-04-13 10:02:00",
            ),
            (
                "legacy-synth-adjust",
                "seg-1",
                "add-2",
                "region-2",
                "adjust",
                30.0,
                31.0,
                "2026-04-13 10:03:00",
                "2026-04-13 10:03:00",
            ),
            (
                "legacy-synth-delete",
                "seg-1",
                "add-3",
                "region-2",
                "delete",
                None,
                None,
                "2026-04-13 10:04:00",
                "2026-04-13 10:04:00",
            ),
        ]
        conn.executemany(
            """
            INSERT INTO event_boundary_corrections (
                id,
                event_segmentation_job_id,
                event_id,
                region_id,
                correction_type,
                start_sec,
                end_sec,
                created_at,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
    finally:
        conn.close()


def _create_target_db(path: Path, *, include_rd_job: bool = True) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.executescript(
            """
            CREATE TABLE region_detection_jobs (
                id TEXT PRIMARY KEY
            );

            CREATE TABLE event_boundary_corrections (
                id TEXT PRIMARY KEY,
                region_detection_job_id TEXT NOT NULL,
                region_id TEXT NOT NULL,
                correction_type TEXT NOT NULL,
                original_start_sec FLOAT,
                original_end_sec FLOAT,
                corrected_start_sec FLOAT,
                corrected_end_sec FLOAT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        if include_rd_job:
            conn.execute(
                "INSERT INTO region_detection_jobs (id) VALUES (?)",
                ("rd-1",),
            )
        conn.commit()
    finally:
        conn.close()


def _write_segmentation_events(storage_root: Path) -> None:
    seg_dir = segmentation_job_dir(storage_root, "seg-1")
    seg_dir.mkdir(parents=True, exist_ok=True)
    write_events(
        seg_dir / "events.parquet",
        [
            Event("event-1", "region-1", 10.0, 11.0, 10.5, 0.9),
            Event("event-2", "region-1", 12.0, 13.0, 12.5, 0.8),
        ],
    )


def test_build_recovery_plan_and_overlay_validation(tmp_path: Path) -> None:
    module = _load_script_module()
    backup_db = tmp_path / "backup.db"
    _create_legacy_backup_db(backup_db)
    _write_segmentation_events(tmp_path)

    plan = module.build_recovery_plan(backup_db, tmp_path)

    assert plan.scanned_rows == 5
    assert len(plan.recovered_rows) == 4
    assert len(plan.skipped_rows) == 1
    assert len(plan.unrecoverable_rows) == 0
    assert plan.strategy_counts == {
        "adjust_from_segmentation_event": 1,
        "delete_from_segmentation_event": 1,
        "direct_add": 1,
        "synthetic_adjust_as_add": 1,
    }

    recovered_by_legacy = {row.legacy_id: row for row in plan.recovered_rows}
    assert recovered_by_legacy["legacy-add"].correction_type == "add"
    assert recovered_by_legacy["legacy-add"].original_start_sec is None
    assert recovered_by_legacy["legacy-adjust"].original_start_sec == 10.0
    assert recovered_by_legacy["legacy-delete"].original_end_sec == 13.0
    assert recovered_by_legacy["legacy-synth-adjust"].correction_type == "add"

    overlay = module.validate_overlay_projection(backup_db, tmp_path, plan)
    assert overlay.success is True
    assert overlay.jobs_compared == 1
    assert overlay.jobs_with_missing_artifacts == 0
    assert overlay.matching_jobs == 1
    assert overlay.mismatched_jobs == []


def test_apply_recovery_and_verify_is_idempotent(tmp_path: Path) -> None:
    module = _load_script_module()
    backup_db = tmp_path / "backup.db"
    target_db = tmp_path / "target.db"
    _create_legacy_backup_db(backup_db)
    _create_target_db(target_db)
    _write_segmentation_events(tmp_path)

    plan = module.build_recovery_plan(backup_db, tmp_path)
    impact_before = module.estimate_target_impact(target_db, plan)
    assert impact_before.inserts == 4
    assert impact_before.updates == 0
    assert impact_before.unchanged == 0

    apply_result = module.apply_recovery(target_db, plan)
    assert apply_result.inserts == 4
    assert apply_result.updates == 0
    assert apply_result.unchanged == 0

    verification = module.verify_applied_rows(target_db, plan.recovered_rows)
    assert verification.success is True
    assert verification.matched_rows == 4
    assert verification.missing_rows == []

    impact_after = module.estimate_target_impact(target_db, plan)
    assert impact_after.inserts == 0
    assert impact_after.updates == 0
    assert impact_after.unchanged == 4


def test_validate_target_db_reports_missing_region_detection_jobs(
    tmp_path: Path,
) -> None:
    module = _load_script_module()
    backup_db = tmp_path / "backup.db"
    target_db = tmp_path / "target.db"
    _create_legacy_backup_db(backup_db)
    _create_target_db(target_db, include_rd_job=False)
    _write_segmentation_events(tmp_path)

    plan = module.build_recovery_plan(backup_db, tmp_path)
    validation = module.validate_target_db(target_db, plan)

    assert validation.success is False
    assert validation.missing_region_detection_job_ids == ["rd-1"]
