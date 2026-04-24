"""Unit tests for scripts/recover_vocalization_corrections.py."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import scripts.recover_vocalization_corrections as recovery_script
from humpback.call_parsing.storage import (
    classification_job_dir,
    segmentation_job_dir,
    write_events,
    write_typed_events,
)
from humpback.call_parsing.types import Event, TypedEvent


def _create_backup_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    try:
        with conn:
            conn.execute(
                """
                CREATE TABLE event_type_corrections (
                    id VARCHAR NOT NULL PRIMARY KEY,
                    event_classification_job_id VARCHAR,
                    event_id VARCHAR NOT NULL,
                    type_name VARCHAR,
                    created_at DATETIME,
                    updated_at DATETIME
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE event_classification_jobs (
                    id VARCHAR NOT NULL PRIMARY KEY,
                    event_segmentation_job_id VARCHAR
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE event_segmentation_jobs (
                    id VARCHAR NOT NULL PRIMARY KEY,
                    region_detection_job_id VARCHAR
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE event_boundary_corrections (
                    id VARCHAR NOT NULL PRIMARY KEY,
                    event_segmentation_job_id VARCHAR NOT NULL,
                    event_id VARCHAR NOT NULL,
                    region_id VARCHAR NOT NULL,
                    correction_type VARCHAR NOT NULL,
                    start_sec FLOAT,
                    end_sec FLOAT,
                    created_at DATETIME,
                    updated_at DATETIME
                )
                """
            )
    finally:
        conn.close()


def _create_target_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    try:
        with conn:
            conn.execute(
                """
                CREATE TABLE vocalization_corrections (
                    id VARCHAR NOT NULL PRIMARY KEY,
                    region_detection_job_id VARCHAR NOT NULL,
                    start_sec FLOAT NOT NULL,
                    end_sec FLOAT NOT NULL,
                    type_name VARCHAR NOT NULL,
                    correction_type VARCHAR NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP NOT NULL,
                    CONSTRAINT uq_vocalization_corrections_job_time_type
                        UNIQUE (region_detection_job_id, start_sec, end_sec, type_name)
                )
                """
            )
    finally:
        conn.close()


def _seed_backup_rows(path: Path) -> None:
    conn = sqlite3.connect(path)
    try:
        with conn:
            conn.executemany(
                """
                INSERT INTO event_classification_jobs (id, event_segmentation_job_id)
                VALUES (?, ?)
                """,
                [
                    ("ec-1", "es-1"),
                    ("ec-2", "es-2"),
                ],
            )
            conn.executemany(
                """
                INSERT INTO event_segmentation_jobs (id, region_detection_job_id)
                VALUES (?, ?)
                """,
                [
                    ("es-1", "rd-1"),
                    ("es-2", "rd-2"),
                ],
            )
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
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        "bc-add-1",
                        "es-1",
                        "add-1",
                        "region-added",
                        "add",
                        30.0,
                        31.0,
                        "2026-04-16 11:00:00",
                        "2026-04-16 11:00:00",
                    )
                ],
            )
            conn.executemany(
                """
                INSERT INTO event_type_corrections (
                    id,
                    event_classification_job_id,
                    event_id,
                    type_name,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        "legacy-1",
                        "ec-1",
                        "event-positive",
                        "Whup",
                        "2026-04-16 10:00:00",
                        "2026-04-16 10:00:00",
                    ),
                    (
                        "legacy-2",
                        "ec-1",
                        "event-negative",
                        None,
                        "2026-04-16 10:05:00",
                        "2026-04-16 10:05:00",
                    ),
                    (
                        "legacy-3",
                        "ec-1",
                        "add-1",
                        "Growl",
                        "2026-04-16 10:10:00",
                        "2026-04-16 10:10:00",
                    ),
                    (
                        "legacy-4",
                        "ec-1",
                        "event-positive",
                        "Whup",
                        "2026-04-16 10:15:00",
                        "2026-04-16 10:15:00",
                    ),
                    (
                        "legacy-orphan",
                        "missing-job",
                        "add-orphan",
                        None,
                        "2026-04-16 10:20:00",
                        "2026-04-16 10:20:00",
                    ),
                ],
            )
    finally:
        conn.close()


def _seed_artifacts(storage_root: Path) -> None:
    write_typed_events(
        classification_job_dir(storage_root, "ec-1") / "typed_events.parquet",
        [
            TypedEvent(
                event_id="event-positive",
                start_sec=10.0,
                end_sec=11.0,
                type_name="Whup",
                score=0.92,
                above_threshold=True,
            ),
            TypedEvent(
                event_id="event-positive",
                start_sec=10.0,
                end_sec=11.0,
                type_name="Buzz",
                score=0.12,
                above_threshold=False,
            ),
            TypedEvent(
                event_id="event-negative",
                start_sec=20.0,
                end_sec=21.0,
                type_name="Buzz",
                score=0.87,
                above_threshold=True,
            ),
            TypedEvent(
                event_id="event-negative",
                start_sec=20.0,
                end_sec=21.0,
                type_name="Whup",
                score=0.34,
                above_threshold=False,
            ),
        ],
    )
    write_events(
        segmentation_job_dir(storage_root, "es-1") / "events.parquet",
        [
            Event(
                event_id="event-positive",
                region_id="region-1",
                start_sec=10.0,
                end_sec=11.0,
                center_sec=10.5,
                segmentation_confidence=0.95,
            ),
            Event(
                event_id="event-negative",
                region_id="region-2",
                start_sec=20.0,
                end_sec=21.0,
                center_sec=20.5,
                segmentation_confidence=0.96,
            ),
        ],
    )


def test_parser_defaults_to_dry_run_mode() -> None:
    parser = recovery_script.build_parser()
    args = parser.parse_args(
        ["--backup-db", "/tmp/backup.db", "--target-db", "/tmp/target.db"]
    )
    assert args.apply is False
    assert args.dry_run is False
    assert args.show_unrecoverable == 10


def test_build_plan_apply_and_verify(tmp_path: Path) -> None:
    backup_db = tmp_path / "backup.db"
    target_db = tmp_path / "target.db"
    storage_root = tmp_path / "data"
    storage_root.mkdir()

    _create_backup_db(backup_db)
    _create_target_db(target_db)
    _seed_backup_rows(backup_db)
    _seed_artifacts(storage_root)

    plan = recovery_script.build_recovery_plan(backup_db, storage_root)

    assert plan.scanned_rows == 5
    assert len(plan.recovered_corrections) == 3
    assert len(plan.unrecoverable_rows) == 1
    assert plan.duplicate_rows == 1
    assert plan.conflicting_rows == 0

    recovered = {
        (
            row.region_detection_job_id,
            row.start_sec,
            row.end_sec,
            row.type_name,
            row.correction_type,
        )
        for row in plan.recovered_corrections
    }
    assert ("rd-1", 10.0, 11.0, "Whup", "add") in recovered
    assert ("rd-1", 20.0, 21.0, "Buzz", "remove") in recovered
    assert ("rd-1", 30.0, 31.0, "Growl", "add") in recovered

    impact = recovery_script.preview_target_impact(target_db, plan)
    assert impact.inserts == 3
    assert impact.updates == 0
    assert impact.unchanged == 0

    apply_result = recovery_script.apply_recovery_plan(target_db, plan)
    assert apply_result.inserts == 3
    assert apply_result.updates == 0
    assert apply_result.unchanged == 0

    verify = recovery_script.verify_recovery(target_db, plan)
    assert verify.success is True
    assert verify.expected_rows == 3
    assert verify.matched_rows == 3
    assert verify.missing_rows == []
    assert verify.mismatched_rows == []

    second_impact = recovery_script.preview_target_impact(target_db, plan)
    assert second_impact.inserts == 0
    assert second_impact.updates == 0
    assert second_impact.unchanged == 3
