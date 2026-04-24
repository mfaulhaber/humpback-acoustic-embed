"""Recover unified event boundary corrections from a pre-#140 backup database.

This script reconstructs rows for the current ``event_boundary_corrections``
table from legacy ``event_boundary_corrections`` data in an older backup DB.

The legacy table was keyed by ``event_segmentation_job_id`` + ``event_id``:

- ``add`` rows already contain the final event bounds
- ``adjust`` rows for original events need the original bounds recovered from
  ``events.parquet``
- ``delete`` rows for original events need the original bounds recovered from
  ``events.parquet``

Two legacy edge cases are handled conservatively:

- synthetic ``add-*`` rows with ``correction_type='adjust'`` are recovered as
  new-style ``add`` rows because the old schema did not retain the prior bounds
  of user-added events
- synthetic ``add-*`` rows with ``correction_type='delete'`` are skipped as
  net-no-op rows, because deleting a synthetic added event leaves no surviving
  correction in the unified schema

By default the script runs in dry-run mode and prints a recovery summary.
Use ``--apply`` to write recovered rows into the target DB and verify them
afterward.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from humpback.call_parsing.storage import read_events, segmentation_job_dir
from humpback.call_parsing.types import Event


@dataclass(frozen=True)
class LegacyBoundaryRow:
    legacy_id: str
    event_segmentation_job_id: str
    region_detection_job_id: str | None
    event_id: str
    region_id: str
    correction_type: str
    start_sec: float | None
    end_sec: float | None
    created_at: str | None
    updated_at: str | None


@dataclass(frozen=True)
class RecoveredBoundaryCorrection:
    legacy_id: str
    event_segmentation_job_id: str
    region_detection_job_id: str
    region_id: str
    correction_type: str
    original_start_sec: float | None
    original_end_sec: float | None
    corrected_start_sec: float | None
    corrected_end_sec: float | None
    recovery_strategy: str
    created_at: str | None
    updated_at: str | None

    @property
    def full_key(
        self,
    ) -> tuple[str, str, str, float | None, float | None, float | None, float | None]:
        return (
            self.region_detection_job_id,
            self.region_id,
            self.correction_type,
            self.original_start_sec,
            self.original_end_sec,
            self.corrected_start_sec,
            self.corrected_end_sec,
        )

    @property
    def identity_key(
        self,
    ) -> tuple[str, str, str, float | None, float | None]:
        if self.correction_type == "add":
            return (
                self.region_detection_job_id,
                self.region_id,
                "add",
                self.corrected_start_sec,
                self.corrected_end_sec,
            )
        return (
            self.region_detection_job_id,
            self.region_id,
            self.correction_type,
            self.original_start_sec,
            self.original_end_sec,
        )


@dataclass(frozen=True)
class SkippedLegacyRow:
    legacy_id: str
    event_segmentation_job_id: str
    region_detection_job_id: str | None
    event_id: str
    reason: str


@dataclass(frozen=True)
class UnrecoverableLegacyRow:
    legacy_id: str
    event_segmentation_job_id: str
    region_detection_job_id: str | None
    event_id: str
    reason: str


@dataclass(frozen=True)
class RecoveryPlan:
    scanned_rows: int
    recovered_rows: list[RecoveredBoundaryCorrection]
    skipped_rows: list[SkippedLegacyRow]
    unrecoverable_rows: list[UnrecoverableLegacyRow]
    strategy_counts: dict[str, int]
    recoverable_region_detection_job_ids: set[str]


@dataclass(frozen=True)
class TargetValidationSummary:
    success: bool
    missing_region_detection_job_ids: list[str]
    target_existing_rows: int


@dataclass(frozen=True)
class TargetImpact:
    inserts: int
    updates: int
    unchanged: int


@dataclass(frozen=True)
class OverlayJobValidation:
    event_segmentation_job_id: str
    legacy_row_count: int
    recovered_row_count: int
    success: bool
    legacy_event_count: int
    recovered_event_count: int
    missing_in_recovered: list[tuple[str, float, float]]
    extra_in_recovered: list[tuple[str, float, float]]
    reason: str | None = None


@dataclass(frozen=True)
class OverlayValidationSummary:
    success: bool
    jobs_compared: int
    jobs_with_missing_artifacts: int
    matching_jobs: int
    mismatched_jobs: list[OverlayJobValidation]


@dataclass(frozen=True)
class ApplyResult:
    inserts: int
    updates: int
    unchanged: int


@dataclass(frozen=True)
class VerificationResult:
    success: bool
    matched_rows: int
    missing_rows: list[RecoveredBoundaryCorrection]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recover unified event_boundary_corrections from a legacy "
            "pre-#140 backup database."
        )
    )
    parser.add_argument(
        "--backup-db",
        type=Path,
        required=True,
        help="Path to the legacy backup SQLite DB (for example humpback.db.20260419).",
    )
    parser.add_argument(
        "--target-db",
        type=Path,
        required=True,
        help="Path to the current SQLite DB that contains unified corrections.",
    )
    parser.add_argument(
        "--storage-root",
        type=Path,
        default=None,
        help=(
            "Root data directory that contains call_parsing artifacts. "
            "Defaults to the parent directory of --target-db."
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the recovery plan without modifying the target DB (default).",
    )
    mode.add_argument(
        "--apply",
        action="store_true",
        help="Write recovered rows into the target DB and verify them afterward.",
    )
    parser.add_argument(
        "--validate-target",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Check whether recovered region_detection_job_ids exist in the target DB.",
    )
    parser.add_argument(
        "--validate-overlay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Compare the projected recovered overlay against legacy overlay "
            "behavior job-by-job using events.parquet."
        ),
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Optional path to write the full recovery report as JSON.",
    )
    parser.add_argument(
        "--show-unrecoverable",
        type=int,
        default=10,
        help="Maximum number of unrecoverable sample rows to print (default: 10).",
    )
    parser.add_argument(
        "--show-mismatches",
        type=int,
        default=10,
        help="Maximum number of overlay-mismatch jobs to print (default: 10).",
    )
    return parser


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_file_exists(path: Path, *, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return row is not None


def _fetch_table_columns(conn: sqlite3.Connection, table_name: str) -> list[str]:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return [str(row["name"]) for row in rows]


def _validate_backup_db(backup_db_path: Path) -> None:
    _ensure_file_exists(backup_db_path, label="Backup DB")
    conn = _connect(backup_db_path)
    try:
        if not _table_exists(conn, "event_boundary_corrections"):
            raise ValueError(
                f"Backup DB does not contain event_boundary_corrections: {backup_db_path}"
            )
        expected = {
            "id",
            "event_segmentation_job_id",
            "event_id",
            "region_id",
            "correction_type",
            "start_sec",
            "end_sec",
            "created_at",
            "updated_at",
        }
        actual = set(_fetch_table_columns(conn, "event_boundary_corrections"))
        if expected != actual:
            raise ValueError(
                "Backup DB event_boundary_corrections schema does not match the "
                f"legacy shape. Expected {sorted(expected)}, got {sorted(actual)}."
            )
    finally:
        conn.close()


def _validate_target_db(target_db_path: Path) -> None:
    _ensure_file_exists(target_db_path, label="Target DB")
    conn = _connect(target_db_path)
    try:
        if not _table_exists(conn, "event_boundary_corrections"):
            raise ValueError(
                f"Target DB does not contain event_boundary_corrections: {target_db_path}"
            )
        expected = {
            "id",
            "region_detection_job_id",
            "region_id",
            "correction_type",
            "original_start_sec",
            "original_end_sec",
            "corrected_start_sec",
            "corrected_end_sec",
            "created_at",
            "updated_at",
        }
        actual = set(_fetch_table_columns(conn, "event_boundary_corrections"))
        if expected != actual:
            raise ValueError(
                "Target DB event_boundary_corrections schema does not match the "
                f"unified shape. Expected {sorted(expected)}, got {sorted(actual)}."
            )
    finally:
        conn.close()


def _load_legacy_rows(backup_db_path: Path) -> list[LegacyBoundaryRow]:
    _validate_backup_db(backup_db_path)
    conn = _connect(backup_db_path)
    try:
        rows = conn.execute(
            """
            SELECT
                ebc.id AS legacy_id,
                ebc.event_segmentation_job_id,
                es.region_detection_job_id,
                ebc.event_id,
                ebc.region_id,
                ebc.correction_type,
                ebc.start_sec,
                ebc.end_sec,
                ebc.created_at,
                ebc.updated_at
            FROM event_boundary_corrections ebc
            LEFT JOIN event_segmentation_jobs es
              ON es.id = ebc.event_segmentation_job_id
            ORDER BY ebc.created_at, ebc.id
            """
        ).fetchall()
        return [
            LegacyBoundaryRow(
                legacy_id=str(row["legacy_id"]),
                event_segmentation_job_id=str(row["event_segmentation_job_id"]),
                region_detection_job_id=(
                    str(row["region_detection_job_id"])
                    if row["region_detection_job_id"] is not None
                    else None
                ),
                event_id=str(row["event_id"]),
                region_id=str(row["region_id"]),
                correction_type=str(row["correction_type"]),
                start_sec=(
                    float(row["start_sec"]) if row["start_sec"] is not None else None
                ),
                end_sec=float(row["end_sec"]) if row["end_sec"] is not None else None,
                created_at=(
                    str(row["created_at"]) if row["created_at"] is not None else None
                ),
                updated_at=(
                    str(row["updated_at"]) if row["updated_at"] is not None else None
                ),
            )
            for row in rows
        ]
    finally:
        conn.close()


def _load_events_for_segmentation_job(
    storage_root: Path, event_segmentation_job_id: str
) -> dict[str, Event] | None:
    path = (
        segmentation_job_dir(storage_root, event_segmentation_job_id) / "events.parquet"
    )
    if not path.exists():
        return None
    return {event.event_id: event for event in read_events(path)}


def _resolve_legacy_row(
    row: LegacyBoundaryRow,
    *,
    events_by_job: dict[str, dict[str, Event] | None],
) -> RecoveredBoundaryCorrection | SkippedLegacyRow | UnrecoverableLegacyRow:
    if row.region_detection_job_id is None:
        return UnrecoverableLegacyRow(
            legacy_id=row.legacy_id,
            event_segmentation_job_id=row.event_segmentation_job_id,
            region_detection_job_id=row.region_detection_job_id,
            event_id=row.event_id,
            reason="missing region_detection_job_id mapping in backup DB",
        )

    if row.correction_type == "add":
        if row.start_sec is None or row.end_sec is None:
            return UnrecoverableLegacyRow(
                legacy_id=row.legacy_id,
                event_segmentation_job_id=row.event_segmentation_job_id,
                region_detection_job_id=row.region_detection_job_id,
                event_id=row.event_id,
                reason="legacy add row is missing start/end bounds",
            )
        return RecoveredBoundaryCorrection(
            legacy_id=row.legacy_id,
            event_segmentation_job_id=row.event_segmentation_job_id,
            region_detection_job_id=row.region_detection_job_id,
            region_id=row.region_id,
            correction_type="add",
            original_start_sec=None,
            original_end_sec=None,
            corrected_start_sec=row.start_sec,
            corrected_end_sec=row.end_sec,
            recovery_strategy="direct_add",
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

    if row.event_id.startswith("add-") and row.correction_type == "adjust":
        if row.start_sec is None or row.end_sec is None:
            return UnrecoverableLegacyRow(
                legacy_id=row.legacy_id,
                event_segmentation_job_id=row.event_segmentation_job_id,
                region_detection_job_id=row.region_detection_job_id,
                event_id=row.event_id,
                reason="synthetic legacy adjust row is missing corrected bounds",
            )
        return RecoveredBoundaryCorrection(
            legacy_id=row.legacy_id,
            event_segmentation_job_id=row.event_segmentation_job_id,
            region_detection_job_id=row.region_detection_job_id,
            region_id=row.region_id,
            correction_type="add",
            original_start_sec=None,
            original_end_sec=None,
            corrected_start_sec=row.start_sec,
            corrected_end_sec=row.end_sec,
            recovery_strategy="synthetic_adjust_as_add",
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

    if row.event_id.startswith("add-") and row.correction_type == "delete":
        return SkippedLegacyRow(
            legacy_id=row.legacy_id,
            event_segmentation_job_id=row.event_segmentation_job_id,
            region_detection_job_id=row.region_detection_job_id,
            event_id=row.event_id,
            reason="synthetic delete is a net no-op in the unified schema",
        )

    events_by_id = events_by_job.get(row.event_segmentation_job_id)
    if events_by_id is None:
        return UnrecoverableLegacyRow(
            legacy_id=row.legacy_id,
            event_segmentation_job_id=row.event_segmentation_job_id,
            region_detection_job_id=row.region_detection_job_id,
            event_id=row.event_id,
            reason="events.parquet not found for segmentation job",
        )

    original_event = events_by_id.get(row.event_id)
    if original_event is None:
        return UnrecoverableLegacyRow(
            legacy_id=row.legacy_id,
            event_segmentation_job_id=row.event_segmentation_job_id,
            region_detection_job_id=row.region_detection_job_id,
            event_id=row.event_id,
            reason="event_id not found in segmentation events",
        )

    if row.correction_type == "adjust":
        if row.start_sec is None or row.end_sec is None:
            return UnrecoverableLegacyRow(
                legacy_id=row.legacy_id,
                event_segmentation_job_id=row.event_segmentation_job_id,
                region_detection_job_id=row.region_detection_job_id,
                event_id=row.event_id,
                reason="legacy adjust row is missing corrected bounds",
            )
        return RecoveredBoundaryCorrection(
            legacy_id=row.legacy_id,
            event_segmentation_job_id=row.event_segmentation_job_id,
            region_detection_job_id=row.region_detection_job_id,
            region_id=row.region_id,
            correction_type="adjust",
            original_start_sec=original_event.start_sec,
            original_end_sec=original_event.end_sec,
            corrected_start_sec=row.start_sec,
            corrected_end_sec=row.end_sec,
            recovery_strategy="adjust_from_segmentation_event",
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

    if row.correction_type == "delete":
        return RecoveredBoundaryCorrection(
            legacy_id=row.legacy_id,
            event_segmentation_job_id=row.event_segmentation_job_id,
            region_detection_job_id=row.region_detection_job_id,
            region_id=row.region_id,
            correction_type="delete",
            original_start_sec=original_event.start_sec,
            original_end_sec=original_event.end_sec,
            corrected_start_sec=None,
            corrected_end_sec=None,
            recovery_strategy="delete_from_segmentation_event",
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

    return UnrecoverableLegacyRow(
        legacy_id=row.legacy_id,
        event_segmentation_job_id=row.event_segmentation_job_id,
        region_detection_job_id=row.region_detection_job_id,
        event_id=row.event_id,
        reason=f"unsupported correction_type {row.correction_type!r}",
    )


def build_recovery_plan(backup_db_path: Path, storage_root: Path) -> RecoveryPlan:
    legacy_rows = _load_legacy_rows(backup_db_path)
    events_by_job: dict[str, dict[str, Event] | None] = {}
    recovered: list[RecoveredBoundaryCorrection] = []
    skipped: list[SkippedLegacyRow] = []
    unrecoverable: list[UnrecoverableLegacyRow] = []

    for row in legacy_rows:
        if row.event_segmentation_job_id not in events_by_job:
            events_by_job[row.event_segmentation_job_id] = (
                _load_events_for_segmentation_job(
                    storage_root, row.event_segmentation_job_id
                )
            )

        resolved = _resolve_legacy_row(row, events_by_job=events_by_job)
        if isinstance(resolved, RecoveredBoundaryCorrection):
            recovered.append(resolved)
        elif isinstance(resolved, SkippedLegacyRow):
            skipped.append(resolved)
        else:
            unrecoverable.append(resolved)

    strategy_counts = Counter(row.recovery_strategy for row in recovered)
    region_detection_job_ids = {row.region_detection_job_id for row in recovered}
    return RecoveryPlan(
        scanned_rows=len(legacy_rows),
        recovered_rows=recovered,
        skipped_rows=skipped,
        unrecoverable_rows=unrecoverable,
        strategy_counts=dict(sorted(strategy_counts.items())),
        recoverable_region_detection_job_ids=region_detection_job_ids,
    )


def validate_target_db(
    target_db_path: Path, plan: RecoveryPlan
) -> TargetValidationSummary:
    _validate_target_db(target_db_path)
    conn = _connect(target_db_path)
    try:
        existing_rd_job_ids = {
            str(row["id"])
            for row in conn.execute("SELECT id FROM region_detection_jobs").fetchall()
        }
        missing_rd_job_ids = sorted(
            plan.recoverable_region_detection_job_ids - existing_rd_job_ids
        )
        existing_rows = int(
            conn.execute("SELECT COUNT(*) FROM event_boundary_corrections").fetchone()[
                0
            ]
        )
        return TargetValidationSummary(
            success=len(missing_rd_job_ids) == 0,
            missing_region_detection_job_ids=missing_rd_job_ids,
            target_existing_rows=existing_rows,
        )
    finally:
        conn.close()


def _legacy_effective_events(
    original_events: dict[str, Event],
    rows: list[LegacyBoundaryRow],
) -> list[tuple[str, float, float]]:
    events_by_id: dict[str, tuple[str, float, float]] = {
        event.event_id: (event.region_id, event.start_sec, event.end_sec)
        for event in original_events.values()
    }
    for row in rows:
        if row.correction_type == "delete":
            events_by_id.pop(row.event_id, None)
            continue
        if row.start_sec is None or row.end_sec is None:
            continue
        events_by_id[row.event_id] = (row.region_id, row.start_sec, row.end_sec)
    return sorted(events_by_id.values())


def _recovered_effective_events(
    original_events: dict[str, Event],
    rows: list[RecoveredBoundaryCorrection],
) -> list[tuple[str, float, float]]:
    events_by_region: dict[str, list[dict[str, float]]] = defaultdict(list)
    for event in original_events.values():
        events_by_region[event.region_id].append(
            {"start_sec": event.start_sec, "end_sec": event.end_sec}
        )

    for row in rows:
        region_events = events_by_region[row.region_id]
        if row.correction_type == "delete":
            region_events[:] = [
                event
                for event in region_events
                if not (
                    event["start_sec"] == row.original_start_sec
                    and event["end_sec"] == row.original_end_sec
                )
            ]
        elif row.correction_type == "adjust":
            if row.corrected_start_sec is None or row.corrected_end_sec is None:
                continue
            for event in region_events:
                if (
                    event["start_sec"] == row.original_start_sec
                    and event["end_sec"] == row.original_end_sec
                ):
                    event["start_sec"] = float(row.corrected_start_sec)
                    event["end_sec"] = float(row.corrected_end_sec)
                    break
        elif row.correction_type == "add":
            if row.corrected_start_sec is None or row.corrected_end_sec is None:
                continue
            region_events.append(
                {
                    "start_sec": float(row.corrected_start_sec),
                    "end_sec": float(row.corrected_end_sec),
                }
            )

    flattened: list[tuple[str, float, float]] = []
    for region_id, region_events in events_by_region.items():
        for event in region_events:
            flattened.append((region_id, event["start_sec"], event["end_sec"]))
    return sorted(flattened)


def validate_overlay_projection(
    backup_db_path: Path,
    storage_root: Path,
    plan: RecoveryPlan,
) -> OverlayValidationSummary:
    legacy_rows = _load_legacy_rows(backup_db_path)
    legacy_by_seg_job: dict[str, list[LegacyBoundaryRow]] = defaultdict(list)
    for row in legacy_rows:
        legacy_by_seg_job[row.event_segmentation_job_id].append(row)

    recovered_by_seg_job: dict[str, list[RecoveredBoundaryCorrection]] = defaultdict(
        list
    )
    for row in plan.recovered_rows:
        recovered_by_seg_job[row.event_segmentation_job_id].append(row)

    jobs_with_missing_artifacts = 0
    matching_jobs = 0
    mismatched_jobs: list[OverlayJobValidation] = []

    compared_job_ids = sorted(set(legacy_by_seg_job) | set(recovered_by_seg_job))
    for seg_job_id in compared_job_ids:
        original_events = _load_events_for_segmentation_job(storage_root, seg_job_id)
        if original_events is None:
            jobs_with_missing_artifacts += 1
            continue

        legacy_effective = _legacy_effective_events(
            original_events, legacy_by_seg_job.get(seg_job_id, [])
        )
        recovered_effective = _recovered_effective_events(
            original_events, recovered_by_seg_job.get(seg_job_id, [])
        )

        legacy_counter = Counter(legacy_effective)
        recovered_counter = Counter(recovered_effective)

        missing_rows: list[tuple[str, float, float]] = []
        extra_rows: list[tuple[str, float, float]] = []
        for item, count in (legacy_counter - recovered_counter).items():
            missing_rows.extend([item] * count)
        for item, count in (recovered_counter - legacy_counter).items():
            extra_rows.extend([item] * count)

        success = not missing_rows and not extra_rows
        if success:
            matching_jobs += 1
            continue

        mismatched_jobs.append(
            OverlayJobValidation(
                event_segmentation_job_id=seg_job_id,
                legacy_row_count=len(legacy_by_seg_job.get(seg_job_id, [])),
                recovered_row_count=len(recovered_by_seg_job.get(seg_job_id, [])),
                success=False,
                legacy_event_count=len(legacy_effective),
                recovered_event_count=len(recovered_effective),
                missing_in_recovered=missing_rows[:10],
                extra_in_recovered=extra_rows[:10],
            )
        )

    jobs_compared = len(compared_job_ids) - jobs_with_missing_artifacts
    return OverlayValidationSummary(
        success=len(mismatched_jobs) == 0,
        jobs_compared=jobs_compared,
        jobs_with_missing_artifacts=jobs_with_missing_artifacts,
        matching_jobs=matching_jobs,
        mismatched_jobs=mismatched_jobs,
    )


def _load_existing_target_rows(
    target_db_path: Path,
    region_detection_job_ids: set[str],
) -> dict[tuple[str, str, str, float | None, float | None], sqlite3.Row]:
    if not region_detection_job_ids:
        return {}
    conn = _connect(target_db_path)
    try:
        placeholders = ",".join("?" for _ in sorted(region_detection_job_ids))
        rows = conn.execute(
            f"""
            SELECT
                id,
                region_detection_job_id,
                region_id,
                correction_type,
                original_start_sec,
                original_end_sec,
                corrected_start_sec,
                corrected_end_sec
            FROM event_boundary_corrections
            WHERE region_detection_job_id IN ({placeholders})
            """,
            tuple(sorted(region_detection_job_ids)),
        ).fetchall()
        result: dict[tuple[str, str, str, float | None, float | None], sqlite3.Row] = {}
        for row in rows:
            if row["correction_type"] == "add":
                key = (
                    str(row["region_detection_job_id"]),
                    str(row["region_id"]),
                    "add",
                    (
                        float(row["corrected_start_sec"])
                        if row["corrected_start_sec"] is not None
                        else None
                    ),
                    (
                        float(row["corrected_end_sec"])
                        if row["corrected_end_sec"] is not None
                        else None
                    ),
                )
            else:
                key = (
                    str(row["region_detection_job_id"]),
                    str(row["region_id"]),
                    str(row["correction_type"]),
                    (
                        float(row["original_start_sec"])
                        if row["original_start_sec"] is not None
                        else None
                    ),
                    (
                        float(row["original_end_sec"])
                        if row["original_end_sec"] is not None
                        else None
                    ),
                )
            result[key] = row
        return result
    finally:
        conn.close()


def estimate_target_impact(target_db_path: Path, plan: RecoveryPlan) -> TargetImpact:
    existing = _load_existing_target_rows(
        target_db_path, plan.recoverable_region_detection_job_ids
    )
    inserts = 0
    updates = 0
    unchanged = 0
    for row in plan.recovered_rows:
        existing_row = existing.get(row.identity_key)
        if existing_row is None:
            inserts += 1
            continue

        existing_full_key = (
            str(existing_row["region_detection_job_id"]),
            str(existing_row["region_id"]),
            str(existing_row["correction_type"]),
            (
                float(existing_row["original_start_sec"])
                if existing_row["original_start_sec"] is not None
                else None
            ),
            (
                float(existing_row["original_end_sec"])
                if existing_row["original_end_sec"] is not None
                else None
            ),
            (
                float(existing_row["corrected_start_sec"])
                if existing_row["corrected_start_sec"] is not None
                else None
            ),
            (
                float(existing_row["corrected_end_sec"])
                if existing_row["corrected_end_sec"] is not None
                else None
            ),
        )
        if existing_full_key == row.full_key:
            unchanged += 1
        else:
            updates += 1

    return TargetImpact(inserts=inserts, updates=updates, unchanged=unchanged)


def apply_recovery(target_db_path: Path, plan: RecoveryPlan) -> ApplyResult:
    _validate_target_db(target_db_path)
    conn = _connect(target_db_path)
    inserts = 0
    updates = 0
    unchanged = 0
    try:
        conn.execute("BEGIN")
        for row in plan.recovered_rows:
            if row.correction_type == "add":
                existing = conn.execute(
                    """
                    SELECT id, correction_type, original_start_sec, original_end_sec,
                           corrected_start_sec, corrected_end_sec
                    FROM event_boundary_corrections
                    WHERE region_detection_job_id = ?
                      AND region_id = ?
                      AND corrected_start_sec = ?
                      AND corrected_end_sec = ?
                    """,
                    (
                        row.region_detection_job_id,
                        row.region_id,
                        row.corrected_start_sec,
                        row.corrected_end_sec,
                    ),
                ).fetchone()
            else:
                existing = conn.execute(
                    """
                    SELECT id, correction_type, original_start_sec, original_end_sec,
                           corrected_start_sec, corrected_end_sec
                    FROM event_boundary_corrections
                    WHERE region_detection_job_id = ?
                      AND region_id = ?
                      AND correction_type = ?
                      AND original_start_sec = ?
                      AND original_end_sec = ?
                    """,
                    (
                        row.region_detection_job_id,
                        row.region_id,
                        row.correction_type,
                        row.original_start_sec,
                        row.original_end_sec,
                    ),
                ).fetchone()

            if existing is None:
                conn.execute(
                    """
                    INSERT INTO event_boundary_corrections (
                        id,
                        region_detection_job_id,
                        region_id,
                        correction_type,
                        original_start_sec,
                        original_end_sec,
                        corrected_start_sec,
                        corrected_end_sec,
                        created_at,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row.legacy_id,
                        row.region_detection_job_id,
                        row.region_id,
                        row.correction_type,
                        row.original_start_sec,
                        row.original_end_sec,
                        row.corrected_start_sec,
                        row.corrected_end_sec,
                        row.created_at,
                        row.updated_at,
                    ),
                )
                inserts += 1
                continue

            existing_full_key = (
                row.region_detection_job_id,
                row.region_id,
                str(existing["correction_type"]),
                (
                    float(existing["original_start_sec"])
                    if existing["original_start_sec"] is not None
                    else None
                ),
                (
                    float(existing["original_end_sec"])
                    if existing["original_end_sec"] is not None
                    else None
                ),
                (
                    float(existing["corrected_start_sec"])
                    if existing["corrected_start_sec"] is not None
                    else None
                ),
                (
                    float(existing["corrected_end_sec"])
                    if existing["corrected_end_sec"] is not None
                    else None
                ),
            )
            if existing_full_key == row.full_key:
                unchanged += 1
                continue

            conn.execute(
                """
                UPDATE event_boundary_corrections
                SET correction_type = ?,
                    original_start_sec = ?,
                    original_end_sec = ?,
                    corrected_start_sec = ?,
                    corrected_end_sec = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    row.correction_type,
                    row.original_start_sec,
                    row.original_end_sec,
                    row.corrected_start_sec,
                    row.corrected_end_sec,
                    row.updated_at,
                    str(existing["id"]),
                ),
            )
            updates += 1

        conn.commit()
        return ApplyResult(inserts=inserts, updates=updates, unchanged=unchanged)
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def verify_applied_rows(
    target_db_path: Path, rows: list[RecoveredBoundaryCorrection]
) -> VerificationResult:
    conn = _connect(target_db_path)
    try:
        missing_rows: list[RecoveredBoundaryCorrection] = []
        matched_rows = 0
        for row in rows:
            matched = conn.execute(
                """
                SELECT 1
                FROM event_boundary_corrections
                WHERE region_detection_job_id = ?
                  AND region_id = ?
                  AND correction_type = ?
                  AND (
                    (original_start_sec = ?)
                    OR (original_start_sec IS NULL AND ? IS NULL)
                  )
                  AND (
                    (original_end_sec = ?)
                    OR (original_end_sec IS NULL AND ? IS NULL)
                  )
                  AND (
                    (corrected_start_sec = ?)
                    OR (corrected_start_sec IS NULL AND ? IS NULL)
                  )
                  AND (
                    (corrected_end_sec = ?)
                    OR (corrected_end_sec IS NULL AND ? IS NULL)
                  )
                """,
                (
                    row.region_detection_job_id,
                    row.region_id,
                    row.correction_type,
                    row.original_start_sec,
                    row.original_start_sec,
                    row.original_end_sec,
                    row.original_end_sec,
                    row.corrected_start_sec,
                    row.corrected_start_sec,
                    row.corrected_end_sec,
                    row.corrected_end_sec,
                ),
            ).fetchone()
            if matched is None:
                missing_rows.append(row)
            else:
                matched_rows += 1
        return VerificationResult(
            success=len(missing_rows) == 0,
            matched_rows=matched_rows,
            missing_rows=missing_rows,
        )
    finally:
        conn.close()


def _print_summary(
    plan: RecoveryPlan,
    *,
    target_validation: TargetValidationSummary | None,
    impact: TargetImpact | None,
    overlay_validation: OverlayValidationSummary | None,
    show_unrecoverable: int,
    show_mismatches: int,
) -> None:
    print("Recovery plan")
    print(f"  scanned legacy rows: {plan.scanned_rows}")
    print(f"  recoverable rows: {len(plan.recovered_rows)}")
    print(f"  skipped legacy rows: {len(plan.skipped_rows)}")
    print(f"  unrecoverable rows: {len(plan.unrecoverable_rows)}")
    if plan.strategy_counts:
        print("  recovery strategies:")
        for strategy, count in plan.strategy_counts.items():
            print(f"    {strategy}: {count}")

    if target_validation is not None:
        print("Target validation")
        print(f"  existing target rows: {target_validation.target_existing_rows}")
        print(
            "  missing region_detection_job_ids: "
            f"{len(target_validation.missing_region_detection_job_ids)}"
        )
        if target_validation.missing_region_detection_job_ids:
            for job_id in target_validation.missing_region_detection_job_ids[:10]:
                print(f"    {job_id}")

    if impact is not None:
        print("Target impact")
        print(f"  inserts: {impact.inserts}")
        print(f"  updates: {impact.updates}")
        print(f"  unchanged: {impact.unchanged}")

    if overlay_validation is not None:
        print("Overlay validation")
        print(f"  jobs compared: {overlay_validation.jobs_compared}")
        print(
            "  jobs with missing artifacts: "
            f"{overlay_validation.jobs_with_missing_artifacts}"
        )
        print(f"  matching jobs: {overlay_validation.matching_jobs}")
        print(f"  mismatched jobs: {len(overlay_validation.mismatched_jobs)}")
        for job in overlay_validation.mismatched_jobs[:show_mismatches]:
            print(f"    job {job.event_segmentation_job_id}")
            print(
                "      legacy/recovered effective events: "
                f"{job.legacy_event_count}/{job.recovered_event_count}"
            )
            if job.missing_in_recovered:
                print(f"      missing sample: {job.missing_in_recovered[0]}")
            if job.extra_in_recovered:
                print(f"      extra sample: {job.extra_in_recovered[0]}")

    if plan.unrecoverable_rows:
        print("Unrecoverable sample rows")
        for row in plan.unrecoverable_rows[:show_unrecoverable]:
            print(
                f"  legacy_id={row.legacy_id} seg_job={row.event_segmentation_job_id} "
                f"event_id={row.event_id} reason={row.reason}"
            )

    if plan.skipped_rows:
        print("Skipped sample rows")
        for row in plan.skipped_rows[: min(5, len(plan.skipped_rows))]:
            print(
                f"  legacy_id={row.legacy_id} seg_job={row.event_segmentation_job_id} "
                f"event_id={row.event_id} reason={row.reason}"
            )


def _report_dict(
    plan: RecoveryPlan,
    *,
    target_validation: TargetValidationSummary | None,
    impact: TargetImpact | None,
    overlay_validation: OverlayValidationSummary | None,
    apply_result: ApplyResult | None,
    verification: VerificationResult | None,
) -> dict[str, Any]:
    return {
        "plan": {
            "scanned_rows": plan.scanned_rows,
            "recovered_rows": len(plan.recovered_rows),
            "skipped_rows": len(plan.skipped_rows),
            "unrecoverable_rows": len(plan.unrecoverable_rows),
            "strategy_counts": plan.strategy_counts,
            "skipped_samples": [asdict(row) for row in plan.skipped_rows[:20]],
            "unrecoverable_samples": [
                asdict(row) for row in plan.unrecoverable_rows[:50]
            ],
        },
        "target_validation": (
            asdict(target_validation) if target_validation is not None else None
        ),
        "target_impact": asdict(impact) if impact is not None else None,
        "overlay_validation": (
            {
                "success": overlay_validation.success,
                "jobs_compared": overlay_validation.jobs_compared,
                "jobs_with_missing_artifacts": (
                    overlay_validation.jobs_with_missing_artifacts
                ),
                "matching_jobs": overlay_validation.matching_jobs,
                "mismatched_jobs": [
                    asdict(job) for job in overlay_validation.mismatched_jobs
                ],
            }
            if overlay_validation is not None
            else None
        ),
        "apply_result": asdict(apply_result) if apply_result is not None else None,
        "verification": asdict(verification) if verification is not None else None,
    }


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    backup_db = args.backup_db
    target_db = args.target_db
    storage_root = args.storage_root or target_db.parent

    if not args.apply:
        args.dry_run = True

    plan = build_recovery_plan(backup_db, storage_root)

    target_validation: TargetValidationSummary | None = None
    if args.validate_target:
        target_validation = validate_target_db(target_db, plan)

    impact = estimate_target_impact(target_db, plan)

    overlay_validation: OverlayValidationSummary | None = None
    if args.validate_overlay:
        overlay_validation = validate_overlay_projection(backup_db, storage_root, plan)

    _print_summary(
        plan,
        target_validation=target_validation,
        impact=impact,
        overlay_validation=overlay_validation,
        show_unrecoverable=args.show_unrecoverable,
        show_mismatches=args.show_mismatches,
    )

    apply_result: ApplyResult | None = None
    verification: VerificationResult | None = None

    if args.apply:
        if target_validation is not None and not target_validation.success:
            raise SystemExit(
                "Refusing to apply recovery because target validation failed."
            )
        apply_result = apply_recovery(target_db, plan)
        verification = verify_applied_rows(target_db, plan.recovered_rows)
        print("Apply result")
        print(f"  inserts: {apply_result.inserts}")
        print(f"  updates: {apply_result.updates}")
        print(f"  unchanged: {apply_result.unchanged}")
        print("Verification")
        print(f"  matched rows: {verification.matched_rows}")
        print(f"  missing rows: {len(verification.missing_rows)}")

    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(
            json.dumps(
                _report_dict(
                    plan,
                    target_validation=target_validation,
                    impact=impact,
                    overlay_validation=overlay_validation,
                    apply_result=apply_result,
                    verification=verification,
                ),
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        print(f"Wrote report: {args.report_json}")

    if target_validation is not None and not target_validation.success:
        return 2
    if overlay_validation is not None and not overlay_validation.success:
        return 3
    if verification is not None and not verification.success:
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
