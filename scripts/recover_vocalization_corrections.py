"""Recover unified vocalization corrections from a pre-#139 backup database.

This script reconstructs rows for the current ``vocalization_corrections``
table from legacy ``event_type_corrections`` data in an older backup DB.

Recovery sources, in priority order:

1. ``typed_events.parquet`` for the legacy classification job
2. matching legacy ``event_boundary_corrections`` rows for synthetic
   ``add-*`` events or adjusted bounds
3. ``events.parquet`` for the legacy segmentation job

By default the script runs in dry-run mode and prints a summary. Use
``--apply`` to write recovered rows into the target DB and verify them
afterward.

Usage::

    uv run python scripts/recover_vocalization_corrections.py \\
        --backup-db /path/to/humpback.db.20260419 \\
        --target-db /path/to/humpback.db \\
        --storage-root /path/to/data \\
        --dry-run

    uv run python scripts/recover_vocalization_corrections.py \\
        --backup-db /path/to/humpback.db.20260419 \\
        --target-db /path/to/humpback.db \\
        --storage-root /path/to/data \\
        --apply
"""

from __future__ import annotations

import argparse
import sqlite3
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from humpback.call_parsing.storage import (
    classification_job_dir,
    read_events,
    read_typed_events,
    segmentation_job_dir,
)
from humpback.call_parsing.types import Event, TypedEvent

RECOVERY_ID_NAMESPACE = uuid.UUID("f5db25cb-3ba8-43c3-b8e5-d7eb9b590315")


@dataclass(frozen=True)
class LegacyCorrectionRow:
    legacy_id: str
    event_classification_job_id: str | None
    event_segmentation_job_id: str | None
    region_detection_job_id: str | None
    event_id: str
    type_name: str | None
    created_at: str | None
    boundary_correction_type: str | None
    boundary_region_id: str | None
    boundary_start_sec: float | None
    boundary_end_sec: float | None


@dataclass(frozen=True)
class RecoveredCorrection:
    region_detection_job_id: str
    start_sec: float
    end_sec: float
    type_name: str
    correction_type: str
    source_event_id: str
    source_classification_job_id: str
    source_segmentation_job_id: str
    source_strategy: str
    legacy_created_at: str | None

    @property
    def key(self) -> tuple[str, float, float, str]:
        return (
            self.region_detection_job_id,
            self.start_sec,
            self.end_sec,
            self.type_name,
        )

    @property
    def stable_id(self) -> str:
        raw = (
            f"{self.region_detection_job_id}|{self.start_sec:.6f}|"
            f"{self.end_sec:.6f}|{self.type_name}"
        )
        return str(uuid.uuid5(RECOVERY_ID_NAMESPACE, raw))


@dataclass(frozen=True)
class UnrecoverableCorrection:
    legacy_id: str
    event_classification_job_id: str | None
    event_segmentation_job_id: str | None
    region_detection_job_id: str | None
    event_id: str
    type_name: str | None
    reason: str


@dataclass(frozen=True)
class RecoveryPlan:
    scanned_rows: int
    recovered_corrections: list[RecoveredCorrection]
    unrecoverable_rows: list[UnrecoverableCorrection]
    duplicate_rows: int
    conflicting_rows: int


@dataclass(frozen=True)
class TargetImpact:
    inserts: int
    updates: int
    unchanged: int


@dataclass(frozen=True)
class ApplyResult:
    inserts: int
    updates: int
    unchanged: int


@dataclass(frozen=True)
class VerificationResult:
    success: bool
    expected_rows: int
    matched_rows: int
    missing_rows: list[RecoveredCorrection]
    mismatched_rows: list[tuple[RecoveredCorrection, str]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recover vocalization_corrections from a legacy backup DB that "
            "still contains event_type_corrections."
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
        help="Path to the current SQLite DB that contains vocalization_corrections.",
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
        help="Preview recovered rows without modifying the target DB (default mode).",
    )
    mode.add_argument(
        "--apply",
        action="store_true",
        help="Write recovered rows into the target DB and verify them afterward.",
    )
    parser.add_argument(
        "--show-unrecoverable",
        type=int,
        default=10,
        help="Maximum number of unrecoverable sample rows to print (default: 10).",
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


def _load_legacy_rows(backup_db_path: Path) -> list[LegacyCorrectionRow]:
    _ensure_file_exists(backup_db_path, label="Backup DB")
    conn = _connect(backup_db_path)
    try:
        if not _table_exists(conn, "event_type_corrections"):
            raise ValueError(
                f"Backup DB does not contain event_type_corrections: {backup_db_path}"
            )

        boundary_join = (
            "LEFT JOIN event_boundary_corrections ebc "
            "ON ebc.event_segmentation_job_id = ec.event_segmentation_job_id "
            "AND ebc.event_id = etc.event_id"
        )
        if not _table_exists(conn, "event_boundary_corrections"):
            boundary_join = ""

        rows = conn.execute(
            f"""
            SELECT
                etc.id AS legacy_id,
                etc.event_classification_job_id,
                ec.event_segmentation_job_id,
                es.region_detection_job_id,
                etc.event_id,
                etc.type_name,
                etc.created_at,
                ebc.correction_type AS boundary_correction_type,
                ebc.region_id AS boundary_region_id,
                ebc.start_sec AS boundary_start_sec,
                ebc.end_sec AS boundary_end_sec
            FROM event_type_corrections etc
            LEFT JOIN event_classification_jobs ec
              ON ec.id = etc.event_classification_job_id
            LEFT JOIN event_segmentation_jobs es
              ON es.id = ec.event_segmentation_job_id
            {boundary_join}
            ORDER BY etc.created_at, etc.id
            """
        ).fetchall()
        return [
            LegacyCorrectionRow(
                legacy_id=row["legacy_id"],
                event_classification_job_id=row["event_classification_job_id"],
                event_segmentation_job_id=row["event_segmentation_job_id"],
                region_detection_job_id=row["region_detection_job_id"],
                event_id=row["event_id"],
                type_name=row["type_name"],
                created_at=row["created_at"],
                boundary_correction_type=row["boundary_correction_type"],
                boundary_region_id=row["boundary_region_id"],
                boundary_start_sec=row["boundary_start_sec"],
                boundary_end_sec=row["boundary_end_sec"],
            )
            for row in rows
        ]
    finally:
        conn.close()


def _load_typed_events_by_job(
    storage_root: Path, job_id: str
) -> dict[str, list[TypedEvent]] | None:
    path = classification_job_dir(storage_root, job_id) / "typed_events.parquet"
    if not path.exists():
        return None
    by_event: dict[str, list[TypedEvent]] = defaultdict(list)
    for row in read_typed_events(path):
        by_event[row.event_id].append(row)
    return dict(by_event)


def _load_segmentation_events_by_job(
    storage_root: Path, job_id: str
) -> dict[str, Event] | None:
    path = segmentation_job_dir(storage_root, job_id) / "events.parquet"
    if not path.exists():
        return None
    return {row.event_id: row for row in read_events(path)}


def _resolve_predicted_type(rows: list[TypedEvent]) -> str | None:
    above = [row for row in rows if row.above_threshold]
    if not above:
        return None
    indexed = list(enumerate(above))
    best = max(indexed, key=lambda item: (item[1].score, -item[0]))[1]
    return best.type_name


def _replacement_priority(correction_type: str) -> int:
    if correction_type == "remove":
        return 1
    return 0


def _resolve_recovered_correction(
    row: LegacyCorrectionRow,
    *,
    typed_by_job: dict[str, dict[str, list[TypedEvent]] | None],
    seg_by_job: dict[str, dict[str, Event] | None],
) -> RecoveredCorrection | UnrecoverableCorrection:
    if (
        row.event_classification_job_id is None
        or row.event_segmentation_job_id is None
        or row.region_detection_job_id is None
    ):
        return UnrecoverableCorrection(
            legacy_id=row.legacy_id,
            event_classification_job_id=row.event_classification_job_id,
            event_segmentation_job_id=row.event_segmentation_job_id,
            region_detection_job_id=row.region_detection_job_id,
            event_id=row.event_id,
            type_name=row.type_name,
            reason="missing job linkage in backup DB",
        )

    typed_by_event = typed_by_job.setdefault(
        row.event_classification_job_id,
        None,
    )
    seg_by_event = seg_by_job.setdefault(
        row.event_segmentation_job_id,
        None,
    )

    typed_rows = (
        typed_by_event.get(row.event_id)
        if typed_by_event is not None and row.event_id in typed_by_event
        else None
    )
    seg_event = (
        seg_by_event.get(row.event_id)
        if seg_by_event is not None and row.event_id in seg_by_event
        else None
    )

    start_sec: float | None = None
    end_sec: float | None = None
    source_strategy: str | None = None

    if typed_rows:
        start_sec = typed_rows[0].start_sec
        end_sec = typed_rows[0].end_sec
        source_strategy = "typed_events"
    elif (
        row.boundary_correction_type in {"add", "adjust"}
        and row.boundary_start_sec is not None
        and row.boundary_end_sec is not None
    ):
        start_sec = row.boundary_start_sec
        end_sec = row.boundary_end_sec
        source_strategy = f"boundary_{row.boundary_correction_type}"
    elif seg_event is not None:
        start_sec = seg_event.start_sec
        end_sec = seg_event.end_sec
        source_strategy = "segmentation_events"

    if start_sec is None or end_sec is None:
        return UnrecoverableCorrection(
            legacy_id=row.legacy_id,
            event_classification_job_id=row.event_classification_job_id,
            event_segmentation_job_id=row.event_segmentation_job_id,
            region_detection_job_id=row.region_detection_job_id,
            event_id=row.event_id,
            type_name=row.type_name,
            reason="could not resolve event bounds from typed events, boundary corrections, or segmentation events",
        )
    assert source_strategy is not None

    if row.type_name is not None:
        correction_type = "add"
        type_name = row.type_name
    else:
        predicted_type = _resolve_predicted_type(typed_rows or [])
        if predicted_type is None:
            return UnrecoverableCorrection(
                legacy_id=row.legacy_id,
                event_classification_job_id=row.event_classification_job_id,
                event_segmentation_job_id=row.event_segmentation_job_id,
                region_detection_job_id=row.region_detection_job_id,
                event_id=row.event_id,
                type_name=row.type_name,
                reason="legacy negative correction has no recoverable predicted type",
            )
        correction_type = "remove"
        type_name = predicted_type

    return RecoveredCorrection(
        region_detection_job_id=row.region_detection_job_id,
        start_sec=start_sec,
        end_sec=end_sec,
        type_name=type_name,
        correction_type=correction_type,
        source_event_id=row.event_id,
        source_classification_job_id=row.event_classification_job_id,
        source_segmentation_job_id=row.event_segmentation_job_id,
        source_strategy=source_strategy,
        legacy_created_at=row.created_at,
    )


def build_recovery_plan(backup_db_path: Path, storage_root: Path) -> RecoveryPlan:
    legacy_rows = _load_legacy_rows(backup_db_path)

    typed_by_job: dict[str, dict[str, list[TypedEvent]] | None] = {}
    seg_by_job: dict[str, dict[str, Event] | None] = {}

    for row in legacy_rows:
        if (
            row.event_classification_job_id is not None
            and row.event_classification_job_id not in typed_by_job
        ):
            typed_by_job[row.event_classification_job_id] = _load_typed_events_by_job(
                storage_root, row.event_classification_job_id
            )
        if (
            row.event_segmentation_job_id is not None
            and row.event_segmentation_job_id not in seg_by_job
        ):
            seg_by_job[row.event_segmentation_job_id] = (
                _load_segmentation_events_by_job(
                    storage_root, row.event_segmentation_job_id
                )
            )

    recovered_by_key: dict[tuple[str, float, float, str], RecoveredCorrection] = {}
    unrecoverable: list[UnrecoverableCorrection] = []
    duplicate_rows = 0
    conflicting_rows = 0

    for row in legacy_rows:
        resolved = _resolve_recovered_correction(
            row,
            typed_by_job=typed_by_job,
            seg_by_job=seg_by_job,
        )
        if isinstance(resolved, UnrecoverableCorrection):
            unrecoverable.append(resolved)
            continue

        existing = recovered_by_key.get(resolved.key)
        if existing is None:
            recovered_by_key[resolved.key] = resolved
            continue

        if existing.correction_type == resolved.correction_type:
            duplicate_rows += 1
            continue

        conflicting_rows += 1
        existing_created = existing.legacy_created_at or ""
        resolved_created = resolved.legacy_created_at or ""
        if resolved_created > existing_created or (
            resolved_created == existing_created
            and _replacement_priority(resolved.correction_type)
            > _replacement_priority(existing.correction_type)
        ):
            recovered_by_key[resolved.key] = resolved

    recovered = sorted(
        recovered_by_key.values(),
        key=lambda item: (
            item.region_detection_job_id,
            item.start_sec,
            item.end_sec,
            item.type_name,
        ),
    )
    return RecoveryPlan(
        scanned_rows=len(legacy_rows),
        recovered_corrections=recovered,
        unrecoverable_rows=unrecoverable,
        duplicate_rows=duplicate_rows,
        conflicting_rows=conflicting_rows,
    )


def _validate_target_db(target_db_path: Path) -> None:
    _ensure_file_exists(target_db_path, label="Target DB")
    conn = _connect(target_db_path)
    try:
        if not _table_exists(conn, "vocalization_corrections"):
            raise ValueError(
                f"Target DB does not contain vocalization_corrections: {target_db_path}"
            )
    finally:
        conn.close()


def _load_target_corrections(
    target_db_path: Path,
    region_detection_job_ids: set[str],
) -> dict[tuple[str, float, float, str], str]:
    if not region_detection_job_ids:
        return {}
    conn = _connect(target_db_path)
    try:
        placeholders = ",".join("?" for _ in sorted(region_detection_job_ids))
        rows = conn.execute(
            f"""
            SELECT region_detection_job_id, start_sec, end_sec, type_name, correction_type
            FROM vocalization_corrections
            WHERE region_detection_job_id IN ({placeholders})
            """,
            tuple(sorted(region_detection_job_ids)),
        ).fetchall()
        return {
            (
                row["region_detection_job_id"],
                float(row["start_sec"]),
                float(row["end_sec"]),
                row["type_name"],
            ): row["correction_type"]
            for row in rows
        }
    finally:
        conn.close()


def preview_target_impact(target_db_path: Path, plan: RecoveryPlan) -> TargetImpact:
    _validate_target_db(target_db_path)
    region_ids = {
        correction.region_detection_job_id for correction in plan.recovered_corrections
    }
    existing = _load_target_corrections(target_db_path, region_ids)
    inserts = 0
    updates = 0
    unchanged = 0
    for correction in plan.recovered_corrections:
        current = existing.get(correction.key)
        if current is None:
            inserts += 1
        elif current == correction.correction_type:
            unchanged += 1
        else:
            updates += 1
    return TargetImpact(inserts=inserts, updates=updates, unchanged=unchanged)


def apply_recovery_plan(target_db_path: Path, plan: RecoveryPlan) -> ApplyResult:
    _validate_target_db(target_db_path)
    impact = preview_target_impact(target_db_path, plan)

    conn = _connect(target_db_path)
    try:
        with conn:
            for correction in plan.recovered_corrections:
                timestamp = correction.legacy_created_at or "CURRENT_TIMESTAMP"
                if correction.legacy_created_at is None:
                    conn.execute(
                        """
                        INSERT INTO vocalization_corrections (
                            id,
                            region_detection_job_id,
                            start_sec,
                            end_sec,
                            type_name,
                            correction_type
                        )
                        VALUES (?, ?, ?, ?, ?, ?)
                        ON CONFLICT(region_detection_job_id, start_sec, end_sec, type_name)
                        DO UPDATE SET
                            correction_type = excluded.correction_type,
                            updated_at = CURRENT_TIMESTAMP
                        """,
                        (
                            correction.stable_id,
                            correction.region_detection_job_id,
                            correction.start_sec,
                            correction.end_sec,
                            correction.type_name,
                            correction.correction_type,
                        ),
                    )
                else:
                    conn.execute(
                        """
                        INSERT INTO vocalization_corrections (
                            id,
                            region_detection_job_id,
                            start_sec,
                            end_sec,
                            type_name,
                            correction_type,
                            created_at,
                            updated_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(region_detection_job_id, start_sec, end_sec, type_name)
                        DO UPDATE SET
                            correction_type = excluded.correction_type,
                            updated_at = excluded.updated_at
                        """,
                        (
                            correction.stable_id,
                            correction.region_detection_job_id,
                            correction.start_sec,
                            correction.end_sec,
                            correction.type_name,
                            correction.correction_type,
                            timestamp,
                            timestamp,
                        ),
                    )
        return ApplyResult(
            inserts=impact.inserts,
            updates=impact.updates,
            unchanged=impact.unchanged,
        )
    finally:
        conn.close()


def verify_recovery(target_db_path: Path, plan: RecoveryPlan) -> VerificationResult:
    _validate_target_db(target_db_path)
    region_ids = {
        correction.region_detection_job_id for correction in plan.recovered_corrections
    }
    actual = _load_target_corrections(target_db_path, region_ids)
    missing: list[RecoveredCorrection] = []
    mismatched: list[tuple[RecoveredCorrection, str]] = []
    matched = 0

    for correction in plan.recovered_corrections:
        current = actual.get(correction.key)
        if current is None:
            missing.append(correction)
        elif current != correction.correction_type:
            mismatched.append((correction, current))
        else:
            matched += 1

    return VerificationResult(
        success=not missing and not mismatched,
        expected_rows=len(plan.recovered_corrections),
        matched_rows=matched,
        missing_rows=missing,
        mismatched_rows=mismatched,
    )


def _print_plan_summary(
    plan: RecoveryPlan,
    impact: TargetImpact,
    *,
    show_unrecoverable: int,
) -> None:
    print("Recovery summary")
    print(f"  scanned legacy rows: {plan.scanned_rows}")
    print(f"  recoverable unique rows: {len(plan.recovered_corrections)}")
    print(f"  unrecoverable rows: {len(plan.unrecoverable_rows)}")
    print(f"  duplicate rows merged: {plan.duplicate_rows}")
    print(f"  conflicting rows resolved: {plan.conflicting_rows}")
    print(f"  target inserts: {impact.inserts}")
    print(f"  target updates: {impact.updates}")
    print(f"  target unchanged: {impact.unchanged}")

    if plan.unrecoverable_rows and show_unrecoverable > 0:
        print("\nSample unrecoverable rows")
        for row in plan.unrecoverable_rows[:show_unrecoverable]:
            print(
                "  "
                f"legacy_id={row.legacy_id} "
                f"event_id={row.event_id} "
                f"classification_job_id={row.event_classification_job_id} "
                f"segmentation_job_id={row.event_segmentation_job_id} "
                f"region_detection_job_id={row.region_detection_job_id} "
                f"type_name={row.type_name!r} "
                f"reason={row.reason}"
            )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    backup_db = args.backup_db.resolve()
    target_db = args.target_db.resolve()
    storage_root = (
        args.storage_root.resolve()
        if args.storage_root is not None
        else target_db.parent.resolve()
    )

    plan = build_recovery_plan(backup_db, storage_root)
    impact = preview_target_impact(target_db, plan)
    _print_plan_summary(plan, impact, show_unrecoverable=args.show_unrecoverable)

    if not args.apply:
        print("\nDry run only. No changes written.")
        return 0

    print("\nApplying recovery plan...")
    apply_result = apply_recovery_plan(target_db, plan)
    print(
        "Applied recovery plan: "
        f"inserts={apply_result.inserts} "
        f"updates={apply_result.updates} "
        f"unchanged={apply_result.unchanged}"
    )

    print("Verifying target DB...")
    verification = verify_recovery(target_db, plan)
    if verification.success:
        print(
            "Verification passed: "
            f"{verification.matched_rows}/{verification.expected_rows} expected rows present."
        )
        return 0

    print(
        "Verification failed: "
        f"matched={verification.matched_rows} "
        f"expected={verification.expected_rows} "
        f"missing={len(verification.missing_rows)} "
        f"mismatched={len(verification.mismatched_rows)}"
    )
    for row in verification.missing_rows[:10]:
        print(
            "  missing "
            f"region_detection_job_id={row.region_detection_job_id} "
            f"start_sec={row.start_sec} "
            f"end_sec={row.end_sec} "
            f"type_name={row.type_name} "
            f"correction_type={row.correction_type}"
        )
    for row, actual in verification.mismatched_rows[:10]:
        print(
            "  mismatched "
            f"region_detection_job_id={row.region_detection_job_id} "
            f"start_sec={row.start_sec} "
            f"end_sec={row.end_sec} "
            f"type_name={row.type_name} "
            f"expected={row.correction_type} actual={actual}"
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
