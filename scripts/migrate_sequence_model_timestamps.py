#!/usr/bin/env python3
"""Migrate Sequence Models artifacts to canonical epoch timestamp fields.

Dry run by default. Use ``--apply`` to rewrite parquet/JSON artifacts.

The script handles only Sequence Models artifacts:

- ``continuous_embeddings/<id>/embeddings.parquet``
- ``continuous_embeddings/<id>/manifest.json``
- ``hmm_sequences/<id>/states.parquet``
- ``hmm_sequences/<id>/pca_overlay.parquet``
- ``hmm_sequences/<id>/exemplars/exemplars.json``

Legacy ``start_time_sec`` / ``end_time_sec`` fields are replaced by
``start_timestamp`` / ``end_timestamp``. If legacy values are job-relative,
the source region job's ``start_timestamp`` is added. Already-epoch legacy
values are renamed without offsetting. Ambiguous values fail loudly.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv

from humpback.config import Settings

LEGACY_START = "start_time_sec"
LEGACY_END = "end_time_sec"
CANONICAL_START = "start_timestamp"
CANONICAL_END = "end_timestamp"
TOLERANCE_SEC = 1.0


@dataclass
class Summary:
    scanned: int = 0
    missing: int = 0
    canonical: int = 0
    would_migrate: int = 0
    migrated: int = 0
    failed: int = 0
    failures: list[str] = field(default_factory=list)

    def add_failure(self, path: Path, exc: Exception) -> None:
        self.failed += 1
        self.failures.append(f"{path}: {type(exc).__name__}: {exc}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Migrate Sequence Models artifact timestamps to epoch fields."
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="SQLite DB path. Defaults to HUMPBACK_DATABASE_URL / Settings().database_url.",
    )
    parser.add_argument(
        "--storage-root",
        type=Path,
        default=None,
        help="Storage root. Defaults to Settings().storage_root.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Rewrite artifacts. Default is dry-run only.",
    )
    return parser


def _sqlite_path_from_url(database_url: str) -> Path:
    prefix = "sqlite+aiosqlite:///"
    if not database_url.startswith(prefix):
        raise ValueError(f"Only sqlite+aiosqlite URLs are supported: {database_url}")
    raw = database_url[len(prefix) :]
    return Path(raw)


def _atomic_write_table(table: pa.Table, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    try:
        pq.write_table(table, tmp)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _atomic_write_json(payload: dict[str, Any], path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    try:
        tmp.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _classify_and_convert(
    values: list[float],
    *,
    job_start: float,
    job_end: float,
    path: Path,
    field_name: str,
) -> tuple[list[float], str]:
    if not values:
        return values, "empty"

    job_duration = max(0.0, job_end - job_start)
    min_value = min(values)
    max_value = max(values)

    is_epoch = (
        min_value >= job_start - TOLERANCE_SEC and max_value <= job_end + TOLERANCE_SEC
    )
    is_relative = (
        min_value >= -TOLERANCE_SEC
        and max_value <= job_duration + TOLERANCE_SEC
        and job_start > TOLERANCE_SEC
    )

    if is_epoch and not is_relative:
        return values, "already_epoch"
    if is_relative and not is_epoch:
        return [v + job_start for v in values], "relative_converted"
    if is_epoch:
        return values, "already_epoch"

    raise ValueError(
        f"Ambiguous {field_name} values in {path}: range {min_value}..{max_value}, "
        f"job range {job_start}..{job_end}"
    )


def _rewrite_time_columns(
    table: pa.Table,
    *,
    path: Path,
    job_start: float,
    job_end: float,
) -> tuple[pa.Table, str]:
    names = table.column_names
    has_canonical = CANONICAL_START in names and CANONICAL_END in names
    has_legacy = LEGACY_START in names and LEGACY_END in names

    if has_canonical and not has_legacy:
        return table, "already_canonical"
    if not has_legacy:
        raise ValueError(
            f"Missing legacy or canonical timestamp columns in {path}: {names}"
        )

    starts = [float(v) for v in table.column(LEGACY_START).to_pylist()]
    ends = [float(v) for v in table.column(LEGACY_END).to_pylist()]
    start_values, start_mode = _classify_and_convert(
        starts,
        job_start=job_start,
        job_end=job_end,
        path=path,
        field_name=LEGACY_START,
    )
    end_values, end_mode = _classify_and_convert(
        ends,
        job_start=job_start,
        job_end=job_end,
        path=path,
        field_name=LEGACY_END,
    )

    arrays: list[pa.Array] = []
    fields: list[pa.Field] = []
    for name in names:
        if name in {CANONICAL_START, CANONICAL_END}:
            continue
        if name == LEGACY_START:
            fields.append(pa.field(CANONICAL_START, pa.float64()))
            arrays.append(pa.array(start_values, type=pa.float64()))
        elif name == LEGACY_END:
            fields.append(pa.field(CANONICAL_END, pa.float64()))
            arrays.append(pa.array(end_values, type=pa.float64()))
        else:
            fields.append(table.schema.field(name))
            arrays.append(table.column(name))

    return pa.Table.from_arrays(arrays, schema=pa.schema(fields)), (
        f"{start_mode}/{end_mode}"
    )


def _rewrite_record_times(
    record: dict[str, Any],
    *,
    path: Path,
    job_start: float,
    job_end: float,
) -> bool:
    if CANONICAL_START in record and CANONICAL_END in record:
        record.pop(LEGACY_START, None)
        record.pop(LEGACY_END, None)
        return False
    if LEGACY_START not in record or LEGACY_END not in record:
        raise ValueError(f"Missing timestamp fields in JSON record at {path}: {record}")

    start_values, _ = _classify_and_convert(
        [float(record[LEGACY_START])],
        job_start=job_start,
        job_end=job_end,
        path=path,
        field_name=LEGACY_START,
    )
    end_values, _ = _classify_and_convert(
        [float(record[LEGACY_END])],
        job_start=job_start,
        job_end=job_end,
        path=path,
        field_name=LEGACY_END,
    )
    record[CANONICAL_START] = start_values[0]
    record[CANONICAL_END] = end_values[0]
    del record[LEGACY_START]
    del record[LEGACY_END]
    return True


def migrate_parquet(
    path: Path,
    *,
    job_start: float,
    job_end: float,
    apply: bool,
    summary: Summary,
) -> None:
    summary.scanned += 1
    if not path.exists():
        summary.missing += 1
        return

    try:
        table = pq.read_table(path)
        new_table, mode = _rewrite_time_columns(
            table,
            path=path,
            job_start=job_start,
            job_end=job_end,
        )
        if mode == "already_canonical":
            summary.canonical += 1
            return
        if apply:
            _atomic_write_table(new_table, path)
            summary.migrated += 1
        else:
            summary.would_migrate += 1
    except Exception as exc:
        summary.add_failure(path, exc)


def migrate_manifest(
    path: Path,
    *,
    job_start: float,
    job_end: float,
    apply: bool,
    summary: Summary,
) -> None:
    summary.scanned += 1
    if not path.exists():
        summary.missing += 1
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        changed = False
        for span in payload.get("spans", []):
            changed = (
                _rewrite_record_times(
                    span, path=path, job_start=job_start, job_end=job_end
                )
                or changed
            )
        if not changed:
            summary.canonical += 1
            return
        if apply:
            _atomic_write_json(payload, path)
            summary.migrated += 1
        else:
            summary.would_migrate += 1
    except Exception as exc:
        summary.add_failure(path, exc)


def migrate_exemplars(
    path: Path,
    *,
    job_start: float,
    job_end: float,
    apply: bool,
    summary: Summary,
) -> None:
    summary.scanned += 1
    if not path.exists():
        summary.missing += 1
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        changed = False
        for records in payload.get("states", {}).values():
            for record in records:
                changed = (
                    _rewrite_record_times(
                        record, path=path, job_start=job_start, job_end=job_end
                    )
                    or changed
                )
        if not changed:
            summary.canonical += 1
            return
        if apply:
            _atomic_write_json(payload, path)
            summary.migrated += 1
        else:
            summary.would_migrate += 1
    except Exception as exc:
        summary.add_failure(path, exc)


def _print_summary(name: str, summary: Summary) -> None:
    print(
        f"{name}: scanned={summary.scanned} missing={summary.missing} "
        f"canonical={summary.canonical} would_migrate={summary.would_migrate} "
        f"migrated={summary.migrated} failed={summary.failed}"
    )
    for failure in summary.failures:
        print(f"  FAILED {failure}")


def main() -> int:
    load_dotenv()
    args = build_parser().parse_args()
    settings = Settings()

    db_path = args.db or _sqlite_path_from_url(settings.database_url)
    storage_root = args.storage_root or Path(settings.storage_root)

    print(f"Mode: {'apply' if args.apply else 'dry-run'}")
    print(f"Database: {db_path}")
    print(f"Storage root: {storage_root}")

    summaries = {
        "cej_parquet": Summary(),
        "cej_manifest": Summary(),
        "hmm_states": Summary(),
        "hmm_overlay": Summary(),
        "hmm_exemplars": Summary(),
    }

    conn = sqlite3.connect(str(db_path))
    try:
        cej_rows = conn.execute(
            """
            SELECT ce.id, ce.parquet_path, rd.start_timestamp, rd.end_timestamp
            FROM continuous_embedding_jobs ce
            JOIN region_detection_jobs rd ON rd.id = ce.region_detection_job_id
            WHERE rd.start_timestamp IS NOT NULL
              AND rd.end_timestamp IS NOT NULL
            """
        ).fetchall()
        for cej_id, parquet_path, job_start, job_end in cej_rows:
            cej_dir = storage_root / "continuous_embeddings" / cej_id
            emb_path = (
                Path(parquet_path) if parquet_path else cej_dir / "embeddings.parquet"
            )
            migrate_parquet(
                emb_path,
                job_start=float(job_start),
                job_end=float(job_end),
                apply=args.apply,
                summary=summaries["cej_parquet"],
            )
            migrate_manifest(
                cej_dir / "manifest.json",
                job_start=float(job_start),
                job_end=float(job_end),
                apply=args.apply,
                summary=summaries["cej_manifest"],
            )

        hmm_rows = conn.execute(
            """
            SELECT hmm.id, hmm.artifact_dir, rd.start_timestamp, rd.end_timestamp
            FROM hmm_sequence_jobs hmm
            JOIN continuous_embedding_jobs ce ON ce.id = hmm.continuous_embedding_job_id
            JOIN region_detection_jobs rd ON rd.id = ce.region_detection_job_id
            WHERE rd.start_timestamp IS NOT NULL
              AND rd.end_timestamp IS NOT NULL
            """
        ).fetchall()
        for hmm_id, artifact_dir, job_start, job_end in hmm_rows:
            hmm_dir = (
                Path(artifact_dir)
                if artifact_dir
                else storage_root / "hmm_sequences" / hmm_id
            )
            migrate_parquet(
                hmm_dir / "states.parquet",
                job_start=float(job_start),
                job_end=float(job_end),
                apply=args.apply,
                summary=summaries["hmm_states"],
            )
            migrate_parquet(
                hmm_dir / "pca_overlay.parquet",
                job_start=float(job_start),
                job_end=float(job_end),
                apply=args.apply,
                summary=summaries["hmm_overlay"],
            )
            migrate_exemplars(
                hmm_dir / "exemplars" / "exemplars.json",
                job_start=float(job_start),
                job_end=float(job_end),
                apply=args.apply,
                summary=summaries["hmm_exemplars"],
            )
    finally:
        conn.close()

    print()
    for name, summary in summaries.items():
        _print_summary(name, summary)

    failed = sum(summary.failed for summary in summaries.values())
    if failed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
