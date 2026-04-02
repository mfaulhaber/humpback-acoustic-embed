#!/usr/bin/env python3
"""Migrate detection job parquet files to stable row_id schema.

Processes all detection job directories under ``{storage_root}/detections/``:
  1. Row store parquets: assigns UUID to each row without one, rewrites atomically.
  2. Embedding parquets: matches entries from old ``(filename, start_sec, end_sec)``
     schema to row store via UTC tolerance matching, rewrites with
     ``(row_id, embedding, confidence)`` schema.
  3. Inference output parquets: maps ``(start_utc, end_utc)`` to ``row_id`` via
     row store lookup; unmatched predictions dropped.

The script is idempotent — running it multiple times produces the same result.
Rows that already have a ``row_id`` are preserved; embeddings and inference
outputs already in the new schema are skipped.

Usage::

    uv run python scripts/migrate_row_ids.py [--storage-root data]
"""

from __future__ import annotations

import argparse
import logging
import shutil
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from humpback.classifier.detection_rows import (
    parse_recording_timestamp,
    read_detection_row_store,
    write_detection_row_store,
)

logger = logging.getLogger(__name__)

_MATCH_TOLERANCE_SEC = 0.5


# ---------------------------------------------------------------------------
# Row store migration
# ---------------------------------------------------------------------------


def migrate_row_store(rs_path: Path) -> tuple[int, int, list[dict[str, str]]]:
    """Assign row_ids to rows without one. Returns (total, newly_assigned, rows)."""
    # Check the raw parquet to see if it already has a row_id column.
    raw_table = pq.read_table(str(rs_path))
    has_row_id_col = "row_id" in set(raw_table.column_names)

    # Count rows missing row_id in the raw data.
    rows_needing_ids = 0
    if has_row_id_col:
        for val in raw_table.column("row_id").to_pylist():
            if not val:
                rows_needing_ids += 1
    else:
        rows_needing_ids = raw_table.num_rows

    # read_detection_row_store normalises and calls ensure_row_ids internally.
    _fields, rows = read_detection_row_store(rs_path)

    if rows_needing_ids > 0:
        _atomic_write_row_store(rs_path, rows)

    return len(rows), rows_needing_ids, rows


def _atomic_write_row_store(path: Path, rows: list[dict[str, str]]) -> None:
    """Write row store atomically via temp file + rename."""
    tmp_fd, tmp_name = tempfile.mkstemp(suffix=".parquet", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        import os

        os.close(tmp_fd)
        write_detection_row_store(tmp_path, rows)
        shutil.move(str(tmp_path), str(path))
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


# ---------------------------------------------------------------------------
# Embedding migration
# ---------------------------------------------------------------------------


def _build_utc_to_row_id(
    rows: list[dict[str, str]],
) -> dict[tuple[float, float], str]:
    """Build (start_utc, end_utc) -> row_id lookup from row store rows."""
    mapping: dict[tuple[float, float], str] = {}
    for row in rows:
        start = float(row.get("start_utc") or "0")
        end = float(row.get("end_utc") or "0")
        rid = row.get("row_id", "")
        if rid:
            mapping[(start, end)] = rid
    return mapping


def _match_utc(
    target_start: float,
    target_end: float,
    utc_to_rid: dict[tuple[float, float], str],
) -> str | None:
    """Find matching row_id within tolerance."""
    for (rs, re), rid in utc_to_rid.items():
        if (
            abs(target_start - rs) <= _MATCH_TOLERANCE_SEC
            and abs(target_end - re) <= _MATCH_TOLERANCE_SEC
        ):
            return rid
    return None


def migrate_embeddings(
    emb_path: Path,
    rows: list[dict[str, str]],
) -> tuple[int, int, int]:
    """Migrate embedding parquet from old schema to row_id schema.

    Returns (total, matched, dropped).
    """
    if not emb_path.exists():
        return 0, 0, 0

    table = pq.read_table(str(emb_path))
    col_names = set(table.column_names)

    # Already migrated — has row_id and no filename column
    if "row_id" in col_names and "filename" not in col_names:
        return table.num_rows, table.num_rows, 0

    # New schema already present — skip
    if "row_id" in col_names:
        return table.num_rows, table.num_rows, 0

    # Old schema: (filename, start_sec, end_sec, embedding[, confidence])
    if "filename" not in col_names or "start_sec" not in col_names:
        logger.warning("Unexpected embedding schema at %s, skipping", emb_path)
        return table.num_rows, 0, table.num_rows

    utc_to_rid = _build_utc_to_row_id(rows)

    filenames = table.column("filename").to_pylist()
    start_secs = table.column("start_sec").to_pylist()
    end_secs = (
        table.column("end_sec").to_pylist()
        if "end_sec" in col_names
        else [float(s) + 5.0 for s in start_secs]
    )
    embeddings = table.column("embedding")
    confidences = (
        table.column("confidence").to_pylist()
        if "confidence" in col_names
        else [None] * table.num_rows
    )

    new_row_ids: list[str] = []
    new_embeddings: list = []
    new_confidences: list = []
    dropped = 0

    for i in range(table.num_rows):
        fname = filenames[i]
        s = float(start_secs[i])
        e = float(end_secs[i])

        # Compute UTC from filename timestamp
        ts = parse_recording_timestamp(fname)
        base_epoch = ts.timestamp() if ts else 0.0
        utc_start = base_epoch + s
        utc_end = base_epoch + e

        rid = _match_utc(utc_start, utc_end, utc_to_rid)
        if rid is None:
            dropped += 1
            continue

        new_row_ids.append(rid)
        new_embeddings.append(embeddings[i].as_py())
        new_confidences.append(confidences[i])

    matched = len(new_row_ids)

    if matched == 0:
        # Remove the file if nothing matched
        emb_path.unlink()
        return table.num_rows, 0, dropped

    # Determine embedding dim from first entry
    vec_dim = len(new_embeddings[0])
    schema = pa.schema(
        [
            ("row_id", pa.string()),
            ("embedding", pa.list_(pa.float32(), vec_dim)),
            ("confidence", pa.float32()),
        ]
    )
    new_table = pa.table(
        {
            "row_id": new_row_ids,
            "embedding": new_embeddings,
            "confidence": new_confidences,
        },
        schema=schema,
    )

    # Atomic write
    tmp_fd, tmp_name = tempfile.mkstemp(suffix=".parquet", dir=str(emb_path.parent))
    tmp_path = Path(tmp_name)
    try:
        import os

        os.close(tmp_fd)
        pq.write_table(new_table, str(tmp_path))
        shutil.move(str(tmp_path), str(emb_path))
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    return table.num_rows, matched, dropped


# ---------------------------------------------------------------------------
# Inference output migration
# ---------------------------------------------------------------------------


def migrate_inference_output(
    output_path: Path,
    rows: list[dict[str, str]],
) -> tuple[int, int, int]:
    """Migrate inference output from (start_utc, end_utc) to row_id keying.

    Returns (total, matched, dropped).
    """
    if not output_path.exists():
        return 0, 0, 0

    table = pq.read_table(str(output_path))
    col_names = set(table.column_names)

    # Already migrated
    if "row_id" in col_names and "start_utc" not in col_names:
        return table.num_rows, table.num_rows, 0

    # Has row_id already (hybrid or already done)
    if "row_id" in col_names:
        return table.num_rows, table.num_rows, 0

    # Old schema uses start_utc/end_utc or filename/start_sec/end_sec
    utc_to_rid = _build_utc_to_row_id(rows)

    matched_indices: list[int] = []
    matched_row_ids: list[str] = []

    if "start_utc" in col_names and "end_utc" in col_names:
        start_utcs = table.column("start_utc").to_pylist()
        end_utcs = table.column("end_utc").to_pylist()
        for i in range(table.num_rows):
            rid = _match_utc(float(start_utcs[i]), float(end_utcs[i]), utc_to_rid)
            if rid is not None:
                matched_indices.append(i)
                matched_row_ids.append(rid)
    elif "filename" in col_names and "start_sec" in col_names:
        filenames = table.column("filename").to_pylist()
        start_secs = table.column("start_sec").to_pylist()
        end_secs = (
            table.column("end_sec").to_pylist()
            if "end_sec" in col_names
            else [float(s) + 5.0 for s in start_secs]
        )
        for i in range(table.num_rows):
            ts = parse_recording_timestamp(filenames[i])
            base_epoch = ts.timestamp() if ts else 0.0
            utc_start = base_epoch + float(start_secs[i])
            utc_end = base_epoch + float(end_secs[i])
            rid = _match_utc(utc_start, utc_end, utc_to_rid)
            if rid is not None:
                matched_indices.append(i)
                matched_row_ids.append(rid)
    else:
        logger.warning(
            "Unexpected inference output schema at %s, skipping", output_path
        )
        return table.num_rows, 0, table.num_rows

    dropped = table.num_rows - len(matched_indices)

    if not matched_indices:
        return table.num_rows, 0, dropped

    # Build new table: row_id + confidence (if present) + score columns
    new_columns: dict[str, list] = {"row_id": matched_row_ids}

    if "confidence" in col_names:
        conf = table.column("confidence").to_pylist()
        new_columns["confidence"] = [conf[i] for i in matched_indices]

    # Copy score columns (vocalization type names)
    skip_cols = {
        "row_id",
        "filename",
        "start_sec",
        "end_sec",
        "start_utc",
        "end_utc",
        "confidence",
    }
    for name in table.column_names:
        if name not in skip_cols:
            vals = table.column(name).to_pylist()
            new_columns[name] = [vals[i] for i in matched_indices]

    new_table = pa.table(new_columns)

    # Atomic write
    tmp_fd, tmp_name = tempfile.mkstemp(suffix=".parquet", dir=str(output_path.parent))
    tmp_path = Path(tmp_name)
    try:
        import os

        os.close(tmp_fd)
        pq.write_table(new_table, str(tmp_path))
        shutil.move(str(tmp_path), str(output_path))
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    return table.num_rows, len(matched_indices), dropped


# ---------------------------------------------------------------------------
# Main migration
# ---------------------------------------------------------------------------


def find_inference_outputs(storage_root: Path) -> dict[str, list[Path]]:
    """Find inference output parquets and map detection_job_id -> output paths.

    Inference outputs are at ``vocalization_inference/*/predictions.parquet``.
    We need the database to know which ones target detection jobs, but as a
    pragmatic fallback we process all of them — non-detection-job outputs
    (embedding set sources) won't have matching UTC keys and will be skipped.
    """
    inf_dir = storage_root / "vocalization_inference"
    if not inf_dir.exists():
        return {}

    result: dict[str, list[Path]] = {}
    for job_dir in inf_dir.iterdir():
        if not job_dir.is_dir():
            continue
        pred_path = job_dir / "predictions.parquet"
        if pred_path.exists():
            # We don't know the detection_job_id from the filesystem alone,
            # so we collect all and try matching against each detection job's
            # row store. In practice, only the right one will match.
            result.setdefault("_all", []).append(pred_path)
    return result


def run_migration(storage_root: Path) -> dict[str, int]:
    """Run the full migration. Returns summary counters."""
    detections_dir = storage_root / "detections"
    if not detections_dir.exists():
        logger.info("No detections directory found at %s", detections_dir)
        return {"jobs_processed": 0}

    stats = {
        "jobs_processed": 0,
        "jobs_skipped_no_row_store": 0,
        "rows_total": 0,
        "rows_assigned_ids": 0,
        "embeddings_total": 0,
        "embeddings_matched": 0,
        "embeddings_dropped": 0,
        "inference_total": 0,
        "inference_matched": 0,
        "inference_dropped": 0,
    }

    # Collect all inference outputs for matching
    inf_dir = storage_root / "vocalization_inference"
    all_inference_outputs: list[Path] = []
    if inf_dir.exists():
        for job_dir in inf_dir.iterdir():
            if not job_dir.is_dir():
                continue
            pred_path = job_dir / "predictions.parquet"
            if pred_path.exists():
                all_inference_outputs.append(pred_path)

    for job_dir in sorted(detections_dir.iterdir()):
        if not job_dir.is_dir():
            continue

        job_id = job_dir.name
        rs_path = job_dir / "detection_rows.parquet"

        if not rs_path.exists():
            stats["jobs_skipped_no_row_store"] += 1
            continue

        stats["jobs_processed"] += 1

        # 1. Migrate row store
        total, assigned, rows = migrate_row_store(rs_path)
        stats["rows_total"] += total
        stats["rows_assigned_ids"] += assigned
        if assigned > 0:
            logger.info(
                "Job %s: assigned %d row_ids (%d total rows)", job_id, assigned, total
            )

        # 2. Migrate embeddings
        emb_path = job_dir / "detection_embeddings.parquet"
        if emb_path.exists():
            e_total, e_matched, e_dropped = migrate_embeddings(emb_path, rows)
            stats["embeddings_total"] += e_total
            stats["embeddings_matched"] += e_matched
            stats["embeddings_dropped"] += e_dropped
            if e_dropped > 0 or (e_matched > 0 and e_matched < e_total):
                logger.info(
                    "Job %s embeddings: %d matched, %d dropped",
                    job_id,
                    e_matched,
                    e_dropped,
                )

        # 3. Migrate inference outputs — try all outputs against this job's rows
        for inf_path in all_inference_outputs:
            i_total, i_matched, i_dropped = migrate_inference_output(inf_path, rows)
            if i_matched > 0:
                stats["inference_total"] += i_total
                stats["inference_matched"] += i_matched
                stats["inference_dropped"] += i_dropped
                logger.info(
                    "Job %s inference %s: %d matched, %d dropped",
                    job_id,
                    inf_path.parent.name,
                    i_matched,
                    i_dropped,
                )

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate detection job parquets to stable row_id schema."
    )
    parser.add_argument(
        "--storage-root",
        type=Path,
        default=Path("data"),
        help="Storage root directory (default: data)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable debug logging"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    storage_root = args.storage_root
    logger.info("Migrating detection jobs in %s", storage_root)

    stats = run_migration(storage_root)

    print("\n=== Migration Summary ===")
    print(f"Jobs processed:       {stats['jobs_processed']}")
    print(f"Jobs skipped (no RS):  {stats['jobs_skipped_no_row_store']}")
    print(f"Row store rows:       {stats['rows_total']}")
    print(f"  Newly assigned IDs: {stats['rows_assigned_ids']}")
    print(f"Embeddings total:     {stats['embeddings_total']}")
    print(f"  Matched:            {stats['embeddings_matched']}")
    print(f"  Dropped:            {stats['embeddings_dropped']}")
    print(f"Inference total:      {stats['inference_total']}")
    print(f"  Matched:            {stats['inference_matched']}")
    print(f"  Dropped:            {stats['inference_dropped']}")


if __name__ == "__main__":
    main()
