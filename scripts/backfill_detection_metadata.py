"""Backfill empty filename/start_sec/end_sec in training dataset parquet files.

Detection rows added after the row_id refactor (commit 3d677cc) were written
with filename="", start_sec=0.0, end_sec=0.0. This script looks up the actual
metadata from the detection row store and rewrites the parquet.

Usage:
    uv run scripts/backfill_detection_metadata.py              # dry run
    uv run scripts/backfill_detection_metadata.py --apply       # rewrite parquet
"""

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
from dotenv import load_dotenv

from humpback.config import Settings
from humpback.storage import detection_embeddings_path, detection_row_store_path

load_dotenv()


def _filename_from_utc(epoch: float) -> str:
    dt = datetime.fromtimestamp(epoch, tz=timezone.utc)
    return dt.strftime("%Y%m%dT%H%M%SZ.wav")


def get_db_path() -> Path:
    url = Settings().database_url
    raw = url.split("///", 1)[1]
    return Path(raw)


def main():
    parser = argparse.ArgumentParser(description="Backfill detection row metadata")
    parser.add_argument(
        "--apply", action="store_true", help="Rewrite parquet files (default: dry run)"
    )
    args = parser.parse_args()

    settings = Settings()
    db_path = get_db_path()
    print(f"Database: {db_path}")

    conn = sqlite3.connect(str(db_path))

    # Find all training datasets
    cur = conn.execute("SELECT id, parquet_path, source_config FROM training_datasets")
    datasets = cur.fetchall()

    total_fixed = 0
    total_skipped = 0

    for ds_id, parquet_path, source_config_json in datasets:
        config = json.loads(source_config_json)
        det_job_ids = config.get("detection_job_ids", [])
        if not det_job_ids:
            continue

        if not Path(parquet_path).exists():
            print(f"\n[{ds_id[:12]}...] parquet not found: {parquet_path}")
            continue

        # Build row_id -> (filename, start_sec, end_sec) from all detection row stores
        row_meta: dict[str, tuple[str, float, float]] = {}
        for djid in det_job_ids:
            rs_path = detection_row_store_path(Path(settings.storage_root), djid)
            if not rs_path.exists():
                print(f"  WARNING: no row store for detection job {djid[:12]}...")
                continue
            rs_table = pq.read_table(
                str(rs_path), columns=["row_id", "start_utc", "end_utc"]
            )
            for j in range(rs_table.num_rows):
                rid = rs_table.column("row_id")[j].as_py()
                s_utc = float(rs_table.column("start_utc")[j].as_py())
                e_utc = float(rs_table.column("end_utc")[j].as_py())
                row_meta[rid] = (_filename_from_utc(s_utc), 0.0, e_utc - s_utc)

        # Read parquet — need all columns to rewrite
        table = pq.read_table(parquet_path)
        source_types = table.column("source_type").to_pylist()
        source_ids = table.column("source_id").to_pylist()
        filenames = table.column("filename").to_pylist()
        start_secs = table.column("start_sec").to_pylist()
        end_secs = table.column("end_sec").to_pylist()

        # Find which detection rows need backfill (empty filename or zero duration)
        needs_fix = []
        for i in range(table.num_rows):
            if source_types[i] != "detection_job":
                continue
            if filenames[i] and (end_secs[i] - start_secs[i]) > 0:
                continue  # already populated
            needs_fix.append(i)

        if not needs_fix:
            continue

        # Match training dataset rows to detection embedding rows by embedding vector.
        # Build embedding-hash -> row_id index from each detection embedding parquet.
        import numpy as np

        fixed = 0
        skipped = 0
        new_filenames = list(filenames)
        new_start_secs = list(start_secs)
        new_end_secs = list(end_secs)

        emb_hash_to_rid: dict[str, dict[bytes, str]] = {}
        for djid in det_job_ids:
            emb_path = detection_embeddings_path(Path(settings.storage_root), djid)
            if not emb_path.exists():
                continue
            emb_table = pq.read_table(str(emb_path), columns=["row_id", "embedding"])
            rid_col = emb_table.column("row_id").to_pylist()
            emb_col = emb_table.column("embedding")
            index: dict[bytes, str] = {}
            for j in range(emb_table.num_rows):
                vec = np.array(emb_col[j].as_py(), dtype=np.float32)
                index[vec.tobytes()] = rid_col[j]
            emb_hash_to_rid[djid] = index

        # Read training dataset embeddings for the rows that need fixing
        td_emb_table = pq.read_table(parquet_path, columns=["embedding"])
        td_emb_col = td_emb_table.column("embedding")

        for i in needs_fix:
            sid = source_ids[i]
            index = emb_hash_to_rid.get(sid, {})
            vec = np.array(td_emb_col[i].as_py(), dtype=np.float32)
            rid = index.get(vec.tobytes())
            if rid and rid in row_meta:
                fname, s_sec, e_sec = row_meta[rid]
                new_filenames[i] = fname
                new_start_secs[i] = s_sec
                new_end_secs[i] = e_sec
                fixed += 1
            else:
                skipped += 1

        total_fixed += fixed
        total_skipped += skipped

        print(
            f"\n[{ds_id[:12]}...] {len(needs_fix)} rows need fix: "
            f"{fixed} fixable, {skipped} no metadata available"
        )

        if not args.apply or fixed == 0:
            # Show a few samples
            shown = 0
            for i in needs_fix:
                if new_filenames[i]:
                    print(
                        f'  row {i}: "" -> "{new_filenames[i]}"'
                        f"  0.0-0.0 -> {new_start_secs[i]}-{new_end_secs[i]}"
                    )
                    shown += 1
                    if shown >= 5:
                        break
            continue

        # Rewrite parquet with fixed metadata
        import pyarrow as pa

        new_table = table.set_column(
            table.schema.get_field_index("filename"),
            "filename",
            pa.array(new_filenames, type=pa.string()),
        )
        new_table = new_table.set_column(
            new_table.schema.get_field_index("start_sec"),
            "start_sec",
            pa.array(new_start_secs, type=pa.float32()),
        )
        new_table = new_table.set_column(
            new_table.schema.get_field_index("end_sec"),
            "end_sec",
            pa.array(new_end_secs, type=pa.float32()),
        )
        pq.write_table(new_table, parquet_path)
        print(f"  Rewrote {parquet_path}")

    conn.close()

    print(f"\nTotal: {total_fixed} rows fixed, {total_skipped} skipped")
    if not args.apply and total_fixed > 0:
        print("Dry run — no files changed. Use --apply to rewrite.")


if __name__ == "__main__":
    main()
