"""Remove duplicate training_dataset_labels rows.

For each (training_dataset_id, row_index, label) group with more than one row,
keeps the oldest record (earliest created_at) and deletes the rest.

Usage:
    uv run scripts/fix_duplicate_labels.py              # dry run (default)
    uv run scripts/fix_duplicate_labels.py --apply       # apply deletes
"""

import argparse
import sqlite3
from pathlib import Path

from dotenv import load_dotenv

from humpback.config import Settings

load_dotenv()


def get_db_path() -> Path:
    url = Settings().database_url
    # sqlite+aiosqlite:///path or sqlite:///path
    raw = url.split("///", 1)[1]
    return Path(raw)


def find_duplicates(conn: sqlite3.Connection) -> list[dict]:
    """Find duplicate (dataset_id, row_index, label) groups."""
    cur = conn.execute("""
        SELECT training_dataset_id, row_index, label, COUNT(*) as cnt
        FROM training_dataset_labels
        GROUP BY training_dataset_id, row_index, label
        HAVING cnt > 1
        ORDER BY training_dataset_id, row_index, label
    """)
    return [
        {
            "training_dataset_id": r[0],
            "row_index": r[1],
            "label": r[2],
            "count": r[3],
        }
        for r in cur.fetchall()
    ]


def get_ids_to_delete(conn: sqlite3.Connection, dup: dict) -> list[str]:
    """For a duplicate group, return IDs of all but the oldest record."""
    cur = conn.execute(
        """
        SELECT id FROM training_dataset_labels
        WHERE training_dataset_id = ? AND row_index = ? AND label = ?
        ORDER BY created_at ASC
        """,
        (dup["training_dataset_id"], dup["row_index"], dup["label"]),
    )
    ids = [r[0] for r in cur.fetchall()]
    # Keep the first (oldest), delete the rest
    return ids[1:]


def main():
    parser = argparse.ArgumentParser(
        description="Fix duplicate training dataset labels"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete duplicates (default: dry run)",
    )
    args = parser.parse_args()

    db_path = get_db_path()
    print(f"Database: {db_path}")

    if not db_path.exists():
        print("ERROR: database file not found")
        return

    conn = sqlite3.connect(str(db_path))
    duplicates = find_duplicates(conn)

    if not duplicates:
        print("No duplicate labels found.")
        conn.close()
        return

    total_excess = 0
    all_delete_ids: list[str] = []

    print(f"\nFound {len(duplicates)} duplicate group(s):\n")
    for dup in duplicates:
        delete_ids = get_ids_to_delete(conn, dup)
        all_delete_ids.extend(delete_ids)
        excess = dup["count"] - 1
        total_excess += excess
        print(
            f"  dataset={dup['training_dataset_id'][:8]}...  "
            f"row={dup['row_index']:4d}  "
            f"label={dup['label']!r:20s}  "
            f"copies={dup['count']}  "
            f"deleting={excess}"
        )

    print(f"\nTotal records to delete: {total_excess}")

    if not args.apply:
        print("\nDry run — no changes made. Use --apply to delete.")
        conn.close()
        return

    # Apply deletes
    for label_id in all_delete_ids:
        conn.execute("DELETE FROM training_dataset_labels WHERE id = ?", (label_id,))
    conn.commit()
    print(f"\nDeleted {len(all_delete_ids)} duplicate label record(s).")

    # Verify
    remaining = find_duplicates(conn)
    if remaining:
        print(f"WARNING: {len(remaining)} duplicate groups still remain!")
    else:
        print("Verified: no duplicates remain.")

    conn.close()


if __name__ == "__main__":
    main()
