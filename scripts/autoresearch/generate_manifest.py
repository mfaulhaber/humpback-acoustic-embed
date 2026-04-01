"""Generate a data manifest from humpback platform classifier training jobs.

Queries the database for embedding set metadata and builds a stable
train/val/test split grouped by audio file.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from sqlalchemy import create_engine, text

from humpback.config import Settings


def _get_sync_db_url(settings: Settings) -> str:
    """Convert async database URL to sync for manifest generation."""
    url = settings.database_url
    return url.replace("sqlite+aiosqlite:", "sqlite:")


def _query_job_embedding_sets(
    db_url: str, job_ids: list[int | str]
) -> tuple[list[str], list[str]]:
    """Query positive and negative embedding set IDs from training jobs."""
    engine = create_engine(db_url)
    positive_ids: list[str] = []
    negative_ids: list[str] = []

    with engine.connect() as conn:
        for job_id in job_ids:
            row = conn.execute(
                text(
                    "SELECT positive_embedding_set_ids, negative_embedding_set_ids "
                    "FROM classifier_training_jobs WHERE id = :id"
                ),
                {"id": str(job_id)},
            ).fetchone()
            if row is None:
                msg = f"Training job {job_id} not found"
                raise ValueError(msg)
            positive_ids.extend(json.loads(row[0]))
            negative_ids.extend(json.loads(row[1]))

    # Deduplicate while preserving order
    positive_ids = list(dict.fromkeys(positive_ids))
    negative_ids = list(dict.fromkeys(negative_ids))
    engine.dispose()
    return positive_ids, negative_ids


def _query_embedding_sets(db_url: str, es_ids: list[str]) -> list[dict[str, Any]]:
    """Query embedding set metadata."""
    engine = create_engine(db_url)
    results = []

    with engine.connect() as conn:
        for es_id in es_ids:
            row = conn.execute(
                text(
                    "SELECT id, parquet_path, audio_file_id "
                    "FROM embedding_sets WHERE id = :id"
                ),
                {"id": es_id},
            ).fetchone()
            if row is not None:
                results.append(
                    {
                        "id": row[0],
                        "parquet_path": row[1],
                        "audio_file_id": row[2],
                    }
                )

    engine.dispose()
    return results


def _count_parquet_rows(parquet_path: str) -> int:
    """Count rows in a Parquet file without loading vectors."""
    metadata = pq.read_metadata(parquet_path)
    return metadata.num_rows


def _assign_splits(
    audio_file_ids: list[str],
    split_ratio: tuple[int, int, int],
    seed: int,
) -> dict[str, str]:
    """Assign audio files to splits. Returns {audio_file_id: split_name}."""
    unique_ids = sorted(set(audio_file_ids))
    rng = random.Random(seed)
    rng.shuffle(unique_ids)

    total = sum(split_ratio)
    n = len(unique_ids)
    train_end = round(n * split_ratio[0] / total)
    val_end = round(n * (split_ratio[0] + split_ratio[1]) / total)

    assignments: dict[str, str] = {}
    for i, fid in enumerate(unique_ids):
        if i < train_end:
            assignments[fid] = "train"
        elif i < val_end:
            assignments[fid] = "val"
        else:
            assignments[fid] = "test"

    return assignments


def generate_manifest(
    job_ids: list[int | str],
    split_ratio: tuple[int, int, int],
    seed: int,
    db_url: str | None = None,
) -> dict[str, Any]:
    """Generate a data manifest from classifier training jobs."""
    if db_url is None:
        settings = Settings()
        db_url = _get_sync_db_url(settings)

    positive_ids, negative_ids = _query_job_embedding_sets(db_url, job_ids)
    pos_sets = _query_embedding_sets(db_url, positive_ids)
    neg_sets = _query_embedding_sets(db_url, negative_ids)

    # Collect all audio file IDs for split assignment
    all_audio_ids = [es["audio_file_id"] for es in pos_sets + neg_sets]
    split_assignments = _assign_splits(all_audio_ids, split_ratio, seed)

    # Build examples
    examples: list[dict[str, Any]] = []

    for es in pos_sets:
        n_rows = _count_parquet_rows(es["parquet_path"])
        split = split_assignments[es["audio_file_id"]]
        for ri in range(n_rows):
            examples.append(
                {
                    "id": f"es{es['id']}_row{ri}",
                    "split": split,
                    "label": 1,
                    "parquet_path": es["parquet_path"],
                    "row_index": ri,
                    "audio_file_id": es["audio_file_id"],
                    "negative_group": None,
                }
            )

    for es in neg_sets:
        n_rows = _count_parquet_rows(es["parquet_path"])
        split = split_assignments[es["audio_file_id"]]
        for ri in range(n_rows):
            examples.append(
                {
                    "id": f"es{es['id']}_row{ri}",
                    "split": split,
                    "label": 0,
                    "parquet_path": es["parquet_path"],
                    "row_index": ri,
                    "audio_file_id": es["audio_file_id"],
                    "negative_group": None,
                }
            )

    manifest = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_job_ids": [str(j) for j in job_ids],
            "positive_embedding_set_ids": positive_ids,
            "negative_embedding_set_ids": negative_ids,
            "split_strategy": "by_audio_file",
            "split_ratio": list(split_ratio),
            "seed": seed,
        },
        "examples": examples,
    }

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate autoresearch data manifest from training jobs"
    )
    parser.add_argument(
        "--job-ids",
        required=True,
        help="Comma-separated classifier training job IDs",
    )
    parser.add_argument(
        "--split-ratio",
        default="70,15,15",
        help="Train,val,test ratio (default: 70,15,15)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splits")
    parser.add_argument(
        "--output", required=True, help="Output path for data_manifest.json"
    )
    args = parser.parse_args()

    job_ids = [j.strip() for j in args.job_ids.split(",")]
    ratio_parts = [int(x) for x in args.split_ratio.split(",")]
    if len(ratio_parts) != 3:
        msg = "split-ratio must have exactly 3 comma-separated integers"
        raise ValueError(msg)
    split_ratio = (ratio_parts[0], ratio_parts[1], ratio_parts[2])

    manifest = generate_manifest(job_ids, split_ratio, args.seed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest written to {output_path} ({len(manifest['examples'])} examples)")


if __name__ == "__main__":
    main()
