"""Generate a data manifest from humpback platform classifier training jobs.

Queries the database for embedding set metadata and optionally detection job
data, then builds a stable train/val/test split grouped by audio file.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from sqlalchemy import create_engine, text

from humpback.config import Settings


# Score band boundaries for negative_group assignment on detection windows.
SCORE_BANDS: list[tuple[float, float, str]] = [
    (0.50, 0.90, "det_0.50_0.90"),
    (0.90, 0.95, "det_0.90_0.95"),
    (0.95, 0.99, "det_0.95_0.99"),
    (0.99, 1.01, "det_0.99_1.00"),  # upper inclusive via > max
]


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


def _query_detection_jobs(
    db_url: str, detection_job_ids: list[str]
) -> list[dict[str, Any]]:
    """Query detection job metadata and verify labels exist."""
    engine = create_engine(db_url)
    results = []

    with engine.connect() as conn:
        for job_id in detection_job_ids:
            row = conn.execute(
                text(
                    "SELECT id, has_positive_labels, output_row_store_path "
                    "FROM detection_jobs WHERE id = :id"
                ),
                {"id": str(job_id)},
            ).fetchone()
            if row is None:
                msg = f"Detection job {job_id} not found"
                raise ValueError(msg)
            if not row[1]:
                msg = (
                    f"Detection job {job_id} has no positive labels — "
                    "only labeled detection jobs can be used"
                )
                raise ValueError(msg)
            results.append(
                {
                    "id": row[0],
                    "row_store_path": row[2],
                }
            )

    engine.dispose()
    return results


def _count_parquet_rows(parquet_path: str) -> int:
    """Count rows in a Parquet file without loading vectors."""
    metadata = pq.read_metadata(parquet_path)
    return metadata.num_rows


def _score_to_band(score: float) -> str | None:
    """Map a confidence score to a score-band negative_group string."""
    for low, high, band_name in SCORE_BANDS:
        if low <= score < high:
            return band_name
    return None


def _collect_detection_examples(
    detection_jobs: list[dict[str, Any]],
    settings: Settings,
    score_range: tuple[float, float],
) -> list[dict[str, Any]]:
    """Build manifest examples from detection job data."""
    from humpback.storage import detection_embeddings_path, detection_row_store_path

    examples: list[dict[str, Any]] = []
    score_min, score_max = score_range

    for dj in detection_jobs:
        job_id = dj["id"]
        job_id_short = job_id[:8]

        # Resolve paths
        emb_path = detection_embeddings_path(settings.storage_root, job_id)
        row_store_path = detection_row_store_path(settings.storage_root, job_id)

        if not emb_path.exists():
            msg = (
                f"Detection embeddings not found for job {job_id} "
                f"at {emb_path} — run a detection embedding job first"
            )
            raise FileNotFoundError(msg)

        # Read detection embeddings (filename, start_sec, end_sec, embedding, confidence)
        emb_table = pq.read_table(str(emb_path))
        filenames = emb_table["filename"].to_pylist()
        confidences = emb_table["confidence"].to_numpy().astype(np.float64)
        n_emb_rows = len(filenames)

        # Read detection row store for labels
        label_map: dict[int, dict[str, str]] = {}
        if row_store_path.exists():
            row_table = pq.read_table(
                str(row_store_path),
                columns=["humpback", "orca", "ship", "background"],
            )
            for i in range(len(row_table)):
                label_map[i] = {
                    "humpback": str(row_table["humpback"][i].as_py() or ""),
                    "orca": str(row_table["orca"][i].as_py() or ""),
                    "ship": str(row_table["ship"][i].as_py() or ""),
                    "background": str(row_table["background"][i].as_py() or ""),
                }

        for ri in range(n_emb_rows):
            filename = filenames[ri]
            confidence = float(confidences[ri])
            labels = label_map.get(ri, {})

            is_humpback = labels.get("humpback") == "1"
            is_orca = labels.get("orca") == "1"
            is_ship = labels.get("ship") == "1"
            is_background = labels.get("background") == "1"
            has_any_label = is_humpback or is_orca or is_ship or is_background

            if is_humpback or is_orca:
                # Labeled positive
                examples.append(
                    {
                        "id": f"det{job_id_short}_row{ri}",
                        "split": "",  # assigned later
                        "label": 1,
                        "source_type": "detection_job",
                        "parquet_path": str(emb_path),
                        "row_index": ri,
                        "audio_file_id": filename,
                        "negative_group": None,
                        "detection_confidence": round(confidence, 6),
                    }
                )
            elif is_ship or is_background:
                # Labeled negative with semantic group
                neg_group = "ship" if is_ship else "background"
                examples.append(
                    {
                        "id": f"det{job_id_short}_row{ri}",
                        "split": "",
                        "label": 0,
                        "source_type": "detection_job",
                        "parquet_path": str(emb_path),
                        "row_index": ri,
                        "audio_file_id": filename,
                        "negative_group": neg_group,
                        "detection_confidence": round(confidence, 6),
                    }
                )
            elif not has_any_label:
                # Unlabeled — hard negative candidate if within score range
                if score_min <= confidence <= score_max:
                    band = _score_to_band(confidence)
                    examples.append(
                        {
                            "id": f"det{job_id_short}_row{ri}",
                            "split": "",
                            "label": 0,
                            "source_type": "detection_job",
                            "parquet_path": str(emb_path),
                            "row_index": ri,
                            "audio_file_id": filename,
                            "negative_group": band,
                            "detection_confidence": round(confidence, 6),
                        }
                    )

    return examples


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
    job_ids: list[int | str] | None = None,
    detection_job_ids: list[str] | None = None,
    split_ratio: tuple[int, int, int] = (70, 15, 15),
    seed: int = 42,
    score_range: tuple[float, float] = (0.5, 0.995),
    db_url: str | None = None,
) -> dict[str, Any]:
    """Generate a data manifest from training jobs and/or detection jobs."""
    if not job_ids and not detection_job_ids:
        msg = "At least one of job_ids or detection_job_ids must be provided"
        raise ValueError(msg)

    settings = Settings.from_repo_env()
    if db_url is None:
        db_url = _get_sync_db_url(settings)

    examples: list[dict[str, Any]] = []
    positive_ids: list[str] = []
    negative_ids: list[str] = []

    # Embedding set sources
    if job_ids:
        positive_ids, negative_ids = _query_job_embedding_sets(db_url, job_ids)
        pos_sets = _query_embedding_sets(db_url, positive_ids)
        neg_sets = _query_embedding_sets(db_url, negative_ids)

        for es in pos_sets:
            n_rows = _count_parquet_rows(es["parquet_path"])
            for ri in range(n_rows):
                examples.append(
                    {
                        "id": f"es{es['id']}_row{ri}",
                        "split": "",
                        "label": 1,
                        "source_type": "embedding_set",
                        "parquet_path": es["parquet_path"],
                        "row_index": ri,
                        "audio_file_id": es["audio_file_id"],
                        "negative_group": None,
                    }
                )

        for es in neg_sets:
            n_rows = _count_parquet_rows(es["parquet_path"])
            for ri in range(n_rows):
                examples.append(
                    {
                        "id": f"es{es['id']}_row{ri}",
                        "split": "",
                        "label": 0,
                        "source_type": "embedding_set",
                        "parquet_path": es["parquet_path"],
                        "row_index": ri,
                        "audio_file_id": es["audio_file_id"],
                        "negative_group": None,
                    }
                )

    # Detection job sources
    det_job_id_list: list[str] = []
    if detection_job_ids:
        detection_jobs = _query_detection_jobs(db_url, detection_job_ids)
        det_job_id_list = [dj["id"] for dj in detection_jobs]
        det_examples = _collect_detection_examples(
            detection_jobs, settings, score_range
        )
        examples.extend(det_examples)

    # Assign splits based on all audio file IDs
    all_audio_ids = [ex["audio_file_id"] for ex in examples]
    split_assignments = _assign_splits(all_audio_ids, split_ratio, seed)
    for ex in examples:
        ex["split"] = split_assignments[ex["audio_file_id"]]

    manifest = {
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_job_ids": [str(j) for j in job_ids] if job_ids else [],
            "positive_embedding_set_ids": positive_ids,
            "negative_embedding_set_ids": negative_ids,
            "detection_job_ids": det_job_id_list,
            "score_range": list(score_range),
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
        default=None,
        help="Comma-separated classifier training job IDs",
    )
    parser.add_argument(
        "--detection-job-ids",
        default=None,
        help="Comma-separated detection job IDs (must have positive labels)",
    )
    parser.add_argument(
        "--score-range",
        default="0.5,0.995",
        help="Min,max confidence for unlabeled hard negatives (default: 0.5,0.995)",
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

    job_ids = None
    if args.job_ids:
        job_ids = [j.strip() for j in args.job_ids.split(",")]

    detection_job_ids = None
    if args.detection_job_ids:
        detection_job_ids = [j.strip() for j in args.detection_job_ids.split(",")]

    if not job_ids and not detection_job_ids:
        parser.error("At least one of --job-ids or --detection-job-ids is required")

    score_parts = [float(x) for x in args.score_range.split(",")]
    if len(score_parts) != 2:
        msg = "score-range must have exactly 2 comma-separated floats"
        raise ValueError(msg)
    score_range = (score_parts[0], score_parts[1])

    ratio_parts = [int(x) for x in args.split_ratio.split(",")]
    if len(ratio_parts) != 3:
        msg = "split-ratio must have exactly 3 comma-separated integers"
        raise ValueError(msg)
    split_ratio = (ratio_parts[0], ratio_parts[1], ratio_parts[2])

    manifest = generate_manifest(
        job_ids=job_ids,
        detection_job_ids=detection_job_ids,
        split_ratio=split_ratio,
        seed=args.seed,
        score_range=score_range,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)

    n_det = sum(
        1 for ex in manifest["examples"] if ex.get("source_type") == "detection_job"
    )
    n_es = sum(
        1 for ex in manifest["examples"] if ex.get("source_type") == "embedding_set"
    )
    print(
        f"Manifest written to {output_path} "
        f"({len(manifest['examples'])} examples: {n_es} from embedding sets, {n_det} from detection jobs)"
    )


if __name__ == "__main__":
    main()
