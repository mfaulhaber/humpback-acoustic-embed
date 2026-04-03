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

import pyarrow.parquet as pq
from sqlalchemy import create_engine, text

from humpback.config import Settings


NEGATIVE_VOCALIZATION_LABEL = "(Negative)"

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


def _query_vocalization_labels(
    db_url: str,
    detection_job_ids: list[str],
) -> dict[str, dict[str, set[str]]]:
    """Query manual vocalization labels keyed by detection_job_id and row_id."""
    engine = create_engine(db_url)
    results: dict[str, dict[str, set[str]]] = {
        job_id: {} for job_id in detection_job_ids
    }

    with engine.connect() as conn:
        for job_id in detection_job_ids:
            rows = conn.execute(
                text(
                    "SELECT row_id, label "
                    "FROM vocalization_labels "
                    "WHERE detection_job_id = :id"
                ),
                {"id": str(job_id)},
            ).fetchall()
            by_row_id: dict[str, set[str]] = {}
            for row_id, label in rows:
                rid = str(row_id or "").strip()
                if not rid:
                    continue
                by_row_id.setdefault(rid, set()).add(str(label))
            results[str(job_id)] = by_row_id

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


def _parse_label_flag(value: Any) -> bool:
    """Interpret row-store label columns stored as strings."""
    return str(value or "").strip() == "1"


def _parse_optional_float(value: Any) -> float | None:
    """Parse a nullable float stored in Parquet/string form."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _row_has_binary_supervision(row: dict[str, Any]) -> bool:
    """Whether a detection row has any binary positive/negative label."""
    return bool(row["humpback"] or row["orca"] or row["ship"] or row["background"])


def _row_id_split_group(job_id: str, start_utc: float) -> str:
    """Derive the split-group key for row-id detection examples."""
    hour_bucket = datetime.fromtimestamp(start_utc, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H"
    )
    return f"det{job_id[:8]}:{hour_bucket}"


def _read_detection_row_store_rows(
    row_store_path: Path,
) -> list[dict[str, Any]]:
    """Read the minimal row-store fields needed by autoresearch."""
    table = pq.read_table(
        str(row_store_path),
        columns=["row_id", "start_utc", "humpback", "orca", "ship", "background"],
    )
    rows: list[dict[str, Any]] = []
    for row in table.to_pylist():
        rows.append(
            {
                "row_id": str(row.get("row_id") or "").strip(),
                "start_utc": _parse_optional_float(row.get("start_utc")),
                "humpback": _parse_label_flag(row.get("humpback")),
                "orca": _parse_label_flag(row.get("orca")),
                "ship": _parse_label_flag(row.get("ship")),
                "background": _parse_label_flag(row.get("background")),
            }
        )
    return rows


def _new_detection_job_summary() -> dict[str, Any]:
    """Create a JSON-serializable summary bucket for one detection job."""
    return {
        "included_positive": 0,
        "included_negative": 0,
        "included_positives_by_source": {
            "vocalization_positive": 0,
            "binary_positive": 0,
        },
        "included_negatives_by_source": {
            "vocalization_negative": 0,
            "ship": 0,
            "background": 0,
            "score_band": 0,
        },
        "included_score_band_negatives": 0,
        "skipped_conflicts": 0,
        "skipped_unlabeled_not_explicit_negative": 0,
        "skipped_null_confidence_unlabeled": 0,
        "skipped_out_of_range_unlabeled": 0,
        "skipped_missing_embeddings": 0,
        "skipped_missing_row_store": 0,
        "skipped_missing_start_utc": 0,
    }


def _record_included_example(
    summary: dict[str, Any],
    label: int,
    label_source: str,
) -> None:
    """Update per-job summary counts after including an example."""
    if label == 1:
        summary["included_positive"] += 1
        summary["included_positives_by_source"][label_source] += 1
        return

    summary["included_negative"] += 1
    summary["included_negatives_by_source"][label_source] += 1
    if label_source == "score_band":
        summary["included_score_band_negatives"] += 1


def _classify_detection_row(
    row: dict[str, Any],
    vocalization_labels: set[str],
    confidence: float | None,
    score_range: tuple[float, float],
    include_unlabeled_hard_negatives: bool,
) -> tuple[dict[str, Any] | None, str | None]:
    """Classify one detection row into autoresearch supervision buckets."""
    has_vocalization_positive = any(
        label != NEGATIVE_VOCALIZATION_LABEL for label in vocalization_labels
    )
    has_vocalization_negative = NEGATIVE_VOCALIZATION_LABEL in vocalization_labels
    has_binary_positive = bool(row["humpback"] or row["orca"])
    has_binary_negative = bool(row["ship"] or row["background"])

    has_conflict = (
        (has_vocalization_positive and has_vocalization_negative)
        or (has_vocalization_positive and has_binary_negative)
        or (has_vocalization_negative and has_binary_positive)
        or (has_binary_positive and has_binary_negative)
    )
    if has_conflict:
        return None, "conflict"

    if has_vocalization_positive:
        return {
            "label": 1,
            "negative_group": None,
            "label_source": "vocalization_positive",
        }, None

    if has_vocalization_negative:
        return {
            "label": 0,
            "negative_group": "vocalization_negative",
            "label_source": "vocalization_negative",
        }, None

    if has_binary_positive:
        return {
            "label": 1,
            "negative_group": None,
            "label_source": "binary_positive",
        }, None

    if row["ship"]:
        return {
            "label": 0,
            "negative_group": "ship",
            "label_source": "ship",
        }, None

    if row["background"]:
        return {
            "label": 0,
            "negative_group": "background",
            "label_source": "background",
        }, None

    if not include_unlabeled_hard_negatives:
        return None, "unlabeled_not_explicit_negative"

    if confidence is None:
        return None, "null_confidence_unlabeled"

    score_min, score_max = score_range
    if not (score_min <= confidence <= score_max):
        return None, "out_of_range_unlabeled"

    band = _score_to_band(confidence)
    if band is None:
        return None, "out_of_range_unlabeled"

    return {
        "label": 0,
        "negative_group": band,
        "label_source": "score_band",
    }, None


def _collect_detection_examples(
    detection_jobs: list[dict[str, Any]],
    settings: Settings,
    score_range: tuple[float, float],
    vocalization_labels_by_job: dict[str, dict[str, set[str]]] | None = None,
    include_unlabeled_hard_negatives: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Build manifest examples from detection job data."""
    from humpback.storage import detection_embeddings_path, detection_row_store_path

    examples: list[dict[str, Any]] = []
    summaries: dict[str, dict[str, Any]] = {}
    vocalization_labels_by_job = vocalization_labels_by_job or {}

    for dj in detection_jobs:
        job_id = dj["id"]
        job_id_short = job_id[:8]
        summary = _new_detection_job_summary()
        summaries[job_id] = summary

        # Resolve paths
        emb_path = detection_embeddings_path(settings.storage_root, job_id)
        row_store_path = detection_row_store_path(settings.storage_root, job_id)

        if not emb_path.exists():
            msg = (
                f"Detection embeddings not found for job {job_id} "
                f"at {emb_path} — run a detection embedding job first"
            )
            raise FileNotFoundError(msg)

        if not row_store_path.exists():
            msg = f"Detection row store not found for job {job_id} at {row_store_path}"
            raise FileNotFoundError(msg)

        row_store_rows = _read_detection_row_store_rows(row_store_path)
        row_store_by_id = {
            row["row_id"]: row for row in row_store_rows if row["row_id"]
        }
        labels_by_row_id = vocalization_labels_by_job.get(job_id, {})

        emb_table = pq.read_table(str(emb_path))
        col_names = set(emb_table.column_names)
        raw_confidences = (
            emb_table.column("confidence").to_pylist()
            if "confidence" in col_names
            else [None] * emb_table.num_rows
        )

        if "row_id" in col_names:
            row_ids = [
                str(rid or "").strip() for rid in emb_table["row_id"].to_pylist()
            ]
            embedding_row_ids = {rid for rid in row_ids if rid}
            supervised_row_ids = {
                rid
                for rid, labels in labels_by_row_id.items()
                if rid
                and (
                    NEGATIVE_VOCALIZATION_LABEL in labels
                    or any(label != NEGATIVE_VOCALIZATION_LABEL for label in labels)
                )
            }
            supervised_row_ids.update(
                rid
                for rid, row in row_store_by_id.items()
                if rid and _row_has_binary_supervision(row)
            )
            summary["skipped_missing_embeddings"] = len(
                supervised_row_ids - embedding_row_ids
            )

            for rid, raw_confidence in zip(row_ids, raw_confidences, strict=False):
                row = row_store_by_id.get(rid)
                if row is None:
                    summary["skipped_missing_row_store"] += 1
                    continue

                start_utc = row["start_utc"]
                if start_utc is None:
                    summary["skipped_missing_start_utc"] += 1
                    continue

                confidence = (
                    float(raw_confidence) if raw_confidence is not None else None
                )
                classification, skip_reason = _classify_detection_row(
                    row=row,
                    vocalization_labels=labels_by_row_id.get(rid, set()),
                    confidence=confidence,
                    score_range=score_range,
                    include_unlabeled_hard_negatives=include_unlabeled_hard_negatives,
                )
                if classification is None:
                    if skip_reason == "conflict":
                        summary["skipped_conflicts"] += 1
                    elif skip_reason == "unlabeled_not_explicit_negative":
                        summary["skipped_unlabeled_not_explicit_negative"] += 1
                    elif skip_reason == "null_confidence_unlabeled":
                        summary["skipped_null_confidence_unlabeled"] += 1
                    else:
                        summary["skipped_out_of_range_unlabeled"] += 1
                    continue

                example = {
                    "id": f"det{job_id_short}_{rid}",
                    "split": "",
                    "label": classification["label"],
                    "source_type": "detection_job",
                    "parquet_path": str(emb_path),
                    "row_id": rid,
                    "audio_file_id": _row_id_split_group(job_id, start_utc),
                    "negative_group": classification["negative_group"],
                    "label_source": classification["label_source"],
                    "detection_confidence": (
                        round(confidence, 6) if confidence is not None else None
                    ),
                    "start_utc": start_utc,
                }
                examples.append(example)
                _record_included_example(
                    summary, example["label"], classification["label_source"]
                )
            continue

        if "filename" not in col_names:
            msg = f"Unsupported detection embeddings format in {emb_path}: {col_names}"
            raise ValueError(msg)

        filenames = emb_table["filename"].to_pylist()
        n_emb_rows = len(filenames)
        for idx, row in enumerate(row_store_rows):
            if idx < n_emb_rows:
                continue
            has_vocalization = bool(labels_by_row_id.get(row["row_id"], set()))
            if has_vocalization or _row_has_binary_supervision(row):
                summary["skipped_missing_embeddings"] += 1

        for ri, filename in enumerate(filenames):
            if ri >= len(row_store_rows):
                summary["skipped_missing_row_store"] += 1
                continue

            row = row_store_rows[ri]
            confidence = (
                float(raw_confidences[ri]) if raw_confidences[ri] is not None else None
            )
            rid = row["row_id"]
            classification, skip_reason = _classify_detection_row(
                row=row,
                vocalization_labels=labels_by_row_id.get(rid, set()),
                confidence=confidence,
                score_range=score_range,
                include_unlabeled_hard_negatives=include_unlabeled_hard_negatives,
            )
            if classification is None:
                if skip_reason == "conflict":
                    summary["skipped_conflicts"] += 1
                elif skip_reason == "unlabeled_not_explicit_negative":
                    summary["skipped_unlabeled_not_explicit_negative"] += 1
                elif skip_reason == "null_confidence_unlabeled":
                    summary["skipped_null_confidence_unlabeled"] += 1
                else:
                    summary["skipped_out_of_range_unlabeled"] += 1
                continue

            example = {
                "id": f"det{job_id_short}_row{ri}",
                "split": "",
                "label": classification["label"],
                "source_type": "detection_job",
                "parquet_path": str(emb_path),
                "row_index": ri,
                "audio_file_id": str(filename),
                "negative_group": classification["negative_group"],
                "label_source": classification["label_source"],
                "detection_confidence": (
                    round(confidence, 6) if confidence is not None else None
                ),
                "row_id": rid or None,
                "start_utc": row["start_utc"],
            }
            examples.append(example)
            _record_included_example(
                summary, example["label"], classification["label_source"]
            )

    return examples, summaries


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
    include_unlabeled_hard_negatives: bool = False,
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
    detection_job_summaries: dict[str, dict[str, Any]] = {}
    if detection_job_ids:
        detection_jobs = _query_detection_jobs(db_url, detection_job_ids)
        det_job_id_list = [dj["id"] for dj in detection_jobs]
        vocalization_labels_by_job = _query_vocalization_labels(db_url, det_job_id_list)
        det_examples, detection_job_summaries = _collect_detection_examples(
            detection_jobs,
            settings,
            score_range,
            vocalization_labels_by_job=vocalization_labels_by_job,
            include_unlabeled_hard_negatives=include_unlabeled_hard_negatives,
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
            "detection_job_summaries": detection_job_summaries,
            "score_range": list(score_range),
            "include_unlabeled_hard_negatives": include_unlabeled_hard_negatives,
            "split_strategy": "by_audio_file",
            "detection_split_strategy": (
                "by_job_hour_utc" if det_job_id_list else None
            ),
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
        help="Min,max confidence for unlabeled hard negatives when explicitly enabled (default: 0.5,0.995)",
    )
    parser.add_argument(
        "--include-unlabeled-hard-negatives",
        action="store_true",
        help="Include unlabeled detections inside --score-range as hard negatives; default is explicit negatives only",
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
        include_unlabeled_hard_negatives=args.include_unlabeled_hard_negatives,
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
