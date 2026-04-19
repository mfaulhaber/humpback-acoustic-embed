"""Backfill training_summary for promoted and detection-manifest classifier models.

1. Autoresearch-candidate models: merges standard metric fields (n_positive,
   cv_accuracy, etc.) from promotion_provenance into training_summary.
2. Detection-manifest models: reads the training manifest, computes per-job
   label breakdowns, and patches training_data_source.per_job_counts.

Usage:
    uv run scripts/backfill_training_summary.py              # dry run
    uv run scripts/backfill_training_summary.py --apply       # update DB
"""

import argparse
import json
import re
import sqlite3
from pathlib import Path

from dotenv import load_dotenv

from humpback.config import Settings

load_dotenv()


def get_db_path() -> Path:
    url = Settings().database_url
    raw = url.split("///", 1)[1]
    return Path(raw)


def _backfill_candidate_models(conn: sqlite3.Connection, *, apply: bool) -> int:
    cursor = conn.execute(
        "SELECT id, name, training_summary, promotion_provenance "
        "FROM classifier_models "
        "WHERE training_source_mode = 'autoresearch_candidate'"
    )
    patched = 0
    for row in cursor.fetchall():
        model_id, name, ts_raw, pp_raw = row
        summary = json.loads(ts_raw) if ts_raw else {}
        provenance = json.loads(pp_raw) if pp_raw else None

        if "n_positive" in summary:
            print(f"  SKIP {name} ({model_id[:8]}) — already has n_positive")
            continue

        tds = summary.get("training_data_source", {})
        n_pos = int(tds.get("positive_count") or 0)
        n_neg = int(tds.get("negative_count") or 0)
        summary["n_positive"] = n_pos
        summary["n_negative"] = n_neg
        summary["balance_ratio"] = round(n_pos / n_neg, 4) if n_neg > 0 else 0.0

        if provenance:
            split_metrics = provenance.get("split_metrics") or {}
            ar_metrics: dict = {}
            for _split_name, split_data in split_metrics.items():
                if isinstance(split_data, dict) and "autoresearch" in split_data:
                    ar_metrics = split_data["autoresearch"]
                    break

            if ar_metrics:
                precision = float(ar_metrics.get("precision", 0))
                recall = float(ar_metrics.get("recall", 0))
                summary["cv_precision"] = precision
                summary["cv_recall"] = recall
                denom = precision + recall
                summary["cv_f1"] = (
                    round(2 * precision * recall / denom, 6) if denom > 0 else 0.0
                )
                tp = int(ar_metrics.get("tp", 0))
                fp = int(ar_metrics.get("fp", 0))
                fn = int(ar_metrics.get("fn", 0))
                tn = int(ar_metrics.get("tn", 0))
                total = tp + fp + fn + tn
                summary["cv_accuracy"] = (
                    round((tp + tn) / total, 6) if total > 0 else 0.0
                )
                summary["train_confusion"] = {
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                }

            trainer_params = provenance.get("trainer_parameters") or {}
            if "classifier_type" in trainer_params:
                summary["classifier_type"] = trainer_params["classifier_type"]
            class_weight = trainer_params.get("class_weight")
            if class_weight:
                summary["effective_class_weights"] = class_weight
                summary["class_weight_strategy"] = "custom"

        fields_added = [
            k
            for k in [
                "n_positive",
                "cv_accuracy",
                "cv_precision",
                "cv_f1",
                "classifier_type",
                "train_confusion",
            ]
            if k in summary
        ]
        print(f"  PATCH {name} ({model_id[:8]}) — adding: {', '.join(fields_added)}")

        if apply:
            conn.execute(
                "UPDATE classifier_models SET training_summary = ? WHERE id = ?",
                (json.dumps(summary), model_id),
            )
        patched += 1
    return patched


def _backfill_detection_manifest_models(
    conn: sqlite3.Connection, *, apply: bool
) -> int:
    cursor = conn.execute(
        "SELECT id, name, training_summary "
        "FROM classifier_models "
        "WHERE training_source_mode = 'detection_manifest'"
    )
    pattern = re.compile(r"/detections/([0-9a-f-]{36})/")
    patched = 0

    for row in cursor.fetchall():
        model_id, name, ts_raw = row
        summary = json.loads(ts_raw) if ts_raw else {}
        tds = summary.get("training_data_source", {})

        if tds.get("per_job_counts"):
            print(f"  SKIP {name} ({model_id[:8]}) — already has per_job_counts")
            continue

        manifest_path = summary.get("manifest_path")
        if not manifest_path or not Path(manifest_path).exists():
            print(
                f"  SKIP {name} ({model_id[:8]}) — manifest not found: {manifest_path}"
            )
            continue

        manifest = json.loads(Path(manifest_path).read_text())
        examples = manifest.get("examples", [])
        split = tds.get("split", "train")
        split_examples = [ex for ex in examples if ex.get("split") == split]

        job_counts: dict[str, dict[str, int]] = {}
        for ex in split_examples:
            if ex.get("source_type") != "detection_job":
                continue
            parquet_path = str(ex.get("parquet_path", ""))
            match = pattern.search(parquet_path)
            if not match:
                continue
            job_id = match.group(1)
            if job_id not in job_counts:
                job_counts[job_id] = {"positive_count": 0, "negative_count": 0}
            if int(ex.get("label", 0)) == 1:
                job_counts[job_id]["positive_count"] += 1
            else:
                job_counts[job_id]["negative_count"] += 1

        per_job_counts = [
            {"detection_job_id": jid, **counts} for jid, counts in job_counts.items()
        ]
        tds["per_job_counts"] = per_job_counts
        summary["training_data_source"] = tds

        total_pos = sum(c["positive_count"] for c in per_job_counts)
        total_neg = sum(c["negative_count"] for c in per_job_counts)
        print(
            f"  PATCH {name} ({model_id[:8]}) — "
            f"{len(per_job_counts)} jobs, {total_pos}+ / {total_neg}-"
        )

        if apply:
            conn.execute(
                "UPDATE classifier_models SET training_summary = ? WHERE id = ?",
                (json.dumps(summary), model_id),
            )
        patched += 1
    return patched


def main():
    parser = argparse.ArgumentParser(
        description="Backfill training_summary for promoted and detection-manifest models"
    )
    parser.add_argument(
        "--apply", action="store_true", help="Actually update the database"
    )
    args = parser.parse_args()

    db_path = get_db_path()
    print(f"Database: {db_path}")
    if not args.apply:
        print("DRY RUN — pass --apply to write changes\n")

    conn = sqlite3.connect(str(db_path))

    print("=== Autoresearch-candidate models ===")
    n_candidate = _backfill_candidate_models(conn, apply=args.apply)

    print("\n=== Detection-manifest models ===")
    n_detection = _backfill_detection_manifest_models(conn, apply=args.apply)

    if args.apply:
        conn.commit()
        print(f"\nCommitted: {n_candidate} candidate + {n_detection} detection models")
    else:
        print(
            f"\nWould patch: {n_candidate} candidate + {n_detection} detection models"
        )

    conn.close()


if __name__ == "__main__":
    main()
