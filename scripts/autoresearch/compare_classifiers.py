"""Compare an autoresearch winner against a production classifier."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
from sqlalchemy import create_engine, text

import sys as _sys
from pathlib import Path as _Path

# Ensure repo root is on sys.path so cross-script imports work
# when invoked directly (e.g. uv run scripts/autoresearch/compare_classifiers.py)
_repo_root = str(_Path(__file__).resolve().parents[2])
if _repo_root not in _sys.path:
    _sys.path.insert(0, _repo_root)

from humpback.config import Settings  # noqa: E402
from scripts.autoresearch.objectives import get_objective  # noqa: E402
from scripts.autoresearch.run_autoresearch import (  # noqa: E402
    _build_trial_manifest,
    _load_hard_negatives,
    _ordered_replay_candidate_ids,
)
from scripts.autoresearch.train_eval import (  # noqa: E402
    _load_parquet_cache,
    evaluate_classifier_on_split,
    fit_autoresearch_classifier,
    load_manifest,
    prepare_embeddings,
)


COMPARISON_METRIC_KEYS = [
    "objective",
    "precision",
    "recall",
    "fp_rate",
    "high_conf_fp_rate",
    "tp",
    "fp",
    "fn",
    "tn",
]


def _get_sync_db_url(settings: Settings) -> str:
    """Convert async database URL to a sync URL for SQLAlchemy."""
    return settings.database_url.replace("sqlite+aiosqlite:", "sqlite:")


def load_best_run(path: str | Path) -> dict[str, Any]:
    """Load an autoresearch best_run.json file."""
    with open(path) as f:
        return json.load(f)


def resolve_production_classifier(
    settings: Settings,
    *,
    classifier_name: str | None,
    classifier_id: str | None,
) -> dict[str, Any]:
    """Resolve a production classifier model record by id or name."""
    if classifier_name and classifier_id:
        raise ValueError("Provide only one of classifier_name or classifier_id")

    db_url = _get_sync_db_url(settings)
    engine = create_engine(db_url)
    row: Any | None = None
    matched_by = "id" if classifier_id else "name"
    try:
        with engine.connect() as conn:
            if classifier_id:
                row = (
                    conn.execute(
                        text(
                            "SELECT id, name, model_path, model_version, "
                            "training_summary, training_job_id, created_at "
                            "FROM classifier_models "
                            "WHERE id = :value "
                            "LIMIT 1"
                        ),
                        {"value": str(classifier_id)},
                    )
                    .mappings()
                    .first()
                )
            else:
                row = (
                    conn.execute(
                        text(
                            "SELECT id, name, model_path, model_version, "
                            "training_summary, training_job_id, created_at "
                            "FROM classifier_models "
                            "WHERE name = :value "
                            "ORDER BY created_at DESC "
                            "LIMIT 1"
                        ),
                        {"value": str(classifier_name)},
                    )
                    .mappings()
                    .first()
                )
    finally:
        engine.dispose()

    if row is None:
        if classifier_id:
            raise ValueError(f"Production classifier not found for id {classifier_id}")
        raise ValueError(f"Production classifier not found for name {classifier_name}")

    training_summary = None
    if row["training_summary"]:
        training_summary = json.loads(row["training_summary"])

    model_path = str(row["model_path"])
    if not Path(model_path).exists():
        raise ValueError(f"Production classifier artifact not found: {model_path}")

    return {
        "id": str(row["id"]),
        "name": str(row["name"]),
        "model_path": model_path,
        "model_version": str(row["model_version"]),
        "training_job_id": row["training_job_id"],
        "created_at": str(row["created_at"]),
        "training_summary": training_summary,
        "matched_by": matched_by,
    }


def build_trial_manifest_for_best_run(
    manifest: dict[str, Any],
    best_run: dict[str, Any],
    hard_negative_from: str | Path | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply phase-2 replay semantics when needed to reproduce a best run."""
    best_metrics = best_run.get("metrics", {})
    config = best_run.get("config", {})
    best_seed = int(config.get("seed", best_metrics.get("seed", 42)))
    replay_fraction = float(config.get("hard_negative_fraction", 0.0))
    available_from_best = int(best_metrics.get("available_hard_negatives", 0))

    if hard_negative_from is None:
        if available_from_best > 0:
            raise ValueError(
                "best_run.json indicates hard-negative replay was available; "
                "provide --hard-negative-from to reproduce that trial manifest"
            )
        return manifest, {
            "hard_negative_from": None,
            "available_hard_negatives": 0,
            "replayed_hard_negatives": 0,
            "used_replay_manifest": False,
        }

    hard_negative_ids = _load_hard_negatives(hard_negative_from)
    replay_candidate_order = _ordered_replay_candidate_ids(
        manifest,
        hard_negative_ids,
        best_seed,
    )
    trial_manifest, replay_count = _build_trial_manifest(
        manifest,
        replay_candidate_order,
        replay_fraction,
    )
    return trial_manifest, {
        "hard_negative_from": str(hard_negative_from),
        "available_hard_negatives": len(replay_candidate_order),
        "replayed_hard_negatives": replay_count,
        "used_replay_manifest": True,
    }


def _manifest_split_counts(manifest: dict[str, Any]) -> dict[str, dict[str, int]]:
    """Summarize label counts per split."""
    counts: dict[str, dict[str, int]] = {}
    for split in ["train", "val", "test", "unused"]:
        split_examples = [ex for ex in manifest["examples"] if ex.get("split") == split]
        if not split_examples:
            continue
        counts[split] = {
            "total": len(split_examples),
            "positive": sum(1 for ex in split_examples if int(ex["label"]) == 1),
            "negative": sum(1 for ex in split_examples if int(ex["label"]) == 0),
        }
    return counts


def _metric_delta(
    autoresearch_metrics: dict[str, Any],
    production_metrics: dict[str, Any],
) -> dict[str, float]:
    """Return autoresearch minus production metric deltas."""
    delta: dict[str, float] = {}
    for key in COMPARISON_METRIC_KEYS:
        if key not in autoresearch_metrics or key not in production_metrics:
            continue
        delta[key] = round(
            float(autoresearch_metrics[key]) - float(production_metrics[key]),
            6,
        )
    return delta


def _build_example_lookup(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index manifest examples by example id."""
    return {str(ex["id"]): ex for ex in manifest["examples"]}


def build_prediction_disagreements(
    manifest: dict[str, Any],
    autoresearch_eval: dict[str, Any],
    production_eval: dict[str, Any],
    *,
    autoresearch_threshold: float,
    production_threshold: float,
    top_n: int,
) -> list[dict[str, Any]]:
    """Return the biggest score disagreements where predictions differ."""
    examples_by_id = _build_example_lookup(manifest)
    auto_indices = {
        example_id: idx
        for idx, example_id in enumerate(autoresearch_eval["example_ids"])
    }
    prod_indices = {
        example_id: idx for idx, example_id in enumerate(production_eval["example_ids"])
    }

    disagreements: list[dict[str, Any]] = []
    common_ids = set(auto_indices) & set(prod_indices)
    for example_id in common_ids:
        auto_idx = auto_indices[example_id]
        prod_idx = prod_indices[example_id]

        auto_score = float(autoresearch_eval["scores"][auto_idx])
        prod_score = float(production_eval["scores"][prod_idx])
        auto_pred = auto_score >= autoresearch_threshold
        prod_pred = prod_score >= production_threshold
        if auto_pred == prod_pred:
            continue

        example = examples_by_id[example_id]
        disagreements.append(
            {
                "id": example_id,
                "label": int(example["label"]),
                "split": example["split"],
                "source_type": example.get("source_type"),
                "label_source": example.get("label_source"),
                "negative_group": example.get("negative_group"),
                "audio_file_id": example.get("audio_file_id"),
                "row_id": example.get("row_id"),
                "row_index": example.get("row_index"),
                "autoresearch_score": round(auto_score, 6),
                "production_score": round(prod_score, 6),
                "autoresearch_pred": int(auto_pred),
                "production_pred": int(prod_pred),
                "score_delta": round(auto_score - prod_score, 6),
                "abs_score_delta": round(abs(auto_score - prod_score), 6),
            }
        )

    disagreements.sort(key=lambda row: row["abs_score_delta"], reverse=True)
    return disagreements[:top_n]


def compare_classifiers(
    manifest: dict[str, Any],
    best_run: dict[str, Any],
    production_classifier: dict[str, Any],
    *,
    objective_name: str = "default",
    splits: list[str] | tuple[str, ...] = ("val", "test"),
    production_context_pooling: str = "center",
    production_threshold: float = 0.5,
    autoresearch_threshold: float | None = None,
    hard_negative_from: str | Path | None = None,
    top_n: int = 25,
) -> dict[str, Any]:
    """Compare one autoresearch winner against one production classifier."""
    objective_fn = get_objective(objective_name)
    comparison_manifest, replay_summary = build_trial_manifest_for_best_run(
        manifest,
        best_run,
        hard_negative_from,
    )

    autoresearch_config = dict(best_run["config"])
    effective_autoresearch_threshold = float(
        autoresearch_threshold
        if autoresearch_threshold is not None
        else autoresearch_config.get("threshold", 0.5)
    )
    effective_production_threshold = float(production_threshold)

    parquet_cache = _load_parquet_cache(comparison_manifest)
    autoresearch_model, autoresearch_transforms, autoresearch_embeddings = (
        fit_autoresearch_classifier(
            comparison_manifest,
            autoresearch_config,
            parquet_cache=parquet_cache,
        )
    )
    production_embeddings = prepare_embeddings(
        comparison_manifest,
        {"context_pooling": production_context_pooling},
        parquet_cache=parquet_cache,
    )
    production_model = joblib.load(production_classifier["model_path"])

    split_results: dict[str, Any] = {}
    for split in splits:
        autoresearch_eval = evaluate_classifier_on_split(
            comparison_manifest,
            autoresearch_embeddings,
            autoresearch_model,
            autoresearch_transforms,
            split=split,
            threshold=effective_autoresearch_threshold,
            top_n=top_n,
        )
        production_eval = evaluate_classifier_on_split(
            comparison_manifest,
            production_embeddings,
            production_model,
            [],
            split=split,
            threshold=effective_production_threshold,
            top_n=top_n,
        )

        autoresearch_metrics = dict(autoresearch_eval["metrics"])
        production_metrics = dict(production_eval["metrics"])
        autoresearch_metrics["objective"] = round(
            float(objective_fn(autoresearch_metrics)),
            6,
        )
        production_metrics["objective"] = round(
            float(objective_fn(production_metrics)),
            6,
        )

        split_results[split] = {
            "autoresearch": {
                "metrics": autoresearch_metrics,
                "top_false_positives": autoresearch_eval["top_false_positives"],
            },
            "production": {
                "metrics": production_metrics,
                "top_false_positives": production_eval["top_false_positives"],
            },
            "delta": _metric_delta(autoresearch_metrics, production_metrics),
            "prediction_disagreements": build_prediction_disagreements(
                comparison_manifest,
                autoresearch_eval,
                production_eval,
                autoresearch_threshold=effective_autoresearch_threshold,
                production_threshold=effective_production_threshold,
                top_n=top_n,
            ),
        }

    return {
        "objective_name": objective_name,
        "manifest_summary": {
            "example_count": len(comparison_manifest["examples"]),
            "split_counts": _manifest_split_counts(comparison_manifest),
            "replay": replay_summary,
        },
        "autoresearch": {
            "trial": best_run.get("trial"),
            "config_hash": best_run.get("config_hash"),
            "config": autoresearch_config,
            "threshold": effective_autoresearch_threshold,
            "source_best_run_metrics": best_run.get("metrics"),
            "source_best_run_objective": best_run.get("objective"),
        },
        "production": {
            **production_classifier,
            "threshold": effective_production_threshold,
            "context_pooling": production_context_pooling,
        },
        "splits": split_results,
    }


def _parse_splits(raw_splits: str) -> list[str]:
    """Parse and validate a comma-separated split list."""
    splits = [part.strip() for part in raw_splits.split(",") if part.strip()]
    allowed = {"train", "val", "test"}
    invalid = [split for split in splits if split not in allowed]
    if invalid:
        raise ValueError(f"Unknown split(s): {', '.join(invalid)}")
    if not splits:
        raise ValueError("At least one split is required")
    return splits


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare an autoresearch winner against a production classifier"
    )
    parser.add_argument("--manifest", required=True, help="Path to data_manifest.json")
    parser.add_argument("--best-run", required=True, help="Path to best_run.json")
    parser.add_argument(
        "--hard-negative-from",
        default=None,
        help=(
            "Optional path to top_false_positives.json when reproducing a phase-2 "
            "best run"
        ),
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--production-classifier-name",
        default="LR-v12",
        help="Classifier model name in classifier_models (default: LR-v12)",
    )
    group.add_argument(
        "--production-classifier-id",
        default=None,
        help="Classifier model id in classifier_models",
    )
    parser.add_argument(
        "--production-threshold",
        type=float,
        default=0.5,
        help="Decision threshold for the production classifier (default: 0.5)",
    )
    parser.add_argument(
        "--autoresearch-threshold",
        type=float,
        default=None,
        help="Override the threshold stored in best_run.json",
    )
    parser.add_argument(
        "--production-context-pooling",
        default="center",
        choices=["center", "mean3", "max3"],
        help="Context pooling used when feeding embeddings into the production model",
    )
    parser.add_argument(
        "--splits",
        default="val,test",
        help="Comma-separated split list to evaluate (default: val,test)",
    )
    parser.add_argument(
        "--objective",
        default="default",
        help="Objective function name used for summary deltas (default: default)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=25,
        help="Number of false positives and disagreements to keep per split",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write comparison JSON; stdout is always emitted",
    )
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    best_run = load_best_run(args.best_run)
    settings = Settings()
    production_classifier = resolve_production_classifier(
        settings,
        classifier_name=args.production_classifier_name,
        classifier_id=args.production_classifier_id,
    )
    comparison = compare_classifiers(
        manifest,
        best_run,
        production_classifier,
        objective_name=args.objective,
        splits=_parse_splits(args.splits),
        production_context_pooling=args.production_context_pooling,
        production_threshold=args.production_threshold,
        autoresearch_threshold=args.autoresearch_threshold,
        hard_negative_from=args.hard_negative_from,
        top_n=args.top_n,
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(comparison, f, indent=2)

    json.dump(comparison, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
