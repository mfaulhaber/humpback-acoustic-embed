"""Compare an autoresearch search winner against a production classifier.

No hard-negative replay — manifests are used as-is.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
from sqlalchemy import create_engine, text

from humpback.config import Settings
from humpback.services.hyperparameter_service.search import default_objective


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


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------


def _get_sync_db_url(settings: Settings) -> str:
    """Convert async database URL to a sync URL for SQLAlchemy."""
    return settings.database_url.replace("sqlite+aiosqlite:", "sqlite:")


def resolve_production_classifier(
    settings: Settings,
    *,
    classifier_name: str | None = None,
    classifier_id: str | None = None,
) -> dict[str, Any]:
    """Resolve a production classifier model record by id or name."""
    if classifier_name and classifier_id:
        raise ValueError("Provide only one of classifier_name or classifier_id")
    if not classifier_name and not classifier_id:
        raise ValueError("Provide classifier_name or classifier_id")

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
        identifier = classifier_id or classifier_name
        raise ValueError(f"Production classifier not found: {identifier}")

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


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _manifest_split_counts(manifest: dict[str, Any]) -> dict[str, dict[str, int]]:
    """Summarize label counts per split."""
    counts: dict[str, dict[str, int]] = {}
    for split in ["train", "val", "test"]:
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


# ---------------------------------------------------------------------------
# Production model config extraction
# ---------------------------------------------------------------------------

_DEFAULT_CONTEXT_POOLING = "center"
_DEFAULT_THRESHOLD = 0.5


def _resolve_production_defaults(
    production_classifier: dict[str, Any],
) -> tuple[str, float]:
    """Extract context_pooling and threshold from a production classifier.

    Checks ``promoted_config`` first, then ``replay_effective_config``,
    falling back to legacy defaults for older models without training
    summary data.
    """
    ts = production_classifier.get("training_summary")
    if not ts or not isinstance(ts, dict):
        return _DEFAULT_CONTEXT_POOLING, _DEFAULT_THRESHOLD

    for key in ("promoted_config", "replay_effective_config"):
        config = ts.get(key)
        if config and isinstance(config, dict):
            pooling = config.get("context_pooling")
            threshold = config.get("threshold")
            if pooling is not None and threshold is not None:
                return str(pooling), float(threshold)

    # One field present in one config but not the other — collect best available
    pooling: str | None = None
    threshold: float | None = None
    for key in ("promoted_config", "replay_effective_config"):
        config = ts.get(key)
        if config and isinstance(config, dict):
            if pooling is None and config.get("context_pooling") is not None:
                pooling = str(config["context_pooling"])
            if threshold is None and config.get("threshold") is not None:
                threshold = float(config["threshold"])

    return (
        pooling if pooling is not None else _DEFAULT_CONTEXT_POOLING,
        threshold if threshold is not None else _DEFAULT_THRESHOLD,
    )


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------


def compare_classifiers(
    manifest: dict[str, Any],
    best_run: dict[str, Any],
    production_classifier: dict[str, Any],
    *,
    splits: list[str] | tuple[str, ...] = ("val", "test"),
    production_context_pooling: str | None = None,
    production_threshold: float | None = None,
    autoresearch_threshold: float | None = None,
    top_n: int = 25,
) -> dict[str, Any]:
    """Compare one autoresearch winner against one production classifier."""
    from humpback.services.hyperparameter_service.train_eval import (
        evaluate_classifier_on_split,
        fit_autoresearch_classifier,
        prepare_embeddings,
    )

    autoresearch_config = dict(best_run["config"])
    effective_autoresearch_threshold = float(
        autoresearch_threshold
        if autoresearch_threshold is not None
        else autoresearch_config.get("threshold", 0.5)
    )

    auto_pooling, auto_threshold = _resolve_production_defaults(production_classifier)
    effective_production_pooling = (
        production_context_pooling
        if production_context_pooling is not None
        else auto_pooling
    )
    effective_production_threshold = (
        float(production_threshold)
        if production_threshold is not None
        else auto_threshold
    )

    from humpback.services.hyperparameter_service.train_eval import (
        load_parquet_cache as _load_parquet_cache,
    )

    parquet_cache = _load_parquet_cache(manifest)
    autoresearch_model, autoresearch_transforms, autoresearch_embeddings = (
        fit_autoresearch_classifier(
            manifest,
            autoresearch_config,
            parquet_cache=parquet_cache,
        )
    )
    production_embeddings = prepare_embeddings(
        manifest,
        {"context_pooling": effective_production_pooling},
        parquet_cache=parquet_cache,
    )
    production_model = joblib.load(production_classifier["model_path"])

    split_results: dict[str, Any] = {}
    for split in splits:
        autoresearch_eval = evaluate_classifier_on_split(
            manifest,
            autoresearch_embeddings,
            autoresearch_model,
            autoresearch_transforms,
            split=split,
            threshold=effective_autoresearch_threshold,
            top_n=top_n,
        )
        production_eval = evaluate_classifier_on_split(
            manifest,
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
            float(default_objective(autoresearch_metrics)),
            6,
        )
        production_metrics["objective"] = round(
            float(default_objective(production_metrics)),
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
                manifest,
                autoresearch_eval,
                production_eval,
                autoresearch_threshold=effective_autoresearch_threshold,
                production_threshold=effective_production_threshold,
                top_n=top_n,
            ),
        }

    return {
        "manifest_summary": {
            "example_count": len(manifest["examples"]),
            "split_counts": _manifest_split_counts(manifest),
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
            "context_pooling": effective_production_pooling,
        },
        "splits": split_results,
    }
