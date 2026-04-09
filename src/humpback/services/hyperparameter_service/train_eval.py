"""Single train/eval cycle for hyperparameter search.

Loads embeddings from a manifest, trains a classifier head, evaluates on a
split, and returns structured metrics. Most heavy lifting is delegated to
``humpback.classifier.replay``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from humpback.classifier.replay import (
    ParquetCacheEntry,
    apply_calibration,
    apply_transforms,
    build_classifier,
    build_embedding_lookup,
    build_feature_pipeline,
    collect_split_arrays,
    compute_metrics,
    evaluate_on_split,
    find_top_false_positives,
    load_manifest,
    load_parquet_cache,
    score_classifier,
)


# ---------------------------------------------------------------------------
# Context pooling wrapper
# ---------------------------------------------------------------------------


def apply_context_pooling(
    manifest: dict[str, Any],
    embedding_lookup: dict[str, np.ndarray],
    parquet_cache: dict[str, Any],
    mode: str,
) -> dict[str, np.ndarray]:
    """Apply context pooling (discards the PoolingReport for simpler return type).

    Normalizes old 3-tuple cache entries to the current 4-tuple
    ``ParquetCacheEntry`` format.
    """
    from humpback.classifier.replay import (
        apply_context_pooling as _apply_context_pooling,
    )

    normalized: dict[str, ParquetCacheEntry] = {}
    for path, entry in parquet_cache.items():
        if len(entry) == 3:
            row_indices, embeddings, filenames = entry
            normalized[path] = (row_indices, embeddings, filenames, None)
        else:
            normalized[path] = entry

    pooled, _report = _apply_context_pooling(
        manifest, embedding_lookup, normalized, mode
    )
    return pooled


# ---------------------------------------------------------------------------
# Embedding preparation
# ---------------------------------------------------------------------------


def prepare_embeddings(
    manifest: dict[str, Any],
    config: dict[str, Any],
    parquet_cache: dict[str, ParquetCacheEntry] | None = None,
    precomputed_embeddings: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Load and context-pool embeddings for a manifest/config pair."""
    if precomputed_embeddings is not None:
        return precomputed_embeddings

    if parquet_cache is None:
        parquet_cache = load_parquet_cache(manifest)
    embedding_lookup = build_embedding_lookup(manifest, parquet_cache)
    pooling_mode = config.get("context_pooling", "center")
    return apply_context_pooling(
        manifest, embedding_lookup, parquet_cache, pooling_mode
    )


# ---------------------------------------------------------------------------
# Classifier fitting
# ---------------------------------------------------------------------------


def fit_autoresearch_classifier(
    manifest: dict[str, Any],
    config: dict[str, Any],
    parquet_cache: dict[str, ParquetCacheEntry] | None = None,
    precomputed_embeddings: dict[str, np.ndarray] | None = None,
) -> tuple[Any, list[Any], dict[str, np.ndarray]]:
    """Fit a classifier on the manifest train split."""
    pooled = prepare_embeddings(
        manifest,
        config,
        parquet_cache=parquet_cache,
        precomputed_embeddings=precomputed_embeddings,
    )

    _train_ids, y_train, X_train, _train_neg_groups = collect_split_arrays(
        manifest,
        pooled,
        "train",
    )
    transforms, X_train = build_feature_pipeline(config, X_train)

    clf = build_classifier(config)
    clf.fit(X_train, y_train)
    clf = apply_calibration(clf, X_train, y_train, config)

    return clf, transforms, pooled


# ---------------------------------------------------------------------------
# Train + eval cycle
# ---------------------------------------------------------------------------


def train_eval(
    manifest: dict[str, Any],
    config: dict[str, Any],
    parquet_cache: dict[str, ParquetCacheEntry] | None = None,
    precomputed_embeddings: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Run a single train/eval cycle and return metrics.

    If ``parquet_cache`` is provided, skips re-reading Parquet files.
    If ``precomputed_embeddings`` is provided (already context-pooled), uses
    those directly.
    """
    clf, transforms, pooled = fit_autoresearch_classifier(
        manifest,
        config,
        parquet_cache=parquet_cache,
        precomputed_embeddings=precomputed_embeddings,
    )
    threshold = config.get("threshold", 0.5)
    split_result = evaluate_on_split(
        manifest,
        pooled,
        clf,
        transforms,
        split="val",
        threshold=threshold,
    )
    metrics = split_result["metrics"]
    top_fps = split_result["top_false_positives"]

    seed = config.get("seed", 42)
    metrics["seed"] = seed
    metrics["config"] = config

    return {"metrics": metrics, "top_false_positives": top_fps}


# Re-export names that callers may need
evaluate_classifier_on_split = evaluate_on_split

__all__ = [
    "ParquetCacheEntry",
    "apply_calibration",
    "apply_context_pooling",
    "apply_transforms",
    "build_classifier",
    "build_embedding_lookup",
    "build_feature_pipeline",
    "collect_split_arrays",
    "compute_metrics",
    "evaluate_classifier_on_split",
    "evaluate_on_split",
    "find_top_false_positives",
    "fit_autoresearch_classifier",
    "load_manifest",
    "load_parquet_cache",
    "prepare_embeddings",
    "score_classifier",
    "train_eval",
]
