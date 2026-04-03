"""Single train/eval run for autoresearch.

Loads embeddings from a manifest, trains a classifier head with the given config,
evaluates on the validation split, and returns structured metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import numpy as np

from humpback.classifier.replay import (
    ParquetCacheEntry,
    apply_calibration,
    apply_transforms,  # noqa: F401 — re-exported for test imports
    build_classifier,
    build_embedding_lookup,
    build_feature_pipeline,
    collect_split_arrays,
    compute_metrics,  # noqa: F401 — re-exported for test imports
    evaluate_on_split,
    find_top_false_positives,  # noqa: F401 — re-exported for test imports
    load_manifest,
    load_parquet_cache,
    score_classifier,  # noqa: F401 — re-exported for test imports
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _unpack_cache_entry(
    entry: ParquetCacheEntry | tuple[np.ndarray, np.ndarray, list[str] | None],
) -> tuple[np.ndarray | None, np.ndarray, list[str] | None, list[str] | None]:
    """Support both the current cache format and older 3-tuple test fixtures."""
    if len(entry) == 3:
        row_indices, embeddings, filenames = entry
        return row_indices, embeddings, filenames, None
    row_indices, embeddings, filenames, row_ids = entry
    return row_indices, embeddings, filenames, row_ids


# Keep the private name as an alias so existing test imports still work.
_load_parquet_cache = load_parquet_cache
_build_embedding_lookup = build_embedding_lookup


# ---------------------------------------------------------------------------
# Context pooling — delegate to shared module, discard PoolingReport for
# backward-compatible return type.
# ---------------------------------------------------------------------------


def apply_context_pooling(
    manifest: dict[str, Any],
    embedding_lookup: dict[str, np.ndarray],
    parquet_cache: dict[str, Any],
    mode: str,
) -> dict[str, np.ndarray]:
    """Apply context pooling (backward-compatible wrapper).

    Delegates to the shared replay module but discards the PoolingReport
    to preserve the existing return type.  Normalizes old 3-tuple cache
    entries to the current 4-tuple ParquetCacheEntry format.
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


# Re-export evaluate_on_split under the original name for backward compat.
evaluate_classifier_on_split = evaluate_on_split


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


def fit_autoresearch_classifier(
    manifest: dict[str, Any],
    config: dict[str, Any],
    parquet_cache: dict[str, ParquetCacheEntry] | None = None,
    precomputed_embeddings: dict[str, np.ndarray] | None = None,
) -> tuple[Any, list[Any], dict[str, np.ndarray]]:
    """Fit the autoresearch classifier on the manifest train split."""
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


def train_eval(
    manifest: dict[str, Any],
    config: dict[str, Any],
    parquet_cache: dict[str, ParquetCacheEntry] | None = None,
    precomputed_embeddings: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Run a single train/eval cycle and return metrics.

    If parquet_cache is provided, skips re-reading Parquet files.
    If precomputed_embeddings is provided (already context-pooled), uses those directly.
    """
    clf, transforms, pooled = fit_autoresearch_classifier(
        manifest,
        config,
        parquet_cache=parquet_cache,
        precomputed_embeddings=precomputed_embeddings,
    )
    threshold = config.get("threshold", 0.5)
    split_result = evaluate_classifier_on_split(
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single autoresearch trial")
    parser.add_argument("--manifest", required=True, help="Path to data_manifest.json")
    parser.add_argument("--config", required=True, help="JSON config string")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    config = json.loads(args.config)
    result = train_eval(manifest, config)

    json.dump(result["metrics"], sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
