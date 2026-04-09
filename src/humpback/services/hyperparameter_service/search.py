"""Hyperparameter search loop for classifier head tuning.

Samples configs from a search space, trains/evaluates each, and tracks the best
result. No hard-negative mining — only human-annotated labels.
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from humpback.services.hyperparameter_service.search_space import (
    config_hash,
    sample_config,
)


# ---------------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------------


def default_objective(metrics: dict[str, Any]) -> float:
    """Conservative FP-penalizing objective.

    Heavily penalizes high-confidence false positives (15x) and
    general false positives (3x) relative to recall.

    Formula: recall - 15.0 * high_conf_fp_rate - 3.0 * fp_rate
    """
    return (
        metrics["recall"]
        - 15.0 * metrics["high_conf_fp_rate"]
        - 3.0 * metrics["fp_rate"]
    )


# ---------------------------------------------------------------------------
# Embedding cache
# ---------------------------------------------------------------------------


def cache_embeddings_by_pooling(
    manifest: dict[str, Any],
    search_space: dict[str, list[Any]],
) -> dict[str, dict[str, np.ndarray]]:
    """Pre-load and context-pool embeddings for each pooling mode in the search space.

    Returns {pooling_mode: {example_id: vector}}.
    """
    from humpback.services.hyperparameter_service.train_eval import (
        apply_context_pooling,
        build_embedding_lookup as _build_embedding_lookup,
        load_parquet_cache as _load_parquet_cache,
    )

    pooling_modes = search_space.get("context_pooling", ["center"])

    parquet_cache = _load_parquet_cache(manifest)
    embedding_lookup = _build_embedding_lookup(manifest, parquet_cache)

    cache: dict[str, dict[str, np.ndarray]] = {}
    for mode in pooling_modes:
        cache[mode] = apply_context_pooling(
            manifest, embedding_lookup, parquet_cache, mode
        )
    return cache


# ---------------------------------------------------------------------------
# JSON persistence
# ---------------------------------------------------------------------------


def _write_json(path: Path, data: Any) -> None:
    """Atomically write JSON to a file."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.rename(path)


# ---------------------------------------------------------------------------
# Search loop
# ---------------------------------------------------------------------------

# Default interval (in trials) between progress callbacks.
PROGRESS_CALLBACK_INTERVAL = 10


def run_search(
    manifest: dict[str, Any],
    search_space: dict[str, list[Any]],
    n_trials: int,
    seed: int,
    results_dir: Path,
    embedding_cache: dict[str, dict[str, np.ndarray]] | None = None,
    progress_callback: Callable[
        [int, float | None, dict[str, Any] | None, dict[str, Any] | None], None
    ]
    | None = None,
) -> dict[str, Any]:
    """Run the search loop and return the best result.

    Args:
        manifest: Data manifest with examples and metadata.
        search_space: Custom search space (dimension name → list of values).
        n_trials: Number of random trials.
        seed: Random seed.
        results_dir: Directory for output files.
        embedding_cache: Pre-cached embeddings by pooling mode. If None, computed
            from the manifest and search_space pooling modes.
        progress_callback: Optional callback invoked periodically with
            (trials_completed, best_objective, best_config, best_metrics).
    """
    from humpback.services.hyperparameter_service.train_eval import train_eval

    results_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    if embedding_cache is None:
        embedding_cache = cache_embeddings_by_pooling(manifest, search_space)

    history: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    best_result: dict[str, Any] | None = None
    best_objective = float("-inf")

    history_path = results_dir / "search_history.json"
    best_path = results_dir / "best_run.json"
    fps_path = results_dir / "top_false_positives.json"

    skipped = 0
    for trial in range(n_trials):
        config = sample_config(rng, search_space)
        config["seed"] = seed

        h = config_hash(config)
        if h in seen_hashes:
            skipped += 1
            continue
        seen_hashes.add(h)

        pooling_mode = config.get("context_pooling", "center")
        precomputed = embedding_cache.get(pooling_mode)

        result = train_eval(
            manifest,
            config,
            precomputed_embeddings=precomputed,
        )
        metrics = result["metrics"]
        obj_value = default_objective(metrics)

        entry = {
            "trial": trial,
            "config_hash": h,
            "config": config,
            "metrics": metrics,
            "objective": round(obj_value, 6),
            "timestamp": time.time(),
        }
        history.append(entry)
        _write_json(history_path, history)

        if obj_value > best_objective:
            best_objective = obj_value
            best_result = entry
            _write_json(best_path, best_result)
            _write_json(fps_path, result["top_false_positives"])

        # Periodic progress reporting
        trials_done = len(history)
        if progress_callback and trials_done % PROGRESS_CALLBACK_INTERVAL == 0:
            progress_callback(
                trials_done,
                round(best_objective, 6) if best_result else None,
                best_result["config"] if best_result else None,
                best_result["metrics"] if best_result else None,
            )

    # Final progress callback
    trials_done = len(history)
    if progress_callback and trials_done % PROGRESS_CALLBACK_INTERVAL != 0:
        progress_callback(
            trials_done,
            round(best_objective, 6) if best_result else None,
            best_result["config"] if best_result else None,
            best_result["metrics"] if best_result else None,
        )

    return {
        "total_trials": len(history),
        "skipped_duplicates": skipped,
        "best_objective": round(best_objective, 6) if best_result else None,
        "best_config": best_result["config"] if best_result else None,
        "best_metrics": best_result["metrics"] if best_result else None,
    }
