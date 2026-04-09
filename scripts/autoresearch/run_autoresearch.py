"""Bounded autoresearch search loop.

.. deprecated::
    This module is deprecated. Use
    ``humpback.services.hyperparameter_service.search`` instead.
    This file is retained for CLI backward compatibility and legacy
    hard-negative replay features not present in the service.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np


import sys
from pathlib import Path as _Path

# Ensure repo root is on sys.path so cross-script imports work
# when invoked directly (e.g. uv run scripts/autoresearch/run_autoresearch.py)
_repo_root = str(_Path(__file__).resolve().parents[2])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from scripts.autoresearch.objectives import get_objective  # noqa: E402
from scripts.autoresearch.search_space import config_hash, sample_config  # noqa: E402
from scripts.autoresearch.train_eval import (  # noqa: E402
    _build_embedding_lookup,
    _load_parquet_cache,
    apply_context_pooling,
    load_manifest,
    train_eval,
)


def _load_hard_negatives(path: str | Path) -> set[str]:
    """Load example IDs from a top_false_positives.json for hard-negative mining."""
    with open(path) as f:
        entries = json.load(f)
    return {e["id"] for e in entries}


def _cache_embeddings_by_pooling(
    manifest: dict[str, Any],
) -> dict[str, dict[str, np.ndarray]]:
    """Pre-load and context-pool embeddings for each pooling mode.

    Returns {pooling_mode: {example_id: vector}}.
    """
    parquet_cache = _load_parquet_cache(manifest)
    embedding_lookup = _build_embedding_lookup(manifest, parquet_cache)

    cache: dict[str, dict[str, np.ndarray]] = {}
    for mode in ["center", "mean3", "max3"]:
        cache[mode] = apply_context_pooling(
            manifest, embedding_lookup, parquet_cache, mode
        )
    return cache


def _write_json(path: Path, data: Any) -> None:
    """Atomically write JSON to a file."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.rename(path)


def _ordered_replay_candidate_ids(
    manifest: dict[str, Any],
    hard_negative_ids: set[str],
    seed: int,
) -> list[str]:
    """Return a deterministic replay order for non-train hard negatives."""
    candidate_ids = sorted(
        {
            ex["id"]
            for ex in manifest["examples"]
            if ex["id"] in hard_negative_ids and ex["split"] != "train"
        }
    )
    rng = random.Random(seed)
    rng.shuffle(candidate_ids)
    return candidate_ids


def _hard_negative_replay_count(total: int, fraction: float) -> int:
    """Convert a replay fraction into a deterministic replay count."""
    if total <= 0:
        return 0

    fraction = max(0.0, min(1.0, fraction))
    if fraction == 0.0:
        return 0

    count = int(round(total * fraction))
    return min(total, max(1, count))


def _build_trial_manifest(
    manifest: dict[str, Any],
    replay_candidate_order: list[str],
    hard_negative_fraction: float,
) -> tuple[dict[str, Any], int]:
    """Create a manifest view with a per-trial hard-negative replay subset.

    Selected replay candidates move to ``train``. Unselected replay candidates
    move to ``unused`` so all phase-2 trials score on the same eval set.
    """
    if not replay_candidate_order:
        return manifest, 0

    replay_count = _hard_negative_replay_count(
        len(replay_candidate_order),
        hard_negative_fraction,
    )
    selected_ids = set(replay_candidate_order[:replay_count])
    candidate_ids = set(replay_candidate_order)

    trial_manifest = copy.deepcopy(manifest)
    for ex in trial_manifest["examples"]:
        if ex["id"] not in candidate_ids or ex["split"] == "train":
            continue
        ex["split"] = "train" if ex["id"] in selected_ids else "unused"

    return trial_manifest, replay_count


def run_search(
    manifest: dict[str, Any],
    n_trials: int,
    objective_name: str,
    seed: int,
    results_dir: Path,
    hard_negative_ids: set[str] | None = None,
    embedding_cache: dict[str, dict[str, np.ndarray]] | None = None,
) -> dict[str, Any]:
    """Run the search loop and return the best result."""
    results_dir.mkdir(parents=True, exist_ok=True)
    objective_fn = get_objective(objective_name)
    rng = random.Random(seed)

    # Pre-cache embeddings if not provided
    if embedding_cache is None:
        embedding_cache = _cache_embeddings_by_pooling(manifest)

    replay_candidate_order: list[str] = []
    if hard_negative_ids:
        replay_candidate_order = _ordered_replay_candidate_ids(
            manifest,
            hard_negative_ids,
            seed,
        )

    history: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    best_result: dict[str, Any] | None = None
    best_objective = float("-inf")

    history_path = results_dir / "search_history.json"
    best_path = results_dir / "best_run.json"
    fps_path = results_dir / "top_false_positives.json"

    skipped = 0
    for trial in range(n_trials):
        config = sample_config(rng)
        config["seed"] = seed
        effective_config = dict(config)
        if not replay_candidate_order:
            effective_config["hard_negative_fraction"] = 0.0

        h = config_hash(effective_config)
        if h in seen_hashes:
            skipped += 1
            continue
        seen_hashes.add(h)

        pooling_mode = effective_config.get("context_pooling", "center")
        precomputed = embedding_cache.get(pooling_mode)
        trial_manifest, replay_count = _build_trial_manifest(
            manifest,
            replay_candidate_order,
            float(effective_config.get("hard_negative_fraction", 0.0)),
        )

        result = train_eval(
            trial_manifest,
            effective_config,
            precomputed_embeddings=precomputed,
        )
        metrics = result["metrics"]
        metrics["available_hard_negatives"] = len(replay_candidate_order)
        metrics["replayed_hard_negatives"] = replay_count
        obj_value = objective_fn(metrics)

        entry = {
            "trial": trial,
            "config_hash": h,
            "config": effective_config,
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

    summary = {
        "total_trials": len(history),
        "skipped_duplicates": skipped,
        "best_objective": round(best_objective, 6) if best_result else None,
        "best_config": best_result["config"] if best_result else None,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run autoresearch search loop")
    parser.add_argument("--manifest", required=True, help="Path to data_manifest.json")
    parser.add_argument(
        "--trials", type=int, default=200, help="Number of trials (default: 200)"
    )
    parser.add_argument(
        "--objective",
        default="default",
        help="Objective function name (default: default)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory for results (default: results)",
    )
    parser.add_argument(
        "--hard-negative-from",
        default=None,
        help="Path to top_false_positives.json for hard-negative mining",
    )
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    results_dir = Path(args.results_dir)

    hard_negative_ids = None
    if args.hard_negative_from:
        hard_negative_ids = _load_hard_negatives(args.hard_negative_from)
        print(f"Loaded {len(hard_negative_ids)} hard negatives")

    print(f"Starting search: {args.trials} trials, objective={args.objective}")
    print(f"Manifest: {len(manifest['examples'])} examples")

    embedding_cache = _cache_embeddings_by_pooling(manifest)
    print("Embeddings cached for all pooling modes")

    summary = run_search(
        manifest=manifest,
        n_trials=args.trials,
        objective_name=args.objective,
        seed=args.seed,
        results_dir=results_dir,
        hard_negative_ids=hard_negative_ids,
        embedding_cache=embedding_cache,
    )

    print(
        f"\nSearch complete: {summary['total_trials']} trials run, "
        f"{summary['skipped_duplicates']} duplicates skipped"
    )
    if summary["best_config"]:
        print(f"Best objective: {summary['best_objective']}")
        print(f"Best config: {json.dumps(summary['best_config'], indent=2)}")
    print(f"Results in: {results_dir}/")


if __name__ == "__main__":
    main()
