"""Stability evaluation: re-run clustering with different random seeds."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np

from humpback.clustering.metrics import (
    compute_category_metrics,
    compute_cluster_metrics,
    compute_fragmentation_report,
)
from humpback.clustering.pipeline import run_clustering_pipeline

logger = logging.getLogger(__name__)


def _generate_seeds(n_runs: int, base_seed: int = 42) -> list[int]:
    """Generate deterministic seed list. First seed is always *base_seed*."""
    if n_runs <= 0:
        return []
    rng = np.random.RandomState(base_seed)
    # First seed is the base itself; rest are drawn from the RNG
    seeds = [base_seed]
    for _ in range(n_runs - 1):
        seeds.append(int(rng.randint(0, 2**31)))
    return seeds


def _compute_run_metrics(
    labels: np.ndarray,
    cluster_input: np.ndarray,
    category_labels: Sequence[str | None] | None,
) -> dict[str, Any]:
    """Compute per-run metrics reusing existing helpers."""
    n_total = len(labels)
    noise_count = int((labels == -1).sum())
    noise_fraction = noise_count / n_total if n_total > 0 else 0.0

    mask = labels != -1
    n_clusters = len(set(labels[mask].tolist())) if mask.any() else 0

    # Silhouette
    sil = None
    try:
        internal = compute_cluster_metrics(cluster_input, labels)
        sil = internal.get("silhouette_score")
    except Exception:
        pass

    # Category metrics
    ari = None
    nmi = None
    frag_index = None

    if category_labels is not None and any(c is not None for c in category_labels):
        try:
            cat = compute_category_metrics(labels, category_labels)
            ari = cat.get("adjusted_rand_index")
            nmi = cat.get("normalized_mutual_info")
        except Exception:
            pass

        try:
            report = compute_fragmentation_report(labels, category_labels, "stability")
            if report is not None:
                frag_index = report["global_fragmentation"]["mean_entropy_norm"]
        except Exception:
            pass

    return {
        "n_clusters": n_clusters,
        "noise_fraction": noise_fraction,
        "silhouette_score": sil,
        "adjusted_rand_index": ari,
        "normalized_mutual_info": nmi,
        "fragmentation_index": frag_index,
    }


def _compute_pairwise_ari(all_labels: list[np.ndarray]) -> dict[str, float | None]:
    """Pairwise ARI between all C(N,2) label vectors."""
    if len(all_labels) < 2:
        return {
            "mean_pairwise_ari": None,
            "std_pairwise_ari": None,
            "min_pairwise_ari": None,
            "max_pairwise_ari": None,
        }

    from sklearn.metrics import adjusted_rand_score

    aris = []
    for i in range(len(all_labels)):
        for j in range(i + 1, len(all_labels)):
            aris.append(float(adjusted_rand_score(all_labels[i], all_labels[j])))

    return {
        "mean_pairwise_ari": float(np.mean(aris)),
        "std_pairwise_ari": float(np.std(aris)),
        "min_pairwise_ari": float(np.min(aris)),
        "max_pairwise_ari": float(np.max(aris)),
    }


def _aggregate_metric(runs: list[dict[str, Any]], key: str) -> dict[str, float | None]:
    """Compute mean/std/min/max for a named metric across runs, skipping None."""
    values = [r[key] for r in runs if r.get(key) is not None]
    if not values:
        return {
            f"{key}_mean": None,
            f"{key}_std": None,
            f"{key}_min": None,
            f"{key}_max": None,
        }
    arr = np.array(values, dtype=np.float64)
    return {
        f"{key}_mean": float(arr.mean()),
        f"{key}_std": float(arr.std()),
        f"{key}_min": float(arr.min()),
        f"{key}_max": float(arr.max()),
    }


def run_stability_evaluation(
    embeddings: np.ndarray,
    parameters: dict[str, Any] | None,
    category_labels: Sequence[str | None] | None,
    n_runs: int = 10,
) -> dict[str, Any]:
    """Re-run the clustering pipeline N times with different random seeds.

    Returns a stability summary with pairwise ARI, aggregate metrics, and
    per-run details.

    Raises ``ValueError`` if *n_runs* < 2.
    """
    if n_runs < 2:
        raise ValueError(f"n_runs must be >= 2, got {n_runs}")

    params = dict(parameters) if parameters else {}
    seeds = _generate_seeds(n_runs)

    all_labels: list[np.ndarray] = []
    per_run: list[dict[str, Any]] = []

    for idx, seed in enumerate(seeds):
        run_params = {**params, "random_state": seed}
        result = run_clustering_pipeline(embeddings, run_params)
        all_labels.append(result.labels)

        run_metrics = _compute_run_metrics(
            result.labels, result.cluster_input, category_labels
        )
        per_run.append({"run_index": idx, "seed": seed, **run_metrics})

    pairwise = _compute_pairwise_ari(all_labels)

    aggregate: dict[str, float | None] = {}
    for key in [
        "n_clusters",
        "noise_fraction",
        "silhouette_score",
        "adjusted_rand_index",
        "normalized_mutual_info",
        "fragmentation_index",
    ]:
        aggregate.update(_aggregate_metric(per_run, key))

    return {
        "n_runs": n_runs,
        "seeds": seeds,
        "pairwise_label_agreement": pairwise,
        "aggregate_metrics": aggregate,
        "per_run": per_run,
    }
