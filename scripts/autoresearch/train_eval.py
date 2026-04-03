"""Single train/eval run for autoresearch.

Loads embeddings from a manifest, trains a classifier head with the given config,
evaluates on the validation split, and returns structured metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.svm import LinearSVC

import pyarrow.parquet as pq

from humpback.processing.embeddings import read_embeddings


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


# Cache entry: (row_indices_or_None, embeddings, filenames_or_None, row_ids_or_None)
# filenames is populated only for legacy detection-format Parquet files.
# row_ids is populated only for canonical row-id detection-format Parquet files.
ParquetCacheEntry = tuple[
    np.ndarray | None,
    np.ndarray,
    list[str] | None,
    list[str] | None,
]


def _unpack_cache_entry(
    entry: ParquetCacheEntry | tuple[np.ndarray, np.ndarray, list[str] | None],
) -> tuple[np.ndarray | None, np.ndarray, list[str] | None, list[str] | None]:
    """Support both the current cache format and older 3-tuple test fixtures."""
    if len(entry) == 3:
        row_indices, embeddings, filenames = entry
        return row_indices, embeddings, filenames, None
    row_indices, embeddings, filenames, row_ids = entry
    return row_indices, embeddings, filenames, row_ids


def load_manifest(manifest_path: str | Path) -> dict[str, Any]:
    """Load a data manifest JSON file."""
    with open(manifest_path) as f:
        return json.load(f)


def _load_parquet_cache(
    manifest: dict[str, Any],
) -> dict[str, ParquetCacheEntry]:
    """Load each unique Parquet file once, auto-detecting schema format.

    Embedding set format: has ``row_index`` column → load via read_embeddings.
    Canonical detection format: has ``row_id`` column.
    Legacy detection format: has ``filename`` column → use positional indices.
    """
    paths = {ex["parquet_path"] for ex in manifest["examples"]}
    cache: dict[str, ParquetCacheEntry] = {}
    for p in sorted(paths):
        schema = pq.read_schema(p)
        col_names = set(schema.names)
        if "row_index" in col_names:
            # Embedding set format
            row_indices, embeddings = read_embeddings(Path(p))
            cache[p] = (row_indices, embeddings, None, None)
        elif "row_id" in col_names:
            # Canonical row-id detection format
            table = pq.read_table(p)
            embeddings = np.array(
                [v.as_py() for v in table["embedding"]], dtype=np.float32
            )
            row_ids = [str(rid or "").strip() for rid in table["row_id"].to_pylist()]
            cache[p] = (None, embeddings, None, row_ids)
        elif "filename" in col_names:
            # Legacy detection embeddings format
            table = pq.read_table(p)
            embeddings = np.array(
                [v.as_py() for v in table["embedding"]], dtype=np.float32
            )
            row_indices = np.arange(len(embeddings), dtype=np.int32)
            filenames = table["filename"].to_pylist()
            cache[p] = (row_indices, embeddings, filenames, None)
        else:
            msg = f"Unknown Parquet format in {p}: columns {col_names}"
            raise ValueError(msg)
    return cache


def _build_embedding_lookup(
    manifest: dict[str, Any],
    parquet_cache: dict[str, ParquetCacheEntry],
) -> dict[str, np.ndarray]:
    """Build {example_id: embedding_vector} from manifest + cached Parquet data."""
    row_index_lookups: dict[str, dict[int, int]] = {}
    row_id_lookups: dict[str, dict[str, int]] = {}
    for path, entry in parquet_cache.items():
        row_indices, _embeddings, _filenames, row_ids = _unpack_cache_entry(entry)
        if row_indices is not None:
            row_index_lookups[path] = {
                int(row_index): idx for idx, row_index in enumerate(row_indices)
            }
        if row_ids is not None:
            row_id_lookups[path] = {
                str(row_id): idx for idx, row_id in enumerate(row_ids) if row_id
            }

    lookup: dict[str, np.ndarray] = {}
    for ex in manifest["examples"]:
        entry = parquet_cache[ex["parquet_path"]]
        _row_indices, embeddings, _filenames, row_ids = _unpack_cache_entry(entry)

        idx: int | None = None
        row_id = str(ex.get("row_id") or "").strip()
        if row_id and row_ids is not None:
            idx = row_id_lookups.get(ex["parquet_path"], {}).get(row_id)
        elif ex.get("row_index") is not None:
            idx = row_index_lookups.get(ex["parquet_path"], {}).get(
                int(ex["row_index"])
            )

        if idx is not None:
            lookup[ex["id"]] = embeddings[idx]
    return lookup


# ---------------------------------------------------------------------------
# Context pooling
# ---------------------------------------------------------------------------


def _build_parquet_row_map(
    parquet_cache: dict[str, ParquetCacheEntry],
) -> dict[str, dict[int, np.ndarray]]:
    """Build {parquet_path: {row_index: vector}} for neighbor lookups."""
    row_map: dict[str, dict[int, np.ndarray]] = {}
    for path, entry in parquet_cache.items():
        row_indices, embeddings, _filenames, _row_ids = _unpack_cache_entry(entry)
        if row_indices is None:
            continue
        row_map[path] = {int(ri): embeddings[i] for i, ri in enumerate(row_indices)}
    return row_map


def _build_parquet_filename_map(
    parquet_cache: dict[str, ParquetCacheEntry],
) -> dict[str, dict[int, str]]:
    """Build {parquet_path: {row_index: filename}} for detection files.

    Only populated for detection-format Parquet files (those with filename column).
    Used to skip cross-file neighbors during context pooling.
    """
    fname_map: dict[str, dict[int, str]] = {}
    for path, entry in parquet_cache.items():
        row_indices, _embeddings, filenames, _row_ids = _unpack_cache_entry(entry)
        if filenames is not None and row_indices is not None:
            fname_map[path] = {
                int(ri): filenames[i] for i, ri in enumerate(row_indices)
            }
    return fname_map


def apply_context_pooling(
    manifest: dict[str, Any],
    embedding_lookup: dict[str, np.ndarray],
    parquet_cache: dict[str, ParquetCacheEntry],
    mode: str,
) -> dict[str, np.ndarray]:
    """Apply context pooling across adjacent windows.

    Modes:
        center: use only the current embedding (passthrough)
        mean3: mean of [left, center, right] neighbors
        max3: element-wise max of [left, center, right] neighbors

    Missing neighbors fall back to center-only. For legacy detection-format files,
    neighbors from a different audio filename are also skipped. Row-id detection
    examples also fall back to center-only because Parquet order is not a stable
    temporal-neighbor contract after embedding sync updates.
    """
    if mode == "center":
        return embedding_lookup

    row_map = _build_parquet_row_map(parquet_cache)
    fname_map = _build_parquet_filename_map(parquet_cache)
    pooled: dict[str, np.ndarray] = {}

    for ex in manifest["examples"]:
        eid = ex["id"]
        center = embedding_lookup.get(eid)
        if center is None:
            continue

        path = ex["parquet_path"]
        _row_indices, _embeddings, _filenames, row_ids = _unpack_cache_entry(
            parquet_cache[path]
        )
        if row_ids is not None and ex.get("row_id") is not None:
            pooled[eid] = center
            continue

        if ex.get("row_index") is None:
            pooled[eid] = center
            continue

        ri = ex["row_index"]
        neighbors = [center]

        # For detection files, check that neighbor is from the same audio file
        fnames = fname_map.get(path)
        current_fname = fnames.get(ri) if fnames else None

        left = row_map.get(path, {}).get(ri - 1)
        if left is not None:
            if fnames is None or fnames.get(ri - 1) == current_fname:
                neighbors.append(left)

        right = row_map.get(path, {}).get(ri + 1)
        if right is not None:
            if fnames is None or fnames.get(ri + 1) == current_fname:
                neighbors.append(right)

        stack = np.array(neighbors)
        if mode == "mean3":
            pooled[eid] = stack.mean(axis=0).astype(np.float32)
        elif mode == "max3":
            pooled[eid] = stack.max(axis=0).astype(np.float32)
        else:
            msg = f"Unknown context pooling mode: {mode!r}"
            raise ValueError(msg)

    return pooled


# ---------------------------------------------------------------------------
# Feature transforms
# ---------------------------------------------------------------------------


def build_feature_pipeline(
    config: dict[str, Any],
    X_train: np.ndarray,
) -> tuple[list[Any], np.ndarray]:
    """Build and fit feature transforms on training data.

    Returns (fitted_transforms, X_train_transformed).
    """
    transforms: list[Any] = []

    norm_mode = config.get("feature_norm", "none")
    if norm_mode == "l2":
        t = Normalizer(norm="l2")
        X_train = t.fit_transform(X_train)
        transforms.append(t)
    elif norm_mode == "standard":
        t = StandardScaler()
        X_train = t.fit_transform(X_train)
        transforms.append(t)

    pca_dim = config.get("pca_dim")
    if pca_dim is not None:
        pca_dim = min(pca_dim, X_train.shape[0], X_train.shape[1])
        t = PCA(n_components=pca_dim, random_state=config.get("seed", 42))
        X_train = t.fit_transform(X_train)
        transforms.append(t)

    return transforms, X_train


def apply_transforms(transforms: list[Any], X: np.ndarray) -> np.ndarray:
    """Apply pre-fitted transforms to data."""
    for t in transforms:
        X = t.transform(X)
    return X


def _feature_dim_from_lookup(embedding_lookup: dict[str, np.ndarray]) -> int:
    """Return the vector dimension for an embedding lookup."""
    if not embedding_lookup:
        return 0
    first = next(iter(embedding_lookup.values()))
    return int(first.shape[0]) if first.ndim > 0 else 0


def collect_split_arrays(
    manifest: dict[str, Any],
    embedding_lookup: dict[str, np.ndarray],
    split: str,
) -> tuple[list[str], np.ndarray, np.ndarray, list[str | None]]:
    """Collect IDs, features, labels, and negative groups for one split."""
    example_ids: list[str] = []
    labels: list[int] = []
    vectors: list[np.ndarray] = []
    negative_groups: list[str | None] = []

    for ex in manifest["examples"]:
        if ex.get("split") != split:
            continue
        eid = ex["id"]
        vec = embedding_lookup.get(eid)
        if vec is None:
            continue
        example_ids.append(eid)
        labels.append(int(ex["label"]))
        vectors.append(vec)
        negative_groups.append(ex.get("negative_group"))

    feature_dim = _feature_dim_from_lookup(embedding_lookup)
    if vectors:
        X = np.array(vectors, dtype=np.float32)
    else:
        X = np.empty((0, feature_dim), dtype=np.float32)
    y = np.array(labels, dtype=np.int32)

    return example_ids, y, X, negative_groups


# ---------------------------------------------------------------------------
# Classifier construction
# ---------------------------------------------------------------------------


def build_classifier(config: dict[str, Any]) -> Any:
    """Build an unfitted classifier from config."""
    seed = config.get("seed", 42)
    class_weight = {
        0: config.get("class_weight_neg", 1.0),
        1: config.get("class_weight_pos", 1.0),
    }
    clf_type = config.get("classifier", "logreg")

    if clf_type == "logreg":
        return LogisticRegression(
            C=1.0,
            class_weight=class_weight,
            max_iter=1000,
            solver="lbfgs",
            random_state=seed,
        )
    elif clf_type == "linear_svm":
        base = LinearSVC(
            class_weight=class_weight,
            max_iter=2000,
            random_state=seed,
        )
        return CalibratedClassifierCV(base, cv=3, method="sigmoid")
    elif clf_type == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(128,),
            early_stopping=True,
            max_iter=500,
            random_state=seed,
        )
    else:
        msg = f"Unknown classifier type: {clf_type!r}"
        raise ValueError(msg)


def apply_calibration(
    clf: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    config: dict[str, Any],
) -> Any:
    """Wrap a fitted classifier in probability calibration if requested."""
    cal_mode = config.get("prob_calibration", "none")
    if cal_mode == "none":
        return clf

    # SVM is already calibrated via CalibratedClassifierCV
    if config.get("classifier") == "linear_svm":
        return clf

    method = "sigmoid" if cal_mode == "platt" else "isotonic"
    calibrated = CalibratedClassifierCV(clf, cv=3, method=method)
    calibrated.fit(X_train, y_train)
    return calibrated


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

HIGH_CONF_THRESHOLD = 0.90


def compute_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float,
    negative_groups: list[str | None] | None = None,
) -> dict[str, Any]:
    """Compute evaluation metrics at the given threshold."""
    y_pred = (y_scores >= threshold).astype(int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    total_neg = int(np.sum(y_true == 0))
    fp_rate = fp / total_neg if total_neg > 0 else 0.0

    # High-confidence FP: negatives scored >= 0.90
    neg_mask = y_true == 0
    neg_scores = y_scores[neg_mask]
    high_conf_fp = int(np.sum(neg_scores >= HIGH_CONF_THRESHOLD))
    high_conf_fp_rate = high_conf_fp / total_neg if total_neg > 0 else 0.0

    metrics: dict[str, Any] = {
        "threshold": threshold,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "fp_rate": round(fp_rate, 6),
        "high_conf_fp_rate": round(high_conf_fp_rate, 6),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }

    # Grouped metrics when negative_group info is available
    if negative_groups is not None:
        group_rates: dict[str, float] = {}
        neg_indices = np.where(neg_mask)[0]
        groups_by_idx: dict[str, list[float]] = defaultdict(list)
        for i, idx in enumerate(neg_indices):
            group = negative_groups[int(idx)]
            if group is not None:
                groups_by_idx[group].append(float(neg_scores[i]))
        for group, scores in sorted(groups_by_idx.items()):
            n_high = sum(1 for s in scores if s >= HIGH_CONF_THRESHOLD)
            group_rates[group] = round(n_high / len(scores), 6) if scores else 0.0
        if group_rates:
            metrics["high_conf_fp_rate_by_group"] = group_rates

    return metrics


def find_top_false_positives(
    example_ids: list[str],
    y_true: np.ndarray,
    y_scores: np.ndarray,
    negative_groups: list[str | None] | None,
    n: int = 50,
) -> list[dict[str, Any]]:
    """Return the N highest-scoring validation negatives."""
    neg_mask = y_true == 0
    neg_indices = np.where(neg_mask)[0]
    neg_scores = y_scores[neg_mask]

    top_idx = np.argsort(neg_scores)[::-1][:n]
    results = []
    for i in top_idx:
        orig_idx = int(neg_indices[i])
        entry: dict[str, Any] = {
            "id": example_ids[orig_idx],
            "score": round(float(neg_scores[i]), 6),
        }
        if negative_groups is not None and negative_groups[orig_idx] is not None:
            entry["negative_group"] = negative_groups[orig_idx]
        results.append(entry)
    return results


def score_classifier(clf: Any, X: np.ndarray) -> np.ndarray:
    """Return positive-class scores for a fitted classifier."""
    if X.shape[0] == 0:
        return np.empty((0,), dtype=np.float32)
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:, 1]
    return clf.decision_function(X)


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
        parquet_cache = _load_parquet_cache(manifest)
    embedding_lookup = _build_embedding_lookup(manifest, parquet_cache)
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


def evaluate_classifier_on_split(
    manifest: dict[str, Any],
    embedding_lookup: dict[str, np.ndarray],
    clf: Any,
    transforms: list[Any],
    split: str,
    threshold: float,
    top_n: int = 50,
) -> dict[str, Any]:
    """Evaluate a fitted classifier on one manifest split."""
    example_ids, y_true, X, negative_groups = collect_split_arrays(
        manifest,
        embedding_lookup,
        split,
    )
    X = apply_transforms(transforms, X)
    y_scores = score_classifier(clf, X)
    metrics = compute_metrics(y_true, y_scores, threshold, negative_groups)
    top_fps = find_top_false_positives(
        example_ids,
        y_true,
        y_scores,
        negative_groups,
        n=top_n,
    )
    return {
        "split": split,
        "example_ids": example_ids,
        "labels": y_true,
        "scores": y_scores,
        "negative_groups": negative_groups,
        "metrics": metrics,
        "top_false_positives": top_fps,
    }


# ---------------------------------------------------------------------------
# Main train/eval entrypoint
# ---------------------------------------------------------------------------


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
