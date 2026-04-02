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


# Cache entry: (row_indices, embeddings, filenames_or_None)
# filenames is populated only for detection-format Parquet files.
ParquetCacheEntry = tuple[np.ndarray, np.ndarray, list[str] | None]


def load_manifest(manifest_path: str | Path) -> dict[str, Any]:
    """Load a data manifest JSON file."""
    with open(manifest_path) as f:
        return json.load(f)


def _load_parquet_cache(
    manifest: dict[str, Any],
) -> dict[str, ParquetCacheEntry]:
    """Load each unique Parquet file once, auto-detecting schema format.

    Embedding set format: has ``row_index`` column → load via read_embeddings.
    Detection format: has ``filename`` column → use positional indices.
    """
    paths = {ex["parquet_path"] for ex in manifest["examples"]}
    cache: dict[str, ParquetCacheEntry] = {}
    for p in sorted(paths):
        schema = pq.read_schema(p)
        col_names = set(schema.names)
        if "row_index" in col_names:
            # Embedding set format
            row_indices, embeddings = read_embeddings(Path(p))
            cache[p] = (row_indices, embeddings, None)
        elif "filename" in col_names:
            # Detection embeddings format
            table = pq.read_table(p)
            embeddings = np.array(
                [v.as_py() for v in table["embedding"]], dtype=np.float32
            )
            row_indices = np.arange(len(embeddings), dtype=np.int32)
            filenames = table["filename"].to_pylist()
            cache[p] = (row_indices, embeddings, filenames)
        else:
            msg = f"Unknown Parquet format in {p}: columns {col_names}"
            raise ValueError(msg)
    return cache


def _build_embedding_lookup(
    manifest: dict[str, Any],
    parquet_cache: dict[str, ParquetCacheEntry],
) -> dict[str, np.ndarray]:
    """Build {example_id: embedding_vector} from manifest + cached Parquet data."""
    lookup: dict[str, np.ndarray] = {}
    for ex in manifest["examples"]:
        row_indices, embeddings, _ = parquet_cache[ex["parquet_path"]]
        idx = int(np.searchsorted(row_indices, ex["row_index"]))
        if idx < len(row_indices) and row_indices[idx] == ex["row_index"]:
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
    for path, (row_indices, embeddings, _) in parquet_cache.items():
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
    for path, (row_indices, _, filenames) in parquet_cache.items():
        if filenames is not None:
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

    Missing neighbors fall back to center-only. For detection-format files,
    neighbors from a different audio filename are also skipped.
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
    # Load and pool embeddings
    if precomputed_embeddings is not None:
        pooled = precomputed_embeddings
    else:
        if parquet_cache is None:
            parquet_cache = _load_parquet_cache(manifest)
        embedding_lookup = _build_embedding_lookup(manifest, parquet_cache)
        pooling_mode = config.get("context_pooling", "center")
        pooled = apply_context_pooling(
            manifest, embedding_lookup, parquet_cache, pooling_mode
        )

    # Split into train/val arrays
    train_ids, train_labels, train_vecs = [], [], []
    val_ids, val_labels, val_vecs = [], [], []
    val_neg_groups: list[str | None] = []

    for ex in manifest["examples"]:
        eid = ex["id"]
        if eid not in pooled:
            continue
        vec = pooled[eid]
        if ex["split"] == "train":
            train_ids.append(eid)
            train_labels.append(ex["label"])
            train_vecs.append(vec)
        elif ex["split"] == "val":
            val_ids.append(eid)
            val_labels.append(ex["label"])
            val_vecs.append(vec)
            val_neg_groups.append(ex.get("negative_group"))

    X_train = np.array(train_vecs, dtype=np.float32)
    y_train = np.array(train_labels, dtype=np.int32)
    X_val = np.array(val_vecs, dtype=np.float32)
    y_val = np.array(val_labels, dtype=np.int32)

    # Feature transforms (fit on train, apply to both)
    transforms, X_train = build_feature_pipeline(config, X_train)
    X_val = apply_transforms(transforms, X_val)

    # Build and train classifier
    clf = build_classifier(config)
    clf.fit(X_train, y_train)

    # Apply calibration if requested
    clf = apply_calibration(clf, X_train, y_train, config)

    # Get probability scores on validation set
    if hasattr(clf, "predict_proba"):
        val_scores = clf.predict_proba(X_val)[:, 1]
    else:
        val_scores = clf.decision_function(X_val)

    # Compute metrics
    threshold = config.get("threshold", 0.5)
    metrics = compute_metrics(y_val, val_scores, threshold, val_neg_groups)

    # Top false positives
    top_fps = find_top_false_positives(val_ids, y_val, val_scores, val_neg_groups)

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
