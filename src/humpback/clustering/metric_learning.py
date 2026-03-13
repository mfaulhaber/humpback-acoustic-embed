"""Metric learning refinement: triplet-loss MLP projection head."""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


def _get_tf_device() -> str:
    """Return the best available TF device for metric learning.

    Checks ``HUMPBACK_TF_FORCE_CPU`` env var first, then tries GPU via
    ``configure_tf_gpu()``, falls back to CPU.
    """
    import tensorflow as tf

    if os.environ.get("HUMPBACK_TF_FORCE_CPU", "").lower() in ("1", "true", "yes"):
        device = tf.config.list_logical_devices("CPU")[0].name
        logger.info("Metric learning: forced CPU by HUMPBACK_TF_FORCE_CPU (%s)", device)
        return device

    from humpback.processing.inference import configure_tf_gpu

    configure_tf_gpu()
    gpus = tf.config.list_logical_devices("GPU")
    if gpus:
        logger.info("Metric learning: using GPU %s", gpus[0].name)
        return gpus[0].name

    device = tf.config.list_logical_devices("CPU")[0].name
    logger.info("Metric learning: no GPU found, using CPU (%s)", device)
    return device


# Metric direction constants for comparison
_HIGHER_IS_BETTER = {
    "silhouette_score",
    "adjusted_rand_index",
    "normalized_mutual_info",
    "calinski_harabasz_score",
}
_LOWER_IS_BETTER = {
    "davies_bouldin_index",
    "noise_fraction",
    "fragmentation_index",
}


def generate_triplets(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_triplets: int,
    strategy: str = "semi-hard",
    margin: float = 1.0,
    model: Any = None,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate triplet indices (anchor, positive, negative).

    Parameters
    ----------
    embeddings : (N, D) array of labeled-subset embeddings
    labels : (N,) int-encoded labels
    n_triplets : number of triplets to generate
    strategy : "random", "hard", or "semi-hard"
    margin : margin for semi-hard mining
    model : tf.keras.Model for projected-space distance (hard/semi-hard)
    random_state : seed for reproducibility

    Returns
    -------
    (anchor_indices, positive_indices, negative_indices) each shape (n_triplets,)
    """
    import tensorflow as tf

    rng = np.random.RandomState(random_state)
    unique_labels = np.unique(labels)
    label_to_indices = {int(lb): np.where(labels == lb)[0] for lb in unique_labels}

    # Pre-compute projected distances for hard/semi-hard
    proj_dists = None
    if strategy in ("hard", "semi-hard") and embeddings.shape[0] < 10000:
        if model is not None:
            with tf.device(_get_tf_device()):
                proj = tf.math.l2_normalize(
                    model(tf.constant(embeddings, dtype=tf.float32), training=False),
                    axis=1,
                ).numpy()
        else:
            proj = embeddings
        proj_dists = cdist(proj, proj, metric="euclidean")

    anchors = np.empty(n_triplets, dtype=np.int64)
    positives = np.empty(n_triplets, dtype=np.int64)
    negatives = np.empty(n_triplets, dtype=np.int64)

    for i in range(n_triplets):
        # Pick anchor
        a_idx = rng.randint(0, len(embeddings))
        a_label = int(labels[a_idx])
        anchors[i] = a_idx

        # Pick positive (same class, different index)
        pos_pool = label_to_indices[a_label]
        if len(pos_pool) > 1:
            p_idx = a_idx
            while p_idx == a_idx:
                p_idx = pos_pool[rng.randint(0, len(pos_pool))]
        else:
            p_idx = a_idx
        positives[i] = p_idx

        # Pick negative (different class)
        neg_labels = [lb for lb in unique_labels if lb != a_label]
        neg_label = neg_labels[rng.randint(0, len(neg_labels))]
        neg_pool = label_to_indices[int(neg_label)]

        if strategy == "random" or proj_dists is None:
            negatives[i] = neg_pool[rng.randint(0, len(neg_pool))]
        elif strategy == "hard":
            # Closest negative
            all_neg_idx = np.concatenate(
                [label_to_indices[int(lb)] for lb in neg_labels]
            )
            dists = proj_dists[a_idx, all_neg_idx]
            negatives[i] = all_neg_idx[np.argmin(dists)]
        else:
            # Semi-hard: d_ap < d_an < d_ap + margin; fallback to random
            d_ap = proj_dists[a_idx, p_idx]
            all_neg_idx = np.concatenate(
                [label_to_indices[int(lb)] for lb in neg_labels]
            )
            dists = proj_dists[a_idx, all_neg_idx]
            semi_hard_mask = (dists > d_ap) & (dists < d_ap + margin)
            if semi_hard_mask.any():
                candidates = all_neg_idx[semi_hard_mask]
                negatives[i] = candidates[rng.randint(0, len(candidates))]
            else:
                negatives[i] = neg_pool[rng.randint(0, len(neg_pool))]

    return anchors, positives, negatives


def _build_projection_head(
    input_dim: int, hidden_dim: int = 512, output_dim: int = 128
):
    """Build a 2-layer MLP projection head."""
    import tensorflow as tf

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(hidden_dim, activation="relu"),
            tf.keras.layers.Dense(output_dim),
        ]
    )
    return model


def _triplet_loss(model, anchors, positives, negatives, margin):
    """Compute triplet loss with L2-normalized projections."""
    import tensorflow as tf

    a_proj = tf.math.l2_normalize(model(anchors, training=True), axis=1)
    p_proj = tf.math.l2_normalize(model(positives, training=True), axis=1)
    n_proj = tf.math.l2_normalize(model(negatives, training=True), axis=1)
    d_ap = tf.reduce_sum(tf.square(a_proj - p_proj), axis=1)
    d_an = tf.reduce_sum(tf.square(a_proj - n_proj), axis=1)
    return tf.reduce_mean(tf.maximum(0.0, d_ap - d_an + margin))


def train_projection(
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_dim: int = 128,
    hidden_dim: int = 512,
    n_epochs: int = 50,
    lr: float = 0.001,
    margin: float = 1.0,
    batch_size: int = 256,
    mining_strategy: str = "semi-hard",
    random_state: int = 42,
) -> dict[str, Any]:
    """Train MLP projection head with triplet loss.

    Returns dict with ``model``, ``refined_embeddings``, ``loss_history``,
    and training params.
    """
    import tensorflow as tf

    input_dim = embeddings.shape[1]

    with tf.device(_get_tf_device()):
        tf.random.set_seed(random_state)
        np.random.seed(random_state)

        model = _build_projection_head(input_dim, hidden_dim, output_dim)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        loss_history: list[float] = []

        for epoch in range(n_epochs):
            anchor_idx, pos_idx, neg_idx = generate_triplets(
                embeddings,
                labels,
                n_triplets=batch_size,
                strategy=mining_strategy,
                margin=margin,
                model=model,
                random_state=random_state + epoch,
            )
            a = tf.constant(embeddings[anchor_idx], dtype=tf.float32)
            p = tf.constant(embeddings[pos_idx], dtype=tf.float32)
            n = tf.constant(embeddings[neg_idx], dtype=tf.float32)

            with tf.GradientTape() as tape:
                loss = _triplet_loss(model, a, p, n, margin)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            loss_history.append(float(loss.numpy()))

        # Project ALL embeddings
        refined = tf.math.l2_normalize(
            model(tf.constant(embeddings, dtype=tf.float32), training=False),
            axis=1,
        ).numpy()

    return {
        "model": model,
        "refined_embeddings": refined,
        "loss_history": loss_history,
        "params": {
            "output_dim": output_dim,
            "hidden_dim": hidden_dim,
            "n_epochs": n_epochs,
            "lr": lr,
            "margin": margin,
            "batch_size": batch_size,
            "mining_strategy": mining_strategy,
        },
    }


def _compute_summary(
    embeddings: np.ndarray,
    category_labels: Sequence[str | None],
    params: dict[str, Any] | None,
) -> dict[str, float | None]:
    """Run clustering + metrics on embeddings and return summary dict."""
    from humpback.clustering.metrics import (
        compute_category_metrics,
        compute_cluster_metrics,
        compute_fragmentation_report,
    )
    from humpback.clustering.pipeline import run_clustering_pipeline

    result = run_clustering_pipeline(embeddings, params)
    labels = result.labels

    summary: dict[str, float | None] = {}

    # Internal metrics
    try:
        internal = compute_cluster_metrics(result.cluster_input, labels)
        summary["silhouette_score"] = internal.get("silhouette_score")
        summary["davies_bouldin_index"] = internal.get("davies_bouldin_index")
        summary["calinski_harabasz_score"] = internal.get("calinski_harabasz_score")
    except Exception:
        pass

    # Cluster count and noise
    mask = labels != -1
    n_clusters = len(set(labels[mask].tolist())) if mask.any() else 0
    noise_fraction = (
        float((labels == -1).sum()) / len(labels) if len(labels) > 0 else 0.0
    )
    summary["n_clusters"] = float(n_clusters)
    summary["noise_fraction"] = noise_fraction

    # Category metrics
    if any(c is not None for c in category_labels):
        try:
            cat = compute_category_metrics(labels, category_labels)
            summary["adjusted_rand_index"] = cat.get("adjusted_rand_index")
            summary["normalized_mutual_info"] = cat.get("normalized_mutual_info")
        except Exception:
            pass

        try:
            report = compute_fragmentation_report(labels, category_labels, "refinement")
            if report is not None:
                summary["fragmentation_index"] = report["global_fragmentation"][
                    "mean_entropy_norm"
                ]
        except Exception:
            pass

    return summary


def run_metric_learning_refinement(
    embeddings: np.ndarray,
    category_labels: Sequence[str | None],
    frag_report: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Train MLP projection, re-cluster refined embeddings, compare metrics.

    Returns None if fewer than 2 categories have >= 5 samples.
    """
    from collections import Counter

    n_total = len(embeddings)
    if n_total == 0 or len(category_labels) != n_total:
        return None

    # Filter to labeled samples
    labeled_mask = [c is not None for c in category_labels]
    labeled_indices = [i for i, m in enumerate(labeled_mask) if m]
    labeled_cats: list[str] = [cast(str, category_labels[i]) for i in labeled_indices]

    if len(labeled_indices) == 0:
        return None

    # Exclude rare categories (< 5 samples)
    counts = Counter(labeled_cats)
    kept_cats = {cat for cat, cnt in counts.items() if cnt >= 5}

    if len(kept_cats) < 2:
        return None

    # Filter to kept categories
    keep_mask = [c in kept_cats for c in labeled_cats]
    filtered_indices = [labeled_indices[i] for i, m in enumerate(keep_mask) if m]
    filtered_cats = [labeled_cats[i] for i, m in enumerate(keep_mask) if m]

    X_labeled = embeddings[filtered_indices]
    le = LabelEncoder()
    y = np.asarray(cast(Any, le.fit_transform(filtered_cats)), dtype=np.int32)
    categories_used = sorted(str(category) for category in cast(Any, le.classes_))

    # Extract ml_* params
    p = params or {}
    ml_output_dim = int(p.get("ml_output_dim", 128))
    ml_hidden_dim = int(p.get("ml_hidden_dim", 512))
    ml_n_epochs = int(p.get("ml_n_epochs", 50))
    ml_lr = float(p.get("ml_lr", 0.001))
    ml_margin = float(p.get("ml_margin", 1.0))
    ml_batch_size = int(p.get("ml_batch_size", 256))
    ml_mining_strategy = str(p.get("ml_mining_strategy", "semi-hard"))
    ml_random_state = int(p.get("random_state", 42))

    # Train projection on labeled subset
    train_result = train_projection(
        X_labeled,
        y,
        output_dim=ml_output_dim,
        hidden_dim=ml_hidden_dim,
        n_epochs=ml_n_epochs,
        lr=ml_lr,
        margin=ml_margin,
        batch_size=ml_batch_size,
        mining_strategy=ml_mining_strategy,
        random_state=ml_random_state,
    )

    # Project ALL embeddings through trained model
    import tensorflow as tf

    with tf.device(_get_tf_device()):
        refined_all = tf.math.l2_normalize(
            train_result["model"](
                tf.constant(embeddings, dtype=tf.float32), training=False
            ),
            axis=1,
        ).numpy()

    # Compute base metrics on original embeddings
    base_summary = _compute_summary(embeddings, category_labels, params)

    # Re-cluster refined embeddings with same params
    refined_summary = _compute_summary(refined_all, category_labels, params)

    # Build comparison
    all_keys = sorted(set(list(base_summary.keys()) + list(refined_summary.keys())))
    key_to_label = {
        "silhouette_score": "Silhouette Score",
        "davies_bouldin_index": "Davies-Bouldin Index",
        "calinski_harabasz_score": "Calinski-Harabasz Score",
        "adjusted_rand_index": "Adjusted Rand Index",
        "normalized_mutual_info": "Normalized Mutual Info",
        "n_clusters": "N Clusters",
        "noise_fraction": "Noise Fraction",
        "fragmentation_index": "Fragmentation Index",
    }

    comparison: list[dict[str, Any]] = []
    for key in all_keys:
        base_val = base_summary.get(key)
        ref_val = refined_summary.get(key)

        if base_val is not None and ref_val is not None:
            delta = round(ref_val - base_val, 6)
            if key in _HIGHER_IS_BETTER:
                improved = delta > 0
            elif key in _LOWER_IS_BETTER:
                improved = delta < 0
            else:
                improved = None
        else:
            delta = None
            improved = None

        comparison.append(
            {
                "metric": key_to_label.get(key, key),
                "key": key,
                "base": round(base_val, 6) if base_val is not None else None,
                "refined": round(ref_val, 6) if ref_val is not None else None,
                "delta": delta,
                "improved": improved,
            }
        )

    loss_history = train_result["loss_history"]

    return {
        "training_params": train_result["params"],
        "n_labeled_samples": len(filtered_indices),
        "n_categories": len(categories_used),
        "n_total_samples": n_total,
        "categories_used": categories_used,
        "loss_history": [round(val, 6) for val in loss_history],
        "final_loss": round(loss_history[-1], 6) if loss_history else None,
        "comparison": comparison,
        "base_summary": {
            k: round(v, 6) if v is not None else None for k, v in base_summary.items()
        },
        "refined_summary": {
            k: round(v, 6) if v is not None else None
            for k, v in refined_summary.items()
        },
        # Internal: refined embeddings for persistence by worker.
        # Underscore prefix signals this should be popped before JSON serialization.
        "_refined_embeddings": refined_all,
    }
