"""Classifier baseline and active learning queue for clustering evaluation."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def run_classifier_baseline(
    embeddings: np.ndarray,
    category_labels: list[str | None],
    frag_report: dict | None = None,
    all_es_ids: list[str] | None = None,
    all_row_indices: list[int] | None = None,
    n_folds: int = 5,
    random_state: int = 42,
) -> dict[str, Any] | None:
    """Train a LogisticRegression classifier on embeddings with cross-validation.

    Returns a dict with ``classifier_report`` and ``label_queue``, or ``None``
    if fewer than 2 categories have enough samples for stratified CV.
    """
    from collections import Counter

    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    from sklearn.preprocessing import LabelEncoder

    n_total = len(embeddings)
    if n_total == 0 or len(category_labels) != n_total:
        return None

    # --- 1. Filter to labeled samples, tracking original indices ---
    labeled_mask = [c is not None for c in category_labels]
    labeled_indices = [i for i, m in enumerate(labeled_mask) if m]
    labeled_cats = [category_labels[i] for i in labeled_indices]

    if len(labeled_indices) == 0:
        return None

    # --- 2. Exclude rare categories ---
    min_samples = max(2, n_folds)
    counts = Counter(labeled_cats)
    excluded = sorted(cat for cat, cnt in counts.items() if cnt < min_samples)
    kept_cats = {cat for cat, cnt in counts.items() if cnt >= min_samples}

    if len(kept_cats) < 2:
        return None

    # Filter to kept categories
    keep_mask = [c in kept_cats for c in labeled_cats]
    filtered_indices = [labeled_indices[i] for i, m in enumerate(keep_mask) if m]
    filtered_cats = [labeled_cats[i] for i, m in enumerate(keep_mask) if m]

    X = embeddings[filtered_indices]
    le = LabelEncoder()
    y = le.fit_transform(filtered_cats)
    n_classes = len(le.classes_)

    # --- 3. Cross-validated predictions ---
    clf = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=random_state)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    proba = cross_val_predict(clf, X, y, cv=skf, method="predict_proba")
    y_pred = np.argmax(proba, axis=1)

    # --- 4. Classification report ---
    target_names = [str(c) for c in le.classes_]
    report_dict = classification_report(
        y, y_pred, target_names=target_names, output_dict=True
    )

    per_class: dict[str, dict] = {}
    for name in target_names:
        r = report_dict[name]
        per_class[name] = {
            "precision": round(float(r["precision"]), 4),
            "recall": round(float(r["recall"]), 4),
            "f1_score": round(float(r["f1-score"]), 4),
            "support": int(r["support"]),
        }

    macro = report_dict["macro avg"]
    weighted = report_dict["weighted avg"]

    def _avg_dict(d: dict) -> dict:
        return {
            "precision": round(float(d["precision"]), 4),
            "recall": round(float(d["recall"]), 4),
            "f1_score": round(float(d["f1-score"]), 4),
            "support": int(d["support"]),
        }

    # --- 5. Confusion matrix ---
    confusion: dict[str, dict[str, int]] = {}
    for true_idx, pred_idx in zip(y, y_pred):
        true_name = target_names[true_idx]
        pred_name = target_names[pred_idx]
        if true_name not in confusion:
            confusion[true_name] = {}
        confusion[true_name][pred_name] = confusion[true_name].get(pred_name, 0) + 1

    # --- 6. Per-sample uncertainty for labeled/filtered samples ---
    entropy = _compute_entropy(proba)
    margin = _compute_margin(proba)
    max_prob = np.max(proba, axis=1)

    # Normalized uncertainty [0, 1]
    log_n_classes = float(np.log(n_classes + 1e-10))
    uncertainty_scores = entropy / log_n_classes if log_n_classes > 0 else entropy

    # --- 7. Build label queue for ALL N samples ---
    frag_cats = {}
    if frag_report is not None:
        cat_frag = frag_report.get("category_fragmentation", {})
        for cat_name, frag_data in cat_frag.items():
            frag_cats[cat_name] = float(frag_data.get("normalized_entropy", 0.0))

    # Map filtered_indices back to queue entries
    filtered_set = set(filtered_indices)
    filtered_idx_to_pos = {idx: pos for pos, idx in enumerate(filtered_indices)}

    queue: list[dict[str, Any]] = []
    for gi in range(n_total):
        cat = category_labels[gi]
        es_id = all_es_ids[gi] if all_es_ids else ""
        row_idx = all_row_indices[gi] if all_row_indices else gi

        if gi in filtered_set:
            pos = filtered_idx_to_pos[gi]
            ent_val = float(entropy[pos])
            mar_val = float(margin[pos])
            mp_val = float(max_prob[pos])
            unc = float(uncertainty_scores[pos])
            pred_cat = target_names[int(y_pred[pos])]
        elif cat is not None:
            # Labeled but excluded (rare category) — treat as uncertain
            ent_val = None
            mar_val = None
            mp_val = None
            unc = 0.9  # high but below truly unlabeled
            pred_cat = None
        else:
            # Unlabeled
            ent_val = None
            mar_val = None
            mp_val = None
            unc = 1.0
            pred_cat = None

        frag_boost = frag_cats.get(cat, 0.0) if cat is not None else 0.0
        priority = (
            min(1.0, unc + 0.3 * frag_boost) if cat is None else unc + 0.3 * frag_boost
        )

        # Cap priority for unlabeled at 1.0
        if cat is None:
            priority = 1.0

        queue.append(
            {
                "global_index": gi,
                "embedding_set_id": es_id,
                "embedding_row_index": row_idx,
                "current_category": cat,
                "predicted_category": pred_cat,
                "entropy": round(ent_val, 6) if ent_val is not None else None,
                "margin": round(mar_val, 6) if mar_val is not None else None,
                "max_prob": round(mp_val, 6) if mp_val is not None else None,
                "fragmentation_boost": round(frag_boost, 6),
                "priority": round(priority, 6),
            }
        )

    # Sort by priority descending, assign ranks
    queue.sort(key=lambda e: e["priority"], reverse=True)
    for rank, entry in enumerate(queue, start=1):
        entry["rank"] = rank

    classifier_report = {
        "n_samples": len(filtered_indices),
        "n_categories": n_classes,
        "n_folds": n_folds,
        "categories_excluded": excluded,
        "overall_accuracy": round(float(report_dict["accuracy"]), 4),
        "per_class": per_class,
        "macro_avg": _avg_dict(macro),
        "weighted_avg": _avg_dict(weighted),
        "confusion_matrix": confusion,
    }

    return {
        "classifier_report": classifier_report,
        "label_queue": queue,
    }


def _compute_entropy(proba: np.ndarray) -> np.ndarray:
    """Shannon entropy per row: -sum(p * log(p + eps))."""
    eps = 1e-10
    return -np.sum(proba * np.log(proba + eps), axis=1)


def _compute_margin(proba: np.ndarray) -> np.ndarray:
    """Difference between top-1 and top-2 probability per row."""
    sorted_proba = np.sort(proba, axis=1)
    return sorted_proba[:, -1] - sorted_proba[:, -2]
