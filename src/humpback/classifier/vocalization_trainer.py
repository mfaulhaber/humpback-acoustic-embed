"""Train per-type binary relevance vocalization classifiers.

Each vocalization type gets an independent binary classifier (present/absent).
Multi-label aware: a window labeled with types A and B is positive for both
A and B classifiers, and negative for neither.
"""

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, StandardScaler

logger = logging.getLogger(__name__)


def _optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find threshold that maximizes F1 on the given predictions."""
    best_f1 = 0.0
    best_t = 0.5
    for t in np.arange(0.1, 0.95, 0.05):
        preds = (y_prob >= t).astype(int)
        f1 = float(f1_score(y_true, preds, zero_division=0.0))  # type: ignore[call-overload]
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return round(best_t, 2)


def train_multilabel_classifiers(
    embeddings: np.ndarray,
    label_sets: list[set[str]],
    parameters: dict[str, Any] | None = None,
) -> tuple[dict[str, Pipeline], dict[str, float], dict[str, dict[str, Any]]]:
    """Train independent binary classifiers for each vocalization type.

    Args:
        embeddings: N x D matrix of embedding vectors.
        label_sets: N sets of type names per window (multi-hot).
            E.g. [{"whup"}, {"whup", "moan"}, {"shriek"}, set(), ...]
        parameters: Optional dict with classifier_type, l2_normalize,
            class_weight, min_examples_per_type.

    Returns:
        (pipelines, thresholds, per_class_metrics) where:
        - pipelines: {type_name: sklearn Pipeline}
        - thresholds: {type_name: float}
        - per_class_metrics: {type_name: {ap, f1, precision, recall, n_positive, ...}}
    """
    parameters = parameters or {}
    min_examples = int(parameters.get("min_examples_per_type", 4))
    l2_normalize = bool(parameters.get("l2_normalize", False))
    class_weight = parameters.get("class_weight", "balanced")

    # Discover all types across all label sets
    all_types: set[str] = set()
    for ls in label_sets:
        all_types.update(ls)

    if not all_types:
        raise ValueError("No vocalization types found in label_sets")

    # Filter types by minimum example count
    type_counts: dict[str, int] = {}
    for t in sorted(all_types):
        count = sum(1 for ls in label_sets if t in ls)
        type_counts[t] = count

    trainable = {t for t, c in type_counts.items() if c >= min_examples}
    filtered = {t: type_counts[t] for t in sorted(all_types - trainable)}

    if filtered:
        logger.info(
            "Filtered %d types below min_examples=%d: %s",
            len(filtered),
            min_examples,
            filtered,
        )

    if not trainable:
        raise ValueError(
            f"No types meet min_examples_per_type={min_examples}. "
            f"Type counts: {type_counts}"
        )

    pipelines: dict[str, Pipeline] = {}
    thresholds: dict[str, float] = {}
    per_class_metrics: dict[str, dict[str, Any]] = {}

    for type_name in sorted(trainable):
        # Build binary labels: 1 if type present, 0 otherwise
        y = np.array([1 if type_name in ls else 0 for ls in label_sets])
        n_pos = int(y.sum())
        n_neg = int(len(y) - n_pos)

        if n_pos == 0 or n_neg == 0:
            logger.warning(
                "Skipping '%s': needs both positive and negative samples "
                "(%d positive, %d negative). Select multiple call types.",
                type_name,
                n_pos,
                n_neg,
            )
            continue

        logger.info(
            "Training '%s' classifier: %d positive, %d negative",
            type_name,
            n_pos,
            n_neg,
        )

        # Build pipeline
        steps: list[tuple[str, Any]] = []
        if l2_normalize:
            steps.append(("l2_norm", Normalizer(norm="l2")))
        steps.append(("scaler", StandardScaler()))
        steps.append(
            (
                "classifier",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=int(parameters.get("max_iter", 1000)),
                    C=float(parameters.get("C", 1.0)),
                    class_weight=class_weight if class_weight else None,
                ),
            )
        )

        pipeline = Pipeline(steps)

        # Cross-validation with threshold optimization
        minority_count = min(n_pos, n_neg)
        n_splits = min(5, minority_count)
        n_splits = max(2, n_splits)

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        # Collect out-of-fold predictions for threshold optimization + metrics
        oof_probs = np.full(len(y), np.nan)
        for train_idx, val_idx in cv.split(embeddings, y):
            X_train, X_val = embeddings[train_idx], embeddings[val_idx]
            y_train = y[train_idx]
            if len(np.unique(y_train)) < 2:
                # Fold has only one class — skip (can happen with tiny datasets)
                continue
            fold_pipe = Pipeline(
                [(name, type(est)(**est.get_params())) for name, est in steps]
            )
            fold_pipe.fit(X_train, y_train)
            oof_probs[val_idx] = fold_pipe.predict_proba(X_val)[:, 1]

        # Mask out samples that were never scored (skipped folds)
        scored_mask = ~np.isnan(oof_probs)
        if scored_mask.sum() == 0:
            # All folds skipped — not enough data for meaningful CV
            logger.warning(
                "  '%s': all CV folds skipped (extreme imbalance), "
                "training on full data without CV metrics",
                type_name,
            )
            pipeline.fit(embeddings, y)
            pipelines[type_name] = pipeline
            thresholds[type_name] = 0.5
            per_class_metrics[type_name] = {
                "ap": 0.0,
                "f1": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "n_positive": n_pos,
                "n_negative": n_neg,
                "n_total": len(y),
                "threshold": 0.5,
                "n_cv_folds": 0,
            }
            continue

        oof_probs_scored = oof_probs[scored_mask]
        y_scored = y[scored_mask]

        # Compute metrics on out-of-fold predictions
        threshold = _optimal_threshold(y_scored, oof_probs_scored)
        oof_preds = (oof_probs_scored >= threshold).astype(int)

        ap = float(average_precision_score(y_scored, oof_probs_scored))
        f1 = float(f1_score(y_scored, oof_preds, zero_division=0.0))  # type: ignore[call-overload]

        tp = int(((oof_preds == 1) & (y_scored == 1)).sum())
        fp = int(((oof_preds == 1) & (y_scored == 0)).sum())
        fn = int(((oof_preds == 0) & (y_scored == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # Final fit on all data
        pipeline.fit(embeddings, y)

        pipelines[type_name] = pipeline
        thresholds[type_name] = threshold
        per_class_metrics[type_name] = {
            "ap": round(ap, 4),
            "f1": round(f1, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "n_positive": n_pos,
            "n_negative": n_neg,
            "n_total": len(y),
            "threshold": threshold,
            "n_cv_folds": n_splits,
        }

        logger.info(
            "  '%s': AP=%.3f, F1=%.3f, threshold=%.2f",
            type_name,
            ap,
            f1,
            threshold,
        )

    if not pipelines:
        raise ValueError(
            "No classifiers could be trained. Each type needs samples from "
            "other types as negatives — select at least 2 call type folders."
        )

    return pipelines, thresholds, per_class_metrics


def save_model_artifacts(
    model_dir: Path,
    pipelines: dict[str, Pipeline],
    thresholds: dict[str, float],
    per_class_metrics: dict[str, dict[str, Any]],
    parameters: dict[str, Any] | None = None,
) -> None:
    """Save per-type .joblib files and metadata.json to model directory."""
    model_dir.mkdir(parents=True, exist_ok=True)

    vocabulary = sorted(pipelines.keys())

    for type_name, pipeline in pipelines.items():
        joblib_path = model_dir / f"{type_name}.joblib"
        joblib.dump(pipeline, joblib_path)

    metadata = {
        "vocabulary": vocabulary,
        "thresholds": thresholds,
        "per_class_metrics": per_class_metrics,
        "parameters": parameters or {},
    }
    metadata_path = model_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    logger.info("Saved %d per-type classifiers to %s", len(pipelines), model_dir)
