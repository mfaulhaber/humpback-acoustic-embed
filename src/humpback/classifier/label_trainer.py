"""Train a multi-class vocalization type classifier on detection embeddings."""

import logging
from typing import Any, cast

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, Normalizer, StandardScaler

logger = logging.getLogger(__name__)


def train_label_classifier(
    embeddings: np.ndarray,
    labels: list[str],
    parameters: dict | None = None,
) -> tuple[Pipeline, dict]:
    """Train a multi-class classifier for vocalization types.

    Args:
        embeddings: N x D matrix of embedding vectors.
        labels: N string labels (e.g. "whup", "moan", "shriek").
        parameters: Optional dict with classifier_type, l2_normalize, etc.

    Returns:
        (pipeline, summary_dict) where pipeline includes a LabelEncoder step
        and summary includes per-class metrics and confusion matrix.
    """
    parameters = parameters or {}

    if len(embeddings) != len(labels):
        raise ValueError(
            f"Embeddings ({len(embeddings)}) and labels ({len(labels)}) "
            f"must have the same length"
        )

    # Encode string labels to integers
    le = LabelEncoder()
    y = le.fit_transform(labels)
    class_names: list[str] = list(le.classes_)  # type: ignore[arg-type]
    n_classes = len(class_names)

    if n_classes < 2:
        raise ValueError(
            f"Need at least 2 distinct classes, got {n_classes}: {class_names}"
        )

    # Check minimum samples per class
    class_counts = {name: int(np.sum(y == i)) for i, name in enumerate(class_names)}
    for name, count in class_counts.items():
        if count < 2:
            raise ValueError(
                f"Class '{name}' has only {count} sample(s), need at least 2"
            )

    X = embeddings
    classifier_type = parameters.get("classifier_type", "logistic_regression")
    l2_normalize = parameters.get("l2_normalize", False)
    class_weight = parameters.get("class_weight", "balanced")

    # Build pipeline
    steps: list[tuple[str, Any]] = []
    if l2_normalize:
        steps.append(("l2_norm", Normalizer(norm="l2")))
    steps.append(("scaler", StandardScaler()))

    if classifier_type == "mlp":
        steps.append(
            (
                "classifier",
                MLPClassifier(
                    hidden_layer_sizes=(128,),
                    max_iter=parameters.get("max_iter", 500),
                    early_stopping=True,
                    random_state=42,
                ),
            )
        )
    else:
        steps.append(
            (
                "classifier",
                LogisticRegression(
                    solver=parameters.get("solver", "lbfgs"),
                    max_iter=parameters.get("max_iter", 1000),
                    C=parameters.get("C", 1.0),
                    class_weight=class_weight,
                ),
            )
        )

    pipeline = Pipeline(steps)

    # Cross-validation
    minority_count = int(min(np.sum(y == c) for c in range(n_classes)))
    n_splits = min(5, minority_count)
    n_splits = max(2, n_splits)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    scoring = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"]
    cv_results = cross_validate(
        pipeline,
        X,
        y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        error_score=cast(Any, "raise"),
    )

    # Final fit on all data
    pipeline.fit(X, y)

    # Per-class metrics on training set
    train_preds = pipeline.predict(X)
    report: dict[str, Any] = classification_report(  # type: ignore[assignment]
        y,
        train_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0,  # type: ignore[arg-type]
    )

    cm = confusion_matrix(y, train_preds)
    # Convert to nested dict: {true_label: {pred_label: count}}
    confusion_dict: dict[str, dict[str, int]] = {}
    for i, true_name in enumerate(class_names):
        confusion_dict[true_name] = {
            pred_name: int(cm[i][j]) for j, pred_name in enumerate(class_names)
        }

    # Per-class summary
    per_class: dict[str, dict[str, Any]] = {}
    for name in class_names:
        cls_report = report[name]
        per_class[name] = {
            "count": class_counts[name],
            "precision": float(cls_report["precision"]),
            "recall": float(cls_report["recall"]),
            "f1": float(cls_report["f1-score"]),
        }

    summary: dict[str, Any] = {
        "class_names": class_names,
        "n_classes": n_classes,
        "n_samples": len(labels),
        "per_class": per_class,
        "classifier_type": classifier_type,
        "l2_normalize": l2_normalize,
        "class_weight_strategy": str(class_weight),
        "cv_accuracy": float(np.nanmean(cv_results["test_accuracy"])),
        "cv_accuracy_std": float(np.nanstd(cv_results["test_accuracy"])),
        "cv_precision_macro": float(np.nanmean(cv_results["test_precision_macro"])),
        "cv_recall_macro": float(np.nanmean(cv_results["test_recall_macro"])),
        "cv_f1_macro": float(np.nanmean(cv_results["test_f1_macro"])),
        "cv_f1_macro_std": float(np.nanstd(cv_results["test_f1_macro"])),
        "cv_f1_weighted": float(np.nanmean(cv_results["test_f1_weighted"])),
        "n_cv_folds": n_splits,
        "confusion_matrix": confusion_dict,
    }

    logger.info(
        "Trained %d-class label classifier: accuracy=%.3f (+-%.3f), "
        "F1-macro=%.3f (+-%.3f), classes=%s",
        n_classes,
        summary["cv_accuracy"],
        summary["cv_accuracy_std"],
        summary["cv_f1_macro"],
        summary["cv_f1_macro_std"],
        class_names,
    )

    return pipeline, summary
