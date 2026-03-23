"""Unit tests for multi-class vocalization label trainer."""

import numpy as np
import pytest

from humpback.classifier.label_trainer import train_label_classifier


def _make_synthetic_data(
    n_per_class: int = 20,
    n_classes: int = 3,
    dim: int = 8,
    seed: int = 42,
) -> tuple[np.ndarray, list[str]]:
    """Create synthetic embeddings with cluster structure."""
    rng = np.random.RandomState(seed)
    class_names = [f"class_{i}" for i in range(n_classes)]
    embeddings = []
    labels = []
    for i, name in enumerate(class_names):
        center = np.zeros(dim, dtype=np.float32)
        center[i % dim] = 3.0  # distinct cluster center per class
        vecs = rng.randn(n_per_class, dim).astype(np.float32) * 0.5 + center
        embeddings.append(vecs)
        labels.extend([name] * n_per_class)
    return np.vstack(embeddings), labels


def test_train_label_classifier_basic():
    """Train with 3 synthetic classes and verify summary structure."""
    X, labels = _make_synthetic_data(n_per_class=20, n_classes=3)

    pipeline, summary = train_label_classifier(X, labels)

    # Pipeline can predict
    preds = pipeline.predict(X)
    assert len(preds) == len(X)

    # Probabilities have correct shape
    probas = pipeline.predict_proba(X)
    assert probas.shape == (len(X), 3)

    # Summary fields
    assert summary["n_classes"] == 3
    assert summary["n_samples"] == 60
    assert set(summary["class_names"]) == {"class_0", "class_1", "class_2"}
    assert summary["cv_accuracy"] > 0.5
    assert summary["cv_f1_macro"] > 0.5
    assert "per_class" in summary
    assert "confusion_matrix" in summary

    # Per-class metrics
    for name in ["class_0", "class_1", "class_2"]:
        cls = summary["per_class"][name]
        assert cls["count"] == 20
        assert 0.0 <= cls["precision"] <= 1.0
        assert 0.0 <= cls["recall"] <= 1.0
        assert 0.0 <= cls["f1"] <= 1.0


def test_train_label_classifier_mlp():
    """MLP classifier type works."""
    X, labels = _make_synthetic_data(n_per_class=15, n_classes=3)

    pipeline, summary = train_label_classifier(
        X, labels, parameters={"classifier_type": "mlp"}
    )

    assert summary["classifier_type"] == "mlp"
    assert summary["cv_accuracy"] > 0.3


def test_train_label_classifier_with_l2_norm():
    """L2 normalization option works."""
    X, labels = _make_synthetic_data(n_per_class=15, n_classes=3)

    _, summary = train_label_classifier(X, labels, parameters={"l2_normalize": True})

    assert summary["l2_normalize"] is True


def test_train_label_classifier_too_few_classes():
    """Raises ValueError with fewer than 2 classes."""
    X = np.random.randn(10, 4).astype(np.float32)
    labels = ["only_one"] * 10

    with pytest.raises(ValueError, match="at least 2 distinct classes"):
        train_label_classifier(X, labels)


def test_train_label_classifier_too_few_samples():
    """Raises ValueError when a class has fewer than 2 samples."""
    X = np.random.randn(3, 4).astype(np.float32)
    labels = ["a", "a", "b"]  # class b has only 1 sample

    with pytest.raises(ValueError, match="only 1 sample"):
        train_label_classifier(X, labels)


def test_train_label_classifier_length_mismatch():
    """Raises ValueError when embeddings and labels have different lengths."""
    X = np.random.randn(10, 4).astype(np.float32)
    labels = ["a", "b"]

    with pytest.raises(ValueError, match="same length"):
        train_label_classifier(X, labels)


def test_confusion_matrix_structure():
    """Confusion matrix is a nested dict of class_name -> class_name -> int."""
    X, labels = _make_synthetic_data(n_per_class=10, n_classes=2)

    _, summary = train_label_classifier(X, labels)

    cm = summary["confusion_matrix"]
    assert set(cm.keys()) == {"class_0", "class_1"}
    for true_name, row in cm.items():
        assert set(row.keys()) == {"class_0", "class_1"}
        for pred_name, count in row.items():
            assert isinstance(count, int)
