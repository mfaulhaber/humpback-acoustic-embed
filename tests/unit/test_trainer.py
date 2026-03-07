"""Tests for binary classifier trainer."""

import numpy as np
import pytest

from humpback.classifier.trainer import train_binary_classifier


def test_train_basic():
    """Train on linearly separable synthetic data, verify pipeline and high AUC."""
    rng = np.random.RandomState(42)
    n = 100
    dim = 32

    # Linearly separable: positives centered at +2, negatives at -2
    positive = rng.randn(n, dim) + 2.0
    negative = rng.randn(n, dim) - 2.0

    pipeline, summary = train_binary_classifier(positive, negative)

    # Verify pipeline structure
    assert hasattr(pipeline, "predict")
    assert hasattr(pipeline, "predict_proba")
    assert len(pipeline.steps) == 2
    assert pipeline.steps[0][0] == "scaler"
    assert pipeline.steps[1][0] == "classifier"

    # Verify summary keys
    assert "n_positive" in summary
    assert "n_negative" in summary
    assert "cv_accuracy" in summary
    assert "cv_roc_auc" in summary
    assert "n_cv_folds" in summary
    assert summary["n_positive"] == n
    assert summary["n_negative"] == n

    # Should achieve high accuracy on separable data
    assert summary["cv_accuracy"] > 0.9
    assert summary["cv_roc_auc"] > 0.9


def test_train_with_parameters():
    """Verify custom parameters are respected."""
    rng = np.random.RandomState(42)
    positive = rng.randn(50, 16) + 1.5
    negative = rng.randn(50, 16) - 1.5

    pipeline, summary = train_binary_classifier(
        positive, negative,
        parameters={"C": 0.1, "max_iter": 500, "solver": "lbfgs"},
    )

    # Pipeline should work
    preds = pipeline.predict(rng.randn(5, 16))
    assert len(preds) == 5
    assert set(preds).issubset({0, 1})


def test_train_small_dataset():
    """Works with very small datasets (adjusts CV folds)."""
    rng = np.random.RandomState(42)
    positive = rng.randn(5, 8) + 2.0
    negative = rng.randn(5, 8) - 2.0

    pipeline, summary = train_binary_classifier(positive, negative)

    assert summary["n_cv_folds"] >= 2
    assert summary["n_cv_folds"] <= 5
    assert pipeline is not None


def test_train_rejects_too_few_positives():
    """Raise ValueError when fewer than 2 positive samples."""
    rng = np.random.RandomState(42)
    positive = rng.randn(1, 16)
    negative = rng.randn(10, 16)

    with pytest.raises(ValueError, match="at least 2 positive"):
        train_binary_classifier(positive, negative)


def test_train_rejects_too_few_negatives():
    """Raise ValueError when fewer than 2 negative samples."""
    rng = np.random.RandomState(42)
    positive = rng.randn(10, 16)
    negative = rng.randn(1, 16)

    with pytest.raises(ValueError, match="at least 2 negative"):
        train_binary_classifier(positive, negative)


def test_predict_proba_shape():
    """Verify predict_proba returns correct shape."""
    rng = np.random.RandomState(42)
    positive = rng.randn(30, 16) + 2.0
    negative = rng.randn(30, 16) - 2.0

    pipeline, _ = train_binary_classifier(positive, negative)

    test_data = rng.randn(10, 16)
    proba = pipeline.predict_proba(test_data)
    assert proba.shape == (10, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_class_weight_balanced_default():
    """With balanced weighting (default), imbalanced data doesn't predict all-positive."""
    rng = np.random.RandomState(42)
    dim = 16
    # Heavily imbalanced: 100 positive, 10 negative — both near zero
    positive = rng.randn(100, dim) + 0.5
    negative = rng.randn(10, dim) - 0.5

    pipeline, summary = train_binary_classifier(positive, negative)

    # Generate test data centered at origin — should not all be positive
    test_data = rng.randn(50, dim)
    preds = pipeline.predict(test_data)
    n_positive = int(np.sum(preds == 1))
    # With balanced weights, not everything should be predicted positive
    assert n_positive < 50, "Balanced weighting should not predict all-positive"


def test_class_weight_none_override():
    """Passing class_weight=None restores old behavior."""
    rng = np.random.RandomState(42)
    positive = rng.randn(50, 16) + 2.0
    negative = rng.randn(50, 16) - 2.0

    pipeline, summary = train_binary_classifier(
        positive, negative,
        parameters={"class_weight": None},
    )

    # Should still produce a working pipeline
    assert hasattr(pipeline, "predict")
    clf = pipeline.named_steps["classifier"]
    assert clf.class_weight is None


def test_imbalance_warning():
    """Ratio > 3:1 produces imbalance_warning in summary."""
    rng = np.random.RandomState(42)
    positive = rng.randn(40, 16) + 2.0
    negative = rng.randn(10, 16) - 2.0

    _, summary = train_binary_classifier(positive, negative)

    assert summary["balance_ratio"] == 4.0
    assert "imbalance_warning" in summary
    assert "imbalance" in summary["imbalance_warning"].lower()


def test_no_warning_when_balanced():
    """1:1 ratio has no imbalance_warning."""
    rng = np.random.RandomState(42)
    positive = rng.randn(30, 16) + 2.0
    negative = rng.randn(30, 16) - 2.0

    _, summary = train_binary_classifier(positive, negative)

    assert summary["balance_ratio"] == 1.0
    assert "imbalance_warning" not in summary
