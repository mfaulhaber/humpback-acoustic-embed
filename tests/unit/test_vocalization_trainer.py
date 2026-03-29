"""Unit tests for multi-label vocalization trainer."""

import json

import numpy as np
import pytest

from humpback.classifier.vocalization_trainer import (
    save_model_artifacts,
    train_multilabel_classifiers,
)


def _make_synthetic_data(
    n_per_type: int = 20, dim: int = 16, seed: int = 42
) -> tuple[np.ndarray, list[set[str]]]:
    """Create synthetic embeddings with separable clusters per type."""
    rng = np.random.RandomState(seed)
    types = ["whup", "moan", "shriek"]
    centers = {
        "whup": rng.randn(dim) * 2,
        "moan": rng.randn(dim) * 2 + 3.0,
        "shriek": rng.randn(dim) * 2 - 3.0,
    }

    embeddings = []
    label_sets: list[set[str]] = []

    for t in types:
        for _ in range(n_per_type):
            vec = centers[t] + rng.randn(dim) * 0.5
            embeddings.append(vec)
            label_sets.append({t})

    # Add a few multi-label windows (whup + moan)
    for _ in range(5):
        vec = (centers["whup"] + centers["moan"]) / 2 + rng.randn(dim) * 0.3
        embeddings.append(vec)
        label_sets.append({"whup", "moan"})

    # Add some unlabeled windows (empty set = negative for all)
    for _ in range(10):
        vec = rng.randn(dim) * 5
        embeddings.append(vec)
        label_sets.append(set())

    X = np.array(embeddings, dtype=np.float32)
    return X, label_sets


class TestMultiLabelTraining:
    def test_basic_training(self):
        """Train per-type classifiers on synthetic data."""
        X, label_sets = _make_synthetic_data()
        pipelines, thresholds, metrics = train_multilabel_classifiers(X, label_sets)

        assert set(pipelines.keys()) == {"whup", "moan", "shriek"}
        assert set(thresholds.keys()) == {"whup", "moan", "shriek"}
        assert set(metrics.keys()) == {"whup", "moan", "shriek"}

        for t in ["whup", "moan", "shriek"]:
            assert 0.0 < thresholds[t] < 1.0
            assert metrics[t]["ap"] > 0
            assert metrics[t]["n_positive"] > 0
            assert metrics[t]["n_negative"] > 0

    def test_multi_label_aware_negatives(self):
        """A window labeled with [A, B] must be positive for A AND B, negative for neither."""
        X, label_sets = _make_synthetic_data()
        pipelines, _, metrics = train_multilabel_classifiers(X, label_sets)

        # Multi-label windows: 5 windows with {"whup", "moan"}
        # whup classifier: 20 single + 5 multi = 25 positive
        assert metrics["whup"]["n_positive"] == 25
        # moan classifier: 20 single + 5 multi = 25 positive
        assert metrics["moan"]["n_positive"] == 25
        # shriek classifier: 20 single, no multi-label overlap
        assert metrics["shriek"]["n_positive"] == 20

        # Total windows = 20*3 + 5 + 10 = 75
        total = 75
        # shriek negatives = total - 20 = 55
        assert metrics["shriek"]["n_negative"] == total - 20
        # whup negatives = total - 25 = 50
        assert metrics["whup"]["n_negative"] == total - 25

    def test_min_examples_filtering(self):
        """Types below min_examples_per_type are filtered out."""
        rng = np.random.RandomState(42)
        dim = 8
        X = rng.randn(30, dim).astype(np.float32)
        label_sets: list[set[str]] = []
        # whup: 15 examples
        for _ in range(15):
            label_sets.append({"whup"})
        # moan: 3 examples (below default min=4)
        for _ in range(3):
            label_sets.append({"moan"})
        # unlabeled: 12 examples
        for _ in range(12):
            label_sets.append(set())

        pipelines, thresholds, metrics = train_multilabel_classifiers(
            X, label_sets, parameters={"min_examples_per_type": 4}
        )

        assert "whup" in pipelines
        assert "moan" not in pipelines
        assert "moan" not in thresholds

    def test_all_types_filtered_raises(self):
        """If all types are below min_examples, raise ValueError."""
        rng = np.random.RandomState(42)
        X = rng.randn(10, 8).astype(np.float32)
        label_sets: list[set[str]] = [{"whup"}] * 2 + [set()] * 8

        with pytest.raises(ValueError, match="No types meet"):
            train_multilabel_classifiers(
                X, label_sets, parameters={"min_examples_per_type": 5}
            )

    def test_empty_label_sets_raises(self):
        """All empty label sets means no types discovered."""
        rng = np.random.RandomState(42)
        X = rng.randn(10, 8).astype(np.float32)
        label_sets: list[set[str]] = [set()] * 10

        with pytest.raises(ValueError, match="No vocalization types"):
            train_multilabel_classifiers(X, label_sets)

    def test_threshold_range(self):
        """Thresholds should be between 0 and 1."""
        X, label_sets = _make_synthetic_data()
        _, thresholds, _ = train_multilabel_classifiers(X, label_sets)

        for t in thresholds.values():
            assert 0.0 < t < 1.0

    def test_inference_produces_probabilities(self):
        """Each trained pipeline's predict_proba returns probabilities."""
        X, label_sets = _make_synthetic_data()
        pipelines, _, _ = train_multilabel_classifiers(X, label_sets)

        test_vec = X[:5]
        for name, pipeline in pipelines.items():
            probs = pipeline.predict_proba(test_vec)
            assert probs.shape == (5, 2)  # binary: [neg_prob, pos_prob]
            assert np.all(probs >= 0) and np.all(probs <= 1)


class TestSaveModelArtifacts:
    def test_save_and_load(self, tmp_path):
        """Save model artifacts and verify structure."""
        X, label_sets = _make_synthetic_data()
        pipelines, thresholds, metrics = train_multilabel_classifiers(X, label_sets)

        model_dir = tmp_path / "model"
        save_model_artifacts(
            model_dir, pipelines, thresholds, metrics, parameters={"C": 1.0}
        )

        # Check files exist
        assert (model_dir / "whup.joblib").exists()
        assert (model_dir / "moan.joblib").exists()
        assert (model_dir / "shriek.joblib").exists()
        assert (model_dir / "metadata.json").exists()

        # Check metadata
        metadata = json.loads((model_dir / "metadata.json").read_text())
        assert sorted(metadata["vocabulary"]) == ["moan", "shriek", "whup"]
        assert "whup" in metadata["thresholds"]
        assert metadata["parameters"]["C"] == 1.0

        # Check loaded pipeline works
        import joblib

        loaded = joblib.load(model_dir / "whup.joblib")
        probs = loaded.predict_proba(X[:3])
        assert probs.shape == (3, 2)
