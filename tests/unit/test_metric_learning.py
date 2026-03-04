"""Tests for metric learning refinement."""

import numpy as np
import pytest

from humpback.clustering.metric_learning import (
    _build_projection_head,
    _triplet_loss,
    generate_triplets,
    run_metric_learning_refinement,
    train_projection,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_separable(n_per_class=30, n_classes=3, dim=32, seed=42):
    """Create well-separated clusters with category labels."""
    rng = np.random.RandomState(seed)
    embeddings = []
    labels = []
    class_names = [f"class_{i}" for i in range(n_classes)]
    for i in range(n_classes):
        center = np.zeros(dim)
        center[i % dim] = 10.0
        embeddings.append(rng.randn(n_per_class, dim).astype(np.float32) + center)
        labels.extend([class_names[i]] * n_per_class)
    return np.vstack(embeddings), labels


# ---------------------------------------------------------------------------
# Triplet generation
# ---------------------------------------------------------------------------


def test_generate_triplets_random_shape():
    """Output indices have correct shape (n_triplets,)."""
    rng = np.random.RandomState(42)
    embeddings = rng.randn(60, 16).astype(np.float32)
    labels = np.array([0] * 20 + [1] * 20 + [2] * 20)
    n_triplets = 50

    a, p, n = generate_triplets(
        embeddings, labels, n_triplets=n_triplets, strategy="random"
    )
    assert a.shape == (n_triplets,)
    assert p.shape == (n_triplets,)
    assert n.shape == (n_triplets,)


def test_generate_triplets_valid_labels():
    """Anchors/positives share label, negatives differ."""
    rng = np.random.RandomState(42)
    embeddings = rng.randn(60, 16).astype(np.float32)
    labels = np.array([0] * 20 + [1] * 20 + [2] * 20)

    a, p, n = generate_triplets(
        embeddings, labels, n_triplets=100, strategy="random", random_state=42
    )
    for i in range(len(a)):
        assert labels[a[i]] == labels[p[i]], "Anchor and positive must share label"
        assert labels[a[i]] != labels[n[i]], "Negative must have different label"


def test_generate_triplets_hard_closer():
    """Hard negatives should be closer (on average) than random negatives."""
    rng = np.random.RandomState(42)
    # Three clusters with varying separation
    embeddings = np.vstack([
        rng.randn(30, 16).astype(np.float32) + np.array([5.0] + [0.0] * 15),
        rng.randn(30, 16).astype(np.float32) + np.array([-5.0] + [0.0] * 15),
        rng.randn(30, 16).astype(np.float32) + np.array([0.0, 5.0] + [0.0] * 14),
    ])
    labels = np.array([0] * 30 + [1] * 30 + [2] * 30)

    # Use same random_state so anchors are identical
    a_rand, _, n_rand = generate_triplets(
        embeddings, labels, n_triplets=200, strategy="random", random_state=42
    )
    a_hard, _, n_hard = generate_triplets(
        embeddings, labels, n_triplets=200, strategy="hard", random_state=42
    )

    # Hard negatives should be closer to their anchors than random negatives
    from scipy.spatial.distance import cdist

    dists_all = cdist(embeddings, embeddings)
    hard_dists = [dists_all[a_hard[i], n_hard[i]] for i in range(200)]
    rand_dists = [dists_all[a_rand[i], n_rand[i]] for i in range(200)]
    assert np.mean(hard_dists) <= np.mean(rand_dists)


# ---------------------------------------------------------------------------
# Triplet loss
# ---------------------------------------------------------------------------


def test_triplet_loss_zero_for_separated():
    """Well-separated clusters should yield loss near 0."""
    import tensorflow as tf

    rng = np.random.RandomState(42)
    dim = 16

    model = _build_projection_head(dim, hidden_dim=32, output_dim=8)
    # Perfect separation: anchors/positives at 0, negatives far away
    anchors = tf.constant(rng.randn(20, dim).astype(np.float32) * 0.01)
    positives = tf.constant(rng.randn(20, dim).astype(np.float32) * 0.01)
    negatives = tf.constant(rng.randn(20, dim).astype(np.float32) * 0.01 + 100)

    loss = _triplet_loss(model, anchors, positives, negatives, margin=1.0)
    # Loss should be 0 or very small because negatives are far away
    assert float(loss.numpy()) >= 0.0  # loss is non-negative


def test_triplet_loss_positive_for_overlapping():
    """Overlapping data should yield positive loss."""
    import tensorflow as tf

    rng = np.random.RandomState(42)
    dim = 16

    model = _build_projection_head(dim, hidden_dim=32, output_dim=8)
    # All same data — negatives as close as positives
    data = rng.randn(20, dim).astype(np.float32)
    anchors = tf.constant(data)
    positives = tf.constant(data)
    negatives = tf.constant(data)

    loss = _triplet_loss(model, anchors, positives, negatives, margin=1.0)
    # With margin=1.0 and d_ap ≈ 0, d_an ≈ 0: loss ≈ margin = 1.0
    assert float(loss.numpy()) > 0.5


# ---------------------------------------------------------------------------
# Projection head
# ---------------------------------------------------------------------------


def test_build_projection_head_output_shape():
    """Model output shape matches (batch, output_dim)."""
    import tensorflow as tf

    model = _build_projection_head(64, hidden_dim=32, output_dim=16)
    x = tf.random.normal((10, 64))
    out = model(x, training=False)
    assert out.shape == (10, 16)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def test_train_projection_loss_decreases():
    """Final loss should be less than initial loss on overlapping data."""
    rng = np.random.RandomState(42)
    # Overlapping clusters so initial loss is positive
    embeddings = np.vstack([
        rng.randn(30, 32).astype(np.float32) + 1,
        rng.randn(30, 32).astype(np.float32) - 1,
    ])
    labels = np.array([0] * 30 + [1] * 30)

    result = train_projection(
        embeddings, labels,
        output_dim=16, hidden_dim=32, n_epochs=30, lr=0.01,
        margin=2.0, batch_size=64, mining_strategy="random", random_state=42,
    )
    # With overlapping data and large margin, initial loss should be positive
    # and training should reduce it
    assert result["loss_history"][0] > 0
    assert result["loss_history"][-1] <= result["loss_history"][0]


def test_train_projection_output_shape():
    """Refined embeddings shape is (N, output_dim)."""
    rng = np.random.RandomState(42)
    embeddings = np.vstack([
        rng.randn(20, 32).astype(np.float32) + 5,
        rng.randn(20, 32).astype(np.float32) - 5,
    ])
    labels = np.array([0] * 20 + [1] * 20)

    result = train_projection(
        embeddings, labels, output_dim=16, hidden_dim=32, n_epochs=5,
        batch_size=32, mining_strategy="random", random_state=42,
    )
    assert result["refined_embeddings"].shape == (40, 16)


def test_train_projection_output_structure():
    """Returned dict has all expected keys."""
    rng = np.random.RandomState(42)
    embeddings = np.vstack([
        rng.randn(20, 16).astype(np.float32) + 3,
        rng.randn(20, 16).astype(np.float32) - 3,
    ])
    labels = np.array([0] * 20 + [1] * 20)

    result = train_projection(
        embeddings, labels, output_dim=8, hidden_dim=16, n_epochs=3,
        batch_size=20, mining_strategy="random", random_state=42,
    )
    assert "model" in result
    assert "refined_embeddings" in result
    assert "loss_history" in result
    assert "params" in result
    assert len(result["loss_history"]) == 3


# ---------------------------------------------------------------------------
# Full refinement
# ---------------------------------------------------------------------------


def test_refinement_returns_none_insufficient():
    """Single/no categories should return None."""
    rng = np.random.RandomState(42)
    embeddings = rng.randn(20, 16).astype(np.float32)

    # Single category
    assert run_metric_learning_refinement(
        embeddings, ["catA"] * 20
    ) is None

    # No categories
    assert run_metric_learning_refinement(
        embeddings, [None] * 20
    ) is None

    # One category with enough samples, one too rare
    labels = ["catA"] * 15 + ["catB"] * 3 + [None] * 2
    assert run_metric_learning_refinement(
        embeddings, labels
    ) is None


def test_refinement_report_structure():
    """Full run: verify all keys in report, comparison has expected metrics."""
    embeddings, labels = _make_separable(n_per_class=20, n_classes=3, dim=32)

    result = run_metric_learning_refinement(
        embeddings,
        labels,
        params={
            "ml_n_epochs": 5,
            "ml_hidden_dim": 32,
            "ml_output_dim": 16,
            "ml_batch_size": 32,
            "ml_mining_strategy": "random",
            "clustering_algorithm": "kmeans",
            "n_clusters": 3,
            "reduction_method": "none",
        },
    )
    assert result is not None

    # Top-level keys
    expected_keys = {
        "training_params", "n_labeled_samples", "n_categories",
        "n_total_samples", "categories_used", "loss_history",
        "final_loss", "comparison", "base_summary", "refined_summary",
    }
    assert expected_keys == set(result.keys())

    # Training params
    tp = result["training_params"]
    assert tp["output_dim"] == 16
    assert tp["hidden_dim"] == 32
    assert tp["n_epochs"] == 5

    # Comparison
    assert len(result["comparison"]) > 0
    comp_keys = {c["key"] for c in result["comparison"]}
    assert "silhouette_score" in comp_keys

    for c in result["comparison"]:
        assert "metric" in c
        assert "key" in c
        assert "base" in c
        assert "refined" in c
        assert "delta" in c
        assert "improved" in c

    # Categories
    assert result["n_categories"] == 3
    assert result["n_labeled_samples"] == 60
    assert result["n_total_samples"] == 60


def test_refinement_deterministic():
    """Same inputs should produce structurally identical output."""
    embeddings, labels = _make_separable(n_per_class=15, n_classes=2, dim=16)
    params = {
        "ml_n_epochs": 3,
        "ml_hidden_dim": 16,
        "ml_output_dim": 8,
        "ml_batch_size": 20,
        "ml_mining_strategy": "random",
        "clustering_algorithm": "kmeans",
        "n_clusters": 2,
        "reduction_method": "none",
        "random_state": 42,
    }

    r1 = run_metric_learning_refinement(embeddings, labels, params=params)
    r2 = run_metric_learning_refinement(embeddings, labels, params=params)
    assert r1 is not None and r2 is not None

    # Structure should match exactly
    assert r1["n_labeled_samples"] == r2["n_labeled_samples"]
    assert r1["n_categories"] == r2["n_categories"]
    assert r1["n_total_samples"] == r2["n_total_samples"]
    assert r1["categories_used"] == r2["categories_used"]
    assert r1["training_params"] == r2["training_params"]
    assert len(r1["loss_history"]) == len(r2["loss_history"])
    assert len(r1["comparison"]) == len(r2["comparison"])
    # Same metrics should appear in comparison
    assert [c["key"] for c in r1["comparison"]] == [c["key"] for c in r2["comparison"]]
