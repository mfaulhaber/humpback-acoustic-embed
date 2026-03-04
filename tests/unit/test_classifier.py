"""Tests for classifier baseline and active learning queue."""

import numpy as np
import pytest

from humpback.clustering.classifier import run_classifier_baseline


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
# Core behavior
# ---------------------------------------------------------------------------


def test_perfect_classification():
    """Well-separated clusters should yield near-perfect accuracy."""
    embeddings, labels = _make_separable(n_per_class=30, n_classes=3)
    result = run_classifier_baseline(embeddings, labels)
    assert result is not None
    report = result["classifier_report"]
    assert report["overall_accuracy"] >= 0.95


def test_returns_none_single_category():
    """All same category should return None (< 2 categories)."""
    rng = np.random.RandomState(42)
    embeddings = rng.randn(20, 16).astype(np.float32)
    labels = ["catA"] * 20
    result = run_classifier_baseline(embeddings, labels)
    assert result is None


def test_returns_none_no_categories():
    """All None categories should return None."""
    rng = np.random.RandomState(42)
    embeddings = rng.randn(20, 16).astype(np.float32)
    labels = [None] * 20
    result = run_classifier_baseline(embeddings, labels)
    assert result is None


def test_filters_rare_categories():
    """Category with fewer than n_folds samples should be excluded."""
    rng = np.random.RandomState(42)
    # Two good categories + one rare
    embeddings = rng.randn(41, 16).astype(np.float32)
    labels = ["catA"] * 20 + ["catB"] * 20 + ["RareCall"]
    result = run_classifier_baseline(embeddings, labels, n_folds=5)
    assert result is not None
    assert "RareCall" in result["classifier_report"]["categories_excluded"]
    assert "RareCall" not in result["classifier_report"]["per_class"]


def test_report_structure():
    """Classifier report should contain all expected top-level keys."""
    embeddings, labels = _make_separable(n_per_class=20, n_classes=2)
    result = run_classifier_baseline(embeddings, labels)
    assert result is not None
    report = result["classifier_report"]
    expected_keys = {
        "n_samples", "n_categories", "n_folds", "categories_excluded",
        "overall_accuracy", "per_class", "macro_avg", "weighted_avg",
        "confusion_matrix",
    }
    assert expected_keys == set(report.keys())


def test_per_class_has_expected_keys():
    """Each per-class entry should have precision, recall, f1_score, support."""
    embeddings, labels = _make_separable(n_per_class=20, n_classes=2)
    result = run_classifier_baseline(embeddings, labels)
    assert result is not None
    for cat, vals in result["classifier_report"]["per_class"].items():
        assert "precision" in vals
        assert "recall" in vals
        assert "f1_score" in vals
        assert "support" in vals
        assert isinstance(vals["support"], int)


def test_confusion_matrix_structure():
    """Confusion matrix rows should sum to support for each category."""
    embeddings, labels = _make_separable(n_per_class=20, n_classes=3)
    result = run_classifier_baseline(embeddings, labels)
    assert result is not None
    report = result["classifier_report"]
    cm = report["confusion_matrix"]
    for cat, row in cm.items():
        row_sum = sum(row.values())
        assert row_sum == report["per_class"][cat]["support"]


def test_label_queue_structure():
    """Each label queue entry should have all expected keys."""
    embeddings, labels = _make_separable(n_per_class=10, n_classes=2)
    result = run_classifier_baseline(embeddings, labels)
    assert result is not None
    expected_keys = {
        "rank", "global_index", "embedding_set_id", "embedding_row_index",
        "current_category", "predicted_category", "entropy", "margin",
        "max_prob", "fragmentation_boost", "priority",
    }
    for entry in result["label_queue"]:
        assert expected_keys == set(entry.keys())


def test_label_queue_sorted_by_priority():
    """Label queue should be sorted by priority descending."""
    embeddings, labels = _make_separable(n_per_class=15, n_classes=2)
    result = run_classifier_baseline(embeddings, labels)
    assert result is not None
    queue = result["label_queue"]
    priorities = [e["priority"] for e in queue]
    assert priorities == sorted(priorities, reverse=True)
    # Ranks should be sequential
    ranks = [e["rank"] for e in queue]
    assert ranks == list(range(1, len(ranks) + 1))


def test_unlabeled_samples_max_priority():
    """Unlabeled samples (None categories) should get priority=1.0."""
    rng = np.random.RandomState(42)
    embeddings = rng.randn(25, 16).astype(np.float32)
    labels = ["catA"] * 10 + ["catB"] * 10 + [None] * 5
    result = run_classifier_baseline(embeddings, labels)
    assert result is not None
    queue = result["label_queue"]
    for entry in queue:
        if entry["current_category"] is None:
            assert entry["priority"] == 1.0
            assert entry["predicted_category"] is None
            assert entry["entropy"] is None


def test_fragmentation_boost():
    """High fragmentation category should yield higher priority than low frag."""
    rng = np.random.RandomState(42)
    embeddings = rng.randn(30, 16).astype(np.float32)
    labels = ["catA"] * 15 + ["catB"] * 15

    frag_report = {
        "category_fragmentation": {
            "catA": {"normalized_entropy": 0.9},  # high fragmentation
            "catB": {"normalized_entropy": 0.1},  # low fragmentation
        }
    }

    result = run_classifier_baseline(embeddings, labels, frag_report=frag_report)
    assert result is not None
    queue = result["label_queue"]

    # Average priority for catA should be higher than catB
    catA_priorities = [e["priority"] for e in queue if e["current_category"] == "catA"]
    catB_priorities = [e["priority"] for e in queue if e["current_category"] == "catB"]
    assert np.mean(catA_priorities) > np.mean(catB_priorities)


def test_deterministic_results():
    """Same random_state should produce identical output."""
    embeddings, labels = _make_separable(n_per_class=15, n_classes=3)
    r1 = run_classifier_baseline(embeddings, labels, random_state=42)
    r2 = run_classifier_baseline(embeddings, labels, random_state=42)
    assert r1 is not None and r2 is not None
    assert r1["classifier_report"] == r2["classifier_report"]
    assert r1["label_queue"] == r2["label_queue"]
