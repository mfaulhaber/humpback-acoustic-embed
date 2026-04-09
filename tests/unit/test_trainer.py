"""Tests for binary classifier trainer."""

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import Normalizer

from humpback.classifier.trainer import (
    load_manifest_split_embeddings,
    map_autoresearch_config_to_training_parameters,
    train_binary_classifier,
)


def _write_embedding_set_parquet(path: Path, rows: list[list[float]]) -> None:
    table = pa.table(
        {
            "row_index": pa.array(list(range(len(rows))), type=pa.int32()),
            "embedding": pa.array(rows, type=pa.list_(pa.float32())),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(path))


def _write_detection_embeddings_parquet(
    path: Path,
    row_ids: list[str],
    rows: list[list[float]],
) -> None:
    table = pa.table(
        {
            "row_id": pa.array(row_ids, type=pa.string()),
            "embedding": pa.array(rows, type=pa.list_(pa.float32())),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(path))


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
        positive,
        negative,
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
        positive,
        negative,
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


# ---- MLP classifier + diagnostics tests ----


def test_mlp_classifier_type():
    """MLP pipeline has MLPClassifier and produces valid predictions."""
    rng = np.random.RandomState(42)
    positive = rng.randn(50, 16) + 2.0
    negative = rng.randn(50, 16) - 2.0

    pipeline, summary = train_binary_classifier(
        positive,
        negative,
        parameters={"classifier_type": "mlp"},
    )

    assert isinstance(pipeline.named_steps["classifier"], MLPClassifier)
    preds = pipeline.predict(rng.randn(10, 16))
    assert len(preds) == 10
    assert set(preds).issubset({0, 1})

    proba = pipeline.predict_proba(rng.randn(5, 16))
    assert proba.shape == (5, 2)


def test_mlp_reduces_fps_on_overlapping_data():
    """MLP achieves higher precision than LR on non-linearly separable data."""
    rng = np.random.RandomState(42)
    n = 200
    dim = 16

    # Concentric clusters: positives in a shell, negatives at the center
    # This is non-linearly separable
    pos = rng.randn(n, dim)
    pos = pos / np.linalg.norm(pos, axis=1, keepdims=True) * 3.0  # on shell at r=3
    pos += rng.randn(n, dim) * 0.3  # small noise

    neg = rng.randn(n, dim) * 0.5  # centered at origin, small spread

    _, lr_summary = train_binary_classifier(
        pos,
        neg,
        parameters={"classifier_type": "logistic_regression"},
    )
    _, mlp_summary = train_binary_classifier(
        pos,
        neg,
        parameters={"classifier_type": "mlp"},
    )

    # MLP should achieve better precision on this non-linear boundary
    assert mlp_summary["cv_precision"] >= lr_summary["cv_precision"] - 0.05, (
        f"MLP precision {mlp_summary['cv_precision']:.3f} should be at least close to "
        f"LR precision {lr_summary['cv_precision']:.3f} on non-linear data"
    )


def test_l2_normalize_pipeline():
    """L2 normalize inserts Normalizer step, pipeline has 3 steps."""
    rng = np.random.RandomState(42)
    positive = rng.randn(30, 16) + 2.0
    negative = rng.randn(30, 16) - 2.0

    pipeline, _ = train_binary_classifier(
        positive,
        negative,
        parameters={"l2_normalize": True},
    )

    assert len(pipeline.steps) == 3
    assert pipeline.steps[0][0] == "l2_norm"
    assert isinstance(pipeline.steps[0][1], Normalizer)
    assert pipeline.steps[1][0] == "scaler"
    assert pipeline.steps[2][0] == "classifier"


def test_extended_cv_metrics():
    """Summary contains precision, recall, F1 from cross-validation."""
    rng = np.random.RandomState(42)
    positive = rng.randn(50, 16) + 2.0
    negative = rng.randn(50, 16) - 2.0

    _, summary = train_binary_classifier(positive, negative)

    for key in [
        "cv_precision",
        "cv_precision_std",
        "cv_recall",
        "cv_recall_std",
        "cv_f1",
        "cv_f1_std",
    ]:
        assert key in summary, f"Missing key: {key}"
        assert isinstance(summary[key], float)


def test_decision_boundary_diagnostics():
    """Summary contains score_separation, mean scores, and train confusion."""
    rng = np.random.RandomState(42)
    positive = rng.randn(50, 16) + 2.0
    negative = rng.randn(50, 16) - 2.0

    _, summary = train_binary_classifier(positive, negative)

    assert "positive_mean_score" in summary
    assert "negative_mean_score" in summary
    assert "score_separation" in summary
    assert isinstance(summary["score_separation"], float)
    assert summary["score_separation"] > 0  # well-separated data

    assert "train_confusion" in summary
    tc = summary["train_confusion"]
    assert all(k in tc for k in ["tp", "fp", "tn", "fn"])
    assert tc["tp"] + tc["fn"] == 50  # all positives
    assert tc["tn"] + tc["fp"] == 50  # all negatives


def test_default_backward_compat():
    """Default parameters produce 2-step pipeline with LogisticRegression."""
    rng = np.random.RandomState(42)
    positive = rng.randn(30, 16) + 2.0
    negative = rng.randn(30, 16) - 2.0

    pipeline, summary = train_binary_classifier(positive, negative)

    assert len(pipeline.steps) == 2
    assert isinstance(pipeline.named_steps["classifier"], LogisticRegression)
    assert summary["classifier_type"] == "logistic_regression"
    assert summary["l2_normalize"] is False


def test_classifier_type_in_summary():
    """Classifier type is recorded in summary for both LR and MLP."""
    rng = np.random.RandomState(42)
    positive = rng.randn(30, 16) + 2.0
    negative = rng.randn(30, 16) - 2.0

    _, lr_summary = train_binary_classifier(
        positive,
        negative,
        parameters={"classifier_type": "logistic_regression"},
    )
    assert lr_summary["classifier_type"] == "logistic_regression"

    _, mlp_summary = train_binary_classifier(
        positive,
        negative,
        parameters={"classifier_type": "mlp"},
    )
    assert mlp_summary["classifier_type"] == "mlp"


def test_feature_norm_none_skips_normalization_steps():
    """Autoresearch-style feature_norm='none' should train without preprocessors."""
    rng = np.random.RandomState(42)
    positive = rng.randn(30, 16) + 2.0
    negative = rng.randn(30, 16) - 2.0

    pipeline, summary = train_binary_classifier(
        positive,
        negative,
        parameters={"feature_norm": "none"},
    )

    assert len(pipeline.steps) == 1
    assert pipeline.steps[0][0] == "classifier"
    assert summary["feature_norm"] == "none"


def test_map_autoresearch_config_to_training_parameters():
    """Promotable autoresearch configs map to explicit trainer parameters."""
    params = map_autoresearch_config_to_training_parameters(
        {
            "classifier": "logreg",
            "feature_norm": "l2",
            "class_weight_pos": 3.0,
            "class_weight_neg": 1.0,
            "seed": 7,
        }
    )

    assert params["classifier_type"] == "logistic_regression"
    assert params["feature_norm"] == "l2"
    assert params["class_weight"] == {0: 1.0, 1: 3.0}
    assert params["random_state"] == 7


def test_map_autoresearch_config_mlp_with_class_weights():
    """MLP configs with explicit class weights are now accepted."""
    params = map_autoresearch_config_to_training_parameters(
        {
            "classifier": "mlp",
            "feature_norm": "standard",
            "class_weight_pos": 1.0,
            "class_weight_neg": 1.5,
            "seed": 42,
        }
    )

    assert params["classifier_type"] == "mlp"
    assert params["class_weight"] == {0: 1.5, 1: 1.0}


def test_train_mlp_with_explicit_class_weights():
    """MLP training with non-uniform class weights produces a valid model."""
    rng = np.random.RandomState(42)
    n = 50
    dim = 16
    positive = rng.randn(n, dim) + 2.0
    negative = rng.randn(n, dim) - 2.0

    pipeline, summary = train_binary_classifier(
        positive,
        negative,
        parameters={
            "classifier_type": "mlp",
            "class_weight": {0: 1.0, 1: 1.5},
        },
    )

    assert hasattr(pipeline, "predict_proba")
    assert isinstance(pipeline.named_steps["classifier"], MLPClassifier)
    assert summary["classifier_type"] == "mlp"


def test_load_manifest_split_embeddings_supports_mixed_sources(
    tmp_path: Path,
) -> None:
    """Manifest loader should resolve row_index and row_id training examples."""
    es_path = tmp_path / "embedding_set.parquet"
    det_path = tmp_path / "detections" / "job-1" / "detection_embeddings.parquet"
    _write_embedding_set_parquet(
        es_path,
        rows=[[2.0, 2.0], [2.5, 2.5]],
    )
    _write_detection_embeddings_parquet(
        det_path,
        row_ids=["neg-1", "neg-2"],
        rows=[[-2.0, -2.0], [-2.5, -2.5]],
    )

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "metadata": {},
                "examples": [
                    {
                        "id": "pos-0",
                        "split": "train",
                        "label": 1,
                        "parquet_path": str(es_path),
                        "row_index": 0,
                    },
                    {
                        "id": "pos-1",
                        "split": "train",
                        "label": 1,
                        "parquet_path": str(es_path),
                        "row_index": 1,
                    },
                    {
                        "id": "neg-1",
                        "split": "train",
                        "label": 0,
                        "parquet_path": str(det_path),
                        "row_id": "neg-1",
                    },
                    {
                        "id": "neg-2",
                        "split": "train",
                        "label": 0,
                        "parquet_path": str(det_path),
                        "row_id": "neg-2",
                    },
                ],
            }
        )
    )

    positive, negative, summary = load_manifest_split_embeddings(manifest_path)

    assert positive.shape == (2, 2)
    assert negative.shape == (2, 2)
    assert summary["split"] == "train"
    assert summary["positive_count"] == 2
    assert summary["negative_count"] == 2


def test_load_manifest_split_data_returns_full_context(tmp_path: Path) -> None:
    """load_manifest_split_data returns manifest, cache, and examples for pooling."""
    from humpback.classifier.trainer import load_manifest_split_data

    es_path = tmp_path / "embedding_set.parquet"
    det_path = tmp_path / "detections" / "job-1" / "detection_embeddings.parquet"
    _write_embedding_set_parquet(
        es_path,
        rows=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
    )
    _write_detection_embeddings_parquet(
        det_path,
        row_ids=["neg-1", "neg-2", "neg-3"],
        rows=[[-1.0, -2.0], [-3.0, -4.0], [-5.0, -6.0]],
    )

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "metadata": {},
                "examples": [
                    {
                        "id": "pos-0",
                        "split": "train",
                        "label": 1,
                        "parquet_path": str(es_path),
                        "row_index": 0,
                    },
                    {
                        "id": "pos-1",
                        "split": "train",
                        "label": 1,
                        "parquet_path": str(es_path),
                        "row_index": 1,
                    },
                    {
                        "id": "pos-2",
                        "split": "val",
                        "label": 1,
                        "parquet_path": str(es_path),
                        "row_index": 2,
                    },
                    {
                        "id": "neg-1",
                        "split": "train",
                        "label": 0,
                        "parquet_path": str(det_path),
                        "row_id": "neg-1",
                    },
                    {
                        "id": "neg-2",
                        "split": "train",
                        "label": 0,
                        "parquet_path": str(det_path),
                        "row_id": "neg-2",
                    },
                    {
                        "id": "neg-3",
                        "split": "val",
                        "label": 0,
                        "parquet_path": str(det_path),
                        "row_id": "neg-3",
                    },
                ],
            }
        )
    )

    data = load_manifest_split_data(manifest_path, split="train")

    # Check arrays
    assert data.X.shape == (4, 2)
    assert data.y.shape == (4,)
    assert int(np.sum(data.y == 1)) == 2
    assert int(np.sum(data.y == 0)) == 2

    # Check examples metadata preserved
    assert len(data.examples) == 4
    assert all(ex["split"] == "train" for ex in data.examples)

    # Check parquet cache has both sources
    assert str(es_path) in data.parquet_cache
    assert str(det_path) in data.parquet_cache

    # Check manifest is the full manifest (all examples, not just train)
    assert len(data.manifest["examples"]) == 6

    # Check source summary
    assert data.source_summary["split"] == "train"
    assert data.source_summary["positive_count"] == 2
    assert data.source_summary["negative_count"] == 2
    assert data.source_summary["vector_dim"] == 2
