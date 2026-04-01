"""Unit tests for autoresearch modules."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.autoresearch.objectives import default_objective, get_objective
from scripts.autoresearch.search_space import (
    SEARCH_SPACE,
    config_hash,
    sample_config,
)
from scripts.autoresearch.train_eval import (
    apply_context_pooling,
    apply_transforms,
    build_classifier,
    build_feature_pipeline,
    compute_metrics,
    find_top_false_positives,
    train_eval,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VECTOR_DIM = 16


def _write_synthetic_parquet(path: Path, n_rows: int, dim: int = VECTOR_DIM) -> None:
    """Write a synthetic Parquet file with sequential embeddings."""
    rng = np.random.RandomState(42)
    schema = pa.schema(
        [
            ("row_index", pa.int32()),
            ("embedding", pa.list_(pa.float32(), dim)),
        ]
    )
    indices = list(range(n_rows))
    embeddings = [rng.randn(dim).astype(np.float32).tolist() for _ in range(n_rows)]
    table = pa.table({"row_index": indices, "embedding": embeddings}, schema=schema)
    pq.write_table(table, str(path))


def _make_manifest(
    parquet_path: str,
    n_pos: int = 10,
    n_neg: int = 10,
    dim: int = VECTOR_DIM,
) -> dict[str, Any]:
    """Create a minimal manifest for testing."""
    examples = []
    for i in range(n_pos):
        examples.append(
            {
                "id": f"es1_row{i}",
                "split": "train" if i < n_pos * 7 // 10 else "val",
                "label": 1,
                "parquet_path": parquet_path,
                "row_index": i,
                "audio_file_id": "file1",
                "negative_group": None,
            }
        )
    for i in range(n_neg):
        ri = n_pos + i
        examples.append(
            {
                "id": f"es2_row{ri}",
                "split": "train" if i < n_neg * 7 // 10 else "val",
                "label": 0,
                "parquet_path": parquet_path,
                "row_index": ri,
                "audio_file_id": "file2",
                "negative_group": "vessel" if i % 2 == 0 else None,
            }
        )
    return {
        "metadata": {
            "created_at": "2026-04-01T00:00:00Z",
            "source_job_ids": ["1"],
            "positive_embedding_set_ids": ["1"],
            "negative_embedding_set_ids": ["2"],
            "split_strategy": "by_audio_file",
        },
        "examples": examples,
    }


# ---------------------------------------------------------------------------
# Objectives tests
# ---------------------------------------------------------------------------


class TestObjectives:
    def test_default_objective_formula(self) -> None:
        metrics = {"recall": 0.9, "high_conf_fp_rate": 0.01, "fp_rate": 0.05}
        result = default_objective(metrics)
        expected = 0.9 - 15.0 * 0.01 - 3.0 * 0.05
        assert abs(result - expected) < 1e-9

    def test_default_objective_penalizes_high_conf_fp(self) -> None:
        good = {"recall": 0.9, "high_conf_fp_rate": 0.0, "fp_rate": 0.05}
        bad = {"recall": 0.9, "high_conf_fp_rate": 0.1, "fp_rate": 0.05}
        assert default_objective(good) > default_objective(bad)

    def test_get_objective_default(self) -> None:
        fn = get_objective("default")
        assert fn is default_objective

    def test_get_objective_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown objective"):
            get_objective("nonexistent")


# ---------------------------------------------------------------------------
# Search space tests
# ---------------------------------------------------------------------------


class TestSearchSpace:
    def test_all_dimensions_present(self) -> None:
        expected = {
            "feature_norm",
            "pca_dim",
            "classifier",
            "class_weight_pos",
            "class_weight_neg",
            "hard_negative_fraction",
            "prob_calibration",
            "threshold",
            "context_pooling",
        }
        assert set(SEARCH_SPACE.keys()) == expected

    def test_sample_config_has_all_keys(self) -> None:
        rng = random.Random(42)
        config = sample_config(rng)
        assert set(config.keys()) == set(SEARCH_SPACE.keys())

    def test_sample_config_values_in_space(self) -> None:
        rng = random.Random(42)
        for _ in range(20):
            config = sample_config(rng)
            for key, val in config.items():
                assert val in SEARCH_SPACE[key], f"{key}={val} not in space"

    def test_sample_config_deterministic(self) -> None:
        c1 = sample_config(random.Random(123))
        c2 = sample_config(random.Random(123))
        assert c1 == c2

    def test_config_hash_deterministic(self) -> None:
        config = {"a": 1, "b": "x"}
        assert config_hash(config) == config_hash(config)

    def test_config_hash_differs(self) -> None:
        c1 = {"a": 1, "b": "x"}
        c2 = {"a": 2, "b": "x"}
        assert config_hash(c1) != config_hash(c2)


# ---------------------------------------------------------------------------
# Context pooling tests
# ---------------------------------------------------------------------------


class TestContextPooling:
    def test_center_passthrough(self, tmp_path: Path) -> None:
        parquet = tmp_path / "test.parquet"
        _write_synthetic_parquet(parquet, 5)

        manifest = _make_manifest(str(parquet), n_pos=3, n_neg=2)
        from humpback.processing.embeddings import read_embeddings

        ri, emb = read_embeddings(parquet)
        lookup = {ex["id"]: emb[ex["row_index"]] for ex in manifest["examples"]}
        cache = {str(parquet): (ri, emb)}

        result = apply_context_pooling(manifest, lookup, cache, "center")
        for eid, vec in result.items():
            np.testing.assert_array_equal(vec, lookup[eid])

    def test_mean3_averages_neighbors(self, tmp_path: Path) -> None:
        parquet = tmp_path / "test.parquet"
        _write_synthetic_parquet(parquet, 5)

        from humpback.processing.embeddings import read_embeddings

        ri, emb = read_embeddings(parquet)
        # Create manifest with a single mid-row example
        manifest = {
            "metadata": {},
            "examples": [
                {
                    "id": "e2",
                    "split": "train",
                    "label": 1,
                    "parquet_path": str(parquet),
                    "row_index": 2,
                    "audio_file_id": "f1",
                    "negative_group": None,
                },
            ],
        }
        lookup = {"e2": emb[2]}
        cache = {str(parquet): (ri, emb)}

        result = apply_context_pooling(manifest, lookup, cache, "mean3")
        expected = np.mean([emb[1], emb[2], emb[3]], axis=0).astype(np.float32)
        np.testing.assert_allclose(result["e2"], expected, atol=1e-6)

    def test_max3_takes_max(self, tmp_path: Path) -> None:
        parquet = tmp_path / "test.parquet"
        _write_synthetic_parquet(parquet, 5)

        from humpback.processing.embeddings import read_embeddings

        ri, emb = read_embeddings(parquet)
        manifest = {
            "metadata": {},
            "examples": [
                {
                    "id": "e2",
                    "split": "train",
                    "label": 1,
                    "parquet_path": str(parquet),
                    "row_index": 2,
                    "audio_file_id": "f1",
                    "negative_group": None,
                },
            ],
        }
        lookup = {"e2": emb[2]}
        cache = {str(parquet): (ri, emb)}

        result = apply_context_pooling(manifest, lookup, cache, "max3")
        expected = np.max([emb[1], emb[2], emb[3]], axis=0).astype(np.float32)
        np.testing.assert_allclose(result["e2"], expected, atol=1e-6)

    def test_boundary_fallback(self, tmp_path: Path) -> None:
        """First row has no left neighbor — should still work."""
        parquet = tmp_path / "test.parquet"
        _write_synthetic_parquet(parquet, 3)

        from humpback.processing.embeddings import read_embeddings

        ri, emb = read_embeddings(parquet)
        manifest = {
            "metadata": {},
            "examples": [
                {
                    "id": "e0",
                    "split": "train",
                    "label": 1,
                    "parquet_path": str(parquet),
                    "row_index": 0,
                    "audio_file_id": "f1",
                    "negative_group": None,
                },
            ],
        }
        lookup = {"e0": emb[0]}
        cache = {str(parquet): (ri, emb)}

        result = apply_context_pooling(manifest, lookup, cache, "mean3")
        # Only center + right neighbor (no left)
        expected = np.mean([emb[0], emb[1]], axis=0).astype(np.float32)
        np.testing.assert_allclose(result["e0"], expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Feature transform tests
# ---------------------------------------------------------------------------


class TestFeatureTransforms:
    def test_l2_produces_unit_vectors(self) -> None:
        rng = np.random.RandomState(42)
        X = rng.randn(20, VECTOR_DIM).astype(np.float32)
        config: dict[str, Any] = {"feature_norm": "l2"}
        transforms, X_out = build_feature_pipeline(config, X)
        norms = np.linalg.norm(X_out, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)
        assert len(transforms) == 1

    def test_standard_zero_mean(self) -> None:
        rng = np.random.RandomState(42)
        X = rng.randn(50, VECTOR_DIM).astype(np.float32) + 5.0
        config: dict[str, Any] = {"feature_norm": "standard"}
        transforms, X_out = build_feature_pipeline(config, X)
        np.testing.assert_allclose(X_out.mean(axis=0), 0.0, atol=1e-5)
        assert len(transforms) == 1

    def test_pca_reduces_dim(self) -> None:
        rng = np.random.RandomState(42)
        X = rng.randn(50, VECTOR_DIM).astype(np.float32)
        config: dict[str, Any] = {"feature_norm": "none", "pca_dim": 4}
        transforms, X_out = build_feature_pipeline(config, X)
        assert X_out.shape == (50, 4)

    def test_pca_clamps_to_min_dim(self) -> None:
        rng = np.random.RandomState(42)
        X = rng.randn(10, 4).astype(np.float32)
        config: dict[str, Any] = {"feature_norm": "none", "pca_dim": 256}
        transforms, X_out = build_feature_pipeline(config, X)
        assert X_out.shape[1] <= min(10, 4)

    def test_apply_transforms_on_val(self) -> None:
        rng = np.random.RandomState(42)
        X_train = rng.randn(50, VECTOR_DIM).astype(np.float32)
        X_val = rng.randn(10, VECTOR_DIM).astype(np.float32)
        config: dict[str, Any] = {"feature_norm": "standard", "pca_dim": 8}
        transforms, _ = build_feature_pipeline(config, X_train)
        X_val_out = apply_transforms(transforms, X_val)
        assert X_val_out.shape == (10, 8)


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_perfect_classifier(self) -> None:
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_scores = np.array([0.99, 0.95, 0.92, 0.1, 0.05, 0.02])
        m = compute_metrics(y_true, y_scores, threshold=0.5)
        assert m["tp"] == 3
        assert m["fp"] == 0
        assert m["fn"] == 0
        assert m["tn"] == 3
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0
        assert m["high_conf_fp_rate"] == 0.0

    def test_high_conf_fp_rate_at_boundary(self) -> None:
        y_true = np.array([0, 0, 0, 0, 0])
        y_scores = np.array([0.89, 0.90, 0.91, 0.50, 0.10])
        m = compute_metrics(y_true, y_scores, threshold=0.5)
        # 0.90 and 0.91 are >= 0.90, so 2 out of 5
        assert m["high_conf_fp_rate"] == 0.4

    def test_grouped_metrics(self) -> None:
        y_true = np.array([0, 0, 0, 0])
        y_scores = np.array([0.95, 0.80, 0.92, 0.70])
        groups: list[str | None] = ["vessel", "vessel", "rain", None]
        m = compute_metrics(y_true, y_scores, threshold=0.5, negative_groups=groups)
        assert "high_conf_fp_rate_by_group" in m
        # vessel: 1/2 >= 0.90, rain: 1/1 >= 0.90
        assert m["high_conf_fp_rate_by_group"]["vessel"] == 0.5
        assert m["high_conf_fp_rate_by_group"]["rain"] == 1.0

    def test_find_top_false_positives(self) -> None:
        ids = ["a", "b", "c", "d", "e"]
        y_true = np.array([1, 0, 0, 0, 1])
        y_scores = np.array([0.9, 0.8, 0.95, 0.3, 0.7])
        groups: list[str | None] = [None, "vessel", None, "rain", None]
        fps = find_top_false_positives(ids, y_true, y_scores, groups, n=2)
        assert len(fps) == 2
        assert fps[0]["id"] == "c"  # highest scoring negative
        assert fps[0]["score"] == 0.95
        assert fps[1]["id"] == "b"
        assert fps[1]["negative_group"] == "vessel"


# ---------------------------------------------------------------------------
# Classifier tests
# ---------------------------------------------------------------------------


class TestClassifiers:
    def test_build_logreg(self) -> None:
        clf = build_classifier({"classifier": "logreg", "seed": 42})
        assert hasattr(clf, "fit")

    def test_build_linear_svm(self) -> None:
        clf = build_classifier({"classifier": "linear_svm", "seed": 42})
        assert hasattr(clf, "fit")

    def test_build_mlp(self) -> None:
        clf = build_classifier({"classifier": "mlp", "seed": 42})
        assert hasattr(clf, "fit")

    def test_unknown_classifier_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown classifier"):
            build_classifier({"classifier": "xgboost"})


# ---------------------------------------------------------------------------
# End-to-end train_eval test
# ---------------------------------------------------------------------------


class TestTrainEval:
    def test_single_run_returns_valid_output(self, tmp_path: Path) -> None:
        """End-to-end: linearly separable data, verify output schema."""
        parquet = tmp_path / "test.parquet"
        n_total = 40
        dim = VECTOR_DIM

        # Create linearly separable embeddings
        rng = np.random.RandomState(42)
        pos_vecs = rng.randn(20, dim).astype(np.float32) + 2.0
        neg_vecs = rng.randn(20, dim).astype(np.float32) - 2.0
        all_vecs = np.vstack([pos_vecs, neg_vecs])

        schema = pa.schema(
            [
                ("row_index", pa.int32()),
                ("embedding", pa.list_(pa.float32(), dim)),
            ]
        )
        table = pa.table(
            {
                "row_index": list(range(n_total)),
                "embedding": [v.tolist() for v in all_vecs],
            },
            schema=schema,
        )
        pq.write_table(table, str(parquet))

        # Build manifest: 14 train + 6 val for each class
        examples = []
        for i in range(20):
            examples.append(
                {
                    "id": f"pos_{i}",
                    "split": "train" if i < 14 else "val",
                    "label": 1,
                    "parquet_path": str(parquet),
                    "row_index": i,
                    "audio_file_id": "f1",
                    "negative_group": None,
                }
            )
        for i in range(20):
            examples.append(
                {
                    "id": f"neg_{i}",
                    "split": "train" if i < 14 else "val",
                    "label": 0,
                    "parquet_path": str(parquet),
                    "row_index": 20 + i,
                    "audio_file_id": "f2",
                    "negative_group": None,
                }
            )

        manifest = {"metadata": {}, "examples": examples}
        config = {
            "classifier": "logreg",
            "feature_norm": "l2",
            "pca_dim": None,
            "threshold": 0.5,
            "context_pooling": "center",
            "class_weight_pos": 1.0,
            "class_weight_neg": 1.0,
            "prob_calibration": "none",
            "hard_negative_fraction": 0.0,
            "seed": 42,
        }

        result = train_eval(manifest, config)

        assert "metrics" in result
        assert "top_false_positives" in result
        m = result["metrics"]
        for key in [
            "threshold",
            "precision",
            "recall",
            "fp_rate",
            "high_conf_fp_rate",
            "tp",
            "fp",
            "fn",
            "tn",
        ]:
            assert key in m, f"Missing metric: {key}"
        assert m["seed"] == 42
