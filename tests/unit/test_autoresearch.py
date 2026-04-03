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
from scripts.autoresearch.run_autoresearch import (
    _build_trial_manifest,
    _hard_negative_replay_count,
    _ordered_replay_candidate_ids,
    run_search,
)
from scripts.autoresearch.search_space import (
    SEARCH_SPACE,
    config_hash,
    sample_config,
)
from scripts.autoresearch.train_eval import (
    _load_parquet_cache,
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


class TestHardNegativeReplay:
    def test_hard_negative_replay_count(self) -> None:
        assert _hard_negative_replay_count(0, 0.4) == 0
        assert _hard_negative_replay_count(50, 0.0) == 0
        assert _hard_negative_replay_count(50, 0.1) == 5
        assert _hard_negative_replay_count(50, 0.4) == 20
        assert _hard_negative_replay_count(3, 0.1) == 1

    def test_ordered_replay_candidates_skip_train_examples(
        self, tmp_path: Path
    ) -> None:
        parquet = tmp_path / "test.parquet"
        _write_synthetic_parquet(parquet, 5)
        manifest = {
            "metadata": {},
            "examples": [
                {
                    "id": "neg_train",
                    "split": "train",
                    "label": 0,
                    "parquet_path": str(parquet),
                    "row_index": 0,
                    "audio_file_id": "f0",
                    "negative_group": "vessel",
                },
                {
                    "id": "neg_val_a",
                    "split": "val",
                    "label": 0,
                    "parquet_path": str(parquet),
                    "row_index": 1,
                    "audio_file_id": "f1",
                    "negative_group": "vessel",
                },
                {
                    "id": "neg_test_b",
                    "split": "test",
                    "label": 0,
                    "parquet_path": str(parquet),
                    "row_index": 2,
                    "audio_file_id": "f2",
                    "negative_group": "vessel",
                },
            ],
        }
        order = _ordered_replay_candidate_ids(
            manifest,
            {"neg_train", "neg_val_a", "neg_test_b"},
            seed=42,
        )
        assert "neg_train" not in order
        assert sorted(order) == ["neg_test_b", "neg_val_a"]

    def test_build_trial_manifest_moves_unsampled_candidates_to_unused(
        self,
        tmp_path: Path,
    ) -> None:
        parquet = tmp_path / "test.parquet"
        _write_synthetic_parquet(parquet, 6)
        manifest = {
            "metadata": {"name": "test"},
            "examples": [
                {
                    "id": "pos_train",
                    "split": "train",
                    "label": 1,
                    "parquet_path": str(parquet),
                    "row_index": 0,
                    "audio_file_id": "f0",
                    "negative_group": None,
                },
                {
                    "id": "hn1",
                    "split": "val",
                    "label": 0,
                    "parquet_path": str(parquet),
                    "row_index": 1,
                    "audio_file_id": "f1",
                    "negative_group": "vessel",
                },
                {
                    "id": "hn2",
                    "split": "val",
                    "label": 0,
                    "parquet_path": str(parquet),
                    "row_index": 2,
                    "audio_file_id": "f2",
                    "negative_group": "vessel",
                },
                {
                    "id": "hn3",
                    "split": "test",
                    "label": 0,
                    "parquet_path": str(parquet),
                    "row_index": 3,
                    "audio_file_id": "f3",
                    "negative_group": "vessel",
                },
                {
                    "id": "hn4",
                    "split": "test",
                    "label": 0,
                    "parquet_path": str(parquet),
                    "row_index": 4,
                    "audio_file_id": "f4",
                    "negative_group": "vessel",
                },
                {
                    "id": "neg_keep",
                    "split": "val",
                    "label": 0,
                    "parquet_path": str(parquet),
                    "row_index": 5,
                    "audio_file_id": "f5",
                    "negative_group": "rain",
                },
            ],
        }

        trial_manifest, replay_count = _build_trial_manifest(
            manifest,
            ["hn1", "hn2", "hn3", "hn4"],
            hard_negative_fraction=0.5,
        )

        assert replay_count == 2
        splits = {ex["id"]: ex["split"] for ex in trial_manifest["examples"]}
        assert splits["hn1"] == "train"
        assert splits["hn2"] == "train"
        assert splits["hn3"] == "unused"
        assert splits["hn4"] == "unused"
        assert splits["neg_keep"] == "val"
        assert splits["pos_train"] == "train"
        assert {ex["id"]: ex["split"] for ex in manifest["examples"]}["hn3"] == "test"

    def test_run_search_uses_effective_hard_negative_fraction(
        self, tmp_path: Path
    ) -> None:
        parquet = tmp_path / "test.parquet"
        _write_synthetic_parquet(parquet, 5)
        manifest = {
            "metadata": {},
            "examples": [
                {
                    "id": "pos_train",
                    "split": "train",
                    "label": 1,
                    "parquet_path": str(parquet),
                    "row_index": 0,
                    "audio_file_id": "f0",
                    "negative_group": None,
                },
                {
                    "id": "hn1",
                    "split": "val",
                    "label": 0,
                    "parquet_path": str(parquet),
                    "row_index": 1,
                    "audio_file_id": "f1",
                    "negative_group": "vessel",
                },
                {
                    "id": "hn2",
                    "split": "val",
                    "label": 0,
                    "parquet_path": str(parquet),
                    "row_index": 2,
                    "audio_file_id": "f2",
                    "negative_group": "vessel",
                },
                {
                    "id": "hn3",
                    "split": "test",
                    "label": 0,
                    "parquet_path": str(parquet),
                    "row_index": 3,
                    "audio_file_id": "f3",
                    "negative_group": "vessel",
                },
                {
                    "id": "keep_val",
                    "split": "val",
                    "label": 0,
                    "parquet_path": str(parquet),
                    "row_index": 4,
                    "audio_file_id": "f4",
                    "negative_group": "rain",
                },
            ],
        }
        configs = iter(
            [
                {
                    "feature_norm": "none",
                    "pca_dim": None,
                    "classifier": "logreg",
                    "class_weight_pos": 1.0,
                    "class_weight_neg": 1.0,
                    "hard_negative_fraction": 0.0,
                    "prob_calibration": "none",
                    "threshold": 0.5,
                    "context_pooling": "center",
                },
                {
                    "feature_norm": "none",
                    "pca_dim": None,
                    "classifier": "logreg",
                    "class_weight_pos": 1.0,
                    "class_weight_neg": 1.0,
                    "hard_negative_fraction": 0.4,
                    "prob_calibration": "none",
                    "threshold": 0.5,
                    "context_pooling": "center",
                },
            ]
        )
        observed: list[dict[str, Any]] = []

        def fake_sample_config(_rng: random.Random) -> dict[str, Any]:
            return dict(next(configs))

        def fake_train_eval(
            trial_manifest: dict[str, Any],
            config: dict[str, Any],
            parquet_cache: dict[str, Any] | None = None,
            precomputed_embeddings: dict[str, np.ndarray] | None = None,
        ) -> dict[str, Any]:
            observed.append(
                {
                    "fraction": config["hard_negative_fraction"],
                    "splits": {
                        ex["id"]: ex["split"]
                        for ex in trial_manifest["examples"]
                        if ex["id"].startswith("hn")
                    },
                }
            )
            return {
                "metrics": {
                    "threshold": config["threshold"],
                    "precision": 1.0,
                    "recall": 0.5,
                    "fp_rate": 0.0,
                    "high_conf_fp_rate": 0.0,
                    "tp": 1,
                    "fp": 0,
                    "fn": 1,
                    "tn": 1,
                },
                "top_false_positives": [{"id": "hn1", "score": 0.8}],
            }

        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(
            "scripts.autoresearch.run_autoresearch.sample_config",
            fake_sample_config,
        )
        monkeypatch.setattr(
            "scripts.autoresearch.run_autoresearch.train_eval",
            fake_train_eval,
        )
        try:
            summary = run_search(
                manifest=manifest,
                n_trials=2,
                objective_name="default",
                seed=42,
                results_dir=tmp_path / "results",
                hard_negative_ids={"hn1", "hn2", "hn3"},
                embedding_cache={"center": {}},
            )
        finally:
            monkeypatch.undo()

        assert summary["total_trials"] == 2
        assert observed[0]["fraction"] == 0.0
        assert set(observed[0]["splits"].values()) == {"unused"}
        assert observed[1]["fraction"] == 0.4
        assert list(observed[1]["splits"].values()).count("train") == 1
        assert list(observed[1]["splits"].values()).count("unused") == 2


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
        cache: dict[str, Any] = {str(parquet): (ri, emb, None)}

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
        cache: dict[str, Any] = {str(parquet): (ri, emb, None)}

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
        cache: dict[str, Any] = {str(parquet): (ri, emb, None)}

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
        cache: dict[str, Any] = {str(parquet): (ri, emb, None)}

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


# ---------------------------------------------------------------------------
# Detection job integration tests
# ---------------------------------------------------------------------------


def _write_detection_parquet(
    path: Path,
    filenames: list[str],
    confidences: list[float],
    dim: int = VECTOR_DIM,
) -> None:
    """Write a synthetic detection embeddings Parquet file."""
    rng = np.random.RandomState(99)
    n = len(filenames)
    schema = pa.schema(
        [
            ("filename", pa.string()),
            ("start_sec", pa.float32()),
            ("end_sec", pa.float32()),
            ("embedding", pa.list_(pa.float32(), dim)),
            ("confidence", pa.float32()),
        ]
    )
    table = pa.table(
        {
            "filename": filenames,
            "start_sec": [float(i * 5) for i in range(n)],
            "end_sec": [float(i * 5 + 5) for i in range(n)],
            "embedding": [rng.randn(dim).astype(np.float32).tolist() for _ in range(n)],
            "confidence": confidences,
        },
        schema=schema,
    )
    pq.write_table(table, str(path))


def _write_row_id_detection_parquet(
    path: Path,
    row_ids: list[str],
    confidences: list[float | None],
    dim: int = VECTOR_DIM,
) -> None:
    """Write a synthetic canonical row-id detection embeddings Parquet file."""
    rng = np.random.RandomState(123)
    schema = pa.schema(
        [
            ("row_id", pa.string()),
            ("embedding", pa.list_(pa.float32(), dim)),
            ("confidence", pa.float32()),
        ]
    )
    table = pa.table(
        {
            "row_id": row_ids,
            "embedding": [rng.randn(dim).astype(np.float32).tolist() for _ in row_ids],
            "confidence": confidences,
        },
        schema=schema,
    )
    pq.write_table(table, str(path))


def _write_detection_row_store(
    path: Path,
    labels: list[dict[str, str]],
) -> None:
    """Write a synthetic detection row store with label columns."""
    from humpback.classifier.detection_rows import (
        ROW_STORE_FIELDNAMES,
        ROW_STORE_SCHEMA,
    )

    rows: list[dict[str, str]] = []
    for i, lab in enumerate(labels):
        row = {field: "" for field in ROW_STORE_FIELDNAMES}
        row["row_id"] = lab.get("row_id", f"row-{i}")
        row["start_utc"] = str(i * 5.0)
        row["end_utc"] = str(i * 5.0 + 5.0)
        row["avg_confidence"] = lab.get("confidence", "0.5")
        row["peak_confidence"] = lab.get("confidence", "0.5")
        row["n_windows"] = "1"
        row.update({k: v for k, v in lab.items() if k in ROW_STORE_FIELDNAMES})
        rows.append(row)

    table = pa.table(
        {field: [r[field] for r in rows] for field in ROW_STORE_FIELDNAMES},
        schema=ROW_STORE_SCHEMA,
    )
    pq.write_table(table, str(path))


class TestScoreBands:
    def test_score_to_band_mapping(self) -> None:
        from scripts.autoresearch.generate_manifest import _score_to_band

        assert _score_to_band(0.5) == "det_0.50_0.90"
        assert _score_to_band(0.89) == "det_0.50_0.90"
        assert _score_to_band(0.90) == "det_0.90_0.95"
        assert _score_to_band(0.949) == "det_0.90_0.95"
        assert _score_to_band(0.95) == "det_0.95_0.99"
        assert _score_to_band(0.99) == "det_0.99_1.00"

    def test_score_below_range_returns_none(self) -> None:
        from scripts.autoresearch.generate_manifest import _score_to_band

        assert _score_to_band(0.1) is None


class TestDetectionParquetLoading:
    def test_auto_detect_detection_format(self, tmp_path: Path) -> None:
        """Detection Parquet (filename column) should be auto-detected."""
        det_parquet = tmp_path / "det_emb.parquet"
        _write_detection_parquet(
            det_parquet,
            filenames=["a.flac", "a.flac", "b.flac"],
            confidences=[0.9, 0.95, 0.8],
        )
        manifest = {
            "metadata": {},
            "examples": [
                {
                    "id": "d0",
                    "split": "train",
                    "label": 0,
                    "parquet_path": str(det_parquet),
                    "row_index": 0,
                    "audio_file_id": "a.flac",
                    "negative_group": None,
                },
            ],
        }
        cache = _load_parquet_cache(manifest)
        row_indices, embeddings, filenames, row_ids = cache[str(det_parquet)]
        assert row_indices is not None
        assert list(row_indices) == [0, 1, 2]
        assert embeddings.shape[0] == 3
        assert filenames is not None
        assert filenames == ["a.flac", "a.flac", "b.flac"]
        assert row_ids is None

    def test_auto_detect_embedding_set_format(self, tmp_path: Path) -> None:
        """Embedding set Parquet (row_index column) should be auto-detected."""
        es_parquet = tmp_path / "es.parquet"
        _write_synthetic_parquet(es_parquet, 5)
        manifest = {
            "metadata": {},
            "examples": [
                {
                    "id": "e0",
                    "split": "train",
                    "label": 1,
                    "parquet_path": str(es_parquet),
                    "row_index": 0,
                    "audio_file_id": "f1",
                    "negative_group": None,
                },
            ],
        }
        cache = _load_parquet_cache(manifest)
        row_indices, _embeddings, filenames, row_ids = cache[str(es_parquet)]
        assert row_indices is not None
        assert filenames is None
        assert row_ids is None

    def test_auto_detect_row_id_detection_format(self, tmp_path: Path) -> None:
        """Canonical detection Parquet (row_id column) should be auto-detected."""
        det_parquet = tmp_path / "det_row_id.parquet"
        _write_row_id_detection_parquet(
            det_parquet,
            row_ids=["rid-1", "rid-2", "rid-3"],
            confidences=[0.9, None, 0.8],
        )
        manifest = {
            "metadata": {},
            "examples": [
                {
                    "id": "d1",
                    "split": "train",
                    "label": 0,
                    "parquet_path": str(det_parquet),
                    "row_id": "rid-2",
                    "audio_file_id": "detjob1:2026-04-03T00",
                    "negative_group": None,
                },
            ],
        }
        cache = _load_parquet_cache(manifest)
        row_indices, embeddings, filenames, row_ids = cache[str(det_parquet)]
        assert row_indices is None
        assert embeddings.shape[0] == 3
        assert filenames is None
        assert row_ids == ["rid-1", "rid-2", "rid-3"]


class TestDetectionContextPooling:
    def test_cross_file_neighbor_skipped(self, tmp_path: Path) -> None:
        """Neighbors from a different filename should be skipped in pooling."""
        det_parquet = tmp_path / "det.parquet"
        # Row 0: file_a, Row 1: file_b — adjacent but different files
        _write_detection_parquet(
            det_parquet,
            filenames=["file_a.flac", "file_b.flac"],
            confidences=[0.9, 0.8],
        )
        manifest = {
            "metadata": {},
            "examples": [
                {
                    "id": "d0",
                    "split": "train",
                    "label": 0,
                    "parquet_path": str(det_parquet),
                    "row_index": 0,
                    "audio_file_id": "file_a.flac",
                    "negative_group": None,
                },
            ],
        }
        cache = _load_parquet_cache(manifest)
        from scripts.autoresearch.train_eval import _build_embedding_lookup

        lookup = _build_embedding_lookup(manifest, cache)
        result = apply_context_pooling(manifest, lookup, cache, "mean3")
        # Should be center-only since right neighbor is a different file
        np.testing.assert_array_equal(result["d0"], lookup["d0"])

    def test_same_file_neighbor_included(self, tmp_path: Path) -> None:
        """Neighbors from the same filename should be included in pooling."""
        det_parquet = tmp_path / "det.parquet"
        _write_detection_parquet(
            det_parquet,
            filenames=["file_a.flac", "file_a.flac", "file_a.flac"],
            confidences=[0.9, 0.8, 0.7],
        )
        manifest = {
            "metadata": {},
            "examples": [
                {
                    "id": "d1",
                    "split": "train",
                    "label": 0,
                    "parquet_path": str(det_parquet),
                    "row_index": 1,
                    "audio_file_id": "file_a.flac",
                    "negative_group": None,
                },
            ],
        }
        cache = _load_parquet_cache(manifest)
        from scripts.autoresearch.train_eval import _build_embedding_lookup

        lookup = _build_embedding_lookup(manifest, cache)
        result = apply_context_pooling(manifest, lookup, cache, "mean3")
        # Should average all 3 rows (left, center, right — all same file)
        _, embeddings, _, _ = cache[str(det_parquet)]
        expected = np.mean(
            [embeddings[0], embeddings[1], embeddings[2]], axis=0
        ).astype(np.float32)
        np.testing.assert_allclose(result["d1"], expected, atol=1e-6)

    def test_row_id_detections_fall_back_to_center(self, tmp_path: Path) -> None:
        """Row-id detection examples should ignore context pooling neighbors."""
        det_parquet = tmp_path / "det_row_id.parquet"
        _write_row_id_detection_parquet(
            det_parquet,
            row_ids=["rid-1", "rid-2", "rid-3"],
            confidences=[0.91, 0.92, 0.93],
        )
        manifest = {
            "metadata": {},
            "examples": [
                {
                    "id": "d1",
                    "split": "train",
                    "label": 0,
                    "parquet_path": str(det_parquet),
                    "row_id": "rid-2",
                    "audio_file_id": "detjob1:2026-04-03T00",
                    "negative_group": None,
                },
            ],
        }
        cache = _load_parquet_cache(manifest)
        from scripts.autoresearch.train_eval import _build_embedding_lookup

        lookup = _build_embedding_lookup(manifest, cache)
        mean_result = apply_context_pooling(manifest, lookup, cache, "mean3")
        max_result = apply_context_pooling(manifest, lookup, cache, "max3")

        np.testing.assert_array_equal(mean_result["d1"], lookup["d1"])
        np.testing.assert_array_equal(max_result["d1"], lookup["d1"])


class TestCollectDetectionExamples:
    def test_row_id_label_classification_and_summary(self, tmp_path: Path) -> None:
        """Default row-id manifests should only include explicit supervision."""
        det_dir = tmp_path / "detections" / "job1"
        det_dir.mkdir(parents=True)
        emb_path = det_dir / "detection_embeddings.parquet"
        row_path = det_dir / "detection_rows.parquet"

        _write_row_id_detection_parquet(
            emb_path,
            row_ids=[
                "rid-pos-voc",
                "rid-neg-voc",
                "rid-pos-binary",
                "rid-ship",
                "rid-band",
                "rid-null",
            ],
            confidences=[0.99, None, 0.92, 0.91, 0.94, None],
        )
        _write_detection_row_store(
            row_path,
            [
                {"row_id": "rid-pos-voc", "start_utc": "1712109600.0"},
                {"row_id": "rid-neg-voc", "start_utc": "1712109660.0"},
                {
                    "row_id": "rid-pos-binary",
                    "start_utc": "1712109720.0",
                    "humpback": "1",
                },
                {"row_id": "rid-ship", "start_utc": "1712109780.0", "ship": "1"},
                {"row_id": "rid-band", "start_utc": "1712109840.0"},
                {"row_id": "rid-null", "start_utc": "1712109900.0"},
            ],
        )

        from unittest.mock import patch

        with (
            patch(
                "humpback.storage.detection_embeddings_path",
                return_value=emb_path,
            ),
            patch(
                "humpback.storage.detection_row_store_path",
                return_value=row_path,
            ),
        ):
            from scripts.autoresearch.generate_manifest import (
                _collect_detection_examples,
            )
            from humpback.config import Settings

            settings = Settings()
            examples, summaries = _collect_detection_examples(
                [{"id": "job1-full-uuid-here"}],
                settings,
                score_range=(0.5, 0.995),
                vocalization_labels_by_job={
                    "job1-full-uuid-here": {
                        "rid-pos-voc": {"whup"},
                        "rid-neg-voc": {"(Negative)"},
                    }
                },
            )

        assert len(examples) == 4
        examples_by_row_id = {ex["row_id"]: ex for ex in examples}
        assert examples_by_row_id["rid-pos-voc"]["label"] == 1
        assert (
            examples_by_row_id["rid-pos-voc"]["label_source"] == "vocalization_positive"
        )
        assert examples_by_row_id["rid-neg-voc"]["label"] == 0
        assert (
            examples_by_row_id["rid-neg-voc"]["negative_group"]
            == "vocalization_negative"
        )
        assert examples_by_row_id["rid-pos-binary"]["label_source"] == "binary_positive"
        assert examples_by_row_id["rid-ship"]["negative_group"] == "ship"
        assert "rid-band" not in examples_by_row_id
        assert "rid-null" not in examples_by_row_id

        from scripts.autoresearch.generate_manifest import _row_id_split_group

        assert examples_by_row_id["rid-pos-voc"][
            "audio_file_id"
        ] == _row_id_split_group(
            "job1-full-uuid-here",
            1712109600.0,
        )
        assert examples_by_row_id["rid-neg-voc"]["detection_confidence"] is None

        summary = summaries["job1-full-uuid-here"]
        assert summary["included_positive"] == 2
        assert summary["included_negative"] == 2
        assert summary["included_positives_by_source"]["vocalization_positive"] == 1
        assert summary["included_positives_by_source"]["binary_positive"] == 1
        assert summary["included_negatives_by_source"]["vocalization_negative"] == 1
        assert summary["included_negatives_by_source"]["ship"] == 1
        assert summary["included_negatives_by_source"]["score_band"] == 0
        assert summary["skipped_unlabeled_not_explicit_negative"] == 2
        assert summary["skipped_null_confidence_unlabeled"] == 0
        assert summary["skipped_conflicts"] == 0

    def test_row_id_conflicts_are_skipped(self, tmp_path: Path) -> None:
        """Contradictory vocalization and row-store labels should be excluded."""
        det_dir = tmp_path / "detections" / "job2"
        det_dir.mkdir(parents=True)
        emb_path = det_dir / "detection_embeddings.parquet"
        row_path = det_dir / "detection_rows.parquet"

        _write_row_id_detection_parquet(
            emb_path,
            row_ids=["rid-conflict-a", "rid-conflict-b", "rid-ok"],
            confidences=[0.91, 0.92, 0.93],
        )
        _write_detection_row_store(
            row_path,
            [
                {"row_id": "rid-conflict-a", "humpback": "1"},
                {"row_id": "rid-conflict-b", "ship": "1"},
                {"row_id": "rid-ok", "background": "1"},
            ],
        )

        from unittest.mock import patch

        with (
            patch(
                "humpback.storage.detection_embeddings_path",
                return_value=emb_path,
            ),
            patch(
                "humpback.storage.detection_row_store_path",
                return_value=row_path,
            ),
        ):
            from scripts.autoresearch.generate_manifest import (
                _collect_detection_examples,
            )
            from humpback.config import Settings

            settings = Settings()
            examples, summaries = _collect_detection_examples(
                [{"id": "job2-full-uuid-here"}],
                settings,
                score_range=(0.5, 0.995),
                vocalization_labels_by_job={
                    "job2-full-uuid-here": {
                        "rid-conflict-a": {"(Negative)"},
                        "rid-conflict-b": {"whup"},
                    }
                },
            )

        assert len(examples) == 1
        assert examples[0]["row_id"] == "rid-ok"
        assert summaries["job2-full-uuid-here"]["skipped_conflicts"] == 2

    def test_row_id_unlabeled_windows_are_excluded_by_default(
        self,
        tmp_path: Path,
    ) -> None:
        """Unlabeled rows should not become negatives unless explicitly enabled."""
        det_dir = tmp_path / "detections" / "job3"
        det_dir.mkdir(parents=True)
        emb_path = det_dir / "detection_embeddings.parquet"
        row_path = det_dir / "detection_rows.parquet"

        _write_row_id_detection_parquet(
            emb_path,
            row_ids=["rid-low", "rid-mid", "rid-high"],
            confidences=[0.30, 0.70, 0.999],
        )
        _write_detection_row_store(
            row_path,
            [
                {"row_id": "rid-low"},
                {"row_id": "rid-mid"},
                {"row_id": "rid-high"},
                {"row_id": "rid-missing", "background": "1"},
            ],
        )

        from unittest.mock import patch

        with (
            patch(
                "humpback.storage.detection_embeddings_path",
                return_value=emb_path,
            ),
            patch(
                "humpback.storage.detection_row_store_path",
                return_value=row_path,
            ),
        ):
            from scripts.autoresearch.generate_manifest import (
                _collect_detection_examples,
            )
            from humpback.config import Settings

            settings = Settings()
            examples, summaries = _collect_detection_examples(
                [{"id": "job3-full-uuid-here"}],
                settings,
                score_range=(0.5, 0.995),
            )

        assert len(examples) == 0
        summary = summaries["job3-full-uuid-here"]
        assert summary["skipped_unlabeled_not_explicit_negative"] == 3
        assert summary["skipped_out_of_range_unlabeled"] == 0
        assert summary["skipped_null_confidence_unlabeled"] == 0
        assert summary["skipped_missing_embeddings"] == 1

    def test_row_id_unlabeled_windows_can_be_opted_in_as_score_band_negatives(
        self,
        tmp_path: Path,
    ) -> None:
        """Opt-in should restore score-band unlabeled negatives."""
        det_dir = tmp_path / "detections" / "job3b"
        det_dir.mkdir(parents=True)
        emb_path = det_dir / "detection_embeddings.parquet"
        row_path = det_dir / "detection_rows.parquet"

        _write_row_id_detection_parquet(
            emb_path,
            row_ids=["rid-low", "rid-mid", "rid-high"],
            confidences=[0.30, 0.70, 0.999],
        )
        _write_detection_row_store(
            row_path,
            [
                {"row_id": "rid-low"},
                {"row_id": "rid-mid"},
                {"row_id": "rid-high"},
            ],
        )

        from unittest.mock import patch

        with (
            patch(
                "humpback.storage.detection_embeddings_path",
                return_value=emb_path,
            ),
            patch(
                "humpback.storage.detection_row_store_path",
                return_value=row_path,
            ),
        ):
            from scripts.autoresearch.generate_manifest import (
                _collect_detection_examples,
            )
            from humpback.config import Settings

            settings = Settings()
            examples, summaries = _collect_detection_examples(
                [{"id": "job3b-full-uuid-here"}],
                settings,
                score_range=(0.5, 0.995),
                include_unlabeled_hard_negatives=True,
            )

        assert len(examples) == 1
        assert examples[0]["row_id"] == "rid-mid"
        assert examples[0]["negative_group"] == "det_0.50_0.90"
        summary = summaries["job3b-full-uuid-here"]
        assert summary["included_negatives_by_source"]["score_band"] == 1
        assert summary["skipped_out_of_range_unlabeled"] == 2

    def test_legacy_detection_embeddings_still_work(self, tmp_path: Path) -> None:
        """Legacy filename-based detection embeddings should still be supported."""
        det_dir = tmp_path / "detections" / "job4"
        det_dir.mkdir(parents=True)
        emb_path = det_dir / "detection_embeddings.parquet"
        row_path = det_dir / "detection_rows.parquet"

        _write_detection_parquet(
            emb_path,
            filenames=["a.flac"] * 3,
            confidences=[0.99, 0.91, 0.72],
        )
        _write_detection_row_store(
            row_path,
            [
                {"row_id": "rid-pos", "humpback": "1"},
                {"row_id": "rid-ship", "ship": "1"},
                {"row_id": "rid-band"},
            ],
        )

        from unittest.mock import patch

        with (
            patch(
                "humpback.storage.detection_embeddings_path",
                return_value=emb_path,
            ),
            patch(
                "humpback.storage.detection_row_store_path",
                return_value=row_path,
            ),
        ):
            from scripts.autoresearch.generate_manifest import (
                _collect_detection_examples,
            )
            from humpback.config import Settings

            settings = Settings()
            examples, summaries = _collect_detection_examples(
                [{"id": "job4-full-uuid-here"}],
                settings,
                score_range=(0.5, 0.995),
            )

        assert [ex["row_index"] for ex in examples] == [0, 1]
        assert [ex["label"] for ex in examples] == [1, 0]
        assert examples[1]["negative_group"] == "ship"
        assert summaries["job4-full-uuid-here"]["included_negative"] == 1
        assert (
            summaries["job4-full-uuid-here"]["skipped_unlabeled_not_explicit_negative"]
            == 1
        )


class TestMixedSourceTrainEval:
    def test_mixed_manifest_runs_end_to_end(self, tmp_path: Path) -> None:
        """Train_eval works with both embedding set and detection source types."""
        dim = VECTOR_DIM
        rng = np.random.RandomState(42)

        # Embedding set Parquet (positives)
        es_parquet = tmp_path / "es.parquet"
        pos_vecs = rng.randn(20, dim).astype(np.float32) + 2.0
        es_schema = pa.schema(
            [
                ("row_index", pa.int32()),
                ("embedding", pa.list_(pa.float32(), dim)),
            ]
        )
        pq.write_table(
            pa.table(
                {
                    "row_index": list(range(20)),
                    "embedding": [v.tolist() for v in pos_vecs],
                },
                schema=es_schema,
            ),
            str(es_parquet),
        )

        # Detection Parquet (negatives)
        det_parquet = tmp_path / "det.parquet"
        neg_vecs = rng.randn(20, dim).astype(np.float32) - 2.0
        det_schema = pa.schema(
            [
                ("row_id", pa.string()),
                ("embedding", pa.list_(pa.float32(), dim)),
                ("confidence", pa.float32()),
            ]
        )
        pq.write_table(
            pa.table(
                {
                    "row_id": [f"rid-{i}" for i in range(20)],
                    "embedding": [v.tolist() for v in neg_vecs],
                    "confidence": [0.9] * 20,
                },
                schema=det_schema,
            ),
            str(det_parquet),
        )

        examples = []
        for i in range(20):
            examples.append(
                {
                    "id": f"pos_{i}",
                    "split": "train" if i < 14 else "val",
                    "label": 1,
                    "source_type": "embedding_set",
                    "parquet_path": str(es_parquet),
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
                    "source_type": "detection_job",
                    "parquet_path": str(det_parquet),
                    "row_id": f"rid-{i}",
                    "audio_file_id": f"detjob1:2026-04-03T{i // 4:02d}",
                    "negative_group": "det_0.90_0.95",
                    "label_source": "score_band",
                }
            )

        manifest = {"metadata": {}, "examples": examples}
        config = {
            "classifier": "logreg",
            "feature_norm": "none",
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
        m = result["metrics"]
        assert "high_conf_fp_rate" in m
        assert m["tp"] + m["fn"] + m["fp"] + m["tn"] > 0
