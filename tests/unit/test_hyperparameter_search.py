"""Unit tests for hyperparameter search and comparison service modules."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from humpback.services.hyperparameter_service.comparison import (
    _manifest_split_counts,
    _metric_delta,
    build_prediction_disagreements,
)
from humpback.services.hyperparameter_service.search import (
    PROGRESS_CALLBACK_INTERVAL,
    _write_json,
    default_objective,
    run_search,
)
from humpback.services.hyperparameter_service.search_space import (
    DEFAULT_SEARCH_SPACE,
    config_hash,
    sample_config,
)


# ---------------------------------------------------------------------------
# Search space
# ---------------------------------------------------------------------------


class TestSearchSpace:
    def test_no_hard_negative_fraction(self) -> None:
        assert "hard_negative_fraction" not in DEFAULT_SEARCH_SPACE

    def test_all_dimensions_present(self) -> None:
        expected = {
            "feature_norm",
            "pca_dim",
            "classifier",
            "class_weight_pos",
            "class_weight_neg",
            "prob_calibration",
            "threshold",
            "context_pooling",
        }
        assert set(DEFAULT_SEARCH_SPACE.keys()) == expected

    def test_sample_config_uses_provided_space(self) -> None:
        space = {"a": [1, 2], "b": ["x", "y"]}
        rng = random.Random(42)
        config = sample_config(rng, space)
        assert set(config.keys()) == {"a", "b"}
        assert config["a"] in [1, 2]
        assert config["b"] in ["x", "y"]

    def test_sample_config_deterministic(self) -> None:
        space = DEFAULT_SEARCH_SPACE
        c1 = sample_config(random.Random(42), space)
        c2 = sample_config(random.Random(42), space)
        assert c1 == c2

    def test_config_hash_deterministic(self) -> None:
        config = {"a": 1, "b": "x"}
        assert config_hash(config) == config_hash(config)

    def test_config_hash_different_configs(self) -> None:
        c1 = {"a": 1}
        c2 = {"a": 2}
        assert config_hash(c1) != config_hash(c2)

    def test_config_hash_key_order_independent(self) -> None:
        c1 = {"a": 1, "b": 2}
        c2 = {"b": 2, "a": 1}
        assert config_hash(c1) == config_hash(c2)


# ---------------------------------------------------------------------------
# Default objective
# ---------------------------------------------------------------------------


class TestDefaultObjective:
    def test_perfect_metrics(self) -> None:
        metrics = {"recall": 1.0, "high_conf_fp_rate": 0.0, "fp_rate": 0.0}
        assert default_objective(metrics) == 1.0

    def test_high_conf_fp_penalized_heavily(self) -> None:
        metrics = {"recall": 1.0, "high_conf_fp_rate": 0.1, "fp_rate": 0.0}
        assert default_objective(metrics) == pytest.approx(-0.5)

    def test_general_fp_penalized(self) -> None:
        metrics = {"recall": 1.0, "high_conf_fp_rate": 0.0, "fp_rate": 0.1}
        assert default_objective(metrics) == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# JSON persistence
# ---------------------------------------------------------------------------


class TestWriteJson:
    def test_write_and_read(self, tmp_path: Path) -> None:
        import json

        data = {"key": "value", "number": 42}
        path = tmp_path / "test.json"
        _write_json(path, data)
        with open(path) as f:
            loaded = json.load(f)
        assert loaded == data

    def test_atomic_no_tmp_leftover(self, tmp_path: Path) -> None:
        path = tmp_path / "test.json"
        _write_json(path, {"a": 1})
        tmp = path.with_suffix(".tmp")
        assert not tmp.exists()


# ---------------------------------------------------------------------------
# Search loop (with mocked train_eval)
# ---------------------------------------------------------------------------

VECTOR_DIM = 16


def _write_synthetic_parquet(path: Path, n_rows: int, dim: int = VECTOR_DIM) -> None:
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
    parquet_path: str, n_pos: int = 10, n_neg: int = 10
) -> dict[str, Any]:
    examples = []
    for i in range(n_pos):
        examples.append(
            {
                "id": f"pos_{i}",
                "split": "train" if i < 7 else "val",
                "label": 1,
                "parquet_path": parquet_path,
                "row_index": i,
                "audio_file_id": "file1",
                "negative_group": None,
            }
        )
    for i in range(n_neg):
        examples.append(
            {
                "id": f"neg_{i}",
                "split": "train" if i < 7 else "val",
                "label": 0,
                "parquet_path": parquet_path,
                "row_index": n_pos + i,
                "audio_file_id": "file2",
                "negative_group": None,
            }
        )
    return {"metadata": {"created_at": "2026-01-01"}, "examples": examples}


class TestRunSearch:
    def test_basic_search(self, tmp_path: Path) -> None:
        pq_path = tmp_path / "embeddings.parquet"
        _write_synthetic_parquet(pq_path, 20)
        manifest = _make_manifest(str(pq_path))
        results_dir = tmp_path / "results"

        search_space = {
            "feature_norm": ["none"],
            "classifier": ["logreg"],
            "class_weight_pos": [1.0],
            "class_weight_neg": [1.0],
            "threshold": [0.5],
            "context_pooling": ["center"],
        }

        # Pre-build a simple embedding cache
        rng = np.random.RandomState(42)
        embedding_cache = {
            "center": {
                ex["id"]: rng.randn(VECTOR_DIM).astype(np.float32)
                for ex in manifest["examples"]
            }
        }

        summary = run_search(
            manifest=manifest,
            search_space=search_space,
            n_trials=3,
            seed=42,
            results_dir=results_dir,
            embedding_cache=embedding_cache,
        )

        assert summary["total_trials"] >= 1
        assert summary["best_objective"] is not None
        assert summary["best_config"] is not None
        assert (results_dir / "search_history.json").exists()
        assert (results_dir / "best_run.json").exists()

    def test_progress_callback_called(self, tmp_path: Path) -> None:
        pq_path = tmp_path / "embeddings.parquet"
        _write_synthetic_parquet(pq_path, 20)
        manifest = _make_manifest(str(pq_path))
        results_dir = tmp_path / "results"

        search_space = {
            "feature_norm": ["none", "l2"],
            "classifier": ["logreg"],
            "class_weight_pos": [1.0],
            "class_weight_neg": [1.0],
            "threshold": [0.5, 0.9],
            "context_pooling": ["center"],
        }

        rng = np.random.RandomState(42)
        embedding_cache = {
            "center": {
                ex["id"]: rng.randn(VECTOR_DIM).astype(np.float32)
                for ex in manifest["examples"]
            }
        }

        callback = MagicMock()
        run_search(
            manifest=manifest,
            search_space=search_space,
            n_trials=PROGRESS_CALLBACK_INTERVAL + 2,
            seed=42,
            results_dir=results_dir,
            embedding_cache=embedding_cache,
            progress_callback=callback,
        )

        # Should have been called at least once (at interval boundary or final)
        assert callback.call_count >= 1
        # First positional arg is trials_completed (int)
        first_call_args = callback.call_args_list[0][0]
        assert isinstance(first_call_args[0], int)

    def test_deduplication(self, tmp_path: Path) -> None:
        pq_path = tmp_path / "embeddings.parquet"
        _write_synthetic_parquet(pq_path, 20)
        manifest = _make_manifest(str(pq_path))
        results_dir = tmp_path / "results"

        # Single-value space → all trials produce the same config hash
        search_space = {
            "feature_norm": ["none"],
            "classifier": ["logreg"],
            "class_weight_pos": [1.0],
            "class_weight_neg": [1.0],
            "threshold": [0.5],
            "context_pooling": ["center"],
        }

        rng = np.random.RandomState(42)
        embedding_cache = {
            "center": {
                ex["id"]: rng.randn(VECTOR_DIM).astype(np.float32)
                for ex in manifest["examples"]
            }
        }

        summary = run_search(
            manifest=manifest,
            search_space=search_space,
            n_trials=5,
            seed=42,
            results_dir=results_dir,
            embedding_cache=embedding_cache,
        )

        # Only 1 unique trial, rest are duplicates
        assert summary["total_trials"] == 1
        assert summary["skipped_duplicates"] == 4


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------


class TestManifestSplitCounts:
    def test_basic(self) -> None:
        manifest = {
            "examples": [
                {"split": "train", "label": 1},
                {"split": "train", "label": 0},
                {"split": "train", "label": 0},
                {"split": "val", "label": 1},
                {"split": "val", "label": 0},
            ]
        }
        counts = _manifest_split_counts(manifest)
        assert counts["train"] == {"total": 3, "positive": 1, "negative": 2}
        assert counts["val"] == {"total": 2, "positive": 1, "negative": 1}
        assert "test" not in counts


class TestMetricDelta:
    def test_basic_delta(self) -> None:
        auto = {"precision": 0.9, "recall": 0.8, "fp_rate": 0.1}
        prod = {"precision": 0.7, "recall": 0.9, "fp_rate": 0.2}
        delta = _metric_delta(auto, prod)
        assert delta["precision"] == pytest.approx(0.2)
        assert delta["recall"] == pytest.approx(-0.1)
        assert delta["fp_rate"] == pytest.approx(-0.1)

    def test_missing_keys_skipped(self) -> None:
        auto = {"precision": 0.9}
        prod = {"recall": 0.8}
        delta = _metric_delta(auto, prod)
        assert delta == {}


class TestPredictionDisagreements:
    def test_finds_disagreements(self) -> None:
        manifest = {
            "examples": [
                {"id": "a", "label": 1, "split": "val", "source_type": "embedding_set"},
                {"id": "b", "label": 0, "split": "val", "source_type": "embedding_set"},
            ]
        }
        auto_eval = {
            "example_ids": ["a", "b"],
            "scores": [0.9, 0.8],  # predicts both positive at threshold 0.5
        }
        prod_eval = {
            "example_ids": ["a", "b"],
            "scores": [0.9, 0.3],  # predicts a positive, b negative at threshold 0.5
        }
        disagreements = build_prediction_disagreements(
            manifest,
            auto_eval,
            prod_eval,
            autoresearch_threshold=0.5,
            production_threshold=0.5,
            top_n=10,
        )
        assert len(disagreements) == 1
        assert disagreements[0]["id"] == "b"
        assert disagreements[0]["autoresearch_pred"] == 1
        assert disagreements[0]["production_pred"] == 0

    def test_no_disagreements_when_same(self) -> None:
        manifest = {
            "examples": [
                {"id": "a", "label": 1, "split": "val", "source_type": "embedding_set"},
            ]
        }
        auto_eval = {"example_ids": ["a"], "scores": [0.9]}
        prod_eval = {"example_ids": ["a"], "scores": [0.8]}
        disagreements = build_prediction_disagreements(
            manifest,
            auto_eval,
            prod_eval,
            autoresearch_threshold=0.5,
            production_threshold=0.5,
            top_n=10,
        )
        assert len(disagreements) == 0

    def test_top_n_limits_results(self) -> None:
        manifest = {
            "examples": [
                {"id": f"ex_{i}", "label": 0, "split": "val", "source_type": "det"}
                for i in range(10)
            ]
        }
        auto_eval = {
            "example_ids": [f"ex_{i}" for i in range(10)],
            "scores": [0.9] * 10,
        }
        prod_eval = {
            "example_ids": [f"ex_{i}" for i in range(10)],
            "scores": [0.1] * 10,
        }
        disagreements = build_prediction_disagreements(
            manifest,
            auto_eval,
            prod_eval,
            autoresearch_threshold=0.5,
            production_threshold=0.5,
            top_n=3,
        )
        assert len(disagreements) == 3
