"""Unit tests for the shared replay module (src/humpback/classifier/replay.py)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from humpback.classifier.replay import (
    ParquetCacheEntry,  # noqa: F401 — used in type annotations
    PoolingReport,
    apply_context_pooling,
    build_replay_pipeline,
    compute_metrics,
    compute_sample_weight,
    evaluate_on_split,
    load_parquet_cache,
)

VECTOR_DIM = 16


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_embedding_set_parquet(
    path: Path, n_rows: int, dim: int = VECTOR_DIM
) -> None:
    """Write a synthetic embedding-set-format Parquet (row_index column)."""
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


def _write_row_id_parquet(path: Path, n_rows: int, dim: int = VECTOR_DIM) -> None:
    """Write a synthetic row-id-format Parquet (row_id column)."""
    rng = np.random.RandomState(99)
    schema = pa.schema(
        [
            ("row_id", pa.string()),
            ("embedding", pa.list_(pa.float32(), dim)),
        ]
    )
    row_ids = [f"rid-{i}" for i in range(n_rows)]
    embeddings = [rng.randn(dim).astype(np.float32).tolist() for _ in range(n_rows)]
    table = pa.table({"row_id": row_ids, "embedding": embeddings}, schema=schema)
    pq.write_table(table, str(path))


def _write_filename_parquet(
    path: Path,
    n_rows: int,
    filenames: list[str] | None = None,
    dim: int = VECTOR_DIM,
) -> None:
    """Write a synthetic legacy detection-format Parquet (filename column)."""
    rng = np.random.RandomState(77)
    if filenames is None:
        filenames = ["file_a.wav"] * n_rows
    schema = pa.schema(
        [
            ("filename", pa.string()),
            ("embedding", pa.list_(pa.float32(), dim)),
        ]
    )
    embeddings = [rng.randn(dim).astype(np.float32).tolist() for _ in range(n_rows)]
    table = pa.table({"filename": filenames, "embedding": embeddings}, schema=schema)
    pq.write_table(table, str(path))


def _make_manifest(
    parquet_path: str,
    n_pos: int = 10,
    n_neg: int = 10,
    use_row_id: bool = False,
) -> dict[str, Any]:
    """Create a minimal manifest for testing."""
    examples = []
    for i in range(n_pos):
        ex: dict[str, Any] = {
            "id": f"pos_{i}",
            "split": "train" if i < n_pos * 7 // 10 else "val",
            "label": 1,
            "parquet_path": parquet_path,
            "negative_group": None,
        }
        if use_row_id:
            ex["row_id"] = f"rid-{i}"
        else:
            ex["row_index"] = i
        examples.append(ex)
    for i in range(n_neg):
        ex = {
            "id": f"neg_{i}",
            "split": "train" if i < n_neg * 7 // 10 else "val",
            "label": 0,
            "parquet_path": parquet_path,
            "negative_group": "vessel" if i % 2 == 0 else None,
        }
        if use_row_id:
            ex["row_id"] = f"rid-{n_pos + i}"
        else:
            ex["row_index"] = n_pos + i
        examples.append(ex)
    return {"metadata": {}, "examples": examples}


# ---------------------------------------------------------------------------
# Context pooling tests
# ---------------------------------------------------------------------------


class TestContextPooling:
    def test_center_returns_passthrough_and_report(self, tmp_path: Path) -> None:
        parquet = tmp_path / "test.parquet"
        _write_embedding_set_parquet(parquet, 5)

        from humpback.processing.embeddings import read_embeddings

        ri, emb = read_embeddings(parquet)
        manifest = _make_manifest(str(parquet), n_pos=3, n_neg=2)
        lookup = {ex["id"]: emb[ex["row_index"]] for ex in manifest["examples"]}
        cache: dict[str, ParquetCacheEntry] = {str(parquet): (ri, emb, None, None)}

        result, report = apply_context_pooling(manifest, lookup, cache, "center")
        assert isinstance(report, PoolingReport)
        assert report.applied_count == 0
        assert report.fallback_count == len(lookup)
        for eid, vec in result.items():
            np.testing.assert_array_equal(vec, lookup[eid])

    def test_mean3_with_pooling_report(self, tmp_path: Path) -> None:
        parquet = tmp_path / "test.parquet"
        _write_embedding_set_parquet(parquet, 5)

        from humpback.processing.embeddings import read_embeddings

        ri, emb = read_embeddings(parquet)
        # Mid-row example should get neighbors
        manifest = {
            "metadata": {},
            "examples": [
                {
                    "id": "e2",
                    "split": "train",
                    "label": 1,
                    "parquet_path": str(parquet),
                    "row_index": 2,
                },
            ],
        }
        lookup = {"e2": emb[2]}
        cache: dict[str, ParquetCacheEntry] = {str(parquet): (ri, emb, None, None)}

        result, report = apply_context_pooling(manifest, lookup, cache, "mean3")
        expected = np.mean([emb[1], emb[2], emb[3]], axis=0).astype(np.float32)
        np.testing.assert_allclose(result["e2"], expected, atol=1e-6)
        assert report.applied_count == 1
        assert report.fallback_count == 0

    def test_max3_with_report(self, tmp_path: Path) -> None:
        parquet = tmp_path / "test.parquet"
        _write_embedding_set_parquet(parquet, 5)

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
                },
            ],
        }
        lookup = {"e2": emb[2]}
        cache: dict[str, ParquetCacheEntry] = {str(parquet): (ri, emb, None, None)}

        result, report = apply_context_pooling(manifest, lookup, cache, "max3")
        expected = np.max([emb[1], emb[2], emb[3]], axis=0).astype(np.float32)
        np.testing.assert_allclose(result["e2"], expected, atol=1e-6)
        assert report.applied_count == 1

    def test_row_id_format_falls_back_to_center(self, tmp_path: Path) -> None:
        parquet = tmp_path / "test.parquet"
        _write_row_id_parquet(parquet, 5)

        cache = load_parquet_cache({"examples": [{"parquet_path": str(parquet)}]})
        _row_indices, emb, _filenames, row_ids = cache[str(parquet)]

        manifest = {
            "metadata": {},
            "examples": [
                {
                    "id": "e0",
                    "split": "train",
                    "label": 1,
                    "parquet_path": str(parquet),
                    "row_id": "rid-1",
                },
                {
                    "id": "e1",
                    "split": "train",
                    "label": 0,
                    "parquet_path": str(parquet),
                    "row_id": "rid-2",
                },
            ],
        }
        lookup = {"e0": emb[1], "e1": emb[2]}

        result, report = apply_context_pooling(manifest, lookup, cache, "mean3")
        # Row-id format should fall back to center for all examples
        np.testing.assert_array_equal(result["e0"], emb[1])
        np.testing.assert_array_equal(result["e1"], emb[2])
        assert report.applied_count == 0
        assert report.fallback_count == 2

    def test_cross_file_neighbor_skipped_filename_format(self, tmp_path: Path) -> None:
        parquet = tmp_path / "test.parquet"
        filenames = ["a.wav", "a.wav", "b.wav", "b.wav", "b.wav"]
        _write_filename_parquet(parquet, 5, filenames=filenames)

        cache = load_parquet_cache({"examples": [{"parquet_path": str(parquet)}]})
        row_indices, emb, fnames, _row_ids = cache[str(parquet)]

        # Row 1 (a.wav) — right neighbor at row 2 is b.wav, should be skipped
        manifest = {
            "metadata": {},
            "examples": [
                {
                    "id": "e1",
                    "split": "train",
                    "label": 1,
                    "parquet_path": str(parquet),
                    "row_index": 1,
                },
            ],
        }
        lookup = {"e1": emb[1]}

        result, report = apply_context_pooling(manifest, lookup, cache, "mean3")
        # Only left neighbor (row 0, a.wav) should be used, right (row 2, b.wav) skipped
        expected = np.mean([emb[0], emb[1]], axis=0).astype(np.float32)
        np.testing.assert_allclose(result["e1"], expected, atol=1e-6)
        assert report.applied_count == 1

    def test_boundary_fallback_counts(self, tmp_path: Path) -> None:
        """Row 0 has no left neighbor — only center + right, still applied."""
        parquet = tmp_path / "test.parquet"
        _write_embedding_set_parquet(parquet, 3)

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
                },
            ],
        }
        lookup = {"e0": emb[0]}
        cache: dict[str, ParquetCacheEntry] = {str(parquet): (ri, emb, None, None)}

        result, report = apply_context_pooling(manifest, lookup, cache, "mean3")
        expected = np.mean([emb[0], emb[1]], axis=0).astype(np.float32)
        np.testing.assert_allclose(result["e0"], expected, atol=1e-6)
        # Has one neighbor (right), so it's applied, not fallback
        assert report.applied_count == 1
        assert report.fallback_count == 0


# ---------------------------------------------------------------------------
# Build replay pipeline tests
# ---------------------------------------------------------------------------


class TestBuildReplayPipeline:
    def _make_data(
        self, n_samples: int = 50, dim: int = VECTOR_DIM
    ) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.RandomState(42)
        X = rng.randn(n_samples, dim).astype(np.float32)
        y = np.array([1] * (n_samples // 2) + [0] * (n_samples // 2), dtype=np.int32)
        return X, y

    def test_norm_only_pipeline(self) -> None:
        X, y = self._make_data()
        config: dict[str, Any] = {
            "feature_norm": "l2",
            "classifier": "logreg",
            "seed": 42,
        }
        pipeline, eff = build_replay_pipeline(config, X, y)
        assert hasattr(pipeline, "predict_proba")
        probs = pipeline.predict_proba(X)
        assert probs.shape == (50, 2)
        assert eff.feature_norm == "l2"
        assert eff.pca_dim is None
        assert eff.prob_calibration == "none"

    def test_norm_pca_pipeline(self) -> None:
        X, y = self._make_data()
        config: dict[str, Any] = {
            "feature_norm": "standard",
            "pca_dim": 8,
            "classifier": "logreg",
            "seed": 42,
        }
        pipeline, eff = build_replay_pipeline(config, X, y)
        probs = pipeline.predict_proba(X)
        assert probs.shape == (50, 2)
        assert eff.pca_dim == 8
        assert eff.pca_components_actual == 8

    def test_pca_clamped_when_n_samples_small(self) -> None:
        X, y = self._make_data(n_samples=6, dim=4)
        config: dict[str, Any] = {
            "feature_norm": "none",
            "pca_dim": 256,
            "classifier": "logreg",
            "seed": 42,
        }
        pipeline, eff = build_replay_pipeline(config, X, y)
        assert eff.pca_components_actual is not None
        assert eff.pca_components_actual <= min(6, 4)
        probs = pipeline.predict_proba(X)
        assert probs.shape[0] == 6

    def test_norm_pca_calibration_platt(self) -> None:
        X, y = self._make_data(n_samples=60)
        config: dict[str, Any] = {
            "feature_norm": "l2",
            "pca_dim": 8,
            "prob_calibration": "platt",
            "classifier": "logreg",
            "seed": 42,
        }
        pipeline, eff = build_replay_pipeline(config, X, y)
        assert eff.prob_calibration == "platt"
        probs = pipeline.predict_proba(X)
        assert probs.shape == (60, 2)
        # Probabilities should be in [0, 1]
        assert np.all(probs >= 0) and np.all(probs <= 1)

    def test_calibration_isotonic_without_pca(self) -> None:
        X, y = self._make_data(n_samples=60)
        config: dict[str, Any] = {
            "feature_norm": "standard",
            "prob_calibration": "isotonic",
            "classifier": "logreg",
            "seed": 42,
        }
        pipeline, eff = build_replay_pipeline(config, X, y)
        assert eff.prob_calibration == "isotonic"
        assert eff.pca_dim is None
        probs = pipeline.predict_proba(X)
        assert probs.shape == (60, 2)

    def test_no_calibration_returns_pipeline_type(self) -> None:
        from sklearn.pipeline import Pipeline

        X, y = self._make_data()
        config: dict[str, Any] = {
            "feature_norm": "none",
            "classifier": "logreg",
            "seed": 42,
        }
        pipeline, eff = build_replay_pipeline(config, X, y)
        assert isinstance(pipeline, Pipeline)
        assert eff.prob_calibration == "none"

    def test_effective_config_to_dict(self) -> None:
        X, y = self._make_data()
        config: dict[str, Any] = {
            "feature_norm": "l2",
            "pca_dim": 4,
            "classifier": "logreg",
            "class_weight_pos": 2.0,
            "class_weight_neg": 1.0,
            "seed": 42,
        }
        _, eff = build_replay_pipeline(config, X, y)
        d = eff.to_dict()
        assert d["feature_norm"] == "l2"
        assert d["pca_dim"] == 4
        assert d["class_weight"] == {"0": 1.0, "1": 2.0}
        assert d["classifier_type"] == "logreg"


# ---------------------------------------------------------------------------
# Evaluate on split tests
# ---------------------------------------------------------------------------


class TestEvaluateOnSplit:
    def test_metrics_for_known_scores(self) -> None:
        """Build a manifest with known labels, use a mock classifier."""
        rng = np.random.RandomState(42)
        dim = 4
        n = 10
        X = rng.randn(n, dim).astype(np.float32)

        # 5 positive, 5 negative
        examples = []
        lookup: dict[str, np.ndarray] = {}
        for i in range(n):
            eid = f"ex_{i}"
            examples.append(
                {
                    "id": eid,
                    "split": "val",
                    "label": 1 if i < 5 else 0,
                }
            )
            lookup[eid] = X[i]

        manifest: dict[str, Any] = {"metadata": {}, "examples": examples}

        # Use a trivial classifier that returns known scores
        class _MockClf:
            def predict_proba(self, X: np.ndarray) -> np.ndarray:
                # First 5 get high scores, last 5 get low scores
                scores = np.zeros((X.shape[0], 2))
                scores[:, 1] = np.linspace(0.9, 0.1, X.shape[0])
                scores[:, 0] = 1 - scores[:, 1]
                return scores

        result = evaluate_on_split(
            manifest, lookup, _MockClf(), [], "val", threshold=0.5
        )
        metrics = result["metrics"]
        assert metrics["tp"] + metrics["fn"] == 5
        assert metrics["fp"] + metrics["tn"] == 5
        assert result["split"] == "val"
        assert len(result["example_ids"]) == n

    def test_compute_metrics_perfect(self) -> None:
        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_scores = np.array([0.99, 0.95, 0.92, 0.1, 0.05, 0.02])
        m = compute_metrics(y_true, y_scores, threshold=0.5)
        assert m["tp"] == 3
        assert m["fp"] == 0
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0


# ---------------------------------------------------------------------------
# Replay verification tests
# ---------------------------------------------------------------------------


class TestVerifyReplay:
    def _build_fixture(
        self, tmp_path: Path, n_train: int = 30, n_val: int = 10
    ) -> tuple[Any, dict[str, Any], dict[str, ParquetCacheEntry], dict[str, Any]]:
        """Build a trained pipeline and matching manifest for verification."""
        from humpback.classifier.replay import (
            build_embedding_lookup,
            build_replay_pipeline,
        )

        dim = 8
        total = n_train + n_val
        n_pos = total // 2
        n_neg = total - n_pos

        parquet = tmp_path / "test.parquet"
        _write_embedding_set_parquet(parquet, n_pos + n_neg, dim=dim)

        examples = []
        for i in range(n_pos):
            examples.append(
                {
                    "id": f"pos_{i}",
                    "split": "train" if i < n_train // 2 else "val",
                    "label": 1,
                    "parquet_path": str(parquet),
                    "row_index": i,
                }
            )
        for i in range(n_neg):
            examples.append(
                {
                    "id": f"neg_{i}",
                    "split": "train" if i < n_train // 2 else "val",
                    "label": 0,
                    "parquet_path": str(parquet),
                    "row_index": n_pos + i,
                }
            )
        manifest: dict[str, Any] = {"metadata": {}, "examples": examples}
        cache = load_parquet_cache(manifest)

        config: dict[str, Any] = {
            "classifier": "logreg",
            "feature_norm": "l2",
            "seed": 42,
            "context_pooling": "center",
        }

        # Train
        from humpback.classifier.replay import collect_split_arrays

        embedding_lookup = build_embedding_lookup(manifest, cache)
        _ids, y_train, X_train, _ng = collect_split_arrays(
            manifest, embedding_lookup, "train"
        )
        pipeline, eff = build_replay_pipeline(config, X_train, y_train)

        return pipeline, manifest, cache, config

    def _get_val_metrics(
        self,
        pipeline: Any,
        manifest: dict[str, Any],
        cache: dict[str, ParquetCacheEntry],
        config: dict[str, Any],
        threshold: float = 0.5,
    ) -> dict[str, Any]:
        """Compute val metrics using the same code path as verify_replay."""
        from humpback.classifier.replay import (
            build_embedding_lookup,
            collect_split_arrays,
        )

        embedding_lookup = build_embedding_lookup(manifest, cache)
        _ids, y_true, X, neg_groups = collect_split_arrays(
            manifest, embedding_lookup, "val"
        )
        y_scores = pipeline.predict_proba(X)[:, 1]
        return compute_metrics(y_true, y_scores, threshold, neg_groups)

    def test_passes_when_metrics_match(self, tmp_path: Path) -> None:
        from humpback.classifier.replay import verify_replay

        pipeline, manifest, cache, config = self._build_fixture(tmp_path)
        actual_metrics = self._get_val_metrics(pipeline, manifest, cache, config)

        result = verify_replay(
            pipeline,
            manifest,
            cache,
            config,
            {"val": actual_metrics},
            threshold=0.5,
            tolerance=0.01,
        )
        assert result["status"] == "verified"
        assert result["tolerance"] == 0.01
        assert result["threshold"] == 0.5
        assert result["splits"]["val"]["pass"] is True

    def test_mismatch_when_rate_exceeds_tolerance(self, tmp_path: Path) -> None:
        from humpback.classifier.replay import verify_replay

        pipeline, manifest, cache, config = self._build_fixture(tmp_path)
        actual_metrics = self._get_val_metrics(pipeline, manifest, cache, config)

        # Tamper with expected precision to create a mismatch
        tampered = dict(actual_metrics)
        tampered["precision"] = actual_metrics["precision"] + 0.05

        result = verify_replay(
            pipeline,
            manifest,
            cache,
            config,
            {"val": tampered},
            threshold=0.5,
            tolerance=0.01,
        )
        assert result["status"] == "mismatch"
        assert result["splits"]["val"]["pass"] is False
        assert result["splits"]["val"]["deltas"]["precision"] >= 0.04

    def test_mismatch_when_counts_differ(self, tmp_path: Path) -> None:
        from humpback.classifier.replay import verify_replay

        pipeline, manifest, cache, config = self._build_fixture(tmp_path)
        actual_metrics = self._get_val_metrics(pipeline, manifest, cache, config)

        # Tamper with expected tp count
        tampered = dict(actual_metrics)
        tampered["tp"] = actual_metrics["tp"] + 1

        result = verify_replay(
            pipeline,
            manifest,
            cache,
            config,
            {"val": tampered},
            threshold=0.5,
            tolerance=0.01,
        )
        assert result["status"] == "mismatch"
        assert result["splits"]["val"]["pass"] is False
        assert result["splits"]["val"]["deltas"]["tp"] == 1

    def test_tolerance_recorded_in_result(self, tmp_path: Path) -> None:
        from humpback.classifier.replay import verify_replay

        pipeline, manifest, cache, config = self._build_fixture(tmp_path)
        actual_metrics = self._get_val_metrics(pipeline, manifest, cache, config)

        result = verify_replay(
            pipeline,
            manifest,
            cache,
            config,
            {"val": actual_metrics},
            threshold=0.5,
            tolerance=0.005,
        )
        assert result["tolerance"] == 0.005

    def test_effective_config_included(self, tmp_path: Path) -> None:
        from humpback.classifier.replay import EffectiveConfig, verify_replay

        pipeline, manifest, cache, config = self._build_fixture(tmp_path)
        actual_metrics = self._get_val_metrics(pipeline, manifest, cache, config)

        eff = EffectiveConfig(
            feature_norm="l2",
            pca_dim=None,
            prob_calibration="none",
            classifier_type="logreg",
            context_pooling_applied_count=0,
            context_pooling_fallback_count=20,
        )

        result = verify_replay(
            pipeline,
            manifest,
            cache,
            config,
            {"val": actual_metrics},
            threshold=0.5,
            tolerance=0.01,
            effective_config=eff,
        )
        assert "effective_config" in result
        assert result["effective_config"]["feature_norm"] == "l2"
        assert result["effective_config"]["context_pooling_fallback_count"] == 20


# ---------------------------------------------------------------------------
# Promotability check tests
# ---------------------------------------------------------------------------


class TestAssessReproducibility:
    def _assess(self, config: dict[str, Any]) -> tuple[bool, list[str]]:
        from humpback.services.classifier_service import _assess_reproducibility

        return _assess_reproducibility(config)

    def test_ar_v1_style_config_is_promotable(self) -> None:
        ok, blockers = self._assess(
            {
                "classifier": "logreg",
                "feature_norm": "l2",
                "pca_dim": 128,
                "prob_calibration": "platt",
                "context_pooling": "mean3",
                "class_weight_pos": 3.0,
                "class_weight_neg": 1.0,
                "hard_negative_fraction": 0.0,
            }
        )
        assert ok is True
        assert blockers == []

    def test_linear_svm_still_blocked(self) -> None:
        ok, blockers = self._assess({"classifier": "linear_svm"})
        assert ok is False
        assert any("linear_svm" in b for b in blockers)

    def test_hard_negative_fraction_still_blocked(self) -> None:
        ok, blockers = self._assess(
            {"classifier": "logreg", "hard_negative_fraction": 0.1}
        )
        assert ok is False
        assert any("hard-negative" in b.lower() for b in blockers)

    def test_mlp_with_class_weights_promotable(self) -> None:
        ok, blockers = self._assess(
            {"classifier": "mlp", "class_weight_pos": 2.0, "class_weight_neg": 1.0}
        )
        assert ok is True
        assert blockers == []

    def test_simple_logreg_promotable(self) -> None:
        ok, blockers = self._assess(
            {
                "classifier": "logreg",
                "feature_norm": "standard",
                "hard_negative_fraction": 0.0,
            }
        )
        assert ok is True
        assert blockers == []


# ---------------------------------------------------------------------------
# compute_sample_weight tests
# ---------------------------------------------------------------------------


class TestComputeSampleWeight:
    def test_uniform_returns_none(self) -> None:
        y = np.array([0, 0, 1, 1])
        result = compute_sample_weight({0: 1.0, 1: 1.0}, y)
        assert result is None

    def test_non_uniform_returns_array(self) -> None:
        y = np.array([0, 0, 1, 1, 0])
        result = compute_sample_weight({0: 1.0, 1: 1.5}, y)
        assert result is not None
        expected = np.array([1.0, 1.0, 1.5, 1.5, 1.0])
        np.testing.assert_array_equal(result, expected)

    def test_both_non_default(self) -> None:
        y = np.array([1, 0, 1])
        result = compute_sample_weight({0: 2.0, 1: 3.0}, y)
        assert result is not None
        np.testing.assert_array_equal(result, np.array([3.0, 2.0, 3.0]))


# ---------------------------------------------------------------------------
# MLP replay pipeline with sample weights
# ---------------------------------------------------------------------------


class TestMlpReplayWithClassWeights:
    def test_mlp_non_uniform_weights_trains(self) -> None:
        rng = np.random.RandomState(42)
        X = np.vstack([rng.randn(30, VECTOR_DIM) + 2, rng.randn(30, VECTOR_DIM) - 2])
        y = np.concatenate([np.ones(30), np.zeros(30)])
        config: dict[str, Any] = {
            "classifier": "mlp",
            "feature_norm": "standard",
            "class_weight_pos": 1.0,
            "class_weight_neg": 1.5,
            "prob_calibration": "none",
            "seed": 42,
        }
        pipeline, effective = build_replay_pipeline(config, X, y)
        assert hasattr(pipeline, "predict_proba")
        assert effective.class_weight == {0: 1.5, 1: 1.0}

    def test_logreg_does_not_use_sample_weight(self) -> None:
        """LogisticRegression uses native class_weight, not sample_weight."""
        rng = np.random.RandomState(42)
        X = np.vstack([rng.randn(30, VECTOR_DIM) + 2, rng.randn(30, VECTOR_DIM) - 2])
        y = np.concatenate([np.ones(30), np.zeros(30)])
        config: dict[str, Any] = {
            "classifier": "logreg",
            "feature_norm": "standard",
            "class_weight_pos": 3.0,
            "class_weight_neg": 1.0,
            "prob_calibration": "none",
            "seed": 42,
        }
        pipeline, effective = build_replay_pipeline(config, X, y)
        clf = pipeline.named_steps["classifier"]
        assert clf.class_weight == {0: 1.0, 1: 3.0}
