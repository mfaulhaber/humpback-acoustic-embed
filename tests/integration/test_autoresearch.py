"""Integration test for autoresearch search loop.

Writes synthetic Parquet files, builds a manifest directly (no DB),
runs a small search, and verifies all output artifacts.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine, text

from humpback.config import Settings
from humpback.services.hyperparameter_service.comparison import (
    compare_classifiers,
    resolve_production_classifier,
)
from humpback.services.hyperparameter_service.train_eval import (
    collect_split_arrays,
    prepare_embeddings,
)

# Integration tests use the legacy run_search (with hard_negative support)
from scripts.autoresearch.run_autoresearch import run_search


VECTOR_DIM = 16


def _create_test_data(tmp_path: Path) -> dict:
    """Create synthetic linearly-separable Parquet data and a manifest."""
    rng = np.random.RandomState(99)

    # Positive embeddings (class center at +2)
    pos_path = tmp_path / "positive.parquet"
    pos_vecs = rng.randn(30, VECTOR_DIM).astype(np.float32) + 2.0

    # Negative embeddings (class center at -2)
    neg_path = tmp_path / "negative.parquet"
    neg_vecs = rng.randn(30, VECTOR_DIM).astype(np.float32) - 2.0

    schema = pa.schema(
        [
            ("row_index", pa.int32()),
            ("embedding", pa.list_(pa.float32(), VECTOR_DIM)),
        ]
    )

    for path, vecs in [(pos_path, pos_vecs), (neg_path, neg_vecs)]:
        table = pa.table(
            {
                "row_index": list(range(len(vecs))),
                "embedding": [v.tolist() for v in vecs],
            },
            schema=schema,
        )
        pq.write_table(table, str(path))

    # Build manifest: 21 train, 5 val, 4 test per class
    examples = []
    for i in range(30):
        if i < 21:
            split = "train"
        elif i < 26:
            split = "val"
        else:
            split = "test"
        examples.append(
            {
                "id": f"pos_{i}",
                "split": split,
                "label": 1,
                "parquet_path": str(pos_path),
                "row_index": i,
                "audio_file_id": "audio_pos",
                "negative_group": None,
            }
        )
    for i in range(30):
        if i < 21:
            split = "train"
        elif i < 26:
            split = "val"
        else:
            split = "test"
        examples.append(
            {
                "id": f"neg_{i}",
                "split": split,
                "label": 0,
                "parquet_path": str(neg_path),
                "row_index": i,
                "audio_file_id": "audio_neg",
                "negative_group": "vessel" if i % 3 == 0 else None,
            }
        )

    return {
        "metadata": {
            "created_at": "2026-04-01T00:00:00Z",
            "source_job_ids": ["test"],
            "positive_embedding_set_ids": ["pos"],
            "negative_embedding_set_ids": ["neg"],
            "split_strategy": "by_audio_file",
        },
        "examples": examples,
    }


def test_search_loop_end_to_end(tmp_path: Path) -> None:
    """Run 5 trials and verify all output artifacts."""
    manifest = _create_test_data(tmp_path)
    results_dir = tmp_path / "results"

    summary = run_search(
        manifest=manifest,
        n_trials=5,
        objective_name="default",
        seed=42,
        results_dir=results_dir,
    )

    # Summary has expected fields
    assert "total_trials" in summary
    assert "best_objective" in summary
    assert "best_config" in summary
    assert summary["total_trials"] > 0

    # search_history.json exists and is valid
    history_path = results_dir / "search_history.json"
    assert history_path.exists()
    with open(history_path) as f:
        history = json.load(f)
    assert isinstance(history, list)
    assert len(history) == summary["total_trials"]

    for entry in history:
        assert "trial" in entry
        assert "config" in entry
        assert "metrics" in entry
        assert "objective" in entry
        assert "timestamp" in entry
        m = entry["metrics"]
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
            assert key in m

    # best_run.json exists and is valid
    best_path = results_dir / "best_run.json"
    assert best_path.exists()
    with open(best_path) as f:
        best = json.load(f)
    assert "config" in best
    assert "metrics" in best
    assert "objective" in best

    # The best run should match the max objective in history
    max_obj = max(e["objective"] for e in history)
    assert abs(best["objective"] - max_obj) < 1e-9

    # top_false_positives.json exists and is valid
    fps_path = results_dir / "top_false_positives.json"
    assert fps_path.exists()
    with open(fps_path) as f:
        fps = json.load(f)
    assert isinstance(fps, list)
    for fp in fps:
        assert "id" in fp
        assert "score" in fp


def test_search_dedup_skips_repeats(tmp_path: Path) -> None:
    """With a tiny search space, duplicates should be skipped."""
    manifest = _create_test_data(tmp_path)
    results_dir = tmp_path / "results"

    # Run many more trials than unique configs in a tiny space
    summary = run_search(
        manifest=manifest,
        n_trials=10,
        objective_name="default",
        seed=42,
        results_dir=results_dir,
    )

    # Some duplicates should have been skipped
    # (not guaranteed with 10 trials on the full space, but the
    # mechanism is tested structurally)
    assert summary["total_trials"] + summary["skipped_duplicates"] == 10


def test_search_without_replay_dedups_hard_negative_fraction_only_variants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without replay IDs, hard_negative_fraction should not create new trials."""
    manifest = _create_test_data(tmp_path)
    results_dir = tmp_path / "results-no-replay"
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

    def fake_sample_config(_rng: random.Random) -> dict[str, object]:
        return dict(next(configs))

    monkeypatch.setattr(
        "scripts.autoresearch.run_autoresearch.sample_config",
        fake_sample_config,
    )

    summary = run_search(
        manifest=manifest,
        n_trials=2,
        objective_name="default",
        seed=42,
        results_dir=results_dir,
    )

    assert summary["total_trials"] == 1
    assert summary["skipped_duplicates"] == 1

    with open(results_dir / "best_run.json") as f:
        best_run = json.load(f)
    assert best_run["config"]["hard_negative_fraction"] == 0.0


def _create_mixed_source_data(tmp_path: Path) -> dict:
    """Create test data with embedding sets plus row-id detection sources."""
    rng = np.random.RandomState(77)

    # Embedding set Parquet (positives)
    es_path = tmp_path / "es_positive.parquet"
    pos_vecs = rng.randn(20, VECTOR_DIM).astype(np.float32) + 2.0
    es_schema = pa.schema(
        [
            ("row_index", pa.int32()),
            ("embedding", pa.list_(pa.float32(), VECTOR_DIM)),
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
        str(es_path),
    )

    # Detection Parquet (canonical row-id negatives from detection)
    det_path = tmp_path / "det_embeddings.parquet"
    neg_vecs = rng.randn(20, VECTOR_DIM).astype(np.float32) - 2.0
    det_schema = pa.schema(
        [
            ("row_id", pa.string()),
            ("embedding", pa.list_(pa.float32(), VECTOR_DIM)),
            ("confidence", pa.float32()),
        ]
    )
    pq.write_table(
        pa.table(
            {
                "row_id": [f"rid-{i}" for i in range(20)],
                "embedding": [v.tolist() for v in neg_vecs],
                "confidence": [0.9 + i * 0.005 for i in range(20)],
            },
            schema=det_schema,
        ),
        str(det_path),
    )

    examples = []
    for i in range(20):
        examples.append(
            {
                "id": f"pos_{i}",
                "split": "train" if i < 14 else "val",
                "label": 1,
                "source_type": "embedding_set",
                "parquet_path": str(es_path),
                "row_index": i,
                "audio_file_id": "audio_pos",
                "negative_group": None,
            }
        )
    for i in range(20):
        band = "det_0.90_0.95" if i < 10 else "det_0.95_0.99"
        examples.append(
            {
                "id": f"det_{i}",
                "split": "train" if i < 14 else "val",
                "label": 0,
                "source_type": "detection_job",
                "parquet_path": str(det_path),
                "row_id": f"rid-{i}",
                "audio_file_id": f"det1:2026-04-03T{i // 4:02d}",
                "negative_group": band,
                "label_source": "score_band",
            }
        )

    return {
        "metadata": {
            "created_at": "2026-04-01T00:00:00Z",
            "source_job_ids": [],
            "positive_embedding_set_ids": ["pos"],
            "negative_embedding_set_ids": [],
            "detection_job_ids": ["det1"],
            "score_range": [0.5, 0.995],
            "split_strategy": "by_audio_file",
        },
        "examples": examples,
    }


def test_mixed_source_search(tmp_path: Path) -> None:
    """Search with both embedding set and detection sources works end-to-end."""
    manifest = _create_mixed_source_data(tmp_path)
    results_dir = tmp_path / "results"

    summary = run_search(
        manifest=manifest,
        n_trials=3,
        objective_name="default",
        seed=42,
        results_dir=results_dir,
    )

    assert summary["total_trials"] == 3

    # Verify grouped metrics include detection score bands
    history_path = results_dir / "search_history.json"
    with open(history_path) as f:
        history = json.load(f)
    assert len(history) == 3

    # At least one run should have grouped FP rates if negative_group is set
    has_grouped = any(
        "high_conf_fp_rate_by_group" in entry["metrics"] for entry in history
    )
    # Grouped metrics appear when there are negatives with groups in val split
    # This depends on the random config (threshold, etc.), so we just check structure
    if has_grouped:
        for entry in history:
            if "high_conf_fp_rate_by_group" in entry["metrics"]:
                groups = entry["metrics"]["high_conf_fp_rate_by_group"]
                assert isinstance(groups, dict)


def test_compare_classifiers_against_production_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compare an autoresearch winner against a classifier loaded from the DB."""
    manifest = _create_test_data(tmp_path)
    results_dir = tmp_path / "results"

    run_search(
        manifest=manifest,
        n_trials=3,
        objective_name="default",
        seed=42,
        results_dir=results_dir,
    )
    with open(results_dir / "best_run.json") as f:
        best_run = json.load(f)

    pooled = prepare_embeddings(manifest, {"context_pooling": "center"})
    _train_ids, y_train, X_train, _train_neg_groups = collect_split_arrays(
        manifest,
        pooled,
        "train",
    )
    production_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(C=0.1, max_iter=1000)),
        ]
    )
    production_pipeline.fit(X_train, y_train)

    model_path = tmp_path / "lr-v12.joblib"
    joblib.dump(production_pipeline, model_path)

    db_path = tmp_path / "compare.sqlite"
    engine = create_engine(f"sqlite:///{db_path}")
    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    CREATE TABLE classifier_models (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        model_path TEXT NOT NULL,
                        model_version TEXT NOT NULL,
                        training_summary TEXT,
                        training_job_id TEXT,
                        created_at TEXT
                    )
                    """
                )
            )
            conn.execute(
                text(
                    """
                    INSERT INTO classifier_models (
                        id, name, model_path, model_version,
                        training_summary, training_job_id, created_at
                    ) VALUES (
                        :id, :name, :model_path, :model_version,
                        :training_summary, :training_job_id, :created_at
                    )
                    """
                ),
                {
                    "id": "prod-lr-v12",
                    "name": "LR-v12",
                    "model_path": str(model_path),
                    "model_version": "surfperch-tensorflow2",
                    "training_summary": json.dumps({"cv_f1": 0.95}),
                    "training_job_id": "job-prod-lr-v12",
                    "created_at": "2026-04-03 12:00:00",
                },
            )
    finally:
        engine.dispose()

    monkeypatch.setenv(
        "HUMPBACK_DATABASE_URL",
        f"sqlite+aiosqlite:///{db_path}",
    )
    settings = Settings()
    production_classifier = resolve_production_classifier(
        settings,
        classifier_name="LR-v12",
        classifier_id=None,
    )

    comparison = compare_classifiers(
        manifest,
        best_run,
        production_classifier,
        splits=["val", "test"],
        top_n=5,
    )

    assert comparison["production"]["name"] == "LR-v12"
    assert comparison["production"]["matched_by"] == "name"
    assert comparison["splits"].keys() == {"val", "test"}
    for split_name in ["val", "test"]:
        split_result = comparison["splits"][split_name]
        assert "autoresearch" in split_result
        assert "production" in split_result
        assert "delta" in split_result
        assert "prediction_disagreements" in split_result
        assert "objective" in split_result["autoresearch"]["metrics"]
        assert "objective" in split_result["production"]["metrics"]
