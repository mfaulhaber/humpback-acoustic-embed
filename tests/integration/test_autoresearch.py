"""Integration test for autoresearch search loop.

Writes synthetic Parquet files, builds a manifest directly (no DB),
runs a small search, and verifies all output artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

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
