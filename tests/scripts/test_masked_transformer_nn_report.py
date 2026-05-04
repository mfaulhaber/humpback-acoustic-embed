"""Tests for masked-transformer nearest-neighbor diagnostic script."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "masked_transformer_nn_report.py"
)
spec = importlib.util.spec_from_file_location(
    "masked_transformer_nn_report", SCRIPT_PATH
)
assert spec is not None and spec.loader is not None
nn_report = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = nn_report
spec.loader.exec_module(nn_report)


def _row(
    idx: int,
    *,
    region_id: str,
    event_id: str,
    label: str,
    duration: float = 1.0,
) -> dict:
    return {
        "idx": idx,
        "region_id": region_id,
        "chunk_index": idx,
        "start_timestamp": float(idx),
        "end_timestamp": float(idx) + 0.25,
        "center_timestamp": float(idx) + 0.125,
        "tier": "event_core",
        "hydrophone_id": "hydrophone",
        "token": idx,
        "token_confidence": 1.0,
        "call_probability": 0.9,
        "event_overlap_fraction": 1.0,
        "nearest_event_id": event_id,
        "event_id": event_id,
        "event_duration": duration,
        "human_types": (label,),
        "effective_types": (label,),
    }


def test_exclude_same_event_region_masks_local_neighbors():
    rows = [
        _row(0, region_id="same-region", event_id="event-a", label="Moan"),
        _row(1, region_id="same-region", event_id="event-b", label="Moan"),
        _row(2, region_id="other-region", event_id="event-c", label="Moan"),
        _row(3, region_id="third-region", event_id="event-d", label="Growl"),
    ]
    vectors = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.99, 0.01, 0.0],
            [0.98, 0.02, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    vectors = nn_report._normalize_rows(vectors)

    _summaries, neighbors, _sample_indices, _pool = nn_report._analyze_neighbors(
        rows,
        vectors,
        n_samples=1,
        topn=2,
        seed=0,
        sample_indices=[0],
        exclude_same_event_region=True,
    )

    assert [record["neighbor_idx"] for record in neighbors] == [2, 3]
    assert all(not record["same_region"] for record in neighbors)
    assert all(not record["same_event"] for record in neighbors)


def test_pc_removal_and_whitening_variants_are_normalized():
    rng = np.random.default_rng(0)
    raw = rng.normal(size=(24, 8)).astype(np.float32)

    variants = nn_report._build_pc_removal_variants(
        raw,
        seed=0,
        remove_counts=[1, 3],
    )
    variants["whiten_pca"] = nn_report._whiten_embeddings(
        raw,
        n_components=4,
        seed=0,
    )

    assert set(variants) == {
        "raw_l2",
        "centered_l2",
        "remove_pc1",
        "remove_pc3",
        "whiten_pca",
    }
    for values in variants.values():
        assert values.shape[0] == raw.shape[0]
        np.testing.assert_allclose(
            np.linalg.norm(values, axis=1),
            np.ones(raw.shape[0]),
            atol=1e-5,
        )
