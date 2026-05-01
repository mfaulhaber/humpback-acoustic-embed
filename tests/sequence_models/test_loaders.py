"""Tests for the source-agnostic HMM interpretation loaders (ADR-059)."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from sklearn.decomposition import PCA

from humpback.models.sequence_models import ContinuousEmbeddingJob, HMMSequenceJob
from humpback.sequence_models.loaders import (
    CrnnRegionLoader,
    OverlayInputs,
    SurfPerchLoader,
    get_loader,
)
from humpback.services.continuous_embedding_service import (
    SOURCE_KIND_REGION_CRNN,
    SOURCE_KIND_SURFPERCH,
)
from humpback.storage import (
    continuous_embedding_parquet_path,
    hmm_sequence_pca_model_path,
    hmm_sequence_states_path,
)


def _fit_pca(n_features: int = 8, n_samples: int = 32) -> PCA:
    rng = np.random.RandomState(0)
    pca = PCA(n_components=4, random_state=0)
    pca.fit(rng.randn(n_samples, n_features).astype(np.float32))
    return pca


def _persist_pca(storage_root: Path, hmm_job_id: str, pca: PCA) -> None:
    dst = hmm_sequence_pca_model_path(storage_root, hmm_job_id)
    dst.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pca, dst)


def _make_hmm_job(job_id: str, l2_normalize: bool = False) -> HMMSequenceJob:
    return HMMSequenceJob(
        id=job_id,
        continuous_embedding_job_id="cej-1",
        n_states=3,
        pca_dims=4,
        pca_whiten=False,
        l2_normalize=l2_normalize,
        covariance_type="diag",
        n_iter=10,
        random_seed=42,
        min_sequence_length_frames=1,
        tol=1e-4,
    )


def _make_cej_surfperch(cej_id: str = "cej-1") -> ContinuousEmbeddingJob:
    return ContinuousEmbeddingJob(
        id=cej_id,
        event_segmentation_job_id="seg-1",
        model_version="surfperch-tensorflow2",
        target_sample_rate=32000,
        encoding_signature=f"sig-{cej_id}",
    )


def _make_cej_crnn(cej_id: str = "cej-1") -> ContinuousEmbeddingJob:
    return ContinuousEmbeddingJob(
        id=cej_id,
        event_segmentation_job_id="seg-1",
        region_detection_job_id="rdj-1",
        model_version="crnn-call-parsing-pytorch",
        target_sample_rate=16000,
        encoding_signature=f"sig-{cej_id}",
    )


def _write_surfperch_fixtures(storage_root: Path, hmm_job_id: str, cej_id: str) -> None:
    """Write a SurfPerch embeddings.parquet + states.parquet fixture."""
    rng = np.random.RandomState(0)
    spans = [(0, 5), (1, 4), (2, 3)]  # span_id, n_windows
    emb_rows: list[dict] = []
    state_rows: list[dict] = []
    for span_id, n in spans:
        for w in range(n):
            emb = rng.randn(8).astype(np.float32).tolist()
            emb_rows.append(
                {
                    "merged_span_id": span_id,
                    "window_index_in_span": w,
                    "audio_file_id": 100 + span_id,
                    "start_timestamp": 1000.0 + span_id * 100 + w,
                    "end_timestamp": 1005.0 + span_id * 100 + w,
                    "embedding": emb,
                }
            )
            state_rows.append(
                {
                    "merged_span_id": span_id,
                    "window_index_in_span": w,
                    "audio_file_id": 100 + span_id,
                    "start_timestamp": 1000.0 + span_id * 100 + w,
                    "end_timestamp": 1005.0 + span_id * 100 + w,
                    "viterbi_state": (span_id + w) % 3,
                    "max_state_probability": 0.5 + 0.05 * w,
                }
            )

    emb_path = continuous_embedding_parquet_path(storage_root, cej_id)
    emb_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(emb_rows), emb_path)

    states_path = hmm_sequence_states_path(storage_root, hmm_job_id)
    states_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(state_rows), states_path)


def _write_crnn_fixtures(
    storage_root: Path, hmm_job_id: str, cej_id: str, *, single_region: bool = False
) -> None:
    """Write a CRNN region-based embeddings.parquet + states.parquet fixture."""
    rng = np.random.RandomState(1)
    if single_region:
        regions = [("region-A", 50.0, 4)]
    else:
        # Out-of-name-order start times to assert sort behavior.
        regions = [
            ("region-zeta", 100.0, 4),
            ("region-alpha", 50.0, 5),
            ("region-mu", 200.0, 3),
        ]
    tier_choices = ["event_core", "near_event", "background"]

    emb_rows: list[dict] = []
    state_rows: list[dict] = []
    for rid, base, n in regions:
        for c in range(n):
            emb = rng.randn(8).astype(np.float32).tolist()
            tier = tier_choices[c % 3]
            emb_rows.append(
                {
                    "region_id": rid,
                    "audio_file_id": 200,
                    "chunk_index_in_region": c,
                    "start_timestamp": base + c * 0.25,
                    "end_timestamp": base + c * 0.25 + 0.25,
                    "embedding": emb,
                }
            )
            state_rows.append(
                {
                    "region_id": rid,
                    "chunk_index_in_region": c,
                    "audio_file_id": 200,
                    "start_timestamp": base + c * 0.25,
                    "end_timestamp": base + c * 0.25 + 0.25,
                    "tier": tier,
                    "viterbi_state": c % 3,
                    "max_state_probability": 0.5 + 0.05 * c,
                }
            )

    emb_path = continuous_embedding_parquet_path(storage_root, cej_id)
    emb_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(emb_rows), emb_path)

    states_path = hmm_sequence_states_path(storage_root, hmm_job_id)
    states_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(state_rows), states_path)


class TestRegistry:
    def test_get_loader_returns_surfperch(self):
        loader = get_loader(SOURCE_KIND_SURFPERCH)
        assert isinstance(loader, SurfPerchLoader)

    def test_get_loader_returns_crnn(self):
        loader = get_loader(SOURCE_KIND_REGION_CRNN)
        assert isinstance(loader, CrnnRegionLoader)

    def test_get_loader_unknown_raises(self):
        with pytest.raises(ValueError, match="unknown sequence-models source kind"):
            get_loader("totally-made-up-source")


class TestSurfPerchLoader:
    def test_returns_overlay_inputs_with_unified_shape(self, tmp_path):
        hmm_job = _make_hmm_job("hmm-1")
        cej = _make_cej_surfperch()
        _persist_pca(tmp_path, hmm_job.id, _fit_pca())
        _write_surfperch_fixtures(tmp_path, hmm_job.id, cej.id)

        result = SurfPerchLoader().load(tmp_path, hmm_job, cej)

        assert isinstance(result, OverlayInputs)
        # 3 spans → 3 sequences.
        assert len(result.raw_sequences) == 3
        assert len(result.viterbi_states) == 3
        # Sequences must be sorted by span id ascending — the loader
        # stringifies the int span id.
        for wm in result.window_metas:
            assert isinstance(wm.sequence_id, str)
            assert wm.extras == {}
            assert wm.audio_file_id is not None
        # Metadata column lengths line up with concatenated sequences.
        n_total = sum(s.shape[0] for s in result.raw_sequences)
        assert len(result.metadata.sequence_ids) == n_total
        assert len(result.metadata.positions_in_sequence) == n_total
        # Stringification: span ids 0, 1, 2 → "0", "1", "2".
        unique_ids = list(dict.fromkeys(result.metadata.sequence_ids))
        assert unique_ids == ["0", "1", "2"]


class TestCrnnRegionLoader:
    def test_orders_regions_by_min_start_timestamp(self, tmp_path):
        hmm_job = _make_hmm_job("hmm-2")
        cej = _make_cej_crnn()
        _persist_pca(tmp_path, hmm_job.id, _fit_pca())
        _write_crnn_fixtures(tmp_path, hmm_job.id, cej.id)

        result = CrnnRegionLoader().load(tmp_path, hmm_job, cej)

        # Regions are seeded with start times alpha=50, zeta=100, mu=200.
        unique_ids = list(dict.fromkeys(result.metadata.sequence_ids))
        assert unique_ids == ["region-alpha", "region-zeta", "region-mu"]

        # Every WindowMeta carries extras["tier"] from one of the three
        # documented tier values.
        for wm in result.window_metas:
            assert wm.extras.get("tier") in {
                "event_core",
                "near_event",
                "background",
            }
            assert isinstance(wm.sequence_id, str)
            assert wm.audio_file_id == 200

    def test_single_region_returns_one_sequence(self, tmp_path):
        hmm_job = _make_hmm_job("hmm-3")
        cej = _make_cej_crnn(cej_id="cej-3")
        _persist_pca(tmp_path, hmm_job.id, _fit_pca())
        _write_crnn_fixtures(tmp_path, hmm_job.id, cej.id, single_region=True)

        result = CrnnRegionLoader().load(tmp_path, hmm_job, cej)

        assert len(result.raw_sequences) == 1
        assert result.raw_sequences[0].shape[0] == 4
        # All chunks share one region id.
        assert set(result.metadata.sequence_ids) == {"region-A"}
