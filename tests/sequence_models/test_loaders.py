"""Tests for the source-agnostic HMM interpretation loaders (ADR-059, ADR-060, ADR-061)."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from sklearn.decomposition import PCA

from humpback.models.call_parsing import EventSegmentationJob, RegionDetectionJob
from humpback.models.processing import JobStatus
from humpback.models.sequence_models import ContinuousEmbeddingJob, HMMSequenceJob
from humpback.sequence_models.loaders import (
    CrnnRegionLoader,
    LabelDistributionInputs,
    OverlayInputs,
    SurfPerchLoader,
    get_loader,
    read_decoded_parquet,
)
from humpback.services.continuous_embedding_service import (
    SOURCE_KIND_REGION_CRNN,
    SOURCE_KIND_SURFPERCH,
)
from humpback.storage import (
    continuous_embedding_parquet_path,
    hmm_sequence_decoded_path,
    hmm_sequence_legacy_states_path,
    hmm_sequence_pca_model_path,
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


def _write_surfperch_fixtures(
    storage_root: Path,
    hmm_job_id: str,
    cej_id: str,
    *,
    label_column: str = "label",
    decoded_filename: str = "decoded.parquet",
) -> None:
    """Write a SurfPerch embeddings.parquet + decoded sequence parquet fixture.

    ``label_column`` and ``decoded_filename`` allow exercising the legacy
    backwards-read shim (legacy on-disk layout uses ``viterbi_state`` +
    ``states.parquet``).
    """
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
                    label_column: (span_id + w) % 3,
                    "max_state_probability": 0.5 + 0.05 * w,
                }
            )

    emb_path = continuous_embedding_parquet_path(storage_root, cej_id)
    emb_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(emb_rows), emb_path)

    decoded_path = hmm_sequence_decoded_path(storage_root, hmm_job_id).with_name(
        decoded_filename
    )
    decoded_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(state_rows), decoded_path)


def _write_crnn_fixtures(
    storage_root: Path,
    hmm_job_id: str,
    cej_id: str,
    *,
    single_region: bool = False,
    label_column: str = "label",
    decoded_filename: str = "decoded.parquet",
) -> Path:
    """Write a CRNN region-based embeddings.parquet + decoded fixture.

    Returns the path to the decoded parquet so per-test variants (e.g.
    masked-transformer per-k paths) can read it explicitly.
    """
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
                    label_column: c % 3,
                    "max_state_probability": 0.5 + 0.05 * c,
                }
            )

    emb_path = continuous_embedding_parquet_path(storage_root, cej_id)
    emb_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(emb_rows), emb_path)

    decoded_path = hmm_sequence_decoded_path(storage_root, hmm_job_id).with_name(
        decoded_filename
    )
    decoded_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(state_rows), decoded_path)
    return decoded_path


class TestRegistry:
    def test_get_loader_returns_surfperch(self, tmp_path):
        loader = get_loader(SOURCE_KIND_SURFPERCH, tmp_path / "decoded.parquet")
        assert isinstance(loader, SurfPerchLoader)

    def test_get_loader_returns_crnn(self, tmp_path):
        loader = get_loader(SOURCE_KIND_REGION_CRNN, tmp_path / "decoded.parquet")
        assert isinstance(loader, CrnnRegionLoader)

    def test_get_loader_unknown_raises(self, tmp_path):
        with pytest.raises(ValueError, match="unknown sequence-models source kind"):
            get_loader("totally-made-up-source", tmp_path / "decoded.parquet")


class TestSurfPerchLoader:
    def test_returns_overlay_inputs_with_unified_shape(self, tmp_path):
        hmm_job = _make_hmm_job("hmm-1")
        cej = _make_cej_surfperch()
        _persist_pca(tmp_path, hmm_job.id, _fit_pca())
        _write_surfperch_fixtures(tmp_path, hmm_job.id, cej.id)

        decoded_path = hmm_sequence_decoded_path(tmp_path, hmm_job.id)
        result = SurfPerchLoader(decoded_artifact_path=str(decoded_path)).load(
            tmp_path, hmm_job, cej
        )

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
        decoded_path = _write_crnn_fixtures(tmp_path, hmm_job.id, cej.id)

        result = CrnnRegionLoader(decoded_artifact_path=str(decoded_path)).load(
            tmp_path, hmm_job, cej
        )

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
        decoded_path = _write_crnn_fixtures(
            tmp_path, hmm_job.id, cej.id, single_region=True
        )

        result = CrnnRegionLoader(decoded_artifact_path=str(decoded_path)).load(
            tmp_path, hmm_job, cej
        )

        assert len(result.raw_sequences) == 1
        assert result.raw_sequences[0].shape[0] == 4
        # All chunks share one region id.
        assert set(result.metadata.sequence_ids) == {"region-A"}

    def test_loader_reads_explicit_per_k_decoded_path(self, tmp_path):
        """The loader honors an explicit ``decoded_artifact_path`` so a
        masked-transformer per-k bundle can plug in without faking an HMM
        directory layout.
        """
        hmm_job = _make_hmm_job("hmm-mt-pseudo")
        cej = _make_cej_crnn(cej_id="cej-mt")
        _persist_pca(tmp_path, hmm_job.id, _fit_pca())
        # Write a "per-k" decoded.parquet to a custom path under tmp_path.
        per_k_dir = tmp_path / "masked_transformer_jobs" / "mt-1" / "k50"
        per_k_dir.mkdir(parents=True, exist_ok=True)
        per_k_decoded = per_k_dir / "decoded.parquet"
        # Reuse the writer but override target filename via a custom dir;
        # construct a synthetic table directly so we hit the explicit path.
        rng = np.random.RandomState(2)
        rows = []
        for c in range(4):
            rng.randn(4)  # advance rng for determinism
            rows.append(
                {
                    "region_id": "region-mt",
                    "chunk_index_in_region": c,
                    "audio_file_id": 200,
                    "start_timestamp": 50.0 + c * 0.25,
                    "end_timestamp": 50.0 + c * 0.25 + 0.25,
                    "tier": "event_core",
                    "label": c % 3,
                    "max_state_probability": 0.7,
                }
            )
        pq.write_table(pa.Table.from_pylist(rows), per_k_decoded)

        # Embeddings live at the standard CEJ path so the loader can find
        # them without trickery.
        emb_path = continuous_embedding_parquet_path(tmp_path, cej.id)
        emb_path.parent.mkdir(parents=True, exist_ok=True)
        emb_rows = []
        for c in range(4):
            emb_rows.append(
                {
                    "region_id": "region-mt",
                    "audio_file_id": 200,
                    "chunk_index_in_region": c,
                    "start_timestamp": 50.0 + c * 0.25,
                    "end_timestamp": 50.0 + c * 0.25 + 0.25,
                    "embedding": np.random.RandomState(3 + c)
                    .randn(8)
                    .astype(np.float32)
                    .tolist(),
                }
            )
        pq.write_table(pa.Table.from_pylist(emb_rows), emb_path)

        result = CrnnRegionLoader(decoded_artifact_path=str(per_k_decoded)).load(
            tmp_path, hmm_job, cej
        )
        assert len(result.raw_sequences) == 1
        assert set(result.metadata.sequence_ids) == {"region-mt"}

    def test_loader_backwards_read_shim_reads_legacy_states_parquet(self, tmp_path):
        """Pre-ADR-061 jobs have ``states.parquet`` with a
        ``viterbi_state`` column. The loader's read shim must transparently
        rename to ``label`` without rewriting the file."""
        hmm_job = _make_hmm_job("hmm-legacy")
        cej = _make_cej_crnn(cej_id="cej-legacy")
        _persist_pca(tmp_path, hmm_job.id, _fit_pca())
        # Simulate legacy on-disk layout: states.parquet w/ viterbi_state.
        _write_crnn_fixtures(
            tmp_path,
            hmm_job.id,
            cej.id,
            label_column="viterbi_state",
            decoded_filename="states.parquet",
        )
        # Caller still asks for decoded.parquet — file does not exist on
        # disk, so the shim falls back to the legacy states.parquet.
        decoded_path = hmm_sequence_decoded_path(tmp_path, hmm_job.id)
        legacy_path = hmm_sequence_legacy_states_path(tmp_path, hmm_job.id)
        assert not decoded_path.exists()
        assert legacy_path.exists()

        # Direct shim test: returns a label-renamed table.
        table = read_decoded_parquet(decoded_path)
        assert "label" in table.column_names
        assert "viterbi_state" not in table.column_names

        # Loader uses the shim under the hood and produces overlay inputs
        # without raising.
        result = CrnnRegionLoader(decoded_artifact_path=str(decoded_path)).load(
            tmp_path, hmm_job, cej
        )
        assert len(result.viterbi_states) > 0


class TestSurfPerchLabelDistributionLoader:
    async def test_returns_aligned_state_rows_and_hydrophone(self, session, tmp_path):
        region_job = RegionDetectionJob(
            status=JobStatus.complete.value,
            hydrophone_id="rpi_orcasound_lab",
            start_timestamp=1000.0,
            end_timestamp=1300.0,
        )
        session.add(region_job)
        await session.flush()
        seg_job = EventSegmentationJob(
            status=JobStatus.complete.value,
            region_detection_job_id=region_job.id,
        )
        session.add(seg_job)
        await session.commit()
        await session.refresh(seg_job)

        cej = ContinuousEmbeddingJob(
            event_segmentation_job_id=seg_job.id,
            model_version="surfperch-tensorflow2",
            window_size_seconds=5.0,
            hop_seconds=1.0,
            pad_seconds=2.0,
            target_sample_rate=32000,
            encoding_signature=f"sig-{seg_job.id}",
            status=JobStatus.complete.value,
        )
        session.add(cej)
        await session.commit()
        await session.refresh(cej)

        hmm_job = _make_hmm_job("hmm-sp-ld")
        decoded_path = hmm_sequence_decoded_path(tmp_path, hmm_job.id)
        decoded_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.table(
                {
                    "start_timestamp": pa.array([1010.0, 1100.0], type=pa.float64()),
                    "end_timestamp": pa.array([1015.0, 1105.0], type=pa.float64()),
                    "label": pa.array([0, 1], type=pa.int16()),
                }
            ),
            decoded_path,
        )

        result = await SurfPerchLoader(
            decoded_artifact_path=str(decoded_path)
        ).load_label_distribution_inputs(session, tmp_path, hmm_job, cej)

        assert isinstance(result, LabelDistributionInputs)
        assert result.hydrophone_id == "rpi_orcasound_lab"
        assert result.tier_per_row is None
        assert len(result.state_rows) == 2
        assert result.state_rows[0]["start_timestamp"] == 1010.0
        assert result.state_rows[0]["viterbi_state"] == 0
        assert result.state_rows[1]["viterbi_state"] == 1

    async def test_missing_event_segmentation_raises(self, session, tmp_path):
        cej = ContinuousEmbeddingJob(
            event_segmentation_job_id="does-not-exist",
            model_version="surfperch-tensorflow2",
            window_size_seconds=5.0,
            hop_seconds=1.0,
            pad_seconds=2.0,
            target_sample_rate=32000,
            encoding_signature="sig-missing-esj",
            status=JobStatus.complete.value,
        )
        session.add(cej)
        await session.commit()
        await session.refresh(cej)

        hmm_job = _make_hmm_job("hmm-sp-missing")
        decoded_path = hmm_sequence_decoded_path(tmp_path, hmm_job.id)
        with pytest.raises(ValueError, match="EventSegmentationJob not found"):
            await SurfPerchLoader(
                decoded_artifact_path=str(decoded_path)
            ).load_label_distribution_inputs(session, tmp_path, hmm_job, cej)


class TestCrnnRegionLabelDistributionLoader:
    async def test_returns_tier_per_row_aligned_with_state_rows(
        self, session, tmp_path
    ):
        region_job = RegionDetectionJob(
            status=JobStatus.complete.value,
            hydrophone_id="rpi_orcasound_lab",
            start_timestamp=1000.0,
            end_timestamp=1300.0,
        )
        session.add(region_job)
        await session.commit()
        await session.refresh(region_job)

        cej = ContinuousEmbeddingJob(
            event_segmentation_job_id="seg-anything",
            region_detection_job_id=region_job.id,
            model_version="crnn-call-parsing-pytorch",
            window_size_seconds=0.25,
            hop_seconds=0.25,
            pad_seconds=0.0,
            target_sample_rate=16000,
            encoding_signature=f"sig-crnn-{region_job.id}",
            status=JobStatus.complete.value,
        )
        session.add(cej)
        await session.commit()
        await session.refresh(cej)

        hmm_job = _make_hmm_job("hmm-crnn-ld")
        decoded_path = hmm_sequence_decoded_path(tmp_path, hmm_job.id)
        decoded_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.table(
                {
                    "start_timestamp": pa.array(
                        [1000.0, 1000.5, 1001.0], type=pa.float64()
                    ),
                    "end_timestamp": pa.array(
                        [1000.5, 1001.0, 1001.5], type=pa.float64()
                    ),
                    "label": pa.array([0, 1, 0], type=pa.int16()),
                    "tier": pa.array(
                        ["event_core", "near_event", "background"], type=pa.string()
                    ),
                }
            ),
            decoded_path,
        )

        result = await CrnnRegionLoader(
            decoded_artifact_path=str(decoded_path)
        ).load_label_distribution_inputs(session, tmp_path, hmm_job, cej)

        assert isinstance(result, LabelDistributionInputs)
        assert result.hydrophone_id == "rpi_orcasound_lab"
        assert result.tier_per_row == ["event_core", "near_event", "background"]
        assert result.tier_per_row is not None
        assert len(result.state_rows) == 3
        assert len(result.tier_per_row) == len(result.state_rows)
        assert result.state_rows[0]["viterbi_state"] == 0
        assert result.state_rows[1]["viterbi_state"] == 1

    async def test_missing_region_detection_job_raises(self, session, tmp_path):
        cej = ContinuousEmbeddingJob(
            event_segmentation_job_id="seg-anything",
            region_detection_job_id="does-not-exist",
            model_version="crnn-call-parsing-pytorch",
            window_size_seconds=0.25,
            hop_seconds=0.25,
            pad_seconds=0.0,
            target_sample_rate=16000,
            encoding_signature="sig-crnn-missing",
            status=JobStatus.complete.value,
        )
        session.add(cej)
        await session.commit()
        await session.refresh(cej)

        hmm_job = _make_hmm_job("hmm-crnn-missing")
        decoded_path = hmm_sequence_decoded_path(tmp_path, hmm_job.id)
        with pytest.raises(ValueError, match="RegionDetectionJob not found"):
            await CrnnRegionLoader(
                decoded_artifact_path=str(decoded_path)
            ).load_label_distribution_inputs(session, tmp_path, hmm_job, cej)

    async def test_null_region_detection_job_id_raises(self, session, tmp_path):
        cej = ContinuousEmbeddingJob(
            event_segmentation_job_id="seg-anything",
            region_detection_job_id=None,
            model_version="crnn-call-parsing-pytorch",
            window_size_seconds=0.25,
            hop_seconds=0.25,
            pad_seconds=0.0,
            target_sample_rate=16000,
            encoding_signature="sig-crnn-null",
            status=JobStatus.complete.value,
        )
        session.add(cej)
        await session.commit()
        await session.refresh(cej)

        hmm_job = _make_hmm_job("hmm-crnn-null")
        decoded_path = hmm_sequence_decoded_path(tmp_path, hmm_job.id)
        with pytest.raises(ValueError, match="missing region_detection_job_id"):
            await CrnnRegionLoader(
                decoded_artifact_path=str(decoded_path)
            ).load_label_distribution_inputs(session, tmp_path, hmm_job, cej)
