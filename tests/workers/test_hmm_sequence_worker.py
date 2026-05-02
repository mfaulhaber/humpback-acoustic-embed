"""End-to-end tests for the HMM sequence worker.

Creates a synthetic embeddings.parquet (matching the continuous-embedding
producer schema) and runs the worker against it to verify all six
artifact outputs, failure handling, and cancellation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from humpback.config import Settings
from humpback.models.call_parsing import EventSegmentationJob, RegionDetectionJob
from humpback.models.processing import JobStatus
from humpback.models.sequence_models import ContinuousEmbeddingJob, HMMSequenceJob
from humpback.schemas.sequence_models import HMMSequenceJobCreate
from humpback.services.hmm_sequence_service import create_hmm_sequence_job
from humpback.storage import (
    continuous_embedding_dir,
    hmm_sequence_dir,
    hmm_sequence_exemplars_path,
    hmm_sequence_hmm_model_path,
    hmm_sequence_overlay_path,
    hmm_sequence_pca_model_path,
    hmm_sequence_decoded_path,
    hmm_sequence_states_path,
    hmm_sequence_summary_path,
    hmm_sequence_training_log_path,
    hmm_sequence_transition_matrix_path,
)
from humpback.workers.hmm_sequence_worker import run_hmm_sequence_job, run_one_iteration
from tests.fixtures.sequence_models.synthetic_sequences import (
    generate_synthetic_sequences,
)

VECTOR_DIM = 16
N_STATES = 3


def _write_synthetic_embeddings_parquet(
    storage_root: Path,
    ce_job_id: str,
    *,
    n_spans: int = 3,
    windows_per_span: int = 40,
    vector_dim: int = VECTOR_DIM,
    seed: int = 42,
) -> Path:
    """Write a fake embeddings.parquet matching the continuous-embedding schema."""
    seqs, _ = generate_synthetic_sequences(
        n_states=N_STATES,
        n_sequences=n_spans,
        min_length=windows_per_span,
        max_length=windows_per_span,
        vector_dim=vector_dim,
        cluster_separation=8.0,
        seed=seed,
    )

    rows = []
    t = 1000.0
    for span_id, seq in enumerate(seqs):
        for win_idx in range(len(seq)):
            rows.append(
                {
                    "merged_span_id": span_id,
                    "window_index_in_span": win_idx,
                    "audio_file_id": None,
                    "start_timestamp": t,
                    "end_timestamp": t + 5.0,
                    "is_in_pad": win_idx < 3,
                    "event_id": f"evt-{span_id}",
                    "embedding": seq[win_idx].tolist(),
                }
            )
            t += 1.0

    schema = pa.schema(
        [
            pa.field("merged_span_id", pa.int32()),
            pa.field("window_index_in_span", pa.int32()),
            pa.field("audio_file_id", pa.int32()),
            pa.field("start_timestamp", pa.float64()),
            pa.field("end_timestamp", pa.float64()),
            pa.field("is_in_pad", pa.bool_()),
            pa.field("event_id", pa.string()),
            pa.field("embedding", pa.list_(pa.float32())),
        ]
    )
    table = pa.Table.from_pylist(rows, schema=schema)

    out_dir = continuous_embedding_dir(storage_root, ce_job_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "embeddings.parquet"
    pq.write_table(table, path)
    return path


async def _seed_complete_ce_job(session, settings) -> ContinuousEmbeddingJob:
    """Create a completed ContinuousEmbeddingJob with a synthetic parquet."""
    region_job = RegionDetectionJob(
        status=JobStatus.complete.value,
        hydrophone_id="rpi_orcasound_lab",
        start_timestamp=1000.0,
        end_timestamp=4000.0,
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

    ce_job = ContinuousEmbeddingJob(
        event_segmentation_job_id=seg_job.id,
        model_version="surfperch-tensorflow2",
        window_size_seconds=5.0,
        hop_seconds=1.0,
        pad_seconds=2.0,
        target_sample_rate=32000,
        encoding_signature=f"test-sig-{seg_job.id}",
        status=JobStatus.complete.value,
        vector_dim=VECTOR_DIM,
        total_windows=120,
        parquet_path="set-below",
    )
    session.add(ce_job)
    await session.commit()
    await session.refresh(ce_job)

    parquet_path = _write_synthetic_embeddings_parquet(settings.storage_root, ce_job.id)
    ce_job.parquet_path = str(parquet_path)
    await session.commit()
    return ce_job


async def _create_hmm_job(session, ce_job_id: str, **overrides: Any) -> HMMSequenceJob:
    defaults: dict[str, Any] = dict(
        continuous_embedding_job_id=ce_job_id,
        n_states=N_STATES,
        pca_dims=8,
        n_iter=20,
        random_seed=42,
        min_sequence_length_frames=5,
    )
    defaults.update(overrides)
    payload = HMMSequenceJobCreate(**defaults)
    return await create_hmm_sequence_job(session, payload)


async def test_happy_path_produces_all_artifacts(session, settings):
    ce_job = await _seed_complete_ce_job(session, settings)
    job = await _create_hmm_job(session, ce_job.id)

    await run_hmm_sequence_job(session, job, settings)
    await session.refresh(job)

    assert job.status == JobStatus.complete.value
    assert job.error_message is None
    assert job.n_train_sequences is not None and job.n_train_sequences > 0
    assert job.n_train_frames is not None and job.n_train_frames > 0
    assert job.n_decoded_sequences == 3
    assert job.train_log_likelihood is not None
    assert job.artifact_dir is not None

    job_id = job.id
    sr = settings.storage_root
    assert hmm_sequence_states_path(sr, job_id).exists()
    assert hmm_sequence_pca_model_path(sr, job_id).exists()
    assert hmm_sequence_hmm_model_path(sr, job_id).exists()
    assert hmm_sequence_transition_matrix_path(sr, job_id).exists()
    assert hmm_sequence_summary_path(sr, job_id).exists()
    assert hmm_sequence_training_log_path(sr, job_id).exists()


async def test_decoded_parquet_schema(session, settings):
    ce_job = await _seed_complete_ce_job(session, settings)
    job = await _create_hmm_job(session, ce_job.id)

    await run_hmm_sequence_job(session, job, settings)

    decoded_path = hmm_sequence_decoded_path(settings.storage_root, job.id)
    assert decoded_path.exists()
    states = pq.read_table(decoded_path)
    expected_cols = {
        "merged_span_id",
        "window_index_in_span",
        "audio_file_id",
        "start_timestamp",
        "end_timestamp",
        "is_in_pad",
        "event_id",
        "label",
        "state_posterior",
        "max_state_probability",
        "was_used_for_training",
    }
    assert set(states.column_names) == expected_cols
    assert states.num_rows == 120  # 3 spans × 40 windows

    posteriors = states.column("state_posterior").to_pylist()
    assert all(len(p) == N_STATES for p in posteriors)


async def test_summary_json_content(session, settings):
    ce_job = await _seed_complete_ce_job(session, settings)
    job = await _create_hmm_job(session, ce_job.id)

    await run_hmm_sequence_job(session, job, settings)

    summary_path = hmm_sequence_summary_path(settings.storage_root, job.id)
    summary = json.loads(summary_path.read_text())
    assert summary["n_states"] == N_STATES
    assert len(summary["states"]) == N_STATES
    total_occ = sum(s["occupancy"] for s in summary["states"])
    assert abs(total_occ - 1.0) < 1e-6


async def test_transition_matrix_shape(session, settings):
    ce_job = await _seed_complete_ce_job(session, settings)
    job = await _create_hmm_job(session, ce_job.id)

    await run_hmm_sequence_job(session, job, settings)

    tm = np.load(hmm_sequence_transition_matrix_path(settings.storage_root, job.id))
    assert tm.shape == (N_STATES, N_STATES)
    np.testing.assert_allclose(tm.sum(axis=1), 1.0, atol=1e-6)


async def test_training_log_content(session, settings):
    ce_job = await _seed_complete_ce_job(session, settings)
    job = await _create_hmm_job(session, ce_job.id)

    await run_hmm_sequence_job(session, job, settings)

    log_path = hmm_sequence_training_log_path(settings.storage_root, job.id)
    log = json.loads(log_path.read_text())
    assert log["n_states"] == N_STATES
    assert log["pca_dims"] == 8
    assert log["random_seed"] == 42
    assert log["n_train_sequences"] > 0


async def test_failure_marks_job_failed(session, settings):
    ce_job = await _seed_complete_ce_job(session, settings)
    job = await _create_hmm_job(session, ce_job.id, pca_dims=8)

    # Corrupt the parquet to force a failure
    assert ce_job.parquet_path is not None
    parquet_path = Path(ce_job.parquet_path)
    parquet_path.write_text("not a parquet file")

    await run_hmm_sequence_job(session, job, settings)
    await session.refresh(job)

    assert job.status == JobStatus.failed.value
    assert job.error_message is not None


async def test_no_temp_files_on_success(session, settings):
    ce_job = await _seed_complete_ce_job(session, settings)
    job = await _create_hmm_job(session, ce_job.id)

    await run_hmm_sequence_job(session, job, settings)

    job_dir = hmm_sequence_dir(settings.storage_root, job.id)
    leftover = list(job_dir.glob("*.tmp")) + list(job_dir.glob("*.tmp.*"))
    assert leftover == []


async def test_cancellation_skips_work(session, settings):
    ce_job = await _seed_complete_ce_job(session, settings)
    job = await _create_hmm_job(session, ce_job.id)

    job.status = JobStatus.canceled.value
    await session.commit()

    await run_hmm_sequence_job(session, job, settings)

    assert not hmm_sequence_states_path(settings.storage_root, job.id).exists()
    await session.refresh(job)
    assert job.status == JobStatus.canceled.value


async def test_run_one_iteration_claims_and_runs(session, settings):
    ce_job = await _seed_complete_ce_job(session, settings)
    job = await _create_hmm_job(session, ce_job.id)
    assert job.status == JobStatus.queued.value

    claimed = await run_one_iteration(session, settings)
    assert claimed is not None
    assert claimed.id == job.id

    await session.refresh(job)
    assert job.status == JobStatus.complete.value


# ---------------------------------------------------------------------------
# CRNN-source worker tests (ADR-057 + ADR-059)
# ---------------------------------------------------------------------------

CRNN_VECTOR_DIM = 16
CRNN_TIERS = ("event_core", "near_event", "background")


def _write_synthetic_crnn_embeddings_parquet(
    storage_root: Path,
    ce_job_id: str,
    *,
    n_regions: int = 3,
    chunks_per_region: int = 50,
    vector_dim: int = CRNN_VECTOR_DIM,
    seed: int = 7,
) -> Path:
    """Write a fake CRNN-source embeddings.parquet matching ADR-057 schema."""
    seqs, _ = generate_synthetic_sequences(
        n_states=N_STATES,
        n_sequences=n_regions,
        min_length=chunks_per_region,
        max_length=chunks_per_region,
        vector_dim=vector_dim,
        cluster_separation=8.0,
        seed=seed,
    )

    rows = []
    for region_idx, seq in enumerate(seqs):
        region_id = f"region-{region_idx:02d}"
        base = 1000.0 + region_idx * 100.0
        for chunk_idx in range(len(seq)):
            tier = CRNN_TIERS[chunk_idx % len(CRNN_TIERS)]
            rows.append(
                {
                    "region_id": region_id,
                    "chunk_index_in_region": chunk_idx,
                    "audio_file_id": 200 + region_idx,
                    "start_timestamp": base + chunk_idx * 0.25,
                    "end_timestamp": base + chunk_idx * 0.25 + 0.25,
                    "is_in_pad": False,
                    "tier": tier,
                    "embedding": seq[chunk_idx].tolist(),
                }
            )

    schema = pa.schema(
        [
            pa.field("region_id", pa.string()),
            pa.field("chunk_index_in_region", pa.int32()),
            pa.field("audio_file_id", pa.int32(), nullable=True),
            pa.field("start_timestamp", pa.float64()),
            pa.field("end_timestamp", pa.float64()),
            pa.field("is_in_pad", pa.bool_()),
            pa.field("tier", pa.string()),
            pa.field("embedding", pa.list_(pa.float32())),
        ]
    )
    table = pa.Table.from_pylist(rows, schema=schema)

    out_dir = continuous_embedding_dir(storage_root, ce_job_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "embeddings.parquet"
    pq.write_table(table, path)
    return path


async def _seed_complete_crnn_ce_job(session, settings) -> ContinuousEmbeddingJob:
    """Create a completed CRNN-source ContinuousEmbeddingJob with synthetic parquet."""
    region_job = RegionDetectionJob(
        status=JobStatus.complete.value,
        hydrophone_id="rpi_orcasound_lab",
        start_timestamp=1000.0,
        end_timestamp=4000.0,
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

    ce_job = ContinuousEmbeddingJob(
        event_segmentation_job_id=seg_job.id,
        region_detection_job_id=region_job.id,
        model_version="crnn-call-parsing-pytorch",
        window_size_seconds=0.25,
        hop_seconds=0.25,
        pad_seconds=0.0,
        target_sample_rate=16000,
        encoding_signature=f"crnn-test-sig-{seg_job.id}",
        status=JobStatus.complete.value,
        vector_dim=CRNN_VECTOR_DIM,
        total_windows=150,
        parquet_path="set-below",
    )
    session.add(ce_job)
    await session.commit()
    await session.refresh(ce_job)

    parquet_path = _write_synthetic_crnn_embeddings_parquet(
        settings.storage_root, ce_job.id
    )
    ce_job.parquet_path = str(parquet_path)
    await session.commit()
    return ce_job


async def _create_crnn_hmm_job(
    session, ce_job_id: str, **overrides: Any
) -> HMMSequenceJob:
    defaults: dict[str, Any] = dict(
        continuous_embedding_job_id=ce_job_id,
        n_states=N_STATES,
        pca_dims=8,
        n_iter=20,
        random_seed=42,
        min_sequence_length_frames=1,
        training_mode="full_region",
        subsequence_length_chunks=16,
        subsequence_stride_chunks=8,
        target_train_chunks=10_000,
    )
    defaults.update(overrides)
    payload = HMMSequenceJobCreate(**defaults)
    return await create_hmm_sequence_job(session, payload)


async def test_crnn_happy_path_writes_overlay_and_exemplars(session, settings):
    ce_job = await _seed_complete_crnn_ce_job(session, settings)
    job = await _create_crnn_hmm_job(session, ce_job.id)

    await run_hmm_sequence_job(session, job, settings)
    await session.refresh(job)

    assert job.status == JobStatus.complete.value
    assert job.error_message is None

    job_id = job.id
    sr = settings.storage_root
    # Standard HMM artifacts.
    assert hmm_sequence_states_path(sr, job_id).exists()
    assert hmm_sequence_pca_model_path(sr, job_id).exists()
    assert hmm_sequence_hmm_model_path(sr, job_id).exists()
    assert hmm_sequence_transition_matrix_path(sr, job_id).exists()
    assert hmm_sequence_summary_path(sr, job_id).exists()
    assert hmm_sequence_training_log_path(sr, job_id).exists()
    # Plan Task 7: interpretation artifacts must be on disk after a CRNN
    # HMM job runs to completion.
    overlay_path = hmm_sequence_overlay_path(sr, job_id)
    exemplars_path = hmm_sequence_exemplars_path(sr, job_id)
    assert overlay_path.exists(), "overlay.parquet missing for CRNN job"
    assert exemplars_path.exists(), "exemplars.json missing for CRNN job"

    overlay_table = pq.read_table(overlay_path)
    assert "sequence_id" in overlay_table.column_names
    assert "position_in_sequence" in overlay_table.column_names
    # CRNN overlay sequence_ids are region strings, never integers.
    sample_ids = overlay_table.column("sequence_id").to_pylist()[:5]
    assert all(isinstance(s, str) and s.startswith("region-") for s in sample_ids)

    exemplars_payload = json.loads(exemplars_path.read_text(encoding="utf-8"))
    assert exemplars_payload["n_states"] == N_STATES
    # Every CRNN exemplar carries extras["tier"] from one of the three
    # documented tier values.
    states = exemplars_payload["states"]
    flat = [rec for recs in states.values() for rec in recs]
    assert flat, "no exemplar records produced"
    for rec in flat:
        assert isinstance(rec.get("sequence_id"), str)
        assert rec["sequence_id"].startswith("region-")
        assert rec.get("extras", {}).get("tier") in {
            "event_core",
            "near_event",
            "background",
        }


async def test_crnn_interpretation_failure_keeps_job_complete(
    session, settings, monkeypatch
):
    """Plan Task 7: injecting a failure in generate_interpretations must NOT
    flip the CRNN HMM job from complete to failed."""
    ce_job = await _seed_complete_crnn_ce_job(session, settings)
    job = await _create_crnn_hmm_job(session, ce_job.id)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("synthetic interpretation failure")

    # Patch the symbol the worker imports at call time.
    import humpback.services.hmm_sequence_service as svc

    monkeypatch.setattr(svc, "generate_interpretations", _boom)

    await run_hmm_sequence_job(session, job, settings)
    await session.refresh(job)

    # Job must still be marked complete; interpretation failure is non-fatal.
    assert job.status == JobStatus.complete.value
    assert job.error_message is None

    job_id = job.id
    sr = settings.storage_root
    # Core HMM artifacts persist.
    assert hmm_sequence_states_path(sr, job_id).exists()
    assert hmm_sequence_summary_path(sr, job_id).exists()
    # Interpretation artifacts must NOT exist (we forced the call to raise).
    assert not hmm_sequence_overlay_path(sr, job_id).exists()
    assert not hmm_sequence_exemplars_path(sr, job_id).exists()


@pytest.fixture
def settings(tmp_path):
    db_path = tmp_path / "test.db"
    return Settings(
        database_url=f"sqlite+aiosqlite:///{db_path}",
        storage_root=tmp_path / "storage",
        use_real_model=False,
    )
