"""End-to-end tests for the masked-transformer worker (ADR-061)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from humpback.models.call_parsing import EventSegmentationJob, RegionDetectionJob
from humpback.models.processing import JobStatus
from humpback.models.sequence_models import ContinuousEmbeddingJob
from humpback.services.masked_transformer_service import (
    create_masked_transformer_job,
)
from humpback.storage import (
    continuous_embedding_dir,
    masked_transformer_contextual_embeddings_path,
    masked_transformer_dir,
    masked_transformer_k_decoded_path,
    masked_transformer_k_dir,
    masked_transformer_k_kmeans_path,
    masked_transformer_k_run_lengths_path,
    masked_transformer_loss_curve_path,
    masked_transformer_model_path,
    masked_transformer_reconstruction_error_path,
)
from humpback.workers.masked_transformer_worker import run_masked_transformer_job


CRNN_VECTOR_DIM = 16
CRNN_TIERS = ("event_core", "near_event", "background")


def _write_synthetic_crnn_embeddings_parquet(
    storage_root: Path,
    ce_job_id: str,
    *,
    n_regions: int = 3,
    chunks_per_region: int = 24,
    vector_dim: int = CRNN_VECTOR_DIM,
    seed: int = 21,
) -> Path:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for r in range(n_regions):
        rid = f"region-{r:02d}"
        base = 100.0 + r * 10.0
        # Use cluster offsets so k-means has structure to find.
        cluster_centers = rng.standard_normal((3, vector_dim)) * 4.0
        for c in range(chunks_per_region):
            tier = CRNN_TIERS[c % len(CRNN_TIERS)]
            center = cluster_centers[c % 3]
            emb = (center + 0.3 * rng.standard_normal(vector_dim)).astype(np.float32)
            rows.append(
                {
                    "region_id": rid,
                    "chunk_index_in_region": c,
                    "audio_file_id": 200 + r,
                    "start_timestamp": base + c * 0.25,
                    "end_timestamp": base + c * 0.25 + 0.25,
                    "is_in_pad": False,
                    "tier": tier,
                    "embedding": emb.tolist(),
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


async def _seed_crnn_ce_job(session, settings) -> ContinuousEmbeddingJob:
    region_job = RegionDetectionJob(
        status=JobStatus.complete.value,
        hydrophone_id="rpi_orcasound_lab",
        start_timestamp=100.0,
        end_timestamp=400.0,
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
        target_sample_rate=16000,
        encoding_signature=f"crnn-mt-sig-{seg_job.id}",
        status=JobStatus.complete.value,
        vector_dim=CRNN_VECTOR_DIM,
        total_chunks=72,
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


async def _seed_surfperch_ce_job(session) -> ContinuousEmbeddingJob:
    region_job = RegionDetectionJob(
        status=JobStatus.complete.value,
        hydrophone_id="rpi_orcasound_lab",
        start_timestamp=100.0,
        end_timestamp=400.0,
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
        target_sample_rate=32000,
        encoding_signature=f"sp-mt-sig-{seg_job.id}",
        status=JobStatus.complete.value,
    )
    session.add(ce_job)
    await session.commit()
    await session.refresh(ce_job)
    return ce_job


def _tiny_config() -> dict[str, Any]:
    return dict(
        preset="small",
        mask_fraction=0.20,
        span_length_min=2,
        span_length_max=3,
        dropout=0.0,
        mask_weight_bias=False,
        cosine_loss_weight=0.0,
        max_epochs=2,
        early_stop_patience=10,
        val_split=0.34,
        seed=11,
        k_values=[10],
    )


async def test_happy_path_persists_all_artifacts(session, settings, monkeypatch):
    monkeypatch.setenv("HUMPBACK_FORCE_CPU", "1")
    cej = await _seed_crnn_ce_job(session, settings)
    job, _ = await create_masked_transformer_job(
        session, continuous_embedding_job_id=cej.id, **_tiny_config()
    )

    await run_masked_transformer_job(session, job, settings)
    await session.refresh(job)

    assert job.status == JobStatus.complete.value, job.error_message
    assert job.error_message is None
    sr = settings.storage_root
    assert masked_transformer_model_path(sr, job.id).exists()
    assert masked_transformer_loss_curve_path(sr, job.id).exists()
    assert masked_transformer_reconstruction_error_path(sr, job.id).exists()
    assert masked_transformer_contextual_embeddings_path(sr, job.id).exists()
    # Per-k bundle.
    k = 10
    k_dir = masked_transformer_k_dir(sr, job.id, k)
    assert k_dir.is_dir()
    assert (k_dir / "decoded.parquet").exists()
    assert (k_dir / "kmeans.joblib").exists()
    assert (k_dir / "run_lengths.json").exists()
    # Worker triggers generate_interpretations per k so the detail page is
    # fully populated when the job completes (spec §4.2).
    assert (k_dir / "overlay.parquet").exists()
    assert (k_dir / "exemplars.json").exists()
    assert (k_dir / "label_distribution.json").exists()

    # Decoded parquet has the canonical schema columns.
    decoded = pq.read_table(masked_transformer_k_decoded_path(sr, job.id, k))
    expected = {
        "sequence_id",
        "position",
        "label",
        "confidence",
        "audio_file_id",
        "start_timestamp",
        "end_timestamp",
        "tier",
        "chunk_index_in_region",
        "region_id",
    }
    assert expected <= set(decoded.column_names)
    # Confidences in [0, 1].
    confs = decoded.column("confidence").to_pylist()
    assert all(0.0 <= c <= 1.0 for c in confs)

    # Loss curve has the expected shape.
    loss_payload = json.loads(
        masked_transformer_loss_curve_path(sr, job.id).read_text()
    )
    assert "epochs" in loss_payload
    assert "train_loss" in loss_payload
    assert len(loss_payload["epochs"]) == len(loss_payload["train_loss"])

    # Job stats persisted.
    assert job.total_sequences == 3
    assert job.total_chunks == 72
    assert job.total_epochs is not None
    assert job.chosen_device == "cpu"


async def test_atomic_per_k_writes_clean_up_on_failure(session, settings, monkeypatch):
    monkeypatch.setenv("HUMPBACK_FORCE_CPU", "1")
    cej = await _seed_crnn_ce_job(session, settings)
    job, _ = await create_masked_transformer_job(
        session,
        continuous_embedding_job_id=cej.id,
        **{**_tiny_config(), "k_values": [10]},
    )

    # Patch ``decode_tokens`` to raise mid-write to k10.tmp/.
    import humpback.workers.masked_transformer_worker as worker_mod

    real_decode = worker_mod.decode_tokens

    def _flaky_decode(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("synthetic decode failure")

    monkeypatch.setattr(worker_mod, "decode_tokens", _flaky_decode)

    await run_masked_transformer_job(session, job, settings)
    await session.refresh(job)
    assert job.status == JobStatus.failed.value
    sr = settings.storage_root
    # Final per-k dir does NOT exist (atomic semantics).
    assert not masked_transformer_k_dir(sr, job.id, 10).exists()
    # Tmp dir is cleaned up by the failure handler.
    assert not (masked_transformer_dir(sr, job.id) / "k10.tmp").exists()

    # Restore the real impl so other tests aren't affected.
    monkeypatch.setattr(worker_mod, "decode_tokens", real_decode)


async def test_device_fallback_records_reason(session, settings, monkeypatch):
    """When validation fails, worker falls back to CPU and persists fallback_reason."""
    monkeypatch.delenv("HUMPBACK_FORCE_CPU", raising=False)
    cej = await _seed_crnn_ce_job(session, settings)
    job, _ = await create_masked_transformer_job(
        session, continuous_embedding_job_id=cej.id, **_tiny_config()
    )

    await run_masked_transformer_job(
        session, job, settings, device_validation_force_fail=True
    )
    await session.refresh(job)

    assert job.status == JobStatus.complete.value, job.error_message
    assert job.chosen_device == "cpu"
    # On Linux/macOS this records the matching backend's mismatch reason;
    # on systems with no accelerator the validate helper returns None.
    if job.fallback_reason is not None:
        assert "_output_mismatch" in job.fallback_reason


async def test_extend_k_sweep_skips_retraining(session, settings, monkeypatch):
    monkeypatch.setenv("HUMPBACK_FORCE_CPU", "1")
    cej = await _seed_crnn_ce_job(session, settings)
    config = _tiny_config()
    config["k_values"] = [10]
    job, _ = await create_masked_transformer_job(
        session, continuous_embedding_job_id=cej.id, **config
    )

    # First pass: trains transformer + writes k=10 bundle.
    await run_masked_transformer_job(session, job, settings)
    await session.refresh(job)
    assert job.status == JobStatus.complete.value, job.error_message
    sr = settings.storage_root
    transformer_path = masked_transformer_model_path(sr, job.id)
    z_path = masked_transformer_contextual_embeddings_path(sr, job.id)
    transformer_mtime = transformer_path.stat().st_mtime_ns
    z_mtime = z_path.stat().st_mtime_ns

    # Simulate extend-k-sweep: append a new k value and requeue.
    job.k_values = json.dumps([10, 12])
    job.status = JobStatus.queued.value
    await session.commit()

    await run_masked_transformer_job(session, job, settings)
    await session.refresh(job)
    assert job.status == JobStatus.complete.value, job.error_message

    # Existing k=10 dir untouched, new k=12 dir created.
    assert masked_transformer_k_dir(sr, job.id, 10).is_dir()
    assert masked_transformer_k_dir(sr, job.id, 12).is_dir()
    # Transformer + Z files unchanged.
    assert transformer_path.stat().st_mtime_ns == transformer_mtime
    assert z_path.stat().st_mtime_ns == z_mtime
    # New k=12 has the per-k artifacts.
    assert masked_transformer_k_decoded_path(sr, job.id, 12).exists()
    assert masked_transformer_k_kmeans_path(sr, job.id, 12).exists()
    assert masked_transformer_k_run_lengths_path(sr, job.id, 12).exists()
    k12_dir = masked_transformer_k_dir(sr, job.id, 12)
    assert (k12_dir / "overlay.parquet").exists()
    assert (k12_dir / "exemplars.json").exists()
    assert (k12_dir / "label_distribution.json").exists()


async def test_idempotency_via_service_signature(session, settings, monkeypatch):
    """Re-creating a job with the same training_signature returns the same id."""
    monkeypatch.setenv("HUMPBACK_FORCE_CPU", "1")
    cej = await _seed_crnn_ce_job(session, settings)
    cfg = _tiny_config()
    job1, created1 = await create_masked_transformer_job(
        session, continuous_embedding_job_id=cej.id, **cfg
    )
    job2, created2 = await create_masked_transformer_job(
        session, continuous_embedding_job_id=cej.id, **cfg
    )
    assert created1 is True
    assert created2 is False
    assert job1.id == job2.id


async def test_rejects_surfperch_upstream(session, settings, monkeypatch):
    monkeypatch.setenv("HUMPBACK_FORCE_CPU", "1")
    cej = await _seed_surfperch_ce_job(session)
    with pytest.raises(ValueError, match="CRNN region-based"):
        await create_masked_transformer_job(
            session, continuous_embedding_job_id=cej.id, **_tiny_config()
        )
