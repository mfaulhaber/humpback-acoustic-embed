"""End-to-end tests for the masked-transformer worker (ADR-061)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from humpback.call_parsing.types import Event
from humpback.models.call_parsing import (
    EventClassificationJob,
    EventSegmentationJob,
    RegionDetectionJob,
)
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
    masked_transformer_k_exemplars_path,
    masked_transformer_k_label_distribution_path,
    masked_transformer_k_overlay_path,
    masked_transformer_k_kmeans_path,
    masked_transformer_k_run_lengths_path,
    masked_transformer_loss_curve_path,
    masked_transformer_model_path,
    masked_transformer_reconstruction_error_path,
    masked_transformer_retrieval_embeddings_path,
)
from humpback.workers.masked_transformer_worker import run_masked_transformer_job
from tests.fixtures.sequence_models.classify_binding import (
    seed_classify_for_segmentation,
)


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


async def _bind_classify(
    session, settings, cej, job, *, events: list[Event] | None = None
) -> None:
    """Bind a fresh complete Classify job to ``job.event_classification_job_id``.

    All MT worker happy-path tests need this so the consolidated
    ``generate_interpretations`` (which raises if the FK is None per spec
    §6.2) runs the label-distribution path.
    """
    cls_id = await seed_classify_for_segmentation(
        session,
        settings.storage_root,
        event_segmentation_job_id=cej.event_segmentation_job_id or "",
        events=events,
    )
    job.event_classification_job_id = cls_id
    await session.commit()
    await session.refresh(job)


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

    session.add(
        EventClassificationJob(
            status=JobStatus.complete.value,
            event_segmentation_job_id=seg_job.id,
        )
    )
    await session.commit()

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
    session.add(
        EventClassificationJob(
            status=JobStatus.complete.value,
            event_segmentation_job_id=seg_job.id,
        )
    )
    await session.commit()
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
    await _bind_classify(session, settings, cej, job)

    await run_masked_transformer_job(session, job, settings)
    await session.refresh(job)

    assert job.status == JobStatus.complete.value, job.error_message
    assert job.error_message is None
    sr = settings.storage_root
    assert masked_transformer_model_path(sr, job.id).exists()
    assert masked_transformer_loss_curve_path(sr, job.id).exists()
    assert masked_transformer_reconstruction_error_path(sr, job.id).exists()
    assert masked_transformer_contextual_embeddings_path(sr, job.id).exists()
    assert not masked_transformer_retrieval_embeddings_path(sr, job.id).exists()
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


async def test_retrieval_head_job_persists_and_tokenizes_retrieval_embeddings(
    session, settings, monkeypatch
):
    monkeypatch.setenv("HUMPBACK_FORCE_CPU", "1")
    cej = await _seed_crnn_ce_job(session, settings)
    job, _ = await create_masked_transformer_job(
        session,
        continuous_embedding_job_id=cej.id,
        **{
            **_tiny_config(),
            "retrieval_head_enabled": True,
            "retrieval_dim": 6,
            "retrieval_hidden_dim": 12,
        },
    )
    await _bind_classify(session, settings, cej, job)

    await run_masked_transformer_job(session, job, settings)
    await session.refresh(job)

    assert job.status == JobStatus.complete.value, job.error_message
    sr = settings.storage_root
    contextual_path = masked_transformer_contextual_embeddings_path(sr, job.id)
    retrieval_path = masked_transformer_retrieval_embeddings_path(sr, job.id)
    assert contextual_path.exists()
    assert retrieval_path.exists()

    contextual = pq.read_table(contextual_path)
    retrieval = pq.read_table(retrieval_path)
    assert contextual.num_rows == retrieval.num_rows
    assert retrieval.column_names == contextual.column_names
    retrieval_vectors = retrieval.column("embedding").to_pylist()
    assert len(retrieval_vectors[0]) == 6
    norms = np.linalg.norm(np.asarray(retrieval_vectors, dtype=np.float32), axis=1)
    np.testing.assert_allclose(norms, np.ones_like(norms), atol=1e-5)

    kmeans_payload = joblib.load(masked_transformer_k_kmeans_path(sr, job.id, 10))
    assert kmeans_payload["kmeans"].cluster_centers_.shape[1] == 6


async def test_retrieval_head_extend_k_sweep_reuses_retrieval_artifact(
    session, settings, monkeypatch
):
    monkeypatch.setenv("HUMPBACK_FORCE_CPU", "1")
    cej = await _seed_crnn_ce_job(session, settings)
    job, _ = await create_masked_transformer_job(
        session,
        continuous_embedding_job_id=cej.id,
        **{
            **_tiny_config(),
            "k_values": [10],
            "retrieval_head_enabled": True,
            "retrieval_dim": 6,
            "retrieval_hidden_dim": 12,
        },
    )
    await _bind_classify(session, settings, cej, job)

    await run_masked_transformer_job(session, job, settings)
    await session.refresh(job)
    assert job.status == JobStatus.complete.value, job.error_message

    sr = settings.storage_root
    transformer_path = masked_transformer_model_path(sr, job.id)
    contextual_path = masked_transformer_contextual_embeddings_path(sr, job.id)
    retrieval_path = masked_transformer_retrieval_embeddings_path(sr, job.id)
    mtimes = {
        "transformer": transformer_path.stat().st_mtime_ns,
        "contextual": contextual_path.stat().st_mtime_ns,
        "retrieval": retrieval_path.stat().st_mtime_ns,
    }

    job.k_values = json.dumps([10, 12])
    job.status = JobStatus.queued.value
    await session.commit()

    await run_masked_transformer_job(session, job, settings)
    await session.refresh(job)

    assert job.status == JobStatus.complete.value, job.error_message
    assert transformer_path.stat().st_mtime_ns == mtimes["transformer"]
    assert contextual_path.stat().st_mtime_ns == mtimes["contextual"]
    assert retrieval_path.stat().st_mtime_ns == mtimes["retrieval"]
    kmeans_payload = joblib.load(masked_transformer_k_kmeans_path(sr, job.id, 12))
    assert kmeans_payload["kmeans"].cluster_centers_.shape[1] == 6


async def test_event_centered_mode_preserves_full_region_artifact_rows(
    session, settings, monkeypatch
):
    monkeypatch.setenv("HUMPBACK_FORCE_CPU", "1")
    cej = await _seed_crnn_ce_job(session, settings)
    job, _ = await create_masked_transformer_job(
        session,
        continuous_embedding_job_id=cej.id,
        **{
            **_tiny_config(),
            "sequence_construction_mode": "event_centered",
            "pre_event_context_sec": 1.0,
            "post_event_context_sec": 1.0,
        },
    )
    await _bind_classify(
        session,
        settings,
        cej,
        job,
        events=[
            Event("e1", "region-00", 1.0, 2.0, 1.5, 0.9),
            Event("e2", "region-01", 11.0, 12.0, 11.5, 0.8),
        ],
    )

    await run_masked_transformer_job(session, job, settings)
    await session.refresh(job)

    assert job.status == JobStatus.complete.value, job.error_message
    assert job.total_sequences == 3
    assert job.total_chunks == 72
    sr = settings.storage_root
    contextual = pq.read_table(
        masked_transformer_contextual_embeddings_path(sr, job.id)
    )
    reconstruction = pq.read_table(
        masked_transformer_reconstruction_error_path(sr, job.id)
    )
    decoded = pq.read_table(masked_transformer_k_decoded_path(sr, job.id, 10))
    assert contextual.num_rows == 72
    assert reconstruction.num_rows == 72
    assert decoded.num_rows == 72

    state = torch.load(
        masked_transformer_model_path(sr, job.id),
        map_location="cpu",
        weights_only=False,
    )
    assert state["config"]["sequence_construction_mode"] == "event_centered"
    assert state["config"]["event_centered_fraction"] == 1.0
    assert state["config"]["pre_event_context_sec"] == 1.0
    assert state["config"]["post_event_context_sec"] == 1.0


async def test_retrieval_head_event_centered_keeps_retrieval_rows_full_region(
    session, settings, monkeypatch
):
    monkeypatch.setenv("HUMPBACK_FORCE_CPU", "1")
    cej = await _seed_crnn_ce_job(session, settings)
    job, _ = await create_masked_transformer_job(
        session,
        continuous_embedding_job_id=cej.id,
        **{
            **_tiny_config(),
            "retrieval_head_enabled": True,
            "retrieval_dim": 6,
            "retrieval_hidden_dim": 12,
            "sequence_construction_mode": "event_centered",
        },
    )
    await _bind_classify(
        session,
        settings,
        cej,
        job,
        events=[Event("e1", "region-00", 1.0, 2.0, 1.5, 0.9)],
    )

    await run_masked_transformer_job(session, job, settings)
    await session.refresh(job)

    assert job.status == JobStatus.complete.value, job.error_message
    retrieval = pq.read_table(
        masked_transformer_retrieval_embeddings_path(settings.storage_root, job.id)
    )
    assert retrieval.num_rows == 72


async def test_mixed_mode_trains_on_region_and_event_windows(
    session, settings, monkeypatch
):
    monkeypatch.setenv("HUMPBACK_FORCE_CPU", "1")
    cej = await _seed_crnn_ce_job(session, settings)
    job, _ = await create_masked_transformer_job(
        session,
        continuous_embedding_job_id=cej.id,
        **{
            **_tiny_config(),
            "sequence_construction_mode": "mixed",
            "event_centered_fraction": 0.5,
            "pre_event_context_sec": 0.5,
            "post_event_context_sec": 0.5,
        },
    )
    await _bind_classify(
        session,
        settings,
        cej,
        job,
        events=[
            Event("e1", "region-00", 1.0, 2.0, 1.5, 0.9),
            Event("e2", "region-01", 11.0, 12.0, 11.5, 0.8),
            Event("e3", "region-02", 21.0, 22.0, 21.5, 0.7),
        ],
    )

    import humpback.workers.masked_transformer_worker as worker_mod

    real_train = worker_mod.train_masked_transformer
    observed_lengths: list[int] = []

    def _spy_train(sequences, *args, **kwargs):  # type: ignore[no-untyped-def]
        observed_lengths[:] = [int(seq.shape[0]) for seq in sequences]
        return real_train(sequences, *args, **kwargs)

    monkeypatch.setattr(worker_mod, "train_masked_transformer", _spy_train)

    await run_masked_transformer_job(session, job, settings)
    await session.refresh(job)

    assert job.status == JobStatus.complete.value, job.error_message
    assert any(length == 24 for length in observed_lengths)
    assert any(length < 24 for length in observed_lengths)


async def test_event_centered_mode_fails_when_no_events_overlap(
    session, settings, monkeypatch
):
    monkeypatch.setenv("HUMPBACK_FORCE_CPU", "1")
    cej = await _seed_crnn_ce_job(session, settings)
    job, _ = await create_masked_transformer_job(
        session,
        continuous_embedding_job_id=cej.id,
        **{**_tiny_config(), "sequence_construction_mode": "event_centered"},
    )
    await _bind_classify(
        session,
        settings,
        cej,
        job,
        events=[Event("e1", "region-00", 500.0, 501.0, 500.5, 0.9)],
    )

    await run_masked_transformer_job(session, job, settings)
    await session.refresh(job)

    assert job.status == JobStatus.failed.value
    assert "no trainable" in (job.error_message or "")


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
    await _bind_classify(session, settings, cej, job)

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
    await _bind_classify(session, settings, cej, job)

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


async def test_requeued_partial_k_bundle_regenerates_missing_interpretations(
    session, settings, monkeypatch
):
    monkeypatch.setenv("HUMPBACK_FORCE_CPU", "1")
    cej = await _seed_crnn_ce_job(session, settings)
    job, _ = await create_masked_transformer_job(
        session, continuous_embedding_job_id=cej.id, **_tiny_config()
    )
    await _bind_classify(session, settings, cej, job)

    await run_masked_transformer_job(session, job, settings)
    await session.refresh(job)
    assert job.status == JobStatus.complete.value, job.error_message

    sr = settings.storage_root
    k = 10
    transformer_path = masked_transformer_model_path(sr, job.id)
    z_path = masked_transformer_contextual_embeddings_path(sr, job.id)
    transformer_mtime = transformer_path.stat().st_mtime_ns
    z_mtime = z_path.stat().st_mtime_ns

    for artifact_path in (
        masked_transformer_k_overlay_path(sr, job.id, k),
        masked_transformer_k_exemplars_path(sr, job.id, k),
        masked_transformer_k_label_distribution_path(sr, job.id, k),
    ):
        artifact_path.unlink()

    # Simulate stale recovery/manual requeue after the k directory was
    # finalized but before interpretations were all present.
    job.status = JobStatus.queued.value
    await session.commit()

    await run_masked_transformer_job(session, job, settings)
    await session.refresh(job)
    assert job.status == JobStatus.complete.value, job.error_message

    assert transformer_path.stat().st_mtime_ns == transformer_mtime
    assert z_path.stat().st_mtime_ns == z_mtime
    assert masked_transformer_k_decoded_path(sr, job.id, k).exists()
    assert masked_transformer_k_kmeans_path(sr, job.id, k).exists()
    assert masked_transformer_k_run_lengths_path(sr, job.id, k).exists()
    assert masked_transformer_k_overlay_path(sr, job.id, k).exists()
    assert masked_transformer_k_exemplars_path(sr, job.id, k).exists()
    assert masked_transformer_k_label_distribution_path(sr, job.id, k).exists()


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
