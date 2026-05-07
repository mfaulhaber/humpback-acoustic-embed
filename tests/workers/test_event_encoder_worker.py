"""Tests for the Event Encoder worker."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from humpback.call_parsing.storage import segmentation_job_dir, write_events
from humpback.call_parsing.types import Event
from humpback.models.call_parsing import EventSegmentationJob, RegionDetectionJob
from humpback.models.processing import JobStatus
from humpback.models.sequence_models import ContinuousEmbeddingJob
from humpback.schemas.sequence_models import (
    EventEncoderJobCreate,
    EventEncoderPreprocessingConfig,
)
from humpback.services.event_encoder_service import create_event_encoder_job
from humpback.storage import continuous_embedding_dir, event_encoder_dir
from humpback.workers.event_encoder_worker import run_event_encoder_job


async def _seed_source(session, settings, *, chunks_region_id: str = "region-1"):
    region = RegionDetectionJob(
        status=JobStatus.complete.value,
        hydrophone_id="rpi_orcasound_lab",
        start_timestamp=100.0,
        end_timestamp=200.0,
    )
    session.add(region)
    await session.flush()
    seg = EventSegmentationJob(
        status=JobStatus.complete.value,
        region_detection_job_id=region.id,
    )
    session.add(seg)
    await session.flush()

    events = [
        Event("event-1", "region-1", 0.0, 1.0, 0.5, 0.9),
        Event("event-2", "region-1", 2.0, 3.0, 2.5, 0.8),
        Event("event-3", "region-1", 4.0, 5.0, 4.5, 0.7),
    ]
    seg_dir = segmentation_job_dir(settings.storage_root, seg.id)
    seg_dir.mkdir(parents=True, exist_ok=True)
    write_events(seg_dir / "events.parquet", events)

    continuous = ContinuousEmbeddingJob(
        status=JobStatus.complete.value,
        event_segmentation_job_id=seg.id,
        event_source_mode="raw",
        region_detection_job_id=region.id,
        model_version="crnn-call-parsing-pytorch",
        target_sample_rate=16000,
        encoding_signature="cej-sig",
        crnn_checkpoint_sha256="abc123",
        chunk_size_seconds=0.5,
        chunk_hop_seconds=0.5,
        projection_kind="identity",
        projection_dim=2,
        total_regions=1,
        total_chunks=6,
    )
    session.add(continuous)
    await session.flush()

    ce_dir = continuous_embedding_dir(settings.storage_root, continuous.id)
    ce_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = ce_dir / "embeddings.parquet"
    rows = []
    for i, start in enumerate([100.0, 100.5, 102.0, 102.5, 104.0, 104.5]):
        rows.append(
            {
                "region_id": chunks_region_id,
                "audio_file_id": None,
                "hydrophone_id": "rpi_orcasound_lab",
                "chunk_index_in_region": i,
                "start_timestamp": start,
                "end_timestamp": start + 0.5,
                "is_in_pad": False,
                "call_probability": 0.2 + 0.1 * i,
                "event_overlap_fraction": 1.0,
                "nearest_event_id": f"event-{1 + i // 2}",
                "distance_to_nearest_event_seconds": 0.0,
                "tier": "event_core",
                "embedding": [float(i), float(i + 1)],
            }
        )
    pq.write_table(pa.Table.from_pylist(rows), parquet_path)
    continuous.parquet_path = str(parquet_path)
    await session.commit()
    await session.refresh(seg)
    await session.refresh(continuous)
    return seg, continuous


def _audio_provider(event: Event, sample_rate: int) -> np.ndarray:
    duration = float(event.end_sec - event.start_sec)
    n = max(1, int(round(duration * sample_rate)))
    t = np.arange(n, dtype=np.float32) / sample_rate
    return np.sin(2 * np.pi * 440.0 * t).astype(np.float32)


async def test_event_encoder_worker_writes_artifacts(session, settings):
    seg, continuous = await _seed_source(session, settings)
    job, _ = await create_event_encoder_job(
        session,
        EventEncoderJobCreate(
            event_segmentation_job_id=seg.id,
            continuous_embedding_job_id=continuous.id,
            k_values=[1, 2, 99],
            preprocessing=EventEncoderPreprocessingConfig(pca_dim=64),
        ),
    )

    await run_event_encoder_job(session, job, settings, audio_provider=_audio_provider)
    await session.refresh(job)

    assert job.status == JobStatus.complete.value
    assert job.total_events == 3
    assert job.encoded_events == 3
    assert job.skipped_events == 0
    assert job.event_vector_dim == 10
    assert job.event_vectors_path is not None
    assert job.event_tokens_path is not None
    assert job.token_sequences_path is not None
    assert job.manifest_path is not None
    assert job.report_path is not None

    vectors = pq.read_table(job.event_vectors_path).to_pylist()
    tokens = pq.read_table(job.event_tokens_path).to_pylist()
    sequences = pq.read_table(job.token_sequences_path).to_pylist()
    manifest = json.loads(Path(job.manifest_path).read_text())
    report = json.loads(Path(job.report_path).read_text())

    assert len(vectors) == 3
    assert len(tokens) == 6
    assert len(sequences) == 6
    assert manifest["valid_k_values"] == [1, 2]
    assert manifest["invalid_k_values"] == [99]
    assert report["summary"]["encoded_events"] == 3
    assert report["token_examples"]["2"]["T00"][0]["event_id"]
    assert (
        event_encoder_dir(settings.storage_root, job.id) / "preprocess.joblib"
    ).exists()
    assert (
        event_encoder_dir(settings.storage_root, job.id) / "kmeans_k2.joblib"
    ).exists()


async def test_event_encoder_worker_fails_when_no_events_are_encodable(
    session, settings
):
    seg, continuous = await _seed_source(
        session, settings, chunks_region_id="different-region"
    )
    job, _ = await create_event_encoder_job(
        session,
        EventEncoderJobCreate(
            event_segmentation_job_id=seg.id,
            continuous_embedding_job_id=continuous.id,
            k_values=[1],
        ),
    )

    await run_event_encoder_job(session, job, settings, audio_provider=_audio_provider)
    await session.refresh(job)

    assert job.status == JobStatus.failed.value
    assert "could not encode any events" in (job.error_message or "")
    assert job.event_vectors_path is None


async def test_event_encoder_worker_honors_pre_canceled_job(session, settings):
    seg, continuous = await _seed_source(session, settings)
    job, _ = await create_event_encoder_job(
        session,
        EventEncoderJobCreate(
            event_segmentation_job_id=seg.id,
            continuous_embedding_job_id=continuous.id,
            k_values=[1],
        ),
    )
    job.status = JobStatus.canceled.value
    await session.commit()

    await run_event_encoder_job(session, job, settings, audio_provider=_audio_provider)
    await session.refresh(job)

    assert job.status == JobStatus.canceled.value
    assert not (
        event_encoder_dir(settings.storage_root, job.id) / "report.json"
    ).exists()
