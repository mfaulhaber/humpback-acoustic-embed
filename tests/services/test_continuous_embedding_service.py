"""Tests for the continuous-embedding service idempotency and validation."""

import json
from datetime import datetime, timezone

import pytest
from sqlalchemy.exc import IntegrityError

from humpback.call_parsing.segmentation.model import SegmentationCRNN
from humpback.database import create_session_factory
from humpback.ml.checkpointing import save_checkpoint
from humpback.models.call_parsing import (
    EventSegmentationJob,
    RegionDetectionJob,
    SegmentationModel,
)
from humpback.models.processing import JobStatus
from humpback.models.sequence_models import ContinuousEmbeddingJob
from humpback.schemas.sequence_models import ContinuousEmbeddingJobCreate
from humpback.services.continuous_embedding_service import (
    CancelTerminalJobError,
    cancel_continuous_embedding_job,
    create_continuous_embedding_job,
    delete_continuous_embedding_job,
    get_continuous_embedding_job,
    list_continuous_embedding_jobs,
)
from humpback.storage import continuous_embedding_dir


async def _seed_region_job(
    session,
    *,
    status: str = JobStatus.complete.value,
    hydrophone: bool = True,
) -> RegionDetectionJob:
    job = RegionDetectionJob(
        status=status,
        hydrophone_id="rpi_orcasound_lab" if hydrophone else None,
        start_timestamp=0.0 if hydrophone else None,
        end_timestamp=300.0 if hydrophone else None,
    )
    session.add(job)
    await session.commit()
    await session.refresh(job)
    return job


async def _seed_seg_job(
    session,
    *,
    status: str = JobStatus.complete.value,
) -> EventSegmentationJob:
    region_job = await _seed_region_job(session)
    seg_job = EventSegmentationJob(
        status=status,
        region_detection_job_id=region_job.id,
    )
    session.add(seg_job)
    await session.commit()
    await session.refresh(seg_job)
    return seg_job


def _save_crnn_checkpoint(path, *, hop_length: int = 512) -> None:
    model = SegmentationCRNN()
    save_checkpoint(
        path,
        model,
        optimizer=None,
        config={
            "model_type": "SegmentationCRNN",
            "n_mels": 64,
            "conv_channels": [32, 64, 96, 128],
            "gru_hidden": 64,
            "gru_layers": 2,
            "feature_config": {
                "sample_rate": 16000,
                "n_fft": 2048,
                "hop_length": hop_length,
                "n_mels": 64,
                "fmin": 20.0,
                "fmax": 4000.0,
                "normalize": "per_region_zscore",
            },
        },
    )


async def _seed_segmentation_model(session, checkpoint_path) -> SegmentationModel:
    model = SegmentationModel(
        name="test-crnn",
        model_family="pytorch_crnn",
        model_path=str(checkpoint_path),
        config_json=None,
    )
    session.add(model)
    await session.commit()
    await session.refresh(model)
    return model


async def test_create_returns_new_job(session):
    seg_job = await _seed_seg_job(session)
    payload = ContinuousEmbeddingJobCreate(event_segmentation_job_id=seg_job.id)
    job, created = await create_continuous_embedding_job(session, payload)
    assert created is True
    assert job.status == JobStatus.queued.value
    assert job.event_segmentation_job_id == seg_job.id
    assert job.window_size_seconds == 5.0
    assert job.target_sample_rate == 32000
    assert job.encoding_signature
    assert job.hop_seconds == 1.0
    assert job.pad_seconds == 2.0


async def test_idempotent_returns_existing_complete(session):
    seg_job = await _seed_seg_job(session)
    payload = ContinuousEmbeddingJobCreate(event_segmentation_job_id=seg_job.id)
    first, _ = await create_continuous_embedding_job(session, payload)
    first.status = JobStatus.complete.value
    await session.commit()

    second, created = await create_continuous_embedding_job(session, payload)
    assert created is False
    assert second.id == first.id


async def test_in_flight_returns_existing_queued_or_running(session):
    seg_job = await _seed_seg_job(session)
    payload = ContinuousEmbeddingJobCreate(event_segmentation_job_id=seg_job.id)
    first, _ = await create_continuous_embedding_job(session, payload)
    first.status = JobStatus.running.value
    await session.commit()

    second, created = await create_continuous_embedding_job(session, payload)
    assert created is False
    assert second.id == first.id

    rows = await list_continuous_embedding_jobs(session)
    assert len(rows) == 1


async def test_signature_differs_for_different_params(session):
    seg_job = await _seed_seg_job(session)
    payload_a = ContinuousEmbeddingJobCreate(event_segmentation_job_id=seg_job.id)
    job_a, _ = await create_continuous_embedding_job(session, payload_a)

    payload_b = ContinuousEmbeddingJobCreate(
        event_segmentation_job_id=seg_job.id, hop_seconds=2.0
    )
    job_b, created_b = await create_continuous_embedding_job(session, payload_b)
    assert created_b is True
    assert job_a.encoding_signature != job_b.encoding_signature


async def test_rejects_unknown_segmentation_job(session):
    payload = ContinuousEmbeddingJobCreate(event_segmentation_job_id="nonexistent")
    with pytest.raises(ValueError, match="event_segmentation_job not found"):
        await create_continuous_embedding_job(session, payload)


async def test_rejects_non_complete_segmentation_job(session):
    seg_job = await _seed_seg_job(session, status=JobStatus.running.value)
    payload = ContinuousEmbeddingJobCreate(event_segmentation_job_id=seg_job.id)
    with pytest.raises(ValueError, match="completed event_segmentation_job"):
        await create_continuous_embedding_job(session, payload)


async def test_rejects_unsupported_model_version(session):
    seg_job = await _seed_seg_job(session)
    payload = ContinuousEmbeddingJobCreate(
        event_segmentation_job_id=seg_job.id,
        model_version="not-a-real-model",
    )
    with pytest.raises(ValueError, match="Unsupported model_version"):
        await create_continuous_embedding_job(session, payload)


async def test_service_defensively_rejects_invalid_hop(session):
    seg_job = await _seed_seg_job(session)

    payload = ContinuousEmbeddingJobCreate(event_segmentation_job_id=seg_job.id)
    payload.hop_seconds = 0.0
    with pytest.raises(ValueError, match="hop_seconds"):
        await create_continuous_embedding_job(session, payload)

    payload2 = ContinuousEmbeddingJobCreate(event_segmentation_job_id=seg_job.id)
    payload2.pad_seconds = -1.0
    with pytest.raises(ValueError, match="pad_seconds"):
        await create_continuous_embedding_job(session, payload2)


async def test_cancel_queued_flips_to_canceled(session):
    seg_job = await _seed_seg_job(session)
    payload = ContinuousEmbeddingJobCreate(event_segmentation_job_id=seg_job.id)
    job, _ = await create_continuous_embedding_job(session, payload)
    canceled = await cancel_continuous_embedding_job(session, job.id)
    assert canceled is not None
    assert canceled.status == JobStatus.canceled.value


async def test_cancel_running_flips_to_canceled(session):
    seg_job = await _seed_seg_job(session)
    payload = ContinuousEmbeddingJobCreate(event_segmentation_job_id=seg_job.id)
    job, _ = await create_continuous_embedding_job(session, payload)
    job.status = JobStatus.running.value
    await session.commit()

    canceled = await cancel_continuous_embedding_job(session, job.id)
    assert canceled is not None
    assert canceled.status == JobStatus.canceled.value


async def test_cancel_complete_raises(session):
    seg_job = await _seed_seg_job(session)
    payload = ContinuousEmbeddingJobCreate(event_segmentation_job_id=seg_job.id)
    job, _ = await create_continuous_embedding_job(session, payload)
    job.status = JobStatus.complete.value
    await session.commit()

    with pytest.raises(CancelTerminalJobError):
        await cancel_continuous_embedding_job(session, job.id)


async def test_cancel_missing_returns_none(session):
    result = await cancel_continuous_embedding_job(session, "nonexistent")
    assert result is None


async def test_resubmitting_failed_job_requeues_existing_row(session):
    seg_job = await _seed_seg_job(session)
    payload = ContinuousEmbeddingJobCreate(event_segmentation_job_id=seg_job.id)
    job, _ = await create_continuous_embedding_job(session, payload)
    job.status = JobStatus.failed.value
    job.error_message = "boom"
    job.parquet_path = "/tmp/old.parquet"
    job.vector_dim = 123
    job.updated_at = datetime.now(timezone.utc)
    await session.commit()

    retried, created = await create_continuous_embedding_job(session, payload)
    assert created is False
    assert retried.id == job.id
    assert retried.status == JobStatus.queued.value
    assert retried.error_message is None
    assert retried.parquet_path is None
    assert retried.vector_dim is None


async def test_integrity_error_race_returns_canonical_existing_job(
    session, engine, monkeypatch
):
    seg_job = await _seed_seg_job(session)
    payload = ContinuousEmbeddingJobCreate(event_segmentation_job_id=seg_job.id)
    factory = create_session_factory(engine)
    original_commit = session.commit
    did_inject_race = False

    async def commit_with_race():
        nonlocal did_inject_race
        if did_inject_race:
            return await original_commit()

        did_inject_race = True
        async with factory() as competing_session:
            competing_job, competing_created = await create_continuous_embedding_job(
                competing_session, payload
            )
            assert competing_created is True
            assert competing_job.encoding_signature
        raise IntegrityError("insert", {"encoding_signature": "dup"}, Exception("dup"))

    monkeypatch.setattr(session, "commit", commit_with_race)

    winner, created = await create_continuous_embedding_job(session, payload)
    assert created is False
    assert winner.status == JobStatus.queued.value

    rows = await list_continuous_embedding_jobs(session)
    assert len(rows) == 1


async def test_get_returns_existing_job(session):
    seg_job = await _seed_seg_job(session)
    payload = ContinuousEmbeddingJobCreate(event_segmentation_job_id=seg_job.id)
    job, _ = await create_continuous_embedding_job(session, payload)

    fetched = await get_continuous_embedding_job(session, job.id)
    assert isinstance(fetched, ContinuousEmbeddingJob)
    assert fetched.id == job.id

    missing = await get_continuous_embedding_job(session, "nonexistent")
    assert missing is None


async def test_delete_removes_db_row_and_disk_artifacts(session, settings):
    seg_job = await _seed_seg_job(session)
    payload = ContinuousEmbeddingJobCreate(event_segmentation_job_id=seg_job.id)
    job, _ = await create_continuous_embedding_job(session, payload)

    artifact_dir = continuous_embedding_dir(settings.storage_root, job.id)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "embeddings.parquet").write_text("fake")

    result = await delete_continuous_embedding_job(session, job.id, settings)
    assert result is True
    assert not artifact_dir.exists()
    assert await get_continuous_embedding_job(session, job.id) is None


async def test_delete_nonexistent_returns_false(session, settings):
    result = await delete_continuous_embedding_job(session, "nonexistent", settings)
    assert result is False


async def test_delete_succeeds_when_no_disk_artifacts(session, settings):
    seg_job = await _seed_seg_job(session)
    payload = ContinuousEmbeddingJobCreate(event_segmentation_job_id=seg_job.id)
    job, _ = await create_continuous_embedding_job(session, payload)

    result = await delete_continuous_embedding_job(session, job.id, settings)
    assert result is True
    assert await get_continuous_embedding_job(session, job.id) is None


async def test_create_region_crnn_snapshots_checkpoint_feature_config(
    session, settings
):
    seg_job = await _seed_seg_job(session)
    checkpoint_path = settings.storage_root / "test-crnn-hop256.pt"
    _save_crnn_checkpoint(checkpoint_path, hop_length=256)
    seg_model = await _seed_segmentation_model(session, checkpoint_path)

    payload = ContinuousEmbeddingJobCreate(
        model_version="crnn-call-parsing-pytorch",
        event_segmentation_job_id=seg_job.id,
        region_detection_job_id=seg_job.region_detection_job_id,
        crnn_segmentation_model_id=seg_model.id,
        chunk_size_seconds=0.25,
        chunk_hop_seconds=0.25,
        projection_kind="identity",
        projection_dim=1024,
    )

    job, created = await create_continuous_embedding_job(session, payload)

    assert created is True
    assert job.target_sample_rate == 16000
    feature_config = json.loads(job.feature_config_json or "{}")
    assert feature_config["hop_length"] == 256
    assert job.projection_dim == 2048
    assert job.encoding_signature
