"""Tests for the continuous-embedding service idempotency and validation."""

import pytest

from humpback.models.call_parsing import RegionDetectionJob
from humpback.models.processing import JobStatus
from humpback.models.sequence_models import ContinuousEmbeddingJob
from humpback.schemas.sequence_models import ContinuousEmbeddingJobCreate
from humpback.services.continuous_embedding_service import (
    CancelTerminalJobError,
    cancel_continuous_embedding_job,
    create_continuous_embedding_job,
    get_continuous_embedding_job,
    list_continuous_embedding_jobs,
)


async def _seed_region_job(session) -> RegionDetectionJob:
    job = RegionDetectionJob(status=JobStatus.complete.value)
    session.add(job)
    await session.commit()
    await session.refresh(job)
    return job


async def test_create_returns_new_job(session):
    region_job = await _seed_region_job(session)
    payload = ContinuousEmbeddingJobCreate(region_detection_job_id=region_job.id)
    job, created = await create_continuous_embedding_job(session, payload)
    assert created is True
    assert job.status == JobStatus.queued.value
    assert job.region_detection_job_id == region_job.id
    assert job.window_size_seconds == 5.0
    assert job.target_sample_rate == 32000
    assert job.encoding_signature
    assert job.hop_seconds == 1.0
    assert job.pad_seconds == 10.0


async def test_idempotent_returns_existing_complete(session):
    region_job = await _seed_region_job(session)
    payload = ContinuousEmbeddingJobCreate(region_detection_job_id=region_job.id)
    first, _ = await create_continuous_embedding_job(session, payload)
    first.status = JobStatus.complete.value
    await session.commit()

    second, created = await create_continuous_embedding_job(session, payload)
    assert created is False
    assert second.id == first.id


async def test_in_flight_returns_existing_queued_or_running(session):
    region_job = await _seed_region_job(session)
    payload = ContinuousEmbeddingJobCreate(region_detection_job_id=region_job.id)
    first, _ = await create_continuous_embedding_job(session, payload)
    first.status = JobStatus.running.value
    await session.commit()

    second, created = await create_continuous_embedding_job(session, payload)
    assert created is False
    assert second.id == first.id

    rows = await list_continuous_embedding_jobs(session)
    assert len(rows) == 1


async def test_signature_differs_for_different_params(session):
    region_job = await _seed_region_job(session)
    payload_a = ContinuousEmbeddingJobCreate(region_detection_job_id=region_job.id)
    job_a, _ = await create_continuous_embedding_job(session, payload_a)

    payload_b = ContinuousEmbeddingJobCreate(
        region_detection_job_id=region_job.id, hop_seconds=2.0
    )
    job_b, created_b = await create_continuous_embedding_job(session, payload_b)
    assert created_b is True
    assert job_a.encoding_signature != job_b.encoding_signature


async def test_rejects_unknown_region_detection_job(session):
    payload = ContinuousEmbeddingJobCreate(region_detection_job_id="nonexistent")
    with pytest.raises(ValueError, match="region_detection_job not found"):
        await create_continuous_embedding_job(session, payload)


async def test_rejects_unsupported_model_version(session):
    region_job = await _seed_region_job(session)
    payload = ContinuousEmbeddingJobCreate(
        region_detection_job_id=region_job.id,
        model_version="not-a-real-model",
    )
    with pytest.raises(ValueError, match="Unsupported model_version"):
        await create_continuous_embedding_job(session, payload)


async def test_service_defensively_rejects_invalid_hop(session):
    region_job = await _seed_region_job(session)

    payload = ContinuousEmbeddingJobCreate(region_detection_job_id=region_job.id)
    payload.hop_seconds = 0.0
    with pytest.raises(ValueError, match="hop_seconds"):
        await create_continuous_embedding_job(session, payload)

    payload2 = ContinuousEmbeddingJobCreate(region_detection_job_id=region_job.id)
    payload2.pad_seconds = -1.0
    with pytest.raises(ValueError, match="pad_seconds"):
        await create_continuous_embedding_job(session, payload2)


async def test_cancel_queued_flips_to_canceled(session):
    region_job = await _seed_region_job(session)
    payload = ContinuousEmbeddingJobCreate(region_detection_job_id=region_job.id)
    job, _ = await create_continuous_embedding_job(session, payload)
    canceled = await cancel_continuous_embedding_job(session, job.id)
    assert canceled is not None
    assert canceled.status == JobStatus.canceled.value


async def test_cancel_running_flips_to_canceled(session):
    region_job = await _seed_region_job(session)
    payload = ContinuousEmbeddingJobCreate(region_detection_job_id=region_job.id)
    job, _ = await create_continuous_embedding_job(session, payload)
    job.status = JobStatus.running.value
    await session.commit()

    canceled = await cancel_continuous_embedding_job(session, job.id)
    assert canceled is not None
    assert canceled.status == JobStatus.canceled.value


async def test_cancel_complete_raises(session):
    region_job = await _seed_region_job(session)
    payload = ContinuousEmbeddingJobCreate(region_detection_job_id=region_job.id)
    job, _ = await create_continuous_embedding_job(session, payload)
    job.status = JobStatus.complete.value
    await session.commit()

    with pytest.raises(CancelTerminalJobError):
        await cancel_continuous_embedding_job(session, job.id)


async def test_cancel_missing_returns_none(session):
    result = await cancel_continuous_embedding_job(session, "nonexistent")
    assert result is None


async def test_get_returns_existing_job(session):
    region_job = await _seed_region_job(session)
    payload = ContinuousEmbeddingJobCreate(region_detection_job_id=region_job.id)
    job, _ = await create_continuous_embedding_job(session, payload)

    fetched = await get_continuous_embedding_job(session, job.id)
    assert isinstance(fetched, ContinuousEmbeddingJob)
    assert fetched.id == job.id

    missing = await get_continuous_embedding_job(session, "nonexistent")
    assert missing is None
