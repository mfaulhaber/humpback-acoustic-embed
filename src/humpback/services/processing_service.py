import json
from typing import Any, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.models.processing import EmbeddingSet, JobStatus, ProcessingJob
from humpback.processing.signature import compute_encoding_signature
from humpback.services.model_registry_service import get_default_model


async def create_processing_job(
    session: AsyncSession,
    audio_file_id: str,
    model_version: Optional[str],
    window_size_seconds: float,
    target_sample_rate: int,
    feature_config: Optional[dict[str, Any]] = None,
) -> tuple[ProcessingJob, bool]:
    """Create a processing job. Returns (job, skipped).
    If an EmbeddingSet already exists for this signature, marks job as complete immediately.
    """
    # Resolve model_version from registry if not provided
    if model_version is None:
        default = await get_default_model(session)
        if default is not None:
            model_version = default.name
        else:
            model_version = "perch_v1"  # fallback

    signature = compute_encoding_signature(
        model_version, window_size_seconds, target_sample_rate, feature_config
    )

    # Check for existing completed embedding set
    existing = await session.execute(
        select(EmbeddingSet).where(
            EmbeddingSet.audio_file_id == audio_file_id,
            EmbeddingSet.encoding_signature == signature,
        )
    )
    if existing.scalar_one_or_none():
        # Create a job record but mark it as already complete (skipped)
        job = ProcessingJob(
            audio_file_id=audio_file_id,
            encoding_signature=signature,
            model_version=model_version,
            window_size_seconds=window_size_seconds,
            target_sample_rate=target_sample_rate,
            feature_config=json.dumps(feature_config) if feature_config else None,
            status=JobStatus.complete.value,
        )
        session.add(job)
        await session.commit()
        return job, True

    job = ProcessingJob(
        audio_file_id=audio_file_id,
        encoding_signature=signature,
        model_version=model_version,
        window_size_seconds=window_size_seconds,
        target_sample_rate=target_sample_rate,
        feature_config=json.dumps(feature_config) if feature_config else None,
    )
    session.add(job)
    await session.commit()
    return job, False


async def get_processing_job(
    session: AsyncSession, job_id: str
) -> Optional[ProcessingJob]:
    result = await session.execute(
        select(ProcessingJob).where(ProcessingJob.id == job_id)
    )
    return result.scalar_one_or_none()


async def list_processing_jobs(session: AsyncSession) -> list[ProcessingJob]:
    result = await session.execute(
        select(ProcessingJob).order_by(ProcessingJob.created_at.desc())
    )
    return list(result.scalars().all())


async def cancel_processing_job(
    session: AsyncSession, job_id: str
) -> Optional[ProcessingJob]:
    job = await get_processing_job(session, job_id)
    if job is None:
        return None
    if job.status in (JobStatus.queued.value, JobStatus.running.value):
        job.status = JobStatus.canceled.value
        await session.commit()
    return job


async def list_embedding_sets(session: AsyncSession) -> list[EmbeddingSet]:
    result = await session.execute(
        select(EmbeddingSet).order_by(EmbeddingSet.created_at.desc())
    )
    return list(result.scalars().all())


async def get_embedding_set(
    session: AsyncSession, es_id: str
) -> Optional[EmbeddingSet]:
    result = await session.execute(
        select(EmbeddingSet).where(EmbeddingSet.id == es_id)
    )
    return result.scalar_one_or_none()
