"""SQL-backed job queue with claim semantics."""

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.models.classifier import ClassifierTrainingJob, DetectionJob
from humpback.models.clustering import ClusteringJob
from humpback.models.processing import EmbeddingSet, JobStatus, ProcessingJob

logger = logging.getLogger(__name__)

STALE_JOB_TIMEOUT = timedelta(minutes=10)


async def recover_stale_jobs(session: AsyncSession) -> int:
    """Reset jobs stuck in 'running' past the stale timeout back to 'queued'."""
    cutoff = datetime.now(timezone.utc) - STALE_JOB_TIMEOUT
    result = await session.execute(
        update(ProcessingJob)
        .where(
            ProcessingJob.status == JobStatus.running.value,
            ProcessingJob.updated_at < cutoff,
        )
        .values(
            status=JobStatus.queued.value,
            updated_at=datetime.now(timezone.utc),
        )
    )
    count = result.rowcount
    if count:
        logger.warning(f"Recovered {count} stale processing job(s)")

    result2 = await session.execute(
        update(ClusteringJob)
        .where(
            ClusteringJob.status == "running",
            ClusteringJob.updated_at < cutoff,
        )
        .values(
            status="queued",
            updated_at=datetime.now(timezone.utc),
        )
    )
    count2 = result2.rowcount
    if count2:
        logger.warning(f"Recovered {count2} stale clustering job(s)")

    result3 = await session.execute(
        update(ClassifierTrainingJob)
        .where(
            ClassifierTrainingJob.status == "running",
            ClassifierTrainingJob.updated_at < cutoff,
        )
        .values(
            status="queued",
            updated_at=datetime.now(timezone.utc),
        )
    )
    count3 = result3.rowcount
    if count3:
        logger.warning(f"Recovered {count3} stale training job(s)")

    result4 = await session.execute(
        update(DetectionJob)
        .where(
            DetectionJob.status == "running",
            DetectionJob.updated_at < cutoff,
        )
        .values(
            status="queued",
            updated_at=datetime.now(timezone.utc),
        )
    )
    count4 = result4.rowcount
    if count4:
        logger.warning(f"Recovered {count4} stale detection job(s)")

    total = count + count2 + count3 + count4
    if total:
        await session.commit()
    return total


async def claim_processing_job(session: AsyncSession) -> Optional[ProcessingJob]:
    """Claim a queued processing job atomically.

    Skips jobs whose encoding_signature already has a running job
    (prevents concurrent processing of same config).
    """
    # Find encoding_signatures that are currently running
    running_sigs = (
        select(ProcessingJob.encoding_signature)
        .where(ProcessingJob.status == JobStatus.running.value)
        .scalar_subquery()
    )

    # Find a queued job not blocked by a running job with same signature
    result = await session.execute(
        select(ProcessingJob)
        .where(
            ProcessingJob.status == JobStatus.queued.value,
            ~ProcessingJob.encoding_signature.in_(running_sigs),
        )
        .order_by(ProcessingJob.created_at)
        .limit(1)
        .with_for_update(skip_locked=True)
    )
    job = result.scalar_one_or_none()
    if job is None:
        return None

    job.status = JobStatus.running.value
    job.updated_at = datetime.now(timezone.utc)
    await session.commit()
    return job


async def complete_processing_job(
    session: AsyncSession, job_id: str, warning_message: str | None = None
) -> None:
    values: dict = {
        "status": JobStatus.complete.value,
        "updated_at": datetime.now(timezone.utc),
    }
    if warning_message is not None:
        values["warning_message"] = warning_message
    await session.execute(
        update(ProcessingJob)
        .where(ProcessingJob.id == job_id)
        .values(**values)
    )
    await session.commit()


async def fail_processing_job(
    session: AsyncSession, job_id: str, error: str
) -> None:
    await session.execute(
        update(ProcessingJob)
        .where(ProcessingJob.id == job_id)
        .values(
            status=JobStatus.failed.value,
            error_message=error,
            updated_at=datetime.now(timezone.utc),
        )
    )
    await session.commit()


async def claim_clustering_job(session: AsyncSession) -> Optional[ClusteringJob]:
    result = await session.execute(
        select(ClusteringJob)
        .where(ClusteringJob.status == "queued")
        .order_by(ClusteringJob.created_at)
        .limit(1)
        .with_for_update(skip_locked=True)
    )
    job = result.scalar_one_or_none()
    if job is None:
        return None

    job.status = "running"
    job.updated_at = datetime.now(timezone.utc)
    await session.commit()
    return job


async def complete_clustering_job(session: AsyncSession, job_id: str) -> None:
    await session.execute(
        update(ClusteringJob)
        .where(ClusteringJob.id == job_id)
        .values(status="complete", updated_at=datetime.now(timezone.utc))
    )
    await session.commit()


async def fail_clustering_job(
    session: AsyncSession, job_id: str, error: str
) -> None:
    await session.execute(
        update(ClusteringJob)
        .where(ClusteringJob.id == job_id)
        .values(
            status="failed",
            error_message=error,
            updated_at=datetime.now(timezone.utc),
        )
    )
    await session.commit()


# ---- Classifier Training Jobs ----


async def claim_training_job(session: AsyncSession) -> Optional[ClassifierTrainingJob]:
    result = await session.execute(
        select(ClassifierTrainingJob)
        .where(ClassifierTrainingJob.status == "queued")
        .order_by(ClassifierTrainingJob.created_at)
        .limit(1)
        .with_for_update(skip_locked=True)
    )
    job = result.scalar_one_or_none()
    if job is None:
        return None

    job.status = "running"
    job.updated_at = datetime.now(timezone.utc)
    await session.commit()
    return job


async def complete_training_job(session: AsyncSession, job_id: str) -> None:
    await session.execute(
        update(ClassifierTrainingJob)
        .where(ClassifierTrainingJob.id == job_id)
        .values(status="complete", updated_at=datetime.now(timezone.utc))
    )
    await session.commit()


async def fail_training_job(
    session: AsyncSession, job_id: str, error: str
) -> None:
    await session.execute(
        update(ClassifierTrainingJob)
        .where(ClassifierTrainingJob.id == job_id)
        .values(
            status="failed",
            error_message=error,
            updated_at=datetime.now(timezone.utc),
        )
    )
    await session.commit()


# ---- Detection Jobs ----


async def claim_detection_job(session: AsyncSession) -> Optional[DetectionJob]:
    """Claim a queued local detection job (not hydrophone)."""
    result = await session.execute(
        select(DetectionJob)
        .where(
            DetectionJob.status == "queued",
            DetectionJob.hydrophone_id.is_(None),
        )
        .order_by(DetectionJob.created_at)
        .limit(1)
        .with_for_update(skip_locked=True)
    )
    job = result.scalar_one_or_none()
    if job is None:
        return None

    job.status = "running"
    job.updated_at = datetime.now(timezone.utc)
    await session.commit()
    return job


async def claim_hydrophone_detection_job(session: AsyncSession) -> Optional[DetectionJob]:
    """Claim a queued hydrophone detection job."""
    result = await session.execute(
        select(DetectionJob)
        .where(
            DetectionJob.status == "queued",
            DetectionJob.hydrophone_id.isnot(None),
        )
        .order_by(DetectionJob.created_at)
        .limit(1)
        .with_for_update(skip_locked=True)
    )
    job = result.scalar_one_or_none()
    if job is None:
        return None

    job.status = "running"
    job.updated_at = datetime.now(timezone.utc)
    await session.commit()
    return job


async def complete_detection_job(session: AsyncSession, job_id: str) -> None:
    await session.execute(
        update(DetectionJob)
        .where(DetectionJob.id == job_id)
        .values(status="complete", updated_at=datetime.now(timezone.utc))
    )
    await session.commit()


async def fail_detection_job(
    session: AsyncSession, job_id: str, error: str
) -> None:
    await session.execute(
        update(DetectionJob)
        .where(DetectionJob.id == job_id)
        .values(
            status="failed",
            error_message=error,
            updated_at=datetime.now(timezone.utc),
        )
    )
    await session.commit()


# ---- Extraction Jobs ----


async def claim_extraction_job(session: AsyncSession) -> Optional[DetectionJob]:
    result = await session.execute(
        select(DetectionJob)
        .where(DetectionJob.extract_status == "queued")
        .order_by(DetectionJob.updated_at)
        .limit(1)
        .with_for_update(skip_locked=True)
    )
    job = result.scalar_one_or_none()
    if job is None:
        return None

    job.extract_status = "running"
    job.updated_at = datetime.now(timezone.utc)
    await session.commit()
    return job


async def complete_extraction_job(session: AsyncSession, job_id: str) -> None:
    await session.execute(
        update(DetectionJob)
        .where(DetectionJob.id == job_id)
        .values(extract_status="complete", updated_at=datetime.now(timezone.utc))
    )
    await session.commit()


async def fail_extraction_job(
    session: AsyncSession, job_id: str, error: str
) -> None:
    await session.execute(
        update(DetectionJob)
        .where(DetectionJob.id == job_id)
        .values(
            extract_status="failed",
            extract_error=error,
            updated_at=datetime.now(timezone.utc),
        )
    )
    await session.commit()
