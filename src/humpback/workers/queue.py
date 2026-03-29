"""SQL-backed job queue with claim semantics."""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.models.classifier import ClassifierTrainingJob, DetectionJob
from humpback.models.clustering import ClusteringJob
from humpback.models.processing import JobStatus, ProcessingJob
from humpback.models.retrain import RetrainWorkflow
from humpback.models.search import SearchJob

logger = logging.getLogger(__name__)

STALE_JOB_TIMEOUT = timedelta(minutes=10)


def _rowcount(result: Any) -> int:
    return int(getattr(result, "rowcount", 0) or 0)


async def _claim_next_job(
    session: AsyncSession,
    model,
    *,
    status_attr,
    queued_value: str,
    running_value: str,
    order_attr,
    extra_filters: tuple = (),
):
    """Atomically claim the next queued job for a model.

    Uses compare-and-set semantics (`WHERE id=:candidate AND status=:queued`) so
    concurrent workers cannot claim the same row.
    """
    candidate_result = await session.execute(
        select(model.id)
        .where(status_attr == queued_value, *extra_filters)
        .order_by(order_attr)
        .limit(1)
    )
    candidate_id = candidate_result.scalar_one_or_none()
    if candidate_id is None:
        return None

    claim_result = await session.execute(
        update(model)
        .where(model.id == candidate_id, status_attr == queued_value)
        .values(
            **{
                status_attr.key: running_value,
                "updated_at": datetime.now(timezone.utc),
            }
        )
    )
    if _rowcount(claim_result) != 1:
        # Another worker claimed it first.
        await session.rollback()
        return None

    await session.commit()
    claimed_result = await session.execute(
        select(model).where(model.id == candidate_id)
    )
    return claimed_result.scalar_one_or_none()


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
    count = _rowcount(result)
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
    count2 = _rowcount(result2)
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
    count3 = _rowcount(result3)
    if count3:
        logger.warning(f"Recovered {count3} stale training job(s)")

    result4 = await session.execute(
        update(DetectionJob)
        .where(
            DetectionJob.status.in_(["running", "paused"]),
            DetectionJob.updated_at < cutoff,
        )
        .values(
            status="queued",
            updated_at=datetime.now(timezone.utc),
        )
    )
    count4 = _rowcount(result4)
    if count4:
        logger.warning(f"Recovered {count4} stale detection job(s)")

    result5 = await session.execute(
        update(RetrainWorkflow)
        .where(
            RetrainWorkflow.status.in_(["importing", "processing", "training"]),
            RetrainWorkflow.updated_at < cutoff,
        )
        .values(
            status="queued",
            updated_at=datetime.now(timezone.utc),
        )
    )
    count5 = _rowcount(result5)
    if count5:
        logger.warning(f"Recovered {count5} stale retrain workflow(s)")

    result6 = await session.execute(
        update(SearchJob)
        .where(
            SearchJob.status == "running",
            SearchJob.updated_at < cutoff,
        )
        .values(
            status="queued",
            updated_at=datetime.now(timezone.utc),
        )
    )
    count6 = _rowcount(result6)
    if count6:
        logger.warning(f"Recovered {count6} stale search job(s)")

    from humpback.models.label_processing import LabelProcessingJob

    result7 = await session.execute(
        update(LabelProcessingJob)
        .where(
            LabelProcessingJob.status == "running",
            LabelProcessingJob.updated_at < cutoff,
        )
        .values(
            status="queued",
            updated_at=datetime.now(timezone.utc),
        )
    )
    count7 = _rowcount(result7)
    if count7:
        logger.warning(f"Recovered {count7} stale label processing job(s)")

    from humpback.models.vocalization import (
        VocalizationInferenceJob,
        VocalizationTrainingJob,
    )

    result8 = await session.execute(
        update(VocalizationTrainingJob)
        .where(
            VocalizationTrainingJob.status == "running",
            VocalizationTrainingJob.updated_at < cutoff,
        )
        .values(
            status="queued",
            updated_at=datetime.now(timezone.utc),
        )
    )
    count8 = _rowcount(result8)
    if count8:
        logger.warning(f"Recovered {count8} stale vocalization training job(s)")

    result9 = await session.execute(
        update(VocalizationInferenceJob)
        .where(
            VocalizationInferenceJob.status == "running",
            VocalizationInferenceJob.updated_at < cutoff,
        )
        .values(
            status="queued",
            updated_at=datetime.now(timezone.utc),
        )
    )
    count9 = _rowcount(result9)
    if count9:
        logger.warning(f"Recovered {count9} stale vocalization inference job(s)")

    total = (
        count + count2 + count3 + count4 + count5 + count6 + count7 + count8 + count9
    )
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
    # Retry a few times to handle races where another worker claims our
    # selected candidate between SELECT and UPDATE.
    for _ in range(3):
        job = await _claim_next_job(
            session,
            ProcessingJob,
            status_attr=ProcessingJob.status,
            queued_value=JobStatus.queued.value,
            running_value=JobStatus.running.value,
            order_attr=ProcessingJob.created_at,
            extra_filters=(~ProcessingJob.encoding_signature.in_(running_sigs),),
        )
        if job is not None:
            return job
    return None


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
        update(ProcessingJob).where(ProcessingJob.id == job_id).values(**values)
    )
    await session.commit()


async def fail_processing_job(session: AsyncSession, job_id: str, error: str) -> None:
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
    for _ in range(3):
        job = await _claim_next_job(
            session,
            ClusteringJob,
            status_attr=ClusteringJob.status,
            queued_value="queued",
            running_value="running",
            order_attr=ClusteringJob.created_at,
        )
        if job is not None:
            return job
    return None


async def complete_clustering_job(session: AsyncSession, job_id: str) -> None:
    await session.execute(
        update(ClusteringJob)
        .where(ClusteringJob.id == job_id)
        .values(status="complete", updated_at=datetime.now(timezone.utc))
    )
    await session.commit()


async def fail_clustering_job(session: AsyncSession, job_id: str, error: str) -> None:
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
    for _ in range(3):
        job = await _claim_next_job(
            session,
            ClassifierTrainingJob,
            status_attr=ClassifierTrainingJob.status,
            queued_value="queued",
            running_value="running",
            order_attr=ClassifierTrainingJob.created_at,
        )
        if job is not None:
            return job
    return None


async def complete_training_job(session: AsyncSession, job_id: str) -> None:
    await session.execute(
        update(ClassifierTrainingJob)
        .where(ClassifierTrainingJob.id == job_id)
        .values(status="complete", updated_at=datetime.now(timezone.utc))
    )
    await session.commit()


async def fail_training_job(session: AsyncSession, job_id: str, error: str) -> None:
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
    for _ in range(3):
        job = await _claim_next_job(
            session,
            DetectionJob,
            status_attr=DetectionJob.status,
            queued_value="queued",
            running_value="running",
            order_attr=DetectionJob.created_at,
            extra_filters=(DetectionJob.hydrophone_id.is_(None),),
        )
        if job is not None:
            return job
    return None


async def claim_hydrophone_detection_job(
    session: AsyncSession,
) -> Optional[DetectionJob]:
    """Claim a queued hydrophone detection job."""
    for _ in range(3):
        job = await _claim_next_job(
            session,
            DetectionJob,
            status_attr=DetectionJob.status,
            queued_value="queued",
            running_value="running",
            order_attr=DetectionJob.created_at,
            extra_filters=(DetectionJob.hydrophone_id.isnot(None),),
        )
        if job is not None:
            return job
    return None


async def complete_detection_job(session: AsyncSession, job_id: str) -> None:
    await session.execute(
        update(DetectionJob)
        .where(DetectionJob.id == job_id)
        .values(status="complete", updated_at=datetime.now(timezone.utc))
    )
    await session.commit()


async def fail_detection_job(session: AsyncSession, job_id: str, error: str) -> None:
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
    for _ in range(3):
        job = await _claim_next_job(
            session,
            DetectionJob,
            status_attr=DetectionJob.extract_status,
            queued_value="queued",
            running_value="running",
            order_attr=DetectionJob.updated_at,
        )
        if job is not None:
            return job
    return None


async def complete_extraction_job(session: AsyncSession, job_id: str) -> None:
    await session.execute(
        update(DetectionJob)
        .where(DetectionJob.id == job_id)
        .values(extract_status="complete", updated_at=datetime.now(timezone.utc))
    )
    await session.commit()


async def fail_extraction_job(session: AsyncSession, job_id: str, error: str) -> None:
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


# ---- Retrain Workflows ----


async def claim_retrain_workflow(
    session: AsyncSession,
) -> Optional[RetrainWorkflow]:
    """Claim a queued retrain workflow (queued → importing)."""
    for _ in range(3):
        wf = await _claim_next_job(
            session,
            RetrainWorkflow,
            status_attr=RetrainWorkflow.status,
            queued_value="queued",
            running_value="importing",
            order_attr=RetrainWorkflow.created_at,
        )
        if wf is not None:
            return wf
    return None


# ---- Search Jobs ----


async def claim_search_job(session: AsyncSession) -> Optional[SearchJob]:
    """Claim a queued search job atomically."""
    for _ in range(3):
        job = await _claim_next_job(
            session,
            SearchJob,
            status_attr=SearchJob.status,
            queued_value="queued",
            running_value="running",
            order_attr=SearchJob.created_at,
        )
        if job is not None:
            return job
    return None


async def complete_search_job(
    session: AsyncSession,
    job_id: str,
    model_version: str,
    embedding_vector: str,
) -> None:
    await session.execute(
        update(SearchJob)
        .where(SearchJob.id == job_id)
        .values(
            status="complete",
            model_version=model_version,
            embedding_vector=embedding_vector,
            updated_at=datetime.now(timezone.utc),
        )
    )
    await session.commit()


async def fail_search_job(session: AsyncSession, job_id: str, error: str) -> None:
    await session.execute(
        update(SearchJob)
        .where(SearchJob.id == job_id)
        .values(
            status="failed",
            error_message=error,
            updated_at=datetime.now(timezone.utc),
        )
    )
    await session.commit()


# ---- Label Processing Jobs ----


async def claim_label_processing_job(
    session: AsyncSession,
) -> Optional[Any]:
    from humpback.models.label_processing import LabelProcessingJob

    for _ in range(3):
        job = await _claim_next_job(
            session,
            LabelProcessingJob,
            status_attr=LabelProcessingJob.status,
            queued_value="queued",
            running_value="running",
            order_attr=LabelProcessingJob.created_at,
        )
        if job is not None:
            return job
    return None


async def complete_label_processing_job(session: AsyncSession, job_id: str) -> None:
    from humpback.models.label_processing import LabelProcessingJob

    await session.execute(
        update(LabelProcessingJob)
        .where(LabelProcessingJob.id == job_id)
        .values(status="complete", updated_at=datetime.now(timezone.utc))
    )
    await session.commit()


async def fail_label_processing_job(
    session: AsyncSession, job_id: str, error: str
) -> None:
    from humpback.models.label_processing import LabelProcessingJob

    await session.execute(
        update(LabelProcessingJob)
        .where(LabelProcessingJob.id == job_id)
        .values(
            status="failed",
            error_message=error,
            updated_at=datetime.now(timezone.utc),
        )
    )
    await session.commit()


# ---- Vocalization Jobs ----


async def claim_vocalization_training_job(session: AsyncSession):
    from humpback.models.vocalization import VocalizationTrainingJob

    for _ in range(3):
        job = await _claim_next_job(
            session,
            VocalizationTrainingJob,
            status_attr=VocalizationTrainingJob.status,
            queued_value="queued",
            running_value="running",
            order_attr=VocalizationTrainingJob.created_at,
        )
        if job is not None:
            return job
    return None


async def claim_vocalization_inference_job(session: AsyncSession):
    from humpback.models.vocalization import VocalizationInferenceJob

    for _ in range(3):
        job = await _claim_next_job(
            session,
            VocalizationInferenceJob,
            status_attr=VocalizationInferenceJob.status,
            queued_value="queued",
            running_value="running",
            order_attr=VocalizationInferenceJob.created_at,
        )
        if job is not None:
            return job
    return None
