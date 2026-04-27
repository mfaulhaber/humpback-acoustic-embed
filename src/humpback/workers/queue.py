"""SQL-backed job queue with claim semantics."""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.models.classifier import ClassifierTrainingJob, DetectionJob
from humpback.models.clustering import ClusteringJob
from humpback.models.detection_embedding_job import DetectionEmbeddingJob  # noqa: E402
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

    result10 = await session.execute(
        update(DetectionEmbeddingJob)
        .where(
            DetectionEmbeddingJob.status == "running",
            DetectionEmbeddingJob.updated_at < cutoff,
        )
        .values(
            status="queued",
            updated_at=datetime.now(timezone.utc),
        )
    )
    count10 = _rowcount(result10)
    if count10:
        logger.warning(f"Recovered {count10} stale detection embedding job(s)")

    from humpback.models.hyperparameter import (
        HyperparameterManifest,
        HyperparameterSearchJob,
    )

    result11 = await session.execute(
        update(HyperparameterManifest)
        .where(
            HyperparameterManifest.status == "running",
            HyperparameterManifest.updated_at < cutoff,
        )
        .values(
            status="queued",
            updated_at=datetime.now(timezone.utc),
        )
    )
    count11 = _rowcount(result11)
    if count11:
        logger.warning(f"Recovered {count11} stale hyperparameter manifest job(s)")

    result12 = await session.execute(
        update(HyperparameterSearchJob)
        .where(
            HyperparameterSearchJob.status == "running",
            HyperparameterSearchJob.updated_at < cutoff,
        )
        .values(
            status="queued",
            updated_at=datetime.now(timezone.utc),
        )
    )
    count12 = _rowcount(result12)
    if count12:
        logger.warning(f"Recovered {count12} stale hyperparameter search job(s)")

    from humpback.models.call_parsing import EventSegmentationJob, RegionDetectionJob

    result13 = await session.execute(
        update(RegionDetectionJob)
        .where(
            RegionDetectionJob.status == "running",
            RegionDetectionJob.updated_at < cutoff,
        )
        .values(
            status="queued",
            updated_at=datetime.now(timezone.utc),
        )
    )
    count13 = _rowcount(result13)
    if count13:
        logger.warning(f"Recovered {count13} stale region detection job(s)")

    result15 = await session.execute(
        update(EventSegmentationJob)
        .where(
            EventSegmentationJob.status == "running",
            EventSegmentationJob.updated_at < cutoff,
        )
        .values(
            status="queued",
            updated_at=datetime.now(timezone.utc),
        )
    )
    count15 = _rowcount(result15)
    if count15:
        logger.warning(f"Recovered {count15} stale event segmentation job(s)")

    from humpback.models.segmentation_training import SegmentationTrainingJob

    result_stj = await session.execute(
        update(SegmentationTrainingJob)
        .where(
            SegmentationTrainingJob.status == "running",
            SegmentationTrainingJob.updated_at < cutoff,
        )
        .values(
            status="queued",
            updated_at=datetime.now(timezone.utc),
        )
    )
    count_stj = _rowcount(result_stj)
    if count_stj:
        logger.warning(f"Recovered {count_stj} stale segmentation training job(s)")

    from humpback.models.feedback_training import EventClassifierTrainingJob

    result17 = await session.execute(
        update(EventClassifierTrainingJob)
        .where(
            EventClassifierTrainingJob.status == "running",
            EventClassifierTrainingJob.updated_at < cutoff,
        )
        .values(
            status="queued",
            updated_at=datetime.now(timezone.utc),
        )
    )
    count17 = _rowcount(result17)
    if count17:
        logger.warning(f"Recovered {count17} stale classifier feedback training job(s)")

    total = (
        count
        + count2
        + count3
        + count4
        + count5
        + count6
        + count7
        + count8
        + count9
        + count10
        + count11
        + count12
        + count13
        + count15
        + count_stj
        + count17
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


# ---- Call Parsing Pipeline Jobs ----


async def claim_region_detection_job(session: AsyncSession):
    from humpback.models.call_parsing import RegionDetectionJob

    for _ in range(3):
        job = await _claim_next_job(
            session,
            RegionDetectionJob,
            status_attr=RegionDetectionJob.status,
            queued_value="queued",
            running_value="running",
            order_attr=RegionDetectionJob.created_at,
        )
        if job is not None:
            return job
    return None


async def claim_event_segmentation_job(session: AsyncSession):
    from humpback.models.call_parsing import EventSegmentationJob

    for _ in range(3):
        job = await _claim_next_job(
            session,
            EventSegmentationJob,
            status_attr=EventSegmentationJob.status,
            queued_value="queued",
            running_value="running",
            order_attr=EventSegmentationJob.created_at,
        )
        if job is not None:
            return job
    return None


async def claim_event_classification_job(session: AsyncSession):
    from humpback.models.call_parsing import EventClassificationJob

    for _ in range(3):
        job = await _claim_next_job(
            session,
            EventClassificationJob,
            status_attr=EventClassificationJob.status,
            queued_value="queued",
            running_value="running",
            order_attr=EventClassificationJob.created_at,
        )
        if job is not None:
            return job
    return None


async def claim_segmentation_training_job(session: AsyncSession):
    from humpback.models.segmentation_training import SegmentationTrainingJob

    for _ in range(3):
        job = await _claim_next_job(
            session,
            SegmentationTrainingJob,
            status_attr=SegmentationTrainingJob.status,
            queued_value="queued",
            running_value="running",
            order_attr=SegmentationTrainingJob.created_at,
        )
        if job is not None:
            return job
    return None


async def claim_classifier_feedback_training_job(session: AsyncSession):
    from humpback.models.feedback_training import EventClassifierTrainingJob

    for _ in range(3):
        job = await _claim_next_job(
            session,
            EventClassifierTrainingJob,
            status_attr=EventClassifierTrainingJob.status,
            queued_value="queued",
            running_value="running",
            order_attr=EventClassifierTrainingJob.created_at,
        )
        if job is not None:
            return job
    return None


# ---- Detection Embedding Jobs ----


# ---- Hyperparameter Jobs ----


async def claim_manifest_job(session: AsyncSession):
    from humpback.models.hyperparameter import HyperparameterManifest

    for _ in range(3):
        job = await _claim_next_job(
            session,
            HyperparameterManifest,
            status_attr=HyperparameterManifest.status,
            queued_value="queued",
            running_value="running",
            order_attr=HyperparameterManifest.created_at,
        )
        if job is not None:
            return job
    return None


async def claim_hyperparameter_search_job(session: AsyncSession):
    from humpback.models.hyperparameter import HyperparameterSearchJob

    for _ in range(3):
        job = await _claim_next_job(
            session,
            HyperparameterSearchJob,
            status_attr=HyperparameterSearchJob.status,
            queued_value="queued",
            running_value="running",
            order_attr=HyperparameterSearchJob.created_at,
        )
        if job is not None:
            return job
    return None


async def claim_continuous_embedding_job(session: AsyncSession):
    from humpback.models.sequence_models import ContinuousEmbeddingJob

    for _ in range(3):
        job = await _claim_next_job(
            session,
            ContinuousEmbeddingJob,
            status_attr=ContinuousEmbeddingJob.status,
            queued_value=JobStatus.queued.value,
            running_value=JobStatus.running.value,
            order_attr=ContinuousEmbeddingJob.created_at,
        )
        if job is not None:
            return job
    return None


async def claim_hmm_sequence_job(session: AsyncSession):
    from humpback.models.sequence_models import HMMSequenceJob

    for _ in range(3):
        job = await _claim_next_job(
            session,
            HMMSequenceJob,
            status_attr=HMMSequenceJob.status,
            queued_value=JobStatus.queued.value,
            running_value=JobStatus.running.value,
            order_attr=HMMSequenceJob.created_at,
        )
        if job is not None:
            return job
    return None


async def claim_window_classification_job(session: AsyncSession):
    from humpback.models.call_parsing import WindowClassificationJob

    for _ in range(3):
        job = await _claim_next_job(
            session,
            WindowClassificationJob,
            status_attr=WindowClassificationJob.status,
            queued_value="queued",
            running_value="running",
            order_attr=WindowClassificationJob.created_at,
        )
        if job is not None:
            return job
    return None


async def claim_detection_embedding_job(
    session: AsyncSession,
) -> DetectionEmbeddingJob | None:
    for _ in range(3):
        job = await _claim_next_job(
            session,
            DetectionEmbeddingJob,
            status_attr=DetectionEmbeddingJob.status,
            queued_value="queued",
            running_value="running",
            order_attr=DetectionEmbeddingJob.created_at,
        )
        if job is not None:
            return job
    return None


async def complete_detection_embedding_job(session: AsyncSession, job_id: str) -> None:
    await session.execute(
        update(DetectionEmbeddingJob)
        .where(DetectionEmbeddingJob.id == job_id)
        .values(status="complete", updated_at=datetime.now(timezone.utc))
    )
    await session.commit()


async def fail_detection_embedding_job(
    session: AsyncSession, job_id: str, error: str
) -> None:
    await session.execute(
        update(DetectionEmbeddingJob)
        .where(DetectionEmbeddingJob.id == job_id)
        .values(
            status="failed",
            error_message=error,
            updated_at=datetime.now(timezone.utc),
        )
    )
    await session.commit()
