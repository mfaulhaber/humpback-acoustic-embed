"""Service layer for the call parsing pipeline.

Creates parent runs and their Pass 1 region detection child jobs with
exactly-one-of source validation and FK resolution against
``audio_files``, ``model_configs``, ``classifier_models``, and the
packaged hydrophone registry. Loads nested run state and cascades
deletion across the four child tables.
"""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from humpback.models.segmentation_training import SegmentationTrainingDataset

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from humpback.call_parsing.storage import (
    classification_job_dir,
    region_job_dir,
    segmentation_job_dir,
)
from humpback.config import Settings, get_archive_source
from humpback.models.audio import AudioFile
from humpback.models.call_parsing import (
    CallParsingRun,
    EventClassificationJob,
    EventSegmentationJob,
    RegionDetectionJob,
)
from humpback.models.classifier import ClassifierModel
from humpback.models.model_registry import ModelConfig
from humpback.schemas.call_parsing import (
    CreateRegionJobRequest,
)


class CallParsingFKError(Exception):
    """Raised when a foreign-key lookup for a region detection job fails.

    The router converts this to an HTTP 404 response. ``field`` names the
    request field that did not resolve; ``value`` echoes the unresolved id
    so the error message points the user at the broken input.
    """

    def __init__(self, field: str, value: str) -> None:
        self.field = field
        self.value = value
        super().__init__(f"{field}={value!r} not found")


async def _require_audio_file(session: AsyncSession, audio_file_id: str) -> None:
    result = await session.execute(
        select(AudioFile.id).where(AudioFile.id == audio_file_id)
    )
    if result.scalar_one_or_none() is None:
        raise CallParsingFKError("audio_file_id", audio_file_id)


async def _require_model_config(session: AsyncSession, model_config_id: str) -> None:
    result = await session.execute(
        select(ModelConfig.id).where(ModelConfig.id == model_config_id)
    )
    if result.scalar_one_or_none() is None:
        raise CallParsingFKError("model_config_id", model_config_id)


async def _require_classifier_model(
    session: AsyncSession, classifier_model_id: str
) -> None:
    result = await session.execute(
        select(ClassifierModel.id).where(ClassifierModel.id == classifier_model_id)
    )
    if result.scalar_one_or_none() is None:
        raise CallParsingFKError("classifier_model_id", classifier_model_id)


def _require_hydrophone(hydrophone_id: str) -> None:
    if get_archive_source(hydrophone_id) is None:
        raise CallParsingFKError("hydrophone_id", hydrophone_id)


async def _validate_region_job_request(
    session: AsyncSession, request: CreateRegionJobRequest
) -> None:
    """Validate all four FKs in request order: source, model, classifier."""
    if request.audio_file_id is not None:
        await _require_audio_file(session, request.audio_file_id)
    elif request.hydrophone_id is not None:
        _require_hydrophone(request.hydrophone_id)
    # Pydantic's exactly-one-of validator guarantees one of the two
    # branches above is reachable — no untyped fallthrough.

    await _require_model_config(session, request.model_config_id)
    await _require_classifier_model(session, request.classifier_model_id)


async def create_region_job(
    session: AsyncSession,
    request: CreateRegionJobRequest,
) -> RegionDetectionJob:
    """Create a queued Pass 1 region detection job from a validated request.

    Raises ``CallParsingFKError`` if any of ``audio_file_id``,
    ``hydrophone_id``, ``model_config_id``, or ``classifier_model_id`` does
    not resolve. The caller owns the session / transaction boundary;
    ``create_region_job`` only flushes so the generated ``job.id`` is
    visible to subsequent operations in the same transaction.
    """
    await _validate_region_job_request(session, request)

    job = RegionDetectionJob(
        status="queued",
        parent_run_id=request.parent_run_id,
        audio_file_id=request.audio_file_id,
        hydrophone_id=request.hydrophone_id,
        start_timestamp=request.start_timestamp,
        end_timestamp=request.end_timestamp,
        model_config_id=request.model_config_id,
        classifier_model_id=request.classifier_model_id,
        config_json=request.config.model_dump_json(),
    )
    session.add(job)
    await session.flush()
    return job


async def create_parent_run(
    session: AsyncSession,
    request: CreateRegionJobRequest,
) -> CallParsingRun:
    """Create a parent run + its queued Pass 1 child in a single transaction.

    The source fields are mirrored onto both the parent row and the Pass 1
    child so downstream list/detail queries can filter the parent by
    source without joining through the child table. The Pass 1 config is
    also snapshotted onto the parent as ``config_snapshot`` — Phase 0
    intended that column to hold the aggregated cross-pass config, and
    with only Pass 1 implemented today it is the Pass 1 config.
    """
    run = CallParsingRun(
        status="queued",
        audio_file_id=request.audio_file_id,
        hydrophone_id=request.hydrophone_id,
        start_timestamp=request.start_timestamp,
        end_timestamp=request.end_timestamp,
        config_snapshot=request.config.model_dump_json(),
    )
    session.add(run)
    await session.flush()

    child_request = request.model_copy(update={"parent_run_id": run.id})
    region_job = await create_region_job(session, child_request)

    run.region_detection_job_id = region_job.id
    await session.commit()
    await session.refresh(run)
    return run


async def get_parent_run(
    session: AsyncSession, run_id: str
) -> Optional[CallParsingRun]:
    result = await session.execute(
        select(CallParsingRun).where(CallParsingRun.id == run_id)
    )
    return result.scalar_one_or_none()


async def list_parent_runs(
    session: AsyncSession, limit: int = 50, offset: int = 0
) -> list[CallParsingRun]:
    result = await session.execute(
        select(CallParsingRun)
        .order_by(CallParsingRun.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    return list(result.scalars().all())


async def _load_child_jobs(
    session: AsyncSession, run: CallParsingRun
) -> tuple[
    Optional[RegionDetectionJob],
    Optional[EventSegmentationJob],
    Optional[EventClassificationJob],
]:
    rd = None
    if run.region_detection_job_id:
        rd = await session.get(RegionDetectionJob, run.region_detection_job_id)
    es = None
    if run.event_segmentation_job_id:
        es = await session.get(EventSegmentationJob, run.event_segmentation_job_id)
    ec = None
    if run.event_classification_job_id:
        ec = await session.get(EventClassificationJob, run.event_classification_job_id)
    return rd, es, ec


async def load_run_with_children(
    session: AsyncSession, run_id: str
) -> Optional[
    tuple[
        CallParsingRun,
        Optional[RegionDetectionJob],
        Optional[EventSegmentationJob],
        Optional[EventClassificationJob],
    ]
]:
    run = await get_parent_run(session, run_id)
    if run is None:
        return None
    rd, es, ec = await _load_child_jobs(session, run)
    return run, rd, es, ec


def _remove_dir(path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


async def delete_parent_run(
    session: AsyncSession, run_id: str, settings: Settings
) -> bool:
    run = await get_parent_run(session, run_id)
    if run is None:
        return False

    rd, es, ec = await _load_child_jobs(session, run)

    if ec is not None:
        _remove_dir(classification_job_dir(settings.storage_root, ec.id))
        await session.delete(ec)
    if es is not None:
        _remove_dir(segmentation_job_dir(settings.storage_root, es.id))
        await session.delete(es)
    if rd is not None:
        _remove_dir(region_job_dir(settings.storage_root, rd.id))
        await session.delete(rd)

    await session.delete(run)
    await session.commit()
    return True


async def list_region_detection_jobs(
    session: AsyncSession,
) -> list[RegionDetectionJob]:
    result = await session.execute(
        select(RegionDetectionJob).order_by(RegionDetectionJob.created_at.desc())
    )
    return list(result.scalars().all())


async def get_region_detection_job(
    session: AsyncSession, job_id: str
) -> Optional[RegionDetectionJob]:
    return await session.get(RegionDetectionJob, job_id)


async def delete_region_detection_job(
    session: AsyncSession, job_id: str, settings: Settings
) -> bool:
    job = await session.get(RegionDetectionJob, job_id)
    if job is None:
        return False
    _remove_dir(region_job_dir(settings.storage_root, job.id))
    await session.delete(job)
    await session.commit()
    return True


async def list_event_segmentation_jobs(
    session: AsyncSession,
) -> list[EventSegmentationJob]:
    result = await session.execute(
        select(EventSegmentationJob).order_by(EventSegmentationJob.created_at.desc())
    )
    return list(result.scalars().all())


async def list_segmentation_jobs_with_correction_counts(
    session: AsyncSession,
) -> list[dict]:
    """Return completed segmentation jobs with correction counts and hydrophone info.

    Uses a LEFT JOIN subquery so there is no N+1 overhead.
    """
    from humpback.models.feedback_training import EventBoundaryCorrection

    correction_counts = (
        select(
            EventBoundaryCorrection.event_segmentation_job_id.label("job_id"),
            func.count(EventBoundaryCorrection.id).label("correction_count"),
        )
        .group_by(EventBoundaryCorrection.event_segmentation_job_id)
        .subquery()
    )

    stmt = (
        select(
            EventSegmentationJob,
            func.coalesce(correction_counts.c.correction_count, 0).label(
                "correction_count"
            ),
            RegionDetectionJob.hydrophone_id,
            RegionDetectionJob.start_timestamp,
            RegionDetectionJob.end_timestamp,
        )
        .join(
            RegionDetectionJob,
            RegionDetectionJob.id == EventSegmentationJob.region_detection_job_id,
        )
        .outerjoin(
            correction_counts,
            correction_counts.c.job_id == EventSegmentationJob.id,
        )
        .where(EventSegmentationJob.status == "complete")
        .order_by(EventSegmentationJob.created_at.desc())
    )

    result = await session.execute(stmt)
    rows = result.all()

    out: list[dict] = []
    for row in rows:
        job = row[0]
        d = {
            "id": job.id,
            "status": job.status,
            "parent_run_id": job.parent_run_id,
            "error_message": job.error_message,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
            "started_at": job.started_at,
            "completed_at": job.completed_at,
            "region_detection_job_id": job.region_detection_job_id,
            "segmentation_model_id": job.segmentation_model_id,
            "config_json": job.config_json,
            "event_count": job.event_count,
            "correction_count": row[1],
            "hydrophone_id": row[2],
            "start_timestamp": row[3],
            "end_timestamp": row[4],
        }
        out.append(d)
    return out


async def get_event_segmentation_job(
    session: AsyncSession, job_id: str
) -> Optional[EventSegmentationJob]:
    return await session.get(EventSegmentationJob, job_id)


async def delete_event_segmentation_job(
    session: AsyncSession, job_id: str, settings: Settings
) -> bool:
    job = await session.get(EventSegmentationJob, job_id)
    if job is None:
        return False
    _remove_dir(segmentation_job_dir(settings.storage_root, job.id))
    await session.delete(job)
    await session.commit()
    return True


async def create_event_classification_job(session: AsyncSession, request):
    """Create a queued Pass 3 event classification job.

    Validates that ``event_segmentation_job_id`` resolves and is
    ``complete``, and that ``vocalization_model_id`` resolves and has
    ``model_family='pytorch_event_cnn'`` + ``input_mode='segmented_event'``.
    """
    from humpback.models.vocalization import (
        VocalizationClassifierModel as _VocModel,
    )
    from humpback.schemas.call_parsing import CreateEventClassificationJobRequest

    assert isinstance(request, CreateEventClassificationJobRequest)

    es = await session.get(EventSegmentationJob, request.event_segmentation_job_id)
    if es is None:
        raise CallParsingFKError(
            "event_segmentation_job_id", request.event_segmentation_job_id
        )
    if es.status != "complete":
        raise CallParsingStateError(
            f"Upstream event segmentation job status is {es.status!r}, not 'complete'"
        )

    vm = await session.get(_VocModel, request.vocalization_model_id)
    if vm is None:
        raise CallParsingFKError("vocalization_model_id", request.vocalization_model_id)
    if vm.model_family != "pytorch_event_cnn" or vm.input_mode != "segmented_event":
        raise CallParsingValidationError(
            f"Vocalization model {request.vocalization_model_id} has "
            f"model_family={vm.model_family!r}, input_mode={vm.input_mode!r}; "
            f"expected pytorch_event_cnn / segmented_event"
        )

    import json

    job = EventClassificationJob(
        status="queued",
        parent_run_id=request.parent_run_id,
        event_segmentation_job_id=request.event_segmentation_job_id,
        vocalization_model_id=request.vocalization_model_id,
        config_json=json.dumps(request.config) if request.config else None,
    )
    session.add(job)
    await session.flush()
    return job


async def list_event_classification_jobs(
    session: AsyncSession,
) -> list[EventClassificationJob]:
    result = await session.execute(
        select(EventClassificationJob).order_by(
            EventClassificationJob.created_at.desc()
        )
    )
    return list(result.scalars().all())


async def get_event_classification_job(
    session: AsyncSession, job_id: str
) -> Optional[EventClassificationJob]:
    return await session.get(EventClassificationJob, job_id)


async def delete_event_classification_job(
    session: AsyncSession, job_id: str, settings: Settings
) -> bool:
    job = await session.get(EventClassificationJob, job_id)
    if job is None:
        return False
    _remove_dir(classification_job_dir(settings.storage_root, job.id))
    await session.delete(job)
    await session.commit()
    return True


# ---- Pass 2: segmentation training + inference jobs --------------------


class CallParsingStateError(Exception):
    """Raised when a create/delete request conflicts with current job state.

    The router maps this to HTTP 409. ``detail`` is surfaced verbatim in
    the response body.
    """

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(detail)


class CallParsingValidationError(Exception):
    """Raised when a request field has a valid FK but wrong semantics.

    The router maps this to HTTP 422.
    """

    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(detail)


async def create_segmentation_job(session, request):
    """Create a queued Pass 2 event segmentation job.

    Validates that both ``region_detection_job_id`` and
    ``segmentation_model_id`` resolve, and that the upstream Pass 1 job
    is in ``complete`` state. Raises ``CallParsingFKError`` (404) on a
    missing FK and ``CallParsingStateError`` (409) when the upstream
    Pass 1 job is not yet complete.
    """
    from humpback.models.call_parsing import (
        RegionDetectionJob as _RegionDetectionJob,
    )
    from humpback.models.call_parsing import (
        SegmentationModel as _SegmentationModel,
    )
    from humpback.schemas.call_parsing import CreateSegmentationJobRequest

    assert isinstance(request, CreateSegmentationJobRequest)

    rd = await session.get(_RegionDetectionJob, request.region_detection_job_id)
    if rd is None:
        raise CallParsingFKError(
            "region_detection_job_id", request.region_detection_job_id
        )
    if rd.status != "complete":
        raise CallParsingStateError(
            f"Upstream region detection job status is {rd.status!r}, not 'complete'"
        )

    sm_result = await session.execute(
        select(_SegmentationModel.id).where(
            _SegmentationModel.id == request.segmentation_model_id
        )
    )
    if sm_result.scalar_one_or_none() is None:
        raise CallParsingFKError("segmentation_model_id", request.segmentation_model_id)

    job = EventSegmentationJob(
        status="queued",
        parent_run_id=request.parent_run_id,
        region_detection_job_id=request.region_detection_job_id,
        segmentation_model_id=request.segmentation_model_id,
        config_json=request.config.model_dump_json(),
    )
    session.add(job)
    await session.flush()
    return job


# ---- Segmentation training datasets / jobs / models read-side ----------


async def list_segmentation_training_datasets(session: AsyncSession):
    """List all training datasets with their sample counts."""
    from humpback.models.segmentation_training import (
        SegmentationTrainingDataset,
        SegmentationTrainingSample,
    )

    count_subq = (
        select(
            SegmentationTrainingSample.training_dataset_id,
            func.count().label("sample_count"),
        )
        .group_by(SegmentationTrainingSample.training_dataset_id)
        .subquery()
    )
    stmt = (
        select(
            SegmentationTrainingDataset.id,
            SegmentationTrainingDataset.name,
            SegmentationTrainingDataset.created_at,
            func.coalesce(count_subq.c.sample_count, 0).label("sample_count"),
        )
        .outerjoin(
            count_subq,
            SegmentationTrainingDataset.id == count_subq.c.training_dataset_id,
        )
        .order_by(SegmentationTrainingDataset.created_at.desc())
    )
    result = await session.execute(stmt)
    return [row._asdict() for row in result.all()]


async def list_segmentation_models(session):
    from humpback.models.call_parsing import SegmentationModel

    result = await session.execute(
        select(SegmentationModel).order_by(SegmentationModel.created_at.desc())
    )
    return list(result.scalars().all())


async def get_segmentation_model(session, model_id: str):
    from humpback.models.call_parsing import SegmentationModel

    return await session.get(SegmentationModel, model_id)


async def delete_segmentation_model(session, model_id: str, settings: Settings) -> bool:
    """Delete a segmentation model row + its checkpoint directory.

    Raises ``CallParsingStateError`` (409) when the model is still
    referenced by an in-flight ``event_segmentation_jobs`` row.
    """
    from pathlib import Path as _Path

    from humpback.models.call_parsing import (
        EventSegmentationJob as _EventSegmentationJob,
    )
    from humpback.models.call_parsing import (
        SegmentationModel,
    )

    model = await session.get(SegmentationModel, model_id)
    if model is None:
        return False

    in_flight = await session.execute(
        select(_EventSegmentationJob.id).where(
            _EventSegmentationJob.segmentation_model_id == model_id,
            _EventSegmentationJob.status.in_(["queued", "running"]),
        )
    )
    if in_flight.scalar_one_or_none() is not None:
        raise CallParsingStateError(
            "Segmentation model is referenced by an in-flight event segmentation job"
        )

    if model.model_path:
        checkpoint_path = _Path(model.model_path)
        if checkpoint_path.exists() and checkpoint_path.is_file():
            checkpoint_dir = checkpoint_path.parent
            _remove_dir(checkpoint_dir)

    await session.delete(model)
    await session.commit()
    return True


# ---- Feedback training: boundary corrections (Pass 2) --------------------


async def upsert_boundary_corrections(
    session: AsyncSession,
    job_id: str,
    corrections: list,
) -> int:
    """Batch upsert boundary corrections for a segmentation job.

    Validates the job exists and is complete. For each correction,
    inserts a new row or updates an existing one keyed by
    ``(event_segmentation_job_id, event_id)``.
    """
    from humpback.models.feedback_training import EventBoundaryCorrection

    es = await session.get(EventSegmentationJob, job_id)
    if es is None:
        raise CallParsingFKError("event_segmentation_job_id", job_id)
    if es.status != "complete":
        raise CallParsingStateError(
            f"Segmentation job status is {es.status!r}, not 'complete'"
        )

    count = 0
    for c in corrections:
        existing = await session.execute(
            select(EventBoundaryCorrection).where(
                EventBoundaryCorrection.event_segmentation_job_id == job_id,
                EventBoundaryCorrection.event_id == c.event_id,
            )
        )
        row = existing.scalar_one_or_none()
        if row is not None:
            row.region_id = c.region_id
            row.correction_type = c.correction_type
            row.start_sec = c.start_sec
            row.end_sec = c.end_sec
        else:
            session.add(
                EventBoundaryCorrection(
                    event_segmentation_job_id=job_id,
                    event_id=c.event_id,
                    region_id=c.region_id,
                    correction_type=c.correction_type,
                    start_sec=c.start_sec,
                    end_sec=c.end_sec,
                )
            )
        count += 1

    await session.commit()
    return count


async def list_boundary_corrections(
    session: AsyncSession,
    job_id: str,
) -> list:
    """List all boundary corrections for a segmentation job."""
    from humpback.models.feedback_training import EventBoundaryCorrection

    result = await session.execute(
        select(EventBoundaryCorrection)
        .where(EventBoundaryCorrection.event_segmentation_job_id == job_id)
        .order_by(EventBoundaryCorrection.created_at)
    )
    return list(result.scalars().all())


async def clear_boundary_corrections(
    session: AsyncSession,
    job_id: str,
) -> None:
    """Delete all boundary corrections for a segmentation job."""
    from humpback.models.feedback_training import EventBoundaryCorrection

    from sqlalchemy import delete as _delete

    await session.execute(
        _delete(EventBoundaryCorrection).where(
            EventBoundaryCorrection.event_segmentation_job_id == job_id
        )
    )
    await session.commit()


# ---- Feedback training: type corrections (Pass 3) ------------------------


async def upsert_type_corrections(
    session: AsyncSession,
    job_id: str,
    corrections: list,
) -> int:
    """Batch upsert type corrections for a classification job.

    Validates the job exists and is complete. Upserts by
    ``(event_classification_job_id, event_id)`` — repeated calls
    overwrite the previous correction for the same event.
    """
    from humpback.models.feedback_training import EventTypeCorrection

    ec = await session.get(EventClassificationJob, job_id)
    if ec is None:
        raise CallParsingFKError("event_classification_job_id", job_id)
    if ec.status != "complete":
        raise CallParsingStateError(
            f"Classification job status is {ec.status!r}, not 'complete'"
        )

    count = 0
    for c in corrections:
        existing = await session.execute(
            select(EventTypeCorrection).where(
                EventTypeCorrection.event_classification_job_id == job_id,
                EventTypeCorrection.event_id == c.event_id,
            )
        )
        row = existing.scalar_one_or_none()
        if row is not None:
            row.type_name = c.type_name
        else:
            session.add(
                EventTypeCorrection(
                    event_classification_job_id=job_id,
                    event_id=c.event_id,
                    type_name=c.type_name,
                )
            )
        count += 1

    await session.commit()
    return count


async def list_type_corrections(
    session: AsyncSession,
    job_id: str,
) -> list:
    """List all type corrections for a classification job."""
    from humpback.models.feedback_training import EventTypeCorrection

    result = await session.execute(
        select(EventTypeCorrection)
        .where(EventTypeCorrection.event_classification_job_id == job_id)
        .order_by(EventTypeCorrection.created_at)
    )
    return list(result.scalars().all())


async def clear_type_corrections(
    session: AsyncSession,
    job_id: str,
) -> None:
    """Delete all type corrections for a classification job."""
    from humpback.models.feedback_training import EventTypeCorrection

    from sqlalchemy import delete as _delete

    await session.execute(
        _delete(EventTypeCorrection).where(
            EventTypeCorrection.event_classification_job_id == job_id
        )
    )
    await session.commit()


# ---- Feedback training: classifier feedback training jobs (Pass 3) -------


async def create_classifier_training_job(session: AsyncSession, request):
    """Create a queued Pass 3 feedback training job.

    Validates that every source classification job ID exists and is complete.
    """
    import json

    from humpback.models.feedback_training import EventClassifierTrainingJob
    from humpback.schemas.call_parsing import CreateClassifierTrainingJobRequest

    assert isinstance(request, CreateClassifierTrainingJobRequest)

    for sid in request.source_job_ids:
        ec = await session.get(EventClassificationJob, sid)
        if ec is None:
            raise CallParsingFKError("source_job_ids", sid)
        if ec.status != "complete":
            raise CallParsingStateError(
                f"Source classification job {sid} status is {ec.status!r}, not 'complete'"
            )

    job = EventClassifierTrainingJob(
        status="queued",
        source_job_ids=json.dumps(request.source_job_ids),
        config_json=request.config.model_dump_json(),
    )
    session.add(job)
    await session.flush()
    return job


async def list_classifier_training_jobs(
    session: AsyncSession,
) -> list:
    from humpback.models.feedback_training import EventClassifierTrainingJob

    result = await session.execute(
        select(EventClassifierTrainingJob).order_by(
            EventClassifierTrainingJob.created_at.desc()
        )
    )
    return list(result.scalars().all())


async def get_classifier_training_job(session: AsyncSession, job_id: str):
    from humpback.models.feedback_training import EventClassifierTrainingJob

    return await session.get(EventClassifierTrainingJob, job_id)


async def delete_classifier_training_job(session: AsyncSession, job_id: str) -> bool:
    from humpback.models.feedback_training import EventClassifierTrainingJob

    job = await session.get(EventClassifierTrainingJob, job_id)
    if job is None:
        return False
    await session.delete(job)
    await session.commit()
    return True


# ---- Feedback training: classifier model management (Pass 3) -------------


async def list_classifier_models(session: AsyncSession) -> list:
    """List vocalization models filtered to ``model_family='pytorch_event_cnn'``."""
    from humpback.models.vocalization import VocalizationClassifierModel

    result = await session.execute(
        select(VocalizationClassifierModel)
        .where(VocalizationClassifierModel.model_family == "pytorch_event_cnn")
        .order_by(VocalizationClassifierModel.created_at.desc())
    )
    return list(result.scalars().all())


async def delete_classifier_model(
    session: AsyncSession, model_id: str, settings: Settings
) -> bool:
    """Delete a ``pytorch_event_cnn`` model + its checkpoint directory.

    Raises ``CallParsingStateError`` (409) when the model is referenced by
    an in-flight classification job or feedback training job.
    """
    from pathlib import Path as _Path

    from humpback.models.feedback_training import EventClassifierTrainingJob
    from humpback.models.vocalization import VocalizationClassifierModel

    model = await session.get(VocalizationClassifierModel, model_id)
    if model is None:
        return False

    if model.model_family != "pytorch_event_cnn":
        return False

    in_flight_cls = await session.execute(
        select(EventClassificationJob.id).where(
            EventClassificationJob.vocalization_model_id == model_id,
            EventClassificationJob.status.in_(["queued", "running"]),
        )
    )
    if in_flight_cls.scalar_one_or_none() is not None:
        raise CallParsingStateError(
            "Classifier model is referenced by an in-flight classification job"
        )

    in_flight_train = await session.execute(
        select(EventClassifierTrainingJob.id).where(
            EventClassifierTrainingJob.vocalization_model_id == model_id,
            EventClassifierTrainingJob.status.in_(["queued", "running"]),
        )
    )
    if in_flight_train.scalar_one_or_none() is not None:
        raise CallParsingStateError(
            "Classifier model is referenced by an in-flight training job"
        )

    if model.model_dir_path:
        model_dir = _Path(model.model_dir_path)
        if model_dir.exists() and model_dir.is_dir():
            _remove_dir(model_dir)

    await session.delete(model)
    await session.commit()
    return True


# ---- Dataset extraction from corrections -----------------------------------


async def create_dataset_from_corrections(
    session: AsyncSession,
    segmentation_job_ids: list[str],
    settings: Settings,
    name: Optional[str] = None,
    description: Optional[str] = None,
) -> tuple["SegmentationTrainingDataset", int]:
    """Extract corrections from one or more segmentation jobs into a new dataset.

    Returns ``(dataset, sample_count)``.

    Each job is validated as existing and complete.  Jobs with zero
    corrections are silently skipped.  Raises ``ValueError`` if no
    samples are collected across all provided jobs.
    """
    from humpback.call_parsing.segmentation.extraction import (
        CorrectedSample,
        collect_corrected_samples,
    )
    from humpback.models.segmentation_training import (
        SegmentationTrainingDataset,
        SegmentationTrainingSample,
    )

    all_samples: list[tuple[str, CorrectedSample]] = []

    for seg_job_id in segmentation_job_ids:
        seg_job = await session.get(EventSegmentationJob, seg_job_id)
        if seg_job is None:
            raise ValueError(f"Segmentation job {seg_job_id} not found")
        if seg_job.status != "complete":
            raise ValueError(
                f"Segmentation job {seg_job_id} is not complete "
                f"(status={seg_job.status})"
            )

        upstream = await session.get(
            RegionDetectionJob, seg_job.region_detection_job_id
        )
        if upstream is None:
            raise ValueError(
                f"Upstream region detection job "
                f"{seg_job.region_detection_job_id} not found"
            )

        job_samples = await collect_corrected_samples(
            session, seg_job_id, settings.storage_root
        )
        for s in job_samples:
            all_samples.append((seg_job_id, s))

    if not all_samples:
        raise ValueError(
            "No corrected regions found across the provided segmentation jobs"
        )

    n_jobs = len(segmentation_job_ids)
    first_id = segmentation_job_ids[0]
    if name:
        dataset_name = name
    elif n_jobs == 1:
        dataset_name = f"corrections-{first_id[:8]}"
    else:
        dataset_name = f"corrections-{n_jobs}jobs-{first_id[:8]}"

    dataset = SegmentationTrainingDataset(name=dataset_name, description=description)
    session.add(dataset)
    await session.flush()

    for job_id, s in all_samples:
        session.add(
            SegmentationTrainingSample(
                training_dataset_id=dataset.id,
                hydrophone_id=s.hydrophone_id,
                start_timestamp=s.start_timestamp,
                end_timestamp=s.end_timestamp,
                crop_start_sec=s.crop_start_sec,
                crop_end_sec=s.crop_end_sec,
                events_json=s.events_json,
                source="boundary_correction",
                source_ref=job_id,
            )
        )

    await session.commit()
    return dataset, len(all_samples)


async def create_dataset_and_train(
    session: AsyncSession,
    segmentation_job_id: str,
    settings: Settings,
) -> tuple[str, str, int]:
    """Create a single-job dataset and queue a training job in one call.

    Convenience wrapper for the SegmentReviewWorkspace quick-retrain flow.
    Returns ``(dataset_id, training_job_id, sample_count)``.
    """
    from humpback.models.segmentation_training import SegmentationTrainingJob
    from humpback.schemas.call_parsing import SegmentationTrainingConfig

    dataset, sample_count = await create_dataset_from_corrections(
        session,
        segmentation_job_ids=[segmentation_job_id],
        settings=settings,
    )

    config = SegmentationTrainingConfig()
    job = SegmentationTrainingJob(
        training_dataset_id=dataset.id,
        config_json=config.model_dump_json(),
    )
    session.add(job)
    await session.commit()

    return dataset.id, job.id, sample_count
