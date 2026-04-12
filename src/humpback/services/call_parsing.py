"""Service layer for the call parsing pipeline.

Creates parent runs and their Pass 1 region detection child jobs with
exactly-one-of source validation and FK resolution against
``audio_files``, ``model_configs``, ``classifier_models``, and the
packaged hydrophone registry. Loads nested run state and cascades
deletion across the four child tables.
"""

from __future__ import annotations

import shutil
from typing import Optional

from sqlalchemy import select
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


async def create_segmentation_training_job(session, request):
    """Create a queued Pass 2 training job.

    Validates that ``training_dataset_id`` resolves; raises
    ``CallParsingFKError`` on miss. The caller owns the transaction
    boundary; this function only flushes so the generated ``job.id`` is
    visible.
    """
    from humpback.models.segmentation_training import (
        SegmentationTrainingDataset,
        SegmentationTrainingJob,
    )
    from humpback.schemas.call_parsing import CreateSegmentationTrainingJobRequest

    assert isinstance(request, CreateSegmentationTrainingJobRequest)
    ds_result = await session.execute(
        select(SegmentationTrainingDataset.id).where(
            SegmentationTrainingDataset.id == request.training_dataset_id
        )
    )
    if ds_result.scalar_one_or_none() is None:
        raise CallParsingFKError("training_dataset_id", request.training_dataset_id)

    job = SegmentationTrainingJob(
        status="queued",
        training_dataset_id=request.training_dataset_id,
        config_json=request.config.model_dump_json(),
    )
    session.add(job)
    await session.flush()
    return job


class CallParsingStateError(Exception):
    """Raised when a create/delete request conflicts with current job state.

    The router maps this to HTTP 409. ``detail`` is surfaced verbatim in
    the response body.
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


async def list_segmentation_training_jobs(session):
    from humpback.models.segmentation_training import SegmentationTrainingJob

    result = await session.execute(
        select(SegmentationTrainingJob).order_by(
            SegmentationTrainingJob.created_at.desc()
        )
    )
    return list(result.scalars().all())


async def get_segmentation_training_job(session, job_id: str):
    from humpback.models.segmentation_training import SegmentationTrainingJob

    return await session.get(SegmentationTrainingJob, job_id)


async def delete_segmentation_training_job(session, job_id: str) -> bool:
    """Delete a training job. Returns True if a row was removed.

    Raises ``CallParsingStateError`` (409) when the linked
    ``segmentation_models`` row is still referenced by an in-flight
    ``event_segmentation_jobs`` row.
    """
    from humpback.models.call_parsing import (
        EventSegmentationJob as _EventSegmentationJob,
    )
    from humpback.models.segmentation_training import SegmentationTrainingJob

    job = await session.get(SegmentationTrainingJob, job_id)
    if job is None:
        return False

    if job.segmentation_model_id:
        in_flight = await session.execute(
            select(_EventSegmentationJob.id).where(
                _EventSegmentationJob.segmentation_model_id
                == job.segmentation_model_id,
                _EventSegmentationJob.status.in_(["queued", "running"]),
            )
        )
        if in_flight.scalar_one_or_none() is not None:
            raise CallParsingStateError(
                "Segmentation training job's model is referenced by an "
                "in-flight event segmentation job"
            )

    await session.delete(job)
    await session.commit()
    return True


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
