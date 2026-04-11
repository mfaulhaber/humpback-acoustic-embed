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
from humpback.schemas.call_parsing import CreateRegionJobRequest


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
