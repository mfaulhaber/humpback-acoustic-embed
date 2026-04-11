"""Service layer for the call parsing pipeline (Phase 0).

Creates parent runs and their initial Pass 1 job, loads nested run
state, and cascades deletion across the four child tables. Phase 0
knows nothing about the pass bodies themselves — those land when Passes
1–3 replace the worker shells.
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
from humpback.config import Settings
from humpback.models.call_parsing import (
    CallParsingRun,
    EventClassificationJob,
    EventSegmentationJob,
    RegionDetectionJob,
)


async def create_parent_run(
    session: AsyncSession,
    audio_source_id: str,
    config_snapshot: Optional[str] = None,
) -> CallParsingRun:
    """Create a parent run + queued Pass 1 job in a single transaction."""
    run = CallParsingRun(
        audio_source_id=audio_source_id,
        status="queued",
        config_snapshot=config_snapshot,
    )
    session.add(run)
    await session.flush()

    region_job = RegionDetectionJob(
        audio_source_id=audio_source_id,
        parent_run_id=run.id,
        status="queued",
    )
    session.add(region_job)
    await session.flush()

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
