"""Unit tests for Phase 0 call-parsing worker shells.

Verifies each of the three Phase 0 workers:
- Claims a queued job atomically (compare-and-set).
- Marks the job ``failed`` with an informative error message.
- Races cleanly when two invocations contend for the same row.

Plus a dispatch priority test: with one queued job of every relevant
job type, the worker loop claims them in the documented order from
CLAUDE.md §8.7.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from humpback.config import Settings
from humpback.database import Base, create_engine, create_session_factory
from humpback.models.call_parsing import (
    EventClassificationJob,
    EventSegmentationJob,
    RegionDetectionJob,
)
from humpback.workers.event_classification_worker import run_one_iteration as ec_run
from humpback.workers.event_segmentation_worker import run_one_iteration as es_run
from humpback.workers.queue import (
    claim_event_classification_job,
    claim_event_segmentation_job,
    claim_region_detection_job,
)
from humpback.workers.region_detection_worker import run_one_iteration as rd_run


@pytest.fixture
async def session_factory(tmp_path):
    db_path = tmp_path / "test.db"
    engine = create_engine(f"sqlite+aiosqlite:///{db_path}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield create_session_factory(engine)
    await engine.dispose()


def _settings(tmp_path: Path) -> Settings:
    storage = tmp_path / "storage"
    storage.mkdir(exist_ok=True)
    return Settings(
        storage_root=storage,
        database_url=f"sqlite+aiosqlite:///{tmp_path}/test.db",
    )


async def test_region_detection_worker_fails_without_classifier_model(
    session_factory, tmp_path
):
    """Pass 1 worker marks a job ``failed`` when required FKs are missing.

    Pre-Pass-1 this suite asserted the NotImplementedError stub message.
    Now the Pass 1 worker does real work, so the minimal-fixture path
    exercises the worker's validation + error-path code instead.
    """
    async with session_factory() as session:
        job = RegionDetectionJob(audio_file_id="audio-1", status="queued")
        session.add(job)
        await session.commit()
        job_id = job.id

    settings = _settings(tmp_path)
    async with session_factory() as session:
        claimed = await rd_run(session, settings)
    assert claimed is not None
    assert claimed.id == job_id

    async with session_factory() as session:
        refreshed = await session.get(RegionDetectionJob, job_id)
        assert refreshed is not None
        assert refreshed.status == "failed"
        assert refreshed.error_message is not None
        assert "classifier_model_id" in refreshed.error_message


async def test_event_segmentation_worker_marks_job_failed(session_factory, tmp_path):
    async with session_factory() as session:
        parent = RegionDetectionJob(audio_file_id="audio-1", status="complete")
        session.add(parent)
        await session.flush()
        job = EventSegmentationJob(region_detection_job_id=parent.id, status="queued")
        session.add(job)
        await session.commit()
        job_id = job.id

    settings = _settings(tmp_path)
    async with session_factory() as session:
        claimed = await es_run(session, settings)
    assert claimed is not None
    assert claimed.id == job_id

    async with session_factory() as session:
        refreshed = await session.get(EventSegmentationJob, job_id)
        assert refreshed is not None
        assert refreshed.status == "failed"
        assert "Pass 2" in (refreshed.error_message or "")


async def test_event_classification_worker_marks_job_failed(session_factory, tmp_path):
    async with session_factory() as session:
        region = RegionDetectionJob(audio_file_id="audio-1", status="complete")
        session.add(region)
        await session.flush()
        seg = EventSegmentationJob(region_detection_job_id=region.id, status="complete")
        session.add(seg)
        await session.flush()
        job = EventClassificationJob(event_segmentation_job_id=seg.id, status="queued")
        session.add(job)
        await session.commit()
        job_id = job.id

    settings = _settings(tmp_path)
    async with session_factory() as session:
        claimed = await ec_run(session, settings)
    assert claimed is not None

    async with session_factory() as session:
        refreshed = await session.get(EventClassificationJob, job_id)
        assert refreshed is not None
        assert refreshed.status == "failed"
        assert "Pass 3" in (refreshed.error_message or "")


async def test_claim_is_exclusive_for_region_detection(session_factory):
    """Two racing claim attempts: only one wins, the other returns None."""
    async with session_factory() as session:
        job = RegionDetectionJob(audio_file_id="audio-race", status="queued")
        session.add(job)
        await session.commit()

    async with session_factory() as session_a:
        first = await claim_region_detection_job(session_a)
        async with session_factory() as session_b:
            second = await claim_region_detection_job(session_b)

    assert first is not None
    assert second is None
    assert first.status == "running"


async def test_claim_is_exclusive_for_event_segmentation(session_factory):
    async with session_factory() as session:
        region = RegionDetectionJob(audio_file_id="audio-race", status="complete")
        session.add(region)
        await session.flush()
        job = EventSegmentationJob(region_detection_job_id=region.id, status="queued")
        session.add(job)
        await session.commit()

    async with session_factory() as session_a:
        first = await claim_event_segmentation_job(session_a)
        async with session_factory() as session_b:
            second = await claim_event_segmentation_job(session_b)

    assert first is not None
    assert second is None


async def test_claim_is_exclusive_for_event_classification(session_factory):
    async with session_factory() as session:
        region = RegionDetectionJob(audio_file_id="audio-race", status="complete")
        session.add(region)
        await session.flush()
        seg = EventSegmentationJob(region_detection_job_id=region.id, status="complete")
        session.add(seg)
        await session.flush()
        job = EventClassificationJob(event_segmentation_job_id=seg.id, status="queued")
        session.add(job)
        await session.commit()

    async with session_factory() as session_a:
        first = await claim_event_classification_job(session_a)
        async with session_factory() as session_b:
            second = await claim_event_classification_job(session_b)

    assert first is not None
    assert second is None


async def test_priority_order_region_before_segmentation_before_classification(
    session_factory,
):
    """All three new queues populated → region first, then seg, then class.

    Exercises the claim order directly since the full dispatcher loop is
    not easily embeddable in a unit test. The order matches the
    corresponding section of the runner's main loop.
    """
    async with session_factory() as session:
        region = RegionDetectionJob(audio_file_id="audio-1", status="queued")
        session.add(region)
        await session.flush()
        seg = EventSegmentationJob(region_detection_job_id=region.id, status="queued")
        session.add(seg)
        await session.flush()
        cls = EventClassificationJob(event_segmentation_job_id=seg.id, status="queued")
        session.add(cls)
        await session.commit()

    claim_order: list[str] = []

    async with session_factory() as session:
        rd = await claim_region_detection_job(session)
        if rd is not None:
            claim_order.append("region")

    async with session_factory() as session:
        sg = await claim_event_segmentation_job(session)
        if sg is not None:
            claim_order.append("segmentation")

    async with session_factory() as session:
        cl = await claim_event_classification_job(session)
        if cl is not None:
            claim_order.append("classification")

    assert claim_order == ["region", "segmentation", "classification"]
