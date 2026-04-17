"""Unit tests for ``create_reembedding_job`` idempotent enqueue contract."""

from __future__ import annotations

import pytest
from sqlalchemy import select

from humpback.models.detection_embedding_job import DetectionEmbeddingJob
from humpback.services.detection_embedding_service import (
    create_reembedding_job,
    get_reembedding_job,
    list_reembedding_jobs,
)


@pytest.mark.asyncio
async def test_create_reembedding_job_new_row(session):
    job = await create_reembedding_job(session, "det-1", "perch_v2")
    assert job.detection_job_id == "det-1"
    assert job.model_version == "perch_v2"
    assert job.status == "queued"
    assert job.mode == "full"
    assert job.rows_processed == 0
    assert job.rows_total is None


@pytest.mark.asyncio
async def test_create_reembedding_job_is_idempotent_when_queued(session):
    first = await create_reembedding_job(session, "det-1", "perch_v2")
    second = await create_reembedding_job(session, "det-1", "perch_v2")
    assert first.id == second.id

    result = await session.execute(
        select(DetectionEmbeddingJob).where(
            DetectionEmbeddingJob.detection_job_id == "det-1",
            DetectionEmbeddingJob.model_version == "perch_v2",
        )
    )
    assert len(list(result.scalars().all())) == 1


@pytest.mark.asyncio
async def test_create_reembedding_job_returns_complete_row_unchanged(session):
    first = await create_reembedding_job(session, "det-1", "perch_v2")
    first.status = "complete"
    await session.commit()

    again = await create_reembedding_job(session, "det-1", "perch_v2")
    assert again.id == first.id
    assert again.status == "complete"


@pytest.mark.asyncio
async def test_create_reembedding_job_resets_failed_row(session):
    first = await create_reembedding_job(session, "det-1", "perch_v2")
    first.status = "failed"
    first.error_message = "boom"
    first.rows_processed = 7
    first.rows_total = 10
    await session.commit()

    retried = await create_reembedding_job(session, "det-1", "perch_v2")
    assert retried.id == first.id
    assert retried.status == "queued"
    assert retried.error_message is None
    assert retried.rows_processed == 0
    assert retried.rows_total is None


@pytest.mark.asyncio
async def test_create_reembedding_job_different_model_versions_coexist(session):
    a = await create_reembedding_job(session, "det-1", "perch_v2")
    b = await create_reembedding_job(session, "det-1", "tf2_other")
    assert a.id != b.id


@pytest.mark.asyncio
async def test_get_and_list_reembedding_jobs(session):
    a = await create_reembedding_job(session, "det-1", "perch_v2")
    await create_reembedding_job(session, "det-2", "perch_v2")
    await create_reembedding_job(session, "det-3", "tf2_other")

    found = await get_reembedding_job(session, "det-1", "perch_v2")
    assert found is not None and found.id == a.id

    listed = await list_reembedding_jobs(
        session, ["det-1", "det-2", "det-3"], "perch_v2"
    )
    assert set(listed.keys()) == {"det-1", "det-2"}
