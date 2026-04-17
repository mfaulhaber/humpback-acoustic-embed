"""Tests for ``GET /classifier/detection-embedding-jobs``."""

from __future__ import annotations

import pytest

from humpback.database import create_engine, create_session_factory
from humpback.models.detection_embedding_job import DetectionEmbeddingJob


async def _insert_job(sf, **kwargs) -> str:
    async with sf() as session:
        job = DetectionEmbeddingJob(**kwargs)
        session.add(job)
        await session.commit()
        return job.id


@pytest.mark.asyncio
async def test_list_mixed_existing_and_missing_pairs(client, app_settings):
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)

    await _insert_job(
        sf,
        detection_job_id="det-a",
        model_version="perch_v2",
        status="running",
        rows_processed=3,
        rows_total=10,
    )
    await _insert_job(
        sf,
        detection_job_id="det-b",
        model_version="perch_v2",
        status="complete",
        rows_processed=5,
        rows_total=5,
    )

    resp = await client.get(
        "/classifier/detection-embedding-jobs",
        params=[
            ("detection_job_ids", "det-a"),
            ("detection_job_ids", "det-b"),
            ("detection_job_ids", "det-c"),
            ("model_version", "perch_v2"),
        ],
    )
    assert resp.status_code == 200
    data = resp.json()
    assert [d["detection_job_id"] for d in data] == ["det-a", "det-b", "det-c"]

    by_id = {d["detection_job_id"]: d for d in data}
    assert by_id["det-a"]["status"] == "running"
    assert by_id["det-a"]["rows_processed"] == 3
    assert by_id["det-a"]["rows_total"] == 10
    assert by_id["det-a"]["model_version"] == "perch_v2"

    assert by_id["det-b"]["status"] == "complete"

    assert by_id["det-c"]["status"] == "not_started"
    assert by_id["det-c"]["rows_processed"] == 0
    assert by_id["det-c"]["rows_total"] is None
    assert by_id["det-c"]["model_version"] == "perch_v2"


@pytest.mark.asyncio
async def test_list_rejects_empty_detection_job_ids(client):
    resp = await client.get(
        "/classifier/detection-embedding-jobs",
        params={"model_version": "perch_v2"},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_list_rejects_missing_model_version(client):
    resp = await client.get(
        "/classifier/detection-embedding-jobs",
        params=[("detection_job_ids", "det-a")],
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_list_filters_by_model_version(client, app_settings):
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)

    await _insert_job(
        sf,
        detection_job_id="det-a",
        model_version="perch_v2",
        status="complete",
    )
    await _insert_job(
        sf,
        detection_job_id="det-a",
        model_version="tf2_other",
        status="failed",
        error_message="nope",
    )

    resp = await client.get(
        "/classifier/detection-embedding-jobs",
        params=[("detection_job_ids", "det-a"), ("model_version", "tf2_other")],
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["status"] == "failed"
    assert data[0]["error_message"] == "nope"
    assert data[0]["model_version"] == "tf2_other"
