"""Integration tests for the Phase 0 call parsing API router."""

from __future__ import annotations

import pytest
from httpx import AsyncClient

BASE = "/call-parsing"


@pytest.mark.asyncio
async def test_create_run_creates_parent_and_pass1_job(client: AsyncClient) -> None:
    resp = await client.post(
        f"{BASE}/runs",
        json={"audio_source_id": "audio-1", "config_snapshot": '{"hop":1.0}'},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["audio_source_id"] == "audio-1"
    assert data["status"] == "queued"
    assert data["region_detection_job"] is not None
    assert data["region_detection_job"]["status"] == "queued"
    assert data["region_detection_job"]["parent_run_id"] == data["id"]
    assert data["event_segmentation_job"] is None
    assert data["event_classification_job"] is None


@pytest.mark.asyncio
async def test_get_run_returns_nested_pass_status(client: AsyncClient) -> None:
    create = await client.post(f"{BASE}/runs", json={"audio_source_id": "audio-nested"})
    run_id = create.json()["id"]

    resp = await client.get(f"{BASE}/runs/{run_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == run_id
    assert data["region_detection_job"]["status"] == "queued"
    assert data["event_segmentation_job"] is None
    assert data["event_classification_job"] is None


@pytest.mark.asyncio
async def test_list_runs_returns_empty_then_populated(client: AsyncClient) -> None:
    empty = await client.get(f"{BASE}/runs")
    assert empty.status_code == 200
    assert empty.json() == []

    await client.post(f"{BASE}/runs", json={"audio_source_id": "audio-a"})
    await client.post(f"{BASE}/runs", json={"audio_source_id": "audio-b"})

    resp = await client.get(f"{BASE}/runs")
    assert resp.status_code == 200
    assert len(resp.json()) == 2


@pytest.mark.asyncio
async def test_delete_run_cascades_children(client: AsyncClient) -> None:
    create = await client.post(f"{BASE}/runs", json={"audio_source_id": "audio-del"})
    run_id = create.json()["id"]
    region_id = create.json()["region_detection_job"]["id"]

    del_resp = await client.delete(f"{BASE}/runs/{run_id}")
    assert del_resp.status_code == 204

    missing_run = await client.get(f"{BASE}/runs/{run_id}")
    assert missing_run.status_code == 404

    missing_region = await client.get(f"{BASE}/region-jobs/{region_id}")
    assert missing_region.status_code == 404


@pytest.mark.asyncio
async def test_get_missing_run_returns_404(client: AsyncClient) -> None:
    resp = await client.get(f"{BASE}/runs/nope")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_list_region_jobs_empty(client: AsyncClient) -> None:
    resp = await client.get(f"{BASE}/region-jobs")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_get_missing_region_job_returns_404(client: AsyncClient) -> None:
    resp = await client.get(f"{BASE}/region-jobs/not-a-real-id")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_list_region_jobs_reflects_parent_run_creation(
    client: AsyncClient,
) -> None:
    await client.post(f"{BASE}/runs", json={"audio_source_id": "audio-list"})
    resp = await client.get(f"{BASE}/region-jobs")
    assert resp.status_code == 200
    assert len(resp.json()) == 1


@pytest.mark.asyncio
async def test_post_region_jobs_returns_501_naming_pass1(client: AsyncClient) -> None:
    resp = await client.post(f"{BASE}/region-jobs", json={})
    assert resp.status_code == 501
    assert "Pass 1" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_post_segmentation_jobs_returns_501_naming_pass2(
    client: AsyncClient,
) -> None:
    resp = await client.post(f"{BASE}/segmentation-jobs", json={})
    assert resp.status_code == 501
    assert "Pass 2" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_post_classification_jobs_returns_501_naming_pass3(
    client: AsyncClient,
) -> None:
    resp = await client.post(f"{BASE}/classification-jobs", json={})
    assert resp.status_code == 501
    assert "Pass 3" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_sequence_endpoint_returns_501_naming_pass4(
    client: AsyncClient,
) -> None:
    create = await client.post(f"{BASE}/runs", json={"audio_source_id": "audio-seq"})
    run_id = create.json()["id"]
    resp = await client.get(f"{BASE}/runs/{run_id}/sequence")
    assert resp.status_code == 501
    assert "Pass 4" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_region_trace_endpoint_is_501(client: AsyncClient) -> None:
    resp = await client.get(f"{BASE}/region-jobs/anything/trace")
    assert resp.status_code == 501
    assert "Pass 1" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_segmentation_events_endpoint_is_501(client: AsyncClient) -> None:
    resp = await client.get(f"{BASE}/segmentation-jobs/anything/events")
    assert resp.status_code == 501
    assert "Pass 2" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_classification_typed_events_endpoint_is_501(
    client: AsyncClient,
) -> None:
    resp = await client.get(f"{BASE}/classification-jobs/anything/typed-events")
    assert resp.status_code == 501
    assert "Pass 3" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_segmentation_and_classification_lists_empty(
    client: AsyncClient,
) -> None:
    s = await client.get(f"{BASE}/segmentation-jobs")
    c = await client.get(f"{BASE}/classification-jobs")
    assert s.status_code == 200 and s.json() == []
    assert c.status_code == 200 and c.json() == []
