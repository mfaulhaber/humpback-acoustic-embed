"""Integration tests for retained Sequence Models / Continuous Embedding APIs."""

from humpback.database import create_engine, create_session_factory
from humpback.models.call_parsing import EventSegmentationJob, RegionDetectionJob
from humpback.models.processing import JobStatus


async def _seed_segmentation_job(app_settings, status: str) -> str:
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)
    async with sf() as session:
        region_job = RegionDetectionJob(
            status=JobStatus.complete.value,
            hydrophone_id="rpi_orcasound_lab",
            start_timestamp=1000.0,
            end_timestamp=1600.0,
        )
        session.add(region_job)
        await session.flush()
        seg_job = EventSegmentationJob(
            status=status,
            region_detection_job_id=region_job.id,
        )
        session.add(seg_job)
        await session.commit()
        await session.refresh(seg_job)
        return seg_job.id


async def test_create_returns_201_then_200_on_idempotent_resubmit(client, app_settings):
    seg_job_id = await _seed_segmentation_job(app_settings, JobStatus.complete.value)
    payload = {"event_segmentation_job_id": seg_job_id}

    first = await client.post("/sequence-models/continuous-embeddings", json=payload)
    assert first.status_code == 201, first.text
    body = first.json()
    job_id = body["id"]
    assert body["status"] == JobStatus.queued.value
    assert body["event_segmentation_job_id"] == seg_job_id

    second = await client.post("/sequence-models/continuous-embeddings", json=payload)
    assert second.status_code == 200, second.text
    assert second.json()["id"] == job_id


async def test_list_filters_by_status(client, app_settings):
    seg_job_id = await _seed_segmentation_job(app_settings, JobStatus.complete.value)
    await client.post(
        "/sequence-models/continuous-embeddings",
        json={"event_segmentation_job_id": seg_job_id},
    )

    listed = await client.get(
        "/sequence-models/continuous-embeddings",
        params={"status": JobStatus.queued.value},
    )
    assert listed.status_code == 200
    items = listed.json()
    assert len(items) == 1
    assert items[0]["status"] == JobStatus.queued.value

    listed_complete = await client.get(
        "/sequence-models/continuous-embeddings",
        params={"status": JobStatus.complete.value},
    )
    assert listed_complete.status_code == 200
    assert listed_complete.json() == []


async def test_get_detail_returns_404_on_missing(client):
    response = await client.get("/sequence-models/continuous-embeddings/missing-id")
    assert response.status_code == 404


async def test_get_detail_returns_job_without_manifest(client, app_settings):
    seg_job_id = await _seed_segmentation_job(app_settings, JobStatus.complete.value)
    create_resp = await client.post(
        "/sequence-models/continuous-embeddings",
        json={"event_segmentation_job_id": seg_job_id},
    )
    job_id = create_resp.json()["id"]

    detail = await client.get(f"/sequence-models/continuous-embeddings/{job_id}")
    assert detail.status_code == 200
    body = detail.json()
    assert body["job"]["id"] == job_id
    assert body["manifest"] is None


async def test_cancel_queued_returns_canceled(client, app_settings):
    seg_job_id = await _seed_segmentation_job(app_settings, JobStatus.complete.value)
    create_resp = await client.post(
        "/sequence-models/continuous-embeddings",
        json={"event_segmentation_job_id": seg_job_id},
    )
    job_id = create_resp.json()["id"]

    cancel_resp = await client.post(
        f"/sequence-models/continuous-embeddings/{job_id}/cancel"
    )
    assert cancel_resp.status_code == 200
    assert cancel_resp.json()["status"] == JobStatus.canceled.value


async def test_cancel_returns_404_on_missing(client):
    response = await client.post(
        "/sequence-models/continuous-embeddings/missing-id/cancel"
    )
    assert response.status_code == 404


async def test_cancel_terminal_returns_409(client, app_settings):
    seg_job_id = await _seed_segmentation_job(app_settings, JobStatus.complete.value)
    create_resp = await client.post(
        "/sequence-models/continuous-embeddings",
        json={"event_segmentation_job_id": seg_job_id},
    )
    job_id = create_resp.json()["id"]

    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)
    async with sf() as session:
        from humpback.models.sequence_models import ContinuousEmbeddingJob

        job = await session.get(ContinuousEmbeddingJob, job_id)
        assert job is not None
        job.status = JobStatus.complete.value
        await session.commit()

    cancel_resp = await client.post(
        f"/sequence-models/continuous-embeddings/{job_id}/cancel"
    )
    assert cancel_resp.status_code == 409


async def test_create_rejects_invalid_hop(client, app_settings):
    seg_job_id = await _seed_segmentation_job(app_settings, JobStatus.complete.value)
    response = await client.post(
        "/sequence-models/continuous-embeddings",
        json={"event_segmentation_job_id": seg_job_id, "hop_seconds": 0},
    )
    assert response.status_code == 422


async def test_create_rejects_missing_segmentation_job(client):
    response = await client.post(
        "/sequence-models/continuous-embeddings",
        json={"event_segmentation_job_id": "nonexistent"},
    )
    assert response.status_code == 422


async def test_create_rejects_non_complete_segmentation_job(client, app_settings):
    seg_job_id = await _seed_segmentation_job(app_settings, JobStatus.running.value)
    response = await client.post(
        "/sequence-models/continuous-embeddings",
        json={"event_segmentation_job_id": seg_job_id},
    )
    assert response.status_code == 422


async def test_retired_sequence_model_routes_are_unregistered(client):
    retired_paths = [
        "/sequence-models/hmm-sequences",
        "/sequence-models/masked-transformers",
        "/sequence-models/motif-extractions",
    ]

    for path in retired_paths:
        response = await client.get(path)
        assert response.status_code == 404
