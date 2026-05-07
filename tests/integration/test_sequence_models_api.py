"""Integration tests for retained Sequence Models / Continuous Embedding APIs."""

from humpback.database import create_engine, create_session_factory
from humpback.models.call_parsing import EventSegmentationJob, RegionDetectionJob
from humpback.models.processing import JobStatus
from humpback.models.sequence_models import ContinuousEmbeddingJob


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


async def _seed_crnn_event_encoder_source(app_settings):
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
            status=JobStatus.complete.value,
            region_detection_job_id=region_job.id,
        )
        session.add(seg_job)
        await session.flush()
        continuous = ContinuousEmbeddingJob(
            status=JobStatus.complete.value,
            event_segmentation_job_id=seg_job.id,
            event_source_mode="raw",
            region_detection_job_id=region_job.id,
            model_version="crnn-call-parsing-pytorch",
            target_sample_rate=16000,
            encoding_signature=f"cej-sig-{seg_job.id}",
            crnn_checkpoint_sha256="abc123",
            chunk_size_seconds=0.25,
            chunk_hop_seconds=0.25,
            projection_kind="identity",
            projection_dim=1024,
            total_regions=1,
            total_chunks=10,
        )
        session.add(continuous)
        await session.commit()
        await session.refresh(seg_job)
        await session.refresh(continuous)
        return seg_job.id, continuous.id


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


async def test_create_event_encoder_returns_201_then_200(client, app_settings):
    seg_job_id, continuous_id = await _seed_crnn_event_encoder_source(app_settings)
    payload = {
        "event_segmentation_job_id": seg_job_id,
        "continuous_embedding_job_id": continuous_id,
        "k_values": [50],
    }

    first = await client.post("/sequence-models/event-encoders", json=payload)
    assert first.status_code == 201, first.text
    body = first.json()
    assert body["status"] == JobStatus.queued.value
    assert body["event_segmentation_job_id"] == seg_job_id
    assert body["continuous_embedding_job_id"] == continuous_id

    second = await client.post("/sequence-models/event-encoders", json=payload)
    assert second.status_code == 200, second.text
    assert second.json()["id"] == body["id"]


async def test_list_event_encoders_filters_by_status(client, app_settings):
    seg_job_id, continuous_id = await _seed_crnn_event_encoder_source(app_settings)
    await client.post(
        "/sequence-models/event-encoders",
        json={
            "event_segmentation_job_id": seg_job_id,
            "continuous_embedding_job_id": continuous_id,
            "k_values": [50],
        },
    )

    listed = await client.get(
        "/sequence-models/event-encoders",
        params={"status": JobStatus.queued.value},
    )
    assert listed.status_code == 200
    assert len(listed.json()) == 1

    complete = await client.get(
        "/sequence-models/event-encoders",
        params={"status": JobStatus.complete.value},
    )
    assert complete.status_code == 200
    assert complete.json() == []


async def test_get_event_encoder_detail_without_sidecars(client, app_settings):
    seg_job_id, continuous_id = await _seed_crnn_event_encoder_source(app_settings)
    created = await client.post(
        "/sequence-models/event-encoders",
        json={
            "event_segmentation_job_id": seg_job_id,
            "continuous_embedding_job_id": continuous_id,
            "k_values": [50],
        },
    )
    job_id = created.json()["id"]

    detail = await client.get(f"/sequence-models/event-encoders/{job_id}")
    assert detail.status_code == 200
    body = detail.json()
    assert body["job"]["id"] == job_id
    assert body["manifest"] is None
    assert body["report"] is None

    missing = await client.get("/sequence-models/event-encoders/missing-id")
    assert missing.status_code == 404


async def test_cancel_event_encoder(client, app_settings):
    seg_job_id, continuous_id = await _seed_crnn_event_encoder_source(app_settings)
    created = await client.post(
        "/sequence-models/event-encoders",
        json={
            "event_segmentation_job_id": seg_job_id,
            "continuous_embedding_job_id": continuous_id,
            "k_values": [50],
        },
    )
    job_id = created.json()["id"]

    canceled = await client.post(f"/sequence-models/event-encoders/{job_id}/cancel")
    assert canceled.status_code == 200
    assert canceled.json()["status"] == JobStatus.canceled.value

    missing = await client.post("/sequence-models/event-encoders/missing-id/cancel")
    assert missing.status_code == 404


async def test_cancel_terminal_event_encoder_returns_409(client, app_settings):
    seg_job_id, continuous_id = await _seed_crnn_event_encoder_source(app_settings)
    created = await client.post(
        "/sequence-models/event-encoders",
        json={
            "event_segmentation_job_id": seg_job_id,
            "continuous_embedding_job_id": continuous_id,
            "k_values": [50],
        },
    )
    job_id = created.json()["id"]

    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)
    async with sf() as session:
        from humpback.models.sequence_models import EventEncoderJob

        job = await session.get(EventEncoderJob, job_id)
        assert job is not None
        job.status = JobStatus.complete.value
        await session.commit()

    response = await client.post(f"/sequence-models/event-encoders/{job_id}/cancel")
    assert response.status_code == 409


async def test_create_event_encoder_rejects_invalid_source(client, app_settings):
    seg_job_id = await _seed_segmentation_job(app_settings, JobStatus.complete.value)
    response = await client.post(
        "/sequence-models/event-encoders",
        json={
            "event_segmentation_job_id": seg_job_id,
            "continuous_embedding_job_id": "missing",
            "k_values": [50],
        },
    )
    assert response.status_code == 422


async def test_delete_event_encoder(client, app_settings):
    seg_job_id, continuous_id = await _seed_crnn_event_encoder_source(app_settings)
    created = await client.post(
        "/sequence-models/event-encoders",
        json={
            "event_segmentation_job_id": seg_job_id,
            "continuous_embedding_job_id": continuous_id,
            "k_values": [50],
        },
    )
    job_id = created.json()["id"]

    deleted = await client.delete(f"/sequence-models/event-encoders/{job_id}")
    assert deleted.status_code == 204

    missing = await client.delete("/sequence-models/event-encoders/missing-id")
    assert missing.status_code == 404
