"""Integration tests for the Sequence Models API router (PR 1)."""

from humpback.database import create_engine, create_session_factory
from humpback.models.call_parsing import RegionDetectionJob
from humpback.models.processing import JobStatus


async def _seed_region_detection_job(app_settings, status: str) -> str:
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)
    async with sf() as session:
        job = RegionDetectionJob(
            status=status,
            hydrophone_id="rpi_orcasound_lab",
            start_timestamp=0.0,
            end_timestamp=600.0,
        )
        session.add(job)
        await session.commit()
        await session.refresh(job)
        return job.id


async def test_create_returns_201_then_200_on_idempotent_resubmit(client, app_settings):
    region_job_id = await _seed_region_detection_job(
        app_settings, JobStatus.complete.value
    )
    payload = {"region_detection_job_id": region_job_id}

    first = await client.post("/sequence-models/continuous-embeddings", json=payload)
    assert first.status_code == 201, first.text
    body = first.json()
    job_id = body["id"]
    assert body["status"] == JobStatus.queued.value
    assert body["region_detection_job_id"] == region_job_id

    second = await client.post("/sequence-models/continuous-embeddings", json=payload)
    assert second.status_code == 200, second.text
    assert second.json()["id"] == job_id


async def test_list_filters_by_status(client, app_settings):
    region_job_id = await _seed_region_detection_job(
        app_settings, JobStatus.complete.value
    )
    await client.post(
        "/sequence-models/continuous-embeddings",
        json={"region_detection_job_id": region_job_id},
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
    region_job_id = await _seed_region_detection_job(
        app_settings, JobStatus.complete.value
    )
    create_resp = await client.post(
        "/sequence-models/continuous-embeddings",
        json={"region_detection_job_id": region_job_id},
    )
    job_id = create_resp.json()["id"]

    detail = await client.get(f"/sequence-models/continuous-embeddings/{job_id}")
    assert detail.status_code == 200
    body = detail.json()
    assert body["job"]["id"] == job_id
    assert body["manifest"] is None


async def test_cancel_queued_returns_canceled(client, app_settings):
    region_job_id = await _seed_region_detection_job(
        app_settings, JobStatus.complete.value
    )
    create_resp = await client.post(
        "/sequence-models/continuous-embeddings",
        json={"region_detection_job_id": region_job_id},
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
    region_job_id = await _seed_region_detection_job(
        app_settings, JobStatus.complete.value
    )
    create_resp = await client.post(
        "/sequence-models/continuous-embeddings",
        json={"region_detection_job_id": region_job_id},
    )
    job_id = create_resp.json()["id"]

    # Manually flip to complete via DB so the cancel produces 409.
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
    region_job_id = await _seed_region_detection_job(
        app_settings, JobStatus.complete.value
    )
    response = await client.post(
        "/sequence-models/continuous-embeddings",
        json={"region_detection_job_id": region_job_id, "hop_seconds": 0},
    )
    assert response.status_code == 422


async def test_create_rejects_unknown_region_job(client):
    response = await client.post(
        "/sequence-models/continuous-embeddings",
        json={"region_detection_job_id": "nonexistent"},
    )
    assert response.status_code == 400


async def test_create_rejects_non_hydrophone_region_job(client, app_settings):
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)
    async with sf() as session:
        job = RegionDetectionJob(status=JobStatus.complete.value)
        session.add(job)
        await session.commit()
        await session.refresh(job)
        region_job_id = job.id

    response = await client.post(
        "/sequence-models/continuous-embeddings",
        json={"region_detection_job_id": region_job_id},
    )
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# HMM Sequence Job endpoints
# ---------------------------------------------------------------------------


async def _seed_complete_continuous_embedding_job(app_settings) -> str:
    from humpback.models.sequence_models import ContinuousEmbeddingJob

    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)
    async with sf() as session:
        region_job = RegionDetectionJob(
            status=JobStatus.complete.value,
            hydrophone_id="rpi_orcasound_lab",
            start_timestamp=0.0,
            end_timestamp=600.0,
        )
        session.add(region_job)
        await session.flush()
        cej = ContinuousEmbeddingJob(
            status=JobStatus.complete.value,
            region_detection_job_id=region_job.id,
            model_version="surfperch-tensorflow2",
            window_size_seconds=5.0,
            hop_seconds=1.0,
            pad_seconds=10.0,
            target_sample_rate=32000,
            encoding_signature="test-sig-hmm",
        )
        session.add(cej)
        await session.commit()
        await session.refresh(cej)
        return cej.id


async def test_hmm_create_returns_201(client, app_settings):
    cej_id = await _seed_complete_continuous_embedding_job(app_settings)
    response = await client.post(
        "/sequence-models/hmm-sequences",
        json={"continuous_embedding_job_id": cej_id, "n_states": 4},
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["status"] == JobStatus.queued.value
    assert body["n_states"] == 4
    assert body["continuous_embedding_job_id"] == cej_id


async def test_hmm_create_rejects_invalid_source(client):
    response = await client.post(
        "/sequence-models/hmm-sequences",
        json={"continuous_embedding_job_id": "nonexistent", "n_states": 3},
    )
    assert response.status_code == 400


async def test_hmm_list_filters_by_status(client, app_settings):
    cej_id = await _seed_complete_continuous_embedding_job(app_settings)
    await client.post(
        "/sequence-models/hmm-sequences",
        json={"continuous_embedding_job_id": cej_id, "n_states": 3},
    )

    listed = await client.get(
        "/sequence-models/hmm-sequences",
        params={"status": JobStatus.queued.value},
    )
    assert listed.status_code == 200
    assert len(listed.json()) >= 1

    listed_complete = await client.get(
        "/sequence-models/hmm-sequences",
        params={"status": JobStatus.complete.value},
    )
    assert listed_complete.status_code == 200
    assert listed_complete.json() == []


async def test_hmm_get_detail_returns_404_on_missing(client):
    response = await client.get("/sequence-models/hmm-sequences/missing-id")
    assert response.status_code == 404


async def test_hmm_get_detail_returns_job_without_summary(client, app_settings):
    cej_id = await _seed_complete_continuous_embedding_job(app_settings)
    create_resp = await client.post(
        "/sequence-models/hmm-sequences",
        json={"continuous_embedding_job_id": cej_id, "n_states": 5},
    )
    job_id = create_resp.json()["id"]

    detail = await client.get(f"/sequence-models/hmm-sequences/{job_id}")
    assert detail.status_code == 200
    body = detail.json()
    assert body["job"]["id"] == job_id
    assert body["summary"] is None


async def test_hmm_detail_includes_region_detection_job_id(client, app_settings):
    from humpback.models.sequence_models import ContinuousEmbeddingJob

    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)
    async with sf() as session:
        region_job = RegionDetectionJob(
            status=JobStatus.complete.value,
            hydrophone_id="rpi_orcasound_lab",
            start_timestamp=0.0,
            end_timestamp=600.0,
        )
        session.add(region_job)
        await session.flush()
        cej = ContinuousEmbeddingJob(
            status=JobStatus.complete.value,
            region_detection_job_id=region_job.id,
            model_version="surfperch-tensorflow2",
            window_size_seconds=5.0,
            hop_seconds=1.0,
            pad_seconds=10.0,
            target_sample_rate=32000,
            encoding_signature="test-sig-hmm-region",
        )
        session.add(cej)
        await session.commit()
        await session.refresh(cej)
        cej_id = cej.id
        region_job_id = region_job.id

    create_resp = await client.post(
        "/sequence-models/hmm-sequences",
        json={"continuous_embedding_job_id": cej_id, "n_states": 4},
    )
    job_id = create_resp.json()["id"]

    detail = await client.get(f"/sequence-models/hmm-sequences/{job_id}")
    assert detail.status_code == 200
    body = detail.json()
    assert body["region_detection_job_id"] == region_job_id


async def test_hmm_cancel_queued_returns_canceled(client, app_settings):
    cej_id = await _seed_complete_continuous_embedding_job(app_settings)
    create_resp = await client.post(
        "/sequence-models/hmm-sequences",
        json={"continuous_embedding_job_id": cej_id, "n_states": 3},
    )
    job_id = create_resp.json()["id"]

    cancel_resp = await client.post(f"/sequence-models/hmm-sequences/{job_id}/cancel")
    assert cancel_resp.status_code == 200
    assert cancel_resp.json()["status"] == JobStatus.canceled.value


async def test_hmm_cancel_returns_404_on_missing(client):
    response = await client.post("/sequence-models/hmm-sequences/missing-id/cancel")
    assert response.status_code == 404


async def test_hmm_cancel_terminal_returns_409(client, app_settings):
    from humpback.models.sequence_models import HMMSequenceJob

    cej_id = await _seed_complete_continuous_embedding_job(app_settings)
    create_resp = await client.post(
        "/sequence-models/hmm-sequences",
        json={"continuous_embedding_job_id": cej_id, "n_states": 3},
    )
    job_id = create_resp.json()["id"]

    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)
    async with sf() as session:
        job = await session.get(HMMSequenceJob, job_id)
        assert job is not None
        job.status = JobStatus.complete.value
        await session.commit()

    cancel_resp = await client.post(f"/sequence-models/hmm-sequences/{job_id}/cancel")
    assert cancel_resp.status_code == 409


# ---------------------------------------------------------------------------
# Delete endpoints
# ---------------------------------------------------------------------------


async def test_delete_continuous_embedding_returns_204(client, app_settings):
    region_job_id = await _seed_region_detection_job(
        app_settings, JobStatus.complete.value
    )
    create_resp = await client.post(
        "/sequence-models/continuous-embeddings",
        json={"region_detection_job_id": region_job_id},
    )
    job_id = create_resp.json()["id"]

    delete_resp = await client.delete(
        f"/sequence-models/continuous-embeddings/{job_id}"
    )
    assert delete_resp.status_code == 204

    get_resp = await client.get(f"/sequence-models/continuous-embeddings/{job_id}")
    assert get_resp.status_code == 404


async def test_delete_continuous_embedding_missing_returns_404(client):
    response = await client.delete("/sequence-models/continuous-embeddings/nonexistent")
    assert response.status_code == 404


async def test_delete_hmm_sequence_returns_204(client, app_settings):
    cej_id = await _seed_complete_continuous_embedding_job(app_settings)
    create_resp = await client.post(
        "/sequence-models/hmm-sequences",
        json={"continuous_embedding_job_id": cej_id, "n_states": 4},
    )
    job_id = create_resp.json()["id"]

    delete_resp = await client.delete(f"/sequence-models/hmm-sequences/{job_id}")
    assert delete_resp.status_code == 204

    get_resp = await client.get(f"/sequence-models/hmm-sequences/{job_id}")
    assert get_resp.status_code == 404


async def test_delete_hmm_sequence_missing_returns_404(client):
    response = await client.delete("/sequence-models/hmm-sequences/nonexistent")
    assert response.status_code == 404
