"""Integration tests for detection embedding status and generation endpoints."""

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from humpback.database import create_engine, create_session_factory
from humpback.models.classifier import ClassifierModel, DetectionJob
from humpback.storage import detection_dir, detection_embeddings_path


async def _create_detection_job(sf, settings, *, with_embeddings: bool = False):
    """Create a test detection job (complete status)."""
    async with sf() as session:
        cm = ClassifierModel(
            name="test-model",
            model_path="/tmp/test.joblib",
            model_version="test_v1",
            vector_dim=1280,
            window_size_seconds=5.0,
            target_sample_rate=32000,
        )
        session.add(cm)
        await session.flush()

        dj = DetectionJob(
            classifier_model_id=cm.id,
            audio_folder="/tmp/test-audio",
            confidence_threshold=0.5,
            hop_seconds=1.0,
            high_threshold=0.7,
            low_threshold=0.45,
            detection_mode="windowed",
            status="complete",
            result_summary=json.dumps({"n_total_windows": 42, "n_detections": 10}),
        )
        session.add(dj)
        await session.commit()
        job_id = dj.id

    if with_embeddings:
        ddir = detection_dir(settings.storage_root, job_id)
        ddir.mkdir(parents=True, exist_ok=True)
        emb_path = detection_embeddings_path(settings.storage_root, job_id)
        table = pa.table(
            {
                "filename": ["test.wav", "test.wav"],
                "start_sec": pa.array([0.0, 5.0], type=pa.float32()),
                "end_sec": pa.array([5.0, 10.0], type=pa.float32()),
                "embedding": pa.array(
                    [[0.1] * 1280, [0.2] * 1280],
                    type=pa.list_(pa.float32(), 1280),
                ),
            }
        )
        pq.write_table(table, str(emb_path))

    return job_id


@pytest.mark.asyncio
async def test_embedding_status_no_embeddings(client, app_settings):
    """Embedding status returns has_embeddings=False when no file exists."""
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)

    job_id = await _create_detection_job(sf, app_settings)

    resp = await client.get(f"/classifier/detection-jobs/{job_id}/embedding-status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["has_embeddings"] is False
    assert data["count"] is None


@pytest.mark.asyncio
async def test_embedding_status_with_embeddings(client, app_settings):
    """Embedding status returns has_embeddings=True with count."""
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)

    job_id = await _create_detection_job(sf, app_settings, with_embeddings=True)

    resp = await client.get(f"/classifier/detection-jobs/{job_id}/embedding-status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["has_embeddings"] is True
    assert data["count"] == 2


@pytest.mark.asyncio
async def test_embedding_status_not_found(client):
    """Embedding status returns 404 for nonexistent job."""
    resp = await client.get("/classifier/detection-jobs/nonexistent/embedding-status")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_generate_embeddings_queues_job(client, app_settings):
    """POST generate-embeddings creates a DetectionEmbeddingJob."""
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)

    job_id = await _create_detection_job(sf, app_settings)

    resp = await client.post(f"/classifier/detection-jobs/{job_id}/generate-embeddings")
    assert resp.status_code == 202
    data = resp.json()
    assert data["status"] == "queued"
    assert data["detection_job_id"] == job_id


@pytest.mark.asyncio
async def test_generate_embeddings_conflict_exists(client, app_settings):
    """POST generate-embeddings returns 409 when embeddings already exist."""
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)

    job_id = await _create_detection_job(sf, app_settings, with_embeddings=True)

    resp = await client.post(f"/classifier/detection-jobs/{job_id}/generate-embeddings")
    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_generate_embeddings_conflict_in_progress(client, app_settings):
    """POST generate-embeddings returns 409 when generation already queued."""
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)

    job_id = await _create_detection_job(sf, app_settings)

    # First request succeeds
    resp1 = await client.post(
        f"/classifier/detection-jobs/{job_id}/generate-embeddings"
    )
    assert resp1.status_code == 202

    # Second request is 409
    resp2 = await client.post(
        f"/classifier/detection-jobs/{job_id}/generate-embeddings"
    )
    assert resp2.status_code == 409


@pytest.mark.asyncio
async def test_embedding_generation_status(client, app_settings):
    """GET embedding-generation-status returns queued job."""
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)

    job_id = await _create_detection_job(sf, app_settings)

    # Queue generation
    resp1 = await client.post(
        f"/classifier/detection-jobs/{job_id}/generate-embeddings"
    )
    assert resp1.status_code == 202

    # Check status
    resp2 = await client.get(
        f"/classifier/detection-jobs/{job_id}/embedding-generation-status"
    )
    assert resp2.status_code == 200
    data = resp2.json()
    assert data["status"] == "queued"
    assert data["detection_job_id"] == job_id
