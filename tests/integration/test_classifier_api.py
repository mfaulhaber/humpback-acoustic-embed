"""Integration tests for classifier API endpoints."""

import struct
import wave
from pathlib import Path


def _make_wav_dir(tmp_path: Path, name: str = "negatives") -> Path:
    """Create a directory with a small WAV file."""
    import math

    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    path = d / "noise.wav"
    sr = 16000
    n = int(sr * 2.0)
    samples = [int(32767 * math.sin(2 * math.pi * 440 * i / sr)) for i in range(n)]
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(struct.pack(f"<{n}h", *samples))
    return d


async def test_create_training_job_missing_embedding_sets(client):
    """400 when embedding sets don't exist."""
    resp = await client.post(
        "/classifier/training-jobs",
        json={
            "name": "test",
            "positive_embedding_set_ids": ["nonexistent"],
            "negative_audio_folder": "/tmp/fake",
        },
    )
    assert resp.status_code == 400


async def test_create_training_job_bad_folder(client, wav_bytes, app_settings):
    """400 when negative audio folder doesn't exist."""
    # First create an audio + embedding set via processing
    upload = await client.post(
        "/audio/upload",
        files={"file": ("test.wav", wav_bytes, "audio/wav")},
    )
    audio_id = upload.json()["id"]

    # Create and run processing job
    pjob = await client.post(
        "/processing/jobs",
        json={"audio_file_id": audio_id},
    )
    assert pjob.status_code == 201

    # Run worker
    from humpback.database import Base, create_engine, create_session_factory
    from humpback.workers.processing_worker import run_processing_job
    from humpback.workers.queue import claim_processing_job

    engine = create_engine(app_settings.database_url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = create_session_factory(engine)
    async with factory() as session:
        job = await claim_processing_job(session)
        if job:
            await run_processing_job(session, job, app_settings)

    # Get embedding set
    es_resp = await client.get("/processing/embedding-sets")
    es_list = es_resp.json()
    if not es_list:
        await engine.dispose()
        return  # skip if no embedding set produced

    es_id = es_list[0]["id"]

    resp = await client.post(
        "/classifier/training-jobs",
        json={
            "name": "test",
            "positive_embedding_set_ids": [es_id],
            "negative_audio_folder": "/nonexistent/path/12345",
        },
    )
    assert resp.status_code == 400
    await engine.dispose()


async def test_list_training_jobs_empty(client):
    resp = await client.get("/classifier/training-jobs")
    assert resp.status_code == 200
    assert resp.json() == []


async def test_get_training_job_not_found(client):
    resp = await client.get("/classifier/training-jobs/nonexistent")
    assert resp.status_code == 404


async def test_list_models_empty(client):
    resp = await client.get("/classifier/models")
    assert resp.status_code == 200
    assert resp.json() == []


async def test_get_model_not_found(client):
    resp = await client.get("/classifier/models/nonexistent")
    assert resp.status_code == 404


async def test_delete_model_not_found(client):
    resp = await client.delete("/classifier/models/nonexistent")
    assert resp.status_code == 404


async def test_list_detection_jobs_empty(client):
    resp = await client.get("/classifier/detection-jobs")
    assert resp.status_code == 200
    assert resp.json() == []


async def test_create_detection_job_bad_model(client):
    resp = await client.post(
        "/classifier/detection-jobs",
        json={
            "classifier_model_id": "nonexistent",
            "audio_folder": "/tmp",
        },
    )
    assert resp.status_code == 400


async def test_get_detection_job_not_found(client):
    resp = await client.get("/classifier/detection-jobs/nonexistent")
    assert resp.status_code == 404


async def test_download_detection_not_found(client):
    resp = await client.get("/classifier/detection-jobs/nonexistent/download")
    assert resp.status_code == 404
