"""Integration tests for spectrogram and embedding visualization endpoints."""

import io
import math
import struct
import wave

import pytest
from httpx import ASGITransport, AsyncClient

from humpback.api.app import create_app
from humpback.config import Settings
from humpback.database import create_engine, create_session_factory
from humpback.processing.inference import FakeTFLiteModel
from humpback.workers.processing_worker import run_processing_job
from humpback.workers.queue import claim_processing_job


def _make_wav_bytes(duration: float = 2.0, sample_rate: int = 16000) -> bytes:
    n = int(sample_rate * duration)
    samples = [int(32767 * math.sin(2 * math.pi * 440 * i / sample_rate)) for i in range(n)]
    buf = io.BytesIO()
    with wave.open(buf, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n}h", *samples))
    return buf.getvalue()


@pytest.fixture
def viz_settings(tmp_path):
    db_path = tmp_path / "viz.db"
    return Settings(
        database_url=f"sqlite+aiosqlite:///{db_path}",
        storage_root=tmp_path / "storage",
        use_real_model=False,
        vector_dim=64,
        window_size_seconds=1.0,
        target_sample_rate=16000,
    )


@pytest.fixture
async def viz_client(viz_settings):
    app = create_app(viz_settings)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        await app.router.startup()
        yield ac
        await app.router.shutdown()


async def _upload_and_process(client, settings):
    """Upload a WAV and run processing to produce an embedding set."""
    wav_data = _make_wav_bytes(duration=2.0, sample_rate=16000)
    resp = await client.post(
        "/audio/upload",
        files={"file": ("test.wav", wav_data, "audio/wav")},
    )
    assert resp.status_code == 201
    audio_id = resp.json()["id"]

    resp = await client.post(
        "/processing/jobs",
        json={
            "audio_file_id": audio_id,
            "model_version": settings.model_version,
            "window_size_seconds": settings.window_size_seconds,
            "target_sample_rate": settings.target_sample_rate,
        },
    )
    assert resp.status_code == 201
    job_id = resp.json()["id"]

    engine = create_engine(settings.database_url)
    session_factory = create_session_factory(engine)
    model = FakeTFLiteModel(vector_dim=settings.vector_dim)

    async with session_factory() as session:
        claimed = await claim_processing_job(session)
        assert claimed is not None
        await run_processing_job(session, claimed, settings, model)

    await engine.dispose()

    resp = await client.get("/processing/embedding-sets")
    es_list = resp.json()
    es_id = es_list[0]["id"]
    return audio_id, es_id


async def test_spectrogram_returns_correct_shape(viz_client, viz_settings):
    """Spectrogram endpoint returns 128x128 data for window 0."""
    wav_data = _make_wav_bytes(duration=2.0, sample_rate=16000)
    resp = await viz_client.post(
        "/audio/upload",
        files={"file": ("spec.wav", wav_data, "audio/wav")},
    )
    audio_id = resp.json()["id"]

    resp = await viz_client.get(
        f"/audio/{audio_id}/spectrogram",
        params={"window_index": 0, "target_sample_rate": 16000, "window_size_seconds": 1.0},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["shape"] == [128, 128]
    assert data["window_index"] == 0
    assert data["total_windows"] == 2
    assert len(data["data"]) == 128
    assert len(data["data"][0]) == 128
    assert isinstance(data["min_db"], float)
    assert isinstance(data["max_db"], float)


async def test_spectrogram_window_out_of_range(viz_client, viz_settings):
    """Requesting a window_index beyond total returns 400."""
    wav_data = _make_wav_bytes(duration=2.0, sample_rate=16000)
    resp = await viz_client.post(
        "/audio/upload",
        files={"file": ("spec2.wav", wav_data, "audio/wav")},
    )
    audio_id = resp.json()["id"]

    resp = await viz_client.get(
        f"/audio/{audio_id}/spectrogram",
        params={"window_index": 999, "target_sample_rate": 16000, "window_size_seconds": 1.0},
    )
    assert resp.status_code == 400


async def test_spectrogram_nonexistent_audio(viz_client):
    """Spectrogram on non-existent audio returns 404."""
    resp = await viz_client.get(
        "/audio/nonexistent/spectrogram",
        params={"window_index": 0},
    )
    assert resp.status_code == 404


async def test_embeddings_returns_similarity_matrix(viz_client, viz_settings):
    """After processing, embeddings endpoint returns cosine similarity matrix."""
    audio_id, es_id = await _upload_and_process(viz_client, viz_settings)

    resp = await viz_client.get(
        f"/audio/{audio_id}/embeddings",
        params={"embedding_set_id": es_id},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["embedding_set_id"] == es_id
    assert data["vector_dim"] == 64
    n = data["num_windows"]
    assert n >= 1
    assert len(data["similarity_matrix"]) == n
    assert len(data["similarity_matrix"][0]) == n
    assert len(data["row_indices"]) == n
    # Matrix should be symmetric
    for i in range(n):
        for j in range(n):
            assert abs(data["similarity_matrix"][i][j] - data["similarity_matrix"][j][i]) < 1e-5
    # Values should be in [-1, 1]
    for row in data["similarity_matrix"]:
        for v in row:
            assert -1.0 - 1e-5 <= v <= 1.0 + 1e-5
    # Diagonal should be 1.0 (or 0.0 if centered vectors are zero)
    for i in range(n):
        diag = data["similarity_matrix"][i][i]
        assert abs(diag - 1.0) < 1e-5 or abs(diag) < 1e-5


async def test_embeddings_wrong_audio_id(viz_client, viz_settings):
    """Requesting embeddings with wrong audio_id returns 404."""
    audio_id, es_id = await _upload_and_process(viz_client, viz_settings)

    resp = await viz_client.get(
        "/audio/nonexistent/embeddings",
        params={"embedding_set_id": es_id},
    )
    assert resp.status_code == 404
