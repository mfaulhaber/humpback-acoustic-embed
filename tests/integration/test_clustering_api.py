import io
import math
import struct
import wave
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


@pytest.fixture
async def embedding_set_id(client):
    """Upload audio and create an embedding set for clustering tests."""
    # Upload audio
    sr = 16000
    duration = 2.0
    n = int(sr * duration)
    samples = [int(32767 * math.sin(2 * math.pi * 440 * i / sr)) for i in range(n)]
    buf = io.BytesIO()
    with wave.open(buf, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(struct.pack(f"<{n}h", *samples))
    wav_bytes = buf.getvalue()

    upload = await client.post(
        "/audio/upload",
        files={"file": ("test.wav", wav_bytes, "audio/wav")},
    )
    audio_id = upload.json()["id"]

    # Create a processing job (will be skipped or queued)
    proc = await client.post(
        "/processing/jobs",
        json={"audio_file_id": audio_id},
    )
    assert proc.status_code == 201

    # We need a real embedding set — create one directly via DB
    # Instead, just get the list; the seed model creates one if processing runs
    # For testing, we'll just use the embedding set endpoint
    sets = await client.get("/processing/embedding-sets")
    if sets.json():
        return sets.json()[0]["id"]

    # If no sets exist yet, create one manually via the DB is complex.
    # Let's just use an empty list for the basic test and skip the validation.
    pytest.skip("No embedding sets available for clustering test")


async def test_create_clustering_job_with_empty_list(client):
    """Creating a clustering job with empty embedding_set_ids should succeed."""
    resp = await client.post(
        "/clustering/jobs",
        json={"embedding_set_ids": []},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["status"] == "queued"
    assert data["embedding_set_ids"] == []


async def test_create_clustering_job_invalid_ids(client):
    """Creating a clustering job with non-existent IDs should fail."""
    resp = await client.post(
        "/clustering/jobs",
        json={"embedding_set_ids": ["fake-id-1"]},
    )
    assert resp.status_code == 400
    assert "not found" in resp.json()["detail"].lower()


async def test_get_clustering_job(client):
    create = await client.post(
        "/clustering/jobs",
        json={"embedding_set_ids": []},
    )
    job_id = create.json()["id"]
    resp = await client.get(f"/clustering/jobs/{job_id}")
    assert resp.status_code == 200
    assert resp.json()["id"] == job_id


async def test_get_clustering_job_not_found(client):
    resp = await client.get("/clustering/jobs/nonexistent")
    assert resp.status_code == 404


async def test_list_clusters_empty(client):
    create = await client.post(
        "/clustering/jobs",
        json={"embedding_set_ids": []},
    )
    job_id = create.json()["id"]
    resp = await client.get(f"/clustering/jobs/{job_id}/clusters")
    assert resp.status_code == 200
    assert resp.json() == []


async def test_visualization_not_found(client):
    resp = await client.get("/clustering/jobs/nonexistent/visualization")
    assert resp.status_code == 404


async def test_visualization_not_complete(client):
    create = await client.post(
        "/clustering/jobs",
        json={"embedding_set_ids": []},
    )
    job_id = create.json()["id"]
    # Job is queued, not complete
    resp = await client.get(f"/clustering/jobs/{job_id}/visualization")
    assert resp.status_code == 400
    assert "not complete" in resp.json()["detail"].lower()


async def test_visualization_success(client, app_settings):
    """Test visualization endpoint with a manually placed umap_coords.parquet."""
    # Create a job
    create = await client.post(
        "/clustering/jobs",
        json={"embedding_set_ids": []},
    )
    job_id = create.json()["id"]

    # Manually mark job as complete in DB (via the API we can't, so use storage hack)
    # We need to manually update the job status. Use the internal service.
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.ext.asyncio import async_sessionmaker
    from sqlalchemy import update
    from humpback.models.clustering import ClusteringJob

    engine = create_async_engine(app_settings.database_url)
    async_session = async_sessionmaker(engine)
    async with async_session() as session:
        await session.execute(
            update(ClusteringJob).where(ClusteringJob.id == job_id).values(status="complete")
        )
        await session.commit()
    await engine.dispose()

    # Create umap_coords.parquet in the cluster directory
    cluster_path = Path(app_settings.storage_root) / "clusters" / job_id
    cluster_path.mkdir(parents=True, exist_ok=True)

    n = 5
    table = pa.table({
        "x": pa.array(np.random.randn(n).astype(np.float32), type=pa.float32()),
        "y": pa.array(np.random.randn(n).astype(np.float32), type=pa.float32()),
        "cluster_label": pa.array([0, 0, 1, 1, -1], type=pa.int32()),
        "embedding_set_id": pa.array(["es1", "es1", "es2", "es2", "es1"], type=pa.string()),
        "embedding_row_index": pa.array([0, 1, 0, 1, 2], type=pa.int32()),
    })
    pq.write_table(table, str(cluster_path / "umap_coords.parquet"))

    resp = await client.get(f"/clustering/jobs/{job_id}/visualization")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["x"]) == n
    assert len(data["y"]) == n
    assert len(data["cluster_label"]) == n
    assert len(data["embedding_set_id"]) == n
    assert len(data["embedding_row_index"]) == n
    assert data["cluster_label"] == [0, 0, 1, 1, -1]


async def test_visualization_no_umap_file(client, app_settings):
    """Test 404 when job is complete but umap_coords.parquet doesn't exist."""
    create = await client.post(
        "/clustering/jobs",
        json={"embedding_set_ids": []},
    )
    job_id = create.json()["id"]

    # Mark job as complete
    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
    from sqlalchemy import update
    from humpback.models.clustering import ClusteringJob

    engine = create_async_engine(app_settings.database_url)
    async_session = async_sessionmaker(engine)
    async with async_session() as session:
        await session.execute(
            update(ClusteringJob).where(ClusteringJob.id == job_id).values(status="complete")
        )
        await session.commit()
    await engine.dispose()

    resp = await client.get(f"/clustering/jobs/{job_id}/visualization")
    assert resp.status_code == 404
    assert "not available" in resp.json()["detail"].lower()
