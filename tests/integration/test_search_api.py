"""Integration tests for embedding similarity search API."""

import uuid

import numpy as np
import pytest

from humpback.database import create_engine, create_session_factory
from humpback.models.audio import AudioFile
from humpback.models.classifier import ClassifierModel, DetectionJob
from humpback.models.processing import EmbeddingSet
from humpback.models.search import SearchJob
from humpback.processing.embeddings import IncrementalParquetWriter


async def _seed_embedding_sets(app_settings, tmp_path):
    """Create audio files + embedding sets with real parquet data.

    Returns dict with keys: es_query, es_other, es_different_model
    containing embedding set IDs.
    """
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)

    # Write parquet files with known vectors (dim=4)
    # Query set: 3 vectors
    query_path = tmp_path / "query.parquet"
    writer = IncrementalParquetWriter(query_path, vector_dim=4)
    writer.add(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))  # row 0: query
    writer.add(np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))  # row 1
    writer.add(np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32))  # row 2
    writer.close()

    # Other set (same model): 3 vectors — row 0 is very similar to query row 0
    other_path = tmp_path / "other.parquet"
    writer = IncrementalParquetWriter(other_path, vector_dim=4)
    writer.add(np.array([0.95, 0.05, 0.0, 0.0], dtype=np.float32))  # row 0: near query
    writer.add(np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32))  # row 1: orthogonal
    writer.add(np.array([-1.0, 0.0, 0.0, 0.0], dtype=np.float32))  # row 2: opposite
    writer.close()

    # Different model set: should be excluded by model_version filtering
    diff_model_path = tmp_path / "diff_model.parquet"
    writer = IncrementalParquetWriter(diff_model_path, vector_dim=4)
    writer.add(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    writer.close()

    async with sf() as session:
        af1 = AudioFile(
            filename="song_a.wav",
            folder_path="whales",
            checksum_sha256=f"a_{uuid.uuid4().hex[:8]}",
        )
        af2 = AudioFile(
            filename="song_b.wav",
            folder_path="whales",
            checksum_sha256=f"b_{uuid.uuid4().hex[:8]}",
        )
        af3 = AudioFile(
            filename="song_c.wav",
            folder_path="other",
            checksum_sha256=f"c_{uuid.uuid4().hex[:8]}",
        )
        session.add_all([af1, af2, af3])
        await session.flush()

        es_query = EmbeddingSet(
            audio_file_id=af1.id,
            encoding_signature="sig_v1",
            model_version="perch_v1",
            window_size_seconds=5.0,
            target_sample_rate=32000,
            vector_dim=4,
            parquet_path=str(query_path),
        )
        es_other = EmbeddingSet(
            audio_file_id=af2.id,
            encoding_signature="sig_v1",
            model_version="perch_v1",
            window_size_seconds=5.0,
            target_sample_rate=32000,
            vector_dim=4,
            parquet_path=str(other_path),
        )
        es_diff_model = EmbeddingSet(
            audio_file_id=af3.id,
            encoding_signature="sig_v2",
            model_version="perch_v2",
            window_size_seconds=5.0,
            target_sample_rate=32000,
            vector_dim=4,
            parquet_path=str(diff_model_path),
        )
        session.add_all([es_query, es_other, es_diff_model])
        await session.flush()
        ids = {
            "es_query": es_query.id,
            "es_other": es_other.id,
            "es_diff_model": es_diff_model.id,
        }
        await session.commit()
    await engine.dispose()
    return ids


@pytest.fixture
async def seeded(app_settings, tmp_path, client):
    """Seed DB with embedding sets and return IDs dict."""
    return await _seed_embedding_sets(app_settings, tmp_path)


async def test_basic_search(seeded, client):
    """Search returns ranked results with correct structure."""
    ids = seeded
    resp = await client.post(
        "/search/similar",
        json={
            "embedding_set_id": ids["es_query"],
            "row_index": 0,
            "top_k": 5,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["query_embedding_set_id"] == ids["es_query"]
    assert data["query_row_index"] == 0
    assert data["model_version"] == "perch_v1"
    assert data["metric"] == "cosine"
    assert data["total_candidates"] == 3  # es_other has 3 rows
    assert len(data["results"]) == 3
    # Best match should be es_other row 0 ([0.95, 0.05, 0, 0])
    top = data["results"][0]
    assert top["embedding_set_id"] == ids["es_other"]
    assert top["row_index"] == 0
    assert top["score"] > 0.9
    assert top["audio_filename"] == "song_b.wav"
    assert top["window_offset_seconds"] == 0.0


async def test_model_version_isolation(seeded, client):
    """Different-model sets are not included in search."""
    ids = seeded
    resp = await client.post(
        "/search/similar",
        json={
            "embedding_set_id": ids["es_query"],
            "row_index": 0,
            "top_k": 100,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    result_es_ids = {r["embedding_set_id"] for r in data["results"]}
    assert ids["es_diff_model"] not in result_es_ids


async def test_embedding_set_ids_filter(seeded, client):
    """Restrict search to specific embedding set IDs."""
    ids = seeded
    # Search only within es_other
    resp = await client.post(
        "/search/similar",
        json={
            "embedding_set_id": ids["es_query"],
            "row_index": 0,
            "embedding_set_ids": [ids["es_other"]],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    result_es_ids = {r["embedding_set_id"] for r in data["results"]}
    assert result_es_ids == {ids["es_other"]}


async def test_exclude_self(seeded, client):
    """By default, query's own embedding set is excluded."""
    ids = seeded
    resp = await client.post(
        "/search/similar",
        json={
            "embedding_set_id": ids["es_query"],
            "row_index": 0,
        },
    )
    data = resp.json()
    result_es_ids = {r["embedding_set_id"] for r in data["results"]}
    assert ids["es_query"] not in result_es_ids


async def test_include_self(seeded, client):
    """With exclude_self=False, query set is included."""
    ids = seeded
    resp = await client.post(
        "/search/similar",
        json={
            "embedding_set_id": ids["es_query"],
            "row_index": 0,
            "exclude_self": False,
        },
    )
    data = resp.json()
    result_es_ids = {r["embedding_set_id"] for r in data["results"]}
    assert ids["es_query"] in result_es_ids


async def test_unknown_embedding_set_404(client):
    """404 for nonexistent embedding set."""
    resp = await client.post(
        "/search/similar",
        json={
            "embedding_set_id": "nonexistent-id",
            "row_index": 0,
        },
    )
    assert resp.status_code == 404


async def test_out_of_range_row_index(seeded, client):
    """404 for row_index that doesn't exist in the parquet."""
    ids = seeded
    resp = await client.post(
        "/search/similar",
        json={
            "embedding_set_id": ids["es_query"],
            "row_index": 999,
        },
    )
    assert resp.status_code == 404


async def test_invalid_top_k(client):
    """422 for top_k outside valid range."""
    resp = await client.post(
        "/search/similar",
        json={
            "embedding_set_id": "any",
            "row_index": 0,
            "top_k": 0,
        },
    )
    assert resp.status_code == 422


async def test_invalid_metric(client):
    """422 for unsupported metric."""
    resp = await client.post(
        "/search/similar",
        json={
            "embedding_set_id": "any",
            "row_index": 0,
            "metric": "hamming",
        },
    )
    assert resp.status_code == 422


async def test_euclidean_search(seeded, client):
    """Euclidean search returns valid results."""
    ids = seeded
    resp = await client.post(
        "/search/similar",
        json={
            "embedding_set_id": ids["es_query"],
            "row_index": 0,
            "metric": "euclidean",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["metric"] == "euclidean"
    assert len(data["results"]) > 0
    # Closest by euclidean should still be es_other row 0
    assert data["results"][0]["embedding_set_id"] == ids["es_other"]
    assert data["results"][0]["row_index"] == 0


async def test_window_offset_seconds(seeded, client):
    """window_offset_seconds is computed from row_index * window_size."""
    ids = seeded
    resp = await client.post(
        "/search/similar",
        json={
            "embedding_set_id": ids["es_query"],
            "row_index": 0,
            "exclude_self": False,
        },
    )
    data = resp.json()
    for r in data["results"]:
        expected_offset = r["row_index"] * 5.0  # window_size_seconds=5.0
        assert r["window_offset_seconds"] == pytest.approx(expected_offset)


# ---------------------------------------------------------------------------
# POST /search/similar-by-vector
# ---------------------------------------------------------------------------


async def test_vector_search_returns_200(seeded, client):
    """POST /search/similar-by-vector returns 200 with valid vector."""
    resp = await client.post(
        "/search/similar-by-vector",
        json={
            "vector": [1.0, 0.0, 0.0, 0.0],
            "model_version": "perch_v1",
            "top_k": 5,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_version"] == "perch_v1"
    assert data["metric"] == "cosine"
    assert data["total_candidates"] > 0
    assert len(data["results"]) > 0
    # Query is [1,0,0,0]; best match should be es_query row 0 or es_other row 0
    top = data["results"][0]
    assert top["score"] > 0.8
    assert "audio_filename" in top
    assert "window_offset_seconds" in top


async def test_vector_search_dimension_mismatch_400(seeded, client):
    """POST /search/similar-by-vector returns 400 for dimension mismatch."""
    # Embedding sets have vector_dim=4, but we send a 3-dim vector
    resp = await client.post(
        "/search/similar-by-vector",
        json={
            "vector": [1.0, 0.0, 0.0],
            "model_version": "perch_v1",
            "top_k": 5,
        },
    )
    assert resp.status_code == 400
    assert "dimension" in resp.json()["detail"].lower()


async def test_vector_search_unknown_model_400(client):
    """POST /search/similar-by-vector returns 400 for unknown model_version."""
    resp = await client.post(
        "/search/similar-by-vector",
        json={
            "vector": [1.0, 0.0, 0.0, 0.0],
            "model_version": "nonexistent_model",
            "top_k": 5,
        },
    )
    assert resp.status_code == 400
    assert "no embedding sets" in resp.json()["detail"].lower()


async def test_vector_search_euclidean(seeded, client):
    """POST /search/similar-by-vector with euclidean metric."""
    resp = await client.post(
        "/search/similar-by-vector",
        json={
            "vector": [1.0, 0.0, 0.0, 0.0],
            "model_version": "perch_v1",
            "metric": "euclidean",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["metric"] == "euclidean"
    assert len(data["results"]) > 0


async def test_vector_search_with_embedding_set_ids_filter(seeded, client):
    """POST /search/similar-by-vector respects embedding_set_ids filter."""
    ids = seeded
    resp = await client.post(
        "/search/similar-by-vector",
        json={
            "vector": [1.0, 0.0, 0.0, 0.0],
            "model_version": "perch_v1",
            "embedding_set_ids": [ids["es_other"]],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    result_es_ids = {r["embedding_set_id"] for r in data["results"]}
    assert result_es_ids == {ids["es_other"]}


# ---------------------------------------------------------------------------
# POST /search/similar-by-audio + GET /search/jobs/{id}
# ---------------------------------------------------------------------------


@pytest.fixture
async def detection_job_id(app_settings, client):
    """Create a detection job for audio search tests."""
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)
    async with sf() as session:
        cm = ClassifierModel(
            name="test_cls",
            model_path="/fake/model.tflite",
            model_version="perch_v1",
            vector_dim=4,
            window_size_seconds=5.0,
            target_sample_rate=32000,
        )
        session.add(cm)
        await session.flush()

        dj = DetectionJob(
            classifier_model_id=cm.id,
            audio_folder="/fake/audio",
            status="complete",
        )
        session.add(dj)
        await session.flush()
        dj_id = dj.id
        await session.commit()
    await engine.dispose()
    return dj_id


async def test_create_audio_search_201(detection_job_id, client):
    """POST /search/similar-by-audio creates a queued search job."""
    resp = await client.post(
        "/search/similar-by-audio",
        json={
            "detection_job_id": detection_job_id,
            "filename": "test.wav",
            "start_sec": 0.0,
            "end_sec": 5.0,
        },
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["status"] == "queued"
    assert "id" in data


async def test_create_audio_search_404_missing_detection(client):
    """POST /search/similar-by-audio returns 404 for nonexistent detection job."""
    resp = await client.post(
        "/search/similar-by-audio",
        json={
            "detection_job_id": "nonexistent-id",
            "filename": "test.wav",
            "start_sec": 0.0,
            "end_sec": 5.0,
        },
    )
    assert resp.status_code == 404


async def test_poll_search_job_queued(detection_job_id, client):
    """GET /search/jobs/{id} returns queued status before worker processes."""
    resp = await client.post(
        "/search/similar-by-audio",
        json={
            "detection_job_id": detection_job_id,
            "filename": "test.wav",
            "start_sec": 0.0,
            "end_sec": 5.0,
        },
    )
    job_id = resp.json()["id"]

    resp2 = await client.get(f"/search/jobs/{job_id}")
    assert resp2.status_code == 200
    assert resp2.json()["status"] == "queued"


async def test_poll_search_job_404(client):
    """GET /search/jobs/{id} returns 404 for nonexistent job."""
    resp = await client.get("/search/jobs/nonexistent-id")
    assert resp.status_code == 404


async def test_poll_completed_search_job(
    detection_job_id, seeded, app_settings, client
):
    """GET /search/jobs/{id} runs search and returns results for a completed job."""
    import json

    # Create a search job
    resp = await client.post(
        "/search/similar-by-audio",
        json={
            "detection_job_id": detection_job_id,
            "filename": "test.wav",
            "start_sec": 0.0,
            "end_sec": 5.0,
        },
    )
    job_id = resp.json()["id"]

    # Manually mark it as complete with a known vector
    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)
    from sqlalchemy import update

    async with sf() as session:
        await session.execute(
            update(SearchJob)
            .where(SearchJob.id == job_id)
            .values(
                status="complete",
                model_version="perch_v1",
                embedding_vector=json.dumps([1.0, 0.0, 0.0, 0.0]),
            )
        )
        await session.commit()
    await engine.dispose()

    # Poll — should get search results
    resp2 = await client.get(f"/search/jobs/{job_id}")
    assert resp2.status_code == 200
    data = resp2.json()
    assert data["status"] == "complete"
    assert data["results"] is not None
    assert data["results"]["model_version"] == "perch_v1"
    assert len(data["results"]["results"]) > 0

    # Verify query_vector and model_version are returned for re-search
    assert data["query_vector"] == [1.0, 0.0, 0.0, 0.0]
    assert data["model_version"] == "perch_v1"

    # Job should be cleaned up — polling again returns 404
    resp3 = await client.get(f"/search/jobs/{job_id}")
    assert resp3.status_code == 404
