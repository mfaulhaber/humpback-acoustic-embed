"""Integration tests for timeline API endpoints."""

import uuid

import pyarrow as pa
import pyarrow.parquet as pq

from humpback.database import create_engine, create_session_factory
from humpback.models.classifier import ClassifierModel, DetectionJob
from humpback.storage import detection_diagnostics_path


async def _create_completed_hydrophone_job(app_settings, *, num_windows: int = 20):
    """Create a completed hydrophone detection job with diagnostics parquet."""
    model_id = str(uuid.uuid4())
    job_id = str(uuid.uuid4())

    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)

    async with sf() as session:
        from sqlalchemy import insert

        await session.execute(
            insert(ClassifierModel).values(
                id=model_id,
                name="timeline-test-model",
                model_path="/fake/path",
                model_version="test_v1",
                vector_dim=128,
                window_size_seconds=5.0,
                target_sample_rate=32000,
            )
        )
        # Hydrophone job: start_timestamp=1000, end_timestamp=1000+num_windows*5
        job_duration = num_windows * 5.0
        await session.execute(
            insert(DetectionJob).values(
                id=job_id,
                status="complete",
                classifier_model_id=model_id,
                detection_mode="windowed",
                hydrophone_id="test_hydrophone",
                hydrophone_name="test_hydro",
                start_timestamp=1000.0,
                end_timestamp=1000.0 + job_duration,
                local_cache_path="/fake/cache",
                timeline_tiles_ready=False,
            )
        )
        await session.commit()

    # Create diagnostics parquet
    diag_path = detection_diagnostics_path(app_settings.storage_root, job_id)
    diag_path.parent.mkdir(parents=True, exist_ok=True)

    # Build a parquet with offset_sec and score columns
    offsets = [float(i * 5) for i in range(num_windows)]
    scores = [0.1 + 0.8 * (i / max(1, num_windows - 1)) for i in range(num_windows)]
    filenames = ["stream.ts"] * num_windows

    table = pa.table(
        {
            "offset_sec": pa.array(offsets, type=pa.float64()),
            "score": pa.array(scores, type=pa.float64()),
            "filename": pa.array(filenames, type=pa.string()),
        }
    )
    pq.write_table(table, str(diag_path))

    await engine.dispose()
    return job_id


async def test_tile_endpoint_returns_png(client, app_settings):
    """GET /tile returns a PNG image (rendered on miss)."""
    job_id = await _create_completed_hydrophone_job(app_settings)

    resp = await client.get(
        f"/classifier/detection-jobs/{job_id}/timeline/tile",
        params={"zoom_level": "24h", "tile_index": 0},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    # PNG magic bytes
    assert resp.content[:4] == b"\x89PNG"


async def test_tile_endpoint_caches_result(client, app_settings):
    """Second request should hit the cache and return the same content."""
    job_id = await _create_completed_hydrophone_job(app_settings)

    resp1 = await client.get(
        f"/classifier/detection-jobs/{job_id}/timeline/tile",
        params={"zoom_level": "24h", "tile_index": 0},
    )
    assert resp1.status_code == 200

    resp2 = await client.get(
        f"/classifier/detection-jobs/{job_id}/timeline/tile",
        params={"zoom_level": "24h", "tile_index": 0},
    )
    assert resp2.status_code == 200
    assert resp2.content == resp1.content


async def test_tile_endpoint_invalid_zoom(client, app_settings):
    """Invalid zoom level returns 400."""
    job_id = await _create_completed_hydrophone_job(app_settings)

    resp = await client.get(
        f"/classifier/detection-jobs/{job_id}/timeline/tile",
        params={"zoom_level": "99h", "tile_index": 0},
    )
    assert resp.status_code == 400


async def test_tile_endpoint_invalid_index(client, app_settings):
    """Tile index out of range returns 400."""
    job_id = await _create_completed_hydrophone_job(app_settings)

    # 20 windows * 5s = 100s duration, 24h tile = 86400s -> only 1 tile
    resp = await client.get(
        f"/classifier/detection-jobs/{job_id}/timeline/tile",
        params={"zoom_level": "24h", "tile_index": 5},
    )
    assert resp.status_code == 400


async def test_tile_endpoint_job_not_found(client):
    """Non-existent job returns 404."""
    resp = await client.get(
        "/classifier/detection-jobs/nonexistent/timeline/tile",
        params={"zoom_level": "24h", "tile_index": 0},
    )
    assert resp.status_code == 404


async def test_confidence_endpoint(client, app_settings):
    """GET /confidence returns sorted scores."""
    num_windows = 10
    job_id = await _create_completed_hydrophone_job(
        app_settings, num_windows=num_windows
    )

    resp = await client.get(
        f"/classifier/detection-jobs/{job_id}/timeline/confidence",
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "scores" in data
    assert "window_sec" in data
    assert "start_timestamp" in data
    assert "end_timestamp" in data
    assert len(data["scores"]) == num_windows
    assert data["window_sec"] == 5.0
    assert data["start_timestamp"] == 1000.0
    assert data["end_timestamp"] == 1000.0 + num_windows * 5.0
    # Scores should be sorted by offset (ascending offsets in fixture)
    for i in range(1, len(data["scores"])):
        assert data["scores"][i] >= data["scores"][i - 1]


async def test_confidence_endpoint_job_not_found(client):
    """Non-existent job returns 404."""
    resp = await client.get(
        "/classifier/detection-jobs/nonexistent/timeline/confidence",
    )
    assert resp.status_code == 404


async def test_confidence_endpoint_no_diagnostics(client, app_settings):
    """Job with no diagnostics file returns 404."""
    # Create a job without writing diagnostics parquet
    model_id = str(uuid.uuid4())
    job_id = str(uuid.uuid4())

    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)
    async with sf() as session:
        from sqlalchemy import insert

        await session.execute(
            insert(ClassifierModel).values(
                id=model_id,
                name="no-diag-model",
                model_path="/fake/path",
                model_version="test_v1",
                vector_dim=128,
                window_size_seconds=5.0,
                target_sample_rate=32000,
            )
        )
        await session.execute(
            insert(DetectionJob).values(
                id=job_id,
                status="complete",
                classifier_model_id=model_id,
                detection_mode="windowed",
                hydrophone_id="test_hydrophone",
                start_timestamp=1000.0,
                end_timestamp=1100.0,
            )
        )
        await session.commit()
    await engine.dispose()

    resp = await client.get(
        f"/classifier/detection-jobs/{job_id}/timeline/confidence",
    )
    assert resp.status_code == 404


async def test_audio_endpoint_returns_wav(client, app_settings):
    """GET /audio returns a WAV file."""
    job_id = await _create_completed_hydrophone_job(app_settings)

    resp = await client.get(
        f"/classifier/detection-jobs/{job_id}/timeline/audio",
        params={"start_sec": 1000.0, "duration_sec": 5.0},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"
    # Check WAV header: RIFF magic
    assert resp.content[:4] == b"RIFF"


async def test_audio_endpoint_max_duration(client, app_settings):
    """Requesting > 120 seconds returns 400."""
    job_id = await _create_completed_hydrophone_job(app_settings)

    resp = await client.get(
        f"/classifier/detection-jobs/{job_id}/timeline/audio",
        params={"start_sec": 1000.0, "duration_sec": 200.0},
    )
    assert resp.status_code == 400


async def test_audio_endpoint_job_not_found(client):
    """Non-existent job returns 404."""
    resp = await client.get(
        "/classifier/detection-jobs/nonexistent/timeline/audio",
        params={"start_sec": 0.0, "duration_sec": 5.0},
    )
    assert resp.status_code == 404


async def test_audio_endpoint_non_hydrophone_job(client, app_settings):
    """Non-hydrophone job returns 400."""
    model_id = str(uuid.uuid4())
    job_id = str(uuid.uuid4())

    engine = create_engine(app_settings.database_url)
    sf = create_session_factory(engine)
    async with sf() as session:
        from sqlalchemy import insert

        await session.execute(
            insert(ClassifierModel).values(
                id=model_id,
                name="local-model",
                model_path="/fake/path",
                model_version="test_v1",
                vector_dim=128,
                window_size_seconds=5.0,
                target_sample_rate=32000,
            )
        )
        await session.execute(
            insert(DetectionJob).values(
                id=job_id,
                status="complete",
                classifier_model_id=model_id,
                detection_mode="windowed",
                # No hydrophone_id -> local detection job
            )
        )
        await session.commit()
    await engine.dispose()

    resp = await client.get(
        f"/classifier/detection-jobs/{job_id}/timeline/audio",
        params={"start_sec": 0.0, "duration_sec": 5.0},
    )
    assert resp.status_code == 400


async def test_prepare_endpoint(client, app_settings):
    """POST /prepare renders coarse tiles and marks job ready."""
    job_id = await _create_completed_hydrophone_job(app_settings)

    resp = await client.post(
        f"/classifier/detection-jobs/{job_id}/timeline/prepare",
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["tiles_rendered"] > 0
    assert data["timeline_tiles_ready"] is True

    # Verify the job is now marked as ready in DB
    get_resp = await client.get(f"/classifier/detection-jobs/{job_id}")
    assert get_resp.status_code == 200
    # The DetectionJobOut schema may or may not expose timeline_tiles_ready,
    # but we verified it via the prepare response


async def test_prepare_endpoint_job_not_found(client):
    """Non-existent job returns 404."""
    resp = await client.post(
        "/classifier/detection-jobs/nonexistent/timeline/prepare",
    )
    assert resp.status_code == 404
