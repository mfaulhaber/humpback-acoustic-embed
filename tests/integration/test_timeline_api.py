"""Integration tests for timeline API endpoints."""

import threading
import time
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

    # Build a parquet matching WINDOW_DIAGNOSTICS_SCHEMA columns
    offsets = [float(i * 5) for i in range(num_windows)]
    scores = [0.1 + 0.8 * (i / max(1, num_windows - 1)) for i in range(num_windows)]
    filenames = ["stream.ts"] * num_windows

    table = pa.table(
        {
            "filename": pa.array(filenames, type=pa.string()),
            "window_index": pa.array(list(range(num_windows)), type=pa.int32()),
            "offset_sec": pa.array(offsets, type=pa.float32()),
            "end_sec": pa.array([o + 5.0 for o in offsets], type=pa.float32()),
            "confidence": pa.array(scores, type=pa.float32()),
            "is_overlapped": pa.array([False] * num_windows, type=pa.bool_()),
            "overlap_sec": pa.array([0.0] * num_windows, type=pa.float32()),
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
    """Requesting > 600 seconds returns 400."""
    job_id = await _create_completed_hydrophone_job(app_settings)

    resp = await client.get(
        f"/classifier/detection-jobs/{job_id}/timeline/audio",
        params={"start_sec": 1000.0, "duration_sec": 601.0},
    )
    assert resp.status_code in (400, 422)


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
    """POST /prepare renders tiles and marks job ready."""
    job_id = await _create_completed_hydrophone_job(app_settings)

    resp = await client.post(
        f"/classifier/detection-jobs/{job_id}/timeline/prepare",
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["timeline_tiles_ready"] is True


async def test_prepare_all_zoom_levels(client, app_settings):
    """Explicit full prepare should report all zoom levels."""
    import time

    job_id = await _create_completed_hydrophone_job(app_settings)
    resp = await client.post(
        f"/classifier/detection-jobs/{job_id}/timeline/prepare",
        json={"scope": "full"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["timeline_tiles_ready"] is True

    # Give background thread a moment to finish (stub audio is fast)
    time.sleep(0.5)

    # Check that tiles were rendered for at least some zoom levels via status
    status_resp = await client.get(
        f"/classifier/detection-jobs/{job_id}/timeline/prepare-status",
    )
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert set(status_data) == {"24h", "6h", "1h", "15m", "5m", "1m"}
    total_rendered = sum(v["rendered"] for v in status_data.values())
    assert total_rendered > 0


async def test_prepare_status_endpoint(client, app_settings):
    """Startup prepare status should reflect the bounded startup target set."""
    import time

    job_id = await _create_completed_hydrophone_job(app_settings)
    # Trigger prepare first
    await client.post(
        f"/classifier/detection-jobs/{job_id}/timeline/prepare",
    )
    # Give background thread a moment to finish
    time.sleep(0.5)

    resp = await client.get(
        f"/classifier/detection-jobs/{job_id}/timeline/prepare-status",
    )
    assert resp.status_code == 200
    data = resp.json()
    assert set(data) == {"1h", "6h"}
    assert data["1h"]["total"] == 1
    assert data["6h"]["total"] == 1
    assert data["1h"]["rendered"] <= data["1h"]["total"]
    assert data["6h"]["rendered"] <= data["6h"]["total"]


async def test_prepare_status_startup_scope_uses_bounded_tile_set(client, app_settings):
    """Startup prepare should target only the requested zoom neighborhood."""
    import time

    job_id = await _create_completed_hydrophone_job(app_settings, num_windows=400)

    resp = await client.post(
        f"/classifier/detection-jobs/{job_id}/timeline/prepare",
        json={
            "scope": "startup",
            "zoom_level": "1h",
            "center_timestamp": 2000.0,
            "radius_tiles": 1,
        },
    )
    assert resp.status_code == 200

    time.sleep(0.2)

    status_resp = await client.get(
        f"/classifier/detection-jobs/{job_id}/timeline/prepare-status",
    )
    assert status_resp.status_code == 200
    data = status_resp.json()
    assert data["1h"]["total"] == 3
    assert data["6h"]["total"] == 1


async def test_prepare_status_job_not_found(client):
    """GET /prepare-status for missing job should 404."""
    resp = await client.get(
        "/classifier/detection-jobs/99999/timeline/prepare-status",
    )
    assert resp.status_code == 404


async def test_prepare_endpoint_job_not_found(client):
    """Non-existent job returns 404."""
    resp = await client.post(
        "/classifier/detection-jobs/nonexistent/timeline/prepare",
    )
    assert resp.status_code == 404


async def test_prepare_endpoint_is_idempotent_while_local_prepare_is_running(
    client, app_settings, monkeypatch
):
    """Repeated prepare requests should not start duplicate local background work."""
    from humpback.api.routers import timeline as timeline_router

    job_id = await _create_completed_hydrophone_job(app_settings)
    started = threading.Event()
    release = threading.Event()
    call_count = 0

    def fake_prepare_tiles_sync(*, job, settings, cache, targets=None):
        nonlocal call_count
        call_count += 1
        started.set()
        assert release.wait(timeout=1.0)
        return 0

    monkeypatch.setattr(timeline_router, "_prepare_tiles_sync", fake_prepare_tiles_sync)

    resp1 = await client.post(
        f"/classifier/detection-jobs/{job_id}/timeline/prepare",
    )
    assert resp1.status_code == 200
    assert started.wait(timeout=1.0)

    resp2 = await client.post(
        f"/classifier/detection-jobs/{job_id}/timeline/prepare",
    )
    assert resp2.status_code == 200
    assert call_count == 1

    release.set()
    time.sleep(0.1)
    assert call_count == 1


async def test_prepare_endpoint_succeeds_when_other_process_holds_lock(
    client, app_settings, monkeypatch
):
    """Prepare should stay idempotent when another process already owns the lock."""
    from humpback.api.routers import timeline as timeline_router

    job_id = await _create_completed_hydrophone_job(app_settings)

    monkeypatch.setattr(
        timeline_router.TimelineTileCache,
        "try_acquire_prepare_lock",
        lambda self, job_id: None,
    )

    def fail_launch_prepare_thread(**kwargs):
        raise AssertionError(
            "prepare thread should not launch when lock is unavailable"
        )

    monkeypatch.setattr(
        timeline_router, "_launch_prepare_thread", fail_launch_prepare_thread
    )

    resp = await client.post(
        f"/classifier/detection-jobs/{job_id}/timeline/prepare",
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["timeline_tiles_ready"] is True


def test_prepare_tiles_sync_renders_each_requested_tile_once(
    app_settings, monkeypatch, tmp_path
):
    """``_prepare_tiles_sync`` should call the per-tile renderer exactly
    once for every requested target (PCEN has no per-job state, so every
    tile is rendered independently).
    """
    from humpback.api.routers import timeline as timeline_router

    job = type(
        "FakeJob",
        (),
        {
            "id": "job-a",
            "hydrophone_id": "test_hydrophone",
            "local_cache_path": "/fake/cache",
            "start_timestamp": 1000.0,
            "end_timestamp": 3000.0,
        },
    )()
    cache = timeline_router.TimelineTileCache(tmp_path / "tile_cache", max_jobs=5)
    seen_targets: list[tuple[str, int]] = []

    def fake_get_or_render_tile(
        *, job, zoom_level, tile_index, settings, repository, renderer=None, **kwargs
    ):
        from humpback.processing.timeline_renderers import DEFAULT_TIMELINE_RENDERER
        from humpback.services.timeline_tile_service import (
            TimelineTileResult,
            source_ref_from_job,
            tile_request_from_settings,
        )

        renderer = renderer or DEFAULT_TIMELINE_RENDERER
        seen_targets.append((zoom_level, tile_index))
        source_ref = source_ref_from_job(job, settings)
        request = tile_request_from_settings(
            zoom_level=zoom_level,
            tile_index=tile_index,
            freq_min=kwargs.get("freq_min", 0),
            freq_max=kwargs.get("freq_max", 3000),
            settings=settings,
        )
        repository.put(
            source_ref, renderer.renderer_id, renderer.version, request, b"\x89PNG"
        )
        return TimelineTileResult(data=b"\x89PNG", cache_hit=False)

    monkeypatch.setattr(timeline_router, "get_or_render_tile", fake_get_or_render_tile)

    rendered = timeline_router._prepare_tiles_sync(
        job=job,
        settings=app_settings,
        cache=cache,
        targets={"1h": [0, 1], "6h": [0]},
    )

    assert rendered == 3
    assert sorted(seen_targets) == sorted([("1h", 0), ("1h", 1), ("6h", 0)])


async def test_tile_endpoint_miss_launches_neighbor_prepare(
    client, app_settings, monkeypatch
):
    """A tile miss should queue bounded same-zoom neighbor warming."""
    from humpback.api.routers import timeline as timeline_router

    job_id = await _create_completed_hydrophone_job(app_settings, num_windows=400)
    launched: dict[str, object] = {}

    def fake_get_or_render_tile(
        *, job, zoom_level, tile_index, settings, repository, renderer=None, **kwargs
    ):
        from humpback.processing.timeline_renderers import DEFAULT_TIMELINE_RENDERER
        from humpback.services.timeline_tile_service import (
            TimelineTileResult,
            source_ref_from_job,
            tile_request_from_settings,
        )

        renderer = renderer or DEFAULT_TIMELINE_RENDERER
        source_ref = source_ref_from_job(job, settings)
        request = tile_request_from_settings(
            zoom_level=zoom_level,
            tile_index=tile_index,
            freq_min=kwargs.get("freq_min", 0),
            freq_max=kwargs.get("freq_max", 3000),
            settings=settings,
        )
        repository.put(
            source_ref, renderer.renderer_id, renderer.version, request, b"\x89PNG"
        )
        return TimelineTileResult(data=b"\x89PNG", cache_hit=False)

    def fake_launch_prepare_thread(*, job, settings, cache, prepare_lock, targets=None):
        launched["targets"] = targets
        prepare_lock.release()
        timeline_router._release_prepare_slot(job.id)

    monkeypatch.setattr(timeline_router, "get_or_render_tile", fake_get_or_render_tile)
    monkeypatch.setattr(
        timeline_router, "_launch_prepare_thread", fake_launch_prepare_thread
    )

    resp = await client.get(
        f"/classifier/detection-jobs/{job_id}/timeline/tile",
        params={"zoom_level": "1h", "tile_index": 1},
    )

    assert resp.status_code == 200
    assert launched["targets"] == {"1h": [0, 2]}


async def test_audio_endpoint_mp3_format(client, app_settings):
    """GET /audio with format=mp3 should return audio/mpeg content."""
    job_id = await _create_completed_hydrophone_job(app_settings)

    resp = await client.get(
        f"/classifier/detection-jobs/{job_id}/timeline/audio",
        params={"start_sec": 1000.0, "duration_sec": 5.0, "format": "mp3"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/mpeg"
    # MP3 starts with ID3 tag or sync word 0xFF
    assert resp.content[:3] == b"ID3" or resp.content[0] == 0xFF


async def test_audio_endpoint_600s_accepted(client, app_settings):
    """GET /audio should accept duration_sec up to 600."""
    job_id = await _create_completed_hydrophone_job(app_settings)

    resp = await client.get(
        f"/classifier/detection-jobs/{job_id}/timeline/audio",
        params={"start_sec": 1000.0, "duration_sec": 600.0},
    )
    assert resp.status_code == 200


async def test_audio_endpoint_601s_rejected(client, app_settings):
    """GET /audio should reject duration_sec > 600."""
    job_id = await _create_completed_hydrophone_job(app_settings)

    resp = await client.get(
        f"/classifier/detection-jobs/{job_id}/timeline/audio",
        params={"start_sec": 1000.0, "duration_sec": 601.0},
    )
    assert resp.status_code in (400, 422)


async def test_audio_endpoint_invalid_format(client, app_settings):
    """GET /audio with invalid format should return 422."""
    job_id = await _create_completed_hydrophone_job(app_settings)

    resp = await client.get(
        f"/classifier/detection-jobs/{job_id}/timeline/audio",
        params={"start_sec": 1000.0, "duration_sec": 5.0, "format": "ogg"},
    )
    assert resp.status_code == 422
