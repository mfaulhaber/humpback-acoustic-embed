"""Integration tests for shared region timeline tile caching."""

from __future__ import annotations

import numpy as np

from humpback.database import create_engine, create_session_factory
from humpback.models.call_parsing import RegionDetectionJob

BASE = "/call-parsing"


async def _create_region_job(app_settings, *, duration_sec: float = 300.0) -> str:
    engine = create_engine(app_settings.database_url)
    try:
        session_factory = create_session_factory(engine)
        async with session_factory() as session:
            job = RegionDetectionJob(
                status="complete",
                hydrophone_id="test-hydro",
                start_timestamp=1_000_000.0,
                end_timestamp=1_000_000.0 + duration_sec,
                config_json="{}",
            )
            session.add(job)
            await session.commit()
            return job.id
    finally:
        await engine.dispose()


async def test_region_tile_hits_shared_cache_on_second_request(
    client,
    app_settings,
    monkeypatch,
) -> None:
    job_id = await _create_region_job(app_settings)
    calls: list[dict] = []
    from humpback.api.routers import timeline as timeline_router

    def _fake_resolve(**kwargs):
        calls.append(kwargs)
        sr = kwargs.get("target_sr", 32000)
        duration = kwargs.get("duration_sec", 50.0)
        n = int(sr * duration)
        t = np.arange(n) / sr
        return (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    monkeypatch.setattr(
        "humpback.processing.timeline_audio.resolve_timeline_audio",
        _fake_resolve,
    )
    monkeypatch.setattr(timeline_router, "_neighbor_prepare_targets", lambda **_: None)

    params = {"zoom_level": "5m", "tile_index": 0}
    first = await client.get(f"{BASE}/region-jobs/{job_id}/tile", params=params)
    second = await client.get(f"{BASE}/region-jobs/{job_id}/tile", params=params)

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.content == first.content
    assert len(calls) == 1
    assert calls[0]["timeline_cache"] is not None
    assert isinstance(calls[0]["job_id"], str)
    assert calls[0]["job_id"]


async def test_region_tile_miss_launches_neighbor_prepare(
    client,
    app_settings,
    monkeypatch,
) -> None:
    from humpback.api.routers import timeline as timeline_router
    from humpback.services import timeline_tile_service
    from humpback.services.timeline_tile_service import TimelineTileResult

    job_id = await _create_region_job(app_settings, duration_sec=900.0)
    launched: dict[str, object] = {}

    def fake_get_or_render_tile(**kwargs):
        return TimelineTileResult(data=b"\x89PNG", cache_hit=False)

    def fake_try_launch_prepare(**kwargs):
        launched["targets"] = kwargs["targets"]

    monkeypatch.setattr(
        timeline_tile_service, "get_or_render_tile", fake_get_or_render_tile
    )
    monkeypatch.setattr(timeline_router, "_try_launch_prepare", fake_try_launch_prepare)

    resp = await client.get(
        f"{BASE}/region-jobs/{job_id}/tile",
        params={"zoom_level": "5m", "tile_index": 1},
    )

    assert resp.status_code == 200
    assert launched["targets"] == {"5m": [0, 2]}
