"""Tests for the shared timeline tile service."""

from __future__ import annotations

import threading
import time


def test_get_or_render_tile_serializes_same_tile_misses(
    settings, monkeypatch, tmp_path
):
    from humpback.processing.timeline_repository import TimelineTileRepository
    from humpback.services import timeline_tile_service as service

    job = type(
        "FakeJob",
        (),
        {
            "id": "job-a",
            "hydrophone_id": "test_hydrophone",
            "local_cache_path": "/fake/cache",
            "start_timestamp": 1000.0,
            "end_timestamp": 4600.0,
        },
    )()
    repository = TimelineTileRepository(tmp_path / "timeline_cache")
    render_count = 0
    render_lock = threading.Lock()

    def fake_render_tile(**kwargs):
        nonlocal render_count
        with render_lock:
            render_count += 1
        time.sleep(0.05)
        return b"tile-bytes"

    monkeypatch.setattr(service, "render_tile", fake_render_tile)

    barrier = threading.Barrier(2)
    results: list[bytes] = []

    def request_tile():
        barrier.wait()
        result = service.get_or_render_tile(
            job=job,
            settings=settings,
            zoom_level="1h",
            tile_index=0,
            repository=repository,
        )
        results.append(result.data)

    thread_a = threading.Thread(target=request_tile)
    thread_b = threading.Thread(target=request_tile)
    thread_a.start()
    thread_b.start()
    thread_a.join()
    thread_b.join()

    assert results == [b"tile-bytes", b"tile-bytes"]
    assert render_count == 1
