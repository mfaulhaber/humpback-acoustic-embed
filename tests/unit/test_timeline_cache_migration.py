"""Tests for ``TimelineTileCache.ensure_job_cache_current`` migration."""

from __future__ import annotations

import time
from pathlib import Path

from humpback.processing.timeline_cache import (
    TIMELINE_CACHE_VERSION,
    TimelineTileCache,
)


def _legacy_cache_layout(cache_dir: Path, job_id: str) -> dict[str, Path]:
    """Populate a job's cache directory with pre-PCEN artifacts."""
    job_dir = cache_dir / job_id
    zoom_dirs = {
        zoom: job_dir / zoom for zoom in ("24h", "6h", "1h", "15m", "5m", "1m")
    }
    for zoom, zoom_dir in zoom_dirs.items():
        zoom_dir.mkdir(parents=True, exist_ok=True)
        for i in range(3):
            (zoom_dir / f"tile_{i:04d}.png").write_bytes(
                b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
            )
    (job_dir / ".ref_db.json").write_text('{"ref_db": -55.0}')
    (job_dir / ".gain_profile.json").write_text(
        '{"segments": [], "global_median_rms_db": -40.0}'
    )
    (job_dir / ".prepare_plan.json").write_text(
        '{"scope": "startup", "zooms": {"1h": [0, 1]}}'
    )
    (job_dir / ".audio_manifest.json").write_text('{"entries": []}')
    (job_dir / ".last_access").touch()
    return {
        "job_dir": job_dir,
        **{f"zoom_{k}": v for k, v in zoom_dirs.items()},
    }


def test_migrates_legacy_cache(tmp_path: Path):
    cache_dir = tmp_path / "timeline_cache"
    job_id = "job-migrate"
    layout = _legacy_cache_layout(cache_dir, job_id)

    cache = TimelineTileCache(cache_dir, max_jobs=5)
    cache.ensure_job_cache_current(job_id)

    job_dir = layout["job_dir"]

    # Legacy sidecars removed.
    assert not (job_dir / ".ref_db.json").exists()
    assert not (job_dir / ".gain_profile.json").exists()

    # All tile PNGs deleted from each zoom subdirectory.
    for zoom in ("24h", "6h", "1h", "15m", "5m", "1m"):
        remaining = list((job_dir / zoom).glob("tile_*.png"))
        assert remaining == []

    # Preserved sidecars (``.prepare_plan.json`` is an active runtime
    # artifact and must survive migration).
    assert (job_dir / ".audio_manifest.json").exists()
    assert (job_dir / ".prepare_plan.json").exists()
    assert (job_dir / ".last_access").exists()

    # Current version marker is written.
    version_path = job_dir / ".cache_version"
    assert version_path.exists()
    assert int(version_path.read_text().strip()) == TIMELINE_CACHE_VERSION


def test_current_cache_is_noop(tmp_path: Path):
    cache_dir = tmp_path / "timeline_cache"
    job_id = "job-current"
    job_dir = cache_dir / job_id
    (job_dir / "1h").mkdir(parents=True)
    tile_path = job_dir / "1h" / "tile_0000.png"
    tile_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)
    version_path = job_dir / ".cache_version"
    version_path.write_text(str(TIMELINE_CACHE_VERSION))

    original_tile_mtime = tile_path.stat().st_mtime
    original_version_mtime = version_path.stat().st_mtime
    time.sleep(0.05)

    cache = TimelineTileCache(cache_dir, max_jobs=5)
    cache.ensure_job_cache_current(job_id)

    # Tile and version marker untouched.
    assert tile_path.exists()
    assert tile_path.stat().st_mtime == original_tile_mtime
    assert version_path.stat().st_mtime == original_version_mtime


def test_missing_job_dir_is_safe(tmp_path: Path):
    cache_dir = tmp_path / "timeline_cache"
    cache = TimelineTileCache(cache_dir, max_jobs=5)
    # Should not raise or create anything.
    cache.ensure_job_cache_current("nonexistent")
    assert not (cache_dir / "nonexistent").exists()


def test_old_version_file_triggers_migration(tmp_path: Path):
    cache_dir = tmp_path / "timeline_cache"
    job_id = "job-v1"
    layout = _legacy_cache_layout(cache_dir, job_id)
    (layout["job_dir"] / ".cache_version").write_text("1")

    cache = TimelineTileCache(cache_dir, max_jobs=5)
    cache.ensure_job_cache_current(job_id)

    assert not (layout["job_dir"] / ".ref_db.json").exists()
    assert not (layout["job_dir"] / ".gain_profile.json").exists()
    version_path = layout["job_dir"] / ".cache_version"
    assert int(version_path.read_text().strip()) == TIMELINE_CACHE_VERSION


def test_migration_drops_in_memory_cache_entries(tmp_path: Path):
    cache_dir = tmp_path / "timeline_cache"
    job_id = "job-memory"
    _legacy_cache_layout(cache_dir, job_id)

    cache = TimelineTileCache(cache_dir, max_jobs=5, memory_cache_max_items=16)
    # Seed the memory cache with a tile lookup for this job.
    assert cache.get(job_id, "1h", 0) == b"\x89PNG\r\n\x1a\n" + b"\x00" * 16

    cache.ensure_job_cache_current(job_id)

    # The now-stale memory entry must not be returned on the next
    # lookup; since the tile file was deleted, the result should be
    # ``None``.
    assert cache.get(job_id, "1h", 0) is None


def test_migration_does_not_touch_shared_span_repository(tmp_path: Path):
    cache_dir = tmp_path / "timeline_cache"
    job_id = "job-legacy"
    _legacy_cache_layout(cache_dir, job_id)
    shared_tile = (
        cache_dir
        / "spans"
        / "span-key"
        / "lifted-ocean"
        / "v1"
        / "1m"
        / "f0-3000"
        / "w512_h256"
        / "tile_0000.png"
    )
    shared_tile.parent.mkdir(parents=True)
    shared_tile.write_bytes(b"shared")

    cache = TimelineTileCache(cache_dir, max_jobs=5)
    cache.ensure_job_cache_current(job_id)

    assert shared_tile.read_bytes() == b"shared"
