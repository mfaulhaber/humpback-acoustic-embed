"""Tests for timeline tile disk cache."""

import threading
import time
from pathlib import Path

import pytest


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    return tmp_path / "tile_cache"


def test_put_and_get(cache_dir: Path):
    """Stored tile should be retrievable."""
    from humpback.processing.timeline_cache import TimelineTileCache

    cache = TimelineTileCache(cache_dir, max_jobs=5)
    cache.put("job1", "6h", 0, b"fake-png-data")
    result = cache.get("job1", "6h", 0)
    assert result == b"fake-png-data"


def test_get_miss(cache_dir: Path):
    """Missing tile should return None."""
    from humpback.processing.timeline_cache import TimelineTileCache

    cache = TimelineTileCache(cache_dir, max_jobs=5)
    assert cache.get("job1", "6h", 99) is None


def test_directory_structure(cache_dir: Path):
    """Tiles should be stored in {cache_dir}/{job_id}/{zoom}/tile_{index:04d}.png."""
    from humpback.processing.timeline_cache import TimelineTileCache

    cache = TimelineTileCache(cache_dir, max_jobs=5)
    cache.put("job-abc", "1h", 5, b"data")
    expected = cache_dir / "job-abc" / "1h" / "tile_0005.png"
    assert expected.is_file()
    assert expected.read_bytes() == b"data"


def test_put_is_atomic(cache_dir: Path):
    """No .tmp files should remain after put."""
    from humpback.processing.timeline_cache import TimelineTileCache

    cache = TimelineTileCache(cache_dir, max_jobs=5)
    cache.put("j1", "1m", 0, b"data")
    tmp_files = list(cache_dir.rglob("*.tmp"))
    assert tmp_files == []


def test_concurrent_put_same_tile_uses_distinct_temp_files(
    cache_dir: Path, monkeypatch
):
    """Concurrent writers for the same tile should not collide on a shared tmp path."""
    import os

    from humpback.processing.timeline_cache import TimelineTileCache

    cache_a = TimelineTileCache(cache_dir, max_jobs=5)
    cache_b = TimelineTileCache(cache_dir, max_jobs=5)

    barrier = threading.Barrier(2)
    real_replace = os.replace
    exceptions: list[Exception] = []

    def replace_with_barrier(src: str | Path, dst: str | Path) -> None:
        if str(src).endswith(".tmp"):
            barrier.wait(timeout=1.0)
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", replace_with_barrier)

    def writer(cache: TimelineTileCache, payload: bytes) -> None:
        try:
            cache.put("job-race", "1h", 7, payload)
        except Exception as exc:  # pragma: no cover - asserted below
            exceptions.append(exc)

    thread_a = threading.Thread(target=writer, args=(cache_a, b"a-data"))
    thread_b = threading.Thread(target=writer, args=(cache_b, b"b-data"))
    thread_a.start()
    thread_b.start()
    thread_a.join()
    thread_b.join()

    assert exceptions == []
    tile_path = cache_dir / "job-race" / "1h" / "tile_0007.png"
    assert tile_path.exists()
    assert tile_path.read_bytes() in {b"a-data", b"b-data"}
    assert list(cache_dir.rglob("*.tmp")) == []


def test_touch_job_creates_sentinel(cache_dir: Path):
    """Accessing a job should create/update .last_access sentinel."""
    from humpback.processing.timeline_cache import TimelineTileCache

    cache = TimelineTileCache(cache_dir=cache_dir, max_jobs=5)
    cache.put("job_a", "1h", 0, b"tile-data")
    sentinel = cache_dir / "job_a" / ".last_access"
    assert sentinel.exists()


def test_get_updates_sentinel_mtime(cache_dir: Path):
    """Cache hit should touch the job's sentinel to keep it fresh."""
    from humpback.processing.timeline_cache import TimelineTileCache

    cache = TimelineTileCache(cache_dir=cache_dir, max_jobs=5)
    cache.put("job_a", "1h", 0, b"tile-data")
    sentinel = cache_dir / "job_a" / ".last_access"
    old_mtime = sentinel.stat().st_mtime
    time.sleep(0.05)
    cache.get("job_a", "1h", 0)
    assert sentinel.stat().st_mtime > old_mtime


def test_lru_eviction_removes_oldest_job(cache_dir: Path):
    """When max_jobs exceeded, oldest-accessed job directory is removed."""
    from humpback.processing.timeline_cache import TimelineTileCache

    cache = TimelineTileCache(cache_dir=cache_dir, max_jobs=2)
    cache.put("job_a", "1h", 0, b"a-data")
    time.sleep(0.05)
    cache.put("job_b", "1h", 0, b"b-data")
    time.sleep(0.05)
    cache.put("job_c", "1h", 0, b"c-data")  # triggers eviction
    assert cache.get("job_a", "1h", 0) is None  # evicted
    assert cache.get("job_b", "1h", 0) == b"b-data"
    assert cache.get("job_c", "1h", 0) == b"c-data"


def test_lru_eviction_preserves_recently_accessed(cache_dir: Path):
    """Accessing an old job refreshes it so a newer-but-untouched job gets evicted."""
    from humpback.processing.timeline_cache import TimelineTileCache

    cache = TimelineTileCache(cache_dir=cache_dir, max_jobs=2)
    cache.put("job_a", "1h", 0, b"a-data")
    time.sleep(0.05)
    cache.put("job_b", "1h", 0, b"b-data")
    time.sleep(0.05)
    cache.get("job_a", "1h", 0)  # refresh job_a
    time.sleep(0.05)
    cache.put("job_c", "1h", 0, b"c-data")  # should evict job_b
    assert cache.get("job_a", "1h", 0) == b"a-data"
    assert cache.get("job_b", "1h", 0) is None  # evicted
    assert cache.get("job_c", "1h", 0) == b"c-data"


def test_job_count_returns_cached_job_count(cache_dir: Path):
    """job_count() should return the number of job directories in the cache."""
    from humpback.processing.timeline_cache import TimelineTileCache

    cache = TimelineTileCache(cache_dir=cache_dir, max_jobs=5)
    assert cache.job_count() == 0
    cache.put("job_a", "1h", 0, b"data")
    cache.put("job_b", "1h", 0, b"data")
    assert cache.job_count() == 2


def test_try_acquire_prepare_lock_is_exclusive(cache_dir: Path):
    """Only one cache instance should own a job prepare lock at a time."""
    from humpback.processing.timeline_cache import TimelineTileCache

    cache_a = TimelineTileCache(cache_dir=cache_dir, max_jobs=5)
    cache_b = TimelineTileCache(cache_dir=cache_dir, max_jobs=5)

    lock_a = cache_a.try_acquire_prepare_lock("job_a")
    assert lock_a is not None
    assert cache_b.try_acquire_prepare_lock("job_a") is None

    lock_a.release()

    lock_b = cache_b.try_acquire_prepare_lock("job_a")
    assert lock_b is not None
    lock_b.release()


def test_tile_count_for_zoom(cache_dir: Path):
    """tile_count_for_zoom() should count .png files in the zoom directory."""
    from humpback.processing.timeline_cache import TimelineTileCache

    cache = TimelineTileCache(cache_dir=cache_dir, max_jobs=5)
    assert cache.tile_count_for_zoom("job_a", "1h") == 0
    cache.put("job_a", "1h", 0, b"data")
    cache.put("job_a", "1h", 1, b"data")
    cache.put("job_a", "6h", 0, b"data")
    assert cache.tile_count_for_zoom("job_a", "1h") == 2
    assert cache.tile_count_for_zoom("job_a", "6h") == 1
