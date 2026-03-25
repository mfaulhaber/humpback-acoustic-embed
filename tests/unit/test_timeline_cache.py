"""Tests for timeline tile disk cache."""

from pathlib import Path

import pytest


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    return tmp_path / "tile_cache"


def test_put_and_get(cache_dir: Path):
    """Stored tile should be retrievable."""
    from humpback.processing.timeline_cache import TimelineTileCache

    cache = TimelineTileCache(cache_dir, max_items=100)
    cache.put("job1", "6h", 0, b"fake-png-data")
    result = cache.get("job1", "6h", 0)
    assert result == b"fake-png-data"


def test_get_miss(cache_dir: Path):
    """Missing tile should return None."""
    from humpback.processing.timeline_cache import TimelineTileCache

    cache = TimelineTileCache(cache_dir, max_items=100)
    assert cache.get("job1", "6h", 99) is None


def test_directory_structure(cache_dir: Path):
    """Tiles should be stored in {cache_dir}/{job_id}/{zoom}/tile_{index:04d}.png."""
    from humpback.processing.timeline_cache import TimelineTileCache

    cache = TimelineTileCache(cache_dir, max_items=100)
    cache.put("job-abc", "1h", 5, b"data")
    expected = cache_dir / "job-abc" / "1h" / "tile_0005.png"
    assert expected.is_file()
    assert expected.read_bytes() == b"data"


def test_fifo_eviction(cache_dir: Path):
    """Oldest tiles should be evicted when global count exceeds max."""
    from humpback.processing.timeline_cache import TimelineTileCache

    cache = TimelineTileCache(cache_dir, max_items=3)
    cache.put("j1", "24h", 0, b"a")
    cache.put("j1", "6h", 0, b"b")
    cache.put("j1", "6h", 1, b"c")
    # All 3 present
    assert cache.get("j1", "24h", 0) is not None
    # Adding 4th should evict oldest
    cache.put("j2", "24h", 0, b"d")
    assert cache.get("j1", "24h", 0) is None
    assert cache.get("j2", "24h", 0) == b"d"


def test_put_is_atomic(cache_dir: Path):
    """No .tmp files should remain after put."""
    from humpback.processing.timeline_cache import TimelineTileCache

    cache = TimelineTileCache(cache_dir, max_items=100)
    cache.put("j1", "1m", 0, b"data")
    tmp_files = list(cache_dir.rglob("*.tmp"))
    assert tmp_files == []
