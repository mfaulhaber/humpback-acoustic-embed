"""Tests for timeline tile rendering and grid math."""

import numpy as np
import pytest


def test_ocean_depth_colormap_endpoints():
    """Ocean Depth colormap should map 0.0 to near-black and 1.0 to near-white."""
    from humpback.processing.timeline_tiles import get_ocean_depth_colormap

    cmap = get_ocean_depth_colormap()
    low = cmap(0.0)
    high = cmap(1.0)
    assert low[0] < 0.05 and low[1] < 0.1 and low[2] < 0.1
    assert high[0] > 0.7 and high[1] > 0.9 and high[2] > 0.8


def test_ocean_depth_colormap_midpoint_is_teal():
    """Midpoint should be in the teal range."""
    from humpback.processing.timeline_tiles import get_ocean_depth_colormap

    cmap = get_ocean_depth_colormap()
    mid = cmap(0.6)
    assert mid[0] < 0.3
    assert mid[1] > 0.3
    assert mid[2] > 0.3


@pytest.mark.parametrize(
    "zoom_level,expected_duration",
    [
        ("24h", 86400.0),
        ("6h", 21600.0),
        ("1h", 600.0),
        ("15m", 150.0),
        ("5m", 50.0),
        ("1m", 10.0),
    ],
)
def test_tile_duration(zoom_level, expected_duration):
    from humpback.processing.timeline_tiles import tile_duration_sec

    assert tile_duration_sec(zoom_level) == expected_duration


def test_tile_count_24h():
    from humpback.processing.timeline_tiles import tile_count

    count = tile_count("24h", job_duration_sec=86400.0)
    assert count == 1


def test_tile_count_6h():
    from humpback.processing.timeline_tiles import tile_count

    count = tile_count("6h", job_duration_sec=86400.0)
    assert count == 4


def test_tile_count_1m():
    from humpback.processing.timeline_tiles import tile_count

    count = tile_count("1m", job_duration_sec=86400.0)
    assert count == 8640


def test_tile_count_partial():
    from humpback.processing.timeline_tiles import tile_count

    count = tile_count("6h", job_duration_sec=45000.0)
    assert count == 3


def test_tile_time_range():
    from humpback.processing.timeline_tiles import tile_time_range

    job_start = 1000000.0
    start, end = tile_time_range("6h", tile_index=1, job_start_timestamp=job_start)
    assert start == job_start + 21600.0
    assert end == job_start + 43200.0


def test_generate_timeline_tile_returns_png():
    from humpback.processing.timeline_tiles import generate_timeline_tile

    sr = 32000
    duration = 10.0
    audio = np.random.randn(int(sr * duration)).astype(np.float32) * 0.01
    result = generate_timeline_tile(
        audio=audio,
        sample_rate=sr,
        freq_min=0,
        freq_max=3000,
        width_px=512,
        height_px=256,
    )
    assert result[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(result) > 100


def test_generate_timeline_tile_custom_freq_range():
    from humpback.processing.timeline_tiles import generate_timeline_tile

    sr = 32000
    audio = np.random.randn(sr * 5).astype(np.float32) * 0.01
    png_narrow = generate_timeline_tile(
        audio=audio,
        sample_rate=sr,
        freq_min=0,
        freq_max=1000,
        width_px=512,
        height_px=256,
    )
    png_wide = generate_timeline_tile(
        audio=audio,
        sample_rate=sr,
        freq_min=0,
        freq_max=8000,
        width_px=512,
        height_px=256,
    )
    assert png_narrow != png_wide


def test_generate_timeline_tile_fixed_ref_db():
    """Tiles with different amplitudes should produce different pixel data
    (not normalized to the same brightness)."""
    from humpback.processing.timeline_tiles import generate_timeline_tile

    sr = 32000
    np.random.seed(42)
    quiet = np.random.randn(sr * 5).astype(np.float32) * 0.001
    loud = np.random.randn(sr * 5).astype(np.float32) * 1.0

    png_quiet = generate_timeline_tile(audio=quiet, sample_rate=sr)
    png_loud = generate_timeline_tile(audio=loud, sample_rate=sr)
    # With fixed ref_db, different amplitudes produce different tiles
    assert png_quiet != png_loud


def test_generate_timeline_tile_same_ref_db_produces_consistent_tiles():
    """Two tiles from audio with the same amplitude but different random content
    should look similar (not wildly different brightness) when using the same ref_db."""
    from humpback.processing.timeline_tiles import generate_timeline_tile

    sr = 32000
    np.random.seed(10)
    audio_a = np.random.randn(sr * 5).astype(np.float32) * 0.01
    np.random.seed(20)
    audio_b = np.random.randn(sr * 5).astype(np.float32) * 0.01

    ref = -60.0
    png_a = generate_timeline_tile(audio=audio_a, sample_rate=sr, ref_db=ref)
    png_b = generate_timeline_tile(audio=audio_b, sample_rate=sr, ref_db=ref)
    # Both are valid PNGs; they may differ in content but both should render
    assert png_a[:8] == b"\x89PNG\r\n\x1a\n"
    assert png_b[:8] == b"\x89PNG\r\n\x1a\n"
    # File sizes should be in the same ballpark (within 50%) since amplitude is the same
    ratio = len(png_a) / len(png_b)
    assert 0.5 < ratio < 2.0


# ---- compute_power_db_stats tests ----


def test_compute_power_db_stats_returns_expected_keys():
    from humpback.processing.timeline_tiles import compute_power_db_stats

    sr = 32000
    audio = np.random.randn(sr * 5).astype(np.float32) * 0.01
    stats = compute_power_db_stats(audio, sr)
    assert set(stats.keys()) == {"min_db", "max_db", "p95_db", "mean_db"}
    for v in stats.values():
        assert isinstance(v, float)


def test_compute_power_db_stats_ordering():
    """min_db <= mean_db <= p95_db <= max_db."""
    from humpback.processing.timeline_tiles import compute_power_db_stats

    sr = 32000
    audio = np.random.randn(sr * 5).astype(np.float32) * 0.05
    stats = compute_power_db_stats(audio, sr)
    assert stats["min_db"] <= stats["mean_db"]
    assert stats["mean_db"] <= stats["p95_db"]
    assert stats["p95_db"] <= stats["max_db"]


def test_compute_power_db_stats_louder_audio_has_higher_values():
    from humpback.processing.timeline_tiles import compute_power_db_stats

    sr = 32000
    np.random.seed(42)
    quiet = np.random.randn(sr * 5).astype(np.float32) * 0.001
    loud = np.random.randn(sr * 5).astype(np.float32) * 1.0

    stats_quiet = compute_power_db_stats(quiet, sr)
    stats_loud = compute_power_db_stats(loud, sr)
    assert stats_loud["max_db"] > stats_quiet["max_db"]
    assert stats_loud["p95_db"] > stats_quiet["p95_db"]


def test_compute_power_db_stats_short_audio():
    """Audio shorter than n_fft should still produce valid stats (via padding)."""
    from humpback.processing.timeline_tiles import compute_power_db_stats

    sr = 32000
    audio = np.random.randn(100).astype(np.float32) * 0.01
    stats = compute_power_db_stats(audio, sr)
    assert stats["min_db"] < stats["max_db"]


# ---- TimelineTileCache ref_db tests ----


def test_cache_ref_db_roundtrip(tmp_path):
    from humpback.processing.timeline_cache import TimelineTileCache

    cache = TimelineTileCache(cache_dir=tmp_path, max_jobs=5)
    job_id = "test-job-123"

    # Initially None
    assert cache.get_ref_db(job_id) is None

    # Store and retrieve
    cache.put_ref_db(job_id, -62.5)
    assert cache.get_ref_db(job_id) == pytest.approx(-62.5)


def test_cache_ref_db_survives_reopen(tmp_path):
    from humpback.processing.timeline_cache import TimelineTileCache

    job_id = "test-job-456"

    cache1 = TimelineTileCache(cache_dir=tmp_path, max_jobs=5)
    cache1.put_ref_db(job_id, -55.0)

    cache2 = TimelineTileCache(cache_dir=tmp_path, max_jobs=5)
    assert cache2.get_ref_db(job_id) == pytest.approx(-55.0)


def test_cache_ref_db_corrupt_file_returns_none(tmp_path):
    from humpback.processing.timeline_cache import TimelineTileCache

    cache = TimelineTileCache(cache_dir=tmp_path, max_jobs=5)
    job_id = "test-job-corrupt"

    # Write corrupt JSON
    job_dir = tmp_path / job_id
    job_dir.mkdir(parents=True)
    (job_dir / ".ref_db.json").write_text("not valid json")

    assert cache.get_ref_db(job_id) is None


def test_generate_timeline_tile_silence():
    from humpback.processing.timeline_tiles import generate_timeline_tile

    sr = 32000
    audio = np.zeros(sr * 5, dtype=np.float32)
    result = generate_timeline_tile(
        audio=audio,
        sample_rate=sr,
        freq_min=0,
        freq_max=3000,
        width_px=512,
        height_px=256,
    )
    assert result[:8] == b"\x89PNG\r\n\x1a\n"
