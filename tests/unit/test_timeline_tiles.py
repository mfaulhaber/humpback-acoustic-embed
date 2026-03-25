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
