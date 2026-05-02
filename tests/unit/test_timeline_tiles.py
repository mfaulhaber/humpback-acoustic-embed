"""Tests for timeline tile rendering and grid math."""

import numpy as np
import pytest


def test_ocean_depth_colormap_endpoints():
    """Ocean Depth compatibility colormap maps 0.0 to near-black and 1.0 to near-white."""
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
        ("30s", 5.0),
        ("10s", 2.0),
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


# ---- PCEN-based tile rendering ----


def _noise(sr: int, duration: float, rms: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(int(sr * duration)).astype(np.float32)
    current = float(np.sqrt(np.mean(x**2)))
    if current > 0:
        x *= rms / current
    return x


def test_generate_timeline_tile_returns_png():
    from humpback.processing.timeline_tiles import generate_timeline_tile

    sr = 8000
    audio = _noise(sr, 10.0, 0.01, seed=1)
    result = generate_timeline_tile(
        audio=audio,
        sample_rate=sr,
        n_fft=1024,
        hop_length=128,
        freq_min=0,
        freq_max=3000,
        width_px=512,
        height_px=256,
    )
    assert result[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(result) > 100


def test_generate_timeline_tile_uses_lifted_ocean_default():
    from humpback.processing.timeline_renderers import DEFAULT_TIMELINE_RENDERER

    assert DEFAULT_TIMELINE_RENDERER.renderer_id == "lifted-ocean"


def test_generate_timeline_tile_custom_freq_range():
    from humpback.processing.timeline_tiles import generate_timeline_tile

    sr = 8000
    audio = _noise(sr, 5.0, 0.01, seed=2)
    png_narrow = generate_timeline_tile(
        audio=audio,
        sample_rate=sr,
        n_fft=1024,
        hop_length=128,
        freq_min=0,
        freq_max=1000,
        width_px=512,
        height_px=256,
    )
    png_wide = generate_timeline_tile(
        audio=audio,
        sample_rate=sr,
        n_fft=1024,
        hop_length=128,
        freq_min=0,
        freq_max=3500,
        width_px=512,
        height_px=256,
    )
    assert png_narrow != png_wide


def test_pcen_flattens_level_differences():
    """With PCEN, spectrograms from quiet and loud stationary noise
    should converge to the same bounded output range (AGC flattens them).
    Comparing the trailing PCEN frames directly avoids PNG compression
    noise from biasing the test.
    """
    from humpback.processing.pcen_rendering import render_tile_pcen

    sr = 8000
    # Same phase, 1000× amplitude — PCEN should equalize them.
    quiet = _noise(sr, 15.0, 0.001, seed=10)
    loud = _noise(sr, 15.0, 1.0, seed=10)

    _, pcen_quiet = render_tile_pcen(
        audio=quiet, sample_rate=sr, n_fft=1024, hop_length=128
    )
    _, pcen_loud = render_tile_pcen(
        audio=loud, sample_rate=sr, n_fft=1024, hop_length=128
    )
    # Compare the well-settled trailing region.
    tail_q = float(np.mean(pcen_quiet[:, 500:]))
    tail_l = float(np.mean(pcen_loud[:, 500:]))
    ratio = tail_l / max(tail_q, 1e-12)
    # gain=0.98 preserves a small residue of the level difference;
    # 1000x → ~1.2x is the expected compression ratio.
    assert 0.7 < ratio < 1.3, (
        f"PCEN should flatten 1000x amplitude difference (ratio={ratio})"
    )


def test_generate_timeline_tile_with_warmup():
    """Rendering a tile with a warm-up prefix should trim those frames
    from the output and still produce a valid PNG."""
    from humpback.processing.timeline_tiles import generate_timeline_tile

    sr = 8000
    # Tile audio = 10 s; prepend 2 s of warm-up = 12 s total
    tile_audio = _noise(sr, 10.0, 0.02, seed=3)
    warmup = _noise(sr, 2.0, 0.02, seed=4)
    audio = np.concatenate([warmup, tile_audio])

    png = generate_timeline_tile(
        audio=audio,
        sample_rate=sr,
        n_fft=1024,
        hop_length=128,
        warmup_samples=2 * sr,
        width_px=512,
        height_px=256,
    )
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_generate_timeline_tile_empty_audio():
    """Passing zero-length audio should produce a flat PNG at the
    configured vmin rather than raising."""
    from humpback.processing.timeline_tiles import generate_timeline_tile

    png = generate_timeline_tile(
        audio=np.zeros(0, dtype=np.float32),
        sample_rate=8000,
        n_fft=1024,
        hop_length=128,
    )
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_generate_timeline_tile_silence():
    from humpback.processing.timeline_tiles import generate_timeline_tile

    sr = 8000
    audio = np.zeros(sr * 5, dtype=np.float32)
    result = generate_timeline_tile(
        audio=audio,
        sample_rate=sr,
        n_fft=1024,
        hop_length=128,
        freq_min=0,
        freq_max=3000,
        width_px=512,
        height_px=256,
    )
    assert result[:8] == b"\x89PNG\r\n\x1a\n"
