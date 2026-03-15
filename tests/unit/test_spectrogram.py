"""Tests for spectrogram generator and FIFO cache."""

import io

import matplotlib.image as mpimg
import numpy as np
import pytest

from humpback.processing.spectrogram import generate_spectrogram_png
from humpback.processing.spectrogram_cache import SpectrogramCache


# ── generate_spectrogram_png ──────────────────────────────────────────


def test_sine_wave_returns_valid_png():
    sr = 32000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    png = generate_spectrogram_png(audio, sr)
    assert png[:4] == b"\x89PNG"
    assert len(png) > 100


def test_custom_params_return_valid_png():
    sr = 16000
    audio = np.random.randn(sr * 3).astype(np.float32)
    png = generate_spectrogram_png(
        audio,
        sr,
        hop_length=256,
        dynamic_range_db=60.0,
        width_px=320,
        height_px=160,
    )
    assert png[:4] == b"\x89PNG"


def test_custom_dimensions_preserved():
    sr = 16000
    audio = np.random.randn(sr * 2).astype(np.float32)
    png = generate_spectrogram_png(audio, sr, width_px=320, height_px=160)

    image = mpimg.imread(io.BytesIO(png), format="png")
    assert image.shape[0] == 160
    assert image.shape[1] == 320


def test_very_short_audio():
    """Audio shorter than n_fft should be padded, not crash."""
    sr = 32000
    audio = np.ones(100, dtype=np.float32)
    png = generate_spectrogram_png(audio, sr, n_fft=2048)
    assert png[:4] == b"\x89PNG"


# ── SpectrogramCache ─────────────────────────────────────────────────


@pytest.fixture
def cache(tmp_path):
    return SpectrogramCache(tmp_path / "cache", max_items=3)


def test_miss_returns_none(cache):
    assert cache.get("nonexistent") is None


def test_put_get_roundtrip(cache):
    data = b"\x89PNG_fake_data"
    key = SpectrogramCache._make_key("j1", "f.wav", 0.0, 5.0, 128, 80.0, 2048, 640, 320)
    cache.put(key, data)
    assert cache.get(key) == data


def test_fifo_eviction(cache):
    """Oldest items evicted when count exceeds max_items (max_items=3)."""
    keys = []
    for i in range(5):
        key = SpectrogramCache._make_key(
            f"j{i}", "f.wav", 0.0, 5.0, 128, 80.0, 2048, 640, 320
        )
        cache.put(key, f"data{i}".encode())
        keys.append(key)

    # After 5 puts with max_items=3, exactly 3 files should survive
    surviving = [k for k in keys if cache.get(k) is not None]
    assert len(surviving) == 3


def test_cache_key_deterministic():
    k1 = SpectrogramCache._make_key("j1", "f.wav", 1.5, 5.0, 128, 80.0, 2048, 640, 320)
    k2 = SpectrogramCache._make_key("j1", "f.wav", 1.5, 5.0, 128, 80.0, 2048, 640, 320)
    assert k1 == k2


def test_cache_key_varies_on_params():
    k1 = SpectrogramCache._make_key("j1", "f.wav", 0.0, 5.0, 128, 80.0, 2048, 640, 320)
    k2 = SpectrogramCache._make_key("j1", "f.wav", 1.0, 5.0, 128, 80.0, 2048, 640, 320)
    assert k1 != k2


def test_cache_key_varies_on_renderer_version():
    k1 = SpectrogramCache._make_key(
        "j1",
        "f.wav",
        0.0,
        5.0,
        128,
        80.0,
        2048,
        640,
        320,
        renderer_version=1,
    )
    k2 = SpectrogramCache._make_key(
        "j1",
        "f.wav",
        0.0,
        5.0,
        128,
        80.0,
        2048,
        640,
        320,
        renderer_version=2,
    )
    assert k1 != k2
