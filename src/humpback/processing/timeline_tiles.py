"""Multi-resolution spectrogram tile renderer for the timeline viewer.

Renders marker-free PNG tiles at fixed pixel dimensions.
"""

import math

import numpy as np

from humpback.processing.pcen_rendering import PcenParams, render_tile_pcen
from humpback.processing.timeline_renderers import (
    DEFAULT_TIMELINE_RENDERER,
    TimelineTileRenderInput,
    get_ocean_depth_colormap,
)

# ---- Ocean Depth Colormap ----

__all__ = [
    "ZOOM_LEVELS",
    "generate_timeline_tile",
    "get_ocean_depth_colormap",
    "tile_count",
    "tile_duration_sec",
    "tile_time_range",
]


# ---- Zoom Level Grid Math ----

ZOOM_LEVELS = ("24h", "6h", "1h", "15m", "5m", "1m", "30s", "10s")

_TILE_DURATIONS: dict[str, float] = {
    "24h": 86400.0,
    "6h": 21600.0,
    "1h": 600.0,
    "15m": 150.0,
    "5m": 50.0,
    "1m": 10.0,
    "30s": 5.0,
    "10s": 2.0,
}


def tile_duration_sec(zoom_level: str) -> float:
    """Return the duration in seconds that one tile covers at this zoom level."""
    return _TILE_DURATIONS[zoom_level]


def tile_count(zoom_level: str, *, job_duration_sec: float) -> int:
    """Return the number of tiles needed to cover the job duration."""
    return math.ceil(job_duration_sec / _TILE_DURATIONS[zoom_level])


def tile_time_range(
    zoom_level: str, *, tile_index: int, job_start_timestamp: float
) -> tuple[float, float]:
    """Return (start_epoch, end_epoch) for a tile."""
    dur = _TILE_DURATIONS[zoom_level]
    start = job_start_timestamp + tile_index * dur
    end = start + dur
    return start, end


# ---- Tile Renderer ----


def generate_timeline_tile(
    audio: np.ndarray,
    sample_rate: int,
    freq_min: int = 0,
    freq_max: int = 3000,
    n_fft: int = 2048,
    hop_length: int = 256,
    warmup_samples: int = 0,
    pcen_params: PcenParams | None = None,
    vmin: float = 0.0,
    vmax: float = 0.15,
    width_px: int = 512,
    height_px: int = 256,
) -> bytes:
    """Render a marker-free spectrogram PNG tile with the default renderer.

    The input ``audio`` is expected to begin with ``warmup_samples`` of
    pre-tile audio so the PCEN filter state can settle before the first
    rendered frame. Those frames are trimmed off the PCEN output before
    rendering.

    Returns raw PNG bytes with no axes, labels, or padding — just pixels.
    """
    return DEFAULT_TIMELINE_RENDERER.render(
        TimelineTileRenderInput(
            audio=audio,
            sample_rate=sample_rate,
            freq_min=freq_min,
            freq_max=freq_max,
            n_fft=n_fft,
            hop_length=hop_length,
            warmup_samples=warmup_samples,
            pcen_params=pcen_params,
            vmin=vmin,
            vmax=vmax,
            width_px=width_px,
            height_px=height_px,
        )
    )


def compute_timeline_pcen(
    audio: np.ndarray,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    warmup_samples: int = 0,
    pcen_params: PcenParams | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return PCEN frequencies and values for diagnostics/tests."""
    return render_tile_pcen(
        audio=audio,
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        warmup_samples=warmup_samples,
        params=pcen_params,
    )
