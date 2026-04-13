"""Multi-resolution spectrogram tile renderer for the timeline viewer.

Uses the Ocean Depth colormap (navy -> teal -> seafoam -> white) and renders
marker-free PNG tiles at fixed pixel dimensions.
"""

import io
import math

import matplotlib
import numpy as np

from humpback.processing.pcen_rendering import PcenParams, render_tile_pcen

matplotlib.use("Agg")

import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# ---- Ocean Depth Colormap ----

_OCEAN_DEPTH_COLORS = [
    (0.0, "#000510"),
    (0.2, "#051530"),
    (0.4, "#0a3050"),
    (0.6, "#108070"),
    (0.8, "#50c8a0"),
    (1.0, "#d0fff0"),
]


def get_ocean_depth_colormap() -> mcolors.LinearSegmentedColormap:
    """Return the Ocean Depth colormap for timeline spectrograms."""
    positions = [p for p, _ in _OCEAN_DEPTH_COLORS]
    hex_colors = [c for _, c in _OCEAN_DEPTH_COLORS]
    rgb_colors = [mcolors.to_rgb(c) for c in hex_colors]
    return mcolors.LinearSegmentedColormap.from_list(
        "ocean_depth", list(zip(positions, rgb_colors))
    )


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
    """Render a marker-free spectrogram PNG tile with Ocean Depth colormap.

    The input ``audio`` is expected to begin with ``warmup_samples`` of
    pre-tile audio so the PCEN filter state can settle before the first
    rendered frame. Those frames are trimmed off the PCEN output before
    rendering.

    Returns raw PNG bytes with no axes, labels, or padding — just pixels.
    """
    freqs, pcen_power = render_tile_pcen(
        audio=audio,
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        warmup_samples=warmup_samples,
        params=pcen_params,
    )

    if pcen_power.shape[1] == 0:
        # Empty tile (no audio available) — render a flat vmin-valued
        # image at the requested pixel size so downstream code still gets
        # a valid PNG.
        pcen_power = np.full((len(freqs), max(1, width_px)), vmin, dtype=np.float32)

    freq_mask = (freqs >= freq_min) & (freqs <= freq_max)
    pcen_cropped = pcen_power[freq_mask, :]
    if pcen_cropped.shape[0] == 0:
        pcen_cropped = pcen_power

    cmap = get_ocean_depth_colormap()

    dpi = 100
    fig, ax = plt.subplots(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_axis_off()

    ax.imshow(
        pcen_cropped,
        aspect="auto",
        origin="lower",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        interpolation="bicubic",
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return buf.read()
