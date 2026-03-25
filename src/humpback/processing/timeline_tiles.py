"""Multi-resolution spectrogram tile renderer for the timeline viewer.

Uses the Ocean Depth colormap (navy -> teal -> seafoam -> white) and renders
marker-free PNG tiles at fixed pixel dimensions.
"""

import io
import math

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from scipy.signal import stft  # noqa: E402

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

ZOOM_LEVELS = ("24h", "6h", "1h", "15m", "5m", "1m")

_TILE_DURATIONS: dict[str, float] = {
    "24h": 86400.0,
    "6h": 21600.0,
    "1h": 600.0,
    "15m": 150.0,
    "5m": 50.0,
    "1m": 10.0,
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
    dynamic_range_db: float = 80.0,
    width_px: int = 512,
    height_px: int = 256,
) -> bytes:
    """Render a marker-free spectrogram PNG tile with Ocean Depth colormap.

    Returns raw PNG bytes with no axes, labels, or padding — just pixels.
    """
    if len(audio) < n_fft:
        audio = np.pad(audio, (0, n_fft - len(audio)))

    noverlap = n_fft - hop_length
    f, _t, Zxx = stft(
        audio, fs=sample_rate, window="hann", nperseg=n_fft, noverlap=noverlap
    )

    power = np.abs(Zxx) ** 2
    power = np.maximum(power, 1e-12)
    power_db = 10.0 * np.log10(power)

    # Frequency cropping
    freq_mask = (f >= freq_min) & (f <= freq_max)
    power_db = power_db[freq_mask, :]

    vmax = float(power_db.max())
    vmin = vmax - dynamic_range_db

    cmap = get_ocean_depth_colormap()

    dpi = 100
    fig, ax = plt.subplots(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_axis_off()

    ax.imshow(
        power_db,
        aspect="auto",
        origin="lower",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        interpolation="bilinear",
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    return buf.read()
