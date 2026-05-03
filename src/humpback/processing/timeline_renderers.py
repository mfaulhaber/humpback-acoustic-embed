"""Renderer implementations for timeline spectrogram PNG tiles."""

from __future__ import annotations

import io
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import matplotlib.colors as mcolors
import numpy as np
from PIL import Image

from humpback.processing.pcen_rendering import PcenParams, render_tile_pcen

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@dataclass(frozen=True)
class TimelineTileRenderInput:
    """Inputs required to render a timeline spectrogram tile."""

    audio: np.ndarray
    sample_rate: int
    freq_min: int
    freq_max: int
    n_fft: int
    hop_length: int
    warmup_samples: int
    pcen_params: PcenParams | None
    vmin: float
    vmax: float
    width_px: int
    height_px: int


def colormap_from_stops(
    name: str,
    colors: tuple[tuple[float, str], ...],
) -> mcolors.LinearSegmentedColormap:
    """Create a Matplotlib colormap from normalized color stops."""
    positions = [p for p, _ in colors]
    rgb_colors = [mcolors.to_rgb(c) for _, c in colors]
    return mcolors.LinearSegmentedColormap.from_list(
        name,
        list(zip(positions, rgb_colors)),
    )


class TimelineTileRenderer(ABC):
    """Abstract renderer for marker-free timeline spectrogram PNG tiles."""

    renderer_id: ClassVar[str]
    version: ClassVar[int]

    @abstractmethod
    def colormap(self) -> mcolors.Colormap:
        """Return the renderer's display colormap."""

    def pcen_params(self, settings) -> PcenParams:
        """Build PCEN parameters from project settings."""
        return PcenParams(
            time_constant=settings.pcen_time_constant_sec,
            gain=settings.pcen_gain,
            bias=settings.pcen_bias,
            power=settings.pcen_power,
            eps=settings.pcen_eps,
        )

    def cache_metadata(self, settings) -> dict[str, object]:
        """Return renderer metadata that affects cached pixels."""
        return {
            "renderer_id": self.renderer_id,
            "version": self.version,
            "pcen": {
                "time_constant": settings.pcen_time_constant_sec,
                "gain": settings.pcen_gain,
                "bias": settings.pcen_bias,
                "power": settings.pcen_power,
                "eps": settings.pcen_eps,
            },
        }

    def render(
        self,
        render_input: TimelineTileRenderInput,
    ) -> bytes:
        """Render a marker-free spectrogram PNG tile."""
        freqs, pcen_power = render_tile_pcen(
            audio=render_input.audio,
            sample_rate=render_input.sample_rate,
            n_fft=render_input.n_fft,
            hop_length=render_input.hop_length,
            warmup_samples=render_input.warmup_samples,
            params=render_input.pcen_params,
        )

        if pcen_power.shape[1] == 0:
            pcen_power = np.full(
                (len(freqs), max(1, render_input.width_px)),
                render_input.vmin,
                dtype=np.float32,
            )

        freq_mask = (freqs >= render_input.freq_min) & (freqs <= render_input.freq_max)
        pcen_cropped = pcen_power[freq_mask, :]
        if pcen_cropped.shape[0] == 0:
            pcen_cropped = pcen_power

        display_values = self.display_values(
            pcen_cropped,
            vmin=render_input.vmin,
            vmax=render_input.vmax,
        )
        return self.encode_png(
            display_values,
            width_px=render_input.width_px,
            height_px=render_input.height_px,
        )

    def display_values(
        self,
        values: np.ndarray,
        *,
        vmin: float,
        vmax: float,
    ) -> np.ndarray:
        """Normalize PCEN values to the renderer's 0..1 display domain."""
        span = max(vmax - vmin, 1e-12)
        normalized = np.nan_to_num(values, nan=vmin, posinf=vmax, neginf=vmin) - vmin
        normalized = normalized / span
        return np.clip(normalized, 0.0, 1.0)

    def encode_png(
        self, display_values: np.ndarray, *, width_px: int, height_px: int
    ) -> bytes:
        """Color-map a normalized matrix and encode it as a PNG."""
        rgba = self.colormap()(display_values)
        rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
        # STFT frequencies ascend with row index; image rows descend from top.
        # Flip so low frequencies render at the bottom, matching imshow(origin="lower").
        image = Image.fromarray(np.flipud(rgb), mode="RGB")
        if image.size != (width_px, height_px):
            image = image.resize((width_px, height_px), Image.Resampling.BICUBIC)
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()


OCEAN_DEPTH_COLORS: tuple[tuple[float, str], ...] = (
    (0.0, "#000510"),
    (0.2, "#051530"),
    (0.4, "#0a3050"),
    (0.6, "#108070"),
    (0.8, "#50c8a0"),
    (1.0, "#d0fff0"),
)


LIFTED_OCEAN_COLORS: tuple[tuple[float, str], ...] = (
    (0.0, "#07101d"),
    (0.18, "#0d2441"),
    (0.38, "#145579"),
    (0.62, "#1fa08d"),
    (0.82, "#6ee4b7"),
    (1.0, "#effff7"),
)


class OceanDepthRenderer(TimelineTileRenderer):
    """Compatibility renderer that preserves the original Ocean Depth style."""

    renderer_id = "ocean-depth"
    version = 7

    def colormap(self) -> mcolors.Colormap:
        return colormap_from_stops("ocean_depth", OCEAN_DEPTH_COLORS)


class LiftedOceanRenderer(TimelineTileRenderer):
    """Renderer with a brighter Ocean Depth-derived display mapping."""

    renderer_id = "lifted-ocean"
    version = 1
    display_ceiling = 0.65
    display_gamma = 0.78

    def colormap(self) -> mcolors.Colormap:
        return colormap_from_stops("lifted_ocean", LIFTED_OCEAN_COLORS)

    def display_values(
        self,
        values: np.ndarray,
        *,
        vmin: float,
        vmax: float,
    ) -> np.ndarray:
        del vmax
        normalized = np.nan_to_num(
            values, nan=vmin, posinf=self.display_ceiling, neginf=vmin
        )
        normalized = np.clip((normalized - vmin) / self.display_ceiling, 0.0, 1.0)
        return normalized**self.display_gamma


class PerFrequencyWhitenedOceanRenderer(LiftedOceanRenderer):
    """Lifted Ocean plus per-frequency background whitening detail."""

    renderer_id = "per-frequency-whitened-ocean"
    version = 3
    background_percentile = 25.0
    foreground_percentile = 95.0
    detail_ceiling = 0.86
    detail_gamma = 0.72
    scale_multiplier = 0.55

    def cache_metadata(self, settings) -> dict[str, object]:
        metadata = super().cache_metadata(settings)
        metadata["whitening"] = {
            "background_percentile": self.background_percentile,
            "foreground_percentile": self.foreground_percentile,
            "detail_ceiling": self.detail_ceiling,
            "detail_gamma": self.detail_gamma,
            "scale_multiplier": self.scale_multiplier,
        }
        return metadata

    def display_values(
        self,
        values: np.ndarray,
        *,
        vmin: float,
        vmax: float,
    ) -> np.ndarray:
        base = super().display_values(values, vmin=vmin, vmax=vmax)
        clean = np.nan_to_num(
            values,
            nan=vmin,
            posinf=self.display_ceiling,
            neginf=vmin,
        )
        background = np.percentile(
            clean,
            self.background_percentile,
            axis=1,
            keepdims=True,
        )
        foreground = np.percentile(
            clean,
            self.foreground_percentile,
            axis=1,
            keepdims=True,
        )
        min_scale = max((vmax - vmin) * 0.03, 1e-6)
        scale = np.maximum((foreground - background) * self.scale_multiplier, min_scale)
        detail = np.clip((clean - background) / scale, 0.0, 1.0)
        detail = (detail**self.detail_gamma) * self.detail_ceiling
        return np.maximum(base, detail)


DEFAULT_TIMELINE_RENDERER = PerFrequencyWhitenedOceanRenderer()


def get_ocean_depth_colormap() -> mcolors.LinearSegmentedColormap:
    """Return the compatibility Ocean Depth colormap."""
    return colormap_from_stops("ocean_depth", OCEAN_DEPTH_COLORS)
