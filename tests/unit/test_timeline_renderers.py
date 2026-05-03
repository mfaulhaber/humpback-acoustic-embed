"""Tests for timeline tile renderer implementations."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image


def _luminance(png: bytes) -> float:
    image = Image.open(io.BytesIO(png)).convert("RGB")
    arr = np.asarray(image, dtype=np.float32) / 255.0
    y = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
    return float(np.mean(y))


def test_renderer_ids_and_versions_are_stable():
    from humpback.processing.timeline_renderers import (
        DEFAULT_TIMELINE_RENDERER,
        LiftedOceanRenderer,
        OceanDepthRenderer,
        PerFrequencyWhitenedOceanRenderer,
    )

    assert OceanDepthRenderer.renderer_id == "ocean-depth"
    assert OceanDepthRenderer.version == 7
    assert LiftedOceanRenderer.renderer_id == "lifted-ocean"
    assert LiftedOceanRenderer.version == 1
    assert PerFrequencyWhitenedOceanRenderer.renderer_id == (
        "per-frequency-whitened-ocean"
    )
    assert PerFrequencyWhitenedOceanRenderer.version == 3
    assert isinstance(DEFAULT_TIMELINE_RENDERER, PerFrequencyWhitenedOceanRenderer)


def test_lifted_ocean_display_values_are_monotonic_and_clamped():
    from humpback.processing.timeline_renderers import LiftedOceanRenderer

    renderer = LiftedOceanRenderer()
    values = np.array([[-1.0, 0.0, 0.1, 0.65, 2.0]], dtype=np.float32)

    display = renderer.display_values(values, vmin=0.0, vmax=1.0)

    assert np.all(np.diff(display[0]) >= 0)
    assert display[0, 0] == 0.0
    assert display[0, -1] == 1.0
    assert 0.0 < display[0, 2] < display[0, 3]


def test_lifted_ocean_low_color_is_not_pure_black():
    from humpback.processing.timeline_renderers import LiftedOceanRenderer

    rgb = LiftedOceanRenderer().colormap()(0.0)

    assert rgb[0] > 0.02
    assert rgb[1] > 0.04
    assert rgb[2] > 0.08


def test_lifted_ocean_is_brighter_than_ocean_depth_for_same_matrix():
    from humpback.processing.timeline_renderers import (
        LiftedOceanRenderer,
        OceanDepthRenderer,
    )

    matrix = np.tile(np.linspace(0.0, 0.7, 128, dtype=np.float32), (64, 1))

    ocean_png = OceanDepthRenderer().encode_png(matrix, width_px=128, height_px=64)
    lifted_png = LiftedOceanRenderer().encode_png(
        LiftedOceanRenderer().display_values(matrix, vmin=0.0, vmax=1.0),
        width_px=128,
        height_px=64,
    )

    assert _luminance(lifted_png) > _luminance(ocean_png) * 1.5


def test_renderer_output_dimensions_are_exact():
    from humpback.processing.timeline_renderers import LiftedOceanRenderer

    matrix = np.ones((12, 24), dtype=np.float32) * 0.4
    png = LiftedOceanRenderer().encode_png(matrix, width_px=111, height_px=37)

    image = Image.open(io.BytesIO(png))
    assert image.size == (111, 37)


def test_frequency_orientation_matches_origin_lower():
    from humpback.processing.timeline_renderers import LiftedOceanRenderer

    matrix = np.zeros((4, 4), dtype=np.float32)
    matrix[0, :] = 1.0

    png = LiftedOceanRenderer().encode_png(matrix, width_px=4, height_px=4)
    image = Image.open(io.BytesIO(png)).convert("RGB")
    arr = np.asarray(image, dtype=np.float32)
    luminance = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]

    assert float(luminance[-1].mean()) > float(luminance[0].mean())


def test_per_frequency_whitening_preserves_lifted_ocean_floor():
    from humpback.processing.timeline_renderers import (
        LiftedOceanRenderer,
        PerFrequencyWhitenedOceanRenderer,
    )

    rows = np.linspace(0.08, 0.45, 32, dtype=np.float32)[:, None]
    columns = np.linspace(0.0, 0.08, 96, dtype=np.float32)[None, :]
    matrix = rows + columns

    lifted = LiftedOceanRenderer().display_values(matrix, vmin=0.0, vmax=1.0)
    whitened = PerFrequencyWhitenedOceanRenderer().display_values(
        matrix,
        vmin=0.0,
        vmax=1.0,
    )

    assert np.all(whitened >= lifted)


def test_per_frequency_whitening_boosts_row_local_detail():
    from humpback.processing.timeline_renderers import (
        LiftedOceanRenderer,
        PerFrequencyWhitenedOceanRenderer,
    )

    x = np.linspace(0.0, 1.0, 160, dtype=np.float32)
    matrix = np.tile(np.linspace(0.18, 0.42, 48, dtype=np.float32)[:, None], (1, 160))
    matrix += (0.012 * np.sin(2.0 * np.pi * 12.0 * x))[None, :]
    matrix[:, 70:75] += 0.04

    lifted = LiftedOceanRenderer().display_values(matrix, vmin=0.0, vmax=1.0)
    whitened = PerFrequencyWhitenedOceanRenderer().display_values(
        matrix,
        vmin=0.0,
        vmax=1.0,
    )

    lifted_detail = float(np.std(lifted - lifted.mean(axis=1, keepdims=True)))
    whitened_detail = float(np.std(whitened - whitened.mean(axis=1, keepdims=True)))
    assert whitened_detail > lifted_detail * 1.5
