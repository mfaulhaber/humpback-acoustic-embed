"""Opt-in microbenchmark for timeline tile PNG encoding."""

from __future__ import annotations

import io
import os
import time

import numpy as np
import pytest
from PIL import Image

from humpback.processing.timeline_renderers import LiftedOceanRenderer


def _matplotlib_png(renderer, values, *, width_px: int, height_px: int) -> bytes:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(width_px / 100, height_px / 100), dpi=100)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.axis("off")
    ax.imshow(
        values,
        origin="lower",
        aspect="auto",
        cmap=renderer.colormap(),
        interpolation="bicubic",
        vmin=0.0,
        vmax=1.0,
    )
    buf = io.BytesIO()
    fig.savefig(buf, format="png", pad_inches=0)
    plt.close(fig)
    return buf.getvalue()


@pytest.mark.skipif(
    os.environ.get("HUMPBACK_RUN_PERFORMANCE_TESTS") != "1",
    reason="Set HUMPBACK_RUN_PERFORMANCE_TESTS=1 to run renderer microbenchmarks.",
)
def test_direct_png_encoding_benchmark_against_matplotlib():
    renderer = LiftedOceanRenderer()
    rng = np.random.default_rng(123)
    values = rng.random((256, 512), dtype=np.float32)

    start = time.perf_counter()
    direct = renderer.encode_png(values, width_px=512, height_px=256)
    direct_elapsed = time.perf_counter() - start

    start = time.perf_counter()
    matplotlib = _matplotlib_png(renderer, values, width_px=512, height_px=256)
    matplotlib_elapsed = time.perf_counter() - start

    assert Image.open(io.BytesIO(direct)).size == (512, 256)
    assert Image.open(io.BytesIO(matplotlib)).size == (512, 256)
    assert direct_elapsed < matplotlib_elapsed
