# Timeline Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Pattern Radio-inspired full-screen zoomable spectrogram viewer for hydrophone detection jobs, with confidence heatmap, audio playback, and detection review popovers.

**Architecture:** Backend serves pre-colored PNG spectrogram tiles (Ocean Depth colormap) at 6 discrete zoom levels via a new tile API. A `resolve_timeline_audio()` function maps timeline-absolute positions to HLS cache segments. The React frontend composites tiles on a Canvas 2D element with crossfade zoom transitions, centered-playhead playback, and a clickable confidence minimap.

**Tech Stack:** Python (scipy STFT, matplotlib colormap, PyArrow), FastAPI, SQLite/Alembic, React 18, Canvas 2D API, TanStack Query, Web Audio API, Tailwind CSS.

**Spec:** `docs/specs/2026-03-24-timeline-viewer-design.md`

---

## File Structure

### Backend — New Files

| File | Responsibility |
|------|---------------|
| `src/humpback/processing/timeline_tiles.py` | Ocean Depth colormap, `generate_timeline_tile()` STFT renderer, tile grid math (zoom levels, tile counts, index-to-time mapping) |
| `src/humpback/processing/timeline_audio.py` | `resolve_timeline_audio()` — maps (job, start_sec, duration_sec) to raw audio via HLS timeline manifest |
| `src/humpback/processing/timeline_cache.py` | `TimelineTileCache` — directory-structured tile cache with global FIFO eviction |
| `src/humpback/api/routers/timeline.py` | FastAPI sub-router: `/timeline/tile`, `/timeline/confidence`, `/timeline/audio`, `/timeline/prepare` |
| `alembic/versions/025_timeline_tiles_ready.py` | Add `timeline_tiles_ready` column to `detection_jobs` |
| `tests/unit/test_timeline_tiles.py` | Tests for colormap, tile renderer, tile grid math |
| `tests/unit/test_timeline_audio.py` | Tests for `resolve_timeline_audio()` |
| `tests/unit/test_timeline_cache.py` | Tests for tile cache storage and eviction |
| `tests/integration/test_timeline_api.py` | Tests for all 4 timeline API endpoints |

### Backend — Modified Files

| File | Change |
|------|--------|
| `src/humpback/models/classifier.py` | Add `timeline_tiles_ready` column |
| `src/humpback/storage.py` | Add `timeline_tiles_dir()` helper |
| `src/humpback/config.py` | Add timeline tile settings |
| `src/humpback/api/routers/classifier.py` | Include timeline sub-router |
| `src/humpback/workers/queue.py` | No change needed — `complete_detection_job` stays as-is; tile prep is triggered by worker after completion |
| `src/humpback/workers/classifier_worker.py` | Call tile pre-rendering after hydrophone job completion |

### Frontend — New Files

| File | Responsibility |
|------|---------------|
| `frontend/src/components/timeline/TimelineViewer.tsx` | Page component — layout, state management, keyboard shortcuts |
| `frontend/src/components/timeline/TimelineHeader.tsx` | Back link, job metadata, settings toggles |
| `frontend/src/components/timeline/Minimap.tsx` | 24h confidence heatmap canvas with viewport indicator |
| `frontend/src/components/timeline/SpectrogramViewport.tsx` | Main canvas: tile compositing, playhead, axes, confidence strip |
| `frontend/src/components/timeline/TileCanvas.tsx` | Canvas element: loads tile images, composites, crossfade |
| `frontend/src/components/timeline/PlaybackControls.tsx` | Transport, speed, zoom buttons |
| `frontend/src/components/timeline/ZoomSelector.tsx` | Discrete zoom level buttons |
| `frontend/src/components/timeline/DetectionPopover.tsx` | Click popover with detection row details |
| `frontend/src/components/timeline/DetectionOverlay.tsx` | Optional label overlay rectangles |
| `frontend/src/components/timeline/constants.ts` | Zoom levels, tile dimensions, color scheme constants |
| `frontend/src/hooks/queries/useTimeline.ts` | TanStack Query hooks for timeline endpoints |
| `frontend/e2e/timeline.spec.ts` | Playwright tests |

### Frontend — Modified Files

| File | Change |
|------|--------|
| `frontend/src/App.tsx` | Add `/app/classifier/timeline/:jobId` route |
| `frontend/src/api/client.ts` | Add timeline API functions |
| `frontend/src/api/types.ts` | Add timeline response types |
| `frontend/src/components/classifier/HydrophoneTab.tsx` | Add "Timeline View" button on completed jobs |

---

## Task 1: Database Migration — Add `timeline_tiles_ready` Column

**Files:**
- Create: `alembic/versions/025_timeline_tiles_ready.py`
- Modify: `src/humpback/models/classifier.py:42-73`

- [ ] **Step 1: Write the migration file**

```python
# alembic/versions/025_timeline_tiles_ready.py
"""Add timeline_tiles_ready column to detection_jobs."""

from alembic import op
import sqlalchemy as sa

revision = "025"
down_revision = "024"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("detection_jobs") as batch_op:
        batch_op.add_column(
            sa.Column("timeline_tiles_ready", sa.Boolean(), server_default="0")
        )


def downgrade() -> None:
    with op.batch_alter_table("detection_jobs") as batch_op:
        batch_op.drop_column("timeline_tiles_ready")
```

- [ ] **Step 2: Add column to ORM model**

In `src/humpback/models/classifier.py`, add to the `DetectionJob` class after the `has_positive_labels` column:

```python
timeline_tiles_ready: Mapped[bool] = mapped_column(default=False)
```

- [ ] **Step 3: Run migration**

Run: `uv run alembic upgrade head`
Expected: "Running upgrade 024 -> 025"

- [ ] **Step 4: Verify migration**

Run: `uv run python -c "from humpback.models.classifier import DetectionJob; print(DetectionJob.timeline_tiles_ready)"`
Expected: Prints column attribute without error

- [ ] **Step 5: Commit**

```bash
git add alembic/versions/025_timeline_tiles_ready.py src/humpback/models/classifier.py
git commit -m "feat: add timeline_tiles_ready column to detection_jobs"
```

---

## Task 2: Storage Helpers and Config

**Files:**
- Modify: `src/humpback/storage.py`
- Modify: `src/humpback/config.py`

- [ ] **Step 1: Add timeline_tiles_dir helper to storage.py**

Add after `detection_embeddings_path`:

```python
def timeline_tiles_dir(storage_root: Path, detection_job_id: str) -> Path:
    return detection_dir(storage_root, detection_job_id) / "timeline_tiles"
```

- [ ] **Step 2: Add timeline settings to config.py**

Add after the spectrogram settings block (after line 47):

```python
# Timeline viewer settings
timeline_tile_width_px: int = 512
timeline_tile_height_px: int = 256
timeline_tile_cache_max_items: int = 5000
timeline_dynamic_range_db: float = 80.0
```

- [ ] **Step 3: Verify imports work**

Run: `uv run python -c "from humpback.storage import timeline_tiles_dir; from humpback.config import Settings; s = Settings(); print(s.timeline_tile_width_px)"`
Expected: `512`

- [ ] **Step 4: Commit**

```bash
git add src/humpback/storage.py src/humpback/config.py
git commit -m "feat: add timeline tile storage helper and config settings"
```

---

## Task 3: Ocean Depth Colormap and Tile Renderer

**Files:**
- Create: `src/humpback/processing/timeline_tiles.py`
- Create: `tests/unit/test_timeline_tiles.py`

- [ ] **Step 1: Write failing tests for the colormap and tile grid math**

```python
# tests/unit/test_timeline_tiles.py
"""Tests for timeline tile rendering and grid math."""

import numpy as np
import pytest


def test_ocean_depth_colormap_endpoints():
    """Ocean Depth colormap should map 0.0 to near-black and 1.0 to near-white."""
    from humpback.processing.timeline_tiles import get_ocean_depth_colormap

    cmap = get_ocean_depth_colormap()
    low = cmap(0.0)
    high = cmap(1.0)
    # Low end should be very dark (navy)
    assert low[0] < 0.05 and low[1] < 0.1 and low[2] < 0.1
    # High end should be very bright (seafoam/white)
    assert high[0] > 0.7 and high[1] > 0.9 and high[2] > 0.8


def test_ocean_depth_colormap_midpoint_is_teal():
    """Midpoint should be in the teal range."""
    from humpback.processing.timeline_tiles import get_ocean_depth_colormap

    cmap = get_ocean_depth_colormap()
    mid = cmap(0.6)
    # Teal: low red, moderate-high green, moderate-high blue
    assert mid[0] < 0.3
    assert mid[1] > 0.3
    assert mid[2] > 0.3


ZOOM_LEVELS = ["24h", "6h", "1h", "15m", "5m", "1m"]


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
    """Each zoom level has a fixed tile duration."""
    from humpback.processing.timeline_tiles import tile_duration_sec

    assert tile_duration_sec(zoom_level) == expected_duration


def test_tile_count_24h():
    """24h zoom for a 24h job should produce 1 tile."""
    from humpback.processing.timeline_tiles import tile_count

    count = tile_count("24h", job_duration_sec=86400.0)
    assert count == 1


def test_tile_count_6h():
    """6h zoom for a 24h job should produce 4 tiles."""
    from humpback.processing.timeline_tiles import tile_count

    count = tile_count("6h", job_duration_sec=86400.0)
    assert count == 4


def test_tile_count_1m():
    """1m zoom (10s tiles) for 24h should produce 8640 tiles."""
    from humpback.processing.timeline_tiles import tile_count

    count = tile_count("1m", job_duration_sec=86400.0)
    assert count == 8640


def test_tile_count_partial():
    """Partial last tile should still count."""
    from humpback.processing.timeline_tiles import tile_count

    # 12.5 hours = 45000 sec; 6h tiles = 21600s each → ceil(45000/21600) = 3
    count = tile_count("6h", job_duration_sec=45000.0)
    assert count == 3


def test_tile_time_range():
    """tile_time_range returns correct absolute start/end for a tile index."""
    from humpback.processing.timeline_tiles import tile_time_range

    job_start = 1000000.0  # arbitrary epoch
    start, end = tile_time_range("6h", tile_index=1, job_start_timestamp=job_start)
    assert start == job_start + 21600.0
    assert end == job_start + 43200.0


def test_generate_timeline_tile_returns_png():
    """generate_timeline_tile should return valid PNG bytes."""
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
    # PNG magic bytes
    assert result[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(result) > 100


def test_generate_timeline_tile_custom_freq_range():
    """Different freq ranges should produce different PNGs."""
    from humpback.processing.timeline_tiles import generate_timeline_tile

    sr = 32000
    audio = np.random.randn(sr * 5).astype(np.float32) * 0.01
    png_narrow = generate_timeline_tile(
        audio=audio, sample_rate=sr, freq_min=0, freq_max=1000,
        width_px=512, height_px=256,
    )
    png_wide = generate_timeline_tile(
        audio=audio, sample_rate=sr, freq_min=0, freq_max=8000,
        width_px=512, height_px=256,
    )
    assert png_narrow != png_wide


def test_generate_timeline_tile_silence():
    """Silence should produce a valid (dark) PNG without errors."""
    from humpback.processing.timeline_tiles import generate_timeline_tile

    sr = 32000
    audio = np.zeros(sr * 5, dtype=np.float32)
    result = generate_timeline_tile(
        audio=audio, sample_rate=sr, freq_min=0, freq_max=3000,
        width_px=512, height_px=256,
    )
    assert result[:8] == b"\x89PNG\r\n\x1a\n"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_timeline_tiles.py -v`
Expected: All tests FAIL with `ModuleNotFoundError` or `ImportError`

- [ ] **Step 3: Implement timeline_tiles.py**

```python
# src/humpback/processing/timeline_tiles.py
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
    fig, ax = plt.subplots(
        figsize=(width_px / dpi, height_px / dpi), dpi=dpi
    )
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_timeline_tiles.py -v`
Expected: All tests PASS

- [ ] **Step 5: Run type checker**

Run: `uv run pyright src/humpback/processing/timeline_tiles.py`
Expected: 0 errors

- [ ] **Step 6: Commit**

```bash
git add src/humpback/processing/timeline_tiles.py tests/unit/test_timeline_tiles.py
git commit -m "feat: add Ocean Depth colormap and timeline tile renderer"
```

---

## Task 4: Timeline Tile Cache

**Files:**
- Create: `src/humpback/processing/timeline_cache.py`
- Create: `tests/unit/test_timeline_cache.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_timeline_cache.py
"""Tests for timeline tile disk cache."""

from pathlib import Path

import pytest


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    return tmp_path / "tile_cache"


def test_put_and_get(cache_dir: Path):
    """Stored tile should be retrievable."""
    from humpback.processing.timeline_cache import TimelineTileCache

    cache = TimelineTileCache(cache_dir, max_items=100)
    cache.put("job1", "6h", 0, b"fake-png-data")
    result = cache.get("job1", "6h", 0)
    assert result == b"fake-png-data"


def test_get_miss(cache_dir: Path):
    """Missing tile should return None."""
    from humpback.processing.timeline_cache import TimelineTileCache

    cache = TimelineTileCache(cache_dir, max_items=100)
    assert cache.get("job1", "6h", 99) is None


def test_directory_structure(cache_dir: Path):
    """Tiles should be stored in {cache_dir}/{job_id}/{zoom}/tile_{index:04d}.png."""
    from humpback.processing.timeline_cache import TimelineTileCache

    cache = TimelineTileCache(cache_dir, max_items=100)
    cache.put("job-abc", "1h", 5, b"data")
    expected = cache_dir / "job-abc" / "1h" / "tile_0005.png"
    assert expected.is_file()
    assert expected.read_bytes() == b"data"


def test_fifo_eviction(cache_dir: Path):
    """Oldest tiles should be evicted when global count exceeds max."""
    from humpback.processing.timeline_cache import TimelineTileCache

    cache = TimelineTileCache(cache_dir, max_items=3)
    cache.put("j1", "24h", 0, b"a")
    cache.put("j1", "6h", 0, b"b")
    cache.put("j1", "6h", 1, b"c")
    # All 3 present
    assert cache.get("j1", "24h", 0) is not None
    # Adding 4th should evict oldest
    cache.put("j2", "24h", 0, b"d")
    assert cache.get("j1", "24h", 0) is None
    assert cache.get("j2", "24h", 0) == b"d"


def test_put_is_atomic(cache_dir: Path):
    """No .tmp files should remain after put."""
    from humpback.processing.timeline_cache import TimelineTileCache

    cache = TimelineTileCache(cache_dir, max_items=100)
    cache.put("j1", "1m", 0, b"data")
    tmp_files = list(cache_dir.rglob("*.tmp"))
    assert tmp_files == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_timeline_cache.py -v`
Expected: All FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement timeline_cache.py**

```python
# src/humpback/processing/timeline_cache.py
"""Directory-structured disk cache for timeline spectrogram tiles.

Tiles are stored as: {cache_dir}/{job_id}/{zoom_level}/tile_{index:04d}.png
Global FIFO eviction (mtime-based) caps total tile count across all jobs.
"""

import os
from pathlib import Path


class TimelineTileCache:
    """Disk-backed tile cache with per-job/zoom directory structure."""

    def __init__(self, cache_dir: Path, max_items: int = 5000) -> None:
        self.cache_dir = cache_dir
        self.max_items = max_items

    def _tile_path(self, job_id: str, zoom_level: str, tile_index: int) -> Path:
        return self.cache_dir / job_id / zoom_level / f"tile_{tile_index:04d}.png"

    def get(self, job_id: str, zoom_level: str, tile_index: int) -> bytes | None:
        p = self._tile_path(job_id, zoom_level, tile_index)
        if p.is_file():
            return p.read_bytes()
        return None

    def put(
        self, job_id: str, zoom_level: str, tile_index: int, data: bytes
    ) -> None:
        p = self._tile_path(job_id, zoom_level, tile_index)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, p)
        self._evict()

    def _evict(self) -> None:
        """Remove oldest tiles globally when count exceeds max_items."""
        files = sorted(self.cache_dir.rglob("*.png"), key=lambda f: f.stat().st_mtime)
        excess = len(files) - self.max_items
        if excess > 0:
            for f in files[:excess]:
                f.unlink(missing_ok=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_timeline_cache.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/humpback/processing/timeline_cache.py tests/unit/test_timeline_cache.py
git commit -m "feat: add directory-structured timeline tile cache"
```

---

## Task 5: Timeline Audio Resolution

**Files:**
- Create: `src/humpback/processing/timeline_audio.py`
- Create: `tests/unit/test_timeline_audio.py`

This is the core function that maps timeline-absolute positions to raw audio via the HLS cache.

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_timeline_audio.py
"""Tests for timeline audio resolution from HLS cache."""

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_provider():
    """Mock ArchiveProvider that returns sine wave audio for any segment."""
    provider = MagicMock()
    provider.kind = "orcasound_hls"
    return provider


def test_resolve_returns_correct_duration():
    """Resolved audio length should match requested duration at target sample rate."""
    from humpback.processing.timeline_audio import resolve_timeline_audio

    sr = 32000
    duration = 10.0
    # Create a mock that simulates the full resolution pipeline
    fake_audio = np.zeros(int(sr * duration), dtype=np.float32)

    with patch(
        "humpback.processing.timeline_audio._resolve_audio_from_hls_cache"
    ) as mock_resolve:
        mock_resolve.return_value = fake_audio
        result = resolve_timeline_audio(
            hydrophone_id="rpi_north_sjc",
            local_cache_path="/fake/cache",
            job_start_timestamp=1000000.0,
            job_end_timestamp=1086400.0,
            start_sec=1000050.0,
            duration_sec=duration,
            target_sr=sr,
        )
    assert len(result) == int(sr * duration)


def test_resolve_silence_for_gap():
    """When HLS cache has no segments for a range, return silence."""
    from humpback.processing.timeline_audio import resolve_timeline_audio

    sr = 32000
    duration = 5.0

    with patch(
        "humpback.processing.timeline_audio._resolve_audio_from_hls_cache"
    ) as mock_resolve:
        mock_resolve.return_value = np.zeros(int(sr * duration), dtype=np.float32)
        result = resolve_timeline_audio(
            hydrophone_id="rpi_north_sjc",
            local_cache_path="/fake/cache",
            job_start_timestamp=1000000.0,
            job_end_timestamp=1086400.0,
            start_sec=1000050.0,
            duration_sec=duration,
            target_sr=sr,
        )
    assert len(result) == int(sr * duration)
    # All zeros is valid (silence for gap)
    assert result.dtype == np.float32


def test_resolve_clamps_to_job_bounds():
    """Requested range beyond job bounds should be clamped."""
    from humpback.processing.timeline_audio import resolve_timeline_audio

    sr = 32000
    job_start = 1000000.0
    job_end = 1086400.0

    with patch(
        "humpback.processing.timeline_audio._resolve_audio_from_hls_cache"
    ) as mock_resolve:
        mock_resolve.return_value = np.zeros(sr * 5, dtype=np.float32)
        # Request starts before job
        result = resolve_timeline_audio(
            hydrophone_id="rpi_north_sjc",
            local_cache_path="/fake/cache",
            job_start_timestamp=job_start,
            job_end_timestamp=job_end,
            start_sec=job_start - 100.0,
            duration_sec=5.0,
            target_sr=sr,
        )
    # Should return valid audio (clamped) without error
    assert len(result) == sr * 5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_timeline_audio.py -v`
Expected: All FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement timeline_audio.py**

```python
# src/humpback/processing/timeline_audio.py
"""Resolve audio from HLS cache for arbitrary timeline-absolute positions.

Maps (hydrophone_id, start_epoch, duration) to a continuous float32 audio
array by reading overlapping HLS segments from the local cache directory
and stitching them together.  Gaps are filled with silence.
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def resolve_timeline_audio(
    *,
    hydrophone_id: str,
    local_cache_path: str,
    job_start_timestamp: float,
    job_end_timestamp: float,
    start_sec: float,
    duration_sec: float,
    target_sr: int = 32000,
) -> np.ndarray:
    """Return audio samples for an arbitrary timeline-absolute range.

    Parameters
    ----------
    hydrophone_id : Orcasound hydrophone ID.
    local_cache_path : Root path of the local HLS segment cache.
    job_start_timestamp, job_end_timestamp : UTC epoch bounds of the detection job.
    start_sec : UTC epoch start of the requested range.
    duration_sec : Length in seconds of the requested range.
    target_sr : Target sample rate.

    Returns
    -------
    1-D float32 array of length ``int(target_sr * duration_sec)``.
    Gaps in the cache are filled with silence.
    """
    # Clamp to job bounds
    effective_start = max(start_sec, job_start_timestamp)
    effective_end = min(start_sec + duration_sec, job_end_timestamp)
    effective_duration = max(0.0, effective_end - effective_start)

    n_samples = int(target_sr * duration_sec)

    if effective_duration <= 0:
        return np.zeros(n_samples, dtype=np.float32)

    audio = _resolve_audio_from_hls_cache(
        hydrophone_id=hydrophone_id,
        local_cache_path=local_cache_path,
        start_sec=effective_start,
        duration_sec=effective_duration,
        target_sr=target_sr,
    )

    # Ensure exact length
    if len(audio) < n_samples:
        audio = np.pad(audio, (0, n_samples - len(audio)))
    elif len(audio) > n_samples:
        audio = audio[:n_samples]

    return audio.astype(np.float32)


def _resolve_audio_from_hls_cache(
    *,
    hydrophone_id: str,
    local_cache_path: str,
    start_sec: float,
    duration_sec: float,
    target_sr: int,
) -> np.ndarray:
    """Resolve audio from HLS .ts segments in the local cache.

    This function discovers HLS folders and segments overlapping the requested
    range, decodes them, and concatenates with silence for gaps.
    """
    from humpback.classifier.s3_stream import (
        build_hls_timeline_for_range,
        decode_segments_to_audio,
    )

    n_samples = int(target_sr * duration_sec)
    end_sec = start_sec + duration_sec

    try:
        timeline = build_hls_timeline_for_range(
            hydrophone_id=hydrophone_id,
            local_cache_path=local_cache_path,
            start_epoch=start_sec,
            end_epoch=end_sec,
        )
        if not timeline:
            logger.debug(
                "No HLS segments found for %s [%.0f–%.0f]",
                hydrophone_id, start_sec, end_sec,
            )
            return np.zeros(n_samples, dtype=np.float32)

        audio = decode_segments_to_audio(
            timeline=timeline,
            start_epoch=start_sec,
            end_epoch=end_sec,
            target_sr=target_sr,
        )
        return audio

    except Exception:
        logger.exception(
            "Failed to resolve HLS audio for %s [%.0f–%.0f]",
            hydrophone_id, start_sec, end_sec,
        )
        return np.zeros(n_samples, dtype=np.float32)
```

- [ ] **Step 3b: Extract reusable HLS timeline functions into s3_stream.py**

The `build_hls_timeline_for_range` and `decode_segments_to_audio` functions called by `timeline_audio.py` must be created in `src/humpback/classifier/s3_stream.py`. These extract and generalize the existing HLS timeline assembly logic.

**Read first:** Study the existing functions in `s3_stream.py` — `_sort_segment_keys()`, `_parse_playlist_segments()`, `_ordered_folder_segments()` — and how the detection worker calls them to build segment timelines.

**Create `build_hls_timeline_for_range`:**

```python
def build_hls_timeline_for_range(
    *,
    hydrophone_id: str,
    local_cache_path: str,
    start_epoch: float,
    end_epoch: float,
) -> list[tuple[str, float, float]]:
    """Build an ordered timeline of HLS segments overlapping [start_epoch, end_epoch].

    Returns a list of (segment_path, segment_start_epoch, segment_duration_sec)
    tuples, ordered by time.  Uses the local HLS cache only (no S3 fallback).

    This function extracts and generalizes the timeline assembly logic used by
    the detection worker's segment processing loop.
    """
    # 1. Discover HLS folders in local_cache_path for the hydrophone
    # 2. Use _ordered_folder_segments() / _parse_playlist_segments() to get
    #    segment metadata with absolute timestamps
    # 3. Filter to segments overlapping [start_epoch, end_epoch]
    # 4. Return ordered list of (segment_path, start_epoch, duration) tuples
    ...
```

**Create `decode_segments_to_audio`:**

```python
def decode_segments_to_audio(
    *,
    timeline: list[tuple[str, float, float]],
    start_epoch: float,
    end_epoch: float,
    target_sr: int,
) -> np.ndarray:
    """Decode HLS segments and stitch into a continuous audio array.

    Gaps between segments are filled with silence.  The output is trimmed
    to exactly [start_epoch, end_epoch] at the target sample rate.
    """
    # 1. Allocate output array of zeros: int((end_epoch - start_epoch) * target_sr)
    # 2. For each segment in timeline:
    #    a. Decode .ts segment to float32 audio (use existing decode_audio or ffmpeg)
    #    b. Calculate where this segment's audio maps into the output array
    #    c. Copy decoded samples into the correct position (overwriting silence)
    # 3. Return the output array
    ...
```

The exact implementation will depend on the existing patterns in `s3_stream.py`. The implementer should read the file thoroughly before writing these functions. The key is to **reuse** existing segment discovery and ordering code, not rewrite it.

Also modify `src/humpback/classifier/s3_stream.py` to add these as public functions.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_timeline_audio.py -v`
Expected: All tests PASS

- [ ] **Step 5: Run type checker**

Run: `uv run pyright src/humpback/processing/timeline_audio.py`
Expected: 0 errors

- [ ] **Step 6: Commit**

```bash
git add src/humpback/processing/timeline_audio.py tests/unit/test_timeline_audio.py
git commit -m "feat: add timeline audio resolution from HLS cache"
```

---

## Task 6: Timeline API Endpoints

**Files:**
- Create: `src/humpback/api/routers/timeline.py`
- Modify: `src/humpback/api/routers/classifier.py`
- Create: `tests/integration/test_timeline_api.py`

- [ ] **Step 1: Write failing integration tests**

```python
# tests/integration/test_timeline_api.py
"""Integration tests for timeline viewer API endpoints."""

import uuid
from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np
import pytest
from sqlalchemy import insert

from humpback.models.classifier import DetectionJob


@pytest.fixture
def hydrophone_job_id():
    return str(uuid.uuid4())


@pytest.fixture
async def completed_hydrophone_job(session, settings, hydrophone_job_id):
    """Create a completed hydrophone detection job with fixture diagnostics."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from humpback.storage import detection_diagnostics_path

    job_start = 1710000000.0  # arbitrary epoch
    job_end = job_start + 86400.0  # 24 hours

    await session.execute(
        insert(DetectionJob).values(
            id=hydrophone_job_id,
            status="complete",
            hydrophone_id="rpi_north_sjc",
            hydrophone_name="rpi_north_sjc",
            start_timestamp=job_start,
            end_timestamp=job_end,
            local_cache_path=str(settings.storage_root / "fake_cache"),
            timeline_tiles_ready=False,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )
    )
    await session.commit()

    # Create fixture diagnostics parquet
    diag_path = detection_diagnostics_path(settings.storage_root, hydrophone_job_id)
    diag_path.parent.mkdir(parents=True, exist_ok=True)
    n_windows = 100
    table = pa.table({
        "filename": ["seg_0.ts"] * n_windows,
        "window_index": list(range(n_windows)),
        "offset_sec": [float(i * 5) for i in range(n_windows)],
        "end_sec": [float(i * 5 + 5) for i in range(n_windows)],
        "confidence": [float(i % 10) / 10.0 for i in range(n_windows)],
        "is_overlapped": [False] * n_windows,
        "overlap_sec": [0.0] * n_windows,
    })
    pq.write_table(table, diag_path)

    return hydrophone_job_id


async def test_tile_endpoint_returns_png(client, completed_hydrophone_job, settings):
    """GET /timeline/tile should return PNG for valid params."""
    job_id = completed_hydrophone_job

    fake_audio = np.random.randn(32000 * 10).astype(np.float32) * 0.01

    with patch(
        "humpback.api.routers.timeline.resolve_timeline_audio"
    ) as mock_resolve:
        mock_resolve.return_value = fake_audio
        resp = await client.get(
            f"/classifier/detection-jobs/{job_id}/timeline/tile",
            params={"zoom_level": "24h", "tile_index": 0},
        )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert resp.content[:8] == b"\x89PNG\r\n\x1a\n"


async def test_tile_endpoint_404_bad_index(client, completed_hydrophone_job):
    """Out-of-range tile index should return 404."""
    job_id = completed_hydrophone_job
    resp = await client.get(
        f"/classifier/detection-jobs/{job_id}/timeline/tile",
        params={"zoom_level": "24h", "tile_index": 999},
    )
    assert resp.status_code == 404


async def test_tile_endpoint_400_bad_zoom(client, completed_hydrophone_job):
    """Invalid zoom level should return 400 or 422."""
    job_id = completed_hydrophone_job
    resp = await client.get(
        f"/classifier/detection-jobs/{job_id}/timeline/tile",
        params={"zoom_level": "3h", "tile_index": 0},
    )
    assert resp.status_code in (400, 422)


async def test_confidence_endpoint(client, completed_hydrophone_job):
    """GET /timeline/confidence should return JSON with scores array."""
    job_id = completed_hydrophone_job
    resp = await client.get(
        f"/classifier/detection-jobs/{job_id}/timeline/confidence",
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "scores" in data
    assert "window_sec" in data
    assert "start_timestamp" in data
    assert isinstance(data["scores"], list)
    assert len(data["scores"]) > 0


async def test_audio_endpoint_returns_wav(client, completed_hydrophone_job):
    """GET /timeline/audio should return WAV audio."""
    job_id = completed_hydrophone_job
    job_start = 1710000000.0

    fake_audio = np.random.randn(32000 * 5).astype(np.float32) * 0.01

    with patch(
        "humpback.api.routers.timeline.resolve_timeline_audio"
    ) as mock_resolve:
        mock_resolve.return_value = fake_audio
        resp = await client.get(
            f"/classifier/detection-jobs/{job_id}/timeline/audio",
            params={"start_sec": job_start + 100, "duration_sec": 5.0},
        )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/wav"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/integration/test_timeline_api.py -v`
Expected: All FAIL

- [ ] **Step 3: Implement the timeline router**

```python
# src/humpback/api/routers/timeline.py
"""Timeline viewer API endpoints.

Sub-router mounted under /classifier/detection-jobs/{job_id}/timeline.
Provides spectrogram tiles, confidence data, and audio for the zoomable viewer.
"""

import asyncio
import io
import logging
from enum import Enum
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select

from humpback.api.deps import SessionDep, SettingsDep
from humpback.models.classifier import DetectionJob
from humpback.processing.timeline_audio import resolve_timeline_audio
from humpback.processing.timeline_cache import TimelineTileCache
from humpback.processing.timeline_tiles import (
    ZOOM_LEVELS,
    generate_timeline_tile,
    tile_count,
    tile_time_range,
)
from humpback.storage import detection_diagnostics_path, timeline_tiles_dir

logger = logging.getLogger(__name__)

router = APIRouter()


class ZoomLevel(str, Enum):
    h24 = "24h"
    h6 = "6h"
    h1 = "1h"
    m15 = "15m"
    m5 = "5m"
    m1 = "1m"


class ConfidenceResponse(BaseModel):
    window_sec: float
    scores: list[Optional[float]]
    start_timestamp: float
    end_timestamp: float


async def _get_hydrophone_job(session: SessionDep, job_id: str) -> DetectionJob:
    """Fetch a hydrophone detection job or raise 404."""
    result = await session.execute(
        select(DetectionJob).where(DetectionJob.id == job_id)
    )
    job = result.scalar_one_or_none()
    if job is None:
        raise HTTPException(404, f"Detection job {job_id} not found")
    if job.hydrophone_id is None:
        raise HTTPException(400, "Timeline viewer is only available for hydrophone jobs")
    if job.start_timestamp is None or job.end_timestamp is None:
        raise HTTPException(400, "Job missing start/end timestamps")
    return job


def _get_tile_cache(settings: SettingsDep) -> TimelineTileCache:
    cache_dir = settings.storage_root / "timeline_tile_cache"
    return TimelineTileCache(cache_dir, max_items=settings.timeline_tile_cache_max_items)


@router.get("/tile")
async def get_timeline_tile(
    job_id: str,
    session: SessionDep,
    settings: SettingsDep,
    zoom_level: ZoomLevel = Query(...),
    tile_index: int = Query(..., ge=0),
    freq_min: int = Query(0, ge=0),
    freq_max: int = Query(3000, gt=0),
):
    """Return a spectrogram tile PNG for the given zoom level and index."""
    job = await _get_hydrophone_job(session, job_id)

    job_duration = job.end_timestamp - job.start_timestamp
    max_tiles = tile_count(zoom_level.value, job_duration_sec=job_duration)
    if tile_index >= max_tiles:
        raise HTTPException(404, f"Tile index {tile_index} out of range (max {max_tiles - 1})")

    # Check tile cache
    cache = _get_tile_cache(settings)
    cache_key_suffix = f"f{freq_min}-{freq_max}"
    cached = cache.get(f"{job_id}_{cache_key_suffix}", zoom_level.value, tile_index)
    if cached is not None:
        from fastapi.responses import Response
        return Response(content=cached, media_type="image/png")

    # Resolve audio for this tile's time range
    start_epoch, end_epoch = tile_time_range(
        zoom_level.value,
        tile_index=tile_index,
        job_start_timestamp=job.start_timestamp,
    )
    duration = end_epoch - start_epoch

    audio = await asyncio.to_thread(
        resolve_timeline_audio,
        hydrophone_id=job.hydrophone_id,
        local_cache_path=job.local_cache_path or "",
        job_start_timestamp=job.start_timestamp,
        job_end_timestamp=job.end_timestamp,
        start_sec=start_epoch,
        duration_sec=duration,
        target_sr=settings.target_sample_rate,
    )

    # Render tile
    png_bytes = await asyncio.to_thread(
        generate_timeline_tile,
        audio=audio,
        sample_rate=settings.target_sample_rate,
        freq_min=freq_min,
        freq_max=freq_max,
        n_fft=2048,
        hop_length=settings.spectrogram_hop_length,
        dynamic_range_db=settings.timeline_dynamic_range_db,
        width_px=settings.timeline_tile_width_px,
        height_px=settings.timeline_tile_height_px,
    )

    # Cache the result
    cache.put(f"{job_id}_{cache_key_suffix}", zoom_level.value, tile_index, png_bytes)

    from fastapi.responses import Response
    return Response(content=png_bytes, media_type="image/png")


@router.get("/confidence")
async def get_timeline_confidence(
    job_id: str,
    session: SessionDep,
    settings: SettingsDep,
) -> ConfidenceResponse:
    """Return confidence scores as a timeline-ordered array."""
    job = await _get_hydrophone_job(session, job_id)

    diag_path = detection_diagnostics_path(settings.storage_root, job_id)
    if not diag_path.is_file():
        raise HTTPException(404, "Window diagnostics not found for this job")

    def _build_confidence():
        """Read diagnostics once, convert per-file offsets to timeline-absolute,
        and return (window_sec, sorted scores)."""
        import pyarrow.parquet as pq
        from humpback.classifier.s3_stream import (
            build_hls_timeline_for_range,
        )

        table = pq.read_table(diag_path)
        filenames = table.column("filename").to_pylist()
        offsets = table.column("offset_sec").to_pylist()
        confidences = table.column("confidence").to_pylist()

        # For hydrophone jobs, offset_sec is per-file (relative to each HLS
        # segment).  We need the HLS timeline manifest to convert (filename,
        # offset_sec) → timeline-absolute position.
        #
        # Build a mapping: filename → segment_start_epoch from the HLS
        # timeline.  For each row, absolute_time = segment_start + offset_sec.
        timeline = build_hls_timeline_for_range(
            hydrophone_id=job.hydrophone_id,
            local_cache_path=job.local_cache_path or "",
            start_epoch=job.start_timestamp,
            end_epoch=job.end_timestamp,
        )
        seg_start_map: dict[str, float] = {}
        for seg_path, seg_start, _seg_dur in (timeline or []):
            seg_name = seg_path.rsplit("/", 1)[-1] if "/" in str(seg_path) else str(seg_path)
            seg_start_map[seg_name] = seg_start

        abs_pairs: list[tuple[float, float]] = []
        for fname, offset, conf in zip(filenames, offsets, confidences):
            seg_start = seg_start_map.get(fname)
            if seg_start is not None:
                abs_pairs.append((seg_start + offset, conf))
            else:
                # Fallback: treat offset as already absolute (legacy or
                # single-file jobs)
                abs_pairs.append((offset, conf))

        abs_pairs.sort(key=lambda x: x[0])

        # Infer window_sec from first two distinct absolute offsets
        window_sec = 5.0
        if len(abs_pairs) >= 2:
            delta = abs_pairs[1][0] - abs_pairs[0][0]
            if delta > 0:
                window_sec = delta

        scores = [c for _, c in abs_pairs]
        return window_sec, scores

    window_sec, scores = await asyncio.to_thread(_build_confidence)

    return ConfidenceResponse(
        window_sec=window_sec,
        scores=scores,
        start_timestamp=job.start_timestamp,
        end_timestamp=job.end_timestamp,
    )


@router.get("/audio")
async def get_timeline_audio(
    job_id: str,
    session: SessionDep,
    settings: SettingsDep,
    start_sec: float = Query(...),
    duration_sec: float = Query(30.0, gt=0, le=120),
):
    """Return WAV audio for an arbitrary timeline-absolute position."""
    job = await _get_hydrophone_job(session, job_id)

    audio = await asyncio.to_thread(
        resolve_timeline_audio,
        hydrophone_id=job.hydrophone_id,
        local_cache_path=job.local_cache_path or "",
        job_start_timestamp=job.start_timestamp,
        job_end_timestamp=job.end_timestamp,
        start_sec=start_sec,
        duration_sec=duration_sec,
        target_sr=settings.target_sample_rate,
    )

    # Encode to WAV
    import wave

    buf = io.BytesIO()
    sr = settings.target_sample_rate
    pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    buf.seek(0)

    from fastapi.responses import Response
    return Response(content=buf.read(), media_type="audio/wav")


@router.post("/prepare")
async def prepare_timeline_tiles(
    job_id: str,
    session: SessionDep,
    settings: SettingsDep,
):
    """Trigger pre-rendering of coarse zoom level tiles (24h + 6h)."""
    job = await _get_hydrophone_job(session, job_id)

    if job.timeline_tiles_ready:
        return {"status": "already_ready"}

    # Pre-render coarse tiles in a background thread (plain sync function)
    def _render_coarse():
        cache = _get_tile_cache(settings)
        for zoom in ("24h", "6h"):
            job_duration = job.end_timestamp - job.start_timestamp
            n_tiles = tile_count(zoom, job_duration_sec=job_duration)
            for idx in range(n_tiles):
                if cache.get(f"{job_id}_f0-3000", zoom, idx) is not None:
                    continue
                start_epoch, end_epoch = tile_time_range(
                    zoom, tile_index=idx, job_start_timestamp=job.start_timestamp,
                )
                audio = resolve_timeline_audio(
                    hydrophone_id=job.hydrophone_id,
                    local_cache_path=job.local_cache_path or "",
                    job_start_timestamp=job.start_timestamp,
                    job_end_timestamp=job.end_timestamp,
                    start_sec=start_epoch,
                    duration_sec=end_epoch - start_epoch,
                    target_sr=settings.target_sample_rate,
                )
                png = generate_timeline_tile(
                    audio=audio,
                    sample_rate=settings.target_sample_rate,
                    freq_min=0,
                    freq_max=3000,
                    n_fft=2048,
                    hop_length=settings.spectrogram_hop_length,
                    dynamic_range_db=settings.timeline_dynamic_range_db,
                    width_px=settings.timeline_tile_width_px,
                    height_px=settings.timeline_tile_height_px,
                )
                cache.put(f"{job_id}_f0-3000", zoom, idx, png)

    await asyncio.to_thread(_render_coarse)

    # Mark ready
    from sqlalchemy import update
    from datetime import datetime, timezone

    await session.execute(
        update(DetectionJob)
        .where(DetectionJob.id == job_id)
        .values(
            timeline_tiles_ready=True,
            updated_at=datetime.now(timezone.utc),
        )
    )
    await session.commit()

    return {"status": "prepared"}
```

- [ ] **Step 4: Mount the sub-router in classifier.py**

In `src/humpback/api/routers/classifier.py`, add the import and include:

```python
from humpback.api.routers.timeline import router as timeline_router

# Mount timeline sub-router (add near other router configuration)
router.include_router(
    timeline_router,
    prefix="/detection-jobs/{job_id}/timeline",
    tags=["timeline"],
)
```

- [ ] **Step 5: Run integration tests**

Run: `uv run pytest tests/integration/test_timeline_api.py -v`
Expected: All tests PASS

- [ ] **Step 6: Run type checker**

Run: `uv run pyright src/humpback/api/routers/timeline.py`
Expected: 0 errors

- [ ] **Step 7: Commit**

```bash
git add src/humpback/api/routers/timeline.py src/humpback/api/routers/classifier.py tests/integration/test_timeline_api.py
git commit -m "feat: add timeline viewer API endpoints (tile, confidence, audio, prepare)"
```

---

## Task 7: Worker Integration — Auto-Prepare Tiles on Job Completion

**Files:**
- Modify: `src/humpback/workers/classifier_worker.py`

- [ ] **Step 1: Read the hydrophone job completion code**

Read `src/humpback/workers/classifier_worker.py` around line 1309 where `complete_detection_job` is called to understand the surrounding context and error handling pattern.

- [ ] **Step 2: Add tile pre-rendering after job completion**

After the `await complete_detection_job(session, job.id)` call for hydrophone jobs (line ~1309), add tile pre-rendering. The pattern should be:

```python
# After: await complete_detection_job(session, job.id)

# Pre-render coarse timeline tiles (best-effort, don't fail the job)
try:
    from humpback.processing.timeline_tiles import (
        generate_timeline_tile,
        tile_count,
        tile_time_range,
    )
    from humpback.processing.timeline_audio import resolve_timeline_audio
    from humpback.processing.timeline_cache import TimelineTileCache

    tile_cache = TimelineTileCache(
        settings.storage_root / "timeline_tile_cache",
        max_items=settings.timeline_tile_cache_max_items,
    )
    for zoom in ("24h", "6h"):
        job_duration = job.end_timestamp - job.start_timestamp
        n_tiles = tile_count(zoom, job_duration_sec=job_duration)
        for idx in range(n_tiles):
            start_epoch, end_epoch = tile_time_range(
                zoom, tile_index=idx, job_start_timestamp=job.start_timestamp,
            )
            audio = resolve_timeline_audio(
                hydrophone_id=job.hydrophone_id,
                local_cache_path=job.local_cache_path or "",
                job_start_timestamp=job.start_timestamp,
                job_end_timestamp=job.end_timestamp,
                start_sec=start_epoch,
                duration_sec=end_epoch - start_epoch,
                target_sr=settings.target_sample_rate,
            )
            png = generate_timeline_tile(
                audio=audio,
                sample_rate=settings.target_sample_rate,
                freq_min=0,
                freq_max=3000,
                hop_length=settings.spectrogram_hop_length,
                dynamic_range_db=settings.timeline_dynamic_range_db,
                width_px=settings.timeline_tile_width_px,
                height_px=settings.timeline_tile_height_px,
            )
            tile_cache.put(f"{job.id}_f0-3000", zoom, idx, png)

    # Mark tiles ready
    from sqlalchemy import update
    await session.execute(
        update(DetectionJob)
        .where(DetectionJob.id == job.id)
        .values(timeline_tiles_ready=True, updated_at=datetime.now(timezone.utc))
    )
    await session.commit()
    logger.info("Pre-rendered timeline tiles for job %s", job.id)
except Exception:
    logger.warning("Failed to pre-render timeline tiles for job %s", job.id, exc_info=True)
```

- [ ] **Step 3: Run full test suite to verify no regressions**

Run: `uv run pytest tests/ -x -q`
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add src/humpback/workers/classifier_worker.py
git commit -m "feat: auto-prepare timeline tiles on hydrophone job completion"
```

---

## Task 8: Frontend Types and API Client

**Files:**
- Modify: `frontend/src/api/types.ts`
- Modify: `frontend/src/api/client.ts`
- Create: `frontend/src/hooks/queries/useTimeline.ts`
- Create: `frontend/src/components/timeline/constants.ts`

- [ ] **Step 1: Add types to types.ts**

Add to `frontend/src/api/types.ts`:

```typescript
// Timeline viewer types

export type ZoomLevel = "24h" | "6h" | "1h" | "15m" | "5m" | "1m";

export interface TimelineConfidenceResponse {
  window_sec: number;
  scores: (number | null)[];
  start_timestamp: number;
  end_timestamp: number;
}
```

- [ ] **Step 2: Add API functions to client.ts**

Add to `frontend/src/api/client.ts`:

```typescript
// Timeline viewer API

export function timelineTileUrl(
  jobId: string,
  zoomLevel: string,
  tileIndex: number,
  freqMin = 0,
  freqMax = 3000,
): string {
  return `/classifier/detection-jobs/${jobId}/timeline/tile?zoom_level=${zoomLevel}&tile_index=${tileIndex}&freq_min=${freqMin}&freq_max=${freqMax}`;
}

export function timelineAudioUrl(
  jobId: string,
  startSec: number,
  durationSec = 30,
): string {
  return `/classifier/detection-jobs/${jobId}/timeline/audio?start_sec=${startSec}&duration_sec=${durationSec}`;
}

export const fetchTimelineConfidence = (jobId: string) =>
  api<TimelineConfidenceResponse>(
    `/classifier/detection-jobs/${jobId}/timeline/confidence`,
  );

export const prepareTimelineTiles = (jobId: string) =>
  post<{ status: string }>(
    `/classifier/detection-jobs/${jobId}/timeline/prepare`,
    {},
  );
```

- [ ] **Step 3: Create constants.ts**

```typescript
// frontend/src/components/timeline/constants.ts
import type { ZoomLevel } from "@/api/types";

export const ZOOM_LEVELS: ZoomLevel[] = ["24h", "6h", "1h", "15m", "5m", "1m"];

/** Tile duration in seconds for each zoom level */
export const TILE_DURATION: Record<ZoomLevel, number> = {
  "24h": 86400,
  "6h": 21600,
  "1h": 600,
  "15m": 150,
  "5m": 50,
  "1m": 10,
};

/** Viewport span in seconds for each zoom level */
export const VIEWPORT_SPAN: Record<ZoomLevel, number> = {
  "24h": 86400,
  "6h": 21600,
  "1h": 3600,
  "15m": 900,
  "5m": 300,
  "1m": 60,
};

export const TILE_WIDTH_PX = 512;
export const TILE_HEIGHT_PX = 256;

/** Pixels per second at each zoom level (for canvas layout) */
export const PIXELS_PER_SEC: Record<ZoomLevel, number> = {
  "24h": TILE_WIDTH_PX / 86400,
  "6h": TILE_WIDTH_PX / 21600,
  "1h": TILE_WIDTH_PX / 600,
  "15m": TILE_WIDTH_PX / 150,
  "5m": TILE_WIDTH_PX / 50,
  "1m": TILE_WIDTH_PX / 10,
};

// Ocean Depth color scheme — UI chrome colors
export const COLORS = {
  bg: "#060d14",
  bgDark: "#040810",
  text: "#a0c8c0",
  textMuted: "#3a6a60",
  textBright: "#5a9a80",
  accent: "#70e0c0",
  accentDim: "#40a080",
  border: "#1a3040",
  headerBg: "#0a1520",
  // Label overlay colors
  labelHumpback: "rgba(64, 224, 192, 0.15)",
  labelOrca: "rgba(224, 176, 64, 0.15)",
  labelShip: "rgba(224, 64, 64, 0.15)",
  labelBackground: "rgba(160, 160, 160, 0.15)",
} as const;

// Confidence heatmap gradient stops (green-yellow)
export const CONFIDENCE_GRADIENT = [
  [0.0, "#0a1a0a"],
  [0.3, "#2a5a20"],
  [0.5, "#60a020"],
  [0.7, "#a0c820"],
  [0.85, "#d0e040"],
  [1.0, "#f0f060"],
] as const;

export const CROSSFADE_DURATION_MS = 300;
export const AUDIO_PREFETCH_SEC = 30;
export const FREQ_AXIS_WIDTH_PX = 44;
```

- [ ] **Step 4: Create TanStack Query hooks**

```typescript
// frontend/src/hooks/queries/useTimeline.ts
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  fetchTimelineConfidence,
  fetchDetectionContent,
  prepareTimelineTiles,
} from "@/api/client";

export function useTimelineConfidence(jobId: string) {
  return useQuery({
    queryKey: ["timelineConfidence", jobId],
    queryFn: () => fetchTimelineConfidence(jobId),
    staleTime: Infinity, // Confidence data doesn't change
  });
}

export function useTimelineDetections(jobId: string) {
  return useQuery({
    queryKey: ["timelineDetections", jobId],
    queryFn: () => fetchDetectionContent(jobId),
    staleTime: Infinity,
  });
}

export function usePrepareTimeline() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (jobId: string) => prepareTimelineTiles(jobId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["hydrophoneDetectionJobs"] });
    },
  });
}
```

- [ ] **Step 5: Type check frontend**

Run: `cd frontend && npx tsc --noEmit`
Expected: 0 errors

- [ ] **Step 6: Commit**

```bash
git add frontend/src/api/types.ts frontend/src/api/client.ts frontend/src/hooks/queries/useTimeline.ts frontend/src/components/timeline/constants.ts
git commit -m "feat: add timeline viewer frontend types, API client, and query hooks"
```

---

## Task 9: TimelineViewer Page Component and Route

**Files:**
- Create: `frontend/src/components/timeline/TimelineViewer.tsx`
- Create: `frontend/src/components/timeline/TimelineHeader.tsx`
- Create: `frontend/src/components/timeline/PlaybackControls.tsx`
- Create: `frontend/src/components/timeline/ZoomSelector.tsx`
- Modify: `frontend/src/App.tsx`

This task builds the page shell — layout, state management, header, zoom selector, and playback controls. The spectrogram canvas and minimap come in the next tasks.

- [ ] **Step 1: Create TimelineHeader.tsx**

```typescript
// frontend/src/components/timeline/TimelineHeader.tsx
import { useNavigate } from "react-router-dom";
import { ArrowLeft, Tag, Settings } from "lucide-react";
import { COLORS } from "./constants";

interface TimelineHeaderProps {
  hydrophone: string;
  startTimestamp: number;
  endTimestamp: number;
  showLabels: boolean;
  onToggleLabels: () => void;
  freqRange: [number, number];
  onFreqRangeChange: (range: [number, number]) => void;
}

export function TimelineHeader({
  hydrophone,
  startTimestamp,
  endTimestamp,
  showLabels,
  onToggleLabels,
  freqRange,
}: TimelineHeaderProps) {
  const navigate = useNavigate();
  const startStr = new Date(startTimestamp * 1000).toISOString().replace("T", " ").slice(0, 19) + "Z";
  const endStr = new Date(endTimestamp * 1000).toISOString().replace("T", " ").slice(0, 19) + "Z";

  return (
    <div
      className="flex items-center justify-between px-4 py-2"
      style={{ background: COLORS.headerBg, borderBottom: `1px solid ${COLORS.border}` }}
    >
      <div className="flex items-center gap-3">
        <button
          onClick={() => navigate("/app/classifier/hydrophone")}
          className="flex items-center gap-1 text-xs hover:opacity-80"
          style={{ color: COLORS.textMuted }}
        >
          <ArrowLeft size={14} /> Back to Jobs
        </button>
        <span className="font-bold text-sm" style={{ color: COLORS.accent }}>
          {hydrophone}
        </span>
        <span className="text-xs" style={{ color: COLORS.textBright }}>
          {startStr} — {endStr}
        </span>
      </div>
      <div className="flex items-center gap-3">
        <button
          onClick={onToggleLabels}
          className="flex items-center gap-1 px-2 py-1 rounded text-[10px]"
          style={{
            background: showLabels ? COLORS.accentDim : COLORS.border,
            color: COLORS.accent,
          }}
        >
          <Tag size={10} /> Labels: {showLabels ? "ON" : "OFF"}
        </button>
        <span
          className="px-2 py-1 rounded text-[10px]"
          style={{ background: COLORS.border, color: COLORS.accent }}
        >
          Freq: {freqRange[0] / 1000}–{freqRange[1] / 1000} kHz
        </span>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Create ZoomSelector.tsx**

```typescript
// frontend/src/components/timeline/ZoomSelector.tsx
import type { ZoomLevel } from "@/api/types";
import { ZOOM_LEVELS, COLORS } from "./constants";

interface ZoomSelectorProps {
  activeLevel: ZoomLevel;
  onChange: (level: ZoomLevel) => void;
}

export function ZoomSelector({ activeLevel, onChange }: ZoomSelectorProps) {
  return (
    <div className="flex justify-center gap-1 py-1">
      {ZOOM_LEVELS.map((level) => (
        <button
          key={level}
          onClick={() => onChange(level)}
          className="px-2 py-0.5 rounded text-[10px] font-mono transition-colors"
          style={{
            background: level === activeLevel ? "rgba(64, 160, 128, 0.2)" : COLORS.bgDark,
            border: level === activeLevel ? `1px solid ${COLORS.accentDim}` : "1px solid transparent",
            color: level === activeLevel ? COLORS.accent : COLORS.textMuted,
          }}
        >
          {level}
        </button>
      ))}
    </div>
  );
}
```

- [ ] **Step 3: Create PlaybackControls.tsx**

```typescript
// frontend/src/components/timeline/PlaybackControls.tsx
import { Play, Pause, SkipBack, SkipForward, Plus, Minus } from "lucide-react";
import type { ZoomLevel } from "@/api/types";
import { COLORS } from "./constants";

interface PlaybackControlsProps {
  centerTimestamp: number;
  isPlaying: boolean;
  speed: number;
  onPlayPause: () => void;
  onSkipBack: () => void;
  onSkipForward: () => void;
  onSpeedChange: (speed: number) => void;
  onZoomIn: () => void;
  onZoomOut: () => void;
}

const SPEEDS = [0.5, 1, 2];

export function PlaybackControls({
  centerTimestamp,
  isPlaying,
  speed,
  onPlayPause,
  onSkipBack,
  onSkipForward,
  onSpeedChange,
  onZoomIn,
  onZoomOut,
}: PlaybackControlsProps) {
  const timeStr = new Date(centerTimestamp * 1000).toISOString().slice(11, 19) + " UTC";

  return (
    <div
      className="flex items-center justify-center gap-6 py-2.5 px-4"
      style={{ borderTop: `1px solid ${COLORS.border}` }}
    >
      <span className="text-[10px] font-mono" style={{ color: COLORS.textMuted }}>
        {timeStr}
      </span>
      <div className="flex items-center gap-4">
        <button onClick={onSkipBack} style={{ color: COLORS.textBright }}>
          <SkipBack size={16} />
        </button>
        <button
          onClick={onPlayPause}
          className="w-9 h-9 rounded-full flex items-center justify-center"
          style={{ border: `1.5px solid ${COLORS.accent}` }}
        >
          {isPlaying ? (
            <Pause size={16} style={{ color: COLORS.accent }} />
          ) : (
            <Play size={16} style={{ color: COLORS.accent, paddingLeft: 2 }} />
          )}
        </button>
        <button onClick={onSkipForward} style={{ color: COLORS.textBright }}>
          <SkipForward size={16} />
        </button>
      </div>
      <button
        onClick={() => {
          const idx = SPEEDS.indexOf(speed);
          onSpeedChange(SPEEDS[(idx + 1) % SPEEDS.length]);
        }}
        className="text-[10px] font-mono"
        style={{ color: COLORS.textMuted }}
      >
        {speed}x
      </button>
      <div className="flex items-center gap-2 ml-10">
        <button onClick={onZoomOut} style={{ color: COLORS.textBright }}>
          <Minus size={14} />
        </button>
        <span className="text-[10px]" style={{ color: COLORS.accent }}>Zoom</span>
        <button onClick={onZoomIn} style={{ color: COLORS.textBright }}>
          <Plus size={14} />
        </button>
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Create TimelineViewer.tsx (page component)**

```typescript
// frontend/src/components/timeline/TimelineViewer.tsx
import { useState, useCallback, useEffect } from "react";
import { useParams } from "react-router-dom";
import type { ZoomLevel } from "@/api/types";
import { useTimelineConfidence, useTimelineDetections } from "@/hooks/queries/useTimeline";
import { useHydrophoneDetectionJobs } from "@/hooks/queries/useClassifier";
import { TimelineHeader } from "./TimelineHeader";
import { ZoomSelector } from "./ZoomSelector";
import { PlaybackControls } from "./PlaybackControls";
import { ZOOM_LEVELS, VIEWPORT_SPAN, COLORS } from "./constants";

export function TimelineViewer() {
  const { jobId } = useParams<{ jobId: string }>();
  const { data: jobs } = useHydrophoneDetectionJobs(0);
  const job = jobs?.find((j) => j.id === jobId);

  // Core state
  const [centerTimestamp, setCenterTimestamp] = useState<number>(0);
  const [zoomLevel, setZoomLevel] = useState<ZoomLevel>("1h");
  const [isPlaying, setIsPlaying] = useState(false);
  const [freqRange, setFreqRange] = useState<[number, number]>([0, 3000]);
  const [showLabels, setShowLabels] = useState(false);
  const [speed, setSpeed] = useState(1);

  // Initialize center to job midpoint
  useEffect(() => {
    if (job?.start_timestamp && job?.end_timestamp && centerTimestamp === 0) {
      setCenterTimestamp(
        job.start_timestamp + (job.end_timestamp - job.start_timestamp) / 2,
      );
    }
  }, [job, centerTimestamp]);

  // Data queries
  const { data: confidence } = useTimelineConfidence(jobId ?? "");
  const { data: detections } = useTimelineDetections(jobId ?? "");

  // Zoom in/out
  const zoomIn = useCallback(() => {
    const idx = ZOOM_LEVELS.indexOf(zoomLevel);
    if (idx < ZOOM_LEVELS.length - 1) setZoomLevel(ZOOM_LEVELS[idx + 1]);
  }, [zoomLevel]);

  const zoomOut = useCallback(() => {
    const idx = ZOOM_LEVELS.indexOf(zoomLevel);
    if (idx > 0) setZoomLevel(ZOOM_LEVELS[idx - 1]);
  }, [zoomLevel]);

  // Play/pause
  const togglePlay = useCallback(() => {
    setIsPlaying((prev) => !prev);
  }, []);

  // Skip to next/prev detection
  // Note: detection row start_sec/end_sec are canonical snapped bounds (timeline-
  // absolute for hydrophone jobs since the detection worker writes them that way).
  // If start_sec is file-relative for your data, you'll need to add job.start_timestamp.
  // Check your detection content endpoint output to confirm.
  const skipForward = useCallback(() => {
    if (!detections || !job) return;
    const jobStart = job.start_timestamp ?? 0;
    const next = detections.find((d) => (jobStart + d.start_sec) > centerTimestamp);
    if (next) setCenterTimestamp(jobStart + next.start_sec);
  }, [detections, centerTimestamp, job]);

  const skipBack = useCallback(() => {
    if (!detections || !job) return;
    const jobStart = job.start_timestamp ?? 0;
    const prev = [...detections].reverse().find((d) => (jobStart + d.end_sec) < centerTimestamp);
    if (prev) setCenterTimestamp(jobStart + prev.start_sec);
  }, [detections, centerTimestamp, job]);

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === " ") { e.preventDefault(); togglePlay(); }
      if (e.key === "+" || e.key === "=") zoomIn();
      if (e.key === "-") zoomOut();
      if (e.key === "ArrowLeft") {
        setCenterTimestamp((prev) => {
          const span = VIEWPORT_SPAN[zoomLevel];
          return Math.max(job?.start_timestamp ?? prev, prev - span / 10);
        });
      }
      if (e.key === "ArrowRight") {
        setCenterTimestamp((prev) => {
          const span = VIEWPORT_SPAN[zoomLevel];
          return Math.min(job?.end_timestamp ?? prev, prev + span / 10);
        });
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [togglePlay, zoomIn, zoomOut, zoomLevel, job]);

  if (!jobId || !job) {
    return (
      <div className="flex items-center justify-center h-screen" style={{ background: COLORS.bg, color: COLORS.text }}>
        Loading...
      </div>
    );
  }

  return (
    <div className="flex flex-col h-screen font-mono text-xs" style={{ background: COLORS.bg, color: COLORS.text }}>
      <TimelineHeader
        hydrophone={job.hydrophone_name ?? job.hydrophone_id ?? ""}
        startTimestamp={job.start_timestamp ?? 0}
        endTimestamp={job.end_timestamp ?? 0}
        showLabels={showLabels}
        onToggleLabels={() => setShowLabels((s) => !s)}
        freqRange={freqRange}
        onFreqRangeChange={setFreqRange}
      />

      {/* Minimap placeholder — Task 10 */}
      <div className="px-4 pt-1.5">
        <div className="h-7 rounded" style={{ background: COLORS.bgDark }} />
      </div>

      {/* Main spectrogram viewport placeholder — Task 11 */}
      <div className="flex-1 mx-4 my-2 rounded relative" style={{ background: COLORS.bgDark }}>
        <div className="flex items-center justify-center h-full" style={{ color: COLORS.textMuted }}>
          Spectrogram viewport — implemented in Task 11
        </div>
      </div>

      <ZoomSelector activeLevel={zoomLevel} onChange={setZoomLevel} />
      <PlaybackControls
        centerTimestamp={centerTimestamp}
        isPlaying={isPlaying}
        speed={speed}
        onPlayPause={togglePlay}
        onSkipBack={skipBack}
        onSkipForward={skipForward}
        onSpeedChange={setSpeed}
        onZoomIn={zoomIn}
        onZoomOut={zoomOut}
      />
    </div>
  );
}
```

- [ ] **Step 5: Add route in App.tsx**

Add import and route:

```typescript
import { TimelineViewer } from "@/components/timeline/TimelineViewer";
```

Add route inside `<Routes>` after the classifier routes:

```typescript
<Route path="/app/classifier/timeline/:jobId" element={<TimelineViewer />} />
```

- [ ] **Step 6: Type check**

Run: `cd frontend && npx tsc --noEmit`
Expected: 0 errors

- [ ] **Step 7: Commit**

```bash
git add frontend/src/components/timeline/TimelineViewer.tsx frontend/src/components/timeline/TimelineHeader.tsx frontend/src/components/timeline/PlaybackControls.tsx frontend/src/components/timeline/ZoomSelector.tsx frontend/src/App.tsx
git commit -m "feat: add TimelineViewer page shell with header, zoom, and playback controls"
```

---

## Task 10: Minimap Component

**Files:**
- Create: `frontend/src/components/timeline/Minimap.tsx`
- Modify: `frontend/src/components/timeline/TimelineViewer.tsx`

- [ ] **Step 1: Create Minimap.tsx**

The minimap renders the 24h confidence heatmap on a `<canvas>` element and draws a viewport indicator rectangle. It supports click-to-jump and drag-to-pan on the viewport indicator.

Key implementation details:
- Use `useRef` for the canvas element
- Use `useEffect` to draw the confidence heatmap when data loads
- Confidence scores are mapped to the green-yellow gradient from `CONFIDENCE_GRADIENT`
- The viewport indicator rectangle width = `(viewportSpan / totalDuration) * canvasWidth`
- The viewport indicator position = `((centerTimestamp - jobStart) / totalDuration) * canvasWidth`
- Click handler: convert click X to timestamp and call `onCenterChange`
- Drag handler: track mousedown on indicator, mousemove updates center, mouseup stops

- [ ] **Step 2: Wire Minimap into TimelineViewer**

Replace the minimap placeholder div with `<Minimap>` passing confidence data, center timestamp, zoom level, job start/end, and the setter.

- [ ] **Step 3: Verify visually**

Run: `cd frontend && npm run dev`
Navigate to a timeline viewer route. The minimap should render a colored heatmap bar with a viewport indicator rectangle.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/timeline/Minimap.tsx frontend/src/components/timeline/TimelineViewer.tsx
git commit -m "feat: add confidence heatmap minimap with click-to-jump and drag-to-pan"
```

---

## Task 11: TileCanvas and SpectrogramViewport

**Files:**
- Create: `frontend/src/components/timeline/TileCanvas.tsx`
- Create: `frontend/src/components/timeline/SpectrogramViewport.tsx`
- Modify: `frontend/src/components/timeline/TimelineViewer.tsx`

This is the most complex frontend component — the canvas that loads, composites, and crossfades spectrogram tile images.

- [ ] **Step 1: Create TileCanvas.tsx**

Key implementation:
- Canvas element fills available width
- Calculate visible tile indices from `centerTimestamp`, `zoomLevel`, and canvas width
- Load `TILE_WIDTH_PX * TILE_HEIGHT_PX` PNG images using `new Image()` with src from `timelineTileUrl()`
- Maintain an LRU `Map<string, HTMLImageElement>` for loaded tiles (cap at ~200 entries)
- On each render frame (requestAnimationFrame during playback, or on state change when paused):
  - Clear canvas
  - Calculate pixel offset for each visible tile based on its time position relative to canvas center
  - Draw each loaded tile image at the correct x position using `ctx.drawImage()`
  - For tiles still loading, draw a dark placeholder
- Crossfade: when `zoomLevel` changes, keep previous zoom's tiles visible at opacity 1→0 while new zoom fades 0→1 over `CROSSFADE_DURATION_MS`

- [ ] **Step 2: Create SpectrogramViewport.tsx**

Wraps TileCanvas with:
- Frequency axis (left gutter, 44px wide) — labels at key frequency points based on `freqRange`
- Confidence strip (20px tall bar below canvas) — rendered client-side from confidence data
- Time axis (UTC labels below confidence strip) — density adapts to zoom level
- Playhead (vertical line at center with triangle marker, z-index above tiles)
- Pan handler: mousedown + mousemove on canvas updates `centerTimestamp` when paused

- [ ] **Step 3: Wire into TimelineViewer**

Replace the spectrogram placeholder with `<SpectrogramViewport>`. Pass all state + setters.

- [ ] **Step 4: Verify visually**

Run backend and frontend dev servers. Navigate to timeline viewer for a completed hydrophone job. Spectrogram tiles should load and display. Zoom level changes should crossfade. Panning should work when paused.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/timeline/TileCanvas.tsx frontend/src/components/timeline/SpectrogramViewport.tsx frontend/src/components/timeline/TimelineViewer.tsx
git commit -m "feat: add spectrogram tile canvas with crossfade zoom and pan"
```

---

## Task 12: Audio Playback Integration

**Files:**
- Modify: `frontend/src/components/timeline/TimelineViewer.tsx`
- Modify: `frontend/src/components/timeline/PlaybackControls.tsx`

- [ ] **Step 1: Add audio playback to TimelineViewer**

Key implementation:
- Use an `<audio>` element (hidden) or `AudioContext` + `fetch`
- When `isPlaying` becomes true:
  - Fetch audio from `timelineAudioUrl(jobId, centerTimestamp, AUDIO_PREFETCH_SEC)`
  - Create an `AudioBufferSourceNode` or set `audio.src`
  - Start playback; use `requestAnimationFrame` loop to advance `centerTimestamp` based on elapsed time and `speed`
- When user pans during playback, auto-pause (set `isPlaying = false`)
- When `isPlaying` becomes false, stop audio
- Pre-fetch next chunk when playback reaches 80% of current chunk

The simplest approach: use a hidden `<audio>` element with sequential WAV chunks. Set `src` to the timeline audio URL, listen for `timeupdate` events to advance `centerTimestamp`.

- [ ] **Step 2: Wire speed multiplier**

`playbackRate` on the audio element maps directly to speed (0.5, 1, 2).

- [ ] **Step 3: Test playback**

Run both servers. Navigate to timeline viewer. Click play. Audio should play and the spectrogram should scroll.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/timeline/TimelineViewer.tsx
git commit -m "feat: add audio playback with spectrogram scroll sync"
```

---

## Task 13: Detection Overlay and Popover

**Files:**
- Create: `frontend/src/components/timeline/DetectionOverlay.tsx`
- Create: `frontend/src/components/timeline/DetectionPopover.tsx`
- Modify: `frontend/src/components/timeline/SpectrogramViewport.tsx`

- [ ] **Step 1: Create DetectionOverlay.tsx**

Renders semi-transparent colored rectangles at labeled detection positions:
- Map each detection row's `start_sec`/`end_sec` to pixel positions based on current zoom and center
- Color by label: use `COLORS.labelHumpback`, etc.
- Full frequency-axis height
- Only visible when `showLabels` is true
- Positioned absolutely over the TileCanvas

- [ ] **Step 2: Create DetectionPopover.tsx**

A floating popover anchored to click position:
- Props: detection row data, position, onClose
- Shows: timestamp range, avg/peak confidence, current labels, "View in table" link
- "View in table" link navigates to `/app/classifier/hydrophone?highlight={row_id}`
- Dismissed by click outside or Esc

- [ ] **Step 3: Add click handler to SpectrogramViewport**

On click (when not playing):
- Convert click X position to timestamp
- Find the detection row whose `[start_sec, end_sec]` contains that timestamp
- If found, show `DetectionPopover` with that row's data
- If not found, dismiss any open popover

- [ ] **Step 4: Wire overlay visibility to showLabels state**

- [ ] **Step 5: Test**

Verify clicking on a detection region shows the popover. Verify label overlay toggles.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/components/timeline/DetectionOverlay.tsx frontend/src/components/timeline/DetectionPopover.tsx frontend/src/components/timeline/SpectrogramViewport.tsx
git commit -m "feat: add detection overlay and click popover to timeline viewer"
```

---

## Task 14: HydrophoneTab — Add Timeline View Button

**Files:**
- Modify: `frontend/src/components/classifier/HydrophoneTab.tsx`

- [ ] **Step 1: Read the existing HydrophoneTab component**

Read `frontend/src/components/classifier/HydrophoneTab.tsx` to understand how completed job rows are rendered and where to add the button.

- [ ] **Step 2: Add "Timeline View" button**

In the job row actions area (where download, extract, delete buttons are), add a button that:
- Only appears for completed hydrophone jobs where `timeline_tiles_ready === true` (or always appears for completed jobs — the viewer can use on-demand rendering as fallback)
- Uses `<Link to={`/app/classifier/timeline/${job.id}`}>` or `navigate()`
- Styled consistently with existing action buttons
- Icon: could use `LineChart` or `BarChart3` from lucide-react

- [ ] **Step 3: Verify**

Navigate to the Hydrophone tab. Completed jobs should show a "Timeline" button.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/components/classifier/HydrophoneTab.tsx
git commit -m "feat: add Timeline View button to completed hydrophone jobs"
```

---

## Task 15: Playwright E2E Tests

**Files:**
- Create: `frontend/e2e/timeline.spec.ts`

- [ ] **Step 1: Write Playwright tests**

```typescript
// frontend/e2e/timeline.spec.ts
import { test, expect } from "@playwright/test";

test.describe("Timeline Viewer", () => {
  test("navigates to timeline from hydrophone tab", async ({ page }) => {
    await page.goto("http://localhost:5173/app/classifier/hydrophone");
    // Find a completed hydrophone job with timeline button
    const timelineBtn = page.locator("text=Timeline").first();
    const visible = await timelineBtn.isVisible().catch(() => false);
    test.skip(!visible, "No completed hydrophone jobs with timeline available");

    await timelineBtn.click();
    await expect(page).toHaveURL(/\/app\/classifier\/timeline\//);
    // Should show the timeline viewer page
    await expect(page.locator("text=Back to Jobs")).toBeVisible();
  });

  test("zoom level buttons change active state", async ({ page }) => {
    await page.goto("http://localhost:5173/app/classifier/hydrophone");
    const timelineBtn = page.locator("text=Timeline").first();
    const visible = await timelineBtn.isVisible().catch(() => false);
    test.skip(!visible, "No completed hydrophone jobs with timeline available");

    await timelineBtn.click();
    await page.waitForURL(/\/app\/classifier\/timeline\//);

    // Click different zoom levels
    const zoomBtn5m = page.locator("button", { hasText: "5m" });
    await zoomBtn5m.click();
    // Active button should have accent styling
    await expect(zoomBtn5m).toHaveCSS("color", /rgb/);
  });

  test("play button toggles state", async ({ page }) => {
    await page.goto("http://localhost:5173/app/classifier/hydrophone");
    const timelineBtn = page.locator("text=Timeline").first();
    const visible = await timelineBtn.isVisible().catch(() => false);
    test.skip(!visible, "No completed hydrophone jobs with timeline available");

    await timelineBtn.click();
    await page.waitForURL(/\/app\/classifier\/timeline\//);

    // Find play button (circle with play icon)
    const playBtn = page.locator("button").filter({ has: page.locator("svg") }).nth(1);
    await playBtn.click();
    // After click, it should show pause icon or changed state
  });

  test("back button returns to hydrophone tab", async ({ page }) => {
    await page.goto("http://localhost:5173/app/classifier/hydrophone");
    const timelineBtn = page.locator("text=Timeline").first();
    const visible = await timelineBtn.isVisible().catch(() => false);
    test.skip(!visible, "No completed hydrophone jobs with timeline available");

    await timelineBtn.click();
    await page.waitForURL(/\/app\/classifier\/timeline\//);

    await page.locator("text=Back to Jobs").click();
    await expect(page).toHaveURL(/\/app\/classifier\/hydrophone/);
  });
});
```

- [ ] **Step 2: Run Playwright tests**

Run: `cd frontend && npx playwright test e2e/timeline.spec.ts`
Expected: Tests pass (or skip gracefully if no completed hydrophone jobs)

- [ ] **Step 3: Commit**

```bash
git add frontend/e2e/timeline.spec.ts
git commit -m "test: add Playwright e2e tests for timeline viewer"
```

---

## Task 16: Documentation Updates

**Files:**
- Modify: `CLAUDE.md`
- Modify: `STATUS.md`
- Modify: `DECISIONS.md`
- Modify: `README.md`

- [ ] **Step 1: Update CLAUDE.md**

Add the timeline viewer route to the frontend file structure section. Add a note in the Hydrophone section about the timeline viewer page.

- [ ] **Step 2: Update STATUS.md**

Add timeline viewer to current capabilities. Note the new migration (025), new API endpoints, and new frontend route.

- [ ] **Step 3: Append ADR to DECISIONS.md**

```markdown
## ADR-NNN: Timeline Viewer Tile Architecture

**Status:** Accepted
**Date:** 2026-03-24

**Context:** Need a zoomable spectrogram viewer for hydrophone detection jobs spanning up to 24 hours.

**Decision:** Canvas 2D with pre-colored PNG tiles at 6 discrete zoom levels. Ocean Depth colormap baked into tiles. Coarse levels pre-rendered on job completion, fine levels rendered on demand with global FIFO cache. Timeline audio resolved from HLS local cache via `resolve_timeline_audio()`.

**Alternatives considered:**
- WebGL shader rendering (rejected: over-complex for discrete zoom levels)
- On-demand only (rejected: minimap/initial view would be slow)

**Consequences:** Simple frontend (no WebGL), human-inspectable tile cache, colormap changes require re-render.
```

- [ ] **Step 4: Update README.md**

Add timeline viewer to the feature list and API endpoint documentation.

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md STATUS.md DECISIONS.md README.md
git commit -m "docs: update project docs for timeline viewer feature"
```

---

## Summary

| Task | Description | Est. Complexity |
|------|------------|-----------------|
| 1 | DB migration — `timeline_tiles_ready` column | Small |
| 2 | Storage helpers and config | Small |
| 3 | Ocean Depth colormap and tile renderer | Medium |
| 4 | Timeline tile cache | Small |
| 5 | Timeline audio resolution from HLS | Medium-Large |
| 6 | Timeline API endpoints (4 endpoints) | Large |
| 7 | Worker auto-prepare on job completion | Small |
| 8 | Frontend types, API client, hooks, constants | Small |
| 9 | TimelineViewer page shell + route | Medium |
| 10 | Minimap (confidence heatmap canvas) | Medium |
| 11 | TileCanvas + SpectrogramViewport (core canvas) | Large |
| 12 | Audio playback integration | Medium |
| 13 | Detection overlay + popover | Medium |
| 14 | HydrophoneTab timeline button | Small |
| 15 | Playwright E2E tests | Small |
| 16 | Documentation updates | Small |

**Dependency chain:** Tasks 1-2 are prerequisites. Tasks 3 and 4 can run in parallel. Task 5 (HLS audio resolution) includes an s3_stream.py refactoring step (Step 3b) that Tasks 6 and 7 also depend on — complete Task 5 before starting 6 or 7. Task 6 depends on 3-5. Task 7 depends on 3-5. Task 8 depends on 6. Tasks 9-14 are sequential (each builds on prior). Tasks 15-16 come last.
