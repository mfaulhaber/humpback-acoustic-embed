# Timeline Viewer Enhancements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve the timeline viewer with background tile caching, positive-only label bars, audio-authoritative playhead sync, and gapless double-buffered MP3 playback.

**Architecture:** Four independent enhancements to the existing timeline viewer. Backend changes: per-job LRU tile cache, enhanced prepare endpoint with progress reporting, MP3 audio encoding. Frontend changes: double-buffered audio elements with audio-authoritative playhead, positive-only detection label bars with hover tooltips, background cache progress indicator.

**Tech Stack:** Python/FastAPI (backend), React/TypeScript (frontend), ffmpeg (MP3 encoding), TanStack Query (server state)

**Spec:** `docs/specs/2026-03-25-timeline-viewer-enhancements-design.md`

---

## File Structure

### Backend — Modified Files

| File | Responsibility | Changes |
|------|---------------|---------|
| `src/humpback/config.py` | Settings | Replace `timeline_tile_cache_max_items` with `timeline_cache_max_jobs` |
| `src/humpback/processing/timeline_cache.py` | Disk tile cache | Rewrite: per-job LRU eviction, `.last_access` sentinel, `touch_job()`, `evict_lru_jobs()` |
| `src/humpback/api/routers/timeline.py` | Timeline API | Enhanced prepare (all zoom levels, background thread), new prepare-status endpoint, MP3 audio format, 600s max duration |

### Frontend — Modified Files

| File | Responsibility | Changes |
|------|---------------|---------|
| `frontend/src/components/timeline/constants.ts` | Shared constants | `AUDIO_PREFETCH_SEC` 30→300, add `AUDIO_FORMAT` |
| `frontend/src/components/timeline/TimelineViewer.tsx` | Main container | Double-buffered audio, audio-authoritative RAF loop, `playbackOriginEpoch` state |
| `frontend/src/components/timeline/DetectionOverlay.tsx` | Label rendering | Positive-only filter, narrow bars, hover tooltips, pointer events |
| `frontend/src/components/timeline/SpectrogramViewport.tsx` | Viewport layout | Extend overlay to cover confidence strip |
| `frontend/src/api/client.ts` | API functions | Add `format` param to `timelineAudioUrl`, add `fetchPrepareStatus`, update `prepareTimelineTiles` |
| `frontend/src/api/types.ts` | Type definitions | Add `PrepareStatusResponse` type |
| `frontend/src/hooks/queries/useTimeline.ts` | Query hooks | Add `usePrepareStatus` polling hook, update prepare mutation |

### Test Files — Modified/Created

| File | Tests |
|------|-------|
| `tests/unit/test_timeline_cache.py` | Add LRU eviction, touch_job, multi-job tests |
| `tests/integration/test_timeline_api.py` | Add prepare-status, MP3 format, 600s duration, cache eviction tests |
| `frontend/e2e/timeline.spec.ts` | Add positive-only labels, hover tooltip tests |

---

## Task 1: Per-Job LRU Tile Cache

Rewrite `TimelineTileCache` from flat FIFO to per-job LRU with configurable job count.

**Files:**
- Modify: `src/humpback/config.py:50-54`
- Modify: `src/humpback/processing/timeline_cache.py` (full rewrite)
- Test: `tests/unit/test_timeline_cache.py`

### Step 1: Write failing tests for per-job LRU cache

- [ ] Add new tests to `tests/unit/test_timeline_cache.py`:

```python
def test_touch_job_creates_sentinel(tmp_path):
    """Accessing a job should create/update .last_access sentinel."""
    cache = TimelineTileCache(cache_dir=tmp_path, max_jobs=5)
    cache.put("job_a", "1h", 0, b"tile-data")
    sentinel = tmp_path / "job_a" / ".last_access"
    assert sentinel.exists()


def test_get_updates_sentinel_mtime(tmp_path):
    """Cache hit should touch the job's sentinel to keep it fresh."""
    cache = TimelineTileCache(cache_dir=tmp_path, max_jobs=5)
    cache.put("job_a", "1h", 0, b"tile-data")
    sentinel = tmp_path / "job_a" / ".last_access"
    old_mtime = sentinel.stat().st_mtime
    import time; time.sleep(0.05)
    cache.get("job_a", "1h", 0)
    assert sentinel.stat().st_mtime > old_mtime


def test_lru_eviction_removes_oldest_job(tmp_path):
    """When max_jobs exceeded, oldest-accessed job directory is removed."""
    cache = TimelineTileCache(cache_dir=tmp_path, max_jobs=2)
    cache.put("job_a", "1h", 0, b"a-data")
    import time; time.sleep(0.05)
    cache.put("job_b", "1h", 0, b"b-data")
    time.sleep(0.05)
    cache.put("job_c", "1h", 0, b"c-data")  # triggers eviction
    assert cache.get("job_a", "1h", 0) is None  # evicted
    assert cache.get("job_b", "1h", 0) == b"b-data"
    assert cache.get("job_c", "1h", 0) == b"c-data"


def test_lru_eviction_preserves_recently_accessed(tmp_path):
    """Accessing an old job refreshes it so a newer-but-untouched job gets evicted."""
    cache = TimelineTileCache(cache_dir=tmp_path, max_jobs=2)
    cache.put("job_a", "1h", 0, b"a-data")
    import time; time.sleep(0.05)
    cache.put("job_b", "1h", 0, b"b-data")
    time.sleep(0.05)
    cache.get("job_a", "1h", 0)  # refresh job_a
    time.sleep(0.05)
    cache.put("job_c", "1h", 0, b"c-data")  # should evict job_b
    assert cache.get("job_a", "1h", 0) == b"a-data"
    assert cache.get("job_b", "1h", 0) is None  # evicted
    assert cache.get("job_c", "1h", 0) == b"c-data"


def test_job_count_returns_cached_job_count(tmp_path):
    """job_count() should return the number of job directories in the cache."""
    cache = TimelineTileCache(cache_dir=tmp_path, max_jobs=5)
    assert cache.job_count() == 0
    cache.put("job_a", "1h", 0, b"data")
    cache.put("job_b", "1h", 0, b"data")
    assert cache.job_count() == 2
```

- [ ] Run tests to verify they fail:

```bash
uv run pytest tests/unit/test_timeline_cache.py -v
```

Expected: FAIL — `TimelineTileCache` does not accept `max_jobs` param, no `touch_job` or `job_count` methods.

### Step 2: Update config setting

- [ ] In `src/humpback/config.py`, replace `timeline_tile_cache_max_items: int = 5000` (line 53) with:

```python
timeline_cache_max_jobs: int = 15
```

### Step 3: Rewrite TimelineTileCache

- [ ] Replace the contents of `src/humpback/processing/timeline_cache.py` with:

```python
"""Disk-backed tile cache with per-job LRU eviction."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


class TimelineTileCache:
    """Stores timeline spectrogram tiles on disk, evicting whole jobs LRU."""

    def __init__(self, cache_dir: str | Path, max_jobs: int = 15) -> None:
        self.cache_dir = Path(cache_dir)
        self.max_jobs = max_jobs

    # -- public API --

    def get(
        self, job_id: str, zoom_level: str, tile_index: int
    ) -> bytes | None:
        path = self._tile_path(job_id, zoom_level, tile_index)
        if not path.exists():
            return None
        self._touch_job(job_id)
        return path.read_bytes()

    def put(
        self, job_id: str, zoom_level: str, tile_index: int, data: bytes
    ) -> None:
        path = self._tile_path(job_id, zoom_level, tile_index)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)
        self._touch_job(job_id)
        self._evict_lru_jobs()

    def job_count(self) -> int:
        if not self.cache_dir.exists():
            return 0
        return sum(
            1
            for p in self.cache_dir.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        )

    def tile_count_for_zoom(self, job_id: str, zoom_level: str) -> int:
        zoom_dir = self.cache_dir / job_id / zoom_level
        if not zoom_dir.exists():
            return 0
        return sum(1 for f in zoom_dir.iterdir() if f.suffix == ".png")

    # -- internals --

    def _tile_path(
        self, job_id: str, zoom_level: str, tile_index: int
    ) -> Path:
        return (
            self.cache_dir / job_id / zoom_level / f"tile_{tile_index:04d}.png"
        )

    def _touch_job(self, job_id: str) -> None:
        job_dir = self.cache_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        sentinel = job_dir / ".last_access"
        sentinel.touch()

    def _evict_lru_jobs(self) -> None:
        if not self.cache_dir.exists():
            return
        job_dirs = [
            p
            for p in self.cache_dir.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        ]
        if len(job_dirs) <= self.max_jobs:
            return
        # Sort by sentinel mtime (oldest first)
        def _access_time(d: Path) -> float:
            sentinel = d / ".last_access"
            if sentinel.exists():
                return sentinel.stat().st_mtime
            return 0.0

        job_dirs.sort(key=_access_time)
        to_remove = len(job_dirs) - self.max_jobs
        for job_dir in job_dirs[:to_remove]:
            logger.info("Evicting tile cache for job %s", job_dir.name)
            shutil.rmtree(job_dir)
```

### Step 4: Update timeline router to use new cache constructor

- [ ] In `src/humpback/api/routers/timeline.py`, update both `TimelineTileCache` instantiation sites (tile endpoint ~line 236 and prepare endpoint ~line 346). The current code creates a **per-job** cache directory via `timeline_tiles_dir(settings.storage_root, job.id)`. The new LRU design needs a **global** cache directory with job subdirectories inside it. Change both sites from:

```python
tiles_dir = timeline_tiles_dir(settings.storage_root, job.id)
cache = TimelineTileCache(tiles_dir, max_items=settings.timeline_tile_cache_max_items)
```

to:

```python
cache = TimelineTileCache(
    cache_dir=settings.storage_root / "timeline_cache",
    max_jobs=settings.timeline_cache_max_jobs,
)
```

The cache now stores tiles at `{storage_root}/timeline_cache/{job_id}/{zoom}/{tile_NNNN.png}` — the `job_id` is passed to `get()`/`put()` calls. Existing `cache.get(job.id, ...)` and `cache.put(job.id, ...)` calls remain correct since `job.id` is already a string. Remove the `timeline_tiles_dir` import if no longer used elsewhere in this file.

### Step 5: Run tests

- [ ] Run the tile cache tests:

```bash
uv run pytest tests/unit/test_timeline_cache.py -v
```

Expected: All tests PASS (old tests may need updating to use `max_jobs` instead of `max_items`).

- [ ] Update any old tests that reference `max_items` to use `max_jobs`.

### Step 6: Run full test suite

- [ ] Verify nothing else broke:

```bash
uv run pytest tests/ -v
```

### Step 7: Commit

```bash
git add src/humpback/config.py src/humpback/processing/timeline_cache.py src/humpback/api/routers/timeline.py tests/unit/test_timeline_cache.py
git commit -m "feat: per-job LRU tile cache with configurable job count"
```

---

## Task 2: Enhanced Prepare Endpoint with Background Rendering

Add all-zoom-level background rendering and a progress status endpoint.

**Files:**
- Modify: `src/humpback/api/routers/timeline.py:130-186` (prepare helpers), `336-372` (prepare endpoint)
- Test: `tests/integration/test_timeline_api.py`

### Step 1: Write failing tests

- [ ] Add to `tests/integration/test_timeline_api.py`:

```python
def test_prepare_all_zoom_levels(client, completed_hydrophone_job):
    """POST /prepare with all zoom levels should render tiles for each level."""
    job_id = completed_hydrophone_job
    resp = client.post(f"/classifier/detection-jobs/{job_id}/timeline/prepare")
    assert resp.status_code == 200
    data = resp.json()
    assert data["tiles_rendered"] > 0
    assert data["timeline_tiles_ready"] is True


def test_prepare_status_endpoint(client, completed_hydrophone_job):
    """GET /prepare-status should return per-zoom progress."""
    job_id = completed_hydrophone_job
    # Trigger prepare first
    client.post(f"/classifier/detection-jobs/{job_id}/timeline/prepare")
    resp = client.get(
        f"/classifier/detection-jobs/{job_id}/timeline/prepare-status"
    )
    assert resp.status_code == 200
    data = resp.json()
    # Should have entries for each zoom level
    assert "1h" in data
    assert "total" in data["1h"]
    assert "rendered" in data["1h"]


def test_prepare_status_job_not_found(client):
    """GET /prepare-status for missing job should 404."""
    resp = client.get(
        "/classifier/detection-jobs/99999/timeline/prepare-status"
    )
    assert resp.status_code == 404
```

- [ ] Run to verify they fail:

```bash
uv run pytest tests/integration/test_timeline_api.py -k "prepare_all or prepare_status" -v
```

### Step 2: Add PrepareStatusResponse model

- [ ] In `src/humpback/api/routers/timeline.py`, add a response model near the existing `PrepareResponse` (around line 42):

```python
class ZoomProgress(BaseModel):
    total: int
    rendered: int

# PrepareStatusResponse is Dict[str, ZoomProgress] — use dict directly in endpoint
```

### Step 3: Enhance _prepare_tiles_sync to support all zoom levels

- [ ] Rewrite `_prepare_tiles_sync()` (lines 130-186) to accept a list of zoom levels and render them in priority order:

```python
_PREPARE_PRIORITY = ["1h", "15m", "5m", "1m", "6h", "24h"]


def _prepare_tiles_sync(
    *,
    job,
    settings,
    cache: TimelineTileCache,
    zoom_levels: list[str] | None = None,
) -> int:
    """Render tiles for requested zoom levels in priority order. Skips cached."""
    duration = _job_duration(job)
    if duration <= 0:
        return 0
    levels = zoom_levels or list(_PREPARE_PRIORITY)
    # Sort by priority order
    priority = {z: i for i, z in enumerate(_PREPARE_PRIORITY)}
    levels.sort(key=lambda z: priority.get(z, 99))

    rendered = 0
    for zoom in levels:
        count = tile_count(zoom, job_duration_sec=duration)
        for idx in range(count):
            if cache.get(job.id, zoom, idx) is not None:
                continue
            try:
                _render_tile_sync(
                    job=job,
                    zoom_level=zoom,
                    tile_index=idx,
                    settings=settings,
                    cache=cache,
                )
                rendered += 1
            except Exception:
                logger.exception(
                    "Failed to render tile %s/%d for job %s",
                    zoom, idx, job.id,
                )
    return rendered
```

Note: Both `_prepare_tiles_sync` and `_render_tile_sync` use keyword-only arguments (the `*` separator). Always call them with keyword syntax. The existing router calls them via `asyncio.to_thread(fn, key=val, ...)` — match that pattern.

Note: `_render_tile_sync` already writes to the cache internally via `cache.put()`, so no duplicate write needed. The function uses keyword-only arguments (`*` separator) — always use `keyword=value` syntax when calling it.

### Step 4: Enhance POST /prepare endpoint

- [ ] Update the prepare endpoint (lines 336-372) to launch background rendering in a thread and return immediately. The existing router uses async session patterns (`SessionDep`, `await session.commit()`). Match that pattern:

```python
import threading

# Module-level set to track in-progress prepare jobs (prevents duplicate work)
_preparing: set[str] = set()
_preparing_lock = threading.Lock()


@router.post("/prepare")
async def prepare_tiles(
    job_id: str,
    session: SessionDep,
    settings: SettingsDep,
):
    job = await _get_job_or_404(session, job_id)
    cache = TimelineTileCache(
        cache_dir=settings.storage_root / "timeline_cache",
        max_jobs=settings.timeline_cache_max_jobs,
    )

    with _preparing_lock:
        already_running = job.id in _preparing
        if not already_running:
            _preparing.add(job.id)

    if not already_running:
        # Launch background thread — non-blocking
        def _background():
            try:
                _prepare_tiles_sync(job=job, settings=settings, cache=cache)
            finally:
                with _preparing_lock:
                    _preparing.discard(job.id)

        threading.Thread(target=_background, daemon=True).start()

    # Mark timeline_tiles_ready via re-query (match existing pattern)
    from humpback.models.classifier import DetectionJob
    from sqlalchemy import select as sa_select

    result = await session.execute(
        sa_select(DetectionJob).where(DetectionJob.id == job_id)
    )
    db_job = result.scalar_one_or_none()
    if db_job is not None and not db_job.timeline_tiles_ready:
        db_job.timeline_tiles_ready = True
        await session.commit()

    return {"status": "preparing", "timeline_tiles_ready": True}
```

This returns immediately. The frontend polls `/prepare-status` for progress. The background thread reads only eagerly-loaded attributes from the `job` object (`id`, `start_timestamp`, `end_timestamp`, `hydrophone_id`, `local_cache_path`) — no lazy relationships are accessed.

### Step 5: Add GET /prepare-status endpoint

- [ ] Add a new endpoint after the prepare endpoint:

```python
@router.get("/prepare-status")
async def prepare_status(
    job_id: str,
    session: SessionDep,
    settings: SettingsDep,
):
    job = await _get_job_or_404(session, job_id)
    duration = _job_duration(job)
    tile_cache = TimelineTileCache(
        cache_dir=settings.storage_root / "timeline_cache",
        max_jobs=settings.timeline_cache_max_jobs,
    )
    status = {}
    for zoom in ZOOM_LEVELS:
        total = tile_count(zoom, duration)
        rendered = tile_cache.tile_count_for_zoom(job.id, zoom)
        status[zoom] = {"total": total, "rendered": min(rendered, total)}
    return status
```

### Step 6: Run tests

```bash
uv run pytest tests/integration/test_timeline_api.py -v
```

### Step 7: Commit

```bash
git add src/humpback/api/routers/timeline.py tests/integration/test_timeline_api.py
git commit -m "feat: prepare all zoom levels with progress status endpoint"
```

---

## Task 3: MP3 Audio Encoding and Larger Segments

Add MP3 format option to the audio endpoint and increase max duration to 600s.

**Files:**
- Modify: `src/humpback/api/routers/timeline.py:188-203` (`_encode_wav`), `297-333` (audio endpoint)
- Test: `tests/integration/test_timeline_api.py`
- Test: `tests/unit/test_timeline_audio.py`

### Step 1: Write failing tests

- [ ] Add to `tests/integration/test_timeline_api.py`:

```python
def test_audio_endpoint_mp3_format(client, completed_hydrophone_job):
    """GET /audio with format=mp3 should return audio/mpeg content."""
    job_id = completed_hydrophone_job
    resp = client.get(
        f"/classifier/detection-jobs/{job_id}/timeline/audio",
        params={"start_sec": 1000000, "duration_sec": 10, "format": "mp3"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "audio/mpeg"
    # MP3 files start with 0xFF 0xFB or ID3 tag
    assert resp.content[:3] == b"ID3" or resp.content[0] == 0xFF


def test_audio_endpoint_600s_accepted(client, completed_hydrophone_job):
    """GET /audio should accept duration_sec up to 600."""
    job_id = completed_hydrophone_job
    resp = client.get(
        f"/classifier/detection-jobs/{job_id}/timeline/audio",
        params={"start_sec": 1000000, "duration_sec": 600},
    )
    assert resp.status_code == 200


def test_audio_endpoint_601s_rejected(client, completed_hydrophone_job):
    """GET /audio should reject duration_sec > 600."""
    job_id = completed_hydrophone_job
    resp = client.get(
        f"/classifier/detection-jobs/{job_id}/timeline/audio",
        params={"start_sec": 1000000, "duration_sec": 601},
    )
    assert resp.status_code == 400
```

- [ ] Run to verify they fail:

```bash
uv run pytest tests/integration/test_timeline_api.py -k "mp3 or 600 or 601" -v
```

### Step 2: Add _encode_mp3 helper

- [ ] Add after the existing `_encode_wav()` function (around line 203):

```python
def _encode_mp3(audio: np.ndarray, sample_rate: int) -> bytes:
    """Encode float32 audio to MP3 via ffmpeg subprocess."""
    import subprocess
    import tempfile

    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak * 0.95
    pcm = (audio * 32767).astype(np.int16)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_f:
        wav_path = wav_f.name
        _write_wav_file(wav_f, pcm, sample_rate)

    mp3_path = wav_path.replace(".wav", ".mp3")
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", wav_path,
                "-codec:a", "libmp3lame", "-b:a", "128k", "-ac", "1",
                mp3_path,
            ],
            capture_output=True,
            check=True,
        )
        return Path(mp3_path).read_bytes()
    finally:
        Path(wav_path).unlink(missing_ok=True)
        Path(mp3_path).unlink(missing_ok=True)


def _write_wav_file(f, pcm: np.ndarray, sample_rate: int) -> None:
    """Write raw PCM data as a WAV file."""
    import struct

    data = pcm.tobytes()
    f.write(b"RIFF")
    f.write(struct.pack("<I", 36 + len(data)))
    f.write(b"WAVE")
    f.write(b"fmt ")
    f.write(struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16))
    f.write(b"data")
    f.write(struct.pack("<I", len(data)))
    f.write(data)
```

Note: The existing `_encode_wav()` uses Python's `wave` module with `io.BytesIO`. The `_encode_mp3` helper needs a temp WAV file on disk because ffmpeg reads files. Use `_write_wav_file` for that disk path; alternatively, reuse the existing `_encode_wav` to produce WAV bytes, write them to a temp file, then ffmpeg-convert. Either approach works — the implementer should choose whichever avoids duplication in the existing code.

### Step 3: Update audio endpoint

- [ ] Modify the audio endpoint (lines 297-333):
  - Add `format: str = Query("wav", regex="^(wav|mp3)$")` parameter
  - Change max duration from 120 to 600
  - Branch on format for encoding and content type:

```python
@router.get("/audio")
async def get_audio(
    job_id: str,
    session: SessionDep,
    settings: SettingsDep,
    start_sec: float = Query(
        ..., description="Timeline-absolute start position (epoch seconds)"
    ),
    duration_sec: float = Query(..., gt=0, le=600, description="Duration in seconds"),
    format: str = Query("wav", pattern="^(wav|mp3)$"),
) -> Response:
    """Return audio for an arbitrary timeline position (WAV or MP3)."""
    if duration_sec > 600.0:
        raise HTTPException(400, "Maximum audio duration is 600 seconds")

    job = await _get_job_or_404(session, job_id)

    if not job.hydrophone_id:
        raise HTTPException(
            400, "Audio endpoint is only available for hydrophone detection jobs"
        )

    from humpback.processing.timeline_audio import resolve_timeline_audio

    audio = await asyncio.to_thread(
        resolve_timeline_audio,
        hydrophone_id=job.hydrophone_id,
        local_cache_path=job.local_cache_path or "",
        job_start_timestamp=job.start_timestamp or 0.0,
        job_end_timestamp=job.end_timestamp or 0.0,
        start_sec=start_sec,
        duration_sec=duration_sec,
        target_sr=32000,
        noaa_cache_path=settings.noaa_cache_path,
    )

    if format == "mp3":
        data = _encode_mp3(audio, 32000)
        return Response(content=data, media_type="audio/mpeg")
    else:
        data = _encode_wav(audio, sample_rate=32000)
        return Response(content=data, media_type="audio/wav")
```

Note: This matches the existing endpoint patterns exactly — `asyncio.to_thread` with keyword args, `SessionDep`/`SettingsDep`, and `await _get_job_or_404(session, job_id)` argument order.

### Step 4: Run tests

```bash
uv run pytest tests/integration/test_timeline_api.py -v
```

### Step 5: Commit

```bash
git add src/humpback/api/routers/timeline.py tests/integration/test_timeline_api.py
git commit -m "feat: MP3 audio format and 600s max duration for timeline"
```

---

## Task 4: Positive-Only Detection Label Bars with Hover Tooltips

Replace full-rectangle overlays with narrow vertical bars for humpback/orca only.

**Files:**
- Modify: `frontend/src/components/timeline/DetectionOverlay.tsx` (full rewrite)
- Modify: `frontend/src/components/timeline/SpectrogramViewport.tsx:388-397` (overlay sizing)
- Test: `frontend/e2e/timeline.spec.ts`

### Step 1: Rewrite DetectionOverlay.tsx

- [ ] Replace `frontend/src/components/timeline/DetectionOverlay.tsx` with:

```tsx
import React, { useState } from "react";
import { DetectionRow, ZoomLevel } from "../../api/types";
import { TILE_WIDTH_PX, TILE_DURATION, COLORS } from "./constants";

interface DetectionOverlayProps {
  detections: DetectionRow[];
  jobStart: number;
  centerTimestamp: number;
  zoomLevel: ZoomLevel;
  width: number;
  height: number;
  visible: boolean;
  onDetectionClick?: (row: DetectionRow, x: number, y: number) => void;
}

const POSITIVE_COLORS: Record<string, string> = {
  humpback: "rgba(64, 224, 192, 0.25)",
  orca: "rgba(224, 176, 64, 0.25)",
};

function getPositiveLabel(row: DetectionRow): string | null {
  if (row.humpback === 1) return "humpback";
  if (row.orca === 1) return "orca";
  return null;
}

function formatTime(epoch: number): string {
  const d = new Date(epoch * 1000);
  return d.toISOString().slice(11, 19) + " UTC";
}

export function DetectionOverlay({
  detections,
  jobStart,
  centerTimestamp,
  zoomLevel,
  width,
  height,
  visible,
  onDetectionClick,
}: DetectionOverlayProps) {
  const [tooltip, setTooltip] = useState<{
    x: number;
    y: number;
    label: string;
    startTime: string;
    endTime: string;
    avgConf: string;
    peakConf: string;
  } | null>(null);

  if (!visible || !width || !height) return null;

  const pxPerSec = TILE_WIDTH_PX / TILE_DURATION[zoomLevel];

  const positiveRows = detections
    .map((row) => ({ row, label: getPositiveLabel(row) }))
    .filter(
      (r): r is { row: DetectionRow; label: string } => r.label !== null
    );

  return (
    <div
      data-testid="detection-overlay"
      style={{
        position: "absolute",
        top: 0,
        left: 0,
        width,
        height,
        overflow: "hidden",
        zIndex: 5,
      }}
    >
      {positiveRows.map(({ row, label }) => {
        const startEpoch = jobStart + row.start_sec;
        const endEpoch = jobStart + row.end_sec;
        const x = (startEpoch - centerTimestamp) * pxPerSec + width / 2;
        const barWidth = Math.max(2, (row.end_sec - row.start_sec) * pxPerSec);

        if (x + barWidth < 0 || x > width) return null;

        const color = POSITIVE_COLORS[label] || POSITIVE_COLORS.humpback;

        return (
          <div
            key={`${row.start_sec}-${label}`}
            style={{
              position: "absolute",
              left: x,
              top: 0,
              width: barWidth,
              height: "100%",
              backgroundColor: color,
              cursor: "pointer",
            }}
            onMouseEnter={(e) => {
              const rect = e.currentTarget.getBoundingClientRect();
              setTooltip({
                x: rect.left + rect.width / 2,
                y: rect.top,
                label: label.charAt(0).toUpperCase() + label.slice(1),
                startTime: formatTime(startEpoch),
                endTime: formatTime(endEpoch),
                avgConf: (row.avg_confidence ?? 0).toFixed(3),
                peakConf: (row.peak_confidence ?? 0).toFixed(3),
              });
            }}
            onMouseLeave={() => setTooltip(null)}
            onClick={(e) => {
              if (onDetectionClick) {
                onDetectionClick(row, e.clientX, e.clientY);
              }
            }}
          />
        );
      })}

      {tooltip && (
        <div
          style={{
            position: "fixed",
            left: tooltip.x,
            top: tooltip.y - 8,
            transform: "translate(-50%, -100%)",
            background: COLORS.bg,
            border: `1px solid ${COLORS.border}`,
            borderRadius: 6,
            padding: "6px 10px",
            fontSize: 12,
            color: COLORS.text,
            pointerEvents: "none",
            zIndex: 60,
            whiteSpace: "nowrap",
          }}
        >
          <div style={{ fontWeight: 600, marginBottom: 2 }}>
            {tooltip.label}
          </div>
          <div>
            {tooltip.startTime} — {tooltip.endTime}
          </div>
          <div>
            Confidence: {tooltip.avgConf} avg / {tooltip.peakConf} peak
          </div>
        </div>
      )}
    </div>
  );
}
```

### Step 2: Extend overlay in SpectrogramViewport

- [ ] In `frontend/src/components/timeline/SpectrogramViewport.tsx`, find where `DetectionOverlay` is rendered (around lines 388-397). The overlay's `height` prop should include the confidence strip height so the bars span both the spectrogram canvas and the strip below it. Update the `height` prop from `canvasHeight` to `canvasHeight + CONFIDENCE_STRIP_HEIGHT`.

- [ ] Also pass `onDetectionClick` through to `DetectionOverlay` to open the existing `DetectionPopover` on bar click.

### Step 3: Add Playwright tests

- [ ] Add to `frontend/e2e/timeline.spec.ts`:

```typescript
test("label overlay shows only positive detections as bars", async ({
  page,
}) => {
  // Setup mocks with mixed detection rows
  await setupTimelineMocks(page);
  await page.goto("/app/classifier/timeline/1");

  // Enable labels
  const labelsBtn = page.getByRole("button", { name: /labels/i });
  await labelsBtn.click();

  // Verify only positive label bars appear
  const overlay = page.locator('[data-testid="detection-overlay"]');
  await expect(overlay).toBeVisible();

  // Bars should exist for humpback/orca rows only
  // No bars for ship or background rows
});

test("label bar shows tooltip on hover", async ({ page }) => {
  await setupTimelineMocks(page);
  await page.goto("/app/classifier/timeline/1");

  const labelsBtn = page.getByRole("button", { name: /labels/i });
  await labelsBtn.click();

  // Hover first detection bar
  const bar = page.locator('[data-testid="detection-overlay"] > div').first();
  await bar.hover();

  // Tooltip should show label type and confidence
  await expect(page.locator("text=Confidence")).toBeVisible();
});
```

### Step 4: Run frontend type check

```bash
cd frontend && npx tsc --noEmit
```

### Step 5: Commit

```bash
git add frontend/src/components/timeline/DetectionOverlay.tsx frontend/src/components/timeline/SpectrogramViewport.tsx frontend/e2e/timeline.spec.ts
git commit -m "feat: positive-only detection label bars with hover tooltips"
```

---

## Task 5: Audio-Authoritative Playhead Sync

Replace RAF-driven playhead with audio element's `currentTime` as source of truth.

**Files:**
- Modify: `frontend/src/components/timeline/TimelineViewer.tsx:20-96` (state, audio setup, RAF loop)
- Modify: `frontend/src/components/timeline/constants.ts:61` (AUDIO_PREFETCH_SEC)

### Step 1: Update constants

- [ ] In `frontend/src/components/timeline/constants.ts`, change:

```typescript
// Line 61: was 30
export const AUDIO_PREFETCH_SEC = 300;

// Add after line 62:
export const AUDIO_FORMAT = "mp3";
```

### Step 2: Rewrite audio and playback logic in TimelineViewer

- [ ] In `frontend/src/components/timeline/TimelineViewer.tsx`:

**Add new state** (near line 26):

```typescript
const [playbackOriginEpoch, setPlaybackOriginEpoch] = useState(0);
```

**Add second audio ref** (near line 28-29):

```typescript
const audioRefA = useRef<HTMLAudioElement>(null);
const audioRefB = useRef<HTMLAudioElement>(null);
const activeRef = useRef<"A" | "B">("A");
```

Remove the old single `audioRef`.

**Helper to get active/standby refs:**

```typescript
const getActiveAudio = () =>
  activeRef.current === "A" ? audioRefA.current : audioRefB.current;
const getStandbyAudio = () =>
  activeRef.current === "A" ? audioRefB.current : audioRefA.current;
```

**Replace the audio setup effect** (lines 40-76) with double-buffer logic:

```typescript
// Load and play a chunk on the active element
const loadChunk = useCallback(
  (startEpoch: number, element: HTMLAudioElement) => {
    element.src = timelineAudioUrl(jobId, startEpoch, AUDIO_PREFETCH_SEC, AUDIO_FORMAT);
    element.playbackRate = speed;
    element.load();
  },
  [jobId, speed]
);

// When play starts, load first chunk and prefetch next
useEffect(() => {
  const active = getActiveAudio();
  if (!active) return;

  if (isPlaying) {
    const origin = centerTimestamp;
    setPlaybackOriginEpoch(origin);
    loadChunk(origin, active);

    const onCanPlay = () => {
      active.play().catch(() => {});
      // Prefetch next chunk into standby
      const standby = getStandbyAudio();
      if (standby) {
        loadChunk(origin + AUDIO_PREFETCH_SEC, standby);
      }
    };
    active.addEventListener("canplay", onCanPlay, { once: true });
    return () => active.removeEventListener("canplay", onCanPlay);
  } else {
    active.pause();
  }
}, [isPlaying]); // eslint-disable-line react-hooks/exhaustive-deps

// Handle chunk end — swap to standby
useEffect(() => {
  const handleEnded = () => {
    const standby = getStandbyAudio();
    if (!standby || standby.readyState < 3) {
      // Standby not ready — reload from current position (gap fallback)
      const active = getActiveAudio();
      if (active) {
        const newOrigin = playbackOriginEpoch + (active.currentTime || AUDIO_PREFETCH_SEC);
        setPlaybackOriginEpoch(newOrigin);
        loadChunk(newOrigin, active);
        active.addEventListener("canplay", () => active.play().catch(() => {}), { once: true });
      }
      return;
    }

    // Swap
    const newOrigin = playbackOriginEpoch + AUDIO_PREFETCH_SEC;
    setPlaybackOriginEpoch(newOrigin);
    activeRef.current = activeRef.current === "A" ? "B" : "A";
    standby.play().catch(() => {});

    // Prefetch next chunk into the now-idle element
    const nowIdle = getStandbyAudio();
    if (nowIdle) {
      loadChunk(newOrigin + AUDIO_PREFETCH_SEC, nowIdle);
    }
  };

  const a = audioRefA.current;
  const b = audioRefB.current;
  a?.addEventListener("ended", handleEnded);
  b?.addEventListener("ended", handleEnded);
  return () => {
    a?.removeEventListener("ended", handleEnded);
    b?.removeEventListener("ended", handleEnded);
  };
}, [playbackOriginEpoch, speed]); // eslint-disable-line react-hooks/exhaustive-deps
```

**Replace the RAF loop** (lines 78-96) with audio-authoritative version:

```typescript
useEffect(() => {
  if (!isPlaying) return;
  let raf: number;
  const tick = () => {
    const active = getActiveAudio();
    if (active && !active.paused) {
      setCenterTimestamp(playbackOriginEpoch + active.currentTime);
    }
    raf = requestAnimationFrame(tick);
  };
  raf = requestAnimationFrame(tick);
  return () => cancelAnimationFrame(raf);
}, [isPlaying, playbackOriginEpoch]);
```

**Sync playback rate:**

```typescript
useEffect(() => {
  const a = audioRefA.current;
  const b = audioRefB.current;
  if (a) a.playbackRate = speed;
  if (b) b.playbackRate = speed;
}, [speed]);
```

**Update JSX** — replace the single `<audio>` element (line 215-216) with two:

```tsx
<audio ref={audioRefA} preload="auto" style={{ display: "none" }} />
<audio ref={audioRefB} preload="auto" style={{ display: "none" }} />
```

### Step 3: Update API client

- [ ] In `frontend/src/api/client.ts`, update `timelineAudioUrl` (lines 585-591) to accept a `format` parameter:

```typescript
export function timelineAudioUrl(
  jobId: number | string,
  startSec: number,
  durationSec = 300,
  format = "mp3"
): string {
  const params = new URLSearchParams({
    start_sec: startSec.toString(),
    duration_sec: durationSec.toString(),
    format,
  });
  return `${BASE}/classifier/detection-jobs/${jobId}/timeline/audio?${params}`;
}
```

### Step 4: Run frontend type check

```bash
cd frontend && npx tsc --noEmit
```

### Step 5: Commit

```bash
git add frontend/src/components/timeline/TimelineViewer.tsx frontend/src/components/timeline/constants.ts frontend/src/api/client.ts
git commit -m "feat: audio-authoritative playhead with double-buffered MP3 playback"
```

---

## Task 6: Frontend Prepare-Status Polling and Progress Indicator

Wire up the frontend to trigger background caching and show progress.

**Files:**
- Modify: `frontend/src/api/client.ts:598-602` (add fetchPrepareStatus)
- Modify: `frontend/src/api/types.ts:855-864` (add PrepareStatusResponse)
- Modify: `frontend/src/hooks/queries/useTimeline.ts` (add usePrepareStatus hook)
- Modify: `frontend/src/components/timeline/TimelineViewer.tsx` (trigger prepare, show progress)

### Step 1: Add API types and client function

- [ ] In `frontend/src/api/types.ts`, add after `TimelineConfidenceResponse`:

```typescript
export interface ZoomProgress {
  total: number;
  rendered: number;
}

export type PrepareStatusResponse = Record<ZoomLevel, ZoomProgress>;
```

- [ ] In `frontend/src/api/client.ts`, add after `prepareTimelineTiles`:

```typescript
export async function fetchPrepareStatus(
  jobId: number | string
): Promise<Record<string, { total: number; rendered: number }>> {
  const resp = await fetch(
    `${BASE}/classifier/detection-jobs/${jobId}/timeline/prepare-status`
  );
  if (!resp.ok) throw new Error(`prepare-status ${resp.status}`);
  return resp.json();
}
```

### Step 2: Add query hook

- [ ] In `frontend/src/hooks/queries/useTimeline.ts`, add:

```typescript
export function usePrepareStatus(jobId: number | string, enabled: boolean) {
  return useQuery({
    queryKey: ["timelinePrepareStatus", jobId],
    queryFn: () => fetchPrepareStatus(jobId),
    enabled,
    refetchInterval: enabled ? 3000 : false, // Poll every 3s while caching
  });
}
```

### Step 3: Wire into TimelineViewer

- [ ] In `frontend/src/components/timeline/TimelineViewer.tsx`:

Add a state for tracking whether caching is complete:

```typescript
const [cacheComplete, setCacheComplete] = useState(false);
```

Trigger prepare on mount and poll status:

```typescript
const prepareMutation = usePrepareTimeline();
const { data: prepareStatus } = usePrepareStatus(jobId, !cacheComplete);

// Trigger background caching on mount
useEffect(() => {
  prepareMutation.mutate(jobId);
}, [jobId]); // eslint-disable-line react-hooks/exhaustive-deps

// Check if caching is complete
useEffect(() => {
  if (!prepareStatus) return;
  const allDone = Object.values(prepareStatus).every(
    (z) => z.rendered >= z.total
  );
  if (allDone) setCacheComplete(true);
}, [prepareStatus]);
```

Add a subtle progress indicator in the header area (pass to `TimelineHeader` or render inline):

```tsx
{!cacheComplete && prepareStatus && (
  <div style={{
    position: "absolute", top: 4, right: 16,
    fontSize: 11, color: COLORS.textMuted, zIndex: 10,
  }}>
    Caching tiles: {
      Object.entries(prepareStatus)
        .filter(([, z]) => z.rendered < z.total)
        .map(([zoom, z]) => `${zoom} ${z.rendered}/${z.total}`)
        .join(", ")
    }
  </div>
)}
```

### Step 4: Run frontend type check

```bash
cd frontend && npx tsc --noEmit
```

### Step 5: Commit

```bash
git add frontend/src/api/client.ts frontend/src/api/types.ts frontend/src/hooks/queries/useTimeline.ts frontend/src/components/timeline/TimelineViewer.tsx
git commit -m "feat: background tile caching with progress indicator"
```

---

## Task 7: Documentation Updates

Update project documentation to reflect all changes.

**Files:**
- Modify: `CLAUDE.md`

### Step 1: Update CLAUDE.md

- [ ] **§8.6 Runtime Configuration** — add `timeline_cache_max_jobs` / `HUMPBACK_TIMELINE_CACHE_JOBS` to the config table with description: "Max detection jobs with fully cached timeline tiles. LRU eviction removes the oldest job when exceeded. Default 15 (~8-16 GB disk)."

- [ ] **§8.5 Storage Layout** — add note about MP3 audio format option: timeline audio endpoint supports `format=mp3` for compressed playback.

- [ ] **§9.1 Implemented Capabilities** — update timeline viewer entry to mention: background tile pre-caching for all zoom levels, positive-only detection label bars with hover tooltips, audio-authoritative playhead sync, gapless double-buffered MP3 playback.

### Step 2: Commit

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md for timeline viewer enhancements"
```

---

## Task 8: Verification

Run all verification gates before claiming completion.

**Files:** None (verification only)

### Step 1: Backend linting and type checking

```bash
uv run ruff format --check src/humpback/processing/timeline_cache.py src/humpback/api/routers/timeline.py src/humpback/config.py
uv run ruff check src/humpback/processing/timeline_cache.py src/humpback/api/routers/timeline.py src/humpback/config.py
uv run pyright src/humpback/processing/timeline_cache.py src/humpback/api/routers/timeline.py src/humpback/config.py
```

### Step 2: Backend tests

```bash
uv run pytest tests/ -v
```

### Step 3: Frontend type check

```bash
cd frontend && npx tsc --noEmit
```

### Step 4: Frontend E2E tests

```bash
cd frontend && npx playwright test e2e/timeline.spec.ts
```

### Step 5: Fix any issues found and re-run

Iterate until all checks pass.
