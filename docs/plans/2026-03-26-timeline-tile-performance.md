# Timeline Tile Performance Implementation Plan

**Goal:** Reduce timeline viewer startup latency and cache miss cost by preparing only the immediately useful tiles first and reusing timeline/decode work across tile renders.
**Spec:** `docs/specs/2026-03-26-timeline-tile-performance-design.md`

---

### Task 1: Add startup-scoped timeline preparation

**Files:**
- Modify: `src/humpback/api/routers/timeline.py`
- Modify: `frontend/src/api/client.ts`
- Modify: `frontend/src/hooks/queries/useTimeline.ts`
- Modify: `frontend/src/components/classifier/HydrophoneTab.tsx`
- Modify: `frontend/src/components/timeline/TimelineViewer.tsx`
- Modify: `tests/integration/test_timeline_api.py`

**Acceptance criteria:**
- [x] `POST /classifier/detection-jobs/{job_id}/timeline/prepare` supports a startup-scoped mode that targets only a bounded set of tiles around a requested center timestamp and zoom level
- [x] The default `Timeline` button path uses the startup-scoped mode rather than an implicit full all-zoom warmup
- [x] The startup-scoped mode prioritizes the requested zoom level before any optional coarse zoom levels
- [x] Existing full-cache behavior remains available through an explicit mode instead of being the default open-path behavior
- [x] Startup-scoped prepare still uses exactly one shared `ref_db` per job and does not reintroduce per-tile normalization
- [x] `prepare-status` reports progress for the active target set in a way that matches the new startup-scoped semantics

**Tests needed:**
- Add integration coverage that the default prepare path schedules only a bounded startup tile set
- Add integration coverage that explicit full mode still reports all-zoom work
- Add regression coverage that startup-scoped prepare continues to use the shared job-level `ref_db`
- Add or update a frontend timeline smoke path so opening the timeline still navigates immediately and shows cache progress

---

### Task 2: Reuse job timeline metadata, decoded PCM, and shared `ref_db` across renders

**Files:**
- Modify: `src/humpback/processing/timeline_audio.py`
- Modify: `src/humpback/processing/timeline_cache.py`
- Modify: `src/humpback/classifier/s3_stream.py`
- Modify: `src/humpback/api/routers/timeline.py`
- Modify: `tests/unit/test_timeline_audio.py`
- Modify: `tests/unit/test_timeline_cache.py`
- Modify: `tests/integration/test_timeline_api.py`

**Acceptance criteria:**
- [x] Timeline preparation builds or loads a reusable job-level manifest instead of rebuilding the provider timeline for every tile
- [x] Shared `ref_db` computation is preserved but reuses the manifest and hot decoded-PCM caches instead of rebuilding work on each startup path
- [x] Adjacent tile renders within the same prepare batch can reuse decoded segment audio when they touch the same source segments and target sample rate
- [x] Reuse is bounded by explicit cache limits so memory growth stays controlled
- [x] Single-tile miss rendering can consume the same manifest helpers and cached `ref_db` without requiring a full prepare run first
- [x] Existing cache locking and idempotency guarantees remain intact across processes

**Tests needed:**
- Add unit coverage that repeated tile audio resolution for the same job reuses manifest data
- Add unit or integration coverage that cached `ref_db` values are reused and not recomputed when present
- Add unit coverage that decoded segment reuse is bounded and keyed correctly
- Add integration coverage that repeated tile requests do not rebuild the timeline unnecessarily when manifest data already exists

---

### Task 3: Add bounded in-memory hot caches and threaded preparation

**Files:**
- Modify: `src/humpback/api/routers/timeline.py`
- Modify: `src/humpback/processing/timeline_audio.py`
- Modify: `src/humpback/processing/timeline_cache.py`
- Modify: `src/humpback/config.py`
- Modify: `tests/unit/test_timeline_cache.py`
- Modify: `tests/integration/test_timeline_api.py`

**Acceptance criteria:**
- [x] The server keeps a bounded in-memory cache for hot timeline manifests and decoded/resampled PCM used by tile preparation
- [x] Timeline startup/full prepare can use a small configurable worker pool to render targeted tile sets concurrently
- [x] Worker-count and in-memory cache limits are configurable and have safe defaults
- [x] Concurrency does not change the chosen shared `ref_db`, violate prepare-lock idempotency, or duplicate same-job work
- [x] Lossy MP3 is not used as an intermediate cache for spectrogram tile rendering

**Tests needed:**
- Add unit coverage that in-memory cache bounds are enforced
- Add integration coverage that concurrent prepare workers remain idempotent and keep shared-normalization behavior stable

---

### Task 4: Smooth the miss path with same-zoom neighbor warming

**Files:**
- Modify: `src/humpback/api/routers/timeline.py`
- Modify: `src/humpback/processing/timeline_cache.py`
- Modify: `tests/integration/test_timeline_api.py`

**Acceptance criteria:**
- [x] When an uncached tile is requested, the requested tile still renders immediately
- [x] After serving a miss, the backend can queue a bounded number of adjacent same-zoom tiles without blocking the response
- [x] Neighbor warming stays idempotent under repeated requests and respects the existing per-job prepare lock coordination
- [x] Neighbor warming never expands into an unbounded all-zoom prepare

**Tests needed:**
- Add integration coverage that a tile miss can trigger bounded neighbor warming
- Add integration coverage that concurrent misses for nearby tiles do not duplicate background work

---

### Verification

Run in order after all tasks:
1. `uv run ruff format --check src/humpback/api/routers/timeline.py src/humpback/processing/timeline_audio.py src/humpback/processing/timeline_cache.py src/humpback/classifier/s3_stream.py src/humpback/config.py tests/unit/test_timeline_audio.py tests/unit/test_timeline_cache.py tests/unit/test_timeline_tiles.py tests/integration/test_timeline_api.py`
2. `uv run ruff check src/humpback/api/routers/timeline.py src/humpback/processing/timeline_audio.py src/humpback/processing/timeline_cache.py src/humpback/classifier/s3_stream.py src/humpback/config.py tests/unit/test_timeline_audio.py tests/unit/test_timeline_cache.py tests/unit/test_timeline_tiles.py tests/integration/test_timeline_api.py`
3. `uv run pyright src/humpback/api/routers/timeline.py src/humpback/processing/timeline_audio.py src/humpback/processing/timeline_cache.py src/humpback/classifier/s3_stream.py src/humpback/config.py tests/unit/test_timeline_audio.py tests/unit/test_timeline_cache.py tests/unit/test_timeline_tiles.py tests/integration/test_timeline_api.py`
4. `uv run pytest tests/unit/test_timeline_audio.py tests/unit/test_timeline_cache.py tests/unit/test_timeline_tiles.py tests/integration/test_timeline_api.py`
5. `cd frontend && npx tsc --noEmit`
6. `cd frontend && npx playwright test e2e/timeline.spec.ts`
