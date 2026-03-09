# Development Plans

---

## Active

(none)

---

## Recently Completed

# Plan: Classifier/Hydrophone Extract — Hydrophone-Partitioned Output Paths

## Outcome (2026-03-09)

- Updated hydrophone labeled-sample extraction paths to include hydrophone short label
  (`hydrophone_id`) under both positive and negative roots:
  `{positive|negative}_root/{hydrophone_id}/{label}/YYYY/MM/DD/*.wav`.
- Preserved local (non-hydrophone) extraction path behavior.
- Added unit coverage for hydrophone positive/negative path routing, including a guard
  that old non-partitioned hydrophone negative paths are no longer used.

## Verification

- `uv run pytest tests/unit/test_extractor.py -q` passed (`21` passed).
- `uv run pytest tests/` passed (`390` passed).

---

# Plan: Hydrophone Tab — Playback Timestamp Mapping + Saved-Label Extract Activation

## Outcome (2026-03-08)

- Implemented shared hydrophone stream-offset audio-slice resolver with anchor order:
  first available folder timestamp, then legacy `job.start_timestamp`.
- Switched hydrophone extraction to the same resolver path and passed detection
  job stream bounds (`start_timestamp`, `end_timestamp`) through worker plumbing.
- Updated Hydrophone tab extract enablement to use saved labels on the expanded
  completed job only; Extract dialog now targets that single job.

## Verification

- `uv run pytest tests/` passed (`389` passed).
- `cd frontend && npx tsc --noEmit` passed.
- `cd frontend && npx playwright test` ran:
  - New hydrophone regression test passed (`frontend/e2e/hydrophone-extract.spec.ts`).
  - Existing unrelated failures remain in classifier training selectors
    (`frontend/e2e/classifier-training.spec.ts`) and one slider-fill test
    (`frontend/e2e/detection-hysteresis.spec.ts`).

---

# Plan: HydrophoneTab — Live Detection Content + Save/Extract Labels

## Context

The Hydrophone detection UI has two gaps compared to the Detect tab:
1. **No live detection results** — the active job panel shows segment progress but never displays the actual detection rows as they arrive. Expanding rows is restricted to completed jobs only.
2. **Missing action buttons** — "Save Labels" and "Extract Labeled Samples" buttons are absent from the previous jobs toolbar, unlike the Detect tab.

## Files to Modify

| File | Action |
|------|--------|
| `frontend/src/components/classifier/ExtractDialog.tsx` | **Create** — extract from DetectionTab |
| `frontend/src/components/classifier/DetectionTab.tsx` | Remove ExtractDialog definition, add import |
| `frontend/src/components/classifier/HydrophoneTab.tsx` | Main changes — live content + buttons |

No backend changes needed. All endpoints already work for hydrophone jobs.

---

### Step 1: Extract `ExtractDialog` into shared component

**Create** `frontend/src/components/classifier/ExtractDialog.tsx`
- Move lines 864-988 from `DetectionTab.tsx` (the `ExtractDialog` function + its imports)
- Export `ExtractDialog` as named export
- Needs imports: `useState`, `useEffect`, `Input`, `Button`, `FolderOpen`, Dialog primitives, `FolderBrowser`, `useExtractionSettings`, `useExtractLabeledSamples`

**Modify** `DetectionTab.tsx`
- Remove the `ExtractDialog` function (lines 864-988)
- Remove now-unused imports: `useExtractionSettings`, `FolderOpen` (if not used elsewhere), Dialog primitives (keep if used elsewhere)
- Add: `import { ExtractDialog } from "./ExtractDialog";`

### Step 2: Enable live detection content for running jobs

**In `HydrophoneTab.tsx`:**

**2a.** `HydrophoneJobRow` — expand `canExpand` guard (line 621):
```typescript
// Before:
const canExpand = job.status === "complete" && !!job.output_tsv_path;
// After:
const isRunning = job.status === "running";
const canExpand =
  (job.status === "complete" || (isRunning && (job.segments_processed ?? 0) > 0)) &&
  !!job.output_tsv_path;
```

**2b.** Pass `isRunning` prop from `HydrophoneJobRow` to `HydrophoneContentTable`.

**2c.** `HydrophoneContentTable` — add polling + sort transition:
- Accept `isRunning: boolean` prop
- Change `useDetectionContent(jobId)` → `useDetectionContent(jobId, isRunning ? 3000 : undefined)`
- Initialize sort: `filename/asc` when running, `avg_confidence/desc` when not
- Add `useEffect` to switch sort from filename→confidence when job transitions from running→complete (matching DetectionTab lines 628-634)

**2d.** Active job panel (lines 401-465) — embed content table after alerts:
```typescript
{(activeJob.segments_processed ?? 0) > 0 && activeJob.output_tsv_path && (
  <HydrophoneContentTable
    jobId={activeJob.id}
    isRunning={true}
    playingKey={playingKey}
    onPlay={handlePlay}
    onLabelChange={handleLabelChange}
    labelEdits={labelEdits.get(activeJob.id) ?? null}
  />
)}
```

### Step 3: Switch from auto-save to buffered label editing + add buttons

**In `HydrophoneTab.tsx`:**

**3a.** Add state for buffered editing (replacing auto-save):
```typescript
const extractMutation = useExtractLabeledSamples();
const [labelEdits, setLabelEdits] = useState<
  Map<string, Map<string, Partial<Record<LabelField, number | null>>>>
>(new Map());
const [dirtyJobs, setDirtyJobs] = useState<Set<string>>(new Set());
const [showExtractDialog, setShowExtractDialog] = useState(false);
```

**3b.** Replace `handleLabelChange` (lines 147-163) — buffer edits locally instead of calling `saveLabelsMutation.mutate()` immediately. Match DetectionTab lines 161-175.

**3c.** Add `handleSaveLabels` — batch-save all dirty jobs. Match DetectionTab lines 177-203.

**3d.** Add buttons to toolbar (line 471-483 area). Three buttons in a flex row:
- **Save Labels** — disabled when `dirtyJobs.size === 0` or expanded job is running
- **Extract Labeled Samples** — disabled when `selectedIds.size === 0`, opens `ExtractDialog`
- **Delete** — existing button, unchanged

**3e.** Pass `labelEdits` through `HydrophoneJobRow` → `HydrophoneContentTable`.

**3f.** Update `HydrophoneContentTable.getEffectiveLabel` to check buffered edits first, fall back to server value (matching DetectionTab lines 660-668).

**3g.** Add `ExtractDialog` render next to `BulkDeleteDialog`.

### Step 4: Add Extract status column to previous jobs table

- Add `<th>Extract</th>` column header after "Download"
- Add extract status badge cell in `HydrophoneJobRow` (matching DetectionTab lines 567-575)
- Update `colSpan` in the expanded content row from 9 to 10

### Imports to Add/Update in HydrophoneTab

```typescript
// Add:
import { Save, PackageOpen } from "lucide-react";
import { useExtractLabeledSamples } from "@/hooks/queries/useClassifier";
import { ExtractDialog } from "./ExtractDialog";
```

---

### Verification

1. **Live content during running job:**
   - Start a hydrophone detection job
   - Confirm the active job panel shows detection rows once `segments_processed > 0`
   - Confirm rows update every 3s as new detections arrive
   - Confirm sort switches from filename/asc → confidence/desc on completion

2. **Save Labels:**
   - Expand a completed job, toggle label checkboxes
   - Confirm Save Labels button enables (dirty state)
   - Click Save Labels, confirm it saves and button disables
   - Confirm Save Labels is disabled while a running job is expanded

3. **Extract Labeled Samples:**
   - Select completed jobs via checkboxes
   - Click Extract, confirm dialog opens with path fields
   - Submit extraction, confirm extract_status badge appears

4. **Type-check:** `cd frontend && npx tsc --noEmit`
5. **Existing tests:** `cd frontend && npx playwright test`

---

# Plan: Classifier/Hydrophone Tab — S3 HLS Streaming Detection

## Context

Users want to run classifier detection on historic Orcasound hydrophone audio stored as HLS streams in public S3 buckets. Currently, detection only works on local audio folders. This feature adds a new "Hydrophone" subtab under Classifier that streams `.ts` segments from S3, decodes them in memory, and runs the existing inference pipeline without writing intermediate files to disk.

Reference: [orca-hls-utils](https://github.com/orcasound/orca-hls-utils) for S3 HLS structure (`s3://audio-orcasound-net/{hydrophone_id}/hls/{unix_timestamp}/live.m3u8` + `.ts` segments).

---

## Key Design Decisions

1. **Extend `detection_jobs` table** (not a new table) — hydrophone detection is fundamentally a detection job with a different audio source. Same TSV output, same content table, same label editing. Add nullable hydrophone columns; `audio_folder` becomes nullable.

2. **New `run_hydrophone_detection()` function** — the existing `run_detection()` is tightly coupled to `audio_folder.rglob()`. Create a parallel pipeline in `hydrophone_detector.py` that reuses core components: `resample()`, `slice_windows_with_metadata()`, `extract_logmel_batch()`, `model.embed()`, `merge_detection_events()`, `append_detections_tsv()`.

3. **Custom S3 client** (not orca-hls-utils as a dependency) — orca-hls-utils writes to disk, has no retry logic. Build a focused module using boto3 with botocore adaptive retry. Anonymous access (UNSIGNED signature) matches the public Orcasound bucket.

4. **Fully in-memory processing** — S3 segments decoded via ffmpeg stdin/stdout pipes (`-i pipe:0 ... pipe:1`). No intermediate files for streaming or inference. Audio playback re-fetches from S3 on demand in the audio-slice endpoint (~200ms latency per play, zero disk artifacts).

5. **Orcasound-only for MVP** — hardcode anonymous access to `audio-orcasound-net` bucket. No configurable credentials or custom buckets.

6. **Auto-save labels** — call existing `PUT /detection-jobs/{job_id}/labels` with single-row array on each toggle. No new endpoint needed; purely a frontend behavior change.

7. **Hydrophone config** — hardcoded list in `config.py`, served via `GET /classifier/hydrophones`. Small stable list, no DB table needed for MVP.

8. **Cancel support** — `threading.Event` flag checked between segments. Background asyncio task polls DB for status="canceled" and sets the flag.

---

## Implementation Phases

### Phase 1: Database Migration

**New file:** `alembic/versions/012_hydrophone_detection_columns.py`

Add nullable columns to `detection_jobs`:
```
hydrophone_id       VARCHAR  nullable  -- e.g. "rpi_orcasound_lab"
hydrophone_name     VARCHAR  nullable  -- e.g. "Orcasound Lab"
start_timestamp     FLOAT    nullable  -- unix epoch start
end_timestamp       FLOAT    nullable  -- unix epoch end
segments_processed  INTEGER  nullable  -- .ts segments fetched so far
segments_total      INTEGER  nullable  -- estimated total segments
time_covered_sec    FLOAT    nullable  -- seconds of audio processed
alerts              TEXT     nullable  -- JSON array of {type, message, timestamp}
```

Make `audio_folder` nullable (currently NOT NULL).

Use `op.batch_alter_table("detection_jobs")` for SQLite compatibility.

**Modify:** `src/humpback/models/classifier.py` — add `Mapped[Optional[...]]` fields to `DetectionJob`. Change `audio_folder: Mapped[str]` to `Mapped[Optional[str]]`.

---

### Phase 2: Hydrophone Configuration

**Modify:** `src/humpback/config.py`

```python
ORCASOUND_HYDROPHONES = [
    {"id": "rpi_orcasound_lab", "name": "Orcasound Lab", "location": "San Juan Islands"},
    {"id": "rpi_north_sjc", "name": "North San Juan Channel", "location": "San Juan Channel"},
    {"id": "rpi_port_townsend", "name": "Port Townsend", "location": "Puget Sound"},
    {"id": "rpi_bush_point", "name": "Bush Point", "location": "Whidbey Island"},
]
ORCASOUND_S3_BUCKET = "audio-orcasound-net"
```

---

### Phase 3: S3 Streaming Module

**New file:** `src/humpback/classifier/s3_stream.py`

**`OrcasoundS3Client` class:**
- boto3 client with `Config(signature_version=UNSIGNED, retries={"max_attempts": 5, "mode": "adaptive"})`
- `list_hls_folders(hydrophone_id, start_ts, end_ts)` — list unix-timestamp folder prefixes via `list_objects_v2` with delimiter
- `list_segments(hydrophone_id, folder_ts)` — list `.ts` keys in a folder
- `fetch_segment(key) -> bytes` — download segment bytes (in memory)

**`decode_ts_bytes(ts_bytes, target_sr=32000) -> np.ndarray`:**
- `subprocess.run(["ffmpeg", "-i", "pipe:0", "-f", "wav", "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(target_sr), "pipe:1"], input=ts_bytes, capture_output=True)`
- Parse WAV bytes from stdout, return float32 numpy array
- No disk I/O

**`iter_audio_chunks(client, hydrophone_id, start_ts, end_ts, chunk_seconds=60, target_sr=32000, on_error=None)`:**
- Generator that yields `(chunk_audio: np.ndarray, chunk_start_utc: datetime, segment_count: int)`
- Iterates HLS folders chronologically, fetches `.ts` segments, decodes, accumulates into ~60s chunks
- Calls `on_error({"type": "warning", "message": ..., "timestamp": ...})` on individual segment failures without stopping
- Tracks total segments for progress reporting

**Dependency:** `uv add boto3`

---

### Phase 4: Hydrophone Detection Pipeline

**New file:** `src/humpback/classifier/hydrophone_detector.py`

```python
def run_hydrophone_detection(
    hydrophone_id: str,
    start_timestamp: float,
    end_timestamp: float,
    pipeline: Pipeline,           # sklearn classifier
    model: EmbeddingModel,
    window_size_seconds: float,
    target_sample_rate: int,
    confidence_threshold: float,
    input_format: str = "spectrogram",
    feature_config: dict | None = None,
    hop_seconds: float = 1.0,
    high_threshold: float = 0.70,
    low_threshold: float = 0.45,
    on_chunk_complete: Callable | None = None,  # progress callback
    on_alert: Callable | None = None,           # error/warning callback
    cancel_check: Callable[[], bool] | None = None,
) -> tuple[list[dict], dict]:
```

Processing flow per chunk:
1. Receive ~60s audio chunk from `iter_audio_chunks()`
2. Slice into windows via `slice_windows_with_metadata()`
3. Batch feature extraction (`extract_logmel_batch()` or waveform pass-through)
4. Batch embed (groups of 64)
5. Classify via `pipeline.predict_proba()`
6. Merge events via `merge_detection_events()`
7. Emit detections with synthetic filename: `{chunk_start_utc_iso}.wav` (e.g., `20260301T143000Z.wav`)
8. Call `on_chunk_complete(events, segments_done, segments_total, time_covered_sec)`
9. Check `cancel_check()` — return early if canceled

Reuses from `detector.py`: `merge_detection_events()`, `append_detections_tsv()`, `write_detections_tsv()`

---

### Phase 5: Worker Integration

**Modify:** `src/humpback/workers/classifier_worker.py`

Add `run_hydrophone_detection_job()`:
- Same structure as `run_detection_job()` — load classifier model, load embedding model
- Set up TSV path in detection_dir
- Create cancel event (`threading.Event`)
- Define `on_chunk_complete` callback: append to TSV + schedule DB progress update via `loop.call_soon_threadsafe()`
- Define `on_alert` callback: append alert JSON to DB `alerts` column
- Run via `asyncio.to_thread(run_hydrophone_detection, ...)`
- Start background `_poll_cancel` coroutine that checks DB status every 2s and sets cancel event
- On completion/cancel: write final TSV, update summary, mark complete/canceled

**Modify:** `src/humpback/workers/queue.py`

- `claim_detection_job()`: add filter `DetectionJob.hydrophone_id.is_(None)` to only claim local detection jobs
- Add `claim_hydrophone_detection_job()`: filter `DetectionJob.hydrophone_id.isnot(None)`, same claim pattern
- `recover_stale_jobs()`: already covers all DetectionJob rows (no change needed)

**Modify:** `src/humpback/workers/runner.py`

Add hydrophone detection claim step after regular detection (line ~107):
```python
# Then hydrophone detection jobs
async with session_factory() as session:
    hjob = await claim_hydrophone_detection_job(session)
if hjob:
    logger.info(f"Hydrophone detection job {hjob.id}")
    async with session_factory() as session:
        await run_hydrophone_detection_job(session, hjob, settings, session_factory=session_factory)
    claimed = True
```

Import `claim_hydrophone_detection_job` and `run_hydrophone_detection_job`.

---

### Phase 6: API Endpoints & Schemas

**Modify:** `src/humpback/schemas/classifier.py`

Add:
```python
class HydrophoneInfo(BaseModel):
    id: str; name: str; location: str

class HydrophoneDetectionJobCreate(BaseModel):
    classifier_model_id: str
    hydrophone_id: str
    start_timestamp: float  # unix epoch
    end_timestamp: float
    confidence_threshold: float = 0.5
    hop_seconds: float = 1.0
    high_threshold: float = 0.70
    low_threshold: float = 0.45
```

Extend `DetectionJobOut` with nullable hydrophone fields: `hydrophone_id`, `hydrophone_name`, `start_timestamp`, `end_timestamp`, `segments_processed`, `segments_total`, `time_covered_sec`, `alerts` (parsed from JSON).

**Modify:** `src/humpback/api/routers/classifier.py`

New endpoints:
| Method | Path | Purpose |
|--------|------|---------|
| GET | `/classifier/hydrophones` | List configured hydrophone locations |
| POST | `/classifier/hydrophone-detection-jobs` | Create hydrophone detection job |
| GET | `/classifier/hydrophone-detection-jobs` | List hydrophone detection jobs |
| POST | `/classifier/hydrophone-detection-jobs/{job_id}/cancel` | Cancel running job |

Modify `get_detection_audio_slice()` (line 694): when `job.hydrophone_id` is set, re-fetch from S3 on demand (no disk cache):
- Parse the synthetic filename (UTC ISO timestamp, e.g., `20260301T143000Z.wav`) to determine the chunk's absolute start time
- Compute which HLS folder + `.ts` segment(s) cover the requested `start_sec` to `start_sec + duration_sec` within that chunk
- Fetch segment(s) from S3 via `OrcasoundS3Client`, decode via `decode_ts_bytes()`
- Extract the requested slice, normalize, return as WAV (same response format as existing endpoint)
- ~200ms latency per request (S3 fetch + decode); acceptable for user-initiated playback

**Modify:** `src/humpback/services/classifier_service.py`

Add service functions:
- `create_hydrophone_detection_job()` — validate model exists, hydrophone_id is known, timestamps valid (max 7 day range), high >= low threshold
- `list_hydrophone_detection_jobs()` — filter `hydrophone_id IS NOT NULL`
- `cancel_hydrophone_detection_job()` — set status to "canceled" if currently "running"

**Modify:** `src/humpback/storage.py` — no changes needed (fully in-memory, no hydrophone disk cache).

---

### Phase 7: Frontend — Types & API Client

**Modify:** `frontend/src/api/types.ts`

```typescript
export interface HydrophoneInfo {
  id: string; name: string; location: string;
}

export interface HydrophoneDetectionJobCreate {
  classifier_model_id: string;
  hydrophone_id: string;
  start_timestamp: number;
  end_timestamp: number;
  confidence_threshold?: number;
  hop_seconds?: number;
  high_threshold?: number;
  low_threshold?: number;
}

// Add to FlashAlert:
export interface FlashAlert {
  type: "error" | "warning" | "info";
  message: string;
  timestamp: string;
}
```

Extend `DetectionJob` with nullable hydrophone fields.

**Modify:** `frontend/src/api/client.ts`

Add: `fetchHydrophones()`, `fetchHydrophoneDetectionJobs()`, `createHydrophoneDetectionJob()`, `cancelHydrophoneDetectionJob()`.

---

### Phase 8: Frontend — Query Hooks

**Modify:** `frontend/src/hooks/queries/useClassifier.ts`

Add hooks:
- `useHydrophones()` — one-time fetch
- `useHydrophoneDetectionJobs(refetchInterval?)` — with polling support
- `useCreateHydrophoneDetectionJob()` — invalidates hydrophoneDetectionJobs
- `useCancelHydrophoneDetectionJob()` — invalidates hydrophoneDetectionJobs

---

### Phase 9: Frontend — HydrophoneTab Component

**New file:** `frontend/src/components/classifier/HydrophoneTab.tsx`

Three sections:

#### Section 1: Job Creation Form (Card)
- Hydrophone dropdown (from `useHydrophones()`)
- Start/end datetime pickers (`<input type="datetime-local">`)
- Classifier model selector (reuses `useClassifierModels()`)
- Parameters: confidence threshold, hop seconds, high/low threshold (same controls as DetectionTab)
- "Start Detection" button

#### Section 2: Active Job Panel (Card, only when running/queued job exists)
- Hydrophone name, date range display
- Progress: `"Processed {segments_processed}/{segments_total} segments ({time_covered_sec}s audio)"`
- Progress bar
- **Stop button** → calls `useCancelHydrophoneDetectionJob()`
- **Flash alerts area** (Cloudscape flashbar-style):
  - Renders `job.alerts` array as dismissable banners
  - Color-coded by type: error=red, warning=amber, info=blue
  - Icon + message + timestamp + X dismiss button
  - Dismissed alerts tracked client-side in `Set<index>` state
  - New alerts appear at top

#### Section 3: Previous Jobs Panel
- Table of completed/failed/canceled hydrophone jobs
- Columns: Status | Hydrophone | Date Range | Threshold | Results | Download
- **Expandable rows** with detection content table (reuses exact pattern from DetectionTab lines 602-862)
- Audio playback, keyboard shortcuts (j/k/space/h/s/b), label checkboxes
- **Key difference:** Label toggles immediately call `saveLabelsMutation` with single-row array (auto-save)
- **No "Save Labels" button**, no "Extract Labels" button
- Bulk delete available

**Modify:** `frontend/src/components/classifier/ClassifierTab.tsx`

```typescript
type SubView = "train" | "detect" | "hydrophone";
// Add third tab button + conditional render
```

---

## File Change Summary

| File | Action | Purpose |
|------|--------|---------|
| `alembic/versions/012_hydrophone_detection_columns.py` | **New** | Migration: add hydrophone columns, make audio_folder nullable |
| `src/humpback/classifier/s3_stream.py` | **New** | S3 HLS client with retry, in-memory segment decoding |
| `src/humpback/classifier/hydrophone_detector.py` | **New** | Streaming detection pipeline |
| `frontend/src/components/classifier/HydrophoneTab.tsx` | **New** | Main UI component |
| `src/humpback/models/classifier.py` | Modify | Add hydrophone columns to DetectionJob |
| `src/humpback/config.py` | Modify | Add hydrophone constants |
| `src/humpback/schemas/classifier.py` | Modify | Add hydrophone schemas |
| `src/humpback/services/classifier_service.py` | Modify | Add hydrophone service functions |
| `src/humpback/api/routers/classifier.py` | Modify | Add hydrophone endpoints, extend audio-slice |
| `src/humpback/workers/classifier_worker.py` | Modify | Add `run_hydrophone_detection_job()` |
| `src/humpback/workers/queue.py` | Modify | Add `claim_hydrophone_detection_job()`, filter existing claim |
| `src/humpback/workers/runner.py` | Modify | Add hydrophone job claim step |
| `src/humpback/storage.py` | No change | Fully in-memory, no disk cache |
| `pyproject.toml` | Modify | Add boto3 dependency |
| `frontend/src/api/types.ts` | Modify | Add hydrophone types |
| `frontend/src/api/client.ts` | Modify | Add hydrophone API functions |
| `frontend/src/hooks/queries/useClassifier.ts` | Modify | Add hydrophone hooks |
| `frontend/src/components/classifier/ClassifierTab.tsx` | Modify | Add "Hydrophone" subtab |

---

## Implementation Order

1. `pyproject.toml` + `uv add boto3` (dependency)
2. Phase 1: Migration + model changes
3. Phase 2: Config constants
4. Phase 3: S3 streaming module (testable independently)
5. Phase 4: Hydrophone detector (testable with mock S3)
6. Phase 5: Worker integration
7. Phase 6: API endpoints + schemas
8. Phase 7-8: Frontend types, client, hooks
9. Phase 9: HydrophoneTab component
10. Phase 6 (audio-slice extension): last, after UI is wired up

---

## Verification

1. **Unit test S3 client** — mock boto3, verify folder listing, segment fetching, retry on 503
2. **Unit test hydrophone detector** — mock S3 client + fake embedding model, verify detections, cancel support, alert propagation
3. **Integration test API** — create hydrophone job, verify DB state, list/cancel endpoints
4. **Manual E2E** — start backend + worker, create hydrophone job via UI with real Orcasound data, verify progress updates, flash alerts, detection results, audio playback, label auto-save, stop button
5. **Playwright test** — verify Hydrophone tab renders, form submits, active job panel shows progress, previous jobs panel expandable with content


---

## Backlog

- Explore GPU-accelerated batch processing for large audio libraries
- Add WebSocket push for real-time job status updates (replace polling)
- Investigate multi-model ensemble clustering
- Optimize `/audio/{id}/spectrogram` window fetch path to avoid materializing all windows when only one index is requested (reduce memory/time on long files)

---

## Completed

- Overlap-back windowing (ADR-001)
- In-place folder import (ADR-002)
- Balanced class weights for detection (ADR-003)
- Negative embedding sets for training (ADR-004)
- Multi-agent memory framework migration
- Overlapping window inference + hysteresis event detection (ADR-005)
- Incremental detection rendering with per-file progress (ADR-006)
- Fix escalating false positives: MLP classifier + diagnostics (ADR-007)
- Queue claim hardening + API validation pass (P0-P2): atomic compare-and-set claims for all worker job types, strict input validation, robust Range parsing, overlap-back-aligned spectrogram offsets (ADR-009)
- Multi-model support: grouping, filtering & validation
- Fix TFLite/TF2 vector_dim mismatch: auto-detect from model output
- Optimize TFLite encoding: batch inference via resize_tensor_input + multi-threading + timing instrumentation
- Optimize spectrogram extraction: vectorized STFT via batched `np.fft.rfft` (10.9x feature speedup) + TFLite inference batch_size tuned to 64 (6.2x vs sequential)
- S3 HLS Streaming Detection (Hydrophone Tab): Orcasound hydrophone integration with in-memory S3 streaming, cancel support, flash alerts, auto-save labels
- S3 Caching, UTC Display & WAV Export: CachingS3Client with write-through cache + 404 markers, UTC range display in detection table, WAV export for hydrophone jobs
