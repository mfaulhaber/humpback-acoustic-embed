# Timeline Export Implementation Plan

**Goal:** Export a detection job's timeline as a self-contained static bundle (tiles, MP3 audio, JSON manifest) for a readonly React viewer hosted on S3.
**Spec:** [docs/specs/2026-04-05-timeline-export-design.md](../specs/2026-04-05-timeline-export-design.md)
**Consumer contract:** [docs/specs/2026-04-05-timeline-export-consumer-contract.md](../specs/2026-04-05-timeline-export-consumer-contract.md)

---

### Task 1: Export service function

**Files:**
- Create: `src/humpback/services/timeline_export.py`

**Acceptance criteria:**
- [ ] `export_timeline(job_id, output_dir, db, settings) -> ExportResult` async function
- [ ] Validates job exists, is complete, and is a hydrophone job (has `hydrophone_id`)
- [ ] Validates all tiles at all 6 zoom levels are fully rendered in the tile cache (compares `tile_count_for_zoom` against expected `tile_count` from `timeline_tiles.py`)
- [ ] Creates output directory `{output_dir}/{job_id}/tiles/{zoom}/` and `{output_dir}/{job_id}/audio/`
- [ ] Copies all tile PNGs from the tile cache, preserving `tile_{index:04d}.png` naming
- [ ] Generates 300-second MP3 audio chunks using existing `resolve_timeline_audio()` and the MP3 encoding logic (extract `_encode_mp3` helper from `timeline.py` router into a shared location, or inline the ffmpeg subprocess call)
- [ ] Builds `manifest.json` with: job metadata from DB, tile layout metadata, audio chunk metadata, confidence scores from diagnostics parquet, detection rows from row store, vocalization labels from DB + inference predictions, vocalization types from DB
- [ ] Returns `ExportResult` dataclass with `job_id`, `output_path`, `tile_count`, `audio_chunk_count`, `manifest_size_bytes`
- [ ] Raises clear errors: 404 job not found, 409 job not complete, 409 tiles not prepared

**Tests needed:**
- Unit test with a mock tile cache directory, fake DB records, and stubbed audio resolution
- Test that missing tiles raises the expected error
- Test that manifest JSON matches the consumer contract schema shape
- Test that tile copy produces correctly named files in the right directory structure

---

### Task 2: Extract MP3 encoding to shared utility

**Files:**
- Create: `src/humpback/processing/audio_encoding.py`
- Modify: `src/humpback/api/routers/timeline.py` (import from new module instead of inline)

**Acceptance criteria:**
- [ ] `encode_mp3(audio: np.ndarray, sample_rate: int) -> bytes` function extracted from `_encode_mp3` in `timeline.py`
- [ ] `encode_wav(audio: np.ndarray, sample_rate: int) -> bytes` also extracted alongside
- [ ] Timeline router imports from the new module instead of defining inline
- [ ] Existing timeline audio endpoint behavior unchanged

**Tests needed:**
- Unit test that `encode_mp3` produces valid MP3 bytes (round-trip check with ffprobe or size sanity check)
- Existing timeline API tests still pass

---

### Task 3: Manifest assembly logic

**Files:**
- Modify: `src/humpback/services/timeline_export.py` (manifest builder within the export service)

**Acceptance criteria:**
- [ ] `_build_manifest(job, db, settings) -> dict` assembles the full manifest matching the consumer contract schema
- [ ] `job` section: id, hydrophone_name, hydrophone_id, start_timestamp, end_timestamp, species (from classifier model), window_selection, model_name, model_version
- [ ] `tiles` section: zoom_levels list, tile_size [512, 256], tile_durations dict, tile_counts dict computed from job duration
- [ ] `audio` section: chunk_duration_sec=300, chunk_count computed, format="mp3", sample_rate=32000
- [ ] `confidence` section: window_sec and scores array read from diagnostics parquet using the same bucketing logic as the existing `/confidence` endpoint
- [ ] `detections` section: reads row store, normalizes rows, extracts row_id, start_utc, end_utc, avg_confidence, peak_confidence, and a single flattened `label` string (first non-null of humpback/orca/ship/background, or null)
- [ ] `vocalization_labels` section: queries VocalizationLabel DB table + inference predictions (same logic as `/vocalization-labels/{id}/all` endpoint), outputs start_utc, end_utc, type, confidence, source
- [ ] `vocalization_types` section: queries VocalizationType table, outputs id and name
- [ ] `version: 1` at top level

**Tests needed:**
- Unit test manifest assembly with known fixture data, verify JSON structure matches TypeScript interfaces in consumer contract
- Test label flattening logic (humpback=1 -> "humpback", all null -> null, mutual exclusivity)
- Test confidence score bucketing matches expected array shape

---

### Task 4: API endpoint

**Files:**
- Modify: `src/humpback/api/routers/timeline.py` (add export endpoint)
- Modify: `src/humpback/schemas/classifier.py` (add request/response models if needed)

**Acceptance criteria:**
- [ ] `POST /classifier/detection-jobs/{job_id}/timeline/export` endpoint
- [ ] Request body: `{ "output_dir": string }`
- [ ] Response (200): `{ "job_id": string, "output_path": string, "tile_count": int, "audio_chunk_count": int, "manifest_size_bytes": int }`
- [ ] Returns 404 if job not found, 409 if job not complete or tiles not prepared
- [ ] Synchronous — blocks until export completes
- [ ] Calls `export_timeline()` service function

**Tests needed:**
- Integration test: create a completed job with cached tiles, call the export endpoint, verify output directory structure
- Integration test: call export for non-existent job, expect 404
- Integration test: call export for job without tiles prepared, expect 409

---

### Task 5: CLI script

**Files:**
- Create: `scripts/export_timeline.py`

**Acceptance criteria:**
- [ ] `--job-id` and `--output-dir` required arguments
- [ ] Loads settings from `.env` via `dotenv`
- [ ] Creates async DB session
- [ ] Calls `export_timeline()` service function
- [ ] Prints progress to stderr (tile copy count, audio chunk encoding progress)
- [ ] Prints JSON summary to stdout on success
- [ ] Non-zero exit code on failure with error message to stderr

**Tests needed:**
- No automated tests for the CLI script itself (thin wrapper); manual verification is sufficient

---

### Task 6: Documentation updates

**Files:**
- Modify: `CLAUDE.md` (add export endpoint to §8.8, mention export capability in §9.1)
- Modify: `README.md` (add export to feature list if applicable)

**Acceptance criteria:**
- [ ] CLAUDE.md §8.8 lists the new export endpoint
- [ ] CLAUDE.md §9.1 mentions timeline export capability
- [ ] Consumer contract spec cross-referenced from export spec

**Tests needed:**
- None

---

### Verification

Run in order after all tasks:
1. `uv run ruff format --check src/humpback/services/timeline_export.py src/humpback/processing/audio_encoding.py src/humpback/api/routers/timeline.py scripts/export_timeline.py`
2. `uv run ruff check src/humpback/services/timeline_export.py src/humpback/processing/audio_encoding.py src/humpback/api/routers/timeline.py scripts/export_timeline.py`
3. `uv run pyright src/humpback/services/timeline_export.py src/humpback/processing/audio_encoding.py src/humpback/api/routers/timeline.py scripts/export_timeline.py`
4. `uv run pytest tests/`
