# Sync Detection Embeddings Implementation Plan

**Goal:** Add a sync operation that keeps the detection embeddings parquet in sync with timeline-edited row stores, plus a dedicated Embeddings page to track sync jobs.
**Spec:** [docs/specs/2026-04-02-sync-detection-embeddings-design.md](../specs/2026-04-02-sync-detection-embeddings-design.md)

---

### Task 1: Alembic Migration — Add mode and result_summary to DetectionEmbeddingJob

**Files:**
- Modify: `src/humpback/models/detection_embedding_job.py`
- Create: `alembic/versions/034_detection_embedding_job_sync_columns.py`

**Acceptance criteria:**
- [ ] `DetectionEmbeddingJob` model has `mode: Mapped[Optional[str]]` (nullable, default None) and `result_summary: Mapped[Optional[str]]` (Text, nullable)
- [ ] Migration uses `op.batch_alter_table()` for SQLite compatibility
- [ ] `uv run alembic upgrade head` succeeds
- [ ] Existing rows retain null for both new columns

**Tests needed:**
- Migration applies cleanly on a fresh database and on a database with existing embedding job rows

---

### Task 2: Diff Logic — Compare Row Store vs Embeddings Store

**Files:**
- Modify: `src/humpback/classifier/detector.py` (add diff function)

**Acceptance criteria:**
- [ ] New function `diff_row_store_vs_embeddings(row_store_path, embeddings_path) -> DiffResult` that returns missing, orphaned, and matched sets
- [ ] Row store entries matched by `(start_utc, end_utc)` against embedding entries (computed from `filename` + `start_sec`/`end_sec` via `_file_base_epoch`)
- [ ] Timestamp comparison uses 0.5-second tolerance
- [ ] Returns a dataclass/NamedTuple with `missing: list[dict]`, `orphaned_indices: list[int]`, `matched_count: int`

**Tests needed:**
- All matched, all missing, all orphaned, mixed cases
- Empty row store, empty embeddings store
- Timestamp tolerance edge cases (0.49s matches, 0.51s doesn't)

---

### Task 3: Audio Resolution Helpers

**Files:**
- Modify: `src/humpback/classifier/detector.py` (add audio resolution functions)

**Acceptance criteria:**
- [ ] New function `resolve_audio_for_window(start_utc, end_utc, audio_folder, target_sr) -> np.ndarray | None` for local detection jobs — finds covering file, decodes, extracts window
- [ ] New function `resolve_audio_for_window_hydrophone(start_utc, end_utc, provider, target_sr, window_size) -> np.ndarray | None` for hydrophone jobs — uses `iter_audio_chunks` with the target time range
- [ ] Both return None (not raise) when audio is unavailable, with a reason string
- [ ] Local resolution reuses `_file_base_epoch()` for timestamp parsing and `decode_audio`/`resample` for audio loading

**Tests needed:**
- Local: correct file selection given multiple files with different timestamps
- Local: correct offset computation within a file
- Local: returns None when no file covers the target UTC range
- Hydrophone: provider called with correct time range arguments

---

### Task 4: Sync Worker Logic

**Files:**
- Modify: `src/humpback/workers/detection_embedding_worker.py`

**Acceptance criteria:**
- [ ] `run_detection_embedding_job` dispatches to new `_run_sync_mode` when `job.mode == "sync"`
- [ ] Sync mode: runs diff, resolves audio for each missing row, generates embeddings, rewrites parquet
- [ ] Loads embedding model via `get_model_by_version` from `classifier_model_id`
- [ ] Handles local jobs (audio_folder) and hydrophone jobs (provider construction via `build_archive_detection_provider`)
- [ ] Skips rows where audio can't be resolved, records reasons
- [ ] Updates `result_summary` JSON on job completion: `{added, removed, unchanged, skipped, skipped_reasons}`
- [ ] Progress tracking: `progress_total` = number of missing rows, `progress_current` incremented per embedding generated

**Tests needed:**
- Integration test with fake model: add rows to row store, run sync, verify embeddings parquet has new entries and orphans removed
- Verify skipped rows recorded correctly when audio unavailable
- Verify result_summary JSON structure

---

### Task 5: API Changes — Sync Mode and Embedding Status

**Files:**
- Modify: `src/humpback/api/routers/classifier.py`
- Modify: `src/humpback/schemas/classifier.py`

**Acceptance criteria:**
- [ ] `generate_embeddings` endpoint accepts `mode: str = Query("full")` parameter
- [ ] `mode=full` preserves existing behavior (409 if embeddings exist)
- [ ] `mode=sync` requires existing embeddings (400 if not), creates job with `mode="sync"`
- [ ] `EmbeddingStatusResponse` gains `sync_needed: bool | None` field
- [ ] `get_embedding_status` computes `sync_needed` by running the diff logic (comparing row store vs embeddings counts or a lightweight check)
- [ ] `DetectionEmbeddingJobOut` schema gains `mode` and `result_summary` fields

**Tests needed:**
- `mode=sync` returns 400 when no embeddings exist
- `mode=sync` creates job with correct mode value
- `mode=full` returns 409 when embeddings exist (existing behavior preserved)
- `sync_needed` is true when row store has unmatched entries
- `sync_needed` is false when stores are in sync

---

### Task 6: API — Embedding Jobs List Endpoint

**Files:**
- Modify: `src/humpback/api/routers/classifier.py`
- Modify: `src/humpback/schemas/classifier.py`

**Acceptance criteria:**
- [ ] `GET /classifier/embedding-jobs` returns paginated list of `DetectionEmbeddingJob` records, newest first
- [ ] Response includes parent detection job context: `hydrophone_name` and `audio_folder` (basename only)
- [ ] Accepts `offset` and `limit` query params
- [ ] Response schema includes all job fields plus detection job context

**Tests needed:**
- Returns jobs ordered by created_at descending
- Pagination works correctly
- Detection job context fields populated

---

### Task 7: Frontend — Sync Embeddings Button on Timeline Viewer

**Files:**
- Modify: `frontend/src/components/timeline/TimelineViewer.tsx` or `TimelineHeader.tsx`
- Modify: `frontend/src/api/client.ts` (update generate-embeddings call to accept mode param)
- Modify: `frontend/src/api/types.ts` (update EmbeddingStatus type with sync_needed)
- Modify: `frontend/src/hooks/queries/useVocalization.ts` (add sync mutation)

**Acceptance criteria:**
- [ ] "Sync Embeddings" button visible in timeline header when `sync_needed` is true
- [ ] Button calls `POST .../generate-embeddings?mode=sync`
- [ ] Polls for completion using existing `useEmbeddingGenerationStatus` pattern
- [ ] Shows sync summary on completion (added/removed counts)
- [ ] Button hidden or disabled when sync is running or not needed

**Tests needed:**
- Playwright: button appears when sync_needed is true, hidden when false
- Playwright: button triggers sync and shows completion summary

---

### Task 8: Frontend — Classifier/Embeddings Page

**Files:**
- Create: `frontend/src/components/classifier/EmbeddingsPage.tsx`
- Modify: `frontend/src/api/client.ts` (add fetchEmbeddingJobs)
- Modify: `frontend/src/api/types.ts` (add EmbeddingJobListItem type)
- Modify: `frontend/src/hooks/queries/useClassifier.ts` (add useEmbeddingJobs hook)
- Modify: `frontend/src/App.tsx` (add route)
- Modify: `frontend/src/components/layout/Sidebar.tsx` or equivalent nav component (add link)

**Acceptance criteria:**
- [ ] Route `/app/classifier/embeddings` renders the page
- [ ] Table displays: Status badge, Detection Job (linked), Mode (Full/Sync), Progress/Summary, Created, Duration
- [ ] Running jobs show progress indicator with polling
- [ ] Completed sync jobs show added/removed/unchanged/skipped summary
- [ ] Failed jobs show error message
- [ ] Navigation link added under Classifier section
- [ ] Table is paginated

**Tests needed:**
- Playwright: page renders with job table
- Playwright: sync job rows show summary breakdown
- TypeScript compiles without errors

---

### Task 9: Documentation Updates

**Files:**
- Modify: `CLAUDE.md`
- Modify: `docs/reference/frontend.md`
- Modify: `docs/reference/data-model.md`

**Acceptance criteria:**
- [ ] CLAUDE.md §9.1 updated to mention embedding sync capability
- [ ] CLAUDE.md §9.2 migration number updated to 034
- [ ] Frontend reference updated with new route and component
- [ ] Data model reference updated with new DetectionEmbeddingJob columns

**Tests needed:**
- None (documentation only)

---

### Verification

Run in order after all tasks:
1. `uv run ruff format --check src/humpback/models/detection_embedding_job.py src/humpback/classifier/detector.py src/humpback/workers/detection_embedding_worker.py src/humpback/api/routers/classifier.py src/humpback/schemas/classifier.py`
2. `uv run ruff check src/humpback/models/detection_embedding_job.py src/humpback/classifier/detector.py src/humpback/workers/detection_embedding_worker.py src/humpback/api/routers/classifier.py src/humpback/schemas/classifier.py`
3. `uv run pyright src/humpback/models/detection_embedding_job.py src/humpback/classifier/detector.py src/humpback/workers/detection_embedding_worker.py src/humpback/api/routers/classifier.py src/humpback/schemas/classifier.py`
4. `uv run pytest tests/`
5. `cd frontend && npx tsc --noEmit`
