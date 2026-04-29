# HMM Event-Span Input Refactor — Implementation Plan

**Goal:** Refactor the continuous embedding job to consume segmentation events instead of detection regions, producing event-scoped HMM training data with dual region/event navigation in the timeline viewer.
**Spec:** [docs/specs/2026-04-28-hmm-event-span-input-design.md](../specs/2026-04-28-hmm-event-span-input-design.md)

---

### Task 1: Alembic Migration

**Files:**
- Create: `alembic/versions/059_continuous_embedding_event_input.py`
- Modify: `src/humpback/models/sequence_models.py`

**Acceptance criteria:**
- [ ] Back up the production database before applying the migration:
  1. Read the database path from `HUMPBACK_DATABASE_URL` in `.env`
  2. Copy to `<path>.2026-04-28-HH:MM.bak` (UTC timestamp)
  3. Verify backup exists and has non-zero size
- [ ] Migration adds `event_segmentation_job_id` (String, FK to `event_segmentation_jobs.id`) to `continuous_embedding_jobs`
- [ ] Migration removes `region_detection_job_id` column
- [ ] Migration renames `total_regions` to `total_events`
- [ ] Migration changes `pad_seconds` server default to `2.0`
- [ ] Migration uses `op.batch_alter_table()` for SQLite compatibility
- [ ] SQLAlchemy model `ContinuousEmbeddingJob` updated: `region_detection_job_id` → `event_segmentation_job_id`, `total_regions` → `total_events`
- [ ] `uv run alembic upgrade head` succeeds against the production database

**Tests needed:**
- Verify migration applies and rolls back cleanly on a fresh database

---

### Task 2: Pydantic Schemas & Encoding Signature

**Files:**
- Modify: `src/humpback/schemas/sequence_models.py`
- Modify: `src/humpback/services/continuous_embedding_service.py`

**Acceptance criteria:**
- [ ] `ContinuousEmbeddingJobCreate` accepts `event_segmentation_job_id` instead of `region_detection_job_id`, `pad_seconds` defaults to `2.0`
- [ ] `ContinuousEmbeddingJobOut` returns `event_segmentation_job_id` and `total_events` instead of `region_detection_job_id` and `total_regions`
- [ ] `compute_continuous_embedding_signature` hashes `event_segmentation_job_id` instead of `region_detection_job_id`
- [ ] `create_continuous_embedding_job` validates segmentation job exists and is complete
- [ ] `ContinuousEmbeddingJobManifest` uses `total_events` instead of `total_regions`

**Tests needed:**
- Unit test for encoding signature with event_segmentation_job_id
- Unit test for schema validation (create/out models)

---

### Task 3: Continuous Embedding Worker — Event-Based Span Construction

**Files:**
- Modify: `src/humpback/workers/continuous_embedding_worker.py`

**Acceptance criteria:**
- [ ] Worker loads `events.parquet` from the segmentation job instead of `regions.parquet` from the detection job
- [ ] Worker resolves `RegionDetectionJob` through `event_segmentation_job.region_detection_job_id` for hydrophone metadata
- [ ] Each event becomes an independent span: `[event.start_sec - pad_seconds, event.end_sec + pad_seconds]`, clamped to audio envelope
- [ ] Events sorted by `start_sec`, assigned sequential `merged_span_id` (0, 1, 2, ...)
- [ ] No merging of overlapping padded events
- [ ] Parquet schema gains `event_id` column (string)
- [ ] `CONTINUOUS_EMBEDDING_SCHEMA` updated with `event_id` field
- [ ] Each row includes the `event_id` from the source event
- [ ] Manifest JSON uses `total_events` and includes `event_id` per span entry
- [ ] Job completion updates `total_events` instead of `total_regions`
- [ ] `_regions_to_window_geometry` and `merge_padded_regions` usage removed, replaced by event-based span construction

**Tests needed:**
- Worker test with stub embedder: verify event-based span construction produces correct parquet with event_id column
- Verify sequential merged_span_id assignment (one per event)
- Verify pad clamping at audio boundaries
- Verify span timestamps are correct (event.start_sec - pad, event.end_sec + pad)

---

### Task 4: Continuous Embedding API & Queue

**Files:**
- Modify: `src/humpback/api/routers/sequence_models.py`
- Modify: `src/humpback/workers/queue.py` (if claim function references region_detection_job_id)

**Acceptance criteria:**
- [ ] `POST /sequence-models/continuous-embeddings` accepts `event_segmentation_job_id`
- [ ] `GET /sequence-models/continuous-embeddings` returns `event_segmentation_job_id` and `total_events`
- [ ] `GET /sequence-models/continuous-embeddings/{job_id}` returns `event_segmentation_job_id` and `total_events`
- [ ] HMM detail endpoint resolves detection job through `cej.event_segmentation_job_id → seg_job.region_detection_job_id`

**Tests needed:**
- Integration test: create continuous embedding job with event_segmentation_job_id, verify 201 response
- Integration test: idempotent resubmit returns 200
- Integration test: HMM detail returns correct region info through the new FK chain

---

### Task 5: Frontend — API Client & Type Updates

**Files:**
- Modify: `frontend/src/api/sequenceModels.ts`

**Acceptance criteria:**
- [ ] `ContinuousEmbeddingJob` type uses `event_segmentation_job_id` and `totalEvents` instead of `region_detection_job_id` and `totalRegions`
- [ ] `createContinuousEmbeddingJob` sends `event_segmentation_job_id`
- [ ] `ContinuousEmbeddingJobManifest` span entries include `eventId`
- [ ] All references to `regionDetectionJobId` and `totalRegions` in sequence model types updated

**Tests needed:**
- TypeScript type-check passes (`npx tsc --noEmit`)

---

### Task 6: Frontend — Continuous Embedding Create Form

**Files:**
- Modify: `frontend/src/components/sequence-models/ContinuousEmbeddingCreateForm.tsx`

**Acceptance criteria:**
- [ ] Form selects a completed segmentation job instead of a detection job
- [ ] Dropdown lists completed `EventSegmentationJob` rows
- [ ] `padSeconds` default changed to `2.0`
- [ ] Submit sends `event_segmentation_job_id` in payload

**Tests needed:**
- TypeScript type-check passes

---

### Task 7: Frontend — HMM Timeline Viewer Dual Navigation

**Files:**
- Modify: `frontend/src/components/sequence-models/HMMSequenceDetailPage.tsx`
- Modify: `frontend/src/components/sequence-models/SpanNavBar.tsx`

**Acceptance criteria:**
- [ ] Two sets of prev/next controls: region-level and event-level
- [ ] Region nav switches detection regions, scrolls to region time range
- [ ] Event nav steps through events (padded spans), keyboard shortcuts A/D
- [ ] Events sorted by `start_sec` as a flat list across all regions
- [ ] Crossing region boundaries auto-switches the active region
- [ ] "Bring into view without centering" scroll behavior matching `SegmentReviewWorkspace`: 15% viewport margin, direction-aware targeting, sequence-numbered scroll requests
- [ ] State bar and spectrogram show the padded event span with Viterbi overlay

**Tests needed:**
- TypeScript type-check passes
- Playwright smoke test: navigate between events with A/D keys, verify spectrogram updates

---

### Verification

Run in order after all tasks:
1. `uv run ruff format --check src/humpback/models/sequence_models.py src/humpback/schemas/sequence_models.py src/humpback/services/continuous_embedding_service.py src/humpback/workers/continuous_embedding_worker.py src/humpback/api/routers/sequence_models.py`
2. `uv run ruff check src/humpback/models/sequence_models.py src/humpback/schemas/sequence_models.py src/humpback/services/continuous_embedding_service.py src/humpback/workers/continuous_embedding_worker.py src/humpback/api/routers/sequence_models.py`
3. `uv run pyright src/humpback/models/sequence_models.py src/humpback/schemas/sequence_models.py src/humpback/services/continuous_embedding_service.py src/humpback/workers/continuous_embedding_worker.py src/humpback/api/routers/sequence_models.py`
4. `uv run pytest tests/`
5. `cd frontend && npx tsc --noEmit`
