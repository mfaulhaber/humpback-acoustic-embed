# Effective Events Implementation Plan

**Goal:** Make reviewed event identity canonical across Call Parsing review surfaces and Sequence Models while preventing duplicate or overlapping effective events.
**Spec:** `docs/specs/2026-05-03-effective-events-design.md`

---

### Task 1: Add Correction Identity Schema

**Files:**
- Create: `alembic/versions/065_effective_event_identity.py`
- Modify: `src/humpback/models/call_parsing.py`
- Modify: `src/humpback/schemas/call_parsing.py`
- Create: `tests/unit/test_migration_065_effective_event_identity.py`

**Acceptance criteria:**
- [ ] Before running Alembic against the local database, back up the SQLite file from `.env` with `DB=/Volumes/External_2TB/data/whale/humpback-acoustic-embed/data/humpback.db`, `TS=$(date -u +%Y%m%dT%H%M%SZ)`, `cp "$DB" "${DB}.${TS}.bak"`, and `test -s "${DB}.${TS}.bak"`.
- [ ] `event_boundary_corrections` has nullable `event_segmentation_job_id` and `source_event_id` columns.
- [ ] New indexes exist for region detection job, segmentation job, and source event lookup.
- [ ] Existing rows are not backfilled or semantically repaired by the migration.
- [ ] Request and response schemas expose optional correction `id`, `event_segmentation_job_id`, and `source_event_id` fields needed by modern clients.

**Tests needed:**
- Migration test verifies columns and indexes are added without requiring historical data backfill.
- Schema tests cover validation for add, adjust, delete, optional IDs, and invalid corrected ranges.

---

### Task 2: Implement Canonical Effective Event Loading

**Files:**
- Modify: `src/humpback/call_parsing/segmentation/extraction.py`
- Modify: `src/humpback/services/call_parsing.py`
- Modify: `tests/unit/test_feedback_training_schemas_service.py`
- Modify: `tests/integration/test_call_parsing_router.py`

**Acceptance criteria:**
- [ ] Add `load_effective_events()` keyed by `event_segmentation_job_id`.
- [ ] Raw events loaded from `events.parquet` remain unchanged.
- [ ] Adjust corrections preserve the source event ID while replacing bounds.
- [ ] Delete corrections remove only their source event.
- [ ] Add corrections synthesize stable event IDs from the correction row ID.
- [ ] Corrections scoped to another segmentation job are ignored.
- [ ] Legacy region-scoped correction rows are ignored by segmentation-scoped reads unless a compatibility path explicitly opts in.
- [ ] Keep `load_corrected_events()` as a compatibility wrapper during the transition.

**Tests needed:**
- Unit tests for raw, adjusted, deleted, added, and cross-segmentation correction behavior.
- Regression test for an adjusted event preserving its original event ID.

---

### Task 3: Harden Boundary Correction Writes

**Files:**
- Modify: `src/humpback/services/call_parsing.py`
- Modify: `src/humpback/api/routers/call_parsing.py`
- Modify: `src/humpback/schemas/call_parsing.py`
- Modify: `tests/unit/test_feedback_training_schemas_service.py`
- Modify: `tests/integration/test_call_parsing_router.py`

**Acceptance criteria:**
- [ ] `adjust` and `delete` upsert by `(event_segmentation_job_id, source_event_id)` for modern clients.
- [ ] Older adjust/delete clients can still fall back to original bounds within the selected segmentation job.
- [ ] Saved add edits update by correction row `id` instead of creating a new add row.
- [ ] New add requests without `id` create one stable correction row.
- [ ] Add and adjust writes are rejected when the final effective event set would overlap another event in the same segmentation job and region.
- [ ] Overlap checks treat event intervals as half-open ranges and tolerate float noise.
- [ ] Batch validation rejects two proposed corrections that would overlap each other.
- [ ] Conflict responses include the conflicting event ID, region ID, and bounds.

**Tests needed:**
- Service tests for modern upsert identity, legacy fallback, saved add updates, and new adds.
- Service and API tests for overlap conflicts from add, adjust, and same-batch corrections.

---

### Task 4: Update Call Parsing Event APIs

**Files:**
- Modify: `src/humpback/api/routers/call_parsing.py`
- Modify: `src/humpback/schemas/call_parsing.py`
- Modify: `tests/integration/test_call_parsing_router.py`

**Acceptance criteria:**
- [ ] Event-boundary correction POST accepts and validates `event_segmentation_job_id`.
- [ ] Event-boundary correction GET prefers `event_segmentation_job_id` and keeps region job filtering only where compatibility is needed.
- [ ] Segmentation events endpoint can return raw events by default and effective events when requested.
- [ ] Classification typed-events endpoint uses the canonical effective-event path for reviewed/added event identity.
- [ ] Classification typed-events endpoint resolves `region_id` from raw segmentation events first, then effective events, so adjusted-after-classification rows still render.
- [ ] Typed events never return empty `region_id` when their raw source event ID exists upstream.

**Tests needed:**
- API tests for raw versus effective segmentation events.
- API regression for classify job behavior where a typed event was adjusted after classification.
- API test for added/effective event IDs resolving through the effective loader.

---

### Task 5: Update Call Parsing Review UI

**Files:**
- Modify: `frontend/src/api/types.ts`
- Modify: `frontend/src/api/callParsing.ts`
- Modify: `frontend/src/components/call-parsing/SegmentReviewWorkspace.tsx`
- Modify: `frontend/src/components/call-parsing/ClassifyReviewWorkspace.tsx`
- Modify: `frontend/e2e/call-parsing-segment.spec.ts`
- Modify: `frontend/e2e/call-parsing-classify-review.spec.ts`

**Acceptance criteria:**
- [ ] Segment Review fetches and saves boundary corrections with the selected `event_segmentation_job_id`.
- [ ] Segment Review sends `source_event_id` for adjust/delete saves.
- [ ] Segment Review retains saved add correction IDs and sends `id` for later drags or deletes.
- [ ] Segment Review locally detects obvious overlaps before save where practical.
- [ ] Backend conflict responses are surfaced without losing the user's current event context.
- [ ] Classify Review uses the classification job's `event_segmentation_job_id` for boundary correction operations.
- [ ] Classify Review continues resolving vocalization labels against effective event bounds.
- [ ] Classify Review renders spectrogram content for adjusted-after-classification events whose `region_id` is resolved from raw source events.

**Tests needed:**
- E2E regression for a saved add being dragged and reloaded without duplication.
- E2E or component coverage for overlap conflict display.
- E2E regression for the observed Classify Review Event 57 rendering path.

---

### Task 6: Update Downstream Consumers

**Files:**
- Modify: `src/humpback/workers/event_classification_worker.py`
- Modify: `src/humpback/workers/event_classifier_feedback_worker.py`
- Modify: `src/humpback/workers/continuous_embedding_worker.py`
- Modify: `src/humpback/workers/motif_extraction_worker.py`
- Modify: `src/humpback/services/continuous_embedding_service.py`
- Modify: `src/humpback/schemas/sequence_models.py`
- Modify: `src/humpback/models/sequence_models.py`
- Modify: `tests/workers/test_continuous_embedding_worker.py`
- Modify: `tests/workers/test_motif_extraction_worker.py`
- Modify: `tests/services/test_continuous_embedding_service.py`
- Modify: `tests/sequence_models/test_loaders.py`

**Acceptance criteria:**
- [ ] Pass 3 classification reads effective events through `load_effective_events()`.
- [ ] Event classifier feedback uses the same effective event loader for crops and label resolution.
- [ ] Continuous embedding jobs that depend on Pass 2 events expose `event_source_mode` with `raw` as the default.
- [ ] `event_source_mode = "raw"` preserves current `events.parquet` behavior.
- [ ] `event_source_mode = "effective"` reads canonical effective events.
- [ ] Event-aware continuous embedding encoding signatures include event source mode and an effective-correction revision fingerprint.
- [ ] Motif extraction follows the parent continuous embedding job's event source mode.

**Tests needed:**
- Worker tests for Pass 3 effective event consumption.
- Service and worker tests for raw versus effective continuous embedding inputs and idempotency.
- Motif extraction tests for following the continuous embedding job's event source semantics.

---

### Task 7: Update Documentation

**Files:**
- Modify: `DECISIONS.md`
- Modify: `docs/reference/call-parsing-api.md`
- Modify: `docs/reference/sequence-models-api.md`
- Modify: `docs/reference/behavioral-constraints.md`

**Acceptance criteria:**
- [ ] Add an ADR entry superseding ADR-054 for event boundary correction ownership.
- [ ] Call Parsing API docs define segmentation-scoped correction fields and effective events query behavior.
- [ ] Sequence Models API docs define `event_source_mode` and correction fingerprint behavior.
- [ ] Behavioral constraints distinguish raw event artifacts from reviewed effective event sets.

**Tests needed:**
- Documentation review for consistency with implemented request/response fields and job metadata.

---

### Verification

Run in order after all tasks:

1. `uv run ruff format --check src/humpback/call_parsing/segmentation/extraction.py src/humpback/services/call_parsing.py src/humpback/api/routers/call_parsing.py src/humpback/schemas/call_parsing.py src/humpback/workers/event_classification_worker.py src/humpback/workers/event_classifier_feedback_worker.py src/humpback/workers/continuous_embedding_worker.py src/humpback/workers/motif_extraction_worker.py src/humpback/services/continuous_embedding_service.py src/humpback/schemas/sequence_models.py src/humpback/models/call_parsing.py src/humpback/models/sequence_models.py`
2. `uv run ruff check src/humpback/call_parsing/segmentation/extraction.py src/humpback/services/call_parsing.py src/humpback/api/routers/call_parsing.py src/humpback/schemas/call_parsing.py src/humpback/workers/event_classification_worker.py src/humpback/workers/event_classifier_feedback_worker.py src/humpback/workers/continuous_embedding_worker.py src/humpback/workers/motif_extraction_worker.py src/humpback/services/continuous_embedding_service.py src/humpback/schemas/sequence_models.py src/humpback/models/call_parsing.py src/humpback/models/sequence_models.py`
3. `uv run pyright src/humpback/call_parsing/segmentation/extraction.py src/humpback/services/call_parsing.py src/humpback/api/routers/call_parsing.py src/humpback/schemas/call_parsing.py src/humpback/workers/event_classification_worker.py src/humpback/workers/event_classifier_feedback_worker.py src/humpback/workers/continuous_embedding_worker.py src/humpback/workers/motif_extraction_worker.py src/humpback/services/continuous_embedding_service.py src/humpback/schemas/sequence_models.py src/humpback/models/call_parsing.py src/humpback/models/sequence_models.py`
4. `uv run pytest tests/`
5. `cd frontend && npx tsc --noEmit`
