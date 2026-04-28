# Sequence Models Epoch Timestamps Implementation Plan

**Goal:** Align Sequence Models artifacts, APIs, and HMM timeline playback on epoch `start_timestamp` / `end_timestamp` coordinates.
**Spec:** [docs/specs/2026-04-28-sequence-models-epoch-timestamps-design.md](../specs/2026-04-28-sequence-models-epoch-timestamps-design.md)

---

### Task 1: Update Sequence Models Backend Schemas And API Contract

**Files:**
- Modify: `src/humpback/schemas/sequence_models.py`
- Modify: `src/humpback/api/routers/sequence_models.py`
- Modify: `docs/reference/sequence-models-api.md`
- Modify: `docs/specs/2026-04-27-sequence-models-design.md`
- Modify: `docs/specs/2026-04-28-hmm-state-timeline-viewer-design.md`

**Acceptance criteria:**
- [ ] Sequence Models response schemas expose `start_timestamp` / `end_timestamp` for CEJ spans, HMM overlay points, and HMM exemplars.
- [ ] HMM detail response includes `region_detection_job_id`, `region_start_timestamp`, and `region_end_timestamp`.
- [ ] HMM artifact endpoints read canonical timestamp fields and reject missing canonical fields with clear errors.
- [ ] Sequence Models docs no longer describe `start_time_sec` / `end_time_sec` as persisted or API fields.

**Tests needed:**
- Integration tests for HMM detail and artifact endpoints using a source region job with nonzero epoch timestamps.
- Schema/API tests asserting canonical fields are present and legacy fields are absent.

---

### Task 2: Write Canonical Epoch Fields From Continuous Embedding Jobs

**Files:**
- Modify: `src/humpback/workers/continuous_embedding_worker.py`
- Modify: `src/humpback/processing/region_windowing.py`
- Modify: `tests/workers/test_continuous_embedding_worker.py`
- Modify: `tests/services/test_continuous_embedding_service.py`

**Acceptance criteria:**
- [ ] Continuous embedding parquet schema replaces `start_time_sec` / `end_time_sec` with `start_timestamp` / `end_timestamp`.
- [ ] Continuous embedding manifest spans use `start_timestamp` / `end_timestamp`.
- [ ] Worker internals may use relative geometry, but every persisted row/span adds the source `RegionDetectionJob.start_timestamp`.
- [ ] Existing idempotency behavior remains unchanged.
- [ ] Tests use `RegionDetectionJob.start_timestamp != 0` and assert persisted timestamps are epoch values.

**Tests needed:**
- Worker test for a nonzero epoch source region job verifying parquet and manifest timestamp values.
- Existing continuous embedding service tests to ensure job creation/reuse is unaffected.

---

### Task 3: Preserve Epoch Fields Through HMM And Interpretation Artifacts

**Files:**
- Modify: `src/humpback/workers/hmm_sequence_worker.py`
- Modify: `src/humpback/services/hmm_sequence_service.py`
- Modify: `src/humpback/sequence_models/overlay.py`
- Modify: `src/humpback/sequence_models/exemplars.py`
- Modify: `src/humpback/sequence_models/label_distribution.py`
- Modify: `tests/workers/test_hmm_sequence_worker.py`
- Modify: `tests/services/test_hmm_sequence_service.py`
- Modify: `tests/sequence_models/test_overlay.py`
- Modify: `tests/sequence_models/test_exemplars.py`
- Modify: `tests/sequence_models/test_label_distribution.py`

**Acceptance criteria:**
- [ ] HMM states parquet copies `start_timestamp` / `end_timestamp` from the CEJ parquet.
- [ ] PCA/UMAP overlay parquet emits `start_timestamp` / `end_timestamp`.
- [ ] Exemplar JSON emits `start_timestamp` / `end_timestamp`.
- [ ] Label distribution consumes epoch HMM state timestamps directly and no longer adds the region job start timestamp.
- [ ] Tests guard against double-offset bugs by using nonzero region job start timestamps.

**Tests needed:**
- HMM worker schema and timestamp preservation tests.
- Label distribution regression test proving the state-to-label join works with epoch HMM state rows.
- Overlay and exemplar unit tests updated to canonical timestamp fields.

---

### Task 4: Make Region Audio Slice API Epoch-Based

**Files:**
- Modify: `src/humpback/api/routers/call_parsing.py`
- Modify: `frontend/src/api/client.ts`
- Modify: `frontend/src/components/call-parsing/RegionDetectionTimeline.tsx`
- Modify: `tests/integration/test_call_parsing_api.py`

**Acceptance criteria:**
- [ ] Region audio-slice endpoint accepts `start_timestamp` as the canonical query parameter.
- [ ] Endpoint resolves audio using the epoch start directly and validates the requested range against the region job epoch bounds.
- [ ] Frontend helper `regionAudioSliceUrl` takes `startTimestamp` rather than job-relative `startSec`.
- [ ] Existing Region Detection Timeline passes epoch coordinates directly.
- [ ] Any remaining backend-only `start_sec` compatibility is either removed or limited to a deliberate test-only transition point.

**Tests needed:**
- API test for nonzero epoch `start_timestamp` returning a valid audio response or exercising the resolver with a mock.
- Frontend type/build coverage proving callers use the renamed helper argument.

---

### Task 5: Update Sequence Models Frontend Timestamp Usage

**Files:**
- Modify: `frontend/src/api/sequenceModels.ts`
- Modify: `frontend/src/components/sequence-models/ContinuousEmbeddingDetailPage.tsx`
- Modify: `frontend/src/components/sequence-models/HMMSequenceDetailPage.tsx`
- Modify: `frontend/src/components/sequence-models/HMMStateBar.tsx`
- Modify: `frontend/src/components/sequence-models/SpanNavBar.tsx`
- Modify: `frontend/e2e/sequence-models/hmm-sequence.spec.ts`

**Acceptance criteria:**
- [ ] TypeScript interfaces use `start_timestamp` / `end_timestamp` for Sequence Models span/window records.
- [ ] Continuous Embedding detail page labels and displays epoch timestamp fields accurately.
- [ ] HMM detail page derives spans from epoch state rows and passes epoch `jobStart` / `jobEnd` into `TimelineProvider`.
- [ ] HMM State Timeline Viewer requests spectrogram tiles for the correct tile indices on nonzero epoch jobs.
- [ ] HMM audio playback uses epoch `start_timestamp` via the canonical region audio helper.
- [ ] `SpanNavBar` and `HMMStateBar` prop names no longer use ambiguous `Sec` / `start_time_sec` naming.

**Tests needed:**
- Playwright test with mocked nonzero epoch state rows.
- Assertion that the audio request uses `start_timestamp` and that tile requests align with the epoch offset into the region job.

---

### Task 6: Add One-Time Artifact Migration Utility

**Files:**
- Create: `scripts/migrate_sequence_model_timestamps.py`
- Create: `tests/scripts/test_migrate_sequence_model_timestamps.py`

**Acceptance criteria:**
- [ ] Script is dry-run by default and rewrites only with `--apply`.
- [ ] Script rewrites CEJ parquet, CEJ manifest, HMM states parquet, HMM overlay parquet, and HMM exemplar JSON from legacy fields to canonical fields.
- [ ] Script adds `RegionDetectionJob.start_timestamp` when legacy values are job-relative.
- [ ] Script leaves already-canonical artifacts unchanged.
- [ ] Script aborts on ambiguous timestamp values instead of guessing.
- [ ] Script writes parquet and JSON artifacts atomically.

**Tests needed:**
- Unit tests using temp SQLite DB and temp storage fixtures for relative legacy artifacts, already-canonical artifacts, and ambiguous artifacts.
- Test that dry-run does not modify files and `--apply` rewrites expected fields.

---

### Task 7: Remove Remaining Sequence Models Legacy Timestamp Names

**Files:**
- Modify: `frontend/src/api/sequenceModels.ts`
- Modify: `frontend/src/components/sequence-models/HMMSequenceDetailPage.tsx`
- Modify: `frontend/src/components/sequence-models/HMMStateBar.tsx`
- Modify: `src/humpback/schemas/sequence_models.py`
- Modify: `src/humpback/services/hmm_sequence_service.py`
- Modify: `src/humpback/workers/continuous_embedding_worker.py`
- Modify: `src/humpback/workers/hmm_sequence_worker.py`
- Modify: `docs/reference/sequence-models-api.md`

**Acceptance criteria:**
- [ ] `rg "start_time_sec|end_time_sec" src/humpback frontend/src/api frontend/src/components/sequence-models docs/reference/sequence-models-api.md docs/specs/2026-04-27-sequence-models-design.md docs/specs/2026-04-28-hmm-state-timeline-viewer-design.md` returns no Sequence Models contract usages.
- [ ] Any remaining matches are either tests for migration from legacy artifacts or internal non-contract helpers with explicit comments.
- [ ] Sequence Models APIs fail clearly when artifacts still contain only legacy fields after migration.

**Tests needed:**
- Search-based verification plus normal backend/frontend test suites.

---

### Verification

Run in order after all tasks:
1. `uv run ruff format --check src/humpback/schemas/sequence_models.py src/humpback/api/routers/sequence_models.py src/humpback/api/routers/call_parsing.py src/humpback/workers/continuous_embedding_worker.py src/humpback/workers/hmm_sequence_worker.py src/humpback/services/hmm_sequence_service.py src/humpback/sequence_models/overlay.py src/humpback/sequence_models/exemplars.py src/humpback/sequence_models/label_distribution.py scripts/migrate_sequence_model_timestamps.py tests/scripts/test_migrate_sequence_model_timestamps.py`
2. `uv run ruff check src/humpback/schemas/sequence_models.py src/humpback/api/routers/sequence_models.py src/humpback/api/routers/call_parsing.py src/humpback/workers/continuous_embedding_worker.py src/humpback/workers/hmm_sequence_worker.py src/humpback/services/hmm_sequence_service.py src/humpback/sequence_models/overlay.py src/humpback/sequence_models/exemplars.py src/humpback/sequence_models/label_distribution.py scripts/migrate_sequence_model_timestamps.py tests/scripts/test_migrate_sequence_model_timestamps.py`
3. `uv run pyright src/humpback/schemas/sequence_models.py src/humpback/api/routers/sequence_models.py src/humpback/api/routers/call_parsing.py src/humpback/workers/continuous_embedding_worker.py src/humpback/workers/hmm_sequence_worker.py src/humpback/services/hmm_sequence_service.py src/humpback/sequence_models/overlay.py src/humpback/sequence_models/exemplars.py src/humpback/sequence_models/label_distribution.py scripts/migrate_sequence_model_timestamps.py`
4. `uv run pytest tests/`
5. `cd frontend && npx tsc --noEmit`
6. `cd frontend && npx playwright test frontend/e2e/sequence-models/hmm-sequence.spec.ts`
7. `uv run python scripts/migrate_sequence_model_timestamps.py`
8. `rg "start_time_sec|end_time_sec" src/humpback frontend/src/api frontend/src/components/sequence-models docs/reference/sequence-models-api.md docs/specs/2026-04-27-sequence-models-design.md docs/specs/2026-04-28-hmm-state-timeline-viewer-design.md`
