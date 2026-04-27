# Sequence Models — Continuous Embedding Producer (PR 1) Implementation Plan

**Goal:** Land the new "Sequence Models" track and its first job type — a region-bounded, hydrophone-only `ContinuousEmbeddingJob` that produces 1-second-hop SurfPerch embeddings padded around Pass-1 region detections, with full backend, frontend, tests, and documentation.

**Spec:** [docs/specs/2026-04-27-sequence-models-design.md](../specs/2026-04-27-sequence-models-design.md) (PR 1 sections: §3 PR 1 scope, §4.1–4.2, §4.4, §5.1, §5.3, §5.5–5.7, §6, §7 PR 1, §8.1–8.2, §8.4 PR 1, §8.5, §10).

---

### Task 1: Alembic migration — `continuous_embedding_jobs` table

**Files:**
- Create: `alembic/versions/057_continuous_embedding_jobs.py`

**Acceptance criteria:**
- [ ] **Pre-migration backup of production DB performed.** Read `HUMPBACK_DATABASE_URL` from `.env`, copy the SQLite file to `<original_path>.YYYY-MM-DD-HH:mm.bak` using a UTC timestamp (e.g., `cp "$DB_PATH" "${DB_PATH}.2026-04-27-18:30.bak"`), and confirm the backup file exists with non-zero size **before** running any migration command. (CLAUDE.md §3.5; this is blocking, not a parenthetical.)
- [ ] Migration uses `op.batch_alter_table()` for SQLite compatibility per project convention.
- [ ] Creates `continuous_embedding_jobs` with all columns from spec §5.1: `id`, `status`, `region_detection_job_id` (FK → `region_detection_jobs.id`, NOT NULL), `model_version`, `window_size_seconds`, `hop_seconds`, `pad_seconds`, `target_sample_rate`, `feature_config_json`, `encoding_signature`, `vector_dim`, `total_regions`, `merged_spans`, `total_windows`, `parquet_path`, `error_message`, `created_at`, `updated_at`.
- [ ] Index on `encoding_signature` to support idempotency lookup.
- [ ] Index on `status` to support active-job filtering.
- [ ] `downgrade()` drops the table cleanly.
- [ ] `uv run alembic upgrade head` runs cleanly against the backed-up production DB.
- [ ] `uv run alembic downgrade -1` followed by `uv run alembic upgrade head` runs cleanly (round-trip check).

**Tests needed:**
- No code-level unit test for migrations (project convention); round-trip migrate-up/down is the verification.

---

### Task 2: ORM model and enum reuse

**Files:**
- Modify: `src/humpback/database.py`

**Acceptance criteria:**
- [ ] Add `ContinuousEmbeddingJob` SQLAlchemy model mapped to `continuous_embedding_jobs`, with all columns matching the migration.
- [ ] Reuse the existing job-status enum used by clustering / call-parsing job types — do not introduce a new status enum.
- [ ] Add `continuous_embedding_jobs` relationship navigation from `RegionDetectionJob` (one-to-many) so we can query producer jobs for a given region detection.
- [ ] `vector_dim`, `total_regions`, `merged_spans`, `total_windows`, `parquet_path`, `error_message` declared nullable (filled at run-time / on completion).
- [ ] `created_at`, `updated_at` use UTC defaults consistent with project convention (CLAUDE.md §3.8).

**Tests needed:**
- Smoke import test exercising the new model can be constructed and persisted (covered transitively in service tests).

---

### Task 3: Pydantic schemas

**Files:**
- Create: `src/humpback/schemas/sequence_models.py`

**Acceptance criteria:**
- [ ] `ContinuousEmbeddingJobCreate` with fields: `region_detection_job_id` (required), `model_version` (default `"surfperch-tensorflow2"`), `hop_seconds` (default `1.0`, validated `> 0`), `pad_seconds` (default `10.0`, validated `>= 0`).
- [ ] `window_size_seconds`, `target_sample_rate`, `feature_config_json` are **derived from the model config at submission time**, not user-supplied — confirmed by validator/service rather than schema fields.
- [ ] `ContinuousEmbeddingJobOut` exposing all DB columns the frontend needs (status, all params, summary stats, parquet_path, error_message, timestamps).
- [ ] `ContinuousEmbeddingJobManifest` schema matching the JSON sidecar produced by the worker (vector_dim, model_version, hop/pad/window settings, span count, total windows, span boundaries summary).
- [ ] All timestamps as UTC epoch seconds or ISO-Z strings per CLAUDE.md §3.8.

**Tests needed:**
- `tests/schemas/test_sequence_models_schemas.py`: validation rejects `hop_seconds <= 0` and `pad_seconds < 0`; defaults populate as expected.

---

### Task 4: Pure region-windowing helpers

**Files:**
- Create: `src/humpback/processing/region_windowing.py`
- Create: `tests/processing/test_region_windowing.py`

**Acceptance criteria:**
- [ ] `merge_padded_regions(regions, pad_seconds, audio_envelope)` — pure function expanding each region by ±pad, merging padded spans whose extents overlap, tracking original `region_id`s per merged span, clipping to `audio_envelope` (start, end) bounds.
- [ ] `iter_windows(span, hop_seconds, window_size_seconds)` — yields `(window_index_in_span, start_time_sec, end_time_sec, is_in_pad, source_region_ids)` records. `is_in_pad=False` iff the window's *center* timestamp falls inside any of the span's original (un-padded) source regions.
- [ ] `source_region_ids` returns the list of original region ids whose un-padded extent contains the window center (may be empty when in pad; may be multi-element if regions touch).
- [ ] No I/O, no DB access, no model calls. Functions are deterministic given inputs.
- [ ] Type annotations sufficient for `pyright` to pass at strict-enough level for the project.

**Tests needed:**
- Non-overlapping regions → as many merged spans as inputs; no merging.
- Two regions whose padded extents overlap → single merged span tracking both region ids.
- Three+ adjacent regions all merged into one span.
- Region at audio start → pad clipped to envelope start; span's pre-pad shorter than `pad_seconds`.
- Region at audio end → analogous clip at end.
- Empty input → empty output.
- `iter_windows` window-center geometry: window centered exactly on a region boundary is treated as in-region (boundary inclusion documented in code).
- `source_region_ids` correctness when a window center sits inside multiple touching regions.
- Hop and window-size geometry: number of windows in a span matches `floor((span_duration - window_size) / hop) + 1` (or analogous, with documented edge handling).

---

### Task 5: Storage helper

**Files:**
- Modify: `src/humpback/storage.py`

**Acceptance criteria:**
- [ ] Add `continuous_embedding_dir(job_id) -> Path` returning `<data_root>/continuous_embeddings/{job_id}/`.
- [ ] Add `continuous_embedding_parquet_path(job_id) -> Path` returning `embeddings.parquet` under that directory.
- [ ] Add `continuous_embedding_manifest_path(job_id) -> Path` returning `manifest.json` under that directory.
- [ ] Helpers create parent directories on demand consistent with how existing storage helpers behave (or document if creation is the worker's responsibility — match the existing convention used by clustering/detection helpers).

**Tests needed:**
- `tests/test_storage.py` (or similar existing module) — smoke tests that the new helpers return paths under the configured data root.

---

### Task 6: Service layer with idempotency

**Files:**
- Create: `src/humpback/services/continuous_embedding_service.py`
- Create: `tests/services/test_continuous_embedding_service.py`

**Acceptance criteria:**
- [ ] `create_continuous_embedding_job(payload)`:
  - Validates `region_detection_job_id` exists.
  - Resolves `window_size_seconds`, `target_sample_rate`, `feature_config_json` from the registered `model_config` for the requested `model_version`. Rejects unsupported `model_version`.
  - Computes `encoding_signature = sha256(...)` per spec §5.1.
  - **Idempotency**: if a row with the same signature exists with `status='complete'`, return it (no new row).
  - **In-flight blocking**: if a row with the same signature exists with `status` in {`queued`, `running`}, return that row (no duplicate queueing).
  - Otherwise insert a new row in `queued` state and return it.
- [ ] `list_continuous_embedding_jobs(filters)` — supports filtering by status; orders newest-first; returns `ContinuousEmbeddingJobOut`.
- [ ] `get_continuous_embedding_job(id)` — single job + manifest if `parquet_path` is set and manifest file exists.
- [ ] `cancel_continuous_embedding_job(id)` — flips `queued` → `canceled` directly; for `running`, sets a `cancellation_requested` flag the worker checks (mirror existing job patterns).
- [ ] No worker invocation here — services only mutate DB rows; the queue picks up `queued` jobs.

**Tests needed:**
- Idempotency: same signature submitted twice with first complete → second returns the existing complete row.
- In-flight blocking: same signature submitted with first `running` → second returns the running row, no new insert.
- Validation: rejects non-existent `region_detection_job_id`.
- Validation: rejects unsupported `model_version`.
- Validation: rejects `hop_seconds <= 0`, `pad_seconds < 0` (covered partly in schema task; service must defensively re-check).
- Cancel from `queued` flips to `canceled`.
- Cancel from `complete` is a no-op (or returns clear error — match existing convention).

---

### Task 7: Worker with end-to-end tests using SurfPerch stub

**Files:**
- Create: `src/humpback/workers/continuous_embedding_worker.py`
- Create: `tests/fixtures/sequence_models/__init__.py`
- Create: `tests/fixtures/sequence_models/surfperch_stub.py`
- Create: `tests/workers/test_continuous_embedding_worker.py`

**Acceptance criteria:**
- [ ] `claim_and_run_continuous_embedding_job` follows the existing claim-then-run pattern (atomic compare-and-set queued → running) used by clustering/detection workers.
- [ ] `run_continuous_embedding_job(job_id)`:
  - Loads job and source `RegionDetectionJob`; aborts with descriptive error if the source is not `complete`.
  - Loads regions sorted by start_time; computes the audio envelope from the region detection's hydrophone time range.
  - Calls `merge_padded_regions` to produce merged spans.
  - For each merged span: decodes audio chunk-by-chunk from the existing hydrophone audio decoder (the same one Pass-1 uses); runs SurfPerch via the existing `model_runners` interface at `hop_seconds`/`window_size_seconds`; iterates `iter_windows` for row metadata; appends rows to a temp parquet writer.
  - Writes `embeddings.parquet` schema exactly per spec §5.3 — `merged_span_id` (int32), `window_index_in_span` (int32), `audio_file_id` (int32 nullable), `start_time_sec` (float64), `end_time_sec` (float64), `is_in_pad` (bool), `source_region_ids` (list<int32>), `embedding` (list<float32>[vector_dim]). Rows sorted by `(merged_span_id, window_index_in_span)`.
  - Writes `manifest.json` with `vector_dim`, `model_version`, hop/pad/window settings, span count, total windows, per-span time-bound summaries.
  - All artifacts written to temp paths under the same job dir, then atomically renamed into place. Job status flipped to `complete` only after all renames succeed.
  - On exception: status `failed`, `error_message` populated; temp files cleaned best-effort; canonical artifact paths must not exist.
  - Cancellation: worker checks `cancellation_requested` between merged spans; on detection cleans temp files and transitions to `canceled`.
- [ ] **No reprocessing inside the worker** — the idempotency check is the service's job. Worker assumes its job row is the canonical one to execute.
- [ ] SurfPerch stub fixture (`surfperch_stub.py`) provides a deterministic stub implementing the SurfPerch runner interface — same audio in → same fixed-shape embedding out, where the shape matches whatever the registered SurfPerch `model_config` reports (vector_dim discovered from config, not hard-coded). Wired via the existing model-stub dependency-injection / monkeypatch path used by other model-stubbed tests.

**Tests needed:**
- End-to-end happy path with stubbed SurfPerch + a tiny synthetic hydrophone audio fixture: produces parquet with expected row count, schema, types, and sort order; manifest.json present with correct fields.
- Failure path: stub raises mid-run → status `failed`, non-empty `error_message`, no canonical parquet at the expected path.
- Atomic write: after success, no temp files remain; after failure, no partial canonical parquet.
- Cancellation: setting `cancellation_requested=true` between regions triggers cleanup + `canceled` status.
- Source-not-complete guard: worker called with a `region_detection_job` in non-complete status fails fast with clear error message and does not write artifacts.

---

### Task 8: API router and FastAPI integration

**Files:**
- Create: `src/humpback/api/sequence_models.py`
- Modify: the FastAPI app entry (the module that registers routers — match existing project convention)

**Acceptance criteria:**
- [ ] `POST /sequence-models/continuous-embeddings` — calls `create_continuous_embedding_job`, returns `ContinuousEmbeddingJobOut`. Returns the existing job (200) on idempotent or in-flight match; returns 201 when creating new.
- [ ] `GET /sequence-models/continuous-embeddings` — list with optional `status` filter query param.
- [ ] `GET /sequence-models/continuous-embeddings/{id}` — detail + manifest summary; 404 on missing.
- [ ] `POST /sequence-models/continuous-embeddings/{id}/cancel` — calls `cancel_continuous_embedding_job`; 404 on missing; 409 on terminal-state cancel attempt.
- [ ] Router mounted under the existing API app with consistent prefix and tags so it shows up in OpenAPI docs.
- [ ] Standard error envelope responses match existing project convention.

**Tests needed:**
- `tests/api/test_sequence_models_api.py`: happy paths for create/list/get/cancel; idempotent re-create returns existing job id; 404 on unknown id; 409 on canceling a complete job.

---

### Task 9: Frontend — Sequence Models nav and Continuous Embedding pages

**Files:**
- Create: `frontend/src/pages/SequenceModels/ContinuousEmbeddingJobsPage.tsx`
- Create: `frontend/src/pages/SequenceModels/ContinuousEmbeddingDetailPage.tsx`
- Create: `frontend/src/components/sequence-models/ContinuousEmbeddingCreateForm.tsx`
- Create: `frontend/src/components/sequence-models/ContinuousEmbeddingJobCard.tsx`
- Create: `frontend/src/api/sequenceModels.ts`
- Modify: top-level nav config (the file that defines top-level routes / tabs — match existing project pattern)
- Modify: top-level router (where Vocalization, Call Parsing, etc. are routed)
- Create: `frontend/tests/sequence-models/continuous-embedding.spec.ts`

**Acceptance criteria:**
- [ ] New top-level "Sequence Models" nav entry visible alongside Vocalization, Call Parsing, etc.
- [ ] Sub-tab "Continuous Embedding" renders the jobs page; future sub-tab slot ("HMM Sequence") wired but disabled / not visible until PR 2.
- [ ] Jobs page splits jobs into Active (queued/running) and Previous (complete/failed/canceled) sections; cards show status, summary stats (region count, span count, window count, vector_dim), and time since created.
- [ ] Create form: dropdown of completed `RegionDetectionJob`s (filterable by hydrophone/time range as already supported elsewhere), inputs for `hop_seconds` (default 1.0) and `pad_seconds` (default 10.0), model_version dropdown defaulting to `surfperch-tensorflow2`. Submit posts to the API.
- [ ] Detail page shows job manifest summary, per-span row counts (from manifest.json), and error message if failed. **No charts** — producer is plumbing only per spec §7 PR 1.
- [ ] TanStack Query hooks: `useContinuousEmbeddingJobs`, `useContinuousEmbeddingJob` with 3000 ms refetch interval on active jobs, no polling on terminal-state jobs.
- [ ] All UTC timestamps rendered with explicit "UTC" labels per CLAUDE.md §3.8.

**Tests needed:**
- Playwright spec exercises: navigating to Sequence Models → Continuous Embedding; filling the create form against a seeded region-detection job; observing the new job appear in Active; mocking the API to flip the job to `complete` and observing the detail page render manifest stats; mocking a `failed` status and observing the error message render.
- `cd frontend && npx tsc --noEmit` passes.

---

### Task 10: Documentation

**Files:**
- Create: `docs/reference/sequence-models-api.md`
- Modify: `docs/reference/storage-layout.md`
- Modify: `CLAUDE.md`
- Modify: `DECISIONS.md`
- Modify: `README.md`

**Acceptance criteria:**
- [ ] `docs/reference/sequence-models-api.md` lists the four PR 1 endpoints with method, path, brief description, and links to the schemas.
- [ ] `docs/reference/storage-layout.md` adds the `continuous_embeddings/{job_id}/` tree under the appropriate section.
- [ ] `CLAUDE.md` §9.1 — append a bullet describing the new Sequence Models / Continuous Embedding capability.
- [ ] `CLAUDE.md` §9.2 — bump latest migration to `057_continuous_embedding_jobs.py` and add `continuous_embedding_jobs` to the table list.
- [ ] `CLAUDE.md` §9.3 (sensitive components) — add `processing/region_windowing.py` if its correctness gates downstream HMM sequencing (likely yes — flag).
- [ ] `CLAUDE.md` §8 add a new §8.X pointer to `docs/reference/sequence-models-api.md` consistent with how other API surfaces are referenced.
- [ ] `DECISIONS.md` — append `ADR-056: Sequence Models track parallel to Call Parsing pipeline` summarizing the architectural placement, the 1:1 source linkage decision, the region-bounded design with padding, and the choice to keep this independent from the four-pass call parsing scaffold.
- [ ] `README.md` — add the four new endpoints to the user-facing endpoint list, and add Sequence Models to the feature list.

**Tests needed:**
- No automated tests — content review during PR review.

---

### Verification

Run after all tasks are complete, in this order:

1. `uv run ruff format --check src/humpback tests` (modified Python files)
2. `uv run ruff check src/humpback tests` (modified Python files)
3. `uv run pyright` (full run since `pyproject.toml` may have changed; otherwise restrict to modified files)
4. `uv run pytest tests/`
5. `cd frontend && npx tsc --noEmit`
6. `cd frontend && npx playwright test`
7. `uv run alembic upgrade head` runs cleanly against the (backed-up) production DB
8. Manual smoke: from the running app, create a continuous-embedding job against a small completed region-detection job; observe job transitions queued → running → complete; inspect the resulting `embeddings.parquet` and `manifest.json` for sane row counts and schema; verify idempotency by submitting the same parameters again and observing the existing job is returned.
