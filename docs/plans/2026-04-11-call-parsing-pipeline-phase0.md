# Call Parsing Pipeline — Phase 0 Implementation Plan

**Goal:** Ship a working-but-empty four-pass call parsing scaffold — new tables, parquet helpers, PyTorch harness, empty worker shells, stub API surface, and a behavior-preserving detector refactor — so that subsequent Pass specs start from a ready-made skeleton.

**Spec:** [docs/specs/2026-04-11-call-parsing-pipeline-phase0-design.md](../specs/2026-04-11-call-parsing-pipeline-phase0-design.md)

---

### Task 1: Bundle PyTorch into TF extras

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock` (regenerated via `uv lock`)

**Acceptance criteria:**
- [ ] `torch` added to the `tf-macos` optional-dependency extra with a wheel selector appropriate for macOS (CPU/MPS)
- [ ] `torch` added to the `tf-linux-cpu` extra with a CPU wheel selector
- [ ] `torch` added to the `tf-linux-gpu` extra with a CUDA wheel selector matching the project's existing CUDA baseline
- [ ] `uv lock` succeeds without conflicts against the existing TF pins
- [ ] `uv sync --group dev --extra tf-macos` on macOS completes and imports `torch` at the REPL
- [ ] `torch` version chosen is the latest stable that supports the project's Python version cap
- [ ] `README.md` dependency-install snippets remain accurate (no mention of a separate torch install)

**Tests needed:**
- A trivial test in `tests/unit/test_pytorch_available.py` that imports torch, creates a zero tensor, runs a 2-element matmul, and asserts the result shape
- No need to parameterize per-platform — CI runs the platform it runs on

---

### Task 2: Database schema and Alembic migration 042

**Files:**
- Create: `alembic/versions/042_call_parsing_tables.py`
- Modify: `src/humpback/database.py`
- Create: `tests/unit/test_migration_042_call_parsing.py`

**Acceptance criteria:**
- [ ] Migration `042_call_parsing_tables.py` creates the following tables: `call_parsing_runs`, `region_detection_jobs`, `event_segmentation_jobs`, `event_classification_jobs`, `segmentation_models`
- [ ] `call_parsing_runs` columns: `id` (string PK), `audio_source_id` (string, indexed), `status` (string), `config_snapshot` (JSON text), `region_detection_job_id` (nullable FK), `event_segmentation_job_id` (nullable FK), `event_classification_job_id` (nullable FK), `error` (nullable text), `created_at`, `updated_at`, `completed_at` (nullable)
- [ ] Each child job table has standard queue fields (`id`, `status`, `error`, `created_at`, `updated_at`, `started_at`, `completed_at`) plus `parent_run_id` (nullable FK to `call_parsing_runs`), an explicit upstream-pass FK where applicable (`region_detection_job_id` on segmentation, `event_segmentation_job_id` on classification), and a `config_json` column
- [ ] `region_detection_jobs` also stores `audio_source_id`, `model_config_id` (FK), `classifier_model_id` (FK), `trace_row_count`, `region_count`
- [ ] `event_segmentation_jobs` also stores `segmentation_model_id` (FK to `segmentation_models`), `event_count`
- [ ] `event_classification_jobs` also stores `vocalization_model_id` (FK to `vocalization_models`), `typed_event_count`
- [ ] `segmentation_models` columns: `id`, `name`, `model_family`, `model_path`, `config_json`, `training_job_id` (nullable), `created_at`
- [ ] Migration extends `vocalization_models` via `op.batch_alter_table()` to add `model_family` (default `'sklearn_perch_embedding'`, NOT NULL after backfill) and `input_mode` (default `'detection_row'`, NOT NULL after backfill). Existing rows backfilled to defaults in the same migration.
- [ ] Migration extends `vocalization_training_jobs` via `op.batch_alter_table()` with the same two columns and same defaults; existing rows backfilled
- [ ] Downgrade cleanly reverses both table creation and column additions
- [ ] `src/humpback/database.py` gains SQLAlchemy model classes matching the new tables, and `VocalizationModel` / `VocalizationTrainingJob` gain the new columns
- [ ] CLAUDE.md §9.2 migration number updated to `042`

**Tests needed:**
- Fresh SQLite: run `alembic upgrade head` from scratch, verify all new tables and columns exist via SQLAlchemy inspection
- Backfill: create a fixture DB with an existing `vocalization_models` row (pre-migration schema), upgrade, assert the row's `model_family = 'sklearn_perch_embedding'` and `input_mode = 'detection_row'`
- Downgrade roundtrip: upgrade → downgrade → upgrade produces identical schema inspection results
- ORM smoke: insert one row per new table via SQLAlchemy, commit, query it back

---

### Task 3: Call-parsing types and parquet storage helpers

**Files:**
- Create: `src/humpback/call_parsing/__init__.py`
- Create: `src/humpback/call_parsing/types.py`
- Create: `src/humpback/call_parsing/storage.py`
- Create: `tests/unit/test_call_parsing_types.py`
- Create: `tests/unit/test_call_parsing_storage.py`

**Acceptance criteria:**
- [ ] `types.py` defines frozen dataclasses `Region`, `Event`, `TypedEvent`, `WindowScore` with the field lists from the spec
- [ ] `types.py` exposes pyarrow schema constants `REGION_SCHEMA`, `EVENT_SCHEMA`, `TYPED_EVENT_SCHEMA`, `TRACE_SCHEMA` matching the dataclass fields
- [ ] `types.py` exposes a `new_uuid()` helper that returns a UUID4 string, used for `region_id` / `event_id` generation
- [ ] `storage.py` exposes symmetric read/write helpers: `write_trace` / `read_trace`, `write_regions` / `read_regions`, `write_events` / `read_events`, `write_typed_events` / `read_typed_events`
- [ ] Write helpers accept either a list of dataclasses or an iterable, cast to the pyarrow schema, and write atomically (temp file + rename) under a target path
- [ ] Read helpers return a list of dataclasses
- [ ] Writing an empty list produces a valid parquet file with zero rows and correct schema
- [ ] Reading a file with a mismatched schema raises `ValueError` with a descriptive message
- [ ] Directory layout constants for `storage_root/call_parsing/<pass>/<job_id>/` paths centralized in `storage.py`

**Tests needed:**
- Roundtrip each of the four artifact types: write a list of N=50 randomized dataclasses, read back, assert field-by-field equality
- Empty-list roundtrip for all four types produces valid zero-row parquet
- Sorted preservation: write `typed_events` in sorted-by-start order, read back, assert order preserved
- Schema mismatch: write a regions file, attempt to read it as events, assert `ValueError`
- UUID uniqueness: 1000 calls to `new_uuid()` all distinct

---

### Task 4: Shared PyTorch harness under `src/humpback/ml/`

**Files:**
- Create: `src/humpback/ml/__init__.py`
- Create: `src/humpback/ml/device.py`
- Create: `src/humpback/ml/training_loop.py`
- Create: `src/humpback/ml/checkpointing.py`
- Create: `tests/unit/test_ml_device.py`
- Create: `tests/unit/test_ml_training_loop.py`
- Create: `tests/unit/test_ml_checkpointing.py`

**Acceptance criteria:**
- [ ] `device.select_device()` returns a `torch.device` — MPS on macOS when available, CUDA when available, else CPU
- [ ] `device.select_device()` honors `HUMPBACK_FORCE_CPU=1` env var to force CPU (for CI determinism)
- [ ] `training_loop.fit()` accepts `model`, `optimizer`, `train_loader`, optional `val_loader`, `epochs`, optional `scheduler`, optional `callbacks` iterable, optional `device`
- [ ] `fit()` handles `.train()` / `.eval()` mode toggling and `torch.no_grad()` wrapping for validation
- [ ] `fit()` returns a `TrainingResult` dataclass with `train_losses: list[float]`, `val_losses: list[float]`, `callback_outputs: dict` collected from callbacks
- [ ] `fit()` respects a `should_stop` flag set by callbacks for early stopping
- [ ] `checkpointing.save_checkpoint(path, model, optimizer, config)` writes `model_state_dict`, `optimizer_state_dict`, and `config` (dict) to a `.pt` file atomically
- [ ] `checkpointing.load_checkpoint(path, model, optimizer=None)` restores state dicts and returns the `config` dict
- [ ] All public functions have type annotations and pass pyright
- [ ] `ml/__init__.py` re-exports the public API so consumers can `from humpback.ml import fit, save_checkpoint, select_device`

**Tests needed:**
- Device selection: patch torch backend availability flags, verify MPS / CUDA / CPU resolution; verify `HUMPBACK_FORCE_CPU=1` overrides
- `fit()` trains a 2-layer MLP (4 → 8 → 1) on a 2D XOR dataset for 100 epochs with Adam; assert final train loss < 0.2 — validates the loop actually learns
- `fit()` with `val_loader` records val_loss per epoch and the list length equals `epochs`
- Callback early-stop: a callback that sets `should_stop` after epoch 2 terminates the loop with exactly 2 train_losses recorded
- Checkpoint roundtrip: save a trained MLP, load into a fresh MLP instance, run a forward pass, assert outputs equal the original on the same input
- Checkpoint load without optimizer: returns the config dict and leaves the model in the loaded state

---

### Task 5: Detector refactor — extract `compute_hysteresis_events` helper

**Files:**
- Modify: `src/humpback/classifier/detector.py`
- Modify: `src/humpback/classifier/detector_utils.py` (only if needed for the extraction)
- Create: `tests/unit/test_detector_refactor.py`
- Create: `tests/fixtures/detector_refactor_snapshot.json` (committed fixture capturing pre-refactor output on a fixture audio)

**Acceptance criteria:**
- [ ] New helper `compute_hysteresis_events(audio, sample_rate, perch_model, classifier, config)` added to `detector.py`, returning a tuple of `(list[WindowScore], list[HysteresisEvent])` using existing internal types
- [ ] Helper encapsulates: windowing → Perch embedding → per-window binary classifier scoring → hysteresis event merging (via existing `detector_utils.merge_detection_events`)
- [ ] `run_detection()` refactored to call `compute_hysteresis_events()` and then proceed with snap-merge + window-selection + row writing unchanged
- [ ] Public signature of `run_detection()` unchanged
- [ ] The refactor produces **bit-identical** detection rows for the committed fixture audio versus a pre-refactor snapshot
- [ ] Helper importable as `from humpback.classifier.detector import compute_hysteresis_events`
- [ ] Helper returns empty lists when audio is shorter than `window_size_seconds`

**Tests needed:**
- Snapshot regression: capture the current `run_detection` output on a small fixture audio file into `detector_refactor_snapshot.json` BEFORE modifying `detector.py`, then the test asserts post-refactor output matches the snapshot exactly (row count, confidences, timestamps)
- `compute_hysteresis_events` return shapes on fixture audio: correct list lengths, events have valid `start_sec <= end_sec`
- Short audio: audio with duration less than `window_size_seconds` returns `([], [])`
- All existing detection unit and smoke tests continue to pass unchanged

---

### Task 6: Empty worker shells and claim-priority wiring

**Files:**
- Create: `src/humpback/workers/region_detection_worker.py`
- Create: `src/humpback/workers/event_segmentation_worker.py`
- Create: `src/humpback/workers/event_classification_worker.py`
- Modify: the worker loop / dispatcher file that owns the claim priority order (discover in `src/humpback/workers/`)
- Create: `tests/unit/test_call_parsing_workers.py`

**Acceptance criteria:**
- [ ] Each of the three workers exposes a `claim_next()` function using atomic compare-and-set `UPDATE ... WHERE id=:id AND status='queued'` (same pattern as existing workers, ADR-009)
- [ ] On successful claim, each worker immediately marks the job `status='failed'` with `error='NotImplementedError: <pass name> not yet implemented in Phase 0'` and returns
- [ ] Workers are registered in the main worker loop/dispatcher with priority ordering matching CLAUDE.md §8.7: after `vocalization_inference`, before `manifest_generation`, in the order `region_detection → event_segmentation → event_classification`
- [ ] Each worker exports the same module surface as existing workers (e.g. a `run_one_iteration()` or equivalent) so the main loop treats them uniformly

**Tests needed:**
- Queue one job per worker type in a test DB; run that worker's single iteration; assert the job transitions to `failed` with the expected error message
- Race test: two worker invocations race on the same queued row; exactly one succeeds the claim (the other sees zero rows updated and no-ops)
- Priority ordering: queue one job of each of the three new types plus one existing `vocalization_inference` job; run a single dispatch cycle; assert `vocalization_inference` claims first, then `region_detection`, then `event_segmentation`, then `event_classification`

---

### Task 7: API layer — schemas, service, router

**Files:**
- Create: `src/humpback/schemas/call_parsing.py`
- Create: `src/humpback/services/call_parsing.py`
- Create: `src/humpback/api/routers/call_parsing.py`
- Modify: the FastAPI app file that mounts routers (`src/humpback/api/main.py` or equivalent)
- Create: `tests/api/test_call_parsing_router.py`

**Acceptance criteria:**
- [ ] Pydantic schemas defined: `CallParsingRunCreate`, `CallParsingRunResponse` (with nested `region_detection_job`, `event_segmentation_job`, `event_classification_job` status summaries), `RegionDetectionJobSummary`, `EventSegmentationJobSummary`, `EventClassificationJobSummary`
- [ ] `services/call_parsing.py::create_parent_run(session, request)` creates a `CallParsingRun` row and a queued `RegionDetectionJob` row with `parent_run_id` set; returns the parent run with child populated
- [ ] `services/call_parsing.py::get_parent_run(session, run_id)` loads a parent run with its three child rows (nullable) and returns a response model
- [ ] `services/call_parsing.py::delete_parent_run(session, run_id)` cascades to all three child tables and associated parquet directories if present
- [ ] Router registered at `/call-parsing` in the main FastAPI app
- [ ] `POST /call-parsing/runs` — functional; 201 on success with the response model
- [ ] `GET /call-parsing/runs` — functional; returns list with pagination parameters
- [ ] `GET /call-parsing/runs/{id}` — functional; 200 with nested status, 404 if not found
- [ ] `DELETE /call-parsing/runs/{id}` — functional; 204 on success, cascades children
- [ ] `GET /call-parsing/runs/{id}/sequence` — returns 501 with body `{"detail": "Pass 4 (sequence export) not yet implemented"}`
- [ ] `POST /call-parsing/region-jobs` — returns 501 with body naming Pass 1
- [ ] `GET /call-parsing/region-jobs` and `GET /call-parsing/region-jobs/{id}` — functional (pure DB queries returning stored rows, no pass logic needed)
- [ ] `DELETE /call-parsing/region-jobs/{id}` — functional (pure DB delete + parquet directory removal if present)
- [ ] Same pattern for `segmentation-jobs` and `classification-jobs`
- [ ] Artifact endpoints (`/trace`, `/regions`, `/events`, `/typed-events`) return 501 in Phase 0
- [ ] All 501 responses include a clear `detail` naming which Pass owns the endpoint

**Tests needed:**
- `POST /call-parsing/runs` with a valid body creates a DB row in `call_parsing_runs` AND a DB row in `region_detection_jobs` with `parent_run_id` set to the new run's id
- `GET /call-parsing/runs/{id}` returns nested pass status summaries (Pass 1 present and queued, Pass 2/3 null)
- `DELETE /call-parsing/runs/{id}` removes the parent row and the Pass 1 child row
- `POST /call-parsing/region-jobs` returns HTTP 501 with a body that mentions Pass 1
- `GET /call-parsing/runs/{id}/sequence` returns HTTP 501 with a body that mentions Pass 4
- `GET /call-parsing/region-jobs` returns an empty list on a clean DB
- `GET /call-parsing/region-jobs/<id-that-doesnt-exist>` returns 404
- Existing API tests continue to pass

---

### Task 8: Documentation updates

**Files:**
- Modify: `CLAUDE.md` (sections 8.7, 8.8, 9.1, 9.2)
- Modify: `DECISIONS.md` (append ADR-048)
- Modify: `README.md` (if capability list surfaces here; check first)

**Acceptance criteria:**
- [ ] CLAUDE.md §8.7 updated worker priority order to include `region detection → event segmentation → event classification` between `vocalization inference` and `manifest generation`
- [ ] CLAUDE.md §8.8 gains a new "Call Parsing Pipeline" subsection documenting the `/call-parsing/` API surface (parent runs + three pass job types + Pass 4 sequence endpoint)
- [ ] CLAUDE.md §9.1 gains a one-line entry noting "Four-pass call parsing pipeline — Phase 0 scaffold (architecture, tables, workers, stub endpoints; pass logic deferred to Passes 1–4)"
- [ ] CLAUDE.md §9.2 "Latest migration" updated to `042_call_parsing_tables.py`
- [ ] CLAUDE.md §9.2 "Tables" list gains `call_parsing_runs`, `region_detection_jobs`, `event_segmentation_jobs`, `event_classification_jobs`, `segmentation_models`
- [ ] CLAUDE.md §8.7 "Job status transitions" note reaffirmed for new job types (same `queued → running → complete|failed|canceled` pattern)
- [ ] DECISIONS.md appends ADR-048 with Date `2026-04-11`, Status `Accepted`, describing: context (event-level parsing gap), decision (four-pass chained job types + parent `CallParsingRun` + individual-runnable contract), consequences (new tables, PyTorch added to TF extras, `detector.py` refactor, per-pass specs deferred)
- [ ] `README.md` feature list updated only if the existing entries mention detection/classification capabilities at that level of granularity; otherwise unchanged

**Tests needed:**
- None (pure documentation)
- Manual verification: `grep -n "042" CLAUDE.md` and cross-check with the migration filename

---

### Verification

Run after all tasks complete, in order:

1. `uv run ruff format --check src/humpback/call_parsing src/humpback/ml src/humpback/workers/region_detection_worker.py src/humpback/workers/event_segmentation_worker.py src/humpback/workers/event_classification_worker.py src/humpback/api/routers/call_parsing.py src/humpback/schemas/call_parsing.py src/humpback/services/call_parsing.py src/humpback/classifier/detector.py src/humpback/database.py alembic/versions/042_call_parsing_tables.py tests/`
2. `uv run ruff check src/humpback/call_parsing src/humpback/ml src/humpback/workers src/humpback/api/routers/call_parsing.py src/humpback/schemas/call_parsing.py src/humpback/services/call_parsing.py src/humpback/classifier/detector.py src/humpback/database.py alembic/versions/042_call_parsing_tables.py tests/`
3. `uv run pyright` (full run, since `pyproject.toml` changed)
4. `uv run alembic upgrade head` — applies cleanly on a fresh DB
5. `uv run pytest tests/`
6. Manual: `uv run python -c "import torch; print(torch.__version__)"` to confirm the TF-extra bundle landed
7. Manual: start the API server, `POST /call-parsing/runs` with a valid payload, confirm `201` + nested Pass 1 job
