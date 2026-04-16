# MPS Inference for Pass 2 & 3 — Implementation Plan

**Goal:** Run Pass 2 (segmentation) and Pass 3 (event classification) inference on MPS/CUDA when available, with load-time validation and per-job CPU fallback, surfaced to the UI.
**Spec:** [docs/specs/2026-04-16-mps-pass-2-3-inference-design.md](../specs/2026-04-16-mps-pass-2-3-inference-design.md)

---

### Task 1: Alembic migration for compute-device columns

**Files:**
- Create: `alembic/versions/048_job_compute_device.py`

**Acceptance criteria:**
- [ ] Migration adds `compute_device TEXT NULL` and `gpu_fallback_reason TEXT NULL` to `event_segmentation_jobs`
- [ ] Same two columns added to `event_classification_jobs`
- [ ] Uses `op.batch_alter_table()` for SQLite compatibility
- [ ] `down_revision` points at `047_drop_event_segmentation_training_jobs`
- [ ] `uv run alembic upgrade head` applied against the DB from `HUMPBACK_DATABASE_URL` (.env override honored)
- [ ] Downgrade path drops both columns on both tables cleanly

**Tests needed:**
- Migration smoke test: upgrade → downgrade → upgrade on a scratch SQLite DB in the test suite.

---

### Task 2: SQLAlchemy model columns

**Files:**
- Modify: `src/humpback/database.py` (or whichever module defines `EventSegmentationJob` / `EventClassificationJob`)

**Acceptance criteria:**
- [ ] `EventSegmentationJob` gains `compute_device: Mapped[str | None]` and `gpu_fallback_reason: Mapped[str | None]`
- [ ] `EventClassificationJob` gains the same two columns
- [ ] Column definitions match the Alembic migration exactly
- [ ] Pyright clean across `src/humpback`, `scripts/`, `tests/`

**Tests needed:**
- None directly (covered by migration + worker tests in later tasks).

---

### Task 3: Shared `select_and_validate_device` helper

**Files:**
- Modify: `src/humpback/ml/device.py`
- Create: `tests/ml/test_device.py` (if not already present; otherwise extend)

**Acceptance criteria:**
- [ ] New function `select_and_validate_device(model, sample_input, *, rtol=1e-4, atol=1e-5) -> tuple[torch.device, str | None]`
- [ ] Short-circuits to `(cpu, None)` when `select_device()` returns CPU (no validation forward calls)
- [ ] On non-CPU target: runs one forward on CPU, moves model to target device, runs one forward, compares with `torch.allclose`
- [ ] Returns `(cpu, "<backend>_load_error")` on exception during target-device path; model moved back to CPU
- [ ] Returns `(cpu, "<backend>_output_mismatch")` on tolerance failure; model moved back to CPU
- [ ] Returns `(target_device, None)` on success; model left on target device
- [ ] `HUMPBACK_FORCE_CPU=1` honored (already by `select_device()`)
- [ ] Module-level constants for `rtol`, `atol` defaults; docstring explains tuning
- [ ] Logs WARNING on any fallback with the reason
- [ ] Pyright clean

**Tests needed:**
- `HUMPBACK_FORCE_CPU=1` → `(cpu, None)`, model unchanged
- `select_device` monkeypatched to return `cpu` → `(cpu, None)`, no forward calls on the model (assert via spy)
- `select_device` monkeypatched to return a fake non-CPU device; inject a model whose forward raises on that device → `(cpu, "<backend>_load_error")`
- Same fake device; inject a model whose forward returns divergent tensors → `(cpu, "<backend>_output_mismatch")`
- Happy-path MPS test gated on `torch.backends.mps.is_available()` — real small nn.Module, real validation passes → `(mps, None)`

---

### Task 4: Pass 2 inference device plumbing

**Files:**
- Modify: `src/humpback/call_parsing/segmentation/inference.py`

**Acceptance criteria:**
- [ ] `_infer_single(model, audio, feature_config, device)` accepts `device: torch.device`; feature tensor moved with `.to(device)` before `model(...)`
- [ ] `_infer_windowed` forwards `device` through to `_infer_single`
- [ ] `run_inference` accepts `device: torch.device` and forwards it
- [ ] Existing call sites that did not pass a device are updated (this module is only called from the worker; no script entry points)
- [ ] Docstrings updated to note the device argument
- [ ] Pyright clean

**Tests needed:**
- Extend existing segmentation inference unit tests to call `run_inference` with an explicit `torch.device("cpu")` and verify outputs are unchanged vs. the prior signature.

---

### Task 5: Pass 2 worker wiring

**Files:**
- Modify: `src/humpback/workers/event_segmentation_worker.py`

**Acceptance criteria:**
- [ ] Worker imports `select_and_validate_device` from `humpback.ml.device`
- [ ] After `_instantiate_model` + `load_checkpoint` + `model.eval()`, worker builds a deterministic sample tensor sized `(1, 1, n_mels, ~500 frames)` using `n_mels` from the resolved feature config
- [ ] Worker calls the helper, unpacks `(device, fallback_reason)`
- [ ] Worker persists `compute_device = str(device.type)` and `gpu_fallback_reason = fallback_reason` on the job row in the same transaction that moves the row to `running`
- [ ] Worker passes `device` into `run_inference` for every region
- [ ] Validation and the main inference loop stay inside `asyncio.to_thread` so the event loop is not blocked
- [ ] On an exception before validation completes, the partial-artifact cleanup still runs and the row flips to `failed` as before
- [ ] Pyright clean

**Tests needed:**
- Run an existing Pass 2 worker integration test under `HUMPBACK_FORCE_CPU=1`; assert `compute_device == "cpu"` and `gpu_fallback_reason is None` on the completed job row.

---

### Task 6: Pass 3 worker wiring

**Files:**
- Modify: `src/humpback/workers/event_classification_worker.py`

**Acceptance criteria:**
- [ ] Worker imports `select_and_validate_device`
- [ ] After `load_event_classifier(model_dir)`, worker builds a deterministic sample tensor sized `(1, 1, n_mels, ~150 frames)` using `n_mels` from the loaded feature config
- [ ] Helper called, `(device, fallback_reason)` unpacked
- [ ] `compute_device` and `gpu_fallback_reason` persisted on the row in the same transaction that moves the row to `running`
- [ ] `classify_events(...)` called with `device=device`
- [ ] Blocking work remains inside `asyncio.to_thread`
- [ ] Pyright clean

**Tests needed:**
- Run an existing Pass 3 worker integration test under `HUMPBACK_FORCE_CPU=1`; assert `compute_device == "cpu"` and `gpu_fallback_reason is None`.

---

### Task 7: Response schemas

**Files:**
- Modify: `src/humpback/schemas/call_parsing.py`

**Acceptance criteria:**
- [ ] The response model for `EventSegmentationJob` gains `compute_device: str | None = None` and `gpu_fallback_reason: str | None = None`
- [ ] The response model for `EventClassificationJob` gains the same two fields
- [ ] Fields match the SQLAlchemy columns from Task 2
- [ ] Pyright clean; FastAPI serialization verified by existing endpoint tests

**Tests needed:**
- Extend an existing endpoint test for each of the two job types to assert the two new fields are present and nullable in the JSON response.

---

### Task 8: Frontend badge component + integration

**Files:**
- Create: `frontend/src/components/ComputeDeviceBadge.tsx`
- Modify: the Pass 2 event-segmentation job detail page component
- Modify: the Pass 3 event-classification job detail page component
- Modify: frontend API client types for the two job types (wherever response DTOs live)

**Acceptance criteria:**
- [ ] Component accepts `device: string | null` and `fallbackReason: string | null` props
- [ ] Renders green "MPS" badge for `mps`
- [ ] Renders green "CUDA" badge for `cuda`
- [ ] Renders neutral "CPU" badge for `cpu` with null reason
- [ ] Renders yellow "CPU (fallback: <reason>)" badge when reason is non-null, with the reason visible
- [ ] Renders nothing (`null`) for null device
- [ ] Rendered adjacent to the existing status badge on both job detail pages
- [ ] Frontend API client types include the two new fields
- [ ] `cd frontend && npx tsc --noEmit` passes

**Tests needed:**
- Playwright test that visits both job detail pages under mocked API responses and asserts the badge renders the three shapes (cpu, mps, cpu-with-fallback).

---

### Task 9: Docs

**Files:**
- Modify: `docs/reference/call-parsing-api.md`
- Modify: `docs/reference/data-model.md`
- Modify: `CLAUDE.md` §9.2 (latest migration) and §9.1 (new capability note — inference device reporting)

**Acceptance criteria:**
- [ ] `call-parsing-api.md` documents the two new response fields on both job types
- [ ] `data-model.md` notes the two new columns on both tables
- [ ] `CLAUDE.md` §9.2 bumps "Latest migration" to `048_job_compute_device.py`
- [ ] `CLAUDE.md` §9.1 mentions MPS inference for Pass 2/3 with CPU fallback and device reporting

---

### Verification

Run in order after all tasks:

1. `uv run ruff format --check` on modified Python files
2. `uv run ruff check` on modified Python files
3. `uv run pyright` (full run — `pyproject.toml` unchanged but covers `src/humpback`, `scripts/`, `tests/`)
4. `uv run alembic upgrade head` against the DB from `HUMPBACK_DATABASE_URL`
5. `uv run pytest tests/`
6. `cd frontend && npx tsc --noEmit`
7. `cd frontend && npx playwright test`
8. Manual smoke on Apple Silicon: run one Pass 2 job and one Pass 3 job, verify the job detail page shows an "MPS" badge and the DB row has `compute_device = "mps"`, `gpu_fallback_reason IS NULL`.
9. Manual smoke with `HUMPBACK_FORCE_CPU=1`: run the same two jobs, verify "CPU" badge and `compute_device = "cpu"`, `gpu_fallback_reason IS NULL`.
