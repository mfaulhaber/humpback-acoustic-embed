# Fix Comparison Defaults Implementation Plan

**Goal:** Auto-detect production model context_pooling and threshold from training_summary instead of using hardcoded defaults, and remove deprecated autoresearch scripts.
**Spec:** `docs/specs/2026-04-09-fix-comparison-defaults-design.md`

---

### Task 1: Add `_resolve_production_defaults` helper and update `compare_classifiers` signature

**Files:**
- Modify: `src/humpback/services/hyperparameter_service/comparison.py`

**Acceptance criteria:**
- [ ] New helper `_resolve_production_defaults(production_classifier)` extracts `context_pooling` and `threshold` from `training_summary`
- [ ] Checks `promoted_config` first, then `replay_effective_config`, then falls back to `"center"` / `0.5`
- [ ] `compare_classifiers` signature changes both params to `str | None = None` and `float | None = None`
- [ ] When `None`, calls `_resolve_production_defaults` to get effective values
- [ ] Explicit non-None values still override auto-detection
- [ ] Output dict still records the effective values used

**Tests needed:**
- Unit test for `_resolve_production_defaults` with promoted_config, replay_effective_config, missing training_summary, and empty training_summary cases

---

### Task 2: Update hyperparameter worker to use auto-detection

**Files:**
- Modify: `src/humpback/workers/hyperparameter_worker.py`

**Acceptance criteria:**
- [ ] Only pass `production_threshold` when `job.comparison_threshold is not None`
- [ ] Do not pass `production_context_pooling` (let auto-detect handle it)

---

### Task 3: Update integration test for auto-detection

**Files:**
- Modify: `tests/integration/test_autoresearch.py`

**Acceptance criteria:**
- [ ] Existing `compare_classifiers` call in integration test works with auto-detection (no explicit pooling/threshold args)
- [ ] Add assertion that `comparison["production"]["context_pooling"]` and `comparison["production"]["threshold"]` reflect auto-detected values
- [ ] Add a test case where the production model has `training_summary` with `promoted_config` containing `context_pooling` and `threshold`, and verify they are picked up

---

### Task 4: Add unit tests for `_resolve_production_defaults`

**Files:**
- Modify: `tests/unit/test_autoresearch.py`

**Acceptance criteria:**
- [ ] Test: `promoted_config` with both fields returns them
- [ ] Test: only `replay_effective_config` present returns those values
- [ ] Test: `training_summary` is `None` returns fallback defaults
- [ ] Test: `training_summary` present but missing both config keys returns fallback defaults
- [ ] Test: explicit overrides in `compare_classifiers` still take precedence over auto-detected values

---

### Task 5: Remove `scripts/autoresearch/`

**Files:**
- Delete: `scripts/autoresearch/` (entire directory)
- Modify: `CLAUDE.md`
- Modify: `README.md`

**Acceptance criteria:**
- [ ] `scripts/autoresearch/` directory is deleted
- [ ] References to `scripts/autoresearch/` CLI wrappers removed from `CLAUDE.md`
- [ ] References to `scripts/autoresearch/` removed from `README.md`
- [ ] Historical references in `docs/specs/` and `docs/plans/` left as-is

---

### Verification

Run in order after all tasks:
1. `uv run ruff format --check src/humpback/services/hyperparameter_service/comparison.py src/humpback/workers/hyperparameter_worker.py tests/unit/test_autoresearch.py tests/integration/test_autoresearch.py`
2. `uv run ruff check src/humpback/services/hyperparameter_service/comparison.py src/humpback/workers/hyperparameter_worker.py tests/unit/test_autoresearch.py tests/integration/test_autoresearch.py`
3. `uv run pyright src/humpback/services/hyperparameter_service/comparison.py src/humpback/workers/hyperparameter_worker.py`
4. `uv run pytest tests/`
