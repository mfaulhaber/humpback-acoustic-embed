# Scripts Folder Cleanout Implementation Plan

**Goal:** Reduce `scripts/` to the retained bootstrap scripts while preserving timeline export service/API and canonical hyperparameter candidate workflows.
**Spec:** [docs/specs/2026-05-07-scripts-folder-cleanout-design.md](../specs/2026-05-07-scripts-folder-cleanout-design.md)
**Primary domain:** core-platform
**Neighbor domains:** ingest-detection, signal-timeline, frontend-shell

---

### Task 1: Delete Non-Bootstrap Scripts And Direct Script Tests

**Files:**
- Delete: `scripts/README.md`
- Delete: `scripts/backfill_detection_metadata.py`
- Delete: `scripts/backfill_training_summary.py`
- Delete: `scripts/benchmark_region_detection.py`
- Delete: `scripts/cleanup_legacy_workflows.py`
- Delete: `scripts/cleanup_sequence_model_artifacts.py`
- Delete: `scripts/cleanup_short_negatives.py`
- Delete: `scripts/convert_audio_to_flac.py`
- Delete: `scripts/deploy.sh`
- Delete: `scripts/export_timeline.py`
- Delete: `scripts/fix_duplicate_labels.py`
- Delete: `scripts/migrate_row_ids.py`
- Delete: `scripts/migrate_sequence_model_timestamps.py`
- Delete: `scripts/noaa_detection_metadata.py`
- Delete: `scripts/recover_event_boundary_corrections.py`
- Delete: `scripts/recover_vocalization_corrections.py`
- Delete: `scripts/reorder_vocalization_layout.py`
- Delete: `scripts/repair_hydrophone_extract_lengths.py`
- Delete: `scripts/sb_diagnostic.py`
- Delete: `scripts/stage_s3_epoch_cache.py`
- Delete: `scripts/validate_gap_filling.py`
- Delete: `tests/unit/test_convert_audio_to_flac.py`
- Delete: `tests/unit/test_migrate_row_ids.py`
- Delete: `tests/unit/test_noaa_detection_metadata.py`
- Delete: `tests/unit/test_repair_hydrophone_extract_lengths.py`
- Delete: `tests/unit/test_stage_s3_epoch_cache.py`
- Delete: `tests/scripts/test_cleanup_legacy_workflows.py`
- Delete: `tests/scripts/test_cleanup_sequence_model_artifacts.py`
- Delete: `tests/scripts/test_migrate_sequence_model_timestamps.py`

**Acceptance criteria:**
- [x] `git ls-files scripts` lists only `scripts/bootstrap_classifier.py`, `scripts/bootstrap_event_classifier_dataset.py`, and `scripts/bootstrap_segmentation_dataset.py`.
- [x] No test imports a removed `scripts.*` module.
- [x] The three retained bootstrap scripts are not modified except for formatting if a tool requires it.
- [x] Bootstrap script tests still cover the retained scripts.

**Tests needed:**
- Run the retained bootstrap script unit tests.
- Run a repo search for removed `scripts.*` imports and non-bootstrap script path references in active tests.

---

### Task 2: Remove Legacy Autoresearch Alias API

**Files:**
- Delete: `src/humpback/api/routers/classifier/autoresearch.py`
- Modify: `src/humpback/api/routers/classifier/__init__.py`
- Modify: `tests/integration/test_hyperparameter_api.py`
- Modify: `tests/integration/test_classifier_api.py`
- Modify: `frontend/e2e/classifier-training.spec.ts`

**Acceptance criteria:**
- [x] The old `/classifier/autoresearch-candidates/*` router is no longer included in the classifier API.
- [x] Tests no longer assert that the old alias endpoints work.
- [x] Candidate import, list, detail, delete, search-result import, and promotion coverage uses canonical `/classifier/hyperparameter/*` paths.
- [x] Candidate-backed replay training and candidate provenance behavior are unchanged.
- [x] Frontend route mocks use the same canonical candidate paths as `frontend/src/api/client.ts`.

**Tests needed:**
- Run hyperparameter and classifier integration tests that cover candidate import and promotion.
- Run frontend TypeScript after updating E2E route mocks.
- Run the classifier-training Playwright spec if the changed E2E mocks are executable in the local environment.

---

### Task 3: Preserve Timeline Export Service/API While Removing CLI References

**Files:**
- Modify: `docs/reference/classifier-api.md`
- Modify: `README.md`

**Acceptance criteria:**
- [x] `src/humpback/services/timeline_export.py` remains unchanged unless implementation discovers a necessary import cleanup.
- [x] `POST /classifier/detection-jobs/{job_id}/timeline/export` remains documented as the supported timeline export path.
- [x] Docs no longer say timeline export is available through `scripts/export_timeline.py`.
- [x] Timeline export service tests still pass.

**Tests needed:**
- Run timeline export unit tests and timeline API integration tests.

---

### Task 4: Prune Dead Cleanup Helpers And Cleanup-Manifest Docs

**Files:**
- Modify: `src/humpback/storage.py`
- Modify: `tests/unit/test_storage.py`
- Modify: `docs/reference/storage-layout.md`

**Acceptance criteria:**
- [x] `cleanup_manifests_dir` is removed if no non-script callers remain.
- [x] `path_within_root` is removed if no non-script callers remain.
- [x] Storage docs no longer describe cleanup manifests as produced by deleted scripts.
- [x] Storage tests remain aligned with the active storage helper surface.

**Tests needed:**
- Run storage unit tests.
- Run pyright to catch any stale storage-helper imports.

---

### Task 5: Remove Remaining Current-Docs References To Deleted Scripts

**Files:**
- Modify: `README.md`
- Modify: `docs/reference/classifier-api.md`
- Modify: `docs/reference/storage-layout.md`
- Modify: `docs/reference/behavioral-constraints.md`
- Modify: `docs/reference/call-parsing-api.md`

**Acceptance criteria:**
- [x] Current docs no longer advertise deleted operator scripts, migration scripts, cleanup scripts, or old alias endpoints.
- [x] Current docs retain references to the three retained bootstrap scripts where those references describe active bootstrap behavior.
- [x] Historical specs, plans, ADRs, and migrations are left alone unless a current-doc reference points users at deleted runtime scripts.
- [x] README deployment wording no longer depends on `scripts/deploy.sh`.

**Tests needed:**
- Run reference searches for `scripts/` and `/classifier/autoresearch-candidates` in current docs.
- Run `git diff --check`.

---

### Verification

Run in order after all tasks:

1. `git ls-files scripts`
2. `rg -n "from scripts|import scripts|scripts\\." tests src/humpback frontend/src --glob '!**/__pycache__/**' --glob '!frontend/node_modules/**'`
3. `rg -n "/classifier/autoresearch-candidates|scripts/export_timeline.py|scripts/deploy.sh|scripts/convert_audio_to_flac.py|scripts/repair_hydrophone_extract_lengths.py" README.md docs/reference src/humpback frontend/src tests --glob '!frontend/node_modules/**'`
4. `uv run ruff format --check src/humpback/api/routers/classifier/__init__.py src/humpback/storage.py tests/integration/test_hyperparameter_api.py tests/integration/test_classifier_api.py tests/unit/test_storage.py`
5. `uv run ruff check src/humpback/api/routers/classifier/__init__.py src/humpback/storage.py tests/integration/test_hyperparameter_api.py tests/integration/test_classifier_api.py tests/unit/test_storage.py`
6. `uv run pyright`
7. `uv run pytest tests/unit/test_bootstrap_classifier.py tests/unit/test_bootstrap_event_classifier.py tests/unit/test_bootstrap_segmentation_dataset.py -q`
8. `uv run pytest tests/unit/test_timeline_export.py tests/integration/test_timeline_api.py -q`
9. `uv run pytest tests/integration/test_hyperparameter_api.py tests/integration/test_classifier_api.py tests/unit/test_storage.py -q`
10. `cd frontend && npx tsc --noEmit`
11. `uv run pytest tests/`
