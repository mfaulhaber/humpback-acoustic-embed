# Scripts Folder Cleanout Design

**Status:** Draft for discussion
**Date:** 2026-05-07
**Track:** Code health, core platform, ingest detection, signal timeline

---

## 1. Goal

Clean out the repo-root `scripts/` folder so it contains only the retained
bootstrap scripts:

- `scripts/bootstrap_classifier.py`
- `scripts/bootstrap_event_classifier_dataset.py`
- `scripts/bootstrap_segmentation_dataset.py`

Everything else under `scripts/` is either a one-off migration/backfill,
operator utility, recovery tool, deprecated compatibility wrapper, or a CLI
wrapper around production code. Removing these scripts should reduce typecheck
surface area, stale docs, and script-specific tests while preserving the active
runtime workflows.

Two production workflows are explicitly retained:

- Timeline export remains a supported service/API workflow.
- The real hyperparameter/candidate workflow remains available under
  `/classifier/hyperparameter/*`; only the old
  `/classifier/autoresearch-candidates/*` alias API is removed.

---

## 2. Review Inputs

The review inspected:

- Tracked files under `scripts/`
- Direct imports from tests into `scripts.*`
- References to script paths in docs, API docs, frontend code, and tests
- Timeline export service/API callers
- Hyperparameter candidate and old autoresearch alias endpoints
- Domain capsules for core platform, ingest detection, signal timeline, and
  frontend shell

Current direct script-test inventory:

- 8 non-bootstrap script-specific test files, 81 test functions
- 3 bootstrap script test files, 38 test functions

---

## 3. Current State

### 3.1 Retained Scripts

The bootstrap scripts are still active bootstrap-era workflows for Call Parsing
training data and classifier setup. They are documented as direct trainer calls,
not queued worker jobs, and their tests should remain.

Retained tests:

- `tests/unit/test_bootstrap_classifier.py`
- `tests/unit/test_bootstrap_event_classifier.py`
- `tests/unit/test_bootstrap_segmentation_dataset.py`

### 3.2 Scripts To Remove

Remove all non-bootstrap tracked files from `scripts/`:

- `scripts/README.md`
- `scripts/backfill_detection_metadata.py`
- `scripts/backfill_training_summary.py`
- `scripts/benchmark_region_detection.py`
- `scripts/cleanup_legacy_workflows.py`
- `scripts/cleanup_sequence_model_artifacts.py`
- `scripts/cleanup_short_negatives.py`
- `scripts/convert_audio_to_flac.py`
- `scripts/deploy.sh`
- `scripts/export_timeline.py`
- `scripts/fix_duplicate_labels.py`
- `scripts/migrate_row_ids.py`
- `scripts/migrate_sequence_model_timestamps.py`
- `scripts/noaa_detection_metadata.py`
- `scripts/recover_event_boundary_corrections.py`
- `scripts/recover_vocalization_corrections.py`
- `scripts/reorder_vocalization_layout.py`
- `scripts/repair_hydrophone_extract_lengths.py`
- `scripts/sb_diagnostic.py`
- `scripts/stage_s3_epoch_cache.py`
- `scripts/validate_gap_filling.py`

These fall into a few buckets:

- One-time migrations/backfills already superseded by current schema/artifacts
- Cleanup/recovery tools for retired workflows
- Local operator utilities that do not belong in the main application surface
- Script-era diagnostics and benchmarking tools
- CLI wrappers over production services

### 3.3 Direct Tests To Remove

Delete the script-only tests that import the removed scripts:

- `tests/unit/test_convert_audio_to_flac.py`
- `tests/unit/test_migrate_row_ids.py`
- `tests/unit/test_noaa_detection_metadata.py`
- `tests/unit/test_repair_hydrophone_extract_lengths.py`
- `tests/unit/test_stage_s3_epoch_cache.py`
- `tests/scripts/test_cleanup_legacy_workflows.py`
- `tests/scripts/test_cleanup_sequence_model_artifacts.py`
- `tests/scripts/test_migrate_sequence_model_timestamps.py`

No direct tests were found for several removed scripts, including the
backfill/recovery utilities, `deploy.sh`, and `validate_gap_filling.py`.

### 3.4 Timeline Export

`scripts/export_timeline.py` is only a CLI wrapper. The production timeline
export behavior lives in `src/humpback/services/timeline_export.py` and is
exposed through `POST /classifier/detection-jobs/{job_id}/timeline/export`.

Keep:

- `src/humpback/services/timeline_export.py`
- `src/humpback/api/routers/timeline.py` export endpoint
- `tests/unit/test_timeline_export.py`

Remove only the CLI script and docs that describe it as an alternate invocation
path.

### 3.5 Hyperparameter Candidate Workflow

The active candidate workflow is the hyperparameter API surface:

- `/classifier/hyperparameter/candidates/import`
- `/classifier/hyperparameter/candidates`
- `/classifier/hyperparameter/candidates/{candidate_id}`
- `/classifier/hyperparameter/candidates/{candidate_id}/training-jobs`
- `/classifier/hyperparameter/searches/{search_id}/import-candidate`

Keep:

- Hyperparameter models, services, workers, schemas, and frontend API calls
- Candidate-backed replay training and provenance behavior
- Tests for the canonical hyperparameter/candidate workflow

Remove:

- `src/humpback/api/routers/classifier/autoresearch.py`
- The router include for the old alias paths in
  `src/humpback/api/routers/classifier/__init__.py`
- Tests that assert old `/classifier/autoresearch-candidates/*` paths still
  work
- Stale frontend E2E route mocks that still intercept the old alias path

### 3.6 Storage Helpers

`cleanup_manifests_dir()` and `path_within_root()` in `src/humpback/storage.py`
appear to become dead after removing the cleanup scripts. If no non-script
callers remain during implementation, remove these helpers and adjust storage
tests if necessary.

---

## 4. Non-Goals

- Do not remove the three retained bootstrap scripts.
- Do not remove timeline export service/API behavior.
- Do not remove `tests/unit/test_timeline_export.py`.
- Do not remove the canonical hyperparameter/candidate API workflow.
- Do not remove candidate-backed replay training, `AutoresearchCandidate`
  persistence, or existing candidate provenance fields.
- Do not remove hyperparameter manifests, search jobs, workers, or frontend
  tuning UI.
- Do not change database schema or write a migration.
- Do not delete historical specs, plans, ADRs, or migration files merely because
  they mention removed scripts.
- Do not replace removed operator scripts with new scripts in another folder in
  this cleanup.

---

## 5. Proposed Changes

### 5.1 Delete Non-Bootstrap Scripts

Delete every tracked file under `scripts/` except `bootstrap_*.py`.

After implementation, `git ls-files scripts` should list only:

- `scripts/bootstrap_classifier.py`
- `scripts/bootstrap_event_classifier_dataset.py`
- `scripts/bootstrap_segmentation_dataset.py`

### 5.2 Delete Script-Only Tests

Delete the eight direct test files for removed scripts. Keep all bootstrap
tests unchanged unless deletion exposes import or pyright issues.

### 5.3 Remove Old Autoresearch Alias API

Remove the alias router and its include:

- Delete `src/humpback/api/routers/classifier/autoresearch.py`
- Remove `autoresearch_router` import and `router.include_router(...)` from
  `src/humpback/api/routers/classifier/__init__.py`

Update tests:

- Remove old-alias assertions from `tests/integration/test_hyperparameter_api.py`
- Move or update candidate import/promotion tests in
  `tests/integration/test_classifier_api.py` to use the canonical
  `/classifier/hyperparameter/candidates/*` paths, or relocate them to
  `tests/integration/test_hyperparameter_api.py`
- Update stale frontend E2E route mocks in
  `frontend/e2e/classifier-training.spec.ts` to use canonical candidate paths

The implementation should not alter service-layer candidate behavior.

### 5.4 Keep Timeline Export Service/API

Delete `scripts/export_timeline.py`, but keep the service and API endpoint.

Update docs so they describe only the API path, not the removed CLI. Retained
timeline export tests continue to cover the service.

### 5.5 Prune Dead Storage Helpers

If implementation confirms there are no remaining non-script callers, delete:

- `cleanup_manifests_dir`
- `path_within_root`

This is a minor follow-on cleanup from removing cleanup scripts. If either
helper is still used outside removed scripts, keep it.

### 5.6 Documentation Updates

Update current docs and references that point to removed scripts:

- `README.md`
  - Remove `scripts/deploy.sh` as the deployment `.env` loader.
  - Remove the Utilities section for `convert_audio_to_flac.py` and
    `repair_hydrophone_extract_lengths.py`.
  - Update the API endpoint table if it still lists old
    `/classifier/autoresearch-candidates/*` paths.
- `docs/reference/classifier-api.md`
  - Remove the note that timeline export is also available through
    `scripts/export_timeline.py`.
  - Remove or rewrite old autoresearch alias documentation. Canonical candidate
    endpoints live under `/classifier/hyperparameter/candidates/*`.
- `docs/reference/storage-layout.md`
  - Remove cleanup-manifest wording that says manifests are written by the
    deleted cleanup scripts.
- `docs/reference/behavioral-constraints.md` and
  `docs/reference/call-parsing-api.md`
  - Keep bootstrap-script references where they refer to retained
    `bootstrap_*.py` scripts.

Historical specs/plans may keep old script references as historical record.

---

## 6. Approaches Considered

### Approach A: Delete Scripts Only

Delete non-bootstrap scripts and their direct tests, but leave all production
APIs and docs alone.

Pros:

- Smallest code diff.
- Lowest immediate runtime risk.

Cons:

- Leaves stale old autoresearch alias API in place.
- Leaves docs advertising removed CLIs.
- Leaves dead storage helpers and compatibility tests around.

### Approach B: Delete Scripts Plus Stale Compatibility Surface

Delete non-bootstrap scripts, their direct tests, stale docs, and the old
`/classifier/autoresearch-candidates/*` alias API while preserving the real
hyperparameter candidate workflow and timeline export service/API.

Pros:

- Aligns code, tests, and docs with the current product surface.
- Removes script-era compatibility without disturbing canonical workflows.
- Keeps timeline export available where it is still a production service/API.

Cons:

- Requires touching API router wiring and integration tests.
- Any external caller still using the old alias path must move to
  `/classifier/hyperparameter/candidates/*`.

### Approach C: Delete Scripts And Timeline Export Workflow

Delete all non-bootstrap scripts and also remove timeline export service/API.

Pros:

- Larger reduction in signal-timeline surface area.

Cons:

- Removes an explicitly retained production workflow.
- Deletes well-covered service behavior unrelated to the scripts cleanup.

### Recommendation

Use Approach B.

It matches the desired cleanup: `scripts/` becomes bootstrap-only, timeline
export remains supported through the API, and the old script-era autoresearch
alias goes away while canonical hyperparameter/candidate behavior stays intact.

---

## 7. Implementation Notes

Suggested implementation order:

1. Delete non-bootstrap scripts and direct script tests.
2. Remove old autoresearch alias router and update integration/E2E references
   to canonical hyperparameter candidate endpoints.
3. Remove dead storage helpers if no callers remain.
4. Update README and reference docs.
5. Run targeted tests, then full backend verification.

Pay attention to:

- `pyproject.toml` and `.pre-commit-config.yaml` can still include `scripts`
  because bootstrap scripts remain in that folder.
- Do not remove `scripts/` from pyright include unless the retained bootstrap
  scripts are moved, which is out of scope.
- If local untracked files such as `scripts/__pycache__/` or `.DS_Store` exist,
  they are workspace hygiene, not tracked code changes.

---

## 8. Test Plan

Targeted backend checks:

- `uv run ruff format --check` on modified Python and docs-adjacent files where
  applicable
- `uv run ruff check` on modified Python files
- `uv run pyright`
- `uv run pytest tests/unit/test_bootstrap_classifier.py tests/unit/test_bootstrap_event_classifier.py tests/unit/test_bootstrap_segmentation_dataset.py -q`
- `uv run pytest tests/unit/test_timeline_export.py tests/integration/test_timeline_api.py -q`
- `uv run pytest tests/integration/test_hyperparameter_api.py tests/integration/test_classifier_api.py -q`
- `uv run pytest tests/unit/test_storage.py -q` if storage helpers are removed

Frontend/API-client checks if E2E route mocks or TypeScript references change:

- `cd frontend && npx tsc --noEmit`
- `cd frontend && npx playwright test e2e/classifier-training.spec.ts` if the
  skipped candidate test is re-enabled or its route mocks are materially changed

Final verification:

- `uv run pytest tests/`

---

## 9. Acceptance Criteria

- `git ls-files scripts` shows only the three retained `bootstrap_*.py` files.
- No test imports a removed `scripts.*` module.
- Bootstrap script tests still pass.
- Timeline export service/API tests still pass.
- `/classifier/hyperparameter/candidates/*` and
  `/classifier/hyperparameter/searches/{id}/import-candidate` remain supported.
- `/classifier/autoresearch-candidates/*` is no longer registered.
- Current docs no longer advertise removed scripts or old alias endpoints.
- `uv run pyright` passes with the remaining bootstrap scripts in scope.
- Full backend tests pass.

---

## 10. Open Questions

- Should external deployment continue to use a repo-managed shell entrypoint
  after `scripts/deploy.sh` is removed, or should deployment remain outside this
  repository?
- Should old `/classifier/autoresearch-candidates/*` callers receive a short
  deprecation window instead of immediate removal? The current requested
  direction is immediate removal of the alias.
- Should the deleted operator utilities be archived outside the repo for
  one-off future recovery, or is git history sufficient?
