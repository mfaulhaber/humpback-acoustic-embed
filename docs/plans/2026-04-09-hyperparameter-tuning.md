# Hyperparameter Tuning Page Implementation Plan

**Goal:** Replace the CLI-based autoresearch scripts with a new Classifier/Hyperparameter Tuning page that manages manifest generation, hyperparameter search, production comparison, and autoresearch candidates through the web UI.

**Spec:** [docs/specs/2026-04-09-hyperparameter-tuning-design.md](../specs/2026-04-09-hyperparameter-tuning-design.md)

---

### Task 1: Database models and migrations

**Files:**
- Create: `src/humpback/models/hyperparameter.py`
- Create: `alembic/versions/040_hyperparameter_manifests.py`
- Create: `alembic/versions/041_hyperparameter_search_jobs.py`
- Modify: `src/humpback/database.py` (import new models so Base.metadata sees them)

**Acceptance criteria:**
- [ ] `HyperparameterManifest` model with all columns from spec (id, name, status, training_job_ids, detection_job_ids, split_ratio, seed, manifest_path, example_count, split_summary, detection_job_summaries, error_message, created_at, completed_at)
- [ ] `HyperparameterSearchJob` model with all columns from spec (id, name, status, manifest_id FK, search_space, n_trials, seed, objective_name, results_dir, trials_completed, best_objective, best_config, best_metrics, comparison_model_id FK, comparison_threshold, comparison_result, error_message, created_at, completed_at)
- [ ] Both migrations use `op.batch_alter_table()` for SQLite compatibility
- [ ] `uv run alembic upgrade head` succeeds

**Tests needed:**
- Migration up/down roundtrip test
- Model instantiation and basic field access

---

### Task 2: Hyperparameter service — manifest generation

**Files:**
- Create: `src/humpback/services/hyperparameter_service/__init__.py`
- Create: `src/humpback/services/hyperparameter_service/manifest.py`
- Modify: `src/humpback/storage.py` (add `hyperparameter_manifest_path` and `hyperparameter_search_results_dir` helpers)

**Acceptance criteria:**
- [ ] `generate_manifest()` function extracted from `scripts/autoresearch/generate_manifest.py` into `manifest.py`
- [ ] Hard-negative mining removed: no `include_unlabeled_hard_negatives` parameter, no score-band classification, no `hard_negative_fraction`
- [ ] Function takes `training_job_ids`, `detection_job_ids`, `split_ratio`, `seed` and returns manifest dict
- [ ] Storage helpers return `{storage_root}/hyperparameter/manifests/{manifest_id}/manifest.json` and `{storage_root}/hyperparameter/searches/{search_id}/`
- [ ] `__init__.py` re-exports public functions

**Tests needed:**
- Manifest generation with training job sources produces correct example structure
- Manifest generation with detection job sources includes only human-labeled examples
- Split assignment distributes examples correctly
- Storage path helpers return expected paths

---

### Task 3: Hyperparameter service — search and comparison

**Files:**
- Create: `src/humpback/services/hyperparameter_service/search.py`
- Create: `src/humpback/services/hyperparameter_service/comparison.py`
- Create: `src/humpback/services/hyperparameter_service/search_space.py`
- Modify: `src/humpback/services/hyperparameter_service/__init__.py` (re-export new functions)

**Acceptance criteria:**
- [ ] `search_space.py` contains `DEFAULT_SEARCH_SPACE` (same values as current `scripts/autoresearch/search_space.py` minus `hard_negative_fraction`), `sample_config()`, `config_hash()`
- [ ] `search.py` contains `run_search()` extracted from `scripts/autoresearch/run_autoresearch.py` — accepts a custom search space dict, a progress callback for periodic DB updates, no hard-negative replay logic
- [ ] `comparison.py` contains `compare_classifiers()` and `resolve_production_classifier()` extracted from `scripts/autoresearch/compare_classifiers.py` — no hard-negative replay parameters
- [ ] Default objective function (`recall - 15*high_conf_fp_rate - 3*fp_rate`) defined in `search.py`

**Tests needed:**
- `sample_config` samples only from the provided search space dimensions
- `config_hash` is deterministic for the same config
- `run_search` with a small trial count produces expected output structure (best_run, history)
- Comparison produces metric deltas and disagreements

---

### Task 4: Rewrite autoresearch scripts as thin CLI wrappers

**Files:**
- Modify: `scripts/autoresearch/generate_manifest.py`
- Modify: `scripts/autoresearch/run_autoresearch.py`
- Modify: `scripts/autoresearch/compare_classifiers.py`
- Modify: `scripts/autoresearch/search_space.py`
- Modify: `scripts/autoresearch/objectives.py`
- Modify: `scripts/autoresearch/train_eval.py`

**Acceptance criteria:**
- [ ] `generate_manifest.py` `main()` parses CLI args, calls `hyperparameter_service.manifest.generate_manifest()`, writes JSON — same CLI interface
- [ ] `run_autoresearch.py` `main()` parses CLI args, calls `hyperparameter_service.search.run_search()` — same CLI interface (hard-negative flags become no-ops or are removed)
- [ ] `compare_classifiers.py` `main()` parses CLI args, calls `hyperparameter_service.comparison.compare_classifiers()` — same CLI interface (hard-negative flags removed)
- [ ] `search_space.py` re-exports from `hyperparameter_service.search_space`
- [ ] `objectives.py` re-exports from `hyperparameter_service.search`
- [ ] `train_eval.py` unchanged (already delegates to `humpback.classifier.replay`)
- [ ] Existing autoresearch tests still pass

**Tests needed:**
- Verify existing test_autoresearch tests pass with the refactored imports

---

### Task 5: Worker jobs — manifest and search

**Files:**
- Create: `src/humpback/workers/hyperparameter_worker.py`
- Modify: `src/humpback/workers/queue.py` (add `claim_manifest_job`, `claim_hyperparameter_search_job`, add both to `recover_stale_jobs`)
- Modify: `src/humpback/workers/runner.py` (add manifest and search job polling at the end of the priority chain)

**Acceptance criteria:**
- [ ] `run_manifest_job()` claims a queued manifest, calls service, writes artifact, updates DB row with results or error
- [ ] `run_hyperparameter_search_job()` claims a queued search, loads manifest, runs search with progress callback that updates `trials_completed`/`best_*` every ~10 trials, runs comparison if `comparison_model_id` set, writes artifacts, updates DB row
- [ ] Both workers follow existing claim pattern (atomic compare-and-set via `_claim_next_job`)
- [ ] `recover_stale_jobs` includes both new job types
- [ ] Worker priority: manifest generation after vocalization inference, search after manifest generation

**Tests needed:**
- Manifest worker sets status to complete and populates results on success
- Manifest worker sets status to failed with error_message on failure
- Search worker updates trials_completed periodically
- Search worker runs comparison when comparison_model_id is set
- Search worker handles missing manifest gracefully

---

### Task 6: Pydantic schemas for hyperparameter API

**Files:**
- Create: `src/humpback/schemas/hyperparameter.py`

**Acceptance criteria:**
- [ ] Request schemas: `ManifestCreate` (name, training_job_ids, detection_job_ids, split_ratio, seed), `SearchCreate` (name, manifest_id, search_space, n_trials, seed, comparison_model_id, comparison_threshold)
- [ ] Response schemas: `ManifestSummary`, `ManifestDetail`, `SearchSummary`, `SearchDetail`, `SearchSpaceDefaults`
- [ ] Validation: at least one of training_job_ids or detection_job_ids required, split_ratio must be 3 integers, n_trials >= 1, search_space values must be non-empty lists

**Tests needed:**
- Schema validation accepts valid payloads
- Schema validation rejects missing required fields and invalid values

---

### Task 7: API router — hyperparameter endpoints

**Files:**
- Create: `src/humpback/api/routers/classifier/hyperparameter.py`
- Modify: `src/humpback/api/routers/classifier/__init__.py` (include new router)

**Acceptance criteria:**
- [ ] `POST /classifier/hyperparameter/manifests` — creates manifest row, queues job, returns manifest summary
- [ ] `GET /classifier/hyperparameter/manifests` — lists all manifests ordered by created_at desc
- [ ] `GET /classifier/hyperparameter/manifests/{id}` — returns full manifest detail
- [ ] `DELETE /classifier/hyperparameter/manifests/{id}` — deletes manifest (409 if referenced by search job)
- [ ] `POST /classifier/hyperparameter/searches` — creates search row, queues job, returns search summary
- [ ] `GET /classifier/hyperparameter/searches` — lists all searches with progress
- [ ] `GET /classifier/hyperparameter/searches/{id}` — returns full search detail including comparison_result
- [ ] `GET /classifier/hyperparameter/searches/{id}/history` — returns search_history.json contents
- [ ] `DELETE /classifier/hyperparameter/searches/{id}` — deletes search + artifacts
- [ ] `GET /classifier/hyperparameter/search-space-defaults` — returns DEFAULT_SEARCH_SPACE
- [ ] `POST /classifier/hyperparameter/searches/{id}/import-candidate` — creates autoresearch candidate from completed search artifacts

**Tests needed:**
- CRUD operations for manifests and searches
- Delete manifest blocked when referenced by search
- Import candidate from search creates candidate with correct artifact paths
- Search space defaults endpoint returns expected structure

---

### Task 8: Relocate autoresearch candidate endpoints

**Files:**
- Modify: `src/humpback/api/routers/classifier/hyperparameter.py` (add candidate endpoints)
- Modify: `src/humpback/api/routers/classifier/autoresearch.py` (alias old paths to new ones or keep as redirects)
- Modify: `src/humpback/api/routers/classifier/__init__.py` (routing adjustments)

**Acceptance criteria:**
- [ ] Candidate endpoints available at `/classifier/hyperparameter/candidates/import`, `/classifier/hyperparameter/candidates`, `/classifier/hyperparameter/candidates/{id}`, `/classifier/hyperparameter/candidates/{id}/training-jobs`
- [ ] Old `/classifier/autoresearch-candidates/*` paths still work (redirect or alias)
- [ ] No behavioral changes to candidate import/list/detail/promote logic

**Tests needed:**
- New candidate endpoint paths return same data as old paths
- Old paths still work

---

### Task 9: Frontend — TypeScript types and API client

**Files:**
- Modify: `frontend/src/api/types.ts` (add hyperparameter types)
- Modify: `frontend/src/api/client.ts` (add hyperparameter API functions)
- Modify: `frontend/src/hooks/queries/useClassifier.ts` (add query hooks for manifests, searches, search space defaults)

**Acceptance criteria:**
- [ ] Types: `HyperparameterManifestSummary`, `HyperparameterManifestDetail`, `HyperparameterSearchSummary`, `HyperparameterSearchDetail`, `SearchSpaceDefaults`, `ManifestCreateRequest`, `SearchCreateRequest`
- [ ] API client functions: `createManifest`, `listManifests`, `getManifest`, `deleteManifest`, `createSearch`, `listSearches`, `getSearch`, `getSearchHistory`, `deleteSearch`, `getSearchSpaceDefaults`, `importCandidateFromSearch`
- [ ] TanStack Query hooks with appropriate polling for running jobs

**Tests needed:**
- TypeScript compilation passes (`npx tsc --noEmit`)

---

### Task 10: Frontend — TuningTab page component

**Files:**
- Create: `frontend/src/components/classifier/TuningTab.tsx`
- Modify: `frontend/src/App.tsx` (add route `/app/classifier/tuning`)
- Modify: `frontend/src/components/layout/AppShell.tsx` (add "Tuning" nav link under Classifier section)

**Acceptance criteria:**
- [ ] Page has three collapsible sections: Manifests, Searches, Candidates
- [ ] Manifests section: table (name, status badge, source summary, example count, split ratio, created date), "New Manifest" dialog with name, multi-select training jobs, multi-select detection jobs, split ratio, seed
- [ ] Completed manifest rows expand to show split summary and detection job breakdown
- [ ] Manifest delete action, blocked with toast if referenced by a search
- [ ] Searches section: table (name, status + progress, manifest name, best objective, comparison model, created date), "New Search" dialog with name, manifest dropdown, search space configurator (dimension rows with value checkboxes), trial count, seed, optional comparison model + threshold
- [ ] Completed search rows expand to show best config/metrics, comparison deltas, "Import as Candidate" button
- [ ] Candidates section: relocated `AutoresearchCandidatesSection` component
- [ ] AutoresearchCandidatesSection API calls updated to use new `/classifier/hyperparameter/candidates/*` paths
- [ ] "Tuning" tab appears in Classifier nav after "Embeddings"
- [ ] Remove AutoresearchCandidatesSection from TrainingTab

**Tests needed:**
- Playwright test: navigate to Tuning tab, verify three sections render
- Playwright test: create manifest dialog opens and submits
- Playwright test: create search dialog opens with search space configurator

---

### Task 11: Documentation updates

**Files:**
- Modify: `CLAUDE.md` (update §8.7 worker priority, §8.8 classifier API surface, §9.1 capabilities, §9.2 schema)
- Modify: `docs/reference/data-model.md` (add new tables)
- Modify: `docs/reference/frontend.md` (add Tuning tab)
- Modify: `docs/reference/storage-layout.md` (add hyperparameter paths)
- Modify: `scripts/autoresearch/README.md` (note that scripts are now thin wrappers, link to UI)

**Acceptance criteria:**
- [ ] CLAUDE.md reflects new worker types, API endpoints, tables, capability
- [ ] Data model reference includes both new tables
- [ ] Frontend reference includes Tuning tab route and components
- [ ] Storage layout includes `hyperparameter/` subtree
- [ ] Autoresearch README notes the UI alternative

**Tests needed:**
- None (documentation only)

---

### Verification

Run in order after all tasks:
1. `uv run alembic upgrade head`
2. `uv run ruff format --check src/humpback/models/hyperparameter.py src/humpback/services/hyperparameter_service/ src/humpback/workers/hyperparameter_worker.py src/humpback/api/routers/classifier/hyperparameter.py src/humpback/schemas/hyperparameter.py src/humpback/storage.py src/humpback/workers/queue.py src/humpback/workers/runner.py scripts/autoresearch/`
3. `uv run ruff check src/humpback/models/hyperparameter.py src/humpback/services/hyperparameter_service/ src/humpback/workers/hyperparameter_worker.py src/humpback/api/routers/classifier/hyperparameter.py src/humpback/schemas/hyperparameter.py src/humpback/storage.py src/humpback/workers/queue.py src/humpback/workers/runner.py scripts/autoresearch/`
4. `uv run pyright src/humpback/models/hyperparameter.py src/humpback/services/hyperparameter_service/ src/humpback/workers/hyperparameter_worker.py src/humpback/api/routers/classifier/hyperparameter.py src/humpback/schemas/hyperparameter.py src/humpback/storage.py src/humpback/workers/queue.py src/humpback/workers/runner.py scripts/autoresearch/`
5. `uv run pytest tests/`
6. `cd frontend && npx tsc --noEmit`
