# Code Cleanup Phase 3: Backend File Splitting — Implementation Plan

**Goal:** Split four oversized backend modules into focused packages, preserving all public APIs via `__init__.py` re-exports.
**Spec:** [docs/specs/2026-04-03-code-cleanup-design.md](../specs/2026-04-03-code-cleanup-design.md)

**Current line counts (post Phase 2):**
- `api/routers/classifier.py` — 1,673 lines
- `workers/classifier_worker.py` — 1,483 lines
- `services/classifier_service.py` — 1,483 lines
- `classifier/detector.py` — 1,073 lines

---

### Task 1: Split `api/routers/classifier.py` → package

**Files:**
- Create: `src/humpback/api/routers/classifier/__init__.py`
- Create: `src/humpback/api/routers/classifier/models.py`
- Create: `src/humpback/api/routers/classifier/training.py`
- Create: `src/humpback/api/routers/classifier/detection.py`
- Create: `src/humpback/api/routers/classifier/autoresearch.py`
- Create: `src/humpback/api/routers/classifier/hydrophone.py`
- Create: `src/humpback/api/routers/classifier/embeddings.py`
- Delete: `src/humpback/api/routers/classifier.py` (replaced by package)
- Modify: `src/humpback/api/routers/vocalization.py` (update lazy imports of `_noaa_provider_registry` and `build_archive_playback_provider` to import from the new sub-module paths or from `humpback.classifier.providers` directly)

**Module assignments:**

`models.py` — model CRUD endpoints:
- `list_models`, `get_model`, `delete_model`, `bulk_delete_models`

`training.py` — training job + retrain workflow endpoints:
- `create_training_job`, `list_training_jobs`, `get_training_job`, `delete_training_job`, `bulk_delete_training_jobs`
- `get_training_summary`
- `get_retrain_info`, `create_retrain_workflow`, `list_retrain_workflows`, `get_retrain_workflow`

`detection.py` — detection job CRUD, labels, row edits, diagnostics, content, audio/media, extraction, settings:
- All detection job endpoints (create, list, get, delete, bulk-delete, download, extract, content)
- Label endpoints (save labels, save row state, batch edit)
- Diagnostics endpoints (diagnostics, diagnostics summary)
- Audio/media endpoints (audio-slice, spectrogram)
- Settings/browse endpoints (extraction-settings, browse-directories)
- Helper classes: `ExtractRequest`, `BulkDeleteRequest`, `DetectionLabelRow`, `DetectionRowStateUpdate`, `_DecodedAudioCache`, `_NoaaPlaybackProviderRegistry`
- Helper functions: `_require_windowed_detection_job`, `_get_classifier_window_size`, `_ensure_detection_row_store_for_job`, `_serialize_label`, `_encode_wav_response`, `_resolve_detection_audio`, `_get_spectrogram_cache`
- Module-level instances: `_decoded_audio_cache`, `_noaa_provider_registry`
- Timeline sub-router mount

`autoresearch.py` — autoresearch candidate endpoints:
- `import_autoresearch_candidate`, `list_autoresearch_candidates`, `get_autoresearch_candidate`, `create_training_job_from_autoresearch_candidate`

`hydrophone.py` — hydrophone detection endpoints:
- `list_hydrophones`, `create_hydrophone_detection_job`, `list_hydrophone_detection_jobs`, `cancel_hydrophone_detection_job`, `pause_hydrophone_detection_job`, `resume_hydrophone_detection_job`

`embeddings.py` — detection embedding endpoints:
- `get_detection_embedding`, `get_embedding_status`, `generate_embeddings`, `get_embedding_generation_status`, `list_embedding_jobs`

`__init__.py` — compose sub-routers via `include_router()` so `app.py` import path does not change. Re-export `router`, `_noaa_provider_registry`, and `build_archive_playback_provider` for backward compatibility with vocalization router lazy imports.

**Acceptance criteria:**
- [ ] `api/routers/classifier/` package exists with all 7 sub-modules
- [ ] `api/routers/classifier.py` single file deleted
- [ ] `app.py` imports unchanged (`from humpback.api.routers import classifier`)
- [ ] `vocalization.py` lazy imports of `_noaa_provider_registry` work via the package
- [ ] Each sub-module has its own `APIRouter` composed into the package router in `__init__.py`
- [ ] All existing API tests pass without modification
- [ ] `uv run pyright` passes on all new files

**Tests needed:**
- No new tests — existing integration tests cover all classifier API endpoints

---

### Task 2: Split `workers/classifier_worker.py` → package

**Files:**
- Create: `src/humpback/workers/classifier/__init__.py`
- Create: `src/humpback/workers/classifier/training.py`
- Create: `src/humpback/workers/classifier/detection.py`
- Create: `src/humpback/workers/classifier/hydrophone.py`
- Delete: `src/humpback/workers/classifier_worker.py` (replaced by package)

**Module assignments:**

`training.py` — training job execution:
- `run_training_job` (handles both embedding-set and autoresearch candidate flows)

`detection.py` — local detection + extraction:
- `run_detection_job`, `run_extraction_job`
- Shared helper: `_detection_dicts_to_store_rows`

`hydrophone.py` — hydrophone detection with subprocess support:
- `run_hydrophone_detection_job`
- All hydrophone helpers: `_peak_rss_mb`, `_avg_audio_x_realtime`, `_hydrophone_provider_mode`, `_augment_hydrophone_summary`, `_resolve_model_runtime`, `_load_embedding_model_from_runtime`, `_hydrophone_detection_subprocess_main`, `_run_hydrophone_detection_in_subprocess`
- Import `_detection_dicts_to_store_rows` from `.detection`

`__init__.py` — re-export `run_training_job`, `run_detection_job`, `run_extraction_job`, `run_hydrophone_detection_job` so `runner.py` import path does not change.

**Acceptance criteria:**
- [ ] `workers/classifier/` package exists with all 4 sub-modules
- [ ] `workers/classifier_worker.py` single file deleted
- [ ] `runner.py` imports unchanged (`from humpback.workers.classifier_worker import run_*`)
- [ ] `queue.py` completion/failure imports unchanged
- [ ] All existing worker tests pass without modification
- [ ] `uv run pyright` passes on all new files

**Tests needed:**
- No new tests — existing tests cover worker functions

---

### Task 3: Split `services/classifier_service.py` → package

**Files:**
- Create: `src/humpback/services/classifier_service/__init__.py`
- Create: `src/humpback/services/classifier_service/models.py`
- Create: `src/humpback/services/classifier_service/training.py`
- Create: `src/humpback/services/classifier_service/detection.py`
- Create: `src/humpback/services/classifier_service/autoresearch.py`
- Create: `src/humpback/services/classifier_service/hydrophone.py`
- Delete: `src/humpback/services/classifier_service.py` (replaced by package)
- Modify: `src/humpback/api/routers/labeling.py` (lazy import of `get_detection_job`)

**Module assignments:**

`models.py` — classifier model CRUD:
- `list_classifier_models`, `get_classifier_model`, `delete_classifier_model`, `bulk_delete_classifier_models`

`training.py` — training job management + retrain workflows + training data summary:
- `create_training_job`, `list_training_jobs`, `get_training_job`, `delete_training_job`, `bulk_delete_training_jobs`
- `get_training_data_summary`
- `trace_folder_roots`, `collect_embedding_sets_for_folders`, `get_retrain_info`, `create_retrain_workflow`, `list_retrain_workflows`, `get_retrain_workflow`

`detection.py` — detection job management:
- `create_detection_job`, `list_detection_jobs`, `get_detection_job`, `delete_detection_job`, `bulk_delete_detection_jobs`
- `DetectionJobDependencyError`, `_check_detection_job_dependencies`
- Constants: `AUDIO_EXTENSIONS`

`autoresearch.py` — candidate import/promotion + all validation helpers:
- `import_autoresearch_candidate`, `create_training_job_from_autoresearch_candidate`
- All `_validate_*`, `_extract_*`, `_derive_*`, `_summarize_*`, `_assess_*`, `_build_*`, `_copy_*`, `_load_*`, `_default_*`, `_resolve_*` helper functions
- Constants: `AUTORESEARCH_CANDIDATE_DIRNAME`, `AUTORESEARCH_CANDIDATE_STATUS_PROMOTABLE`, `AUTORESEARCH_CANDIDATE_STATUS_BLOCKED`

`hydrophone.py` — hydrophone detection job management:
- `create_hydrophone_detection_job`, `list_hydrophone_detection_jobs`, `cancel_hydrophone_detection_job`, `pause_hydrophone_detection_job`, `resume_hydrophone_detection_job`

`__init__.py` — re-export all public functions and `DetectionJobDependencyError` so callers (classifier router, labeling router, vocalization router, retrain worker) import path does not change.

**Acceptance criteria:**
- [ ] `services/classifier_service/` package exists with all 6 sub-modules
- [ ] `services/classifier_service.py` single file deleted
- [ ] All router imports unchanged (`from humpback.services import classifier_service` or `from humpback.services.classifier_service import ...`)
- [ ] `retrain_worker.py` imports unchanged
- [ ] `labeling.py` lazy import unchanged
- [ ] All existing tests pass without modification
- [ ] `uv run pyright` passes on all new files

**Tests needed:**
- No new tests — existing integration tests cover the entire service API surface

---

### Task 4: Split `classifier/detector.py` — extract utilities (optional)

Only attempt if Tasks 1–3 complete cleanly.

**Files:**
- Create: `src/humpback/classifier/detector_utils.py`
- Modify: `src/humpback/classifier/detector.py` (move internal helpers, import from utils)

**Module assignments:**

`detector_utils.py` — window processing, I/O helpers, embedding I/O:
- Window processing: `merge_detection_spans`, `merge_detection_events`, `snap_event_bounds`, `snap_and_merge_detection_events`, `_smooth_scores`, `select_peak_windows_from_events`
- Diagnostics I/O: `_window_diagnostics_table`, `write_window_diagnostics`, `write_window_diagnostics_shard`, `read_window_diagnostics_table`, `WINDOW_DIAGNOSTICS_SCHEMA`
- TSV I/O: `read_detections_tsv`, `write_detections_tsv`, `append_detections_tsv`, `TSV_FIELDNAMES`
- Embedding I/O: `match_embedding_records_to_row_store`, `write_detection_embeddings`, `read_detection_embedding`, `EmbeddingDiffResult`, `diff_row_store_vs_embeddings`
- Audio resolution: `_build_file_timeline`, `resolve_audio_for_window`, `resolve_audio_for_window_hydrophone`
- Utility: `_file_base_epoch`

`detector.py` retains:
- `run_detection()` (main public API, imports helpers from `detector_utils`)
- `AUDIO_EXTENSIONS` constant

Re-export all moved names from `detector.py` for backward compatibility — external callers (workers, API routers, extractor, hydrophone_detector, detection_rows) continue importing from `humpback.classifier.detector`.

**Acceptance criteria:**
- [ ] `detector_utils.py` contains all extracted functions
- [ ] `detector.py` imports from `detector_utils` and re-exports for backward compatibility
- [ ] All external callers continue importing from `humpback.classifier.detector` with no changes
- [ ] All existing tests pass without modification
- [ ] `uv run pyright` passes on both files

**Tests needed:**
- No new tests — existing tests cover all detector functions

---

### Verification

Run in order after all tasks:
1. `uv run ruff format --check src/humpback/api/routers/classifier/ src/humpback/workers/classifier/ src/humpback/services/classifier_service/ src/humpback/classifier/detector.py src/humpback/classifier/detector_utils.py`
2. `uv run ruff check src/humpback/api/routers/classifier/ src/humpback/workers/classifier/ src/humpback/services/classifier_service/ src/humpback/classifier/detector.py src/humpback/classifier/detector_utils.py`
3. `uv run pyright src/humpback/api/routers/classifier/ src/humpback/workers/classifier/ src/humpback/services/classifier_service/ src/humpback/classifier/detector.py src/humpback/classifier/detector_utils.py`
4. `uv run pytest tests/`
5. `cd frontend && npx tsc --noEmit`
