# Multi-Label Vocalization Type Classifier Implementation Plan

**Goal:** Build a standalone vocalization type classification system with managed vocabulary, multi-label binary relevance training, durable inference jobs, and a dedicated Vocalization UI section — while removing the unused sub-window annotation system.

**Spec:** `docs/specs/2026-03-29-vocalization-classifier-design.md`

---

### Task 1: Remove Sub-Window Annotation System

Clean up the annotation infrastructure that is being replaced by window-level multi-hot labels.

**Files:**
- Modify: `src/humpback/models/labeling.py` — remove `LabelingAnnotation` class
- Modify: `src/humpback/schemas/labeling.py` — remove `AnnotationCreate`, `AnnotationOut`, `AnnotationUpdate` schemas
- Modify: `src/humpback/api/routers/labeling.py` — remove annotation CRUD endpoints, remove `LabelingAnnotation` imports
- Modify: `src/humpback/workers/classifier_worker.py` — remove annotation fallback logic from `_run_vocalization_training_job`
- Modify: `frontend/src/components/classifier/LabelingTab.tsx` — remove annotation mode state and annotation creation flow
- Delete: `frontend/src/components/classifier/AnnotationOverlay.tsx`
- Delete: `frontend/src/components/classifier/AnnotationList.tsx`
- Modify: `frontend/src/api/client.ts` — remove annotation API functions
- Modify: `frontend/src/api/types.ts` — remove annotation type interfaces
- Modify: `frontend/src/hooks/queries/useLabeling.ts` — remove annotation query hooks
- Create: `alembic/versions/029_drop_labeling_annotations.py` — migration to drop `labeling_annotations` table

**Acceptance criteria:**
- [ ] `LabelingAnnotation` model removed from codebase
- [ ] All annotation API endpoints removed (no `/labeling/annotations/` routes)
- [ ] `AnnotationOverlay` and `AnnotationList` components deleted
- [ ] Annotation mode removed from `LabelingTab`
- [ ] `_run_vocalization_training_job` no longer queries `labeling_annotations`
- [ ] Alembic migration drops `labeling_annotations` table
- [ ] Migration runs cleanly: `uv run alembic upgrade head`
- [ ] Existing vocalization label functionality still works

**Tests needed:**
- Verify annotation endpoints return 404 / are absent from router
- Verify vocalization label CRUD still functions after annotation removal
- Verify migration up/down works

---

### Task 2: Data Model — Vocalization Tables and Migration

Add the four new vocalization tables via SQLAlchemy models and Alembic migration.

**Files:**
- Create: `src/humpback/models/vocalization.py` — `VocalizationType`, `VocalizationModel`, `VocalizationTrainingJob`, `VocalizationInferenceJob` SQLAlchemy models
- Modify: `src/humpback/database.py` — import new models so they register with `Base.metadata` (if needed for table creation)
- Create: `alembic/versions/030_vocalization_tables.py` — migration adding `vocalization_types`, `vocalization_models`, `vocalization_training_jobs`, `vocalization_inference_jobs`

**Acceptance criteria:**
- [ ] Four new SQLAlchemy models with fields matching spec section 4
- [ ] `VocalizationType` has unique constraint on `name`
- [ ] `VocalizationModel` has `is_active` with constraint that at most one is active
- [ ] `VocalizationTrainingJob` has nullable FK to `vocalization_models`
- [ ] `VocalizationInferenceJob` has FK to `vocalization_models`
- [ ] All models use `UUIDMixin` and `TimestampMixin` consistent with existing models
- [ ] Migration runs cleanly: `uv run alembic upgrade head`
- [ ] Uses `op.batch_alter_table()` for SQLite compatibility

**Tests needed:**
- Model creation and query round-trip tests
- Unique constraint on `VocalizationType.name`
- Active model constraint enforcement

---

### Task 3: Pydantic Schemas for Vocalization API

Define request/response schemas for all vocalization endpoints.

**Files:**
- Create: `src/humpback/schemas/vocalization.py` — all Pydantic models for vocabulary CRUD, training jobs, models, inference jobs, results

**Acceptance criteria:**
- [ ] `VocalizationTypeCreate` / `VocalizationTypeOut` for vocabulary CRUD
- [ ] `VocalizationTypeImportRequest` — list of embedding set IDs to scan
- [ ] `VocalizationTrainingJobCreate` — source_config (embedding set IDs + detection job IDs), parameters (classifier_type, l2_normalize, class_weight, min_examples_per_type)
- [ ] `VocalizationTrainingJobOut` — status, result_summary, vocalization_model_id
- [ ] `VocalizationModelOut` — per_class_thresholds, per_class_metrics, vocabulary_snapshot, is_active
- [ ] `VocalizationInferenceJobCreate` — vocalization_model_id, source_type, source_id
- [ ] `VocalizationInferenceJobOut` — status, result_summary
- [ ] `VocalizationPredictionRow` — window identity + per-type scores
- [ ] All schemas pass Pyright

**Tests needed:**
- Schema validation for create requests (required fields, valid enums)
- Serialization round-trip tests

---

### Task 4: Vocabulary Service — CRUD and Embedding Set Import

Business logic for managing the vocalization type vocabulary, including auto-import from embedding set folder structure.

**Files:**
- Create: `src/humpback/services/vocalization_service.py` — vocabulary CRUD, embedding set folder scanning, import logic

**Acceptance criteria:**
- [ ] `list_types()`, `create_type()`, `update_type()`, `delete_type()` CRUD operations
- [ ] `delete_type()` fails if the type is referenced by an active model's vocabulary snapshot
- [ ] `import_from_embedding_sets(embedding_set_ids)` scans selected embedding sets, reads subfolder names from their parquet `filename` column, extracts unique folder names, creates `VocalizationType` rows for each
- [ ] Import normalizes folder names to lowercase
- [ ] Import skips folder names that already exist in the vocabulary
- [ ] Import returns a summary of added vs skipped types

**Tests needed:**
- CRUD operations with validation
- Import from embedding set with mock parquet data containing folder-structured filenames
- Import deduplication (re-importing same set doesn't create duplicates)
- Delete protection when type is in active model vocabulary

---

### Task 5: Multi-Label Training Pipeline

Replace the existing single-label `train_label_classifier` with a multi-label binary relevance trainer that produces N independent classifiers.

**Files:**
- Create: `src/humpback/classifier/vocalization_trainer.py` — multi-label training logic: data collection, multi-label-aware negative construction, per-type sklearn pipeline training, threshold optimization, model artifact saving
- Modify: `src/humpback/services/vocalization_service.py` — add training job creation and data collection from embedding sets + detection jobs

**Acceptance criteria:**
- [ ] Collects embeddings from curated embedding sets (folder name → type mapping) and detection job vocalization labels
- [ ] Deduplicates embeddings by `(filename, start_sec, end_sec)` when combining sources
- [ ] Multi-label-aware negative construction: a window labeled with types A and B is positive for both A and B classifiers, negative for neither
- [ ] Filters types below `min_examples_per_type` (default 4), reports filtered types with counts
- [ ] Trains independent sklearn Pipeline per type: optional L2 Normalizer → StandardScaler → LogisticRegression (class_weight='balanced')
- [ ] Runs 5-fold stratified CV per type, computes per-class AP, F1, precision, recall
- [ ] Auto-optimizes per-class threshold (maximizes F1 on held-out folds)
- [ ] Saves per-type `.joblib` files + `metadata.json` to model directory
- [ ] Creates `VocalizationModel` row with vocabulary snapshot, thresholds, metrics

**Tests needed:**
- Multi-label-aware negative construction with synthetic data: verify a window with labels [A, B] is positive for A, positive for B, negative for C, and NOT negative for A or B
- Training with min_examples_per_type filtering: types with fewer than threshold are skipped
- Threshold optimization produces per-type values between 0 and 1
- End-to-end training with synthetic embeddings produces correct number of `.joblib` files
- Model metadata.json contains correct vocabulary snapshot and thresholds

---

### Task 6: Inference Pipeline

Score embedding windows through per-type classifiers and persist results.

**Files:**
- Create: `src/humpback/classifier/vocalization_inference.py` — load model artifacts, score embeddings, write predictions parquet
- Modify: `src/humpback/services/vocalization_service.py` — add inference job creation, embedding loading from detection jobs and embedding sets

**Acceptance criteria:**
- [ ] Loads N per-type `.joblib` pipelines from model directory
- [ ] Scores each embedding through all N pipelines, produces raw probability scores
- [ ] Supports three source types: detection_job, embedding_set, rescore
- [ ] For detection_job source: loads embeddings from detection job's parquet, includes UTC bounds
- [ ] For embedding_set source: loads embeddings from embedding set's parquet
- [ ] For rescore source: re-scores windows from a previous inference job's parquet
- [ ] Writes predictions parquet with window identity columns + one float column per type
- [ ] Creates result_summary with per-type counts at stored thresholds

**Tests needed:**
- Inference with synthetic model and embeddings produces correct parquet schema
- All three source types load embeddings correctly
- Threshold application produces correct tag counts in result_summary

---

### Task 7: Vocalization API Router

FastAPI router with all vocalization endpoints.

**Files:**
- Create: `src/humpback/api/routers/vocalization.py` — router with all endpoints from spec section 7
- Modify: `src/humpback/api/app.py` (or equivalent app setup file) — register the new router

**Acceptance criteria:**
- [ ] Router mounted at `/vocalization/` prefix
- [ ] Vocabulary endpoints: GET/POST types, PUT/DELETE type by id, POST import
- [ ] Training endpoints: GET models, GET model detail, PUT activate, POST/GET training jobs
- [ ] Inference endpoints: POST/GET inference jobs, GET job detail, GET results (paginated with threshold overrides), GET export (TSV/CSV)
- [ ] Results endpoint accepts optional per-type threshold query params
- [ ] Export endpoint applies thresholds and generates downloadable file
- [ ] All endpoints use proper Pydantic schemas and HTTP status codes

**Tests needed:**
- Integration tests for each endpoint group (vocabulary, training, inference)
- Vocabulary import endpoint with real embedding set
- Results endpoint with and without threshold overrides
- Export produces valid TSV with applied thresholds

---

### Task 8: Worker Integration

Add vocalization training and inference job processing to the worker loop.

**Files:**
- Create: `src/humpback/workers/vocalization_worker.py` — `run_vocalization_training_job()` and `run_vocalization_inference_job()` functions
- Modify: `src/humpback/workers/queue.py` — add `claim_vocalization_training_job()`, `claim_vocalization_inference_job()`, extend `recover_stale_jobs()`
- Modify: `src/humpback/workers/runner.py` — add vocalization job claims after retrain in the priority loop

**Acceptance criteria:**
- [ ] `claim_vocalization_training_job()` uses atomic compare-and-set pattern
- [ ] `claim_vocalization_inference_job()` uses atomic compare-and-set pattern
- [ ] Worker loop tries vocalization training after retrain, then vocalization inference
- [ ] `recover_stale_jobs()` recovers stuck vocalization jobs
- [ ] Training worker calls vocalization_trainer, persists model artifacts, updates job status
- [ ] Inference worker calls vocalization_inference, persists predictions, updates job status
- [ ] Failed jobs get status='failed' with error message in result_summary

**Tests needed:**
- Claim semantics with compare-and-set
- Job lifecycle: queued → running → complete
- Job lifecycle: queued → running → failed (with error capture)
- Stale job recovery

---

### Task 9: Frontend — API Client, Types, and Query Hooks

Add TypeScript types, API client functions, and TanStack Query hooks for the vocalization domain.

**Files:**
- Modify: `frontend/src/api/types.ts` — add vocalization interfaces (VocalizationType, VocalizationModel, VocalizationTrainingJob, VocalizationInferenceJob, VocalizationPredictionRow)
- Modify: `frontend/src/api/client.ts` — add vocalization API functions
- Create: `frontend/src/hooks/queries/useVocalization.ts` — TanStack Query hooks for vocabulary, models, training jobs, inference jobs, results

**Acceptance criteria:**
- [ ] TypeScript interfaces match Pydantic response schemas
- [ ] API client functions for all vocalization endpoints
- [ ] Query hooks with appropriate polling intervals for active jobs
- [ ] Mutation hooks for vocabulary CRUD, training/inference job creation, model activation
- [ ] Results query hook accepts per-type threshold overrides
- [ ] All types pass `npx tsc --noEmit`

**Tests needed:**
- TypeScript compilation passes

---

### Task 10: Frontend — Navigation and Routing

Add the Vocalization nav section and route configuration.

**Files:**
- Modify: `frontend/src/components/layout/SideNav.tsx` — add Vocalization nav group with Training and Labeling children
- Modify: `frontend/src/App.tsx` — add routes for `/app/vocalization/training` and `/app/vocalization/labeling`

**Acceptance criteria:**
- [ ] "Vocalization" appears in side nav with Training and Labeling sub-items
- [ ] Vocalization nav uses an appropriate icon (e.g., `AudioWaveform` or `Waves` from lucide-react)
- [ ] Routes render placeholder components (replaced in tasks 11–12)
- [ ] Existing Classifier nav and routes unchanged
- [ ] Remove "Labeling" from under Classifier nav (it moves to Vocalization)

**Tests needed:**
- Navigation renders without errors
- Routes resolve correctly

---

### Task 11: Frontend — Vocalization Training Page

Build the training page with vocabulary manager, training form, and model list.

**Files:**
- Create: `frontend/src/components/vocalization/VocalizationTrainingTab.tsx` — main training page with three panels
- Create: `frontend/src/components/vocalization/VocabularyManager.tsx` — type table with add/edit/delete, import from embedding set dialog
- Create: `frontend/src/components/vocalization/VocalizationTrainForm.tsx` — source selector, parameter controls, train button
- Create: `frontend/src/components/vocalization/VocalizationModelList.tsx` — model table with expandable per-class metrics, activate/delete actions

**Acceptance criteria:**
- [ ] Vocabulary table shows all types with name, description, created date
- [ ] Add type: inline form or dialog with name + optional description
- [ ] Edit type: inline edit or dialog
- [ ] Delete type: confirmation dialog, fails gracefully if type is in active model
- [ ] Import dialog: lists embedding sets, user selects which to scan, shows preview of types to import
- [ ] Train form: multi-select for embedding sets and detection jobs, parameter inputs, train button
- [ ] Active training job shows status with polling
- [ ] Model table: name, date, types count, mean F1, active badge
- [ ] Expandable model rows: per-class AP, F1, precision, recall, sample count, threshold
- [ ] Activate button sets model as active (deactivates previous)
- [ ] All components use shadcn/ui primitives

**Tests needed:**
- Vocabulary CRUD interactions
- Training job submission and status polling
- Model activation toggle

---

### Task 12: Frontend — Vocalization Labeling Page

Build the labeling page with inference job management and results browser.

**Files:**
- Create: `frontend/src/components/vocalization/VocalizationLabelingTab.tsx` — main labeling page with two panels
- Create: `frontend/src/components/vocalization/VocalizationInferenceForm.tsx` — model selector, source selector, queue button
- Create: `frontend/src/components/vocalization/VocalizationResultsBrowser.tsx` — paginated results with spectrogram, playback, type tags, threshold sliders, export

**Acceptance criteria:**
- [ ] Inference form: select active model (or specific model), select source (detection job or embedding set), queue button
- [ ] Inference job list: status, source, model, result summary, expandable per-type tag counts
- [ ] Results browser appears when a completed job is selected
- [ ] Each result row: spectrogram thumbnail, audio playback button, type tags as colored badges
- [ ] Type tags show only types above their threshold
- [ ] Per-type threshold sliders: one slider per type, applied client-side to filter tags
- [ ] Export button: downloads TSV/CSV with currently applied thresholds
- [ ] Rescore option: re-run inference on previous results with a different model
- [ ] All components use shadcn/ui primitives

**Tests needed:**
- Inference job submission and status polling
- Results display with threshold filtering
- Export download

---

### Task 13: Remove Legacy Vocalization Training from Classifier

Clean up the old vocalization training code that lived on the binary classifier's infrastructure.

**Files:**
- Modify: `src/humpback/workers/classifier_worker.py` — remove `_run_vocalization_training_job` function and related imports
- Modify: `src/humpback/api/routers/labeling.py` — remove training job endpoints that are now on `/vocalization/`
- Delete: `src/humpback/classifier/label_trainer.py` — replaced by `vocalization_trainer.py`
- Delete: `frontend/src/components/classifier/VocalizationTrainingPanel.tsx` — replaced by vocalization training page
- Modify: `frontend/src/components/classifier/TrainingTab.tsx` — remove vocalization training panel references
- Modify: `frontend/src/components/classifier/LabelingTab.tsx` — remove vocalization-specific UI elements that moved to vocalization section

**Acceptance criteria:**
- [ ] No vocalization training code remains in `classifier_worker.py`
- [ ] No vocalization training endpoints remain on `/labeling/` router
- [ ] `label_trainer.py` deleted (functionality replaced by `vocalization_trainer.py`)
- [ ] `VocalizationTrainingPanel.tsx` deleted
- [ ] Classifier TrainingTab no longer references vocalization training
- [ ] Binary classifier training still works end-to-end

**Tests needed:**
- Binary classifier training unaffected
- No stale imports or references

---

### Task 14: Documentation Updates

Update project documentation to reflect the new vocalization system.

**Files:**
- Modify: `CLAUDE.md` — update §3.7 (frontend structure), §8.3 (data model), §8.5 (storage layout), §9.1 (implemented capabilities), §9.2 (schema)
- Modify: `DECISIONS.md` — append ADR for multi-label vocalization classifier design
- Modify: `README.md` — add vocalization section to feature list and API docs

**Acceptance criteria:**
- [ ] CLAUDE.md reflects new tables, routes, components, storage paths
- [ ] Frontend file structure in §3.7 includes `vocalization/` component directory
- [ ] Data model summary in §8.3 includes all four vocalization tables
- [ ] Storage layout in §8.5 includes vocalization_models and vocalization_inference paths
- [ ] DECISIONS.md has ADR for multi-label binary relevance approach
- [ ] README.md lists vocalization classification as a feature

**Tests needed:**
- N/A (documentation only)

---

### Verification

Run in order after all tasks:
1. `uv run ruff format --check src/humpback/models/vocalization.py src/humpback/schemas/vocalization.py src/humpback/services/vocalization_service.py src/humpback/classifier/vocalization_trainer.py src/humpback/classifier/vocalization_inference.py src/humpback/api/routers/vocalization.py src/humpback/workers/vocalization_worker.py src/humpback/workers/queue.py src/humpback/workers/runner.py`
2. `uv run ruff check src/humpback/ tests/`
3. `uv run pyright src/humpback/ tests/`
4. `uv run pytest tests/`
5. `cd frontend && npx tsc --noEmit`
