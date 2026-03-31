# Training Dataset Review & Editing — Implementation Plan

**Goal:** Introduce training datasets as a first-class concept with a review/editing UI for vocalization model training data.
**Spec:** [docs/specs/2026-03-31-training-dataset-review-design.md](../specs/2026-03-31-training-dataset-review-design.md)

---

### Task 1: Database Migration, Models, Storage, and Schemas

**Files:**
- Create: `alembic/versions/032_training_datasets.py`
- Create: `src/humpback/models/training_dataset.py`
- Modify: `src/humpback/models/vocalization.py` (add training_dataset_id FK to VocalizationClassifierModel and VocalizationTrainingJob)
- Modify: `src/humpback/storage.py` (add training_dataset_dir, training_dataset_parquet_path helpers)
- Modify: `src/humpback/schemas/vocalization.py` (add training dataset and label request/response schemas)

**Acceptance criteria:**
- [ ] `training_datasets` table created with columns: id, name, source_config, parquet_path, total_rows, vocabulary, created_at, updated_at
- [ ] `training_dataset_labels` table created with columns: id, training_dataset_id (FK), row_index, label, source, created_at, updated_at; indexed on (training_dataset_id, row_index)
- [ ] `vocalization_models.training_dataset_id` nullable FK column added
- [ ] `vocalization_training_jobs.training_dataset_id` nullable FK column added
- [ ] Migration uses `op.batch_alter_table()` for SQLite compatibility on ALTER operations
- [ ] SQLAlchemy models TrainingDataset and TrainingDatasetLabel defined with UUIDMixin and TimestampMixin
- [ ] Storage helpers return `{storage_root}/training_datasets/{dataset_id}/` and `.../embeddings.parquet`
- [ ] Pydantic schemas defined: TrainingDatasetOut, TrainingDatasetRowOut, TrainingDatasetLabelCreate, TrainingDatasetLabelOut, TrainingDatasetExtendRequest
- [ ] `uv run alembic upgrade head` succeeds

**Tests needed:**
- Migration applies and rolls back cleanly
- Storage helpers return expected paths

---

### Task 2: Snapshot Logic

**Files:**
- Create: `src/humpback/services/training_dataset.py`

**Acceptance criteria:**
- [ ] `create_training_dataset_snapshot()` async function that takes a SQLAlchemy session, source_config, storage_root, and returns a TrainingDataset record
- [ ] Detection job sources: reads detection_embeddings.parquet, queries vocalization_labels, matches by UTC key, writes rows to unified parquet, copies labels to training_dataset_labels with source="snapshot"
- [ ] Embedding set sources: reads embedding parquet, looks up EmbeddingSet + AudioFile for filename and processing params, computes window start_sec/end_sec from row_index, writes rows with source_type="embedding_set", creates labels from folder-inferred type name
- [ ] Only explicitly labeled detection job rows are included (unlabeled windows excluded)
- [ ] Unified parquet schema: row_index, embedding, source_type, source_id, filename, start_sec, end_sec, confidence
- [ ] Deduplication by composite key (same logic as current worker)
- [ ] `extend_training_dataset()` async function that appends new source rows to an existing dataset parquet, inserts new training_dataset_labels, updates total_rows and vocabulary on the dataset record
- [ ] Row indices in extension continue from existing total_rows

**Tests needed:**
- Snapshot from detection job source produces correct parquet rows and labels
- Snapshot from embedding set source computes correct window positions and infers type
- Extend appends rows with correct row_index continuation
- Unlabeled detection rows are excluded
- "(Negative)" labels snapshot as empty label set (or as "(Negative)" label — match spec)
- Deduplication prevents duplicate rows

---

### Task 3: Vocalization Worker Changes

**Files:**
- Modify: `src/humpback/workers/vocalization_worker.py`

**Acceptance criteria:**
- [ ] `run_vocalization_training_job()` handles two modes: (a) source_config provided → calls create_training_dataset_snapshot, then trains; (b) training_dataset_id provided → loads existing dataset, trains from it
- [ ] Training reads embeddings from `training_datasets/{id}/embeddings.parquet` and labels from `training_dataset_labels` table (replaces current source-walking logic in lines 50-158)
- [ ] Multi-hot label assembly reads from training_dataset_labels grouped by row_index
- [ ] Model record gets training_dataset_id set after successful training
- [ ] Training job record gets training_dataset_id set
- [ ] Existing source-walking code (direct embedding set folder inference, direct detection job label matching) removed from training path — moved to snapshot logic in Task 2

**Tests needed:**
- Training job with source_config creates dataset and trains
- Training job with training_dataset_id reuses existing dataset
- Label edits on training_dataset_labels are reflected in retrain results
- Worker handles empty datasets gracefully (no labeled rows)

---

### Task 4: API Endpoints

**Files:**
- Modify: `src/humpback/api/routers/vocalization.py`

**Acceptance criteria:**
- [ ] `GET /vocalization/training-datasets` — lists all datasets, newest first
- [ ] `GET /vocalization/training-datasets/{id}` — dataset details including total_rows, vocabulary, source_config
- [ ] `GET /vocalization/training-datasets/{id}/rows` — paginated rows with labels; query params: type (filter by vocalization type), group ("positive" or "negative" relative to type), offset, limit (default 50); returns row_index, filename, start_sec, end_sec, source_type, source_id, confidence, labels array
- [ ] `GET /vocalization/training-datasets/{id}/spectrogram?row_index=N` — reads source_type/source_id from parquet row, delegates to detection job audio resolver or embedding set AudioFile path; returns PNG
- [ ] `GET /vocalization/training-datasets/{id}/audio-slice?row_index=N` — same delegation for audio; returns WAV
- [ ] `POST /vocalization/training-datasets/{id}/extend` — body: {detection_job_ids, embedding_set_ids}; calls extend_training_dataset; returns updated dataset
- [ ] `POST /vocalization/training-datasets/{id}/labels` — body: {row_index, label}; creates training_dataset_label; enforces "(Negative)" mutual exclusivity
- [ ] `DELETE /vocalization/training-datasets/{id}/labels/{label_id}` — removes label
- [ ] `POST /vocalization/training-jobs` updated to accept optional training_dataset_id for retrain (mutually exclusive with source_config)
- [ ] `GET /vocalization/models/{id}` response includes training_dataset_id

**Tests needed:**
- /rows endpoint filters correctly by type and positive/negative group
- /rows returns labels from training_dataset_labels (not vocalization_labels)
- /spectrogram delegates to correct audio source for detection job vs embedding set rows
- /extend appends rows and returns updated total_rows
- Label create/delete work and enforce mutual exclusivity
- Training job creation with source_config creates dataset; with training_dataset_id reuses

---

### Task 5: Frontend Types, API Client, and Query Hooks

**Files:**
- Modify: `frontend/src/api/types.ts`
- Modify: `frontend/src/api/client.ts`
- Modify: `frontend/src/hooks/queries/useVocalization.ts`

**Acceptance criteria:**
- [ ] TypeScript types added: TrainingDataset, TrainingDatasetRow, TrainingDatasetLabel, TrainingDatasetExtendRequest
- [ ] API client functions: fetchTrainingDatasets, fetchTrainingDataset, fetchTrainingDatasetRows, trainingDatasetSpectrogramUrl, trainingDatasetAudioSliceUrl, extendTrainingDataset, createTrainingDatasetLabel, deleteTrainingDatasetLabel
- [ ] Query hooks: useTrainingDataset, useTrainingDatasetRows (with type/group/pagination params), useExtendTrainingDataset (mutation), useCreateTrainingDatasetLabel (mutation), useDeleteTrainingDatasetLabel (mutation)
- [ ] Training job creation hook updated to support training_dataset_id parameter
- [ ] VocClassifierModel type updated with training_dataset_id field

**Tests needed:**
- TypeScript compiles without errors (`npx tsc --noEmit`)

---

### Task 6: Frontend Training Data View

**Files:**
- Create: `frontend/src/components/vocalization/TrainingDataView.tsx`
- Modify: `frontend/src/App.tsx` (add route)

**Acceptance criteria:**
- [ ] New route at `/app/vocalization/training-data` accessible from Vocalization nav
- [ ] Model selector dropdown loads trained models with training_dataset_id; selecting one loads the dataset
- [ ] Type filter bar shows vocabulary types as horizontal pills plus "(All)"; selecting a type enables positive/negative toggle
- [ ] Positive/Negative toggle visible when type selected; defaults to "Positive"
- [ ] Paginated row list (50 per page) with each row showing:
  - Large inline spectrogram (~400px wide) loaded from training dataset spectrogram endpoint
  - Label tags with colored pills (saved/pending-add/pending-remove states)
  - Source badge (detection job or embedding set name)
  - Confidence value when available
  - Click spectrogram opens full-width dialog with audio playback
- [ ] Label editing: click to remove (strikethrough), "+" to add from vocabulary, "(Negative)" mutual exclusivity enforced
- [ ] Pending changes accumulated locally in Maps (same pattern as LabelingWorkspace)
- [ ] Sticky Save/Cancel bar appears when dirty, shows pending change count
- [ ] Save batches all label creates/deletes to API, clears pending state on success
- [ ] Cancel discards pending state without API calls
- [ ] Retrain button in footer, disabled when pending edits exist; creates training job with training_dataset_id

**Tests needed:**
- Playwright test: navigate to training data view, select model, verify rows load
- Playwright test: filter by type and toggle positive/negative
- Playwright test: add/remove label, verify save/cancel flow

---

### Task 7: Retrain Integration on Labeling Page

**Files:**
- Modify: `frontend/src/components/vocalization/RetrainFooter.tsx`
- Modify: `frontend/src/components/vocalization/VocalizationLabelingTab.tsx`

**Acceptance criteria:**
- [ ] RetrainFooter updated: when retraining a model that has a training_dataset_id, the flow is (1) extend the dataset with the current detection job source, (2) create training job with training_dataset_id
- [ ] If the model has no training_dataset_id (legacy model), fall back to current behavior (create training job with source_config) — or show a message that retraining requires a new model
- [ ] User sees progress feedback during extend + retrain sequence
- [ ] After successful retrain, UI reflects the new model

**Tests needed:**
- Playwright test: retrain from Labeling page extends dataset and creates training job

---

### Task 8: Documentation Updates

**Files:**
- Modify: `CLAUDE.md` (§9.1 capabilities, §9.2 schema, §8.3 data model reference)
- Modify: `docs/reference/data-model.md`
- Modify: `docs/reference/storage-layout.md`
- Modify: `docs/reference/frontend.md`

**Acceptance criteria:**
- [ ] CLAUDE.md §9.1 lists training dataset review/editing capability
- [ ] CLAUDE.md §9.2 lists training_datasets and training_dataset_labels tables, latest migration updated to 032
- [ ] Data model reference includes new tables
- [ ] Storage layout includes `training_datasets/{id}/embeddings.parquet`
- [ ] Frontend reference includes new route and component

**Tests needed:**
- None (documentation only)

---

### Verification

Run in order after all tasks:
1. `uv run ruff format --check src/humpback/models/training_dataset.py src/humpback/services/training_dataset.py src/humpback/workers/vocalization_worker.py src/humpback/api/routers/vocalization.py src/humpback/schemas/vocalization.py src/humpback/storage.py alembic/versions/032_training_datasets.py`
2. `uv run ruff check src/humpback/models/training_dataset.py src/humpback/services/training_dataset.py src/humpback/workers/vocalization_worker.py src/humpback/api/routers/vocalization.py src/humpback/schemas/vocalization.py src/humpback/storage.py alembic/versions/032_training_datasets.py`
3. `uv run pyright src/humpback/models/training_dataset.py src/humpback/services/training_dataset.py src/humpback/workers/vocalization_worker.py src/humpback/api/routers/vocalization.py src/humpback/schemas/vocalization.py src/humpback/storage.py`
4. `uv run pytest tests/`
5. `cd frontend && npx tsc --noEmit`
6. `cd frontend && npx playwright test`
