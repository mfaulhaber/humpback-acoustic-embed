# Call Parsing — Pass 3: Event Classifier Implementation Plan

**Goal:** Train a PyTorch event-level CNN classifier on variable-length event crops, extend the vocalization model infrastructure with a `pytorch_event_cnn` family, and produce `typed_events.parquet` from Pass 2's `events.parquet`.
**Spec:** [docs/specs/2026-04-11-call-parsing-pass3-event-classifier-design.md](../specs/2026-04-11-call-parsing-pass3-event-classifier-design.md)

---

### Task 1: Event Classifier CNN Model

**Files:**
- Create: `src/humpback/call_parsing/event_classifier/__init__.py`
- Create: `src/humpback/call_parsing/event_classifier/model.py`
- Create: `tests/unit/test_event_classifier_model.py`

**Acceptance criteria:**
- [ ] `EventClassifierCNN(n_types, channels=(32,64,128,256))` module with 4 Conv2d/BN/ReLU blocks
- [ ] `MaxPool2d((2,1))` after each block — pools frequency only, preserves time dimension
- [ ] `AdaptiveAvgPool2d((1,1))` collapses variable (freq, time) to fixed (256,) vector
- [ ] `Linear(256, n_types)` head outputs raw logits (no sigmoid)
- [ ] Forward pass accepts `(B, 1, 64, T)` for any T >= 1, returns `(B, n_types)`

**Tests needed:**
- Forward pass shape correctness for various T values (6, 15, 50, 156 frames)
- Parameter count is in expected range (~200–500k for typical n_types)
- Deterministic output with fixed seed
- Gradient flows through all parameters

---

### Task 2: Event Crop Dataset and DataLoader

**Files:**
- Create: `src/humpback/call_parsing/event_classifier/dataset.py`
- Create: `tests/unit/test_event_classifier_dataset.py`

**Acceptance criteria:**
- [ ] `EventCropDataset` takes a list of training samples (event bounds + type label + audio source ref) and an `AudioLoader` callable
- [ ] `__getitem__` crops audio at `[start_sec, end_sec]`, runs `extract_logmel` + `normalize_per_region_zscore` from `segmentation.features`, returns `(features, label_vector)` where features is `(1, 64, T)` and label_vector is multi-hot `(n_types,)`
- [ ] `collate_fn` pads spectrograms to max-T within batch, returns `(features_batch, labels_batch)` tensors
- [ ] `AudioLoader = Callable[[Any], np.ndarray]` type alias matches Pass 2 pattern
- [ ] Events at audio boundaries are clipped to available audio (no crash on edge cases)

**Tests needed:**
- Dataset returns correct shapes for a synthetic audio sample
- collate_fn correctly pads variable-length spectrograms and preserves label vectors
- Edge case: event at start/end of audio file

---

### Task 3: Training Driver

**Files:**
- Create: `src/humpback/call_parsing/event_classifier/trainer.py`
- Create: `tests/unit/test_event_classifier_trainer.py`

**Acceptance criteria:**
- [ ] `train_event_classifier(samples, feature_config, audio_loader, config, checkpoint_path)` orchestrates full training
- [ ] `EventClassifierTrainingConfig` dataclass with: `epochs`, `batch_size`, `learning_rate`, `weight_decay`, `early_stopping_patience`, `grad_clip`, `seed`, `min_examples_per_type` (default 10)
- [ ] Per-audio-source train/val split via `split_by_audio_source` (import from segmentation trainer)
- [ ] `BCEWithLogitsLoss` with per-type `pos_weight` auto-computed from label frequency
- [ ] Types with fewer than `min_examples_per_type` examples excluded from training; vocabulary filtered accordingly
- [ ] Reuses `ml.training_loop.fit()` with early-stopping and val-F1 callbacks
- [ ] After training, sweeps per-type thresholds on val set to maximize per-type F1
- [ ] Saves checkpoint via `ml.checkpointing.save_checkpoint`, writes `config.json`, `thresholds.json`, `metrics.json` to model directory
- [ ] Returns an `EventClassifierTrainingResult` with train/val losses, per-type metrics, threshold values, sample counts

**Tests needed:**
- End-to-end training on a tiny synthetic dataset (3–5 samples, 2 types, 2 epochs) produces checkpoint files and result object
- Types below `min_examples_per_type` are excluded
- Per-type pos_weight computation is correct for known label distributions
- Threshold optimization returns values in [0, 1]

---

### Task 4: Inference Module

**Files:**
- Create: `src/humpback/call_parsing/event_classifier/inference.py`
- Create: `tests/unit/test_event_classifier_inference.py`

**Acceptance criteria:**
- [ ] `load_event_classifier(model_dir)` loads `config.json` → reconstructs `EventClassifierCNN` → loads `model.pt` state dict → loads `thresholds.json` → returns `(model, vocabulary, thresholds, feature_config)`
- [ ] `classify_events(model, events, audio_loader, feature_config, vocabulary, thresholds)` takes a list of `Event` dataclasses, crops audio, extracts features, runs batch inference, applies thresholds, returns `list[TypedEvent]`
- [ ] Batch inference: collates variable-length crops, runs forward pass on device, applies sigmoid, compares to thresholds
- [ ] Each event produces one `TypedEvent` row per vocabulary type (with `score` and `above_threshold`)

**Tests needed:**
- Round-trip: save a model checkpoint, load it, run inference on synthetic events, verify TypedEvent output structure
- Threshold application: scores above threshold → `above_threshold=True`
- Correct event_id / start_sec / end_sec propagation from input Event to output TypedEvent

---

### Task 5: Vocalization Training Worker Dispatch

**Files:**
- Modify: `src/humpback/workers/vocalization_worker.py`
- Modify: `tests/unit/test_event_classifier_trainer.py` (or existing vocalization worker tests)

**Acceptance criteria:**
- [ ] `run_vocalization_training_job` checks `job.model_family` at the top
- [ ] `model_family='sklearn_perch_embedding'` (or `None` for legacy) → existing sklearn training path (unchanged)
- [ ] `model_family='pytorch_event_cnn'` → calls new event classifier trainer, creates `VocalizationClassifierModel` row with `model_family='pytorch_event_cnn'` and `input_mode='segmented_event'`
- [ ] The pytorch path mirrors the sklearn path's error handling: catches exceptions, sets job `status='failed'` with `error_message`
- [ ] Result summary JSON stored on the training job row

**Tests needed:**
- Dispatch routes to correct trainer based on model_family
- Model row created with correct model_family and input_mode values
- Error handling sets failed status on exception

---

### Task 6: Event Classification Worker

**Files:**
- Modify: `src/humpback/workers/event_classification_worker.py`
- Modify: `tests/unit/test_call_parsing_workers.py`

**Acceptance criteria:**
- [ ] Replace Phase 0 stub with full implementation
- [ ] Claims queued `EventClassificationJob`, sets status to `running`
- [ ] Validates upstream `EventSegmentationJob` is `complete`
- [ ] Loads `VocalizationClassifierModel` by `vocalization_model_id`, verifies `model_family='pytorch_event_cnn'`
- [ ] Reads `events.parquet` from upstream segmentation job directory
- [ ] Resolves audio source transitively from Pass 1 job (through Pass 2's `region_detection_job_id`)
- [ ] Calls `classify_events` from the inference module
- [ ] Writes `typed_events.parquet` atomically via `write_typed_events` to `classification_job_dir`
- [ ] Updates job: `status='complete'`, `typed_event_count=len(typed_events)`
- [ ] Crash safety: on exception, deletes partial parquet + `.tmp` files, sets `status='failed'` with `error_message`

**Tests needed:**
- Worker processes a job end-to-end with mocked model and synthetic events
- Crash safety: exception mid-processing leaves no partial artifacts, job is `failed`
- Validates upstream job status (rejects non-complete upstream)

---

### Task 7: Unstub API Endpoints and Request Schema

**Files:**
- Modify: `src/humpback/api/routers/call_parsing.py`
- Modify: `src/humpback/schemas/call_parsing.py`
- Modify: `src/humpback/services/call_parsing.py`
- Modify: `tests/integration/test_call_parsing_router.py`

**Acceptance criteria:**
- [ ] `CreateEventClassificationJobRequest` Pydantic schema with `event_segmentation_job_id` (required), `vocalization_model_id` (required), `parent_run_id` (optional), `config` (optional dict)
- [ ] `POST /classification-jobs` validates:
  - `vocalization_model_id` exists (404) and has `model_family='pytorch_event_cnn'` + `input_mode='segmented_event'` (422)
  - `event_segmentation_job_id` exists (404) and is `complete` (409)
  - `parent_run_id` exists if provided (404)
- [ ] Creates queued `EventClassificationJob` row, returns `EventClassificationJobSummary`
- [ ] `GET /classification-jobs/{id}/typed-events` reads `typed_events.parquet` via `read_typed_events`, returns sorted JSON rows
  - 409 while job is not `complete`
  - 404 if parquet file is missing
- [ ] Service layer methods for create and typed-events retrieval

**Tests needed:**
- POST with valid inputs creates job and returns 200
- POST with wrong model family returns 422
- POST with non-existent model returns 404
- POST with non-complete upstream returns 409
- GET typed-events on complete job returns sorted rows
- GET typed-events on non-complete job returns 409

---

### Task 8: Bootstrap Script

**Files:**
- Create: `scripts/bootstrap_event_classifier_dataset.py`
- Create: `tests/unit/test_bootstrap_event_classifier.py`

**Acceptance criteria:**
- [ ] Analogous structure to `scripts/bootstrap_segmentation_dataset.py`
- [ ] Accepts detection job IDs as input, filters to single-label vocalization-labeled windows only (excludes multi-label and `(Negative)`)
- [ ] For each qualifying window: resolves audio source, determines window time range, runs Pass 2 segmentation model to find events within the window
- [ ] Transfers the window's single vocalization type label to all events whose bounds fall within the window's time range
- [ ] Produces training samples compatible with `EventCropDataset` (event bounds + type label + audio source ref)
- [ ] Per-audio-source aware: preserves source identity on each sample for downstream split
- [ ] Dry-run mode for inspection before commit
- [ ] Idempotent: re-running does not create duplicate samples

**Tests needed:**
- Single-label window produces correct labeled events
- Multi-label window is excluded
- `(Negative)` window is excluded
- Events outside window time range are not labeled

---

### Task 9: Smoke Test

**Files:**
- Create: `tests/integration/test_event_classifier_smoke.py`

**Acceptance criteria:**
- [ ] Trains a tiny event classifier on synthetic crops (3+ samples, 2+ types, 2 epochs)
- [ ] Runs inference on a fixture event through the inference module
- [ ] Confirms `typed_events.parquet` is written with correct schema and sane per-type scores
- [ ] Exercises the full worker path: create job → run worker → verify complete status + typed_event_count

**Tests needed:**
- End-to-end pipeline: synthetic audio → events → train model → classify → typed_events.parquet
- Typed events have expected columns and value ranges

---

### Task 10: Documentation Updates

**Files:**
- Modify: `CLAUDE.md`
- Modify: `DECISIONS.md`

**Acceptance criteria:**
- [ ] CLAUDE.md §9.1: add Pass 3 event classifier to implemented capabilities
- [ ] CLAUDE.md §8.7: add behavioral constraints for Pass 3 (model family validation, bootstrap single-label filter, frequency-only pooling rationale)
- [ ] CLAUDE.md §8.9: update Pass 3 endpoints from 501 stubs to functional, document training workflow
- [ ] CLAUDE.md §9.2: update latest migration number if a migration was added
- [ ] DECISIONS.md: ADR for event classifier architecture (frequency-only pooling, variable-length crops, model family coexistence, bootstrap data strategy)

---

### Verification

Run in order after all tasks:
1. `uv run ruff format --check src/humpback/call_parsing/event_classifier/ src/humpback/workers/event_classification_worker.py src/humpback/workers/vocalization_worker.py src/humpback/api/routers/call_parsing.py src/humpback/schemas/call_parsing.py src/humpback/services/call_parsing.py scripts/bootstrap_event_classifier_dataset.py`
2. `uv run ruff check src/humpback/call_parsing/event_classifier/ src/humpback/workers/event_classification_worker.py src/humpback/workers/vocalization_worker.py src/humpback/api/routers/call_parsing.py src/humpback/schemas/call_parsing.py src/humpback/services/call_parsing.py scripts/bootstrap_event_classifier_dataset.py`
3. `uv run pyright src/humpback/call_parsing/event_classifier/ src/humpback/workers/event_classification_worker.py src/humpback/workers/vocalization_worker.py src/humpback/api/routers/call_parsing.py src/humpback/schemas/call_parsing.py src/humpback/services/call_parsing.py scripts/bootstrap_event_classifier_dataset.py`
4. `uv run pytest tests/`
