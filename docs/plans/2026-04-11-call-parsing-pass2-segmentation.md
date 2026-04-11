# Call Parsing — Pass 2: Event Segmentation (Skeletal Plan)

**Status:** Skeletal — requires a Pass 2 brainstorm cycle before execution. This is the heaviest Pass by design: it introduces the project's first PyTorch training workload.
**Goal:** Train a learned framewise segmentation model (PyTorch CRNN or transformer), run inference to produce `events.parquet` from `regions.parquet`, and expose the training and inference surfaces via API.
**Architecture inherits from:** [Phase 0 spec](../specs/2026-04-11-call-parsing-pipeline-phase0-design.md)
**Pass 2 design spec (to be written):** `docs/specs/YYYY-MM-DD-call-parsing-pass2-design.md` — produced by the Pass 2 brainstorm; this plan gets elaborated afterward.
**Depends on:** Pass 1 complete (need `regions.parquet` as input)

---

## Inherited from Phase 0 (do NOT re-derive)

- Table `event_segmentation_jobs` with standard queue columns, `parent_run_id` FK, `region_detection_job_id` upstream FK, `segmentation_model_id` FK, `event_count`
- Worker shell at `src/humpback/workers/event_segmentation_worker.py`
- Table `segmentation_models` with `id`, `name`, `model_family`, `model_path`, `config_json`, `training_job_id`
- `call_parsing/types.py` defines `Event` dataclass with parquet schema
- `call_parsing/storage.py` exposes `write_events` / `read_events`
- `src/humpback/ml/` shared PyTorch harness (`device.select_device`, `training_loop.fit`, `checkpointing.save_checkpoint` / `load_checkpoint`)
- Reserved module namespace `src/humpback/call_parsing/segmentation/` (not yet created)
- PyTorch bundled into all TF extras
- Stub API endpoints at `/call-parsing/segmentation-jobs` (all return 501 except list/detail/delete)

## Brainstorm checklist — Pass 2 TBDs

The Pass 2 brainstorm must settle these. This list is large because Pass 2 introduces the first learned model in the pipeline:

### Model architecture
- [ ] CRNN vs transformer (design spec recommends CRNN baseline; transformer as alternative)
- [ ] Backbone depth, channel counts, recurrent layer width
- [ ] Framewise sigmoid head vs two-head (onset / offset) vs segmentation head
- [ ] Parameter budget target (mobile-lite vs full)

### Input features
- [ ] Log-mel vs PCEN vs log-mel-with-PCEN
- [ ] STFT parameters: frame length (20–40 ms), hop (10–20 ms), n_fft, n_mels
- [ ] Frequency band limit (humpback range, e.g. 20 Hz – 4 kHz)
- [ ] Whether to reuse `processing/features.extract_logmel_batch` or introduce a new helper
- [ ] Input normalization (per-sample, per-dataset, running)

### Training data and labels
- [ ] Label format: per-frame binary presence? per-frame onset/offset targets? bounded-box style?
- [ ] Source of onset/offset labels from the existing vocalization labeling pipeline (user has stated this is the plan but the format needs design)
- [ ] Training dataset schema for `segmentation_training_jobs` table (this table does NOT exist yet — Pass 2 creates it)
- [ ] Train/val/test split policy (per-recording? per-region?)
- [ ] Class imbalance handling (weighted BCE, focal loss, balanced sampling)

### Loss and optimization
- [ ] Loss function: framewise BCE, focal loss, or something task-specific
- [ ] Optimizer and LR schedule defaults
- [ ] Number of epochs, early stopping policy

### Event decoding
- [ ] Frame threshold (primary) and low threshold (hysteresis continuation)
- [ ] Minimum event duration
- [ ] Maximum gap for merging adjacent frames into one event
- [ ] Whether to persist `frame_probs.parquet` or just the decoded events

### Evaluation metrics
- [ ] Framewise F1, event F1, onset error, offset error — which are primary?
- [ ] Tolerance window for onset/offset error

### Operational
- [ ] Training worker type — new `segmentation_training_worker` or reuse an existing pattern
- [ ] Inference batching strategy (per-region, per-batch-of-regions)
- [ ] Checkpoint naming and model directory layout under `storage_root/segmentation_models/`

## Tasks (skeletal — expand after brainstorm)

### Task 1: Migration for `segmentation_training_jobs` table
New Alembic migration creating the training job table and adding any Pass 2 config fields to `event_segmentation_jobs` that weren't in Phase 0.

### Task 2: Input feature pipeline
**Files:**
- Create or modify: `src/humpback/call_parsing/segmentation/features.py` (or extend existing `processing/features.py`)
- Tests

### Task 3: Dataset + DataLoader
**Files:**
- Create: `src/humpback/call_parsing/segmentation/dataset.py`
- Create: `tests/unit/test_segmentation_dataset.py`

Loads audio regions + labels from storage, applies feature extraction, yields PyTorch tensors.

### Task 4: Model architecture
**Files:**
- Create: `src/humpback/call_parsing/segmentation/model.py`
- Create: `tests/unit/test_segmentation_model.py`

Implements the CRNN (or transformer) decided in the brainstorm. Tests: forward pass shapes, parameter count, deterministic output under fixed seed.

### Task 5: Training driver
**Files:**
- Create: `src/humpback/call_parsing/segmentation/trainer.py`
- Create: `tests/unit/test_segmentation_trainer.py`

Thin wrapper over `ml.training_loop.fit()` — assembles model, dataset, loss, optimizer, delegates.

### Task 6: Inference module
**Files:**
- Create: `src/humpback/call_parsing/segmentation/inference.py`
- Create: `tests/unit/test_segmentation_inference.py`

Loads a checkpoint, runs framewise inference on a region, decodes to events.

### Task 7: Segmentation training worker
**Files:**
- Create: `src/humpback/workers/segmentation_training_worker.py`
- Modify: worker loop dispatcher (add to priority order)

### Task 8: Event segmentation inference worker
**Files:**
- Modify: `src/humpback/workers/event_segmentation_worker.py`
- Create: integration test

### Task 9: Unstub Pass 2 API endpoints
- `POST /call-parsing/segmentation-jobs` functional
- `POST /call-parsing/segmentation-training-jobs` functional (new endpoint)
- Artifact endpoints functional

### Task 10: Smoke test
Train a tiny toy segmentation model on synthetic labeled regions; run inference on a fixture region; assert events are produced with reasonable temporal bounds.

### Task 11: Documentation updates
- CLAUDE.md §9.1: Pass 2 capability
- CLAUDE.md §8.8: Pass 2 API surface
- DECISIONS.md: ADR for the model architecture / label format decisions

## Verification

1. `uv run ruff format --check` on modified files
2. `uv run ruff check` on modified files
3. `uv run pyright` on modified files
4. `uv run alembic upgrade head`
5. `uv run pytest tests/`
6. Manual: train a toy model end-to-end, run inference, confirm events land in `events.parquet`
