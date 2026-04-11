# Call Parsing — Pass 3: Event Classifier (Skeletal Plan)

**Status:** Skeletal — requires a Pass 3 brainstorm cycle before execution.
**Goal:** Train a PyTorch event-level classifier (BirdNET-like CNN) on variable-length event crops, extend the existing vocalization model infrastructure with a `pytorch_event_cnn` family, and produce `typed_events.parquet` from `events.parquet`.
**Architecture inherits from:** [Phase 0 spec](../specs/2026-04-11-call-parsing-pipeline-phase0-design.md)
**Pass 3 design spec (to be written):** `docs/specs/YYYY-MM-DD-call-parsing-pass3-design.md`
**Depends on:** Pass 2 complete (need `events.parquet` as input)

---

## Inherited from Phase 0 (do NOT re-derive)

- Table `event_classification_jobs` with standard queue columns, `parent_run_id` FK, `event_segmentation_job_id` upstream FK, `vocalization_model_id` FK, `typed_event_count`
- Worker shell at `src/humpback/workers/event_classification_worker.py`
- `vocalization_models` table extended with `model_family` and `input_mode` columns (defaults: `sklearn_perch_embedding` / `detection_row`)
- `vocalization_training_jobs` table extended with the same two columns
- `call_parsing/types.py` defines `TypedEvent` dataclass with parquet schema
- `call_parsing/storage.py` exposes `write_typed_events` / `read_typed_events`
- `src/humpback/ml/` shared PyTorch harness
- Existing vocabulary (`vocalization_types` table) carries over unchanged — **same types apply to event-level classification as to 5s-window classification**
- Reserved module namespace `src/humpback/call_parsing/event_classifier/` (not yet created)
- Stub API endpoints at `/call-parsing/classification-jobs`

## Brainstorm checklist — Pass 3 TBDs

### Model architecture
- [ ] BirdNET-like CNN structure (EfficientNet-lite backbone? custom convnet?)
- [ ] Number of layers, channel counts, pooling strategy
- [ ] Multi-label sigmoid head sized to `len(vocalization_types)` at training time
- [ ] How to handle vocabulary changes after training (retrain vs partial head update)

### Input features
- [ ] Log-mel vs PCEN (may or may not match Pass 2 — discuss)
- [ ] STFT parameters and n_mels
- [ ] Whether to share a feature pipeline module with Pass 2

### Event crop strategy
- [ ] Fixed-length crop (pad/truncate to e.g. 3s) or variable-length (architecture handles variable time dimension)
- [ ] Context padding (extend event bounds by N seconds of surrounding audio)
- [ ] Multi-crop aggregation for long events (split + pool predictions)

### Training data
- [ ] Where do labeled event-level examples come from? Options:
      - Historical detection-job rows promoted to events via Pass 2 inference
      - Manually labeled event crops in a new labeling UI flow
      - Both
- [ ] Training/val/test split policy
- [ ] Class imbalance handling (per-type sampling, weighted loss)
- [ ] Minimum examples per type before a type enters training (mirror existing `min_examples_per_type` convention)

### Training workflow
- [ ] Does `vocalization_training_jobs` handle both sklearn and pytorch families in one worker, or is there a new worker?
- [ ] Per-type F1 threshold optimization (carry existing pattern from ADR-042)
- [ ] Training eval metrics reporting

### Coexistence with existing sklearn family
- [ ] What happens to vocalization label browsing / inference result browsing UI when two families coexist? Spec in §8.8 path.
- [ ] Default `model_family` in training job creation API — keep `sklearn_perch_embedding` or switch to `pytorch_event_cnn`
- [ ] Whether to deprecate the sklearn family or keep as a permanent alternative

### Operational
- [ ] Checkpoint layout under `storage_root/vocalization_models/<model_id>/` for the pytorch family — file names, config.json schema
- [ ] Inference batching strategy

## Tasks (skeletal — expand after brainstorm)

### Task 1: Optional migration for Pass 3 config fields
New Alembic migration if Pass 3 needs config columns beyond what Phase 0 provided on `event_classification_jobs` or `vocalization_training_jobs`.

### Task 2: Dataset + DataLoader for event crops
**Files:**
- Create: `src/humpback/call_parsing/event_classifier/dataset.py`
- Tests

### Task 3: Model architecture
**Files:**
- Create: `src/humpback/call_parsing/event_classifier/model.py`
- Tests (forward pass shapes, param count, deterministic output)

### Task 4: Training driver
**Files:**
- Create: `src/humpback/call_parsing/event_classifier/trainer.py`
- Tests

Reuses `ml.training_loop.fit()`.

### Task 5: Inference module
**Files:**
- Create: `src/humpback/call_parsing/event_classifier/inference.py`
- Tests

Loads a pytorch event-cnn checkpoint, runs inference on event crops, emits per-type probability rows.

### Task 6: Vocalization training worker updates
**Files:**
- Modify: existing vocalization training worker (or create a dispatcher) so `model_family='pytorch_event_cnn'` jobs route to the new trainer

### Task 7: Event classification inference worker
**Files:**
- Modify: `src/humpback/workers/event_classification_worker.py`
- Create: integration test

### Task 8: Unstub Pass 3 API endpoints
- `POST /call-parsing/classification-jobs` functional
- Training API (either new endpoint or extension of `/vocalization/training-jobs` to accept pytorch family)
- Artifact endpoint `GET /{id}/typed-events` functional

### Task 9: Smoke test
Train a tiny event classifier on synthetic crops; run inference on a fixture event; confirm typed events land in `typed_events.parquet` with sane per-type scores.

### Task 10: Documentation updates
- CLAUDE.md §9.1: Pass 3 capability
- CLAUDE.md §8.8: Pass 3 API surface + how the two model families coexist
- DECISIONS.md: ADR for architecture and the coexistence semantics

## Verification

1. `uv run ruff format --check` on modified files
2. `uv run ruff check` on modified files
3. `uv run pyright` on modified files
4. `uv run alembic upgrade head`
5. `uv run pytest tests/`
6. Manual: train a toy event classifier, run inference through the full chain (Pass 1 → 2 → 3), confirm typed events
