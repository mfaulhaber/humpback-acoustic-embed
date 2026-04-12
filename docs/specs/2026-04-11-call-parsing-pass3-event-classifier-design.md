# Call Parsing — Pass 3: Event Classifier Design

**Date:** 2026-04-11
**Status:** Approved
**Depends on:** Pass 2 event segmentation (complete), Phase 0 scaffold (complete)
**Architecture inherits from:** [Phase 0 spec](2026-04-11-call-parsing-pipeline-phase0-design.md)

---

## 1. Purpose

Pass 3 classifies individual whale vocalization events (produced by Pass 2) into
vocalization types from the existing `vocalization_types` vocabulary. It trains a
PyTorch CNN on variable-length event crops and produces `typed_events.parquet` as
its output artifact.

Pass 3 extends the existing vocalization model infrastructure with a new
`pytorch_event_cnn` model family, coexisting alongside the current
`sklearn_perch_embedding` family.

---

## 2. Data Flow

```
events.parquet (Pass 2)
  + original audio (inherited from Pass 1)
  → crop audio at each event's [start_sec, end_sec]
  → extract log-mel spectrogram (reuse Pass 2 feature pipeline)
  → CNN inference per crop
  → per-type sigmoid scores + threshold application
  → typed_events.parquet
```

**Input:** `events.parquet` from a completed Pass 2 job. Each row has `event_id`,
`region_id`, `start_sec`, `end_sec`, `center_sec`, `segmentation_confidence`. The
audio source is resolved transitively from the upstream Pass 1 job's
`audio_file_id` or `hydrophone_id`.

**Output:** `typed_events.parquet` using the existing `TypedEvent` schema — one row
per (event, type) pair: `event_id`, `start_sec`, `end_sec`, `type_name`, `score`,
`above_threshold`.

---

## 3. Event Crop Extraction

Each event is cropped at its exact boundaries (`start_sec` to `end_sec`). No
context padding, no fixed-length target. The model sees only the vocalization —
Pass 2 already found the boundaries precisely.

- Events at audio file boundaries are clipped to available audio.
- Humpback call durations are typically 1–3s (max ~5s). Pass 2's decoder
  `min_event_sec=0.2` sets the lower bound.
- The variable-length architecture (Section 4) handles the full 0.2–5s range.

---

## 4. Feature Extraction

Reuse Pass 2's feature pipeline verbatim — same module imports, same config object:

- `extract_logmel()` from `call_parsing.segmentation.features`
- `normalize_per_region_zscore()` from the same module
- `SegmentationFeatureConfig` defaults: 16kHz sample rate, n_fft=512, hop=512
  (32ms frames), 64 mel bins, fmin=20Hz, fmax=4000Hz

Output per event: `(64, T)` log-mel spectrogram where T varies with event
duration (~6 frames at 0.2s, ~156 frames at 5s).

---

## 5. Model Architecture

Custom small convnet (~200–500k parameters):

```
Input: (B, 1, 64, T)  — variable T

Block 1: Conv2d(1,  32,  3×3, pad=1) → BatchNorm2d → ReLU → MaxPool2d((2,1))
Block 2: Conv2d(32, 64,  3×3, pad=1) → BatchNorm2d → ReLU → MaxPool2d((2,1))
Block 3: Conv2d(64, 128, 3×3, pad=1) → BatchNorm2d → ReLU → MaxPool2d((2,1))
Block 4: Conv2d(128,256, 3×3, pad=1) → BatchNorm2d → ReLU → MaxPool2d((2,1))

After block 4: (B, 256, 4, T)  — frequency halved 4× (64→4), time preserved

AdaptiveAvgPool2d((1,1)) → (B, 256)
Linear(256, n_types) → (B, n_types)  — raw logits, sigmoid applied externally
```

**Key design choice:** `MaxPool2d((2,1))` pools only in the frequency axis, never
in time. This ensures the time dimension never collapses for short events (a 0.2s
event has ~6 time frames that survive all 4 blocks). `AdaptiveAvgPool2d` handles
the final collapse to a fixed-size vector regardless of T.

The multi-label sigmoid head is sized to `len(vocalization_types)` at training
time. Vocabulary changes require retraining.

---

## 6. Training

### 6.1 Bootstrap Training Data

Training data is bootstrapped from existing vocalization-labeled detection jobs.
Only detection windows with **exactly one** vocalization type label (excluding
`(Negative)`) are used — multi-label windows are ambiguous at the event level and
are excluded.

Bootstrap process:
1. Collect single-label vocalization-labeled detection windows
2. Run Pass 2 segmentation on those audio regions to get event bounds
3. Transfer the window's vocalization type label to all events whose bounds fall
   within the window's time range
4. Assemble into a training dataset

A bootstrap script (analogous to `scripts/bootstrap_segmentation_dataset.py`)
implements this pipeline.

Future work: extend the Vocalization Labeling Workflow and UI for direct
event-level annotation in a human-in-the-loop improvement cycle, matching the
pattern of the existing vocalization labeling workspace.

### 6.2 Train/Val Split

Per-audio-source split via `split_by_audio_source` — no audio source appears in
both train and val. Same rationale as Pass 2: prevents background noise signature
leakage between splits.

### 6.3 Loss Function

`BCEWithLogitsLoss` with per-type `pos_weight` auto-computed from label frequency
in the training set. Handles class imbalance without manual weight tuning.

### 6.4 Minimum Examples Per Type

Types with fewer than `min_examples_per_type` (default 10) examples are excluded
from training. Matches the existing vocalization training convention.

### 6.5 Training Loop

Reuses `ml.training_loop.fit()` with:
- Early-stopping callback (patience-based, monitoring val loss)
- Val F1 callback (per-epoch macro F1 on validation set)
- Gradient clipping

### 6.6 Per-Type Threshold Optimization

After training, sweep per-type classification thresholds on the validation set to
maximize per-type F1. Store optimized thresholds in `thresholds.json`. Applied at
inference time to populate `above_threshold` on each `TypedEvent` row. Same
pattern as the existing sklearn vocalization models (ADR-042).

### 6.7 Variable-Length Batching

DataLoader uses a custom `collate_fn` that pads spectrograms to the max time
dimension within each batch. Since `AdaptiveAvgPool2d` averages over the full
spatial extent, zero-padded time frames (post z-score normalization) contribute
near-zero values that dilute slightly but do not corrupt the pooled
representation. For small batches with similar-duration events (the common case),
the padding is minimal.

---

## 7. Inference & Worker

### 7.1 Event Classification Worker

The existing stub at `workers/event_classification_worker.py` is replaced with:

1. Claim a queued `EventClassificationJob`
2. Validate upstream `EventSegmentationJob` is complete
3. Load the `VocalizationClassifierModel` (must be `pytorch_event_cnn` family)
4. Read `events.parquet` from the upstream job
5. Resolve audio source from the Pass 1 job (transitive through Pass 2)
6. For each event: crop audio at `[start_sec, end_sec]`, extract log-mel features
7. Batch inference through the CNN
8. Apply per-type thresholds
9. Write `typed_events.parquet` atomically
10. Update job row: `status='complete'`, `typed_event_count`

**Crash safety:** On exception, delete partial `typed_events.parquet` and any
`.tmp` sidecars, set status to `failed` with `error_message`. No partial-result
resume — restart from scratch on retry.

### 7.2 Vocalization Training Worker Dispatch

The existing `vocalization_worker.py` gains a dispatcher at the top of its
training path:

- `model_family='sklearn_perch_embedding'` → existing sklearn trainer (unchanged)
- `model_family='pytorch_event_cnn'` → new event classifier trainer

One worker, one claim loop, branching on `model_family`. The worker priority list
is unchanged.

---

## 8. Checkpoint Layout

```
storage_root/vocalization_models/<model_id>/
├── config.json       # feature config, vocabulary snapshot, architecture params,
│                     #   training params, min_examples_per_type
├── model.pt          # PyTorch state_dict (the CNN weights)
├── thresholds.json   # {type_name: float} per-type optimized thresholds
└── metrics.json      # {type_name: {precision, recall, f1}} from validation
```

The sklearn family stores per-type `.joblib` pipeline files. The pytorch family
stores a single `model.pt` since it is one multi-head model. Both families share
the same parent directory pattern under `vocalization_models/`.

---

## 9. API & Validation

### 9.1 Unstubbed Endpoints

**`POST /call-parsing/classification-jobs`** — create and queue a Pass 3 job:
- Validates `vocalization_model_id` exists and has
  `model_family='pytorch_event_cnn'` + `input_mode='segmented_event'` (422 if
  wrong family)
- Validates `event_segmentation_job_id` exists (404) and is `complete` (409)
- Stores config in `config_json`

**`GET /call-parsing/classification-jobs/{id}/typed-events`** — returns
`typed_events.parquet` as sorted JSON rows:
- 409 while job is not `complete`
- 404 if parquet file is missing

### 9.2 Coexistence with sklearn Family

Existing vocalization inference endpoints (detection-job rescoring, embedding-set
inference) continue to only accept `sklearn_perch_embedding` models. The
`pytorch_event_cnn` family is only usable through the call parsing classification
job flow. No cross-contamination, no UI changes in this spec.

---

## 10. Inherited Infrastructure (from Phase 0)

The following are already implemented and should not be re-derived:

- `event_classification_jobs` table with standard queue columns, `parent_run_id`
  FK, `event_segmentation_job_id` FK, `vocalization_model_id` FK,
  `typed_event_count`
- `vocalization_models` and `vocalization_training_jobs` tables with
  `model_family` and `input_mode` columns
- `TypedEvent` dataclass and PyArrow schema in `call_parsing/types.py`
- `write_typed_events` / `read_typed_events` in `call_parsing/storage.py`
- `classification_job_dir()` path helper
- Worker shell at `workers/event_classification_worker.py`
- Shared PyTorch harness at `ml/` (training_loop, checkpointing, device selection)
- Stub API endpoints at `/call-parsing/classification-jobs`
- `EventClassificationJobSummary` Pydantic schema

---

## 11. Out of Scope

- Event-level labeling UI (future spec — extend Vocalization Labeling Workflow)
- Vocabulary management changes (existing `vocalization_types` table is reused)
- Deprecation of the sklearn family (both families coexist indefinitely)
- Pass 4 sequence export (separate spec)
- Pretrained/transfer-learning backbones (start with training from scratch)
