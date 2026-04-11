# Call Parsing — Pass 2: Event Segmentation

**Date:** 2026-04-11
**Status:** Approved
**Inherits from:** [Phase 0 spec](2026-04-11-call-parsing-pipeline-phase0-design.md) (ADR-048), [Pass 1 spec](2026-04-11-call-parsing-pass1-region-detector-design.md) (ADR-049)

---

## Problem

Phase 0 shipped the Pass 2 table scaffold and worker shell (`event_segmentation_jobs`,
`segmentation_models`, `Event` dataclass + parquet schema, `write_events`/`read_events`,
`src/humpback/ml/` PyTorch harness). Pass 1 now produces `regions.parquet` — padded
whale-active regions with raw hysteresis bounds and dense per-window trace data.

Pass 2 turns each region into per-event onset/offset bounds. A PyTorch CRNN learns
framewise binary presence from curated human labels, and a hysteresis decoder converts
the frame head into `Event` rows with absolute audio timestamps. Those rows are Pass 3's
upstream input: Pass 3 crops each event's audio and runs a multi-label call-type
classifier on it.

This is the project's first learned model in a feedback-loop workflow, so the design
deliberately prioritizes **simplest-thing-that-could-work** over precision. Every
sophistication is flagged as an explicit upgrade knob with a named migration path.

Three structural constraints shape the design:

1. **No frame-level ground-truth labels exist today.** `VocalizationLabel` rows attach
   multi-label type strings to detection rows at 5 s window granularity. Pass 2 needs
   frame-level onset/offset supervision, and there is no existing UI or table that
   produces it. The design handles this with a one-shot bootstrap script and a forever
   training-dataset schema that the future UI workflow can write into without a schema
   migration.
2. **The training dataset is the permanent contract, the bootstrap script is
   temporary.** Whatever the bootstrap script writes has to be shape-compatible with
   what a future timeline-viewer event-bound editor will write. The schema is designed
   for both writers from day one; the bootstrap is just the first writer.
3. **Framewise regression is a different task from per-row multi-label classification.**
   Reusing `vocalization_training_jobs` by growing another `model_family` branch would
   tangle trainer code that wants to stay separate. Pass 2 gets its own training-job
   table, its own trainer module, and its own model registry row family.

---

## Inherited from Phase 0 / Pass 1 (NOT re-derived here)

- `event_segmentation_jobs` table with standard queue columns, `parent_run_id` FK,
  `region_detection_job_id` upstream FK, `segmentation_model_id` FK, `event_count`,
  `config_json`.
- `segmentation_models` table with `id`, `name`, `model_family`, `model_path`,
  `config_json`, `training_job_id`, `created_at`.
- Worker shell `src/humpback/workers/event_segmentation_worker.py` that claims and
  fails.
- `Event` frozen dataclass + `EVENT_SCHEMA` pyarrow schema in
  `src/humpback/call_parsing/types.py`.
- `write_events` / `read_events` atomic parquet helpers in
  `src/humpback/call_parsing/storage.py`.
- Stub API endpoints at `/call-parsing/segmentation-jobs` (`POST` and
  `GET /{id}/events` return 501 in Phase 0; list / detail / delete are functional).
- `src/humpback/ml/device.select_device()`,
  `ml/training_loop.fit()`, `ml/checkpointing.save_checkpoint` / `load_checkpoint`.
- PyTorch bundled into all `tf-*` extras.
- Pass 1's `regions.parquet` format, loaded via `call_parsing.storage.read_regions`.
- `RegionDetectionJob.audio_file_id` / `hydrophone_id` / `start_timestamp` /
  `end_timestamp` source columns — Pass 2 **does not** carry its own source columns;
  it inherits the source from its upstream Pass 1 job at inference time.

---

## Scope

**Pass 2 ships:**

- Migration `044` creating three new tables: `segmentation_training_datasets`,
  `segmentation_training_samples`, `segmentation_training_jobs`.
- SQLAlchemy model classes for the three new tables.
- `src/humpback/call_parsing/segmentation/` subpackage with `features.py`,
  `dataset.py`, `model.py`, `decoder.py`, `trainer.py`, `inference.py`.
- `src/humpback/workers/segmentation_training_worker.py` — new worker that claims
  `segmentation_training_jobs` rows and runs the trainer.
- `src/humpback/workers/event_segmentation_worker.py` — unstub the Phase 0 shell.
- `src/humpback/workers/queue.py` — add both `SegmentationTrainingJob` and
  `EventSegmentationJob` to the stale-job recovery sweep, and insert
  `segmentation_training` into the claim priority order between
  `vocalization_inference` and `region_detection`.
- `src/humpback/schemas/call_parsing.py` — new Pydantic models:
  `SegmentationTrainingConfig`, `SegmentationDecoderConfig`,
  `CreateSegmentationTrainingJobRequest`, `CreateSegmentationJobRequest`, and
  response models.
- `src/humpback/services/call_parsing.py` — new service methods
  `create_segmentation_training_job` and `create_segmentation_job`.
- `src/humpback/api/routers/call_parsing.py` — unstub the two Phase 0 endpoints and
  add the five new ones (see the API section below).
- `scripts/bootstrap_segmentation_dataset.py` — one-shot CLI that reads a text file
  of detection `row_id`s, looks up the detection rows and their existing
  `VocalizationLabel`s, and writes `segmentation_training_samples` rows.
- New ADR capturing the framewise α decision, the no-PCEN-for-now decision, the
  per-audio-file split, the persistent training dataset contract, and the CRNN
  architecture choice as a single group.
- Unit, integration, API, migration, and smoke tests (see Testing section).
- Documentation updates in `CLAUDE.md`, `docs/reference/data-model.md`, and
  `DECISIONS.md`.

**Pass 2 does NOT ship:**

- Any Pass 3 or Pass 4 logic.
- Any frontend work — no timeline-viewer editor for event bounds, no dataset UI, no
  training-job status UI. The timeline-viewer extension for editing event bounds and
  the dataset/sample mutation API are a future UI branch.
- Dataset or sample mutation API endpoints. The bootstrap script writes directly via
  SQLAlchemy; the future UI branch adds its own CRUD endpoints.
- GPU-specific tuning. The design runs on CPU / MPS / CUDA via `ml/device.py` but
  doesn't specialize for any target.
- Streaming or real-time inference. Pass 2 is batch-per-region.
- Multi-label frame heads. The frame head is a single sigmoid logit per frame
  representing "call present"; per-type classification is Pass 3's job.
- Data augmentation (SpecAugment, pitch shift, gain jitter, random cropping). Simple
  first; augmentation is an upgrade knob.
- Hydrophone-path integration test. Same rationale as Pass 1 — no
  `ArchivePlaybackProvider` mock surface yet. Correctness of the source-resolution
  path is covered by the `audio_file_id` integration test; the hydrophone path falls
  out of the same code path.
- A manual end-to-end training run against real curated bootstrap data. Pass 2's
  definition of done is "code paths work on synthetic fixtures and the test suite is
  green." The first real training run is a follow-up session.

---

## Training data contract

### Source — bootstrap only

A one-shot `scripts/bootstrap_segmentation_dataset.py` CLI consumes a user-curated
list of detection `row_id`s (the stable identifier established in migration `035`).
Users browse the existing detection/vocalization labeling UI, pick detection rows
with a single clean call type, and collect their IDs into a text file. The script
reads that file, looks up each row in the detection row store, looks up the
associated `VocalizationLabel`s to confirm the row is really labeled, computes a
fixed-length audio crop centered on the row, and inserts a
`segmentation_training_samples` row.

The bootstrap script is explicitly temporary. It's not wired into the workers, not
exposed via the API, and not part of any scheduled workflow.

### Schema is forever

A future branch will extend the timeline viewer to let users add, move, and delete
event bounds directly on audio and persist those edits into
`segmentation_training_samples`. When that lands, the UI will write into exactly the
same columns the bootstrap script writes today, with no schema migration. The Pass 2
schema is designed for both writers from day one — the bootstrap is just the first
writer.

### Migration `044` — three new tables

File: `alembic/versions/044_segmentation_training_tables.py`, using
`op.batch_alter_table()` for SQLite compatibility.

```
segmentation_training_datasets
  id            (String, PK, uuid4)
  name          (String, not null)
  description   (Text, nullable)
  created_at    (DateTime, not null)
  updated_at    (DateTime, not null)
```

```
segmentation_training_samples
  id                    (String, PK, uuid4)
  training_dataset_id   (String, not null, FK segmentation_training_datasets.id)

  -- Audio source — exactly-one-of enforced in Pydantic, not DB CHECK
  audio_file_id         (String, nullable)          -- uploaded file
  hydrophone_id         (String, nullable)          -- hydrophone source
  start_timestamp       (Float,  nullable)          -- UTC epoch seconds, hydrophone only
  end_timestamp         (Float,  nullable)

  -- Sample crop, audio-relative
  crop_start_sec        (Float, not null)
  crop_end_sec          (Float, not null)

  -- Event bounds inside the crop (audio-relative, NOT crop-relative)
  -- JSON array: [{"start_sec": 52.1, "end_sec": 57.3}, ...]
  -- Empty list is valid — represents a negative-only sample.
  events_json           (Text, not null)

  -- Provenance
  source                (String, not null)          -- "bootstrap_vocalization_row", ...
  source_ref            (String, nullable)          -- e.g. detection row_id for bootstrap
  notes                 (Text, nullable)

  created_at            (DateTime, not null)
  updated_at            (DateTime, not null)
```

Indexed on `training_dataset_id` for list queries and on
`(training_dataset_id, source_ref)` to support the bootstrap script's idempotency
check.

```
segmentation_training_jobs
  id                        (String, PK, uuid4)
  status                    (String, not null, default "queued")
  training_dataset_id       (String, not null, FK segmentation_training_datasets.id)
  config_json               (Text, not null)
  segmentation_model_id     (String, nullable, FK segmentation_models.id,
                             populated on success)
  result_summary            (Text, nullable)  -- JSON blob
  error_message             (Text, nullable)
  started_at                (DateTime, nullable)
  completed_at              (DateTime, nullable)
  created_at                (DateTime, not null)
  updated_at                (DateTime, not null)
```

**Downgrade:** drops all three tables. No data preservation, no backfill — this is
new-table creation, not column modification.

### Why not extend existing tables

- **Not `training_datasets` / `training_dataset_labels`.** Those hold
  `(embedding_set_id, row_id, label_string)` pairs — 5 s-window-level, tied to Perch
  embeddings, no audio-timestamp fields. Pass 2 needs audio-timestamp-bound samples
  with multiple event bounds per sample. Forcing that onto the existing tables would
  require adding nullable columns and splitting semantics on every consumer.
- **Not `vocalization_training_jobs`.** Already carries a `model_family` /
  `input_mode` axis for Pass 3 coexistence. Pass 2 is framewise regression, a
  different task from multi-label per-row classification; the trainer code stays
  cleanly separate if the training-job table stays separate. Growing another branch
  on `vocalization_training_jobs` would make the inspection-time trainer dispatch
  three-way and cement a pattern I'd rather break.

---

## Framewise supervision target — α

At training time, the trainer takes each sample's `events_json` and constructs a
framewise target vector of shape `(T,)` where `T = crop_duration / hop_sec`. Frames
whose center time falls inside any event → `1.0`, frames outside all events → `0.0`.
Loss is masked `BCEWithLogitsLoss` with auto-computed `pos_weight`. The frame head
outputs raw logits; sigmoid is applied at eval and inference time only.

**Why α (binary framewise presence), not β (two-head onset/offset) or γ (three-level
with ignore band).**

- β (Gaussian-smeared onset/offset target peaks) is precise only when the label
  bounds are precise. Bootstrap bounds come from NMS / prominence / tiling detection
  selection — they approximate the event but rarely mark the true onset and offset
  tightly. Training the onset/offset heads on systematic label noise would waste
  capacity without buying precision.
- γ (ignore band near event edges) handles loose bounds by not penalizing frames
  within ±δ of the raw edge. It's a strictly better match for the data quality than
  α. But γ is a loss-function change only — the schema, the model, and the decoder
  are identical. It stays a zero-cost migration if α's edge calibration proves bad
  on held-out data.
- α is the simplest thing that could work, produces the cleanest gradient signal in
  the center of every event, and gives the hysteresis decoder a clean `(T,)` frame
  probability vector to consume. Start simple, upgrade when data demands it.

A second upgrade knob: weak supervision from Pass 1's `trace.parquet`. Frames where
the Pass 1 detector scored above `low_threshold` but are outside the user's labeled
event could be downweighted or ignored (they might be other calls the user didn't
label). That path also stays open with no schema change — it's purely a training-time
computation over the trace.

---

## Input features

File: `src/humpback/call_parsing/segmentation/features.py`.

Pass 2 has its own feature extractor rather than extending `processing/features.py`.
The Perch feature pipeline is in the sensitive-components list (signal integrity) and
has a different parameter set, frequency range, and normalization. Pass 2's features
should live in the Pass 2 subpackage and import low-level helpers from `librosa`
directly if needed.

```
Parameters (frozen constants in SegmentationFeatureConfig):
    sample_rate       = 16000     # matches the rest of the pipeline
    n_fft             = 2048      # 128 ms window
    hop_length        = 512       # 32 ms hop
    n_mels            = 64
    fmin              = 20        # Hz
    fmax              = 4000      # Hz
    normalize         = "per_region_zscore"
```

**Function signatures:**

```
def extract_logmel(audio, config) -> np.ndarray of shape (n_mels, T)
def normalize_per_region_zscore(logmel) -> np.ndarray of shape (n_mels, T)
def frame_index_to_audio_sec(frame_idx, config) -> float
def audio_sec_to_frame_index(time_sec, config) -> int
```

**Rationale for each parameter:**

- `n_mels=64` (down from the 128 used in the Perch pipeline). Humpback spectral
  content is narrow enough that 64 bins carry all the information a segmenter needs,
  and halving the input dimension speeds up training and inference on a small CRNN.
- `fmin=20`, `fmax=4000`. Humpback calls span roughly 20 Hz – 4 kHz, with most energy
  below 1 kHz. At a 16 kHz sample rate, Nyquist is 8 kHz — the top half is mostly
  non-humpback noise. Clamping to 20 Hz – 4 kHz removes a distraction band and
  shrinks the input further without losing signal.
- `n_fft=2048`, `hop_length=512`. 32 ms hop gives ~62 frames per 2 s event — enough
  temporal resolution for humpback onset/offset. Matches the existing pipeline so
  there's no new window-size tuning.
- `per_region_zscore`. Computed on the feature tensor of each sample (during
  training) or each region (during inference). Robust to gain variation without the
  bookkeeping of a dataset-wide statistic.

**No PCEN for Pass 2 features.** ADR-047 introduced PCEN for timeline rendering only
and explicitly left the door open for extending it to the classifier feature pipeline
if SNR demands. Pass 2 deliberately does not walk through that door yet — per-region
z-score is the simpler default, and PCEN is a known upgrade knob if the first model's
performance on noisy hydrophone data is limited by gain variation. The decision and
its escape hatch are documented in the new ADR.

---

## Model architecture

File: `src/humpback/call_parsing/segmentation/model.py`.

CRNN at ~300,000 parameters. Input is the `(n_mels=64, T)` log-mel tensor; output is
a `(T,)` logit vector (one sigmoid pre-activation per frame).

```
SegmentationCRNN:
  conv_block_1: Conv2d(1, 32, k=3, pad=1) → BatchNorm2d → ReLU
  conv_block_2: Conv2d(32, 64, k=3, pad=1) → BatchNorm2d → ReLU
  conv_block_3: Conv2d(64, 96, k=3, pad=1) → BatchNorm2d → ReLU
  conv_block_4: Conv2d(96, 128, k=3, pad=1) → BatchNorm2d → ReLU
  (stride in time only on the last block → reduce T by 2)

  reshape: (B, 128, n_mels', T') → (B, T', 128 * n_mels')

  bigru_1: GRU(input=128*n_mels', hidden=64, bidirectional=True)
  bigru_2: GRU(input=128,          hidden=64, bidirectional=True)

  frame_head: Linear(128, 1) → squeeze → (B, T')
  upsample:   nearest-neighbor × 2 → (B, T) — undo the last conv block's time stride
```

**Design notes:**

- Only the last conv block strides in time; earlier blocks keep frame resolution so
  the BiGRU sees a long sequence.
- Frequency is preserved through the conv stack — the network can learn which mel
  bins carry call energy.
- The reshape flattens `(128 channels × n_mels')` into channels for the GRU. Pyright
  ~`n_mels'` is a compile-time constant once the conv stack is fixed; the trainer
  asserts the shape.
- Frame head upsampling is nearest-neighbor so the decoded frame index math stays
  integer-aligned with the hop length.
- Parameter count target is ~300k. Exact count will be asserted in a unit test so a
  future refactor can't accidentally balloon it.
- Variable-length input is length-agnostic by construction — 2D conv + BiGRU handle
  any `T` up to GPU memory limits. No max-length guard in the model itself.

**Why CRNN and not a small 1D-TCN or a transformer.**

- 1D-TCN would flatten frequency into channels on the first layer, so the network
  can't learn hierarchical frequency features. Wrong inductive bias for
  spectrogram-based bioacoustic work.
- A small transformer would need more training data than the bootstrap dataset is
  going to provide. Data efficiency matters more than expressiveness for the first
  working model.
- CRNN is the field-standard architecture for framewise bioacoustic segmentation
  (Stowell, Piczak, BirdNET variants). Good inductive bias (2D conv over a
  spectrogram), good receptive field (BiGRU), small parameter count for the capacity.

---

## Training worker

File: `src/humpback/workers/segmentation_training_worker.py` (new).

Claims `segmentation_training_jobs` rows via the project's standard atomic
compare-and-set pattern. Slotted into `workers/queue.py`'s claim priority order
between `vocalization_inference` and `region_detection`:

```
... vocalization_training → vocalization_inference → segmentation_training →
region_detection → event_segmentation → event_classification → ...
```

`SegmentationTrainingJob` also joins the stale-job recovery sweep in `queue.py`, same
rule as every other worker type.

### Per-job flow

1. Claim the row, deserialize `config_json` into `SegmentationTrainingConfig`.
2. Read all `segmentation_training_samples` for the job's `training_dataset_id`.
3. Build a train / val split **by distinct audio source** (`audio_file_id` OR
   `hydrophone_id`), ~80/20, deterministic under `config.seed`. Samples from the same
   recording never cross the split boundary. Per-sample random splits leak background
   noise signature in bioacoustic ML — this rule is mandatory even on small
   datasets.
4. Instantiate a `SegmentationSampleDataset` that lazy-loads each sample's crop audio
   (via `AudioLoader` for file sources, via the hydrophone provider for hydrophone
   sources), extracts features with `extract_logmel` + `normalize_per_region_zscore`,
   builds the framewise target from `events_json`, returns
   `(features, target, mask)`.
5. Wrap in `DataLoader(batch_size=config.batch_size)` with a collate that pads to the
   max `T` in each batch and produces a matching mask. Bootstrap crops are fixed
   length so padding is typically zero.
6. Build the `SegmentationCRNN` model, `Adam` optimizer, masked `BCEWithLogitsLoss`
   with auto-computed `pos_weight`. `pos_weight` is computed once over the train set
   at the start of training by iterating the sample targets — this is cheap and
   avoids per-batch recomputation.
7. Call `ml.training_loop.fit(model, optimizer, loss_fn, train_loader, epochs,
   val_loader, callbacks=[early_stop, per_epoch_metrics])`.
8. On completion:
   - Run final eval over the val set (full decoder, event-level metrics — see below).
   - `ml.checkpointing.save_checkpoint(checkpoint_path, model, optimizer=None,
     config=...)` writes to `storage_root/segmentation_models/<model_id>/checkpoint.pt`.
   - Write `config.json` next to the checkpoint.
   - Insert a `segmentation_models` row with `model_family="pytorch_crnn"`,
     `model_path=<checkpoint_path>`, `training_job_id=<this job>`.
   - Set `segmentation_training_jobs.segmentation_model_id` and
     `segmentation_training_jobs.result_summary`.
   - Status `complete`, `completed_at` stamped.
9. On exception: delete any partial checkpoint / `.tmp` sidecars, set
   status `failed`, populate `error_message`, re-raise for the worker loop's
   error-path handling.

### Hyperparameter defaults

```
loss:            BCEWithLogitsLoss  (masked)
pos_weight:      auto = total_neg_frames / total_pos_frames  (train set only)
optimizer:       Adam(lr=1e-3, weight_decay=1e-4)
lr_schedule:     constant
batch_size:      16
epochs:          30
early_stopping:  patience=5 on val_loss
grad_clip:       1.0 (L2 norm)
seed:            42
```

All overridable via `config_json`. Defaults are deliberately conservative — bigger
LR, more epochs, different weight decay are all opt-in.

### Per-epoch callback metrics

`train_loss` (from `fit` directly), `val_loss`, `val_framewise_f1` at threshold 0.5.
Framewise is the training-time surrogate because event decoding is non-trivial and
we don't want to run it every epoch.

### Final eval metrics (one-shot over val set)

- Framewise precision / recall / F1 at threshold 0.5.
- Event-level precision / recall / F1 at IoU ≥ 0.3 — runs the full hysteresis
  decoder on val-set frame probs, matches predicted events to ground-truth events
  with IoU, counts hits / misses / extras.
- Mean absolute onset error / offset error on matched events (seconds).
- Final `val_loss`.

All persisted to `segmentation_training_jobs.result_summary` as JSON. A condensed
snapshot goes into `segmentation_models.config_json` so
`GET /segmentation-models/{id}` can report quality without loading the training job.

---

## Inference worker + decoder

File: `src/humpback/workers/event_segmentation_worker.py` (unstub the Phase 0 shell).

Claims `event_segmentation_jobs` via compare-and-set. Also joins `queue.py`'s
stale-job recovery sweep (the Phase 0 scaffold may have already added this — the
implementation should verify and add it if missing).

### Per-job flow

1. Deserialize `config_json` into `SegmentationDecoderConfig`.
2. Read the upstream `region_detection_jobs.id` row and confirm its
   `status == 'complete'`. If not, fail the Pass 2 job with a clear error.
3. Read `regions.parquet` from the upstream job's directory via
   `call_parsing.storage.read_regions`.
4. Load the `segmentation_models` row + checkpoint via `ml.checkpointing.load_checkpoint`
   into a fresh `SegmentationCRNN`, then `model.eval()`.
5. Resolve the audio source by reading the upstream Pass 1 job's source columns:
   `audio_file_id` → `AudioLoader`, hydrophone triple → archive playback provider.
   Pass 2 does not carry its own source columns — the source is whatever Pass 1 ran
   on.
6. For each region in `regions.parquet`:
   - Fetch audio for `[padded_start_sec, padded_end_sec]` (padded, not raw — matches
     what Pass 1 wrote and gives the model the context the curator saw at train time).
   - Run `extract_logmel` + `normalize_per_region_zscore`.
   - One CRNN forward pass, `batch_size=1`, no chunking.
   - Apply sigmoid to get `(T,)` frame probabilities.
   - Decode via the hysteresis decoder (below) to a list of `Event`s with
     `region_id=region.region_id` and absolute audio timestamps.
   - Append to an in-memory list.
7. After all regions: one atomic
   `call_parsing.storage.write_events(<job_dir>/events.parquet, events)`.
8. Update the row: `event_count`, `completed_at`, `status='complete'`.
9. On exception: delete any partial `events.parquet` / `.tmp` sidecars under the job
   directory, set `status='failed'`, populate `error_message`.

### Decoder module

File: `src/humpback/call_parsing/segmentation/decoder.py`.

```
def decode_events(
    frame_probs: np.ndarray,          # shape (T,), values in [0, 1]
    region_id: str,
    region_start_sec: float,          # padded_start of the region in the audio
    hop_sec: float,                   # feature-extractor hop in seconds
    config: SegmentationDecoderConfig,
) -> list[Event]:
    """Turn frame probabilities into Event rows via hysteresis.

    Pure function — no I/O, no audio, no models. Absolute audio-relative
    timestamps are computed from region_start_sec + frame_idx * hop_sec.
    """
```

**Parameters (stored in `event_segmentation_jobs.config_json`):**

```
high_threshold  = 0.5    # frame must exceed this to START an event
low_threshold   = 0.3    # frame must stay above this to CONTINUE
min_event_sec   = 0.2    # drop events shorter than this (post-merge)
merge_gap_sec   = 0.1    # fuse events whose gap is < this (frame-quantized)
```

### Algorithm

1. Walk `frame_probs` left-to-right. When a frame first crosses `high_threshold`,
   open an event. Extend it while subsequent frames stay ≥ `low_threshold`. Close the
   event when a frame dips below `low_threshold`.
2. Merge adjacent closed events whose gap in seconds is `< merge_gap_sec`. Gaps are
   frame-quantized — `merge_gap_sec` is rounded to the nearest frame count before the
   walk.
3. Drop events whose duration is `< min_event_sec`.
4. For each survivor compute:
   - `start_sec = region_start_sec + first_frame_idx * hop_sec`
   - `end_sec   = region_start_sec + (last_frame_idx + 1) * hop_sec`
   - `center_sec = (start_sec + end_sec) / 2`
   - `segmentation_confidence = max(frame_probs[first_frame_idx : last_frame_idx + 1])`
   - `event_id = uuid4().hex`
   - `region_id = <caller-provided>`
5. Return sorted by `start_sec` (already sorted by construction).

**Exhaustively unit-tested edge cases:** all-below-high (no events), all-above-low
with one peak (one event), two peaks separated by > `merge_gap_sec` (two events), two
peaks separated by < `merge_gap_sec` (one merged event), one frame above threshold
but duration < `min_event_sec` (dropped), boundary conditions (starts above
threshold, ends above threshold), single-frame events at the edges of the region.

### Why `max` for `segmentation_confidence`

Max is the most interpretable ("the loudest frame in this event was this confident")
and most robust to long events with narrow peaks. Mean would penalize calls whose
peak is narrow relative to the event duration. `p90` is a middle ground, but its
behavior changes with event length in non-obvious ways. Max stays.

---

## Bootstrap script

File: `scripts/bootstrap_segmentation_dataset.py`.

```
uv run python scripts/bootstrap_segmentation_dataset.py \
    --row-ids-file ids.txt \
    --dataset-name "bootstrap-2026-04-11" \
    --crop-seconds 10.0 \
    [--dataset-id <existing-uuid>]
    [--dry-run]
    [--allow-multi-label]
```

### Arguments

- `--row-ids-file` (required): path to a text file with one detection `row_id` per
  line. Blank lines and lines starting with `#` are ignored.
- `--dataset-name` (required if no `--dataset-id`): name for a new
  `segmentation_training_datasets` row.
- `--dataset-id` (optional): append to an existing dataset instead of creating one.
  Mutually exclusive with `--dataset-name` (Pydantic-style validation).
- `--crop-seconds` (default `10.0`): fixed crop width in seconds; the sample spans
  `[event_center - crop_seconds/2, event_center + crop_seconds/2]` clamped to audio
  bounds.
- `--dry-run` (default `False`): print planned inserts, commit nothing.
- `--allow-multi-label` (default `False`): by default, rows with more than one
  `VocalizationLabel` type are skipped. This flag accepts them anyway (user knows
  what they're doing).

### Behavior per row_id

1. Look up the detection row by stable `row_id`. If not found, log a warning and
   skip.
2. Look up the source detection job to resolve the audio source (audio_file vs
   hydrophone triple) and the row's `start_sec` / `end_sec` in audio coordinates.
3. Look up `VocalizationLabel`s for this `row_id`. If zero labels, skip with reason
   "no vocalization label". If more than one distinct type label and
   `--allow-multi-label` is not set, skip with reason "multi-label".
4. Compute the crop center as `(row.start_sec + row.end_sec) / 2`. Compute
   `crop_start_sec` and `crop_end_sec` by symmetric half-width around the center,
   clamped to `[0, audio_duration_sec]`. If the clamped crop is shorter than
   `crop_seconds * 0.5`, skip with reason "crop too short at boundary".
5. Build `events_json = [{"start_sec": row.start_sec, "end_sec": row.end_sec}]`.
   Event bounds are audio-relative, matching the schema.
6. Check idempotency: if a `segmentation_training_samples` row exists with the same
   `(training_dataset_id, source_ref=row_id)`, skip with reason "already present".
7. Insert the row with `source="bootstrap_vocalization_row"` and
   `source_ref=row_id`.

### Output

Prints dataset id, sample count (inserted / skipped / reason breakdown), and
commits the transaction. `--dry-run` prints the same summary but rolls back.

### Error handling

Strict fail-fast: any unexpected DB error, file-not-found, or unknown SQLAlchemy
exception aborts the whole run with a stack trace. The script is one-shot and
idempotent — re-running after a fix is safe.

### Why direct SQLAlchemy instead of HTTP

The bootstrap script matches the existing project pattern for utility scripts
(`scripts/export_timeline.py`, etc.): direct SQLAlchemy session via
`humpback.database`. No running API server required, simpler error handling, faster
for bulk inserts.

---

## API surface

All endpoints under `/call-parsing/`. Changes on top of Phase 0 / Pass 1:

| Method | Path | Phase 0 / Pass 1 | After Pass 2 |
|---|---|---|---|
| POST | `/segmentation-jobs` | 501 | Functional — `CreateSegmentationJobRequest` |
| GET | `/segmentation-jobs` | Functional | Unchanged |
| GET / DELETE | `/segmentation-jobs/{id}` | Functional | Unchanged |
| GET | `/segmentation-jobs/{id}/events` | 501 | Functional — streams `events.parquet` |
| POST | `/segmentation-training-jobs` | (new) | Functional — `CreateSegmentationTrainingJobRequest` |
| GET | `/segmentation-training-jobs` | (new) | Functional — list |
| GET / DELETE | `/segmentation-training-jobs/{id}` | (new) | Functional — detail + delete |
| GET | `/segmentation-models` | (new) | Functional — list |
| GET / DELETE | `/segmentation-models/{id}` | (new) | Functional — detail + delete |

**Deferred to the future UI branch** (NOT shipped in Pass 2):

- `POST/GET/DELETE /segmentation-datasets` — dataset CRUD
- `POST/GET/DELETE /segmentation-datasets/{id}/samples` — sample CRUD
- Timeline-viewer UI for editing event bounds

### Request schemas

```python
class SegmentationDecoderConfig(BaseModel):
    high_threshold: float = 0.5
    low_threshold: float = 0.3
    min_event_sec: float = 0.2
    merge_gap_sec: float = 0.1

class SegmentationTrainingConfig(BaseModel):
    epochs: int = 30
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    early_stopping_patience: int = 5
    grad_clip: float = 1.0
    seed: int = 42
    # Model knobs
    n_mels: int = 64
    conv_channels: list[int] = [32, 64, 96, 128]
    gru_hidden: int = 64
    gru_layers: int = 2

class CreateSegmentationJobRequest(BaseModel):
    region_detection_job_id: str
    segmentation_model_id: str
    parent_run_id: str | None = None
    config: SegmentationDecoderConfig = Field(default_factory=...)

class CreateSegmentationTrainingJobRequest(BaseModel):
    training_dataset_id: str
    config: SegmentationTrainingConfig = Field(default_factory=...)
```

### Error codes

- `404` — unknown `region_detection_job_id`, `segmentation_model_id`, or
  `training_dataset_id` at create time.
- `409` — `GET /segmentation-jobs/{id}/events` called on a job whose status is not
  `complete`; `DELETE /segmentation-models/{id}` called on a model referenced by an
  in-flight segmentation job; `DELETE /segmentation-training-jobs/{id}` where the
  resulting model is referenced by an in-flight segmentation job.
- `422` — Pydantic validator error on request payload.

### Service layer

`create_segmentation_training_job(session, request) -> SegmentationTrainingJob` —
validates the FK, serializes `request.config.model_dump_json()` into `config_json`,
inserts a queued row, commits, returns the model.

`create_segmentation_job(session, request) -> EventSegmentationJob` — same pattern,
validates both upstream FKs (including that the upstream Pass 1 job is `complete`).

---

## Testing

### 1. Unit — feature extractor

File: `tests/unit/test_segmentation_features.py`.

- `extract_logmel` output shape matches `(n_mels, expected_T)` for a fixture audio
  buffer.
- `fmin` / `fmax` clamping is respected (bins outside range are effectively zero).
- `normalize_per_region_zscore` produces zero mean and unit variance on a random
  input.
- `frame_index_to_audio_sec` / `audio_sec_to_frame_index` round-trip to integer
  precision.

### 2. Unit — model

File: `tests/unit/test_segmentation_model.py`.

- Forward-pass output shape `(batch, T_out)` matches the expected frame count for a
  given input.
- Parameter count matches the target (~300k, asserted to a fixed number).
- Deterministic output under fixed seed on a known input.
- Variable-length inputs in a batch are handled correctly (with padding + mask).

### 3. Unit — decoder

File: `tests/unit/test_segmentation_decoder.py`.

Exhaustive edge cases on synthetic frame-prob inputs:

- Empty input (all zeros) → no events.
- Single peak above `high` → one event, bounds at the threshold crossings.
- Single peak that only crosses `low` → no events.
- Two peaks separated by > `merge_gap_sec` → two events.
- Two peaks separated by < `merge_gap_sec` → one merged event.
- Event shorter than `min_event_sec` → dropped.
- Hysteresis: peak briefly dips below `high` but stays above `low` → one event, not
  two.
- Boundary conditions: starts above threshold, ends above threshold.
- `max` confidence correctness on a known input (e.g. frames `[0.8, 0.9, 0.7]` →
  confidence `0.9`).

### 4. Unit — dataset

File: `tests/unit/test_segmentation_dataset.py`.

- Target vector construction for a sample with one event: frames inside → `1`,
  outside → `0`.
- Target vector construction for a sample with multiple events: all events marked.
- Target vector for an empty-events sample: all zeros.
- Frame at event boundary: handled deterministically (inclusive or exclusive, just
  consistent — specify and assert).
- Mask shape matches target shape.

### 5. Unit — bootstrap script

File: `tests/unit/test_bootstrap_segmentation_dataset.py`.

- Happy path: existing detection row + single vocalization label → sample inserted.
- No vocalization label → skipped.
- Multi-label without `--allow-multi-label` → skipped.
- Multi-label with `--allow-multi-label` → inserted.
- Idempotency: re-running with the same row_id → no duplicate insert.
- `--dry-run` → no DB changes.
- Unknown row_id → skipped with warning.
- Crop too short at audio boundary → skipped.

### 6. Integration — training worker

File: `tests/integration/test_segmentation_training_worker.py`.

End-to-end training run with minimum-size config (1 conv block, 1 GRU, 2 epochs) on
procedurally-generated synthetic data:

- Create a `segmentation_training_datasets` row + a handful of
  `segmentation_training_samples` with synthetic audio fixtures and deterministic
  event bounds.
- Create a `segmentation_training_jobs` row.
- Run one worker iteration.
- Assert: status `complete`, checkpoint file exists at the expected path,
  `segmentation_models` row inserted, `result_summary` populated with the expected
  metric keys, `segmentation_training_jobs.segmentation_model_id` set.
- Failure path: force the trainer to raise, assert status `failed` and
  `error_message` populated, assert no partial `segmentation_models` row.

### 7. Integration — event segmentation worker

File: `tests/integration/test_event_segmentation_worker.py`.

- Use the checkpoint from the training worker test (or a separately-built tiny
  checkpoint).
- Create a Pass 1 `RegionDetectionJob` with a synthetic fixture region.
- Create an `EventSegmentationJob` pointing at it.
- Run one worker iteration.
- Assert: status `complete`, `events.parquet` exists, at least one event decoded,
  bounds within the region's `[padded_start_sec, padded_end_sec]`.
- Failure path: corrupt the checkpoint or stub the model to raise, assert status
  `failed` and no partial parquet.

### 8. Migration round-trip

File: `tests/test_migrations.py` (extended).

- Fresh-DB upgrade to `044` → all three tables exist with the expected columns.
- Upgrade + downgrade + upgrade → no errors, schema matches.

### 9. API router

File: `tests/api/test_call_parsing_router.py` (extended).

- `POST /segmentation-training-jobs` happy path.
- `404` on unknown `training_dataset_id`.
- `422` on malformed request.
- `POST /segmentation-jobs` happy path.
- `404` on unknown `region_detection_job_id` / `segmentation_model_id`.
- `409` on upstream Pass 1 job not complete.
- `409` on `GET /segmentation-jobs/{id}/events` before job completes.
- `GET /segmentation-jobs/{id}/events` happy path after running worker
  synchronously in-test.
- `DELETE /segmentation-models/{id}` happy path + checkpoint file cleanup +
  `409` when referenced by in-flight job.

### 10. Smoke

File: `tests/api/test_call_parsing_smoke.py` (new or extended).

- Use existing Pass 1 fixture machinery to produce a `regions.parquet`.
- Create a tiny synthetic training dataset.
- Run the training worker synchronously.
- Run the event segmentation worker synchronously against the new model + the Pass 1
  fixture.
- Assert `events.parquet` exists on disk and at least one event was decoded.

### 11. Deferred

- Hydrophone-path integration test for the event segmentation worker — no
  `ArchivePlaybackProvider` mock surface. Noted in `docs/plans/backlog.md`.
- Manual end-to-end training run against curated real bootstrap data — a follow-up
  session, not part of Pass 2's definition of done.

---

## ADR-050 — Pass 2 bootstrap-era design decisions

A single new ADR appended to `DECISIONS.md`, capturing as a grouped decision:

- **Framewise α supervision target.** Binary presence inside/outside event bounds,
  masked `BCEWithLogitsLoss` with auto `pos_weight`. Rationale: bootstrap row bounds
  are too loose for onset/offset point targets, and α is the simplest thing that
  gives the hysteresis decoder a clean frame probability vector to consume. γ
  (ignore band at edges) is a cost-free loss-swap upgrade if α's edge calibration
  proves bad.
- **No PCEN for Pass 2 features.** Per-region z-score is the default; ADR-047
  already opened the door for extending PCEN to the classifier feature pipeline if
  SNR demands. Pass 2 deliberately does not walk through that door yet — simpler
  first, upgrade when data demands it.
- **Per-audio-file train/val split.** Per-sample random splits leak background noise
  signature in bioacoustic ML. The rule is mandatory even on small bootstrap
  datasets.
- **Persistent training dataset contract.** `segmentation_training_datasets` +
  `segmentation_training_samples` are designed for both the one-shot bootstrap
  script and the future UI timeline-editor workflow. The bootstrap is the first
  writer, not the only writer. No schema migration when the UI extension lands.
- **CRNN at ~300k parameters as the first shipping architecture.** Field-standard
  inductive bias for bioacoustic framewise segmentation, data-efficient relative to
  a transformer, produces a clean frame head for the hysteresis decoder. The
  `segmentation_models.model_family` column is the extension hook for when a
  different architecture earns its place.
- **Separate `segmentation_training_jobs` table.** Framewise regression is a
  different task from Pass 3's per-event multi-label classification; keeping the
  training-job table separate lets the trainer code stay cleanly isolated and avoids
  growing yet another `model_family` branch on `vocalization_training_jobs`.

---

## Documentation updates

- **`CLAUDE.md` §8.9** — mark `POST /call-parsing/segmentation-jobs`,
  `GET /segmentation-jobs/{id}/events`, and the new segmentation-training-job and
  segmentation-model endpoints as functional; remove their 501 callouts. Add the new
  endpoints to the API surface table.
- **`CLAUDE.md` §8.7** — add behavioral-constraint bullets for (a) the Pass 2
  framewise α contract, (b) the per-audio-file train/val split rule, (c) the Pass 2
  inherits-source-from-upstream rule (inference worker resolves the audio source
  from the upstream Pass 1 job's columns, not its own).
- **`CLAUDE.md` §9.1** — append "Pass 2 event segmentation (training + inference)"
  to Implemented Capabilities; update the call-parsing pipeline status line.
- **`CLAUDE.md` §9.2** — bump latest migration to
  `044_segmentation_training_tables.py`; add `segmentation_training_datasets`,
  `segmentation_training_samples`, `segmentation_training_jobs` to the Tables list.
- **`docs/reference/data-model.md`** — add condensed entries for all three new
  tables.
- **`DECISIONS.md`** — append ADR-050 as described above.
- **`README.md`** — if it lists endpoints, add the new Pass 2 endpoints; otherwise
  no-op.
- **`docs/plans/backlog.md`** — add the deferred hydrophone-path integration test
  and the deferred manual real-data training run.

---

## Non-goals

See Scope section for the full list. Headline items:

- All frontend work (timeline editor, dataset UI, training-job status page).
- Dataset / sample mutation API endpoints.
- GPU-specific optimization.
- Real-time / streaming inference.
- Multi-label frame heads (Pass 3).
- Data augmentation.
- Hydrophone-path integration test.
- Pass 3 / Pass 4 logic.
- Manual training run on real curated data.
