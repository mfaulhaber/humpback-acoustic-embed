# Call Parsing — Pass 2: Event Segmentation Implementation Plan

**Goal:** Turn the Phase 0 Pass 2 stubs into a working training + inference pipeline — a PyTorch CRNN that learns framewise humpback call presence, with a hysteresis decoder that produces `events.parquet` from `regions.parquet`, plus a one-shot bootstrap script and the supporting API surface.
**Spec:** [docs/specs/2026-04-11-call-parsing-pass2-segmentation-design.md](../specs/2026-04-11-call-parsing-pass2-segmentation-design.md)
**Branch:** `feature/call-parsing-pass2-segmentation`

---

## Task ordering

Tasks 1 (migration) and 2 (schemas + service layer) are foundational — everything else depends on them. Tasks 3–7 (feature / model / decoder / dataset / trainer / inference modules) are mostly independent pure-code modules that can be written in any order once the schemas land. Task 8 (training worker) depends on 3–7. Task 9 (event segmentation worker) depends on 7 and the decoder. Task 10 (API router) depends on 2 and can land after the workers are ready. Task 11 (bootstrap script) depends only on 1 + 2 and can land any time after those. Tasks 12 (ADR) and 13 (docs) are documentation and run after everything else is green. Task 14 (smoke test) is the final gate.

---

### Task 1: Migration `044` — segmentation training tables

**Files:**
- Create: `alembic/versions/044_segmentation_training_tables.py`
- Create: `src/humpback/models/segmentation_training.py` (new SQLAlchemy model module)
- Modify: `src/humpback/models/__init__.py` to export the three new model classes

**Acceptance criteria:**
- [ ] Alembic migration uses `op.batch_alter_table()` for SQLite compatibility
- [ ] Creates `segmentation_training_datasets` with columns `id` (String, PK), `name` (String, not null), `description` (Text, nullable), `created_at` / `updated_at` (DateTime, not null)
- [ ] Creates `segmentation_training_samples` with all columns specified in the spec (id, training_dataset_id, audio_file_id, hydrophone_id, start_timestamp, end_timestamp, crop_start_sec, crop_end_sec, events_json, source, source_ref, notes, created_at, updated_at)
- [ ] Creates an index on `segmentation_training_samples.training_dataset_id`
- [ ] Creates a composite index on `segmentation_training_samples(training_dataset_id, source_ref)` to support the bootstrap script's idempotency check
- [ ] Creates `segmentation_training_jobs` with all columns specified in the spec (id, status, training_dataset_id, config_json, segmentation_model_id, result_summary, error_message, started_at, completed_at, created_at, updated_at)
- [ ] `downgrade()` drops all three tables and all their indexes
- [ ] SQLAlchemy models in `src/humpback/models/segmentation_training.py` match the migration exactly, using `UUIDMixin` and `TimestampMixin` where appropriate
- [ ] `src/humpback/models/__init__.py` exports `SegmentationTrainingDataset`, `SegmentationTrainingSample`, `SegmentationTrainingJob`
- [ ] `uv run alembic upgrade head` applies cleanly on a fresh DB and on a snapshot DB already at migration `043`

**Tests needed:**
- Migration round-trip test in `tests/test_migrations.py` covering upgrade → downgrade → upgrade for `044`, asserting all three tables and indexes exist after the final upgrade and are gone after downgrade

---

### Task 2: Pydantic schemas and service layer

**Files:**
- Modify: `src/humpback/schemas/call_parsing.py`
- Modify: `src/humpback/services/call_parsing.py`

**Acceptance criteria:**
- [ ] New `SegmentationFeatureConfig` Pydantic model with frozen defaults (sample_rate=16000, n_fft=2048, hop_length=512, n_mels=64, fmin=20, fmax=4000, normalize="per_region_zscore")
- [ ] New `SegmentationTrainingConfig` Pydantic model with the hyperparameter defaults specified in the spec (epochs=30, batch_size=16, learning_rate=1e-3, weight_decay=1e-4, early_stopping_patience=5, grad_clip=1.0, seed=42, n_mels=64, conv_channels=[32,64,96,128], gru_hidden=64, gru_layers=2)
- [ ] New `SegmentationDecoderConfig` Pydantic model with defaults high_threshold=0.5, low_threshold=0.3, min_event_sec=0.2, merge_gap_sec=0.1, and a validator that enforces `low_threshold < high_threshold` and both in `[0, 1]`
- [ ] New `CreateSegmentationTrainingJobRequest` Pydantic model with required `training_dataset_id` and nested `config: SegmentationTrainingConfig`
- [ ] New `CreateSegmentationJobRequest` Pydantic model with required `region_detection_job_id`, `segmentation_model_id`, optional `parent_run_id`, and nested `config: SegmentationDecoderConfig`
- [ ] New response models as needed for list / detail / result endpoints — at minimum `SegmentationTrainingJobResponse`, `SegmentationJobResponse`, `SegmentationModelResponse`, consistent with the existing Pass 1 response model style
- [ ] New `create_segmentation_training_job(session, request) -> SegmentationTrainingJob` service method — validates `training_dataset_id` exists (404 on missing), serializes `request.config.model_dump_json()` into `config_json`, inserts a queued row, commits, returns the model
- [ ] New `create_segmentation_job(session, request) -> EventSegmentationJob` service method — validates `region_detection_job_id` exists AND is in status `complete` (409 if not complete, 404 if not found), validates `segmentation_model_id` exists (404 on missing), serializes `request.config.model_dump_json()` into `config_json`, inserts a queued row, commits, returns the model

**Tests needed:**
- `SegmentationTrainingConfig` default values and round-trip JSON serialization
- `SegmentationDecoderConfig` validator: rejects low >= high, rejects values outside [0,1], accepts valid values
- `CreateSegmentationTrainingJobRequest` validator on the whole request payload
- `CreateSegmentationJobRequest` validator
- `create_segmentation_training_job` happy path + 404 on missing `training_dataset_id`
- `create_segmentation_job` happy path + 404 on missing FKs + 409 when upstream Pass 1 job is not `complete`

---

### Task 3: Feature extractor module

**Files:**
- Create: `src/humpback/call_parsing/segmentation/__init__.py` (empty, package marker)
- Create: `src/humpback/call_parsing/segmentation/features.py`
- Create: `tests/unit/test_segmentation_features.py`

**Acceptance criteria:**
- [ ] `SegmentationFeatureConfig` dataclass (or reuse the Pydantic model from Task 2) with the parameters frozen in the spec
- [ ] `extract_logmel(audio: np.ndarray, config: SegmentationFeatureConfig) -> np.ndarray` returns shape `(n_mels, T)` where `T` is computed from `audio.shape[0]`, `n_fft`, and `hop_length`
- [ ] Uses `librosa.feature.melspectrogram` with `fmin` / `fmax` / `n_mels` / `n_fft` / `hop_length` respected, then `librosa.power_to_db` — matching the style used in `processing/features.py` but as an independent function that does NOT import from there
- [ ] `normalize_per_region_zscore(logmel: np.ndarray) -> np.ndarray` returns the input z-scored (subtract mean, divide by std), with a small epsilon to avoid divide-by-zero on silent inputs
- [ ] `frame_index_to_audio_sec(frame_idx: int, config: SegmentationFeatureConfig) -> float` computes `frame_idx * hop_length / sample_rate`
- [ ] `audio_sec_to_frame_index(time_sec: float, config: SegmentationFeatureConfig) -> int` computes the reverse, rounding down
- [ ] All four helpers are pure functions — no I/O, no models, no global state

**Tests needed:**
- `extract_logmel` output shape matches the expected `(n_mels, T)` for a known-length fixture audio buffer
- `fmin` / `fmax` clamping is observable (bins for frequencies outside [20, 4000] are effectively zero on a sine-wave input above `fmax` or below `fmin`)
- `normalize_per_region_zscore` produces near-zero mean and near-unit variance on a random input
- `frame_index_to_audio_sec` / `audio_sec_to_frame_index` round-trip to integer precision across a range of values
- Pure-function property: same input → same output, no hidden state

---

### Task 4: Model module

**Files:**
- Create: `src/humpback/call_parsing/segmentation/model.py`
- Create: `tests/unit/test_segmentation_model.py`

**Acceptance criteria:**
- [ ] `SegmentationCRNN(nn.Module)` class whose `__init__` accepts `n_mels`, `conv_channels` (default `[32, 64, 96, 128]`), `gru_hidden` (default `64`), `gru_layers` (default `2`)
- [ ] Conv stack: four `Conv2d → BatchNorm2d → ReLU` blocks with `k=3, pad=1`; only the final block strides `(1, 2)` to halve the time dimension
- [ ] Reshape to `(B, T', C × n_mels')` where `n_mels'` is the post-conv mel dimension (unchanged if no frequency striding)
- [ ] Two-layer bidirectional GRU with `hidden_size=gru_hidden`, batch_first
- [ ] Frame head: `Linear(2 * gru_hidden, 1)` producing raw logits (no sigmoid)
- [ ] Time-axis upsample by `2×` (nearest-neighbor) to undo the last conv's stride and restore the original `T` frame count
- [ ] `forward(x: Tensor) -> Tensor`: accepts `(B, 1, n_mels, T)` input (1 channel), returns `(B, T)` logits
- [ ] Parameter count is approximately 300k — assert within ±20% in a test (allows small architecture tweaks without having to re-derive the exact count each time)
- [ ] Handles variable-length `T` across forward passes
- [ ] Fully deterministic under a fixed seed

**Tests needed:**
- Forward-pass output shape `(batch, T)` for a known input shape
- Parameter count lies in `[240_000, 360_000]`
- Deterministic output under `torch.manual_seed(42)` on a fixed input
- Two different input `T` values produce two different-length outputs (variable-length smoke test)
- Gradients flow through the whole network (one backward pass on a dummy loss, assert all leaf parameters have non-None grads)

---

### Task 5: Decoder module

**Files:**
- Create: `src/humpback/call_parsing/segmentation/decoder.py`
- Create: `tests/unit/test_segmentation_decoder.py`

**Acceptance criteria:**
- [ ] `decode_events(frame_probs, region_id, region_start_sec, hop_sec, config) -> list[Event]` is a pure function: no I/O, no audio, no models
- [ ] Accepts `frame_probs: np.ndarray` of shape `(T,)` with values in `[0, 1]`
- [ ] Applies the hysteresis algorithm specified in the spec: walk left-to-right, open events on crossings of `high_threshold`, extend while ≥ `low_threshold`, close on dips
- [ ] Merges adjacent closed events whose inter-event gap in seconds is `< merge_gap_sec` (gap is frame-quantized)
- [ ] Drops events whose duration is `< min_event_sec`
- [ ] Computes `start_sec = region_start_sec + first_frame_idx * hop_sec`, `end_sec = region_start_sec + (last_frame_idx + 1) * hop_sec`, `center_sec = (start_sec + end_sec) / 2`
- [ ] Computes `segmentation_confidence = max(frame_probs[first : last + 1])`
- [ ] Assigns `event_id = uuid4().hex` per surviving event
- [ ] Carries `region_id` from the argument to each emitted `Event`
- [ ] Returns a `list[Event]` sorted by `start_sec` (already sorted by construction)
- [ ] Handles empty input (all zeros) and single-frame inputs without errors

**Tests needed:**
- Empty input (all zeros) → empty list
- Single peak above `high` → one event with bounds at the threshold crossings
- Single peak that only crosses `low` → no events (never crossed `high`)
- Two peaks separated by > `merge_gap_sec` → two events
- Two peaks separated by < `merge_gap_sec` → one merged event
- Hysteresis: peak dips briefly below `high` but stays above `low` → one event, not two
- Event duration < `min_event_sec` → dropped
- Boundary: starts above threshold → valid event starting at frame 0
- Boundary: ends above threshold → valid event ending at the last frame
- `max` confidence correctness on a known input (frames `[0.8, 0.9, 0.7]` → confidence `0.9`)
- Absolute timestamp computation: `region_start_sec=100.0, hop_sec=0.032, first_frame_idx=10` → `start_sec=100.32`
- All emitted `event_id` values are unique UUID4 hex strings
- Every emitted `Event` carries the caller-provided `region_id`

---

### Task 6: Dataset module

**Files:**
- Create: `src/humpback/call_parsing/segmentation/dataset.py`
- Create: `tests/unit/test_segmentation_dataset.py`

**Acceptance criteria:**
- [ ] `build_framewise_target(events_json, crop_start_sec, crop_end_sec, feature_config) -> np.ndarray` returns shape `(T,)` float32 target vector with `1.0` inside any event bound and `0.0` outside
- [ ] `build_framewise_target` is a pure function suitable for unit testing without audio or models
- [ ] `SegmentationSampleDataset(torch.utils.data.Dataset)` class whose `__init__` takes a list of `SegmentationTrainingSample` rows, a `SegmentationFeatureConfig`, and an `AudioLoader`-like callable for fetching audio
- [ ] `__len__` returns the sample count; `__getitem__` returns a tuple `(features: Tensor, target: Tensor, mask: Tensor)` for one sample
- [ ] Audio fetch is lazy — done in `__getitem__`, not `__init__` — to keep Dataset construction fast
- [ ] `__getitem__` resolves the sample's audio source (`audio_file_id` XOR hydrophone triple), fetches `[crop_start_sec, crop_end_sec]` audio, extracts log-mel, normalizes, builds the framewise target, returns tensors
- [ ] A custom `collate_fn(batch)` pads to max `T` in the batch and produces a matching boolean mask (`True` where real, `False` where padded)
- [ ] Empty `events_json` (`[]`) produces an all-zeros target, handled without errors
- [ ] `compute_pos_weight(dataset) -> float` helper iterates `build_framewise_target` over all samples and returns `total_neg_frames / total_pos_frames` (handling the edge case of zero positives by returning `1.0` and logging a warning)

**Tests needed:**
- `build_framewise_target` with one event: frames whose center is inside → 1, outside → 0
- `build_framewise_target` with multiple events: all events represented
- `build_framewise_target` with empty events: all zeros
- `build_framewise_target` frame-boundary behavior: the frame whose center sits exactly at the event boundary is consistent (spec the behavior — inclusive or exclusive — and assert)
- `collate_fn` with two samples of different `T`: output batch has shape `(2, max_T)`, mask correctly marks padded frames
- `compute_pos_weight` on a dataset with known balance returns the expected ratio
- `compute_pos_weight` on an all-empty-events dataset returns `1.0` without division error

---

### Task 7: Trainer + inference modules

**Files:**
- Create: `src/humpback/call_parsing/segmentation/trainer.py`
- Create: `src/humpback/call_parsing/segmentation/inference.py`
- Create: `tests/unit/test_segmentation_trainer.py` — smoke test for the trainer driver only, not the full worker

**Acceptance criteria:**
- [ ] `train_model(dataset, config, checkpoint_path) -> TrainingResult` in `trainer.py` — assembles model, optimizer, loss, splits by audio source, builds loaders, calls `ml.training_loop.fit`, runs final eval, saves the checkpoint via `ml.checkpointing.save_checkpoint`, returns a dataclass with train/val history and final metrics
- [ ] `split_by_audio_source(samples, val_fraction, seed) -> (train_samples, val_samples)` — groups samples by `audio_file_id` OR `hydrophone_id`, shuffles groups under the seed, assigns the first ~`val_fraction` of groups to val, ensures no audio source appears in both splits
- [ ] Auto `pos_weight` computed once on the train set before training begins
- [ ] Per-epoch callback records `train_loss`, `val_loss`, `val_framewise_f1` at threshold 0.5
- [ ] Early stopping callback with configurable patience on `val_loss`
- [ ] Final eval over val set produces framewise P/R/F1 at threshold 0.5, event-level P/R/F1 at IoU ≥ 0.3 (runs the full decoder on val-set frame probs, matches predicted events to ground-truth), mean absolute onset/offset error on matched events
- [ ] Event matching helper: `match_events_by_iou(pred_events, gt_events, iou_threshold) -> (hits, misses, extras, onset_errors, offset_errors)` — pure function, unit-testable independently
- [ ] `run_inference(model, region, audio_loader, feature_config, decoder_config) -> list[Event]` in `inference.py` — fetches the region audio, extracts features, runs one forward pass, applies sigmoid, calls the decoder, returns events
- [ ] `run_inference` is the pure per-region step used by the event segmentation worker; it takes a live model + region + audio source and returns events, no DB access

**Tests needed:**
- `split_by_audio_source` with synthetic samples across three distinct audio sources: assert no source appears in both splits, assert deterministic under fixed seed
- `split_by_audio_source` with a single audio source: all samples go to one split, other split is empty, no crash
- `match_events_by_iou` edge cases: empty predictions, empty ground truth, perfect match, partial match, one-to-many handling
- Trainer smoke: a tiny 1-conv-block CRNN trained for 2 epochs on synthetic tensors converges (train loss decreases), final eval metrics populated in the result
- `run_inference` on a synthetic region + a seeded random model produces a deterministic list of events

---

### Task 8: Segmentation training worker

**Files:**
- Create: `src/humpback/workers/segmentation_training_worker.py`
- Modify: `src/humpback/workers/queue.py` — add `SegmentationTrainingJob` to the stale-job recovery sweep and the claim priority order
- Create: `tests/integration/test_segmentation_training_worker.py`

**Acceptance criteria:**
- [ ] Worker claims `segmentation_training_jobs` rows via the project's standard atomic compare-and-set pattern
- [ ] Worker is added to `workers/queue.py`'s claim priority order between `vocalization_inference` and `region_detection`
- [ ] Worker is added to `workers/queue.py`'s stale-job recovery sweep
- [ ] On claim: deserialize `config_json` into `SegmentationTrainingConfig`, read samples for `training_dataset_id`, call `train_model`, handle success + failure
- [ ] On success: save checkpoint to `storage_root/segmentation_models/<model_id>/checkpoint.pt`, write `config.json` alongside, insert a `segmentation_models` row with `model_family="pytorch_crnn"`, link the new model back via `segmentation_training_jobs.segmentation_model_id`, write `result_summary` JSON, set status `complete`, stamp `completed_at`
- [ ] The condensed metrics snapshot written into `segmentation_models.config_json` contains at least the framewise F1 and event F1 at IoU ≥ 0.3 so a downstream list endpoint can display them without loading the training job row
- [ ] On exception: delete any partial checkpoint / `.tmp` sidecar / `config.json` written mid-save, set status `failed`, populate `error_message`, do not leave a partial `segmentation_models` row
- [ ] Dataset FK is validated at claim time — missing dataset fails the job immediately with a clear error

**Tests needed:**
- Integration test with a minimum-size config (1 conv block, 1 GRU, 2 epochs) trained on procedurally-generated synthetic data: create a `segmentation_training_datasets` row + a handful of `segmentation_training_samples` with synthetic audio, create a `segmentation_training_jobs` row, run one worker iteration, assert status `complete`, assert checkpoint file exists, assert `segmentation_models` row inserted, assert `result_summary` contains the expected keys, assert `segmentation_training_jobs.segmentation_model_id` is set
- Failure path: stub the trainer to raise mid-training, assert status `failed`, assert `error_message` populated, assert no `segmentation_models` row inserted, assert no partial checkpoint file on disk
- Missing dataset: create a training job row with a nonexistent `training_dataset_id`, run the worker, assert status `failed` with a clear error message
- Stale recovery: insert a running training job with a stale `updated_at`, run the recovery sweep, assert status `queued`

---

### Task 9: Event segmentation worker (unstub)

**Files:**
- Modify: `src/humpback/workers/event_segmentation_worker.py` — replace the Phase 0 stub
- Modify: `src/humpback/workers/queue.py` — verify `EventSegmentationJob` is in the stale-job recovery sweep (Phase 0 may or may not have added it; this task ensures it is)
- Create: `tests/integration/test_event_segmentation_worker.py`

**Acceptance criteria:**
- [ ] Worker claims `event_segmentation_jobs` rows via the standard atomic compare-and-set
- [ ] Deserializes `config_json` into `SegmentationDecoderConfig`
- [ ] Reads the upstream `region_detection_jobs` row, confirms its `status == 'complete'`; if not, fails the Pass 2 job with a clear error
- [ ] Reads `regions.parquet` from the upstream job's directory via `call_parsing.storage.read_regions`
- [ ] Loads the `segmentation_models` row + checkpoint via `ml.checkpointing.load_checkpoint` into a fresh `SegmentationCRNN`, then `model.eval()`
- [ ] Resolves the audio source from the upstream Pass 1 job's source columns (`audio_file_id` or hydrophone triple), NOT from any Pass 2 column
- [ ] For each region: fetches `[padded_start_sec, padded_end_sec]` audio, extracts features, runs one CRNN forward pass (batch_size=1), applies sigmoid, calls `decode_events` with the region's `region_id` and `padded_start_sec` as `region_start_sec`
- [ ] Accumulates all events into one in-memory list, then one atomic `write_events(<job_dir>/events.parquet, events)` call at the end
- [ ] Updates `event_count`, `completed_at`, `status='complete'`
- [ ] On exception: deletes partial `events.parquet` / `.tmp` sidecars under the job directory, sets `status='failed'`, populates `error_message`
- [ ] `EventSegmentationJob` is confirmed in the stale-job recovery sweep in `queue.py`

**Tests needed:**
- Integration test with a small synthetic checkpoint (can reuse the tiny model trained by the training worker integration test, or build a new one): create a Pass 1 `RegionDetectionJob` with a synthetic fixture region (audio file source), create an `EventSegmentationJob` pointing at it, run one worker iteration, assert status `complete`, assert `events.parquet` exists, assert at least one event decoded, assert every event's bounds lie inside `[region.padded_start_sec, region.padded_end_sec]`
- Failure path: corrupt the checkpoint path or stub the model to raise, run one worker iteration, assert status `failed`, assert `error_message` populated, assert no partial `events.parquet`
- Upstream not complete: create a Pass 1 job with `status='queued'`, create a Pass 2 job, run the worker, assert status `failed` with a clear upstream error
- Stale recovery sweep covers `EventSegmentationJob`

---

### Task 10: API router

**Files:**
- Modify: `src/humpback/api/routers/call_parsing.py`
- Modify: `tests/api/test_call_parsing_router.py`

**Acceptance criteria:**
- [ ] `POST /call-parsing/segmentation-jobs` accepts `CreateSegmentationJobRequest`, calls `create_segmentation_job`, returns the new row (no longer 501)
- [ ] `GET /call-parsing/segmentation-jobs/{id}/events` reads `events.parquet` via `call_parsing.storage.read_events` and returns it as JSON; 409 when the job is not `complete`, 404 when the file is missing (no longer 501)
- [ ] `POST /call-parsing/segmentation-training-jobs` accepts `CreateSegmentationTrainingJobRequest`, calls `create_segmentation_training_job`, returns the new row
- [ ] `GET /call-parsing/segmentation-training-jobs` lists rows with pagination
- [ ] `GET /call-parsing/segmentation-training-jobs/{id}` returns the detail including `result_summary` when present
- [ ] `DELETE /call-parsing/segmentation-training-jobs/{id}` removes the row; returns 409 if the resulting `segmentation_models` row is referenced by an in-flight `event_segmentation_jobs` row
- [ ] `GET /call-parsing/segmentation-models` lists rows with minimal fields (id, name, model_family, created_at, condensed metrics snapshot)
- [ ] `GET /call-parsing/segmentation-models/{id}` returns the full row
- [ ] `DELETE /call-parsing/segmentation-models/{id}` removes the row AND deletes the checkpoint directory on disk; returns 409 if referenced by an in-flight `event_segmentation_jobs` row
- [ ] All error paths return JSON bodies consistent with the existing Pass 1 router style

**Tests needed:**
- `POST /segmentation-training-jobs` happy path
- `POST /segmentation-training-jobs` 404 on unknown `training_dataset_id`
- `POST /segmentation-training-jobs` 422 on malformed payload
- `POST /segmentation-jobs` happy path
- `POST /segmentation-jobs` 404 on unknown FK
- `POST /segmentation-jobs` 409 when upstream Pass 1 job is not `complete`
- `GET /segmentation-jobs/{id}/events` 409 while the job is not `complete`
- `GET /segmentation-jobs/{id}/events` happy path after running the worker synchronously in-test
- `DELETE /segmentation-models/{id}` happy path — asserts DB row gone AND checkpoint directory removed
- `DELETE /segmentation-models/{id}` 409 when referenced by an in-flight job

---

### Task 11: Bootstrap script

**Files:**
- Create: `scripts/bootstrap_segmentation_dataset.py`
- Create: `tests/unit/test_bootstrap_segmentation_dataset.py`

**Acceptance criteria:**
- [ ] CLI parses `--row-ids-file` (required), `--dataset-name` / `--dataset-id` (mutually exclusive, exactly one required), `--crop-seconds` (default `10.0`), `--dry-run` (flag), `--allow-multi-label` (flag)
- [ ] Opens a direct SQLAlchemy session via `humpback.database` — no HTTP, no API
- [ ] Reads the row ids file, skipping blank lines and lines starting with `#`
- [ ] If `--dataset-name` is given, creates a new `segmentation_training_datasets` row; if `--dataset-id` is given, loads the existing one (404-style error if not found)
- [ ] Per row id: looks up the detection row by stable `row_id`; on not-found, logs a warning and continues
- [ ] Resolves the audio source and row timestamps by joining against the detection job row store; reads the source audio duration to validate the crop doesn't go out of bounds
- [ ] Looks up `VocalizationLabel`s for the row id; if zero → skip with reason "no vocalization label"; if >1 distinct type labels and `--allow-multi-label` is not set → skip with reason "multi-label"
- [ ] Computes `crop_start_sec` / `crop_end_sec` symmetrically around the event center, clamped to `[0, audio_duration_sec]`; if the clamped crop is shorter than `crop_seconds * 0.5`, skip with reason "crop too short at boundary"
- [ ] Builds `events_json = [{"start_sec": row.start_sec, "end_sec": row.end_sec}]` (audio-relative)
- [ ] Checks idempotency: if a `segmentation_training_samples` row exists with `(training_dataset_id=<target>, source_ref=row_id)`, skip with reason "already present"
- [ ] Inserts the row with `source="bootstrap_vocalization_row"`, `source_ref=row_id`
- [ ] `--dry-run`: performs all lookups and validation, prints the would-be inserts, rolls back the transaction
- [ ] Prints a final summary: dataset id, inserted count, skipped count broken down by reason
- [ ] Fail-fast on unexpected errors — no swallowed exceptions

**Tests needed:**
- Happy path: existing detection row + single vocalization label → sample inserted, return code 0
- No vocalization label → skipped with correct reason
- Multi-label without `--allow-multi-label` → skipped with correct reason
- Multi-label with `--allow-multi-label` → inserted
- Idempotency: re-running on the same row id → no duplicate insert (same dataset)
- `--dry-run` → no DB changes, summary still printed
- Unknown row id in the file → skipped with warning, other rows still processed
- Crop too short at audio boundary (row near start of audio, `crop_seconds` too large) → skipped
- Mutually exclusive `--dataset-name` / `--dataset-id` → argparse error or explicit validation error
- `--dataset-id` pointing at a nonexistent dataset → clear error, no partial writes

---

### Task 12: ADR-050

**Files:**
- Modify: `DECISIONS.md`

**Acceptance criteria:**
- [ ] New ADR-050 entry appended to `DECISIONS.md` following the existing format (Date, Status, Context, Decision, Consequences)
- [ ] Captures, with rationale, each of the bootstrap-era decisions listed in the spec's ADR section: framewise α supervision target, no PCEN for Pass 2 features, per-audio-file train/val split, persistent training dataset contract, CRNN at ~300k parameters, separate `segmentation_training_jobs` table
- [ ] Cross-references ADR-047 (PCEN) and ADR-048 / ADR-049 (Phase 0 and Pass 1) where appropriate

---

### Task 13: Documentation updates

**Files:**
- Modify: `CLAUDE.md` — §8.9, §8.7, §9.1, §9.2
- Modify: `docs/reference/data-model.md`
- Modify: `README.md` (if it lists endpoints — otherwise no-op)
- Modify: `docs/plans/backlog.md`

**Acceptance criteria:**
- [ ] CLAUDE.md §8.9 — `POST /call-parsing/segmentation-jobs`, `GET /call-parsing/segmentation-jobs/{id}/events`, `POST/GET/DELETE /call-parsing/segmentation-training-jobs*`, `GET/DELETE /call-parsing/segmentation-models*` marked as functional; 501 callouts removed; new endpoints added to the API surface table
- [ ] CLAUDE.md §8.7 — new behavioral-constraint bullets for (a) the Pass 2 framewise α contract, (b) the per-audio-file train/val split mandatory rule, (c) the Pass 2 inherits-source-from-upstream rule (inference worker resolves audio source from upstream Pass 1 job, not from its own columns)
- [ ] CLAUDE.md §9.1 — "Pass 2 event segmentation (training + inference)" appended to Implemented Capabilities; call-parsing pipeline status line updated to "Phase 0 + Pass 1 + Pass 2"
- [ ] CLAUDE.md §9.2 — latest migration bumped to `044_segmentation_training_tables.py`; `segmentation_training_datasets`, `segmentation_training_samples`, `segmentation_training_jobs` added to the Tables list
- [ ] `docs/reference/data-model.md` — condensed entries for all three new tables with column lists and purpose
- [ ] `docs/plans/backlog.md` — new entries for (a) deferred hydrophone-path integration test for the event segmentation worker, (b) deferred manual end-to-end training run on real curated bootstrap data, (c) future UI branch for timeline-viewer event-bound editor + dataset/sample CRUD API

---

### Task 14: Smoke test

**Files:**
- Modify: `tests/api/test_call_parsing_smoke.py` (or wherever the existing Pass 1 smoke test lives) — or create a new file under `tests/api/` if none exists

**Acceptance criteria:**
- [ ] New smoke-test scenario: use existing Pass 1 fixture machinery to produce a `regions.parquet`, create a tiny synthetic `segmentation_training_datasets` + `segmentation_training_samples` set, run the training worker synchronously, assert model persisted, run the event segmentation worker synchronously against the new model + the Pass 1 fixture, `GET /segmentation-jobs/{id}/events`, assert at least one event returned, `DELETE` the job and assert parquet cleanup
- [ ] Uses the same mock Perch + mock classifier setup from the Pass 1 tests where applicable
- [ ] Runs end-to-end in < 30 seconds on CPU

---

## Verification

Run in order after all tasks:

1. `uv run ruff format --check src/humpback/ tests/ scripts/bootstrap_segmentation_dataset.py alembic/versions/044_segmentation_training_tables.py`
2. `uv run ruff check src/humpback/ tests/ scripts/bootstrap_segmentation_dataset.py alembic/versions/044_segmentation_training_tables.py`
3. `uv run pyright src/humpback/call_parsing src/humpback/ml src/humpback/workers/segmentation_training_worker.py src/humpback/workers/event_segmentation_worker.py src/humpback/workers/queue.py src/humpback/services/call_parsing.py src/humpback/schemas/call_parsing.py src/humpback/api/routers/call_parsing.py src/humpback/models/segmentation_training.py scripts/bootstrap_segmentation_dataset.py tests/`
4. `uv run alembic upgrade head`
5. `uv run pytest tests/`
6. Manual smoke: create a synthetic fixture, run the smoke test path manually end-to-end, confirm `events.parquet` is produced and readable via the API
