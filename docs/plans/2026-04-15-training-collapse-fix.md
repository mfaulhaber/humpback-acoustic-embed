# Event Classifier Training Collapse Fix — Implementation Plan

**Goal:** Fix the coordinate mismatch that causes 100% empty crops in feedback training, and add targeted guards to prevent silent failures.
**Spec:** [docs/specs/2026-04-15-event-classifier-training-collapse-fix-design.md](../specs/2026-04-15-event-classifier-training-collapse-fix-design.md)

---

### Task 1: Update AudioLoader type and fix dataset cropping

**Files:**
- Modify: `src/humpback/call_parsing/event_classifier/dataset.py`

**Acceptance criteria:**
- [ ] `AudioLoader` type alias changed from `Callable[[Any], np.ndarray]` to `Callable[[Any], tuple[np.ndarray, float]]`, with docstring matching the convention in `inference.py`'s `EventAudioLoader`
- [ ] `EventCropDataset.__getitem__` unpacks `(audio, ctx_start)` from the loader
- [ ] Crop indices computed as `(sample.start_sec - ctx_start) * sr` and `(sample.end_sec - ctx_start) * sr`
- [ ] Existing fallback for `end_sample <= start_sample` preserved

**Tests needed:**
- Unit test: construct an `EventCropDataset` with a mock loader that returns `(audio, ctx_start)` where `ctx_start` is a large epoch value. Verify the crop extracts the correct audio segment using relative offsets.
- Unit test: verify that when `ctx_start=0.0`, behavior is unchanged from the file-based case.

---

### Task 2: Update feedback worker audio loader to return context offset

**Files:**
- Modify: `src/humpback/workers/event_classifier_feedback_worker.py`

**Acceptance criteria:**
- [ ] `_build_audio_loader`'s inner `_load` function returns `(audio, ctx_start)` instead of bare `audio`
- [ ] Type annotation on `_build_audio_loader` updated to return the new `AudioLoader` type (or `Any` if currently untyped)

**Tests needed:**
- Covered by Task 1's integration-level test — the loader is simple enough that the dataset test exercises the contract.

---

### Task 3: Add crop guard in dataset

**Files:**
- Modify: `src/humpback/call_parsing/event_classifier/dataset.py`

**Acceptance criteria:**
- [ ] After computing `crop`, check `len(crop) < self.feature_config.n_fft`
- [ ] If too short, log a warning with sample details (`start_sec`, `end_sec`, crop length, minimum required)
- [ ] Return a zero-feature tensor `(1, n_mels, 1)` and the correct label vector
- [ ] Add `import logging` and `logger = logging.getLogger(__name__)` if not present

**Tests needed:**
- Unit test: mock loader returns audio shorter than n_fft for the event's bounds. Verify warning is logged and zero-feature tensor is returned with correct shape and label.

---

### Task 4: Add constant spectrogram warning in feature extraction

**Files:**
- Modify: `src/humpback/call_parsing/segmentation/features.py`

**Acceptance criteria:**
- [ ] In `normalize_per_region_zscore`, when `std < eps`, log a warning with the mean and std values
- [ ] No behavior change — normalization proceeds as before
- [ ] Add `import logging` and `logger = logging.getLogger(__name__)` if not present

**Tests needed:**
- Unit test: pass an all-zeros spectrogram to `normalize_per_region_zscore`. Verify the warning is logged and the function still returns an all-zeros result.

---

### Task 5: Add gradient clipping and per-epoch logging to training loop

**Files:**
- Modify: `src/humpback/ml/training_loop.py`
- Modify: `src/humpback/call_parsing/event_classifier/trainer.py`

**Acceptance criteria:**
- [ ] `fit()` gains a `grad_clip: float | None = None` parameter
- [ ] When `grad_clip` is set, `_run_epoch` calls `torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)` after `loss.backward()` and before `optimizer.step()`
- [ ] `grad_clip` is threaded from `_run_epoch` parameter or closure — whatever is cleanest given `_run_epoch`'s current signature
- [ ] `fit()` logs at INFO level after each epoch: `Epoch {n}/{total} — train_loss={:.4f}` (with `, val_loss={:.4f}` when val_loader is present)
- [ ] `train_event_classifier` in `trainer.py` passes `config.grad_clip` to `fit()`
- [ ] Add `import logging` and `logger = logging.getLogger(__name__)` to `training_loop.py` if not present

**Tests needed:**
- Unit test: train a tiny model for 2 epochs with `grad_clip=0.5`. Verify gradient norms are clipped (check parameter gradients after a manual backward pass, or verify `clip_grad_norm_` is called via mock).
- Unit test: verify per-epoch log lines are emitted at INFO level.

---

### Verification

Run in order after all tasks:
1. `uv run ruff format --check src/humpback/call_parsing/event_classifier/dataset.py src/humpback/call_parsing/segmentation/features.py src/humpback/ml/training_loop.py src/humpback/call_parsing/event_classifier/trainer.py src/humpback/workers/event_classifier_feedback_worker.py`
2. `uv run ruff check src/humpback/call_parsing/event_classifier/dataset.py src/humpback/call_parsing/segmentation/features.py src/humpback/ml/training_loop.py src/humpback/call_parsing/event_classifier/trainer.py src/humpback/workers/event_classifier_feedback_worker.py`
3. `uv run pyright src/humpback/call_parsing/event_classifier/dataset.py src/humpback/call_parsing/segmentation/features.py src/humpback/ml/training_loop.py src/humpback/call_parsing/event_classifier/trainer.py src/humpback/workers/event_classifier_feedback_worker.py`
4. `uv run pytest tests/`
