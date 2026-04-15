# Event Classifier Training Collapse Fix

**Date:** 2026-04-15
**Status:** Draft

## Problem

The first corrections-only feedback-trained event classifier model
(`event-classifier-fb-ca8a09ba`) collapsed: every event received identical
scores (~0.5 per type). Training loss was flat at ~1.25 across all 19
epochs with no learning.

### Root cause

A coordinate system mismatch in `EventCropDataset.__getitem__`
(`dataset.py:64-65`). The dataset uses `sample.start_sec` — an absolute
epoch timestamp (~1.7e9 seconds) — as a relative offset into a short
(~10 second) audio buffer. The resulting sample index is astronomically
larger than the array length, so every crop is empty, every spectrogram is
all zeros after normalization, and the model converges to a trivial
constant-output solution.

The audio loader (`_build_audio_loader` in the feedback worker) correctly
loads audio for the right time range but does not communicate the context
window's start timestamp back to the dataset. The dataset has no way to
compute relative offsets.

### Compounding factors

The pipeline has no defensive logging at any stage:

| Stage | What happens | Warning logged? |
|-------|-------------|-----------------|
| `dataset.py:66-67` crop fallback | `end_sample <= start_sample` always true | No |
| `features.py:42` zero mel | `power_to_db(zeros, ref=np.max)` → all 0.0 | No |
| `features.py:56-59` z-score | `(0 - 0) / eps = 0` for all features | No |
| `training_loop.py:84` loss | Flat loss ~1.25 every epoch | No per-epoch logging |
| Audio loader | Returns valid audio that goes unused | No |

Additionally, `EventClassifierTrainingConfig.grad_clip = 1.0` is declared
but never applied in the training loop.

## Approach

Fix the coordinate mismatch and add targeted guards at the stages that
would have caught this collapse. No comprehensive observability overhaul —
just the minimum checks that make silent failure modes visible.

## Changes

### 1. Fix the coordinate mismatch

**Files:** `event_classifier_feedback_worker.py`, `dataset.py`

Change the `AudioLoader` protocol from returning `np.ndarray` to returning
`tuple[np.ndarray, float]` — the audio array plus the context window's
start timestamp (`ctx_start`).

In `_build_audio_loader`, the `_load` function returns `(audio, ctx_start)`.

In `EventCropDataset.__getitem__`, the crop computation becomes:

```python
audio, ctx_start = self.audio_loader(sample)
sr = self.feature_config.sample_rate
rel_start = sample.start_sec - ctx_start
rel_end = sample.end_sec - ctx_start
start_sample = max(0, int(round(rel_start * sr)))
end_sample = min(len(audio), int(round(rel_end * sr)))
```

The inference path (`inference.py`) already defines `EventAudioLoader` as
`Callable[[Event], tuple[np.ndarray, float]]` with the second element
documented as `audio_start_sec`. The training dataset's `AudioLoader`
type in `dataset.py` needs to adopt this same convention. The feedback
worker's `_build_audio_loader` returns `(audio, ctx_start)` to match.

### 2. Dataset guard: reject empty/degenerate crops

**File:** `dataset.py`

After computing the crop, check that it meets a minimum length of
`n_fft` samples (2048 at current config — 0.128s at 16kHz). If too short,
log a warning with event details and return a zero-feature tensor. The
existing `collate_fn` handles variable lengths already.

```python
min_samples = self.feature_config.n_fft
if len(crop) < min_samples:
    logger.warning(
        "Event crop too short (%d samples, need %d): "
        "start_sec=%.2f end_sec=%.2f",
        len(crop), min_samples, sample.start_sec, sample.end_sec,
    )
    features = torch.zeros(1, self.feature_config.n_mels, 1)
    label = torch.zeros(self.n_types, dtype=torch.float32)
    label[sample.type_index] = 1.0
    return features, label
```

This is a safety net, not the normal path. The coordinate fix should
eliminate this case. If it fires frequently, something else is wrong and
the warnings make it visible.

### 3. Feature extraction guard: detect constant spectrograms

**File:** `features.py`

Add a warning in `normalize_per_region_zscore` when the input has zero
variance (all values identical — the exact signature of the collapse):

```python
if std < eps:
    logger.warning(
        "Constant spectrogram detected (mean=%.4f, std=%.2e) — "
        "likely silent or degenerate audio input",
        mean, std,
    )
```

No behavior change — just makes the failure mode visible in logs.

### 4. Training loop: per-epoch logging and gradient clipping

**File:** `training_loop.py`

**Gradient clipping:** `fit` gains an optional `grad_clip: float | None`
parameter. When set, `_run_epoch` calls
`torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)` after
`loss.backward()` and before `optimizer.step()`. The event classifier
trainer passes `config.grad_clip` through.

**Per-epoch logging:** `fit` logs train loss and val loss at `INFO` level
after each epoch:

```
Epoch 3/30 — train_loss=0.842, val_loss=0.917
```

One line per epoch. Flat or NaN loss becomes immediately obvious.

## Scope

### In scope

- The four changes above
- Updating the bootstrap training AudioLoader for protocol compatibility
- Unit tests: coordinate fix (relative offset computation), crop guard
  (warning on short crops), grad_clip application

### Not in scope

- Restructuring the AudioLoader contract beyond the tuple return type
- Comprehensive training diagnostics (gradient norm tracking, sample-level
  validation, pre-flight checks)
- Changes to `resolve_timeline_audio` or the upstream audio pipeline
- Rerunning the collapsed training job (separate step after fix ships)
- Changes to `ref=np.max` in `extract_logmel` — all-zeros output is
  correct for silent input; the bug is that the input shouldn't be silent
