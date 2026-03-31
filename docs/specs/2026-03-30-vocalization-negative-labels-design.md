# Vocalization Negative Labels & Training Fix

**Date**: 2026-03-30
**Status**: Approved

## 1. Problem

Three issues with the vocalization labeling and training system:

1. **No explicit negative labeling** — Users can label windows with vocalization
   types but cannot mark a window as "confirmed noise/background." There is no way
   to build a curated pool of negatives for training.

2. **Incorrect negative construction** — The training data assembly
   (`vocalization_worker.py` line 149) includes ALL unlabeled windows as implicit
   negatives via `labels_by_utc.get(utc_key, set())`. If a detection job has 1000
   windows but only 50 are labeled, the remaining 950 unlabeled (unreviewed)
   windows all become negatives. This dilutes training signal with unreviewed data.

3. **Model detail display bugs** — The frontend model metrics table references
   `m.n_samples` but the trainer stores `n_positive` and `n_negative`, so the
   Samples column is always empty. There is no dedicated Negatives column.

## 2. Solution

### 2.1 Explicit "(Negative)" Label

Add "(Negative)" as a special label in the vocalization labeling UI. It is stored
as a regular `vocalization_labels` row with `label = "(Negative)"`. It is NOT a
vocalization type — it is a reserved label string.

**Mutual exclusivity**: "(Negative)" and type labels cannot coexist on the same
window. Adding "(Negative)" removes any existing type labels. Adding a type label
removes any existing "(Negative)" label. Enforced in both the API and the frontend.

### 2.2 Fix Training Data Assembly

Change the training data assembly from detection jobs to only include explicitly
labeled windows:

- **Before**: All embedding rows included; unlabeled get empty set (implicit negatives)
- **After**: Only rows with at least one label in `labels_by_utc` are included;
  unlabeled rows are skipped entirely

For included windows:
- Type labels (e.g., `{"whup", "moan"}`) pass through as-is
- `"(Negative)"` label is converted to empty set `{}` so the binary relevance
  trainer treats it as negative for all types

### 2.3 Model Detail Fixes

- **Samples column**: Display `n_positive + n_negative` (was referencing nonexistent `n_samples`)
- **New Negatives column**: Show `n_negative` per type in the per-class metrics table

## 3. Data Model

No schema changes. No migrations. "(Negative)" uses the existing `vocalization_labels`
table with `label = "(Negative)"`.

The vocalization types API (`POST /vocalization/types`) rejects creating a type
named "(Negative)" to prevent collision.

## 4. API Changes

### 4.1 Label Save Mutual Exclusivity

`POST /labeling/vocalization-labels/{job_id}`:
- When `label = "(Negative)"`: delete any existing type labels on that (start_utc, end_utc) window
- When `label` is a type name: delete any existing "(Negative)" label on that window

### 4.2 Vocabulary Guard

`POST /vocalization/types`:
- Reject requests where `name` matches "(Negative)" (case-insensitive)

## 5. Frontend Changes

### 5.1 Labeling UI

In `LabelingRow`, "(Negative)" appears as a special option:
- Listed at the bottom of the `+` popover, separated by a divider
- Styled with red theme (`bg-red-100 text-red-800 border-red-200`)
- Mutual exclusivity in pending state: adding "(Negative)" clears pending type
  adds and marks existing type labels for removal; adding a type clears pending
  "(Negative)"

### 5.2 Model Detail

In `VocalizationModelList`:
- Fix Samples column: `n_positive + n_negative` instead of `m.n_samples`
- Add Negatives column: `n_negative` per type

## 6. Training Pipeline Changes

In `vocalization_worker.py`, detection job data assembly loop:
- Skip rows where `utc_key` has no entry in `labels_by_utc`
- For rows with "(Negative)" in their label set, convert to empty set `{}`
- No changes to `vocalization_trainer.py` — it already handles empty sets correctly

## 7. Testing

- Training data assembly: verify unlabeled windows are excluded
- Training data assembly: verify "(Negative)" windows become empty-set negatives
- Training data assembly: verify type-labeled windows remain positives
- API: mutual exclusivity enforcement on label save
- API: vocalization type name guard against "(Negative)"
- Frontend: model detail shows correct sample counts and negatives column
