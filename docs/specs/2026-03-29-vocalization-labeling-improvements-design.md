# Vocalization Labeling Page Improvements — Design Spec

**Date:** 2026-03-29
**Status:** Approved

## Goal

Improve the Vocalization / Labeling page with a source abstraction, local-state label accumulation with explicit Save/Cancel, spectrogram enhancements, and fixes to label interaction feedback and retrain visibility.

---

## 1. Source Abstraction

### Source Type

A discriminated union drives the entire labeling page:

```
LabelingSource =
  | { type: "detection_job"; jobId: string }
  | { type: "embedding_set"; embeddingSetId: string }
  | { type: "local"; folderPath: string }
```

### Source Selector UI

Replaces the current `DetectionJobPicker` with a two-part selector:

1. **Segmented toggle** (top): `Detection Jobs | Embedding Set | Local`
2. **Source-specific input** (below the toggle):
   - **Detection Jobs**: existing dropdown with hydrophone + local groups
   - **Embedding Set**: dropdown of embedding sets, display value = top-level folder from `parquet_path` (e.g., "accepted")
   - **Local**: single text input for a raw audio folder path

### Behavioral Flags

Derived from source type:
- `readonly`: `true` for `embedding_set`, `false` otherwise
- Pipeline steps:
  - `detection_job`: EmbeddingStatus -> Inference -> Labeling
  - `embedding_set`: Inference -> Labeling (embeddings already exist, readonly)
  - `local`: ProcessAudio -> EmbeddingStatus -> Inference -> Labeling

### Embedding Set Source

- Dropdown populated from existing `fetchEmbeddingSets()` API
- Display value derived from the `parquet_path` top-level folder name
- Backend already supports `source_type: "embedding_set"` for inference jobs
- Inference results are view-only: no label add/remove, no Save/Cancel bar, no retrain

### Local Source

- User enters a folder path in a text input
- Triggers the full progressive pipeline: process audio into embeddings, then inference
- EmbeddingStatusPanel adapts to work with a folder path, checking for existing processing jobs / embedding sets for that folder
- Requires a backend endpoint to find-or-create an embedding set for a folder path

---

## 2. Label Interaction Rework

### Local State Accumulation

Labels accumulate in component state, not via immediate API calls:

- `pendingAdds`: Map keyed by row identity `(start_utc, end_utc)`, value is a Set of label strings to add
- `pendingRemovals`: Map keyed by row identity, value is a Set of existing label IDs to delete
- `isDirty`: derived boolean — true when any pending changes exist

### Visual Feedback (Three States per Label)

1. **Inference-suggested** (from model predictions above threshold): muted/outline badges. Clicking promotes to a pending add — badge fills solid.
2. **Saved labels** (persisted from previous sessions): solid badges. Clicking X stages a removal — badge dims or shows strikethrough.
3. **Pending new labels** (added this session, unsaved): solid badges with a dot/unsaved indicator distinct from saved labels.

### Save / Cancel Bar

- Sticky bar at the top or bottom of the labeling workspace
- Shows: `"N unsaved changes"` | `Cancel` | `Save`
- **Save**: batch POST/DELETE of all pending changes to backend, then clear pending state and increment `labelCount` for retrain footer
- **Cancel**: discard all pending state, revert rows to last-saved labels
- Bar only visible when `isDirty` is true
- Hidden entirely for readonly sources (embedding_set)

---

## 3. Default Sort

Change `LabelingWorkspace` initial sort from `"uncertainty"` to `"score_desc"` (Score, high first).

---

## 4. Retrain Footer Fix

The `RetrainFooter` component exists but fails to appear because:
- `labelCount` is only incremented on individual label API calls, which no longer happen under the new local-state model
- The condition `labelCount === 0 && !trainingJobId` hides the footer

Fix: increment `labelCount` on successful batch Save (by the count of net new labels in that save). The retrain footer then appears naturally after the first Save with new labels.

Additionally:
- Hidden for `embedding_set` source (readonly, no labels to train on)

---

## 5. Spectrogram Enhancements

### Double-Height Detection Items

- Increase row height, making the spectrogram ~120x80px (filling the left portion of the card)
- Audio controls, score, labels, and time info stack vertically to the right

### Click-to-Expand Popup

- Clicking the spectrogram opens a Dialog with a larger rendering
- Dialog contains: large spectrogram image, time range, audio playback controls, close button
- No label editing in the popup — inspection only
- Reuse the existing spectrogram endpoint with larger requested dimensions

---

## 6. Out of Scope

- Batch label operations (select multiple rows, apply label to all)
- Search/filter within labeling results
- Multiple model comparison in inference
- New vocabulary creation during labeling (managed separately)
