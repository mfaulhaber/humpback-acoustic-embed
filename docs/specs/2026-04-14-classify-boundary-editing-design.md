# Boundary Editing in Classify Review

**Date**: 2026-04-14
**Status**: Approved

## Problem

When reviewing classified events (Pass 3), users frequently notice incorrect event
boundaries but cannot fix them without leaving the classify review, navigating to
the segment review for the upstream segmentation job, finding the right event, and
correcting it there. This friction means boundary corrections are often skipped,
degrading both segmentation and classification model quality.

## Goals

1. Allow boundary editing (adjust, add, delete) directly in the classify review UI
2. Flow boundary corrections to the upstream segmentation job's correction table
3. Ensure corrected boundaries are used by both classification inference and
   classifier feedback training
4. Surface an indicator in the segment training UI when new boundary corrections
   are available for dataset extraction

## Non-Goals

- Segmentation model retraining from the classify review (remains in segment review)
- Modifying `events.parquet` or `typed_events.parquet` files (immutability preserved)
- Multi-select boundary editing or batch operations

## Design

### 1. Backend — Shared Correction Overlay Utility

A new function `load_corrected_events(session, segmentation_job_id, storage_root)`
in `call_parsing/segmentation/extraction.py`:

1. Reads `events.parquet` from the segmentation job directory
2. Queries `event_boundary_corrections` for that job
3. Calls the existing `apply_corrections()` to merge them
4. Returns the corrected event list

If no boundary corrections exist, returns the original events unchanged.

Two callers adopt this function:

- **`event_classification_worker.py`** — replaces the current direct
  `read_events()` call during inference. Added events are classified; deleted
  events are excluded.
- **`event_classifier_feedback_worker.py`** — uses corrected boundaries when
  cropping audio for training data assembly. Type labels are resolved as today
  (type correction wins, then above-threshold inference, then excluded). Added
  events with no type label are excluded from classifier training.

**ADR-054** documents: corrections are applied as read-time overlays via a shared
utility; parquet artifacts remain immutable inference snapshots; all downstream
consumers must use the shared loader rather than reading parquet directly.

### 2. Frontend — Boundary Editing in ClassifyReviewWorkspace

#### Passive Boundary Editing (Selected Event Only)

- Drag handles appear only on the currently selected event's bar in
  `EventBarOverlay`
- Dragging a handle adjusts `start_sec` or `end_sec`; the spectrogram and detail
  panel update immediately
- Delete key removes the selected event (boundary correction with
  `correction_type: "delete"`); navigation advances to the next event

#### Right-Click to Add

- Right-click on empty space in the spectrogram/event bar area opens a context
  menu with "Add event"
- Creates a new event at the click position with a default duration (~1 second)
- The new event is selected with drag handles visible; user adjusts boundaries
- No type assigned — event shows as unlabeled in the type palette and detail panel

#### Pending State Tracking

- New `pendingBoundaryCorrections` map alongside the existing
  `pendingTypeCorrections` map
- The unsaved counter reflects both maps combined
- Boundary-corrected events show a visual indicator (subtle border change for
  adjusted events; deleted events fade out before removal)

#### Save Behavior

Single Save button persists both correction types in parallel:

- `POST /classification-jobs/{id}/corrections` — type corrections (existing)
- `POST /segmentation-jobs/{segJobId}/corrections` — boundary corrections
  (existing endpoint)

No new API endpoints required. Save failure on either type rolls back UI state
for that correction type.

#### Keyboard Shortcuts

- `Delete` / `Backspace` — delete selected event (boundary delete)
- All existing shortcuts preserved: arrow keys navigate, space plays, number/letter
  keys for type palette

### 3. Classification Inference — Consuming Corrected Boundaries

`event_classification_worker.py` replaces `read_events(events_path)` with
`load_corrected_events(session, segmentation_job_id, storage_root)`.

- Added events are classified like any other
- Deleted events are excluded from inference
- Adjusted events use corrected boundaries for audio cropping

### 4. Classifier Feedback Training — Consuming Corrected Boundaries

`event_classifier_feedback_worker.py` calls `load_corrected_events()` for the
upstream segmentation job when assembling training samples.

- Event boundaries come from the corrected set
- Type labels resolved as today: type correction > above-threshold inference >
  excluded
- Added events with no type correction have no label and are excluded from
  classifier training (no label = no training signal)
- Boundary-deleted events are removed before type resolution; any type correction
  on a deleted event is harmless dead weight

### 5. Boundary Delete and Type Correction Interaction

A boundary delete on the segmentation job removes the event from the corrected
event list. No corresponding type correction is created. If a type correction
already exists for the deleted event, it becomes moot — the event is gone before
type resolution runs.

### 6. Segment Training "Update Available" Indicator

**Detection logic**: Compare the most recent
`event_boundary_corrections.updated_at` for a segmentation job against the
`created_at` of the latest `segmentation_training_dataset` that consumed samples
from that job. If corrections are newer, an update is available.

**API surface**: Extend the existing
`GET /call-parsing/segmentation-jobs/with-correction-counts` endpoint to include
a `has_new_corrections` boolean and `latest_correction_at` timestamp.

**UI treatment**: A badge or text indicator ("New corrections available") on the
segment training dataset table. Clicking it pre-populates the "create dataset
from corrections" action.

### 7. Retrain Button Scope

The Retrain button in classify review retrains the classifier only (existing
behavior). Boundary corrections are persisted to the segmentation job and
available for segmentation retraining from the segment review workflow. The
"update available" indicator (§6) surfaces this to the user.
