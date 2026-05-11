# Delete Confirmation Dialogs - Design

**Date:** 2026-05-11
**Status:** Draft
**Primary domain:** Call Parsing
**Neighbor domains:** Sequence Models, Frontend Shell

## 1. Goal

Add an explicit delete confirmation dialog before every user-visible Delete
button in the Call Parsing and Sequence Models workflows, apply a shared delete
button style, and add dataset deletion to the Segment Training training dataset
table.

The experience should reduce accidental destructive actions while keeping the
existing table and review workflows fast to scan.

## 2. Reference Guidance

Use the Cloudscape "delete with simple confirmation" pattern as the content
model:

- Dialog title format: `Delete [resource type]`.
- Reassurance copy identifies what is being deleted and makes the irreversible
  nature clear.
- Consequence copy states the scope or cascading effects.
- Dialog actions are `Cancel` and `Delete`.
- The destructive confirmation action is appropriate when deletion will not
  break other infrastructure.

Reference:
https://cloudscape.design/patterns/resource-management/delete/delete-with-simple-confirmation/

## 3. Current State

Call Parsing has a mix of delete behavior:

- Some resource tables already use `BulkDeleteDialog`, imported from the
  Classifier area, for multi-select deletes.
- Several single-row delete buttons call mutations immediately without an
  intermediate dialog.
- Some model and training-job delete buttons still use native `confirm()`.
- Review workspaces have event-level delete buttons that create pending
  boundary-correction deletes rather than deleting rows immediately.
- Segment Training training datasets can be listed and used to queue training,
  but the dataset rows do not have a delete button and the API does not expose a
  dataset delete endpoint.

Sequence Models has similar direct delete actions:

- Continuous Embedding has an existing bulk dialog but row-level deletes call the
  mutation immediately.
- Event Encoder row-level deletes call the mutation immediately.

The shared `Button` component already has a `destructive` variant, but delete
buttons do not consistently use a rounded red button with white delete text.

## 4. Scope

### In scope

- Add a shared frontend delete confirmation component under
  `frontend/src/components/shared/`.
- Add a shared delete action button style so Delete buttons render as a rounded
  red button with white text.
- Replace native browser delete confirmations with the shared dialog in Call
  Parsing.
- Add dialogs around immediate row-level delete mutations in Call Parsing and
  Sequence Models.
- Move Call Parsing and Sequence Models bulk delete dialogs onto the shared
  component.
- Add Segment Training dataset deletion:
  - backend service helper,
  - API route,
  - frontend API client and query hook,
  - per-row Delete button,
  - confirmation dialog.
- Add focused backend and frontend tests.

### Non-goals

- Do not change delete behavior outside Call Parsing and Sequence Models.
- Do not change non-delete confirmations such as retrain, re-segment, discard
  unsaved changes, or navigation warnings.
- Do not add typed name-entry confirmation; this request follows the simple
  confirmation pattern.
- Do not add a database migration.
- Do not delete existing models or completed training-job history when deleting
  a Segment Training dataset.

## 5. Delete Dialog Contract

The shared component should support both single-resource and multi-resource
deletes.

Required props:

- `open`
- `onOpenChange`
- `resourceType`
- `resourceName` or `count`
- `consequence`
- `onConfirm`
- `isPending`

Dialog content:

- Title: `Delete <resource type>` or `Delete <plural resource type>`.
- Body: identify the resource by name or short id, using bold text for the
  identifier where practical.
- Consequence: explain what will be removed or marked for deletion.
- Footer: `Cancel` and `Delete`; the confirm button may show `Deleting...`
  while pending.

For event-level review deletes, the copy should reflect the actual semantics:
the event is marked for deletion as a pending correction and can be undone before
saving, but once corrections are saved downstream effective-event readers will
treat it as deleted.

## 6. Shared Button Style

Every Delete trigger and Delete confirm action in scope should use the same
visual treatment:

- Rounded box button.
- Red background.
- White text.
- Text begins with `Delete`; bulk triggers may include the selected count.
- Disabled and focus states remain accessible through the existing shared
  `Button` component behavior.

Implementation can use either a dedicated `delete` button variant or a small
shared `DeleteActionButton` wrapper. The important contract is that feature
components no longer hand-roll red ghost buttons or native `<button>` delete
styles.

## 7. Segment Training Dataset Delete Semantics

Add `DELETE /call-parsing/segmentation-training-datasets/{dataset_id}`.

Behavior:

- Return `204` when the dataset is deleted.
- Return `404` when the dataset does not exist.
- Return `409` when a queued or running segmentation training job references the
  dataset.
- Delete the `SegmentationTrainingDataset` row and its
  `SegmentationTrainingSample` rows.
- Leave completed or failed `SegmentationTrainingJob` rows and produced
  segmentation models intact.

The confirmation consequence should say that the dataset and its saved samples
will be removed, while existing models remain.

## 8. Approaches Considered

### Approach A: Shared Dialog and Shared Delete Button Style

Create a reusable delete dialog and shared delete button style, then refactor
Call Parsing and Sequence Models delete flows onto it.

Pros:

- One consistent confirmation and button pattern.
- Keeps dialog copy close to Cloudscape guidance.
- Removes ad hoc native `confirm()` from delete flows.
- Avoids importing a Classifier-specific bulk dialog into other workflows.
- Gives Segment Training dataset deletion a normal path through existing query
  invalidation.

Cons:

- Touches several table components.
- Requires small local state additions for pending delete targets.

Verdict: recommended.

### Approach B: Patch Existing Buttons In Place

Add `confirm()` or local dialog markup next to each delete button and apply
classes manually.

Pros:

- Small per-file changes.
- No shared component design needed.

Cons:

- Keeps copy and styling inconsistent.
- Encourages future delete buttons to drift again.
- Does not solve the cross-domain import of `BulkDeleteDialog`.

Verdict: reject.

### Approach C: Use Additional Typed Confirmation Everywhere

Require users to type a resource name before every delete.

Pros:

- Maximum friction for accidental deletion.

Cons:

- More friction than requested.
- Cloudscape recommends additional confirmation for higher-risk deletes, while
  these job/model/dataset deletes are bounded application resources.
- Poor fit for frequent cleanup of previous jobs.

Verdict: reject.

## 9. Recommended Design

Implement Approach A.

Create a shared delete confirmation component and shared Delete action style,
then migrate every Call Parsing and Sequence Models Delete button to:

1. Open the confirmation dialog with the selected resource metadata.
2. Keep `Cancel` side-effect-free.
3. Run the existing mutation only from the dialog's `Delete` action.
4. Preserve existing toasts, query invalidation, selection clearing, and pending
   disabled states.

For Segment Training datasets, add the missing backend delete route first, then
wire the table row button to the same shared dialog.

## 10. Test Strategy

Backend:

- Add service/router tests for Segment Training dataset delete success,
  not-found, sample cleanup, and conflict with queued/running training jobs.

Frontend:

- Add a shared dialog component test for title, consequence text, cancel, and
  confirm behavior.
- Add focused component coverage for at least one Call Parsing table and one
  Sequence Models table proving row Delete opens the dialog and does not call
  the mutation until confirmation.
- Update existing Call Parsing review tests where Delete Event now opens a
  dialog before marking a correction.
- Run TypeScript after frontend changes.

Manual verification:

- Confirm all in-scope Delete buttons are visually red rounded buttons with
  white text.
- Confirm single, bulk, and Segment Training dataset delete flows show the
  dialog and keep non-delete actions unchanged.
