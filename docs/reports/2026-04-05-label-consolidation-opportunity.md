# Label Editing Consolidation Opportunity

**Date**: 2026-04-05
**Context**: Bug fix for duplicate labels on Training Data page

## Finding

The Training Data page (`TrainingDataView.tsx`) and the Labeling page (`LabelingWorkspace.tsx`) independently implement the same pending-label editing pattern:

- `pendingAdds`: `Map<key, Set<string>>` for labels to create
- `pendingRemovals`: `Map<key, Set<string>>` for labels to delete
- `isDirty` / `pendingChangeCount` derived state
- Sticky save bar with Cancel/Save buttons
- Batch save via `Promise.all` (creates + deletes), then query invalidation

## Bug That Prompted This

`TrainingDataView` stored label **names** in `pendingRemovals`, but the delete API expects label **IDs** (UUIDs). Every removal silently failed. Combined with no server-side duplicate guard, edit cycles accumulated duplicate labels.

`LabelingWorkspace` was implemented correctly from the start — its API returns label objects with IDs, and `pendingRemovals` stores those IDs.

The bug existed because the same pattern was reimplemented independently with a subtle difference in what gets stored in the removal set.

## Consolidation Scope

A shared implementation would cover:

1. **`usePendingLabels` hook** — encapsulates pendingAdds, pendingRemovals, isDirty, pendingChangeCount, add/remove/cancel/save helpers. Parameterized by key type (number for TrainingDataView row_index, string for LabelingWorkspace row keys) and API functions.

2. **`SaveBar` component** — the sticky save/cancel bar with change count. Currently copy-pasted between both views with identical markup.

3. **Label badge rendering** — saved labels (click to mark removal), pending adds (ring highlight, click to undo), pending removals (strikethrough + X). Same visual pattern in both components.

## Files Involved

| Component | File |
|-----------|------|
| Training Data page | `frontend/src/components/vocalization/TrainingDataView.tsx` |
| Labeling page | `frontend/src/components/vocalization/LabelingWorkspace.tsx` |

## Priority

Low — both pages now work correctly. This is a code quality improvement to prevent future divergence bugs, not a functional issue.
