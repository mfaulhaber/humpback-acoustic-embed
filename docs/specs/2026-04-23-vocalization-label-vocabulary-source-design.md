# Vocalization Label Vocabulary Source

**Date:** 2026-04-23
**Status:** Approved

## Problem

The Vocalization Labeling page does not show all vocalization types defined in
the Vocabulary Manager (Training page). New types added to `vocalization_types`
are invisible on the Labeling page until they appear in a trained model's
`vocabulary_snapshot` or have been manually applied to at least one row.

The same issue affects the classifier detection timeline viewer's label popover
(`VocLabelPopover`), which merges three vocabulary sources but still misses
types that haven't been used yet.

## Decision

Use the `vocalization_types` table as the single authoritative source for
label choices in both the Labeling page and the timeline label popover.
Drop the `/labeling/label-vocabulary` query (historical distinct labels)
from both UIs. The `(Negative)` label remains available as a separate option
in both locations, unchanged.

**Option B was chosen over:**
- **A (merge all three sources):** More permissive but accumulates stale/orphaned
  labels and has no single source of truth.
- **C (types + model vocab):** Unnecessary complexity; if types are the authority,
  the model vocab adds nothing.

## Affected Components

### 1. LabelingWorkspace.tsx

**Current behavior:** Merges `vocabulary` (from active model's
`vocabulary_snapshot`) and `labelVocab` (from `/labeling/label-vocabulary`
endpoint) into `allTypes`.

**New behavior:** Fetch `vocalization_types` via `useVocalizationTypes()` and
use the type names as the sole source for `allTypes`. Remove
`useLabelVocabulary()` call. The dropdown populates from `vocalization_types`
regardless of whether an inference job is selected.

### 2. VocLabelPopover.tsx

**Current behavior:** Merges `allTypeNames` (color palette), `vocTypes`
(from `useVocalizationTypes()`), and `labelVocab` (from
`useLabelVocabulary()`) into the available label set.

**New behavior:** Use `vocTypes` from `useVocalizationTypes()` as the sole
source. Keep `allTypeNames` from saved/pending labels for color palette
continuity (so orphaned labels on existing rows still render with their
assigned color). Remove `useLabelVocabulary()` call.

### 3. Classifier LabelingTab.tsx

**Current behavior:** Uses `useLabelVocabulary()` for autocomplete suggestions
when manually typing labels.

**New behavior:** Fetch `vocalization_types` via `useVocalizationTypes()` and
map to type names for the suggestion list. Remove `useLabelVocabulary()` call.

### 4. Backend

No changes. The `/vocalization/types` GET endpoint already returns all
defined types. The `/labeling/label-vocabulary` endpoint remains in the
codebase (no deletion) but is no longer called from these two UIs.

## Edge Cases

- **Orphaned labels:** If a row has a label from a deleted type, it still
  displays on that row (read path unchanged). The label cannot be added to
  new rows. This is the intended behavior.
- **No inference job selected:** The Labeling page still shows all
  `vocalization_types` in the dropdown, since the source no longer depends
  on model vocabulary.
- **(Negative) handling:** Unchanged in both locations. It is excluded from
  the type list and rendered as a separate option.

## Testing

- Verify all `vocalization_types` appear in the Labeling page dropdown.
- Verify all `vocalization_types` appear in the timeline popover dropdown.
- Verify `(Negative)` still works in both locations.
- Verify orphaned labels display on existing rows but are not in the dropdown.
- Verify adding/removing labels still works end-to-end in both UIs.
