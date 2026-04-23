# Vocalization Label Vocabulary Source — Implementation Plan

**Goal:** Use `vocalization_types` as the single source of truth for label choices in both the Labeling page and the timeline label popover.
**Spec:** [docs/specs/2026-04-23-vocalization-label-vocabulary-source-design.md](../specs/2026-04-23-vocalization-label-vocabulary-source-design.md)

---

### Task 1: Switch LabelingWorkspace to vocalization_types

**Files:**
- Modify: `frontend/src/components/vocalization/LabelingWorkspace.tsx`

**Acceptance criteria:**
- [ ] Import `useVocalizationTypes` from `@/hooks/queries/useVocalization`
- [ ] Remove `useLabelVocabulary` import and its call
- [ ] Replace `allTypes` computation: derive from `useVocalizationTypes()` response (map to `.name`, exclude `(Negative)`, sort)
- [ ] Remove `vocabulary` prop from `LabelingRow` — replace with the new `allTypes` derived from vocalization types
- [ ] Update `typeColorMap` to build from `allTypes` (vocalization types) instead of `vocabulary` (model snapshot)
- [ ] `(Negative)` option remains available unchanged

---

### Task 2: Simplify VocLabelPopover to vocalization_types

**Files:**
- Modify: `frontend/src/components/timeline/VocLabelPopover.tsx`

**Acceptance criteria:**
- [ ] Remove `useLabelVocabulary` import and its call
- [ ] Simplify `dropdownTypeNames` computation: merge `allTypeNames` (palette continuity for existing rows) with `vocTypes` names only
- [ ] Orphaned labels on existing rows still render with their color via `allTypeNames`
- [ ] `(Negative)` handling unchanged

---

### Task 3: Manual verification

**Acceptance criteria:**
- [ ] Start dev servers (`uv run uvicorn` + `npm run dev`)
- [ ] On Vocalization / Training page, confirm types listed in Vocabulary Manager
- [ ] On Vocalization / Labeling page, confirm all those types appear in the add-label dropdown
- [ ] On a classifier detection timeline, confirm all types appear in the popover dropdown
- [ ] Confirm `(Negative)` works in both locations
- [ ] Confirm adding and removing labels works end-to-end in both UIs

---

### Verification

Run in order after all tasks:
1. `cd frontend && npx tsc --noEmit`
2. `cd frontend && npx playwright test`
