# Masked Transformer — Motif Token Count Selector Implementation Plan

**Goal:** Add a `Token Count: 2 | 3 | 4` selector to the Masked Transformer detail page timeline legend that highlights every visible occurrence of every motif of the chosen length, color-coded per motif.

**Spec:** [docs/specs/2026-05-02-masked-transformer-motif-token-selector-design.md](../specs/2026-05-02-masked-transformer-motif-token-selector-design.md)

---

### Task 1: Deterministic per-motif color helper

**Files:**
- Create: `frontend/src/lib/motifColor.ts`
- Create: `frontend/src/lib/motifColor.test.ts`

**Acceptance criteria:**
- [ ] Exports `MOTIF_PALETTE` — fixed array of ~12 visually distinct hues, defined as objects with both fill (uniform alpha) and border (higher alpha) CSS color strings.
- [ ] Exports `colorForMotifKey(motifKey: string): { fill: string; border: string }` that hashes the key via djb2 (or equivalent), modulo palette length, and returns the palette entry.
- [ ] Pure function — same input always yields the same output across renders / calls.
- [ ] No imports from React, the timeline package, or anything page-specific.

**Tests needed:**
- Same key returns the same color object across multiple calls.
- A small fixture of distinct keys (e.g., `"0-1-2"`, `"3-4-5"`, `"7-2-9"`) maps to distinct palette indices when palette is large enough.
- Empty string key is handled without throwing.

---

### Task 2: `useMotifsByLength` hook + `MotifOccurrencesLoader` collector

**Files:**
- Modify: `frontend/src/api/sequenceModels.ts`
- Create: `frontend/src/components/sequence-models/MotifOccurrencesLoader.tsx`
- Create: `frontend/src/api/sequenceModels.useMotifsByLength.test.ts` (Vitest)

**Acceptance criteria:**
- [ ] `useMotifsByLength(jobId: number, length: number | null)` exported from `sequenceModels.ts`. When `length` is null the hook returns an idle empty state and performs no work.
- [ ] Hook reads from the existing `useMotifs(jobId)` cache (does not refetch the motif list) and derives `motifsOfLength` by filtering on `motif.length === length`.
- [ ] Hook returns `{ motifs: MotifSummary[]; occurrences: MotifOccurrence[]; isLoading: boolean }` where `occurrences` is the flat concatenation of every length-N motif's occurrences, sorted ascending by `start_timestamp`.
- [ ] `MotifOccurrencesLoader` is a small invisible child component that takes `motifKey`, `jobId`, and `onLoaded(motifKey, occurrences)` props, calls the existing per-motif `useMotifOccurrences` hook, and invokes the callback when the data resolves. The hook composes one loader per motif key so the rules of hooks are not violated.
- [ ] React Query dedup is preserved — selecting a length, switching to single-motif mode for one of its motifs, and switching back does not refetch occurrences for already-loaded motifs.
- [ ] No new backend endpoints introduced.

**Tests needed:**
- Given a fixture motif list with mixed lengths, `motifsOfLength` filtering returns only the requested length.
- Given multiple motifs each with a fixture occurrence array, the flattened result is sorted by `start_timestamp` and contains every occurrence exactly once.
- `isLoading` is `true` while any underlying per-motif fetch is pending and `false` once all resolve.
- When `length === null`, no fetches occur and `occurrences` is empty.

---

### Task 3: Optional color prop on `MotifHighlightOverlay`

**Files:**
- Modify: `frontend/src/components/timeline/overlays/MotifHighlightOverlay.tsx`
- Modify or create: `frontend/src/components/timeline/overlays/MotifHighlightOverlay.test.tsx`

**Acceptance criteria:**
- [ ] Adds an optional `colorForMotifKey?: (motifKey: string) => { fill: string; border: string }` prop. When omitted the overlay renders identically to today (single shared highlight color).
- [ ] When provided, each rendered occurrence rectangle uses `fill` for `backgroundColor` and `border` for `borderColor`. All rectangles render at the same uniform alpha — no per-occurrence darkening or fade.
- [ ] The `activeOccurrenceIndex` indicator changes from a fill / border emphasis to a separate dashed outline ring (e.g., 2px dashed) drawn outside the rectangle, regardless of whether a color mapper is supplied.
- [ ] Props remain otherwise backwards-compatible. No imports from `sequence-models/`.

**Tests needed:**
- Without `colorForMotifKey`, rendered rectangles match the existing color (snapshot or computed-style assertion).
- With `colorForMotifKey`, rectangles for different `motif_key` values render different `backgroundColor` values, and the same key yields the same color.
- The active occurrence renders a dashed outline ring; non-active occurrences do not.

---

### Task 4: Optional `tokenSelector` slot on `MotifTimelineLegend`

**Files:**
- Modify: `frontend/src/components/sequence-models/MotifTimelineLegend.tsx`
- Modify or create: `frontend/src/components/sequence-models/MotifTimelineLegend.test.tsx`

**Acceptance criteria:**
- [ ] Adds an optional `tokenSelector?: ReactNode` prop. When omitted the legend renders identically to today.
- [ ] When provided, the slot is rendered to the right of the existing prev / next / Play controls.
- [ ] Legend continues to consume a flat `occurrences[] + activeOccurrenceIndex` shape with no awareness of single vs. byLength modes.
- [ ] No new dependencies on page-specific state or hooks.

**Tests needed:**
- Legend without the prop matches existing output (HMM-page composability).
- Legend with the prop renders the slot content to the right of the controls (DOM order assertion).

---

### Task 5: `MotifTokenCountSelector` component

**Files:**
- Create: `frontend/src/components/sequence-models/MotifTokenCountSelector.tsx`
- Create: `frontend/src/components/sequence-models/MotifTokenCountSelector.test.tsx`

**Acceptance criteria:**
- [ ] Renders three toggle buttons labeled `2`, `3`, `4` with leading `Token Count:` label, using shadcn `<ToggleGroup type="single">` (or equivalent existing pattern in the codebase).
- [ ] Props: `value: 2 | 3 | 4 | null`, `onChange(next: 2 | 3 | 4 | null)`, `availableLengths: Set<number>`, `isMotifsLoading: boolean`.
- [ ] Clicking the currently-active button calls `onChange(null)`; clicking another button calls `onChange(thatValue)`.
- [ ] A button is disabled with a tooltip "No length-N motifs" when its value is not in `availableLengths`.
- [ ] All buttons are disabled with a small spinner badge while `isMotifsLoading` is `true`.

**Tests needed:**
- Clicking an inactive value fires `onChange(value)`; clicking the active value fires `onChange(null)`.
- Buttons not in `availableLengths` are disabled and have a tooltip; buttons in the set are enabled.
- All buttons disabled while `isMotifsLoading`.

---

### Task 6: `MaskedTransformerDetailPage` state union + wiring

**Files:**
- Modify: `frontend/src/components/sequence-models/MaskedTransformerDetailPage.tsx`
- Modify: `frontend/src/components/sequence-models/MotifExtractionPanel.tsx`
- Modify or create: `frontend/src/components/sequence-models/MaskedTransformerDetailPage.motifSelection.test.tsx`

**Acceptance criteria:**
- [ ] `MaskedTransformerDetailPage` owns a discriminated union `MotifSelection = { kind: "none" } | { kind: "single"; … } | { kind: "byLength"; length; motifs; occurrences }` plus `activeOccurrenceIndex: number`.
- [ ] When the user picks a motif row in the panel, the page sets `kind: "single"` and any prior `byLength` mode (and the selector's value) is cleared.
- [ ] When the user picks a token count from the new selector, the page sets `kind: "byLength"` and any prior single-motif row selection is cleared.
- [ ] Clicking the active token count again returns the page to `kind: "none"`.
- [ ] The page reads `viewStart` / `viewEnd` from `useTimelineContext()` and computes `visibleOccurrences = occurrences.filter(o => o.end_timestamp >= viewStart && o.start_timestamp <= viewEnd)` via `useMemo`. The same filter is applied in `single` and `byLength` modes; `none` mode has no overlay.
- [ ] `activeOccurrenceIndex` is clamped to `[0, visibleOccurrences.length - 1]`; if the prior index falls out of range it snaps to 0.
- [ ] The page passes a `colorForMotifKey` prop to `MotifHighlightOverlay` only in `byLength` mode (single mode keeps existing behavior).
- [ ] The page passes a `<MotifTokenCountSelector>` to `MotifTimelineLegend` via the new `tokenSelector` slot.
- [ ] `MotifExtractionPanel` accepts a new `selection: MotifSelection` prop and visually clears its highlighted row whenever `selection.kind !== "single"`.
- [ ] Existing motif-bounded Play wiring is reused unchanged — `onPlayMotif` is invoked with the active visible occurrence's `[start, end]` regardless of mode.

**Tests needed:**
- Clicking a motif row when in `byLength` mode flips to `single` and clears the selector value.
- Clicking a token count when in `single` mode flips to `byLength` and clears the panel row.
- Clicking the active token count returns to `none`; overlays disappear.
- Visible-filter excludes occurrences whose interval lies fully outside the current view.
- Active index clamps to 0 when the prior active occurrence scrolls out of view.

---

### Task 7: Playwright E2E for the new selector

**Files:**
- Modify or create: `frontend/tests/playwright/sequence-models-masked-transformer.spec.ts`

**Acceptance criteria:**
- [ ] On a seeded masked-transformer job, loading the detail page shows the toggle group with no active value.
- [ ] Clicking `Token Count: 3` causes ≥1 colored highlight rectangle to render in the timeline overlay.
- [ ] Panning the timeline updates the rendered overlay rectangle count.
- [ ] Clicking next moves the dashed active outline to a different occurrence.
- [ ] Clicking Play sets the audio playback bounded span to the active occurrence's `[start, end]`.
- [ ] Clicking the active `Token Count: 3` button again removes all overlays and returns to single-motif mode (toggle group has no active value).
- [ ] Selecting a motif row in the Motif panel during `byLength` mode clears the toggle group and renders a single-motif highlight.

**Tests needed:**
- Listed above as acceptance criteria; each is a discrete Playwright assertion.

---

### Verification

Run in order after all tasks (frontend-only change — no Python, no Alembic):

1. `cd frontend && npx tsc --noEmit`
2. `cd frontend && npx vitest run` (or the project's standard Vitest invocation)
3. `cd frontend && npx playwright test sequence-models-masked-transformer`
4. Manual smoke: start the dev server, open the Masked Transformer detail page on a job with motifs, exercise the selector at `Token Count: 2`, `3`, `4`, confirm overlays render with distinct per-motif colors and prev / next / Play behave as specified.
