# Masked Transformer Motif UX Implementation Plan

**Goal:** Add motif occurrence highlights to the masked-transformer timeline, route motif "Play" through the shared timeline playback handle, hide the confidence/reconstruction strips behind a constant, and show 20 motif occurrences in the alignment list.

**Spec:** [docs/specs/2026-05-02-masked-transformer-motif-ux-design.md](../specs/2026-05-02-masked-transformer-motif-ux-design.md)

---

### Task 1: Add `MotifHighlightOverlay` opt-in timeline layer

**Files:**

- Create: `frontend/src/components/timeline/overlays/MotifHighlightOverlay.tsx`
- Modify: `frontend/src/components/timeline/overlays/overlays.test.ts` (or sibling test file) — register a unit test if other overlays have one.

**Acceptance criteria:**

- [ ] New component exports `MotifHighlightOverlay` with props `{ occurrences: MotifOccurrence[]; activeOccurrenceIndex: number; colorIndex: number; numLabels: number }`.
- [ ] Reads viewport state via `useOverlayContext()` (`viewStart`, `viewEnd`, `pxPerSec`, `canvasHeight`); does not import or reach into `DiscreteSequenceBar`, `Spectrogram`, or `TimelineProvider` internals.
- [ ] For each occurrence intersecting `[viewStart, viewEnd]`, renders an absolutely-positioned `<div>` at the correct x/width/height, with `pointerEvents: "none"` and a stable `data-testid="mt-motif-highlight-band"` plus `data-active` boolean attribute.
- [ ] Inactive bands use ~15% alpha fill and a 1px left border at ~40% alpha; active band uses ~35% alpha fill and a 2px left border at ~80% alpha. Hue is derived from `labelColor(colorIndex, numLabels)` imported from `frontend/src/components/sequence-models/constants.ts`.
- [ ] Returns `null` when `occurrences.length === 0`.
- [ ] No new required props on any existing shared component; the overlay is purely additive.

**Tests needed:**

- Unit-level smoke test (Vitest if other overlays have one, otherwise covered by Task 5 Playwright tests): given a fake overlay context with known `viewStart`/`viewEnd`/`pxPerSec` and three mock occurrences (one before view, one inside, one after), the component renders exactly one band at the expected pixel offset/width.

---

### Task 2: Route motif "Play" through the timeline playback handle

**Files:**

- Modify: `frontend/src/components/sequence-models/MotifExampleAlignment.tsx`
- Modify: `frontend/src/components/sequence-models/MotifExtractionPanel.tsx`
- Modify: `frontend/src/components/sequence-models/MaskedTransformerDetailPage.tsx`

**Acceptance criteria:**

- [ ] `MotifExampleAlignment` accepts a new optional `onPlayMotif?: (occurrence: MotifOccurrence) => void` prop. When supplied, the "Play" button calls `onPlayMotif(occ)` (and does not construct `new Audio(...)`). When not supplied, the existing standalone-audio behavior is preserved unchanged.
- [ ] `MotifExtractionPanel` forwards an optional `onPlayMotif` prop down to `MotifExampleAlignment` without changing any existing prop.
- [ ] `MaskedTransformerDetailPage` provides `onPlayMotif` to `MotifExtractionPanel`. The handler:
  - calls `timelineHandleRef.current?.seekTo(occ.start_timestamp)`,
  - calls `timelineHandleRef.current?.play(occ.start_timestamp, occ.end_timestamp - occ.start_timestamp)` (no `±1s` padding),
  - updates `motifSelection.activeOccurrenceIndex` to the played occurrence's index in the current `occurrences` array.
- [ ] HMM detail page (also a consumer of `MotifExtractionPanel`/`MotifExampleAlignment`) is unaffected: it does not pass `onPlayMotif` and continues to use the standalone-audio fallback.
- [ ] No change to "Jump" button behavior.

**Tests needed:**

- Playwright (in Task 5): clicking Play on an occurrence triggers a `play` request whose audio src URL contains the exact `start_timestamp` and `end_timestamp - start_timestamp` from the occurrence (assert via network mock or via reading the `<audio>` element's `src` attribute).
- Playwright (existing HMM page test): no regression — the HMM Motif panel's Play button still produces audio playback on click, no console errors.

---

### Task 3: Mount `MotifHighlightOverlay` in the masked-transformer timeline

**Files:**

- Modify: `frontend/src/components/sequence-models/MaskedTransformerDetailPage.tsx` (the `TimelineBody` and/or `TimelineSection` components).

**Acceptance criteria:**

- [ ] Inside the masked-transformer `TimelineProvider` subtree, an `OverlayProvider` is composed (if not already present) and `<MotifHighlightOverlay />` is mounted as a sibling of the spectrogram canvas.
- [ ] The overlay only renders when `motifSelection.motifKey != null` and `motifSelection.occurrences.length > 0`.
- [ ] `colorIndex` is derived from the first state of the selected motif (`motifSelection.motif.states[0]`), and `numLabels` is the current `k`.
- [ ] `activeOccurrenceIndex` is forwarded from `motifSelection.activeOccurrenceIndex`.
- [ ] No prop or behavior changes for users that do not select a motif (overlay returns `null`).
- [ ] HMM detail page composition is not modified.

**Tests needed:**

- Playwright (in Task 5): with a motif selected, at least one band renders in the timeline column with `data-testid="mt-motif-highlight-band"`. With no motif selected, no bands render.
- Playwright (in Task 5): after clicking Jump on a different occurrence, the band whose timestamps match the new active occurrence carries `data-active="true"`, and previously active band drops to `data-active="false"`.

---

### Task 4: Hide confidence/reconstruction strips and bump occurrences to 20

**Files:**

- Modify: `frontend/src/components/sequence-models/MaskedTransformerDetailPage.tsx`
- Modify: `frontend/src/components/sequence-models/MotifExampleAlignment.tsx`
- Modify: `frontend/src/components/sequence-models/MotifExtractionPanel.tsx`

**Acceptance criteria:**

- [ ] `SHOW_CONFIDENCE_STRIPS = false` constant added near the top of the masked-transformer detail page module (or its inner `TimelineBody` module scope).
- [ ] Both `<ChunkConfidenceStrip … testId="mt-token-confidence-strip" />` and `<ChunkConfidenceStrip … testId="mt-reconstruction-error-strip" />` are wrapped in `{SHOW_CONFIDENCE_STRIPS && (…)}` so they do not render when the flag is `false`.
- [ ] The `tokenScores`/`reconstructionScores`/`reconstructionMax` `useMemo` hooks remain in place (no logic deletion).
- [ ] `MotifExampleAlignment` slices to **20** rows instead of 10 (either by changing the literal or introducing a `maxRows` prop with default `20`).
- [ ] The wrapper around `<MotifExampleAlignment />` inside `MotifExtractionPanel` (the right-hand pane of the panel) sets `max-h-[<value>] overflow-y-auto` so the list scrolls in place rather than expanding the page; pick a `max-h` that matches (or roughly matches) the height of the motifs table on the left.
- [ ] HMM detail page is unaffected (HMM path through `MotifExampleAlignment` continues to render whatever max-rows default we set; this is an explicit cross-page change to the shared component, called out in the spec).

**Tests needed:**

- Playwright (in Task 5): assert `[data-testid="mt-token-confidence-strip"]` and `[data-testid="mt-reconstruction-error-strip"]` are not present on the masked-transformer detail page.
- Playwright (in Task 5): with a motif having ≥20 occurrences in fixture data, `[data-testid="motif-example-row-19"]` exists; the alignment-list container has CSS `overflow-y: auto`.
- Playwright (HMM regression): HMM detail page renders the motif alignment list with up to 20 rows and no errors.

---

### Task 5: Playwright tests for new behavior + shared-timeline regression checks

**Files:**

- Create or extend: `frontend/tests/sequence-models-masked-transformer-motif-ux.spec.ts` (or merge into an existing masked-transformer spec file).
- Modify (if needed): existing `frontend/tests/sequence-models-*.spec.ts` for HMM regression coverage.

**Acceptance criteria:**

- [ ] New Playwright test file (or new test cases) covering all user-visible behavior added in Tasks 1–4: motif-highlight bands visible, active-occurrence styling, motif-bounded Play (no `±1s` padding), 20-row alignment list with scrolling, conf/recon strips hidden.
- [ ] Existing HMM detail page Playwright tests still pass (motif panel Play uses the standalone-audio fallback, no overlay rendered).
- [ ] Existing Pass 3 review and hydrophone timeline Playwright tests still pass (no regression in shared timeline behavior).
- [ ] All tests use stable `data-testid` selectors; no reliance on Tailwind class strings that may shift.

**Tests needed:**

- (Tests are the deliverable for this task — see the acceptance criteria.)

---

### Verification

Run in order after all tasks:

1. `cd frontend && npx tsc --noEmit`
2. `cd frontend && npx playwright test` (full frontend Playwright suite — confirms HMM/Pass 3/hydrophone regressions are clean alongside new tests).
3. `uv run pytest tests/` (sanity check; this feature touches no Python, but the project's Definition of Done in CLAUDE.md §6 requires the full suite to pass).
4. Manual smoke in the dev server (`npm run dev` from `frontend/`):
   - Open a complete masked-transformer job's detail page.
   - Select a motif → confirm light bands appear across the timeline column at occurrence positions; selected occurrence has darker shade and bolder left border.
   - Click "Play" in the right-hand list → audio plays exactly between motif start and end; playhead tracks; playback stops at end.
   - Click "Jump" → playhead seeks to motif midpoint, no audio plays.
   - Confirm conf/recon heatmap strips are absent.
   - Confirm right-hand alignment list shows up to 20 rows and scrolls when there are more.
   - Open an HMM detail page → confirm motif Play still works via standalone audio (no overlay band rendered there).
