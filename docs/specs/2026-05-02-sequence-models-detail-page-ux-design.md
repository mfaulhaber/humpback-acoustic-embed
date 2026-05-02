# Sequence Models Detail Page UX — Design

**Status:** approved (brainstorm 2026-05-02)
**Pages affected:** Masked Transformer detail, HMM Sequence detail
**Shared components touched:** `MotifExtractionPanel`, `MotifExampleAlignment`

---

## Goals

Five UI changes plus one backend bug fix:

1. Make every panel on both Sequence Model detail pages collapsible, except the
   Timeline viewer (and the top metadata Card).
2. On the Masked Transformer detail page, move the Motifs Card directly below
   the Token Timeline Card. (HMM detail page already has it there.)
3. Fix the Motif example alignment strip — the `relative_start_seconds` /
   `relative_end_seconds` axis renders with an epoch-scale max
   (`+1635638400.00s`) and zero-width occurrence bars. Backend write/read
   pipeline produces absolute epoch values where it should produce offsets
   from the motif anchor.
4. Add a motif legend inside the Timeline Card on both detail pages, showing
   the currently-selected motif's state-sequence color swatches and an
   occurrence counter (`3 / 172`).
5. Add prev/next navigation buttons inside that legend that seek the timeline
   to the previous/next occurrence of the selected motif.

## Non-goals

- No backfill of existing `motif_occurrences` parquet rows — users re-run
  motif extraction for jobs they care about. Release note covers this.
- No change to the Motifs table columns (Core/Bg/Call/etc.). Their identical
  values for masked-transformer parents are real (event-scoped CRNN inputs)
  and out of scope.
- No autoplay on prev/next — seek-only.
- No collapse of the top metadata Card on either page.

## Affected files

### New
- `frontend/src/components/sequence-models/CollapsiblePanelCard.tsx`
- `frontend/src/components/sequence-models/MotifTimelineLegend.tsx`

### Modified — frontend
- `frontend/src/components/sequence-models/MaskedTransformerDetailPage.tsx`
  — wrap non-timeline panels with `CollapsiblePanelCard`, reorder to put
  Motifs directly under Token Timeline, render `MotifTimelineLegend` inside
  the Token Timeline Card, lift motif selection into local state.
- `frontend/src/components/sequence-models/HMMSequenceDetailPage.tsx`
  — wrap non-timeline panels with `CollapsiblePanelCard`, render
  `MotifTimelineLegend` inside the HMM State Timeline Viewer Card, lift motif
  selection into local state.
- `frontend/src/components/sequence-models/MotifExtractionPanel.tsx`
  — add optional `onSelectionChange` callback and controlled
  `activeOccurrenceIndex` + `onActiveOccurrenceChange` props so the parent can
  read selection state and drive prev/next from outside.

### Modified — backend (depends on diagnosis)
- One of: `src/humpback/sequence_models/motifs.py`,
  `src/humpback/schemas/sequence_models.py`,
  `src/humpback/api/routers/sequence_models.py`. The writer's arithmetic
  (`start - anchor_timestamp` on motifs.py:511–512) appears correct in HEAD,
  so the diagnosis task may turn up a stale parquet, a schema/field mismatch,
  or a dtype precision loss. Concrete fix is contingent on that finding.

### Modified — docs
- `docs/reference/frontend.md` — add `CollapsiblePanelCard` and the
  `seq-models:panel:{storageKey}` localStorage convention.

## Design

### 1. CollapsiblePanelCard

Drop-in replacement for `<Card>` on non-timeline panels.

**Props:**
- `title: ReactNode`
- `storageKey: string` — appended to `seq-models:panel:` when persisted.
- `defaultOpen?: boolean = true`
- `headerExtra?: ReactNode` — optional right-side slot for things like the
  Refresh button on Label Distribution.
- `testId?: string`
- `children: ReactNode`

**Behavior:**
- Reads open state from `localStorage[seq-models:panel:{storageKey}]` on
  mount; falls back to `defaultOpen` if missing or unparsable.
- Header row contains a chevron toggle (▾ when open, ▸ when closed).
  Clicking the chevron — or the title row — toggles the panel and writes
  back to localStorage.
- When closed, `<CardContent>` is omitted from the DOM (not just
  `display: none`) so charts stop rendering / fetching is paused naturally
  by component unmount.
- Accessibility: header button has `aria-expanded`, `aria-controls`.

**Storage keys (per-page namespacing):**
- HMM detail: `hmm:label-distribution`, `hmm:transition-matrix`, `hmm:dwell`,
  `hmm:tier-composition`, `hmm:overlay`, `hmm:exemplars`,
  `hmm:state-summary`, `hmm:state-timeline-per-span`, `hmm:motifs`
- Masked Transformer detail: `mt:loss-curve`, `mt:run-lengths`, `mt:overlay`,
  `mt:exemplars`, `mt:label-distribution`, `mt:motifs`

**Not collapsible:**
- Top metadata Card (job header + status) on both pages
- HMM State Timeline Viewer Card
- Masked Transformer Token Timeline Card
- The KPicker (already inline, not a Card)

### 2. Motifs panel placement

On the Masked Transformer detail page, move the Motifs Card from its current
position (last) to immediately below the Token Timeline Card. Final order:

1. Header / metadata
2. KPicker
3. Token Timeline (not collapsible) — embeds `MotifTimelineLegend`
4. **Motifs (collapsible)** ← moved up
5. Loss Curve (collapsible)
6. Run-Length Histograms (collapsible)
7. Overlay (collapsible)
8. Exemplars (collapsible)
9. Label Distribution (collapsible)

HMM detail page already has Motifs directly below the Timeline Viewer Card
— no reorder there.

### 3. Alignment-strip backend bug

**Symptom:** with a real motif extraction job, the Motif example alignment
strip's right axis label reads `+1635638400.00s` and individual occurrence
bars render at zero width pinned to the right edge.

**Inspection:** `motifs.py:511–512` writes
`relative_start_seconds = start - anchor_timestamp`, which is correct.
`_anchor_for_occurrence` always returns a meaningful anchor (event midpoint
or occurrence midpoint). So either:
- the on-disk parquet was written by an earlier buggy version, or
- the read/serialize path drops the field and the frontend is reading
  `start_timestamp` instead, or
- a dtype precision issue (`float32` for ~1.6e9 values would be lossy if a
  bug ever caused the wrong field to be persisted).

**Plan:** start with a diagnostic task that loads `occurrences.parquet` from
a real completed job and dumps the actual stored values. Implement the fix
that the diagnosis points at. Add a regression test asserting
`abs(relative_start_seconds) < anchor_timestamp` (i.e., the field is small,
not absolute-epoch-scale) for a synthetic motif occurrence.

**No backfill.** Existing buggy parquet rows stay buggy until the user
re-runs motif extraction. Release note in the PR description.

### 4. MotifTimelineLegend

Component rendered inside the Timeline Card on both detail pages, between
the Card header (and any per-page nav like `SpanNavBar`) and the timeline
body.

**Props:**
- `selectedMotifKey: string | null`
- `selectedStates: number[]` — pre-parsed from the motif key
  (e.g., `"23-50"` → `[23, 50]`).
- `numLabels: number` — palette modulus (k for masked transformer, n_states
  for HMM).
- `occurrencesTotal: number`
- `activeOccurrenceIndex: number`
- `onPrev: () => void`
- `onNext: () => void`
- `palette?: string[]` — defaults to `LABEL_COLORS`.

**Render:**
- If `selectedMotifKey == null`, render `null` (layout collapses cleanly).
- Otherwise render one row inside the Card's content area:
  - Static label `Selected motif:`
  - State sequence: for each state in `selectedStates`, a colored swatch
    (12×12 px or so) using `palette[state % palette.length]`, with state
    index next to it (e.g., `[swatch] 23 → [swatch] 50`).
  - Counter: `{activeOccurrenceIndex + 1} / {occurrencesTotal}`
  - Prev (`◀`) and Next (`▶`) buttons. Disabled at index 0 and
    `occurrencesTotal - 1` respectively. No wrap-around.
- Test IDs: `motif-timeline-legend`, `motif-timeline-legend-prev`,
  `motif-timeline-legend-next`, `motif-timeline-legend-counter`.

### 5. Prev/Next behavior + wiring

`MotifExtractionPanel.tsx` extends its API to expose selection state and
accept controlled active-occurrence-index. It currently owns
`selectedMotif: string | null` internally; that becomes optionally
controlled via parent.

**New / modified props:**
- `onSelectionChange?: (s: { motifKey: string | null; motif: MotifSummary | null; occurrencesTotal: number; activeOccurrenceIndex: number }) => void`
  — fires whenever the user picks a different motif row, the occurrences
  list refetches and changes the total, or the active occurrence index
  changes.
- `activeOccurrenceIndex?: number` — controlled override of the panel's
  internal "currently-shown" occurrence highlight in
  `MotifExampleAlignment`. When omitted, panel keeps its existing
  uncontrolled behavior (defaulting to 0 on motif selection change).
- `onActiveOccurrenceChange?: (idx: number) => void` — called when the
  per-row Jump button is clicked or when the controlled prop changes via
  prev/next from the parent.

**Per-row Jump button** updates the active occurrence index to that row's
index and calls the existing `onJumpToTimestamp` callback.

**Page-level integration:**
- Each detail page tracks one `motifSelection` state object populated from
  the panel's `onSelectionChange`.
- The Timeline Card renders `MotifTimelineLegend` reading from
  `motifSelection`.
- Prev/Next handlers in the page:
  - Compute `nextIdx = clamp(idx ± 1, 0, occurrences.length - 1)`.
  - Push `nextIdx` back into the panel via `onActiveOccurrenceChange`.
  - Call `onJumpToTimestamp((occurrences[nextIdx].start + occurrences[nextIdx].end) / 2)`.
- HMM page reuses its existing `handleJumpToTimestamp` (lines 769–784) which
  also re-syncs the active span.
- Masked Transformer page reuses `timelineHandleRef.current?.seekTo`.
- The page also needs the full `occurrences` array (not just the count) to
  pull each occurrence's center timestamp. Two options:
  - **(a)** Lift the `occurrences` array up via `onSelectionChange`.
  - **(b)** Have `MotifExtractionPanel` expose prev/next imperatively via a
    handle so the page just calls `panelRef.current.prev()`.
  — Use **(a)** because the legend needs to render the counter
  (`occurrencesTotal`) anyway, and lifting the array up is symmetrical with
  how motif selection is exposed.

## Testing

### Frontend (vitest + Playwright)

- `CollapsiblePanelCard.test.tsx` (vitest):
  - Defaults to open, toggles closed on chevron click.
  - Persists state to localStorage under the supplied key.
  - On remount with the same key, restores prior state.
  - Closed state unmounts children (assert via test ID absence).
- `MotifTimelineLegend.test.tsx` (vitest):
  - Renders nothing when `selectedMotifKey == null`.
  - Renders one swatch per state with the correct
    `LABEL_COLORS[state % palette.length]`.
  - Counter renders `{idx+1} / {total}`.
  - Prev disabled at index 0; Next disabled at index `total-1`.
  - Calls `onPrev` / `onNext` otherwise.
- Playwright (extend the existing masked-transformer detail spec):
  - Land on a masked-transformer job that has a complete motif extraction
    job. Click a motif row. Assert the legend appears in the Token Timeline
    Card with the right swatches and counter `1 / N`. Click `▶`, assert
    counter advances and timeline scrolls (assert via timeline view-start
    change or selected-occurrence highlight in the alignment strip).

### Backend (pytest)

- Regression test for the alignment-strip bug: after the diagnosis-driven
  fix, persist a synthetic motif occurrence with a known anchor and assert
  the API returns `relative_start_seconds == start - anchor` (small float,
  same sign and magnitude). Test fails before fix, passes after.

## Verification gates

Per CLAUDE.md §10.2:
1. `uv run ruff format --check` on touched Python.
2. `uv run ruff check` on touched Python.
3. `uv run pyright` on touched Python.
4. `uv run pytest tests/`
5. `cd frontend && npx tsc --noEmit`
6. `cd frontend && npx playwright test` for the masked-transformer detail
   spec (and any HMM detail spec touched).

## Doc updates

Per CLAUDE.md §10.2 doc-update matrix:
- `docs/reference/frontend.md` — new `CollapsiblePanelCard` and the
  `seq-models:panel:{storageKey}` localStorage convention.
- No CLAUDE.md §9.1 update (no new capability).
- No DECISIONS.md ADR (no architecture decision; this is UX polish + a
  bugfix).
