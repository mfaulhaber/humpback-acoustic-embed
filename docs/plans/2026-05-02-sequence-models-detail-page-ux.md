# Sequence Models Detail Page UX — Implementation Plan

**Goal:** Polish both Sequence Model detail pages (collapsible panels, Motifs panel placement on Masked Transformer, motif legend + prev/next on the Timeline Card) and fix the motif example alignment strip bug where `relative_*_seconds` renders at epoch scale.

**Spec:** [docs/specs/2026-05-02-sequence-models-detail-page-ux-design.md](../specs/2026-05-02-sequence-models-detail-page-ux-design.md)

---

### Task 1: Diagnose the alignment-strip `relative_*_seconds` bug

**Files:**
- Read-only inspection: existing `occurrences.parquet` for a completed motif extraction job under the project's storage root.
- Read-only inspection: `src/humpback/sequence_models/motifs.py`, `src/humpback/schemas/sequence_models.py`, `src/humpback/api/routers/sequence_models.py`.

**Acceptance criteria:**
- [ ] Identify a real motif extraction job whose Motifs panel renders the `+1635638400.00s`-style epoch axis (the masked-transformer job in the screenshot, or any equivalently broken job in the production DB).
- [ ] Read its `occurrences.parquet` directly and record the actual on-disk values of `relative_start_seconds`, `relative_end_seconds`, `start_timestamp`, and `anchor_timestamp` for one occurrence.
- [ ] Determine which of these is the source of the bug:
  - (A) Parquet was written by an older buggy version (HEAD writer is fine; existing data is bad).
  - (B) Schema or API serializer maps `start_timestamp` into the `relative_start_seconds` field.
  - (C) `_anchor_for_occurrence` returns `0.0` (or a degenerate value) under some path, making `start - anchor == start`.
  - (D) Something else.
- [ ] Document the finding inline in this plan (edit Task 2 description below) before starting Task 2.

**Tests needed:**
- None for this task — diagnosis only. Test goes in Task 2.

---

### Task 2: Fix the alignment-strip bug

**Task 1 finding (D):** Time-domain mismatch in the motif worker.
- `decoded.parquet` `start_timestamp` is **absolute UTC epoch seconds** (built by adding `region_detection_job.start_timestamp` as `timestamp_offset` to chunk-relative starts; see `continuous_embedding_worker.py:990-1052`).
- `events.parquet` `start_sec` / `end_sec` are in the **source-audio relative timeline** (per `call_parsing/types.py:88-95`, despite the docstring saying "absolute"; the worker stores them region-local).
- `_load_event_lookup` in `motif_extraction_worker.py:65-82` builds the lookup straight from `events.parquet` without applying the offset, so `_anchor_for_occurrence` mixes domains and `start_timestamp - anchor_timestamp ≈ start_timestamp`. Float32 precision in `relative_*_seconds` then snaps the value to the nearest representable epoch (e.g. `1635638400.0`).
- Verified on disk for masked-transformer job `f0ae9b24-…`: `start_timestamp=1635646521.0`, `anchor_timestamp=8124.392`, `relative_start_seconds=1635638400.0`. Same shape on HMM job `62ce2445-…`. The audio file's UTC start is `region_detection_job.start_timestamp = 1635638400.0`.
- Fix: in the worker, look up `region_detection_job.start_timestamp` via `cej.event_segmentation_job_id → event_segmentation_job.region_detection_job_id` and pass it to `_load_event_lookup` as a `timestamp_offset` so the returned `(ev_start, ev_end)` are absolute UTC.

**Files:**
- Modify: `src/humpback/workers/motif_extraction_worker.py` (compute and apply `timestamp_offset` in `_load_event_lookup`).
- Modify: `tests/workers/test_motif_extraction_worker.py` (regression test).

**Acceptance criteria:**
- [ ] Code change addresses the root cause identified in Task 1. If Task 1 finding is (A) (stale parquet only), no production code change is required for this task — go straight to Task 3 and rely on the release note. If finding is (B), (C), or (D), make the targeted fix.
- [ ] Regression test: persist or synthesize a motif occurrence with `start_timestamp = T + 0.4`, `end_timestamp = T + 1.0`, `anchor_timestamp = T` for some realistic `T` (e.g., `1.7e9` epoch seconds). Round-trip through the same write/read path the bug uses. Assert `0 < relative_start_seconds < 1` and `0 < relative_end_seconds < 2` (i.e., values are second-scale offsets, not epoch-scale).
- [ ] Test fails on `main` (or before the fix); passes after the fix.

**Tests needed:**
- One pytest case asserting the bullet above.
- If Task 1 finding is (A) only, add a Pydantic-level test that exercises `MotifOccurrence.model_validate` from a synthetic dict with the correct field values to lock in field semantics.

---

### Task 3: New `CollapsiblePanelCard` component + unit tests

**Files:**
- Create: `frontend/src/components/sequence-models/CollapsiblePanelCard.tsx`
- Create: `frontend/src/components/sequence-models/CollapsiblePanelCard.test.tsx`

**Acceptance criteria:**
- [ ] Component accepts `title`, `storageKey`, `defaultOpen?`, `headerExtra?`, `testId?`, `children`.
- [ ] On mount, reads `localStorage[\`seq-models:panel:${storageKey}\`]`. If `"true"` or `"false"`, uses that. Otherwise falls back to `defaultOpen` (default `true`).
- [ ] Renders `<Card>` + `<CardHeader>` matching existing detail-page pattern. Header row contains a chevron toggle (▾ open / ▸ closed) plus the `title` and optional `headerExtra` slot.
- [ ] Clicking the chevron or the header row toggles open/closed and writes the new state back to `localStorage` synchronously.
- [ ] When closed, `<CardContent>` is not rendered (children are unmounted, not just hidden).
- [ ] `aria-expanded` and `aria-controls` set correctly on the toggle button.
- [ ] Component renders test ID `${testId ?? "collapsible-panel"}`.

**Tests needed:**
- Default-open render: title visible, children visible.
- Click toggle: children unmount, localStorage updated.
- Mount with localStorage pre-populated to `"false"`: starts closed.
- Mount with `defaultOpen={false}` and no localStorage entry: starts closed.
- `headerExtra` renders next to the title and isn't affected by toggle clicks (clicking inside `headerExtra` does not toggle the panel — stop propagation in `headerExtra` wrapper, OR header click handler ignores clicks inside the `headerExtra` slot).

---

### Task 4: New `MotifTimelineLegend` component + unit tests

**Files:**
- Create: `frontend/src/components/sequence-models/MotifTimelineLegend.tsx`
- Create: `frontend/src/components/sequence-models/MotifTimelineLegend.test.tsx`

**Acceptance criteria:**
- [ ] Accepts props: `selectedMotifKey`, `selectedStates`, `numLabels`, `occurrencesTotal`, `activeOccurrenceIndex`, `onPrev`, `onNext`, `palette?`.
- [ ] When `selectedMotifKey == null`, returns `null`.
- [ ] Renders the row described in the spec: `Selected motif:` label, swatch sequence with `→` separators, counter (`{idx+1} / {total}`), prev/next buttons.
- [ ] Each swatch uses `palette[state % palette.length]` (defaults to `LABEL_COLORS`).
- [ ] Prev disabled when `activeOccurrenceIndex === 0`. Next disabled when `activeOccurrenceIndex >= occurrencesTotal - 1`.
- [ ] Clicking enabled prev/next calls `onPrev` / `onNext`.
- [ ] Test IDs: `motif-timeline-legend`, `motif-timeline-legend-prev`, `motif-timeline-legend-next`, `motif-timeline-legend-counter`.

**Tests needed:**
- Returns null when `selectedMotifKey == null`.
- Renders correct number of swatches with correct background colors for `selectedStates = [23, 50]` and `numLabels = 100`.
- Counter text matches `{idx+1} / {total}`.
- Prev disabled at index 0; Next disabled at `total - 1`.
- Click handlers fire when enabled.

---

### Task 5: Lift selection state in `MotifExtractionPanel`

**Files:**
- Modify: `frontend/src/components/sequence-models/MotifExtractionPanel.tsx`
- Modify: `frontend/src/components/sequence-models/MotifExampleAlignment.tsx` (highlight controlled active occurrence row).

**Acceptance criteria:**
- [ ] Add optional props to `MotifExtractionPanel`:
  - `onSelectionChange?: (s: { motifKey: string | null; motif: MotifSummary | null; occurrences: MotifOccurrence[]; occurrencesTotal: number; activeOccurrenceIndex: number }) => void`
  - `activeOccurrenceIndex?: number`
  - `onActiveOccurrenceChange?: (idx: number) => void`
- [ ] Existing uncontrolled callers (none today, but keep backward-compat) keep working when the new props are omitted.
- [ ] Internal `activeOccurrenceIndex` defaults to 0 when motif selection changes; resets to 0 whenever `selectedMotif` changes.
- [ ] When the prop is provided, panel uses it as the source of truth and calls `onActiveOccurrenceChange` whenever the user clicks a row's Jump button (passes that row's index).
- [ ] When motif row is clicked, `setSelectedMotif(motif.motif_key)` and the panel emits `onSelectionChange` with the new motif and `activeOccurrenceIndex = 0`.
- [ ] Whenever `useMotifOccurrences` returns new data for the active motif, the panel emits `onSelectionChange` with the updated `occurrences` array and `occurrencesTotal`.
- [ ] `MotifExampleAlignment` highlights the row matching `activeOccurrenceIndex` (e.g., faint background or border ring) so the user can see which occurrence prev/next is currently sitting on.

**Tests needed:**
- Existing motif panel tests still pass.
- New unit test: render the panel in controlled mode (`activeOccurrenceIndex={2}`), verify the highlighted row is the third row and that clicking another row's Jump button calls `onActiveOccurrenceChange(<that row's index>)`.

---

### Task 6: Wire collapsible cards + motif legend into `MaskedTransformerDetailPage`

**Files:**
- Modify: `frontend/src/components/sequence-models/MaskedTransformerDetailPage.tsx`

**Acceptance criteria:**
- [ ] All non-timeline panels (Loss Curve, Run-Length Histograms, Overlay, Exemplars, Label Distribution, Motifs) wrapped with `CollapsiblePanelCard` using the storage keys `mt:loss-curve`, `mt:run-lengths`, `mt:overlay`, `mt:exemplars`, `mt:label-distribution`, `mt:motifs`.
- [ ] Top metadata Card and Token Timeline Card remain plain `<Card>`.
- [ ] Motifs Card moved directly below the Token Timeline Card. Final visible order: header, KPicker, Token Timeline, Motifs, Loss Curve, Run-Length Histograms, Overlay, Exemplars, Label Distribution.
- [ ] Page tracks one `motifSelection` state populated from `MotifExtractionPanel`'s `onSelectionChange`.
- [ ] `MotifTimelineLegend` rendered inside the Token Timeline Card's `<CardContent>`, above the `TimelineProvider`. Receives `selectedMotifKey`, `selectedStates` (parsed from motif key), `numLabels = k`, `occurrencesTotal`, `activeOccurrenceIndex`, and prev/next handlers.
- [ ] Prev/next handlers compute `nextIdx`, call `onActiveOccurrenceChange(nextIdx)` on the panel, and call `timelineHandleRef.current?.seekTo((occ.start + occ.end) / 2)` for the new occurrence.
- [ ] Existing per-row Jump and Play buttons in the panel still work.
- [ ] `MotifSection` (the wrapper around `MotifExtractionPanel`) still renders the panel inside its Card; the only change is wiring through the new selection callbacks and pulling the panel's resulting render up to the page.

**Tests needed:**
- Extend `frontend/tests/playwright/<masked-transformer-detail>.spec.ts` (find current spec; create if none): land on a job with a complete motif extraction. Click a motif row. Assert the legend appears in the Token Timeline Card. Click the Next button. Assert the counter increments and the timeline view-start changes (or the alignment-strip active row changes).
- Existing masked-transformer detail Playwright tests still pass.

---

### Task 7: Wire collapsible cards + motif legend into `HMMSequenceDetailPage`

**Files:**
- Modify: `frontend/src/components/sequence-models/HMMSequenceDetailPage.tsx`

**Acceptance criteria:**
- [ ] All non-timeline-viewer panels (State Timeline per-span, PCA/UMAP Overlay, Transition Matrix, Per-State Tier Composition, Label Distribution, Dwell-Time Histograms, State Exemplars, State Summary, Motifs) wrapped with `CollapsiblePanelCard` using the storage keys `hmm:state-timeline-per-span`, `hmm:overlay`, `hmm:transition-matrix`, `hmm:tier-composition`, `hmm:label-distribution`, `hmm:dwell`, `hmm:exemplars`, `hmm:state-summary`, `hmm:motifs`.
- [ ] Top metadata Card and HMM State Timeline Viewer Card remain plain `<Card>`.
- [ ] Existing `headerExtra` use cases preserved: Label Distribution's Refresh button passed through `headerExtra`; Per-Span State Timeline's span selector passed through `headerExtra`.
- [ ] Page tracks one `motifSelection` state populated from `MotifExtractionPanel`'s `onSelectionChange`.
- [ ] `MotifTimelineLegend` rendered inside the HMM State Timeline Viewer Card's `<CardContent>`, between `SpanNavBar` and the `TimelineProvider`. Receives `selectedMotifKey`, `selectedStates`, `numLabels = job.n_states`, `occurrencesTotal`, `activeOccurrenceIndex`, and prev/next handlers.
- [ ] Prev/next handlers compute `nextIdx`, call `onActiveOccurrenceChange(nextIdx)` on the panel, and call the existing `handleJumpToTimestamp((occ.start + occ.end) / 2)` (which already updates the active span and seeks).

**Tests needed:**
- Extend `frontend/tests/playwright/<hmm-detail>.spec.ts` similarly: land on an HMM job with a complete motif extraction, select a motif, assert legend renders, click Next, assert counter increments and the active span/timeline scroll happens.
- Existing HMM detail Playwright tests still pass.

---

### Task 8: Documentation update

**Files:**
- Modify: `docs/reference/frontend.md`

**Acceptance criteria:**
- [ ] Add a short subsection (under whatever existing component-conventions or sequence-models section is most appropriate) describing `CollapsiblePanelCard` — props, the `seq-models:panel:{storageKey}` localStorage convention, and that closed state unmounts children.
- [ ] If there's a sequence-models page-layout section, update the Masked Transformer detail page panel order to reflect the Motifs-just-below-Timeline change.

**Tests needed:**
- None.

---

### Verification

Run in order after all tasks:

1. `uv run ruff format --check src/humpback/sequence_models/motifs.py src/humpback/schemas/sequence_models.py src/humpback/api/routers/sequence_models.py tests/sequence_models/test_motifs.py tests/integration/test_motif_extraction_api.py` (only on files actually touched by Task 2)
2. `uv run ruff check` on the same set
3. `uv run pyright` on the same set
4. `uv run pytest tests/`
5. `cd frontend && npx tsc --noEmit`
6. `cd frontend && npx playwright test` (at minimum the masked-transformer detail and HMM detail specs)

### Release note

Include in PR description:

> The motif example alignment strip on Sequence Model detail pages was rendering with an absolute-epoch axis. Existing motif extraction jobs in the database keep their original (broken) `relative_*_seconds` parquet values; re-run motif extraction on any HMM Sequence or Masked Transformer job whose Motifs panel you care about to get correct alignment visuals.
