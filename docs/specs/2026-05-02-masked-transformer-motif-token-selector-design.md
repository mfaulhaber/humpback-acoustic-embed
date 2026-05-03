# Masked Transformer — Motif Token Count Selector

**Status:** Design
**Date:** 2026-05-02
**Page:** `/sequence-models/masked-transformer/:jobId`
**Related ADRs:** ADR-058 (motif extraction), ADR-061 (masked-transformer)
**Related PRs:** [#162](https://github.com/mfaulhaber/humpback-acoustic-embed/pull/162), [#163](https://github.com/mfaulhaber/humpback-acoustic-embed/pull/163), [#164](https://github.com/mfaulhaber/humpback-acoustic-embed/pull/164)

## Goal

Let a reviewer surface **all motifs of a given token count** that are visible in the current timeline view, instead of inspecting motifs one at a time from the Motif panel. The reviewer picks `Token Count: 2 | 3 | 4`; every visible occurrence of every motif of that length is highlighted on the timeline, color-coded per motif, so recurring patterns become legible at a glance.

## Non-goals

- New backend endpoints or data model changes. Existing per-motif occurrence endpoint suffices.
- Motif lengths outside 2 / 3 / 4 (extraction range can produce up to 8; longer motifs remain reachable through the Motif panel).
- Persisting selector state in the URL or across page reloads.
- Changes to the HMM Sequence detail page or any other timeline consumer.

## User flow

1. Reviewer is on the Masked Transformer detail page; selector is unset on load (single-motif mode).
2. Reviewer clicks `Token Count: 3` in the legend strip above the spectrogram.
3. Any prior single-motif selection in the Motif panel is cleared.
4. The timeline highlights every occurrence of every length-3 motif whose `[start, end]` overlaps the current `[viewStart, viewEnd]`.
5. Each motif gets a deterministic color (hash of `motif_key` → palette); all rectangles render at the same uniform alpha. The "active" occurrence (driven by prev / next) is indicated by a separate dashed outline ring, not by darkening the fill.
6. Prev / next step through `visibleOccurrences` (sorted by `start_timestamp`); Play plays the bounded span of the active occurrence, matching the current motif-bounded Play behavior per-occurrence.
7. Panning / zooming the timeline updates `visibleOccurrences` live; the active index clamps to 0 if it falls outside the new visible set.
8. Clicking the active token-count button again deselects, returning to `none`. Clicking a motif row in the panel switches mode to `single` and visually deselects the toggle group.

## Architecture

### State model (page-local)

`MaskedTransformerDetailPage` owns a single discriminated union; mutual exclusion between the two highlight modes is enforced by the union itself.

```ts
type MotifSelection =
  | { kind: "none" }
  | { kind: "single"; motif: MotifSummary; occurrences: MotifOccurrence[] }
  | { kind: "byLength"; length: 2 | 3 | 4; motifs: MotifSummary[]; occurrences: MotifOccurrence[] };
```

- `activeOccurrenceIndex: number` (already exists) drives prev / next / Play in both modes.
- The `MotifExtractionPanel` receives `selection` as a prop and clears its row highlight when `kind !== "single"`.

### Composability boundaries (timeline is shared)

The timeline / spectrogram / overlay primitives are shared by HMM, masked-transformer, and call-parsing pages. All shared changes are additive and opt-in:

- `MotifHighlightOverlay` gains one optional prop:
  ```ts
  colorForMotifKey?: (motifKey: string) => string;
  ```
  Default behavior unchanged when omitted (single shared highlight color, today's behavior).
- `MotifTimelineLegend` gains one optional slot:
  ```ts
  rightSlot?: ReactNode;  // or named `tokenSelector?: ReactNode`
  ```
  Other consumers (HMM detail page) render the legend without passing it and see no change.
- The new `MotifSelection` union, the token-count selector component, and the `byLength` flatten / filter logic all live inside the masked-transformer page tree. Nothing in `frontend/src/components/timeline/` imports them.
- The selector itself does not call `useTimelineContext()`. The page reads `viewStart` / `viewEnd` once (where the legend already sits in the timeline tree) and hands the filtered `visibleOccurrences` array down to the legend and the overlay — same shape both already accept.

### Data flow

Single source of truth at the page level, fed by two existing hooks:

1. `useMotifs(jobId)` — already loaded for the Motif panel. Cached.
2. `useMotifsByLength(jobId, length)` — new wrapper that:
   - Filters cached motifs to `motif.length === length`.
   - Spawns a per-motif `useMotifOccurrences` via a small `<MotifOccurrencesLoader motifKey onLoaded />` child-component pattern (hooks cannot be called in a loop; one child per motif key satisfies the rules of hooks while keeping React Query dedup intact).
   - Returns `{ motifs, occurrences, isLoading }` with `occurrences` flattened and sorted by `start_timestamp`.

Visible-set computation lives in the page, recomputed on view changes:

```ts
const { viewStart, viewEnd } = useTimelineContext();
const visibleOccurrences = useMemo(
  () => occurrences.filter(o => o.end_timestamp >= viewStart && o.start_timestamp <= viewEnd),
  [occurrences, viewStart, viewEnd],
);
```

`activeOccurrenceIndex` is clamped to `[0, visibleOccurrences.length - 1]` on every recompute; if it would fall out of range it snaps to 0.

### UI placement

The selector is rendered to the right of the existing motif prev / next / Play controls in `MotifTimelineLegend`, above the spectrogram. Layout left → right: occurrence count · prev · next · Play · `Token Count: [2] [3] [4]`.

- Three-button toggle (shadcn `<ToggleGroup type="single">`).
- A button is disabled (with tooltip) when no motif of that length exists for the job.
- A button is disabled with a small spinner while `useMotifs` is still resolving on first load.
- Selecting an active button again deselects to `none`.
- Selecting a motif row in the panel during `byLength` mode switches the page to `single` and the toggle group reflects no active value.

### Color mapping

`colorForMotifKey(motif_key)` is deterministic:

- Hash `motif_key` (string djb2 or equivalent) → integer.
- Modulo over a fixed palette of ~12 visually distinct hues.
- All occurrences render at the same uniform alpha; no active-occurrence darkening.
- Border color matches the fill at a higher alpha.
- The active occurrence (prev / next target) is drawn with a separate 2px dashed outline ring outside the rectangle — not a fill change.

### Playback

`onPlayMotif` is reused unchanged. In `byLength` mode the bounded span is `visibleOccurrences[activeOccurrenceIndex]` instead of a single-motif occurrence; the bounded-Play handler already accepts a `[start, end]` and a stop on bound exit, so no new logic is needed. Switching modes mid-playback stops playback (existing behavior — bounded span changes).

## Edge cases

| Case | Behavior |
|---|---|
| No motifs of length N | Token-count button disabled; tooltip "No length-N motifs." |
| 0 length-N occurrences in current view | Legend shows "0 in view"; prev / next / Play disabled. Selector remains active so panning recovers them. |
| Active occurrence scrolls out of view | `activeOccurrenceIndex` clamps to 0 (first visible). |
| `useMotifs` still pending | Buttons disabled with spinner badge until resolved. |
| Mode switch mid-Play | Existing motif-bounded Play handler stops playback when the bounded span changes. |
| Lengths > 4 (extractor default `max_ngram=8`) | Out of scope. Reviewer falls back to the Motif panel for long motifs. |

## Testing

### Unit (Vitest)

- `useMotifsByLength` derives `motifsOfLength` correctly and concatenates / sorts occurrences by `start_timestamp`.
- Visible-filter helper retains overlapping occurrences and excludes ones fully outside `[viewStart, viewEnd]`.
- `colorForMotifKey` is deterministic — same input → same output across renders; distinct keys get distinct palette indices for a small fixture.

### Component (Vitest + Testing Library)

- `MotifTimelineLegend` renders the optional selector slot only when supplied (composability — HMM page sees no change).
- Toggling a token count clears any prior single-motif selection in `MaskedTransformerDetailPage`.
- Selecting a motif row in `MotifExtractionPanel` clears the active token count.
- Clicking the active token-count button again returns the page to `kind: "none"`.

### E2E (Playwright)

On a seeded masked-transformer job:

1. Load the detail page; assert toggle group has no active value.
2. Click `Token Count: 3`; assert ≥1 colored highlight rectangle is rendered.
3. Pan the timeline; assert the rendered overlay count updates.
4. Click prev / next; assert the dashed outline ring moves to a different occurrence.
5. Click Play; assert playback bounds match the active occurrence span.
6. Click `Token Count: 3` again; assert overlays disappear and the page returns to single-motif mode.
7. Open the Motif panel and click a motif row mid-`byLength` mode; assert the toggle group deselects and a single-motif highlight renders.

## Files touched

- `frontend/src/components/sequence-models/MaskedTransformerDetailPage.tsx` — new state union, fan-out hook usage, visible-filter memo, prop wiring.
- `frontend/src/components/sequence-models/MotifTimelineLegend.tsx` — new optional `rightSlot` (or `tokenSelector`) prop.
- `frontend/src/components/sequence-models/MotifTokenCountSelector.tsx` — new component (toggle group).
- `frontend/src/components/sequence-models/MotifOccurrencesLoader.tsx` — new collector child component for per-motif occurrence fetches.
- `frontend/src/api/sequenceModels.ts` — new `useMotifsByLength(jobId, length)` hook (wrapper over existing endpoints; no API changes).
- `frontend/src/components/timeline/overlays/MotifHighlightOverlay.tsx` — new optional `colorForMotifKey` prop; default unchanged.
- `frontend/src/lib/motifColor.ts` — new `colorForMotifKey` helper + palette constants.
- `frontend/src/components/sequence-models/MotifExtractionPanel.tsx` — accept `selection` prop; clear row highlight when not in `single` mode.
- Tests: new Vitest specs alongside the new components/hooks; Playwright spec extended on the masked-transformer detail page suite.

## Out of scope (deferred)

- Server-side `GET /motif_extraction_jobs/:id/occurrences?length=N` endpoint (Approach 2 in brainstorming) — revisit if profiling shows the per-motif fan-out is too chatty.
- Allowing token counts > 4 — same UI pattern would scale, but the on-screen density past 4-grams is rarely useful for review.
- URL persistence of selector state — easy follow-up if the workflow demands shareable views.
- Color-customizing the per-motif palette per user.
