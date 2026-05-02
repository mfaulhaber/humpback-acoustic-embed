# Masked Transformer Motif UX — Design

**Date:** 2026-05-02
**Scope:** Frontend-only UX additions to the Masked Transformer sequence-model detail page (`/app/sequence-models/masked-transformer/:jobId`).

## Motivation

After clicking a motif in the Motif panel today, the only visible feedback is the legend strip at the top of the timeline and the per-occurrence "Jump"/"Play" buttons inside the right-hand alignment list. There is no way to see, at a glance, *where in the current timeline view* the selected motif occurs, and the "Play" button currently auditions a `±1s`-padded clip via a detached `<audio>` element so playback is decoupled from the timeline playhead. The conf and recon heatmap strips below the token bar are also adding visual noise that the user wants to defer for now.

## Goals

1. **Motif occurrence highlights in the timeline.** When a motif is selected in the Motif panel, every visible occurrence of that motif type in the current timeline viewport is shaded with a vertical band (left border + light fill) drawn on the spectrogram canvas. The active occurrence (the one the user last Jumped/Played) is rendered with a stronger fill and bolder left border. (The band sits on the spectrogram only, not the token bar below; both share the same `pxPerSec`, so the band visually points at the corresponding tokens. See "Detailed Design" §1 for why this is mounted inside `<Spectrogram>` rather than spanning the full column.)
2. **Motif-bounded playback.** The Motif panel's "Play" button plays the audio interval bounded *exactly* by the motif's `start_timestamp`/`end_timestamp` (no padding), routed through the shared `TimelinePlaybackHandle` so the timeline playhead tracks playback and stops at motif end.
3. **Hide conf and recon heatmap strips** in the masked-transformer timeline control. Reversible via a single constant.
4. **Show 20 motif occurrences** (up from 10) in the right-hand alignment list, single column, scrollable.

## Non-Goals

- HMM detail page is not changed. (The timeline-overlay primitive added here is opt-in and HMM can adopt it later without code changes in this feature.)
- No new backend endpoints, no schema changes, no API shape changes. `useMotifOccurrences` already returns up to 100 occurrences per request.
- No user-facing toggle for confidence strips.
- No pagination beyond the first 20 occurrences in the alignment list.
- No new keyboard shortcuts.
- No changes to the standalone "play occurrence" affordance inside `MotifExampleAlignment` beyond rerouting it through the shared playback handle (see Goal 2).

## Compose-Safety Constraints (Timeline Compound Component)

The timeline viewer is a shared compound component used by HMM detail, masked-transformer detail, Pass 1/2/3 review, hydrophone timeline, and labeling. Per CLAUDE.md §8.10, any change must preserve composability:

- The motif highlight is implemented as a **new opt-in overlay layer** (`MotifHighlightOverlay`) that consumers compose explicitly inside `<TimelineProvider>`. Existing consumers' render trees do not change.
- **No new required props** are added to `TimelineProvider`, `DiscreteSequenceBar`, `Spectrogram`, `OverlayProvider`, or `TimelinePlaybackHandle`. The handle's `play(start, duration)` contract is reused as-is.
- Motif state stays **local to the masked-transformer page** (`motifSelection` in `MaskedTransformerDetailPage`). The overlay receives it as props. The provider stays motif-agnostic.
- The new overlay reads viewport via the existing `useOverlayContext()` hook (same API region overlays already use).
- Color parity with the token bar uses the existing `labelColor(state, numLabels)` helper from `frontend/src/components/sequence-models/constants.ts`. No reaching into `DiscreteSequenceBar` internals.
- Confidence strips removal is page-local: a `SHOW_CONFIDENCE_STRIPS` boolean inside `MaskedTransformerDetailPage`'s `TimelineBody`. `ChunkConfidenceStrip` itself is untouched.
- `MotifExampleAlignment`'s standalone `<audio>` is replaced by an `onPlayMotif(start, end)` callback prop that bubbles to `MaskedTransformerDetailPage`, which calls `timelineHandleRef.current?.play(start, end - start)`. Other consumers of `MotifExampleAlignment` (today: `MotifExtractionPanel`, used on both HMM and masked-transformer detail pages) supply the new prop or fall back to the legacy standalone-audio behavior so HMM is unaffected.
- Verification includes Playwright smoke runs against the HMM detail page, Pass 3 review, and hydrophone timeline to confirm no regression in shared timeline behavior.

## Detailed Design

### 1. Motif Highlight Overlay

**New file:** `frontend/src/components/timeline/overlays/MotifHighlightOverlay.tsx`

The overlay is a sibling of the spectrogram canvas inside the timeline column, mounted in the masked-transformer `TimelineBody` next to the existing region-boundary indicators. It uses `useOverlayContext()` to read `viewStart`, `viewEnd`, `pxPerSec`, and `canvasHeight`, mirroring `RegionBandOverlay`.

**Props:**

- `occurrences: MotifOccurrence[]` — comes from `motifSelection.occurrences`.
- `activeOccurrenceIndex: number` — index into `occurrences` for the active band.
- `colorIndex: number` — the label/state index used to derive band hue.
- `numLabels: number` — palette size (k); passed to `labelColor`.

**Render rules:**

- For each occurrence whose `[start_timestamp, end_timestamp]` intersects `[viewStart, viewEnd]`, draw an absolutely-positioned band from `(start - viewStart) * pxPerSec` for `(end - start) * pxPerSec` pixels, full `canvasHeight` tall, behind the playhead (lower `zIndex` than the playhead, equal to or just above region bands).
- Hue: `labelColor(colorIndex, numLabels)`.
- Inactive: `background` at ~15% alpha, left border 1px in same hue at ~40% alpha.
- Active (`idx === activeOccurrenceIndex`): `background` at ~35% alpha, left border 2px in same hue at ~80% alpha.
- `pointerEvents: "none"` — the overlay never intercepts clicks.
- Returns `null` when `occurrences.length === 0`.

**Color index source:** Motif occurrences carry a `states: number[]` array (the n-gram). The hue used for the band is the color of the motif's first state — this matches the leftmost-cell color the user sees in the legend strip and the alignment-list mini-spectrogram. (If the user later wants per-cell color blending in the band, that is a follow-up.)

**Mounting:** `OverlayContext` is provided by `<Spectrogram>` and is bound to its own canvas (see `Spectrogram.tsx`'s `OverlayContext.Provider` at the overlay container — `width: canvasWidth, height: canvasHeight`). The overlay is therefore mounted as a child of `<Spectrogram>` (matching every other overlay's pattern: `RegionBandOverlay`, `EventBarOverlay`, etc.). This means the band covers the spectrogram canvas only — not the `DiscreteSequenceBar` below. Extending coverage into the bar would require either lifting `OverlayContext` out of `Spectrogram` (a breaking change to a shared component) or duplicating viewport math; both violate compose-safety. The bar and spectrogram share the same `pxPerSec` from `TimelineProvider`, so the band aligns with the tokens beneath it visually. The component is conditionally rendered when `motifSelection.motifKey != null` and `motifSelection.occurrences.length > 0`.

### 2. Motif-Bounded Playback

**Files modified:** `MotifExampleAlignment.tsx`, `MotifExtractionPanel.tsx`, `MaskedTransformerDetailPage.tsx`.

**Changes:**

- Add an optional `onPlayMotif?: (occurrence: MotifOccurrence) => void` prop to `MotifExampleAlignment`. When present, the "Play" button calls it instead of constructing `new Audio()`. When absent, the existing standalone-audio fallback runs (preserves HMM behavior).
- `MotifExtractionPanel` forwards a new optional `onPlayMotif` prop down to `MotifExampleAlignment`.
- `MaskedTransformerDetailPage` supplies `onPlayMotif` to `MotifExtractionPanel`. The handler:
  1. Calls `timelineHandleRef.current?.seekTo(occurrence.start_timestamp)` so the band is in view.
  2. Calls `timelineHandleRef.current?.play(occurrence.start_timestamp, occurrence.end_timestamp - occurrence.start_timestamp)`.
  3. Updates `activeOccurrenceIndex` to the played occurrence (so the active-band styling lights up).
- "Jump" button keeps its current behavior (no playback, just seek to midpoint, set active occurrence).

The `regionAudioSliceUrl(regionDetectionJobId, start, duration)` URL builder is already what `TimelineProvider` wires up via `audioUrlBuilder`, so the audio source for motif-bounded play is identical to what the timeline already serves.

### 3. Hide Conf and Recon Heatmaps

**File modified:** `frontend/src/components/sequence-models/MaskedTransformerDetailPage.tsx` (the inner `TimelineBody` component).

Add at the top of the `TimelineSection`/`TimelineBody` module scope:

```ts
const SHOW_CONFIDENCE_STRIPS = false;
```

Wrap the two `<ChunkConfidenceStrip />` instances (`mt-token-confidence-strip` and `mt-reconstruction-error-strip`) in `{SHOW_CONFIDENCE_STRIPS && (…)}`. The `tokenScores`/`reconstructionScores`/`reconstructionMax` `useMemo` hooks remain (cheap; trivially reversible by flipping the constant).

### 4. Twenty Motif Occurrences

**File modified:** `frontend/src/components/sequence-models/MotifExampleAlignment.tsx`.

- Change `const rows = occurrences.slice(0, 10);` to `const rows = occurrences.slice(0, 20);` (or expose a `maxRows` prop defaulting to 20).
- The container's outer wrapper inside `MotifExtractionPanel` (around the `<MotifExampleAlignment …/>` mount, line ~337) gets `max-h-[480px] overflow-y-auto` (or equivalent Tailwind utility) so the list scrolls within its panel rather than stretching the page. Exact `max-h` value picked during implementation to match the height of the table on the left.
- `useMotifOccurrences` is already called with `limit=100` in `MotifExtractionPanel` (line ~104), so no API change is needed.

## Test Plan

Backend: no changes, no new tests required.

Frontend:

- **Playwright (new tests in `frontend/tests/`):**
  - On the masked-transformer detail page, after the page loads with a complete job, select a motif in the panel; assert that overlay band(s) (`data-testid="mt-motif-highlight-band"`) render in the timeline column at viewport-intersecting positions, in the expected hue.
  - Click "Jump" on an occurrence; assert that band's `data-active="true"` and the playhead is near the motif midpoint.
  - Click "Play" on an occurrence; assert the timeline player is in the playing state, the audio source URL contains `start=<occ.start>` and `duration=<occ.end-occ.start>` (no `±1s` padding), and that `activeOccurrenceIndex` updates.
  - Assert that the elements `[data-testid="mt-token-confidence-strip"]` and `[data-testid="mt-reconstruction-error-strip"]` are not present.
  - Assert that the alignment list renders up to 20 rows when ≥20 occurrences exist (`data-testid="motif-example-row-19"` is present), and the container scrolls.
- **Playwright regression (existing tests):** ensure HMM detail page motif panel still works (Play button uses the standalone-audio fallback, no exception, no overlay rendered there).
- **Type check:** `cd frontend && npx tsc --noEmit`.

## Risks / Open Questions

- **Overlay z-order with the playhead.** The new overlay must sit *behind* the playhead red line and *above* the spectrogram pixels. The existing region-band overlay already solves this; we mirror its `zIndex`.
- **Color contrast at low alpha.** The band uses ~15% / ~35% alpha of the categorical palette color. Some hues (yellow/cyan) read weakly on the spectrogram background. Acceptable for a v1; we can darken inactive borders if needed.
- **Active-occurrence tracking during play.** Setting `activeOccurrenceIndex` on Play only marks the band when the user clicks Play. We do not auto-advance the active occurrence as the playhead crosses other bands — out of scope.
- **`MotifExampleAlignment` shared with HMM.** HMM detail does not pass `onPlayMotif`, so it keeps the legacy standalone-audio path. No HMM behavior change.
