# Timeline Viewer: Redraw + Overlay Clipping Fix

**Date:** 2026-05-04
**Status:** Approved (brainstorming)

## Background

Two bugs in the shared timeline viewer compound (`Spectrogram` + `TileCanvas` + overlays) surface as visual regressions during navigation. The compound is consumed by ~10 callsites covering 5 user-facing workspaces (HMM Sequence detail, Masked Transformer detail, Classify Review, Window Classify Review, Segment Review; with secondary consumers in Region Detection, Vocalization Labeling, Hydrophone, ClassifierTimeline, Training Data view). Fixes must preserve the compound contract that today lets each consumer compose its own overlays into a single shared `OverlayContext`.

### Bug 1 — Tile placeholder stuck after navigation

`TileCanvas.tsx:323–340` runs an rAF poll that calls `draw()` only while `loadingTiles.size > 0`. When the last in-flight image's `onload` fires:

- Frame N: `loadingTiles = {url}` → poll fires `draw()` → placeholder rendered (image not yet cached on this frame).
- `Image.onload` fires between frames → tile inserted into cache, URL removed from `loadingTiles`.
- Frame N+1: `loadingTiles = {}` → poll skips `draw()` → placeholder stays on screen.

Hand-scrubbing masks the bug because every drag event re-runs `draw()` via the `[draw]` effect, which finds the now-cached tile. Keyboard navigation, programmatic seek, and zoom transitions surface it.

### Bug 2 — Highlights bleed past canvas bounds

`Spectrogram.tsx:135` declares the overlay container as `<div className="absolute inset-0" style={{ width: canvasWidth, height: canvasHeight }}>` with no clip. `MotifHighlightOverlay.tsx:64` computes `x = (start - viewStart) * pxPerSec`; when an occurrence partially extends past the visible view, the rectangle's `left` or `left + width` lands outside the canvas and visually bleeds onto `FrequencyAxis` (left) or the right gutter (right). Navigation surfaces the bug because seek-to-occurrence places neighboring motifs partially out of frame.

## Goals

- Fix both bugs without changing the public API of `TileCanvas`, `Spectrogram`, or `OverlayContext` for the 6 of 8 overlays that don't render tooltips.
- Preserve tooltip rendering in `DetectionOverlay` and `VocalizationOverlay` (they intentionally extend past canvas edges to remain readable at the boundary).
- Apply the clipping fix once at the compound boundary so all 8 overlay variants — and any future overlay — inherit it without code changes.

## Non-goals

- New overlay types.
- Performance optimization beyond removing the now-unnecessary idle rAF.
- Changes to time/frequency axes, playhead, or confidence strip layout.
- Refactoring the module-level `tileCache` / `loadingTiles` singletons.

## Design

### Bug 1 — Event-driven tile redraw

Replace the rAF poll with a notification from the loader.

**Loader change (`TileCanvas.tsx`):**

- The module-level `loadTile` function already centralizes `Image.onload` for every tile load. Add a module-level `Set<() => void>` of subscribers.
- When `Image.onload` fires (after `loadingTiles.delete(url)` and `putCachedTile(url, img)`), notify every subscriber. `Image.onerror` does NOT notify — error paths should not retrigger redraws.
- Subscriber callbacks must be cheap; the redraw scheduling (rAF dedup) lives in the consumer.

**Consumer change (`TileCanvas` component):**

- Remove the `useEffect` that runs the rAF poll (lines 322–340).
- Add a `useEffect` that subscribes to the loader: on tile-loaded notification, request a redraw via a single rAF handle. If a redraw is already pending for this instance, drop the notification (dedup). Cancel any pending rAF in cleanup.
- The existing `[draw]` effect at lines 313–320 stays. It already handles redraw on prop changes (zoom, center, freq range, etc.).

**Result:**

- The frame-after-last-tile gap closes — the redraw is scheduled the moment the last `Image.onload` runs.
- No idle rAF burning a frame slot per consumer.
- All 10 callsites are unaffected — `TileCanvas`'s prop contract is unchanged.

### Bug 2 — Two-layer overlay container

Split the single overlay container at `Spectrogram.tsx:135` into two siblings:

1. **Clipped band layer** — `position: absolute; inset: 0; overflow: hidden;` sized to `canvasWidth × canvasHeight`. The `OverlayContext.Provider` and the `{children}` slot move inside this layer. Every overlay child is now clipped at the canvas edge — the fix applies uniformly to `MotifHighlightOverlay`, `RegionBandOverlay`, `RegionOverlay`, `RegionEditOverlay`, `EventBarOverlay`, `RegionBoundaryMarkers`, `DetectionOverlay`'s bands, `VocalizationOverlay`'s bands, and any future overlay.
2. **Unclipped tooltip layer** — sibling of the clipped layer, also `position: absolute; inset: 0`, no clip, higher z-index, `pointerEvents: none`. Used only as a portal target for tooltips so they remain readable at canvas edges.

`OverlayContext` gains one new field: `tooltipPortalTarget: HTMLElement | null`. The `Spectrogram` captures a ref to the unclipped layer and exposes the resolved DOM node through context.

`DetectionOverlay` and `VocalizationOverlay` render their tooltip via `createPortal(tooltipNode, ctx.tooltipPortalTarget)` when the target is non-null, falling back to inline rendering when null. The tooltip's `tooltipPos` is already computed in canvas-relative pixels, so the portal mount preserves coordinates because the tooltip layer is `inset: 0` of the same offset parent (the canvas wrapper at `Spectrogram.tsx:111`).

The 6 overlays that don't render tooltips ignore `tooltipPortalTarget`. No API change for them.

### Why two layers, not one

A single clipped container regresses the two tooltip-bearing overlays. A single unclipped container is the bug. Two siblings cleanly separate "geometry inside the spectrogram window" from "UI chrome floating above it" — the same split the `Playhead`, `ConfidenceStrip`, and `TimeAxis` already use as siblings of the overlay container today.

## Verification matrix

For each of the 5 primary use cases (HMM Sequence detail, Masked Transformer detail, Classify Review, Window Classify Review, Segment Review):

- **Bug 1:** navigate to a new region/event/zoom level; assert tile renders without sticking on placeholder.
- **Bug 2:** select an item near the right edge of view; assert the overlay band is clipped at the canvas right edge and does not bleed into the right gutter or `FrequencyAxis`.

Region Detection and Vocalization Labeling exercise Bug 2 via `RegionBandOverlay` / `VocalizationOverlay`; smoke-check those.

For tooltip preservation: hover a detection or vocalization band near the right edge; assert the tooltip renders past the canvas edge (not clipped).

## Test plan

- **Unit (`TileCanvas`)** — mock the loader; assert that completing the last in-flight tile load triggers exactly one redraw and that no rAF callback remains scheduled afterward; assert rAF dedup (two notifications in the same frame produce one redraw).
- **Unit (`Spectrogram`)** — render with a child overlay that draws a band extending past `canvasWidth`; assert the clipped layer has `overflow: hidden` and the band's overflow is not visible. Assert the unclipped tooltip layer is present and exposed via context.
- **Unit (`DetectionOverlay`/`VocalizationOverlay`)** — render with `tooltipPortalTarget` set; assert the tooltip mounts into the portal target. With `tooltipPortalTarget` null, assert it mounts inline (back-compat for tests that don't set up the layer).
- **Playwright** — smoke that the motif highlight at view edge does not visually overflow the `FrequencyAxis` on Masked Transformer detail; tile loads on navigation in HMM Sequence detail.

## Files

**Modify:**
- `frontend/src/components/timeline/TileCanvas.tsx` — replace rAF poll with event-driven redraw; add subscriber registry to module-level loader.
- `frontend/src/components/timeline/spectrogram/Spectrogram.tsx` — split overlay container into clipped + unclipped tooltip layers; pass tooltip layer ref through `OverlayContext`.
- `frontend/src/components/timeline/overlays/OverlayContext.tsx` — add `tooltipPortalTarget: HTMLElement | null` to `OverlayContextValue`.
- `frontend/src/components/timeline/overlays/DetectionOverlay.tsx` — render tooltip via portal when target present.
- `frontend/src/components/timeline/overlays/VocalizationOverlay.tsx` — render tooltip via portal when target present.
- `frontend/src/components/timeline/overlays/overlays.test.ts` — extend existing tests for portal behavior.
- `frontend/src/components/timeline/spectrogram/Spectrogram.test.ts` — extend for clipped-layer assertion.

**Create:** none.

## Risks

- **Tooltip portal back-compat:** existing unit tests that render `DetectionOverlay`/`VocalizationOverlay` outside a `Spectrogram` may pass a null `tooltipPortalTarget` (or no provider). Inline-fallback behavior preserves them.
- **Resize-handle clipping in `RegionEditOverlay`:** 4px nubs at `left: -4`/`right: -4` could be visually clipped if a region sits flush against the canvas edge. Functional impact is minimal (drag area shrinks by up to 4px on one side); accepted as a tradeoff for the broader clipping benefit.
- **Loader subscriber leaks:** every `TileCanvas` instance must unsubscribe on unmount. Verified via cleanup function in the subscription `useEffect`.
