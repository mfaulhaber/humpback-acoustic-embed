# Timeline Viewer Redraw + Overlay Clip Implementation Plan

**Goal:** Fix the stuck-tile-after-navigation bug and the overlay-bleeds-past-canvas bug in the shared timeline compound, preserving every existing consumer's behavior including detection/vocalization tooltips at canvas edges.
**Spec:** [docs/specs/2026-05-04-timeline-redraw-overlay-clip-design.md](../specs/2026-05-04-timeline-redraw-overlay-clip-design.md)

---

### Task 1: Replace rAF poll with event-driven redraw in TileCanvas

**Files:**
- Modify: `frontend/src/components/timeline/TileCanvas.tsx`

**Acceptance criteria:**
- [ ] Module-level loader exposes a subscriber registry (a `Set<() => void>`) and notifies all subscribers from inside `Image.onload` after `loadingTiles.delete(url)` and `putCachedTile(url, img)`.
- [ ] `Image.onerror` does not notify subscribers.
- [ ] `TileCanvas` component subscribes to the loader in a `useEffect`. The callback schedules a redraw via a single rAF handle, deduped: if a redraw is already pending for this instance, drop the notification.
- [ ] The subscription `useEffect` returns a cleanup that unsubscribes and cancels any pending rAF handle.
- [ ] The previous rAF poll (`useEffect` at lines 322–340 polling on `loadingTiles.size`) is removed.
- [ ] The existing `[draw]` effect that redraws on prop changes (zoom, center, freq range) is preserved unchanged.
- [ ] No change to `TileCanvasProps`.

**Tests needed:**
- Unit test: simulate two tile loads completing back-to-back in the same frame — assert exactly one `draw` invocation occurs (rAF dedup).
- Unit test: complete the last in-flight load and tick rAF — assert one redraw fires and no rAF callback remains scheduled.
- Unit test: an `Image.onerror` does not trigger a redraw.
- Unit test: unmounting the `TileCanvas` removes the subscription (a subsequent tile load notification produces no redraw).

---

### Task 2: Split overlay container into clipped + unclipped tooltip layers

**Files:**
- Modify: `frontend/src/components/timeline/spectrogram/Spectrogram.tsx`
- Modify: `frontend/src/components/timeline/overlays/OverlayContext.tsx`

**Acceptance criteria:**
- [ ] `OverlayContextValue` gains a new field `tooltipPortalTarget: HTMLElement | null`.
- [ ] `Spectrogram` renders two siblings inside the canvas wrapper at the location of the previous single overlay container:
  - A clipped band layer with inline style `position: absolute; inset: 0; width: canvasWidth; height: canvasHeight; overflow: hidden;` containing the `OverlayContext.Provider` and the `{children}` slot.
  - An unclipped tooltip layer (sibling) with inline style `position: absolute; inset: 0; pointer-events: none;` and a higher z-index than the band layer. Sized by `inset: 0` to match the canvas wrapper, so its origin matches the band layer.
- [ ] `Spectrogram` captures a ref to the tooltip-layer DOM node and exposes the resolved node through `OverlayContextValue.tooltipPortalTarget`. The first render passes `null`; the next render after the ref attaches passes the node (a state hook for the resolved node, set in a layout effect, is acceptable).
- [ ] All existing siblings of the overlay container (`Playhead`, `ConfidenceStrip`, `TimeAxis`, `FrequencyAxis`) remain at their current positions in the JSX tree and are unaffected.
- [ ] `useOverlayContext()` consumers continue to read `viewStart`, `viewEnd`, `pxPerSec`, `canvasWidth`, `canvasHeight`, `epochToX`, `xToEpoch` unchanged.

**Tests needed:**
- Unit test: render `Spectrogram` with a child overlay that draws a div extending past `canvasWidth`. Assert the band layer has `overflow: hidden` in its computed inline style.
- Unit test: render `Spectrogram` and assert the tooltip layer is present in the DOM as a sibling of the clipped layer with no overflow style.
- Unit test: `useOverlayContext()` inside a child returns a non-null `tooltipPortalTarget` after layout (the ref has resolved).

---

### Task 3: Route tooltips through the unclipped portal

**Files:**
- Modify: `frontend/src/components/timeline/overlays/DetectionOverlay.tsx`
- Modify: `frontend/src/components/timeline/overlays/VocalizationOverlay.tsx`

**Acceptance criteria:**
- [ ] When `useOverlayContext().tooltipPortalTarget` is non-null, the tooltip element is rendered via `createPortal` into that target. The tooltip's `tooltipPos` (already in canvas-relative pixels) is unchanged.
- [ ] When `tooltipPortalTarget` is null (e.g., rendered outside a `Spectrogram` in a unit test), the tooltip renders inline as today (back-compat).
- [ ] The band rectangles continue to render inside the band layer (no portal). Only the tooltip element moves to the portal target.
- [ ] No visual change to the tooltip itself — content, sizing, hover/leave behavior all preserved.

**Tests needed:**
- Unit test (`DetectionOverlay`): mount inside a wrapper that supplies a `tooltipPortalTarget` element; trigger the hover state; assert the tooltip DOM lives inside the portal target, not inside the band container.
- Unit test (`DetectionOverlay`): mount with `tooltipPortalTarget: null`; trigger hover; assert tooltip renders inline (existing behavior).
- Symmetric tests for `VocalizationOverlay`.

---

### Task 4: Cross-consumer Playwright smoke

**Files:**
- Modify: `frontend/e2e/` — extend the most relevant existing smoke spec(s) for the affected pages, or add a focused spec covering both bugs.

**Acceptance criteria:**
- [ ] One Playwright assertion per primary consumer (HMM Sequence detail, Masked Transformer detail, Classify Review, Window Classify Review, Segment Review) that covers at least one of the two bugs. Reuse existing fixtures where they exist; do not add full new test infrastructure.
- [ ] At least one test asserts that a motif/region highlight near the right edge of the visible window is visually clipped at the canvas edge (does not extend over the right gutter).
- [ ] At least one test asserts that after navigating to a new region/event in HMM Sequence detail, the tile renders (no stuck placeholder visible after the loader settles).
- [ ] At least one test asserts that a `DetectionOverlay` or `VocalizationOverlay` tooltip near the right edge is visible (not clipped) — exercising the unclipped tooltip layer.

**Tests needed:**
- Per acceptance criteria above; the tests are the deliverable.

---

### Verification

Run in order after all tasks:
1. `cd frontend && npx tsc --noEmit`
2. `cd frontend && npx playwright test`
3. `uv run pytest tests/` (sanity — no backend changes expected, but project gate)
4. Manual smoke (auto-mode acceptable substitute when consumers' Playwright coverage is exercised): each of the 5 primary consumers verified for both bugs per the spec's verification matrix.
