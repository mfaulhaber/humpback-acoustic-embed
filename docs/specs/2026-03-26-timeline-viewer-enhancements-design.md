# Timeline Viewer Enhancements — Design Spec

**Date:** 2026-03-26

## Problem

The timeline viewer has three usability issues and a workflow documentation ambiguity:

1. **Inconsistent tile brightness.** Spectrogram tiles are normalized per-tile (each tile's peak becomes `vmax`), so quiet tiles appear as bright as loud tiles. Adjacent tiles have visible brightness discontinuities.
2. **Confidence strip scroll desync.** The confidence heatmap below the spectrogram positions bars by array index rather than by timestamp, so it does not scroll in sync with the spectrogram and detection overlays.
3. **Page overflow / scrollbar.** The stacked layout (header + minimap + viewport + zoom selector + playback controls) exceeds the viewport height, producing a window-level scrollbar that hides controls.
4. **Workflow doc ambiguity.** CLAUDE.md and session-plan.md brainstorming overrides imply the spec is "written but not committed on main," which is misleading — the spec should not be written to disk at all during brainstorming.

## Design

### 1. Consistent tile brightness via fixed reference level

**File:** `src/humpback/processing/timeline_tiles.py` — `generate_timeline_tile()`

Replace per-tile normalization with a fixed global reference level:

- Add `ref_db: float = -20.0` parameter. This becomes `vmax` for the colormap.
- `vmin = ref_db - dynamic_range_db` (default: -20 - 80 = -100 dB).
- Remove the two lines `vmax = float(power_db.max())` and `vmin = vmax - dynamic_range_db`.
- The `dynamic_range_db` parameter (default 80.0) still controls the visible range width.
- No changes to the caching pipeline, tile API, or frontend.
- The reference level is a parameter with a sensible default; no cross-tile coordination or pre-scan needed.

### 2. Confidence strip scroll sync

**File:** `frontend/src/components/timeline/SpectrogramViewport.tsx` — `confidenceStrip` useMemo

Fix the bar positioning to use the same timestamp-to-pixel formula as time labels and detection overlays:

- Each score `i` corresponds to time `windowStart = jobStart + i * windowSec`.
- Bar x-position: `(windowStart - centerTimestamp) * pxPerSec + canvasWidth / 2`.
- Bar width: `windowSec * pxPerSec`.
- This replaces the current index-based `barWidth = canvasWidth / visibleWindowCount` approach.
- No backend changes — the confidence data from `/timeline/confidence` is already correct.

### 3. Layout: remove Minimap, pin footer controls

**Files:** `frontend/src/components/timeline/TimelineViewer.tsx`, `frontend/src/components/timeline/Minimap.tsx`

Remove the top Minimap and pin bottom controls as a fixed footer:

- Delete `Minimap.tsx` (no other consumers).
- Remove the `<Minimap>` component and its import from `TimelineViewer.tsx`.
- The `useTimelineConfidence` query stays — its data feeds the confidence strip inside `SpectrogramViewport`.
- Wrap `ZoomSelector`, `PlaybackControls`, and `LabelToolbar` (when visible) in a footer div pinned to the viewport bottom.
- The spectrogram viewport (`flex-1`) fills the space between the header and footer.
- Outer container: `h-screen` with `overflow: hidden`.
- Footer uses `COLORS.headerBg` background and top border for visual consistency with the header.

### 4. Clarify brainstorming workflow docs

**Files:** `CLAUDE.md`, `docs/workflows/session-plan.md`

Fix the ambiguous wording about spec writing during brainstorming:

- CLAUDE.md §10.1 brainstorming overrides: change "Spec is written but not committed on main" to "Spec is NOT written to disk during brainstorming — session-plan writes and commits it on the feature branch."
- `docs/workflows/session-plan.md` preconditions: change "uncommitted on main" to clarify the spec exists as an approved design in conversation context, not as a file on disk.

## Files changed

| File | Change |
|------|--------|
| `src/humpback/processing/timeline_tiles.py` | Add `ref_db` parameter, replace per-tile normalization |
| `frontend/src/components/timeline/SpectrogramViewport.tsx` | Fix `confidenceStrip` positioning math |
| `frontend/src/components/timeline/TimelineViewer.tsx` | Remove Minimap, wrap bottom controls in pinned footer |
| `frontend/src/components/timeline/Minimap.tsx` | Delete file |
| `CLAUDE.md` | Clarify brainstorming override wording in §10.1 |
| `docs/workflows/session-plan.md` | Clarify preconditions wording |

## Testing

- **Tile brightness:** Visual verification — adjacent tiles should have consistent brightness. Existing tile rendering tests should pass with the new default `ref_db`.
- **Confidence strip:** Visual verification — scroll the spectrogram and confirm the confidence strip scrolls in sync with detection overlays.
- **Layout:** Visual verification — no window scrollbar at any zoom level or label mode state. Controls always visible.
- **Regression:** Existing Playwright tests in `frontend/e2e/` should pass. Label editing, playback, and detection overlays should be unaffected.
