# Event Encoder Piano Roll Spectrogram Strip - Design

**Date:** 2026-05-19
**Status:** Approved
**Primary domain:** Sequence Models
**Neighbor domains:** Signal Timeline, Frontend Shell

## 1. Goal

Add a synchronized spectrogram timeline strip to the dedicated Event Encoder
piano roll page. The strip gives researchers acoustic context while they use
the piano roll's token-by-frequency view, without replacing the piano roll's
smooth time zoom, frequency zoom, token filtering, event selection, or
playback behavior.

The strip reuses existing timeline spectrogram tiles from the Call Parsing
region job that already backs Event Encoder timeline playback and review.

## 2. Motivation

The Event Encoder piano roll is optimized for seeing token families, timing
patterns, F0 ranges, and slope character. The existing Event Encoder detail
timeline is better for verifying events against the spectrogram. Moving between
the two views interrupts review when a researcher wants quick acoustic context
for the same visible time range.

A spectrogram strip below the piano roll keeps the current piano roll workflow
intact while making it easier to correlate token rectangles with visible
energy, harmonics, silences, and broadband regions.

## 3. Scope

### In Scope

- Add a visible, collapsible spectrogram strip to
  `/app/sequence-models/event-encoder/:jobId/piano-roll`.
- Render the strip from existing PCEN-normalized Call Parsing region timeline
  tiles.
- Keep the piano roll's `timeRange` as the single source of truth for the strip
  viewport.
- Support smooth piano roll zoom by continuously scaling the selected tile LOD
  and crossfading when the chosen LOD changes.
- Align the strip's horizontal plot area with the piano roll plot area.
- Use the strip's bottom time ticks as the single shared horizontal time axis.
- Draw a strip playhead synchronized to the piano roll's visible playhead.
- Allow strip drag and wheel interactions to update the same piano roll
  `timeRange`.
- Let the strip frequency range follow the piano roll frequency range.
- Preserve current Event Encoder timeline, piano roll, token, and playback
  semantics.
- Remove the piano roll minimap control.

### Non-goals

- No backend API changes.
- No new spectrogram artifacts or Event Encoder artifacts.
- No change to Event Encoder tokenization, descriptors, or projections.
- No replacement of the piano roll canvas with `TimelineProvider`.
- No zoom-level buttons on the piano roll page.
- No editing of events, labels, tokens, or spectrogram content.

## 4. Existing System

The existing Event Encoder detail timeline uses `TimelineProvider`,
`Spectrogram`, `TileCanvas`, and `EventEncoderTokenOverlay` to show tokenized
events over a spectrogram. This works well for the detail page because the
timeline provider owns discrete zoom presets, center timestamp, drag panning,
keyboard shortcuts, and playback state.

The piano roll intentionally does not use `TimelineProvider`. It owns local
state for:

- Smooth arbitrary `timeRange`.
- Smooth wheel-based time zoom centered on the cursor.
- Shift-wheel frequency zoom.
- Token filtering and event selection.
- Playback through shared `usePlayback`.

The spectrogram strip should not introduce a second viewport owner. It should
reuse the low-level tile renderer while remaining controlled by the piano roll
state.

## 5. Proposed Approach

Create an `EventEncoderSpectrogramStrip` component used only by the piano roll
page. The component receives controlled state and callbacks from
`EventEncoderPianoRollViewer`:

- `timeline`
- `timeRange`
- `frequencyRange`
- `playheadTime`
- `onTimeRangeChange`
- `onZoomTime`
- `onZoomFrequency`

The strip renders a left gutter matching the piano roll's left margin, a right
gutter matching the piano roll's right margin, and a central tile canvas. The
central canvas uses `TileCanvas` directly rather than mounting `Spectrogram` or
`TimelineProvider`.

The tile canvas uses:

- `jobId = timeline.region_detection_job_id`
- `jobStart = timeline.job_start_timestamp`
- `jobEnd = timeline.job_end_timestamp`
- `centerTimestamp = (timeRange.start + timeRange.end) / 2`
- `viewportSpanOverride = timeRange.end - timeRange.start`
- `freqRange = [frequencyRange.min, frequencyRange.max]`
- `tileUrlBuilder = regionTileUrl`

The strip overlays a playhead line at the same timestamp used by the piano roll
canvas. It can also draw light event markers in a later follow-up, but the
initial strip should stay visually quiet.

## 6. Smooth Zoom and Tile LOD

Timeline tiles are discrete level-of-detail assets. The backend currently
supports fixed zoom levels such as 5m, 1m, 30s, and 10s. The piano roll uses
smooth arbitrary viewport spans, so the strip needs a bridge between smooth
viewport state and discrete tile LODs.

The strip chooses the tile LOD from the current seconds-per-pixel ratio:

- Compute `secondsPerPixel = (timeRange.end - timeRange.start) / stripWidth`.
- Compare that ratio with each supported tile LOD's native seconds per tile
  pixel.
- Choose the nearest reasonable LOD.
- Apply hysteresis so small wheel movements near a threshold do not flicker
  between adjacent LODs.
- Pass the chosen LOD as `zoomLevel` and its tile duration as
  `tileDurationOverride`.
- Continue passing the smooth viewport span as `viewportSpanOverride`.

This produces map-style zooming. While the user scrolls, the current tile LOD
scales smoothly. When the viewport crosses an LOD threshold, `TileCanvas`
switches tile levels and uses its existing zoom-level crossfade.

The helper that chooses LOD should be deterministic and unit-tested separately.

## 7. Interaction

The strip is primarily contextual, but its time interactions should mirror the
piano roll:

- Wheel over the strip zooms time, centered on the cursor timestamp.
- Shift-wheel over the strip zooms frequency, centered on the cursor
  frequency.
- Dragging the strip pans time.
- Clicking the strip moves the visible playhead only if this can be done
  without changing current selected-event semantics. Otherwise, click remains a
  no-op for the first implementation.
- Strip interactions must not create a separate audio element or duplicate
  playback state.

The piano roll's existing keyboard shortcuts remain unchanged.

## 8. Layout

The piano roll page becomes four vertical bands:

1. Top toolbar.
2. Main piano roll canvas area with token legend, selection, and tooltip.
3. Spectrogram strip with the shared horizontal time axis.
4. Status bar.

The strip should be approximately 160 px tall, default visible, and collapsible
through a compact icon button in the toolbar or strip header. When collapsed,
the piano roll canvas receives the recovered vertical space.

The strip must not hide the token legend, which remains owned by the main
canvas area.

## 9. API and Backend

No backend changes are required. The existing Call Parsing region tile endpoint
already accepts `freq_min` and `freq_max`.

The frontend `regionTileUrl` helper should be extended to accept optional
frequency bounds while preserving the current three-argument call sites.

## 10. Testing

Add focused frontend tests:

- Unit-test the smooth viewport to tile LOD helper, including hysteresis around
  thresholds.
- Component or Playwright coverage verifies that the spectrogram strip renders
  on the piano roll route.
- Playwright coverage verifies that smooth wheel zoom changes the piano roll
  `timeRange` and the strip stays mounted.
- Playwright coverage verifies that tile requests include the selected
  frequency range.
- Existing Event Encoder piano roll tests should continue to pass.

No backend tests are required unless implementation discovers a backend issue.

## 11. Risks and Follow-ups

- Choosing LOD too aggressively can cause visible tile churn during smooth
  scroll. Hysteresis and preloading should keep this calm.
- A very low strip height may make frequency-specific context hard to read.
  Default to a readable bottom strip before adding user-resizable behavior.
- The existing `TileCanvas` crossfade only triggers when `zoomLevel` changes,
  not while smooth-scaling within one LOD. This is expected.
- Future follow-ups could add event tick overlays or selected-event shading if
  the simple strip proves useful.
