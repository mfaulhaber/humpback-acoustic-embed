# Event Encoder Piano Roll Spectrogram Strip Implementation Plan

**Goal:** Add a synchronized PCEN spectrogram strip above the Event Encoder piano roll using existing Call Parsing region timeline tiles while preserving smooth piano roll zoom.
**Spec:** `docs/specs/2026-05-19-event-encoder-piano-roll-spectrogram-strip-design.md`
**Primary domain:** sequence-models
**Neighbor domains:** signal-timeline, frontend-shell

---

### Task 1: Add Smooth Tile LOD Selection

**Files:**
- Create: `frontend/src/components/sequence-models/eventEncoderSpectrogramLod.ts`
- Create: `frontend/src/components/sequence-models/eventEncoderSpectrogramLod.test.ts`

**Acceptance criteria:**
- [ ] Helper accepts viewport span, strip width, current zoom key, and supported tile LOD definitions
- [ ] Helper chooses a stable tile LOD from seconds-per-pixel rather than requiring piano roll zoom buttons
- [ ] Helper applies hysteresis so tiny wheel movements near a threshold keep the current LOD
- [ ] Helper supports backend-backed zoom levels including `5m`, `1m`, `30s`, and `10s`
- [ ] Tests cover coarse, medium, fine, and threshold-hysteresis cases

**Tests needed:**
- Unit tests for LOD selection and hysteresis behavior

---

### Task 2: Extend Region Tile URL Helper

**Files:**
- Modify: `frontend/src/api/client.ts`

**Acceptance criteria:**
- [ ] `regionTileUrl` accepts optional `freqMin` and `freqMax` arguments with defaults matching current behavior
- [ ] Existing three-argument `regionTileUrl` callers continue to work unchanged
- [ ] Generated URLs include `freq_min` and `freq_max` query parameters

**Tests needed:**
- Covered by frontend typecheck and piano roll Playwright tile request assertions

---

### Task 3: Build Controlled Spectrogram Strip Component

**Files:**
- Create: `frontend/src/components/sequence-models/EventEncoderSpectrogramStrip.tsx`

**Acceptance criteria:**
- [ ] Component renders a low-height strip using `TileCanvas` directly
- [ ] Component receives piano roll `timeRange`, `frequencyRange`, and playhead timestamp as controlled props
- [ ] Component aligns its tile canvas with the piano roll plot margins
- [ ] Component uses the existing Call Parsing region tile source from the Event Encoder timeline response
- [ ] Component passes smooth `viewportSpanOverride` to `TileCanvas`
- [ ] Component passes selected tile LOD and tile duration from the LOD helper
- [ ] Component draws a synchronized playhead over the tile canvas
- [ ] Component renders a compact label or affordance without adding explanatory in-app text

**Tests needed:**
- Component-level or Playwright coverage that the strip renders with mocked timeline data

---

### Task 4: Wire Strip Into Piano Roll Layout And Interactions

**Files:**
- Modify: `frontend/src/components/sequence-models/EventEncoderPianoRollPage.tsx`

**Acceptance criteria:**
- [ ] Piano roll page renders the spectrogram strip between the toolbar and main canvas
- [ ] Strip is visible by default and can be collapsed without losing piano roll state
- [ ] Strip wheel zoom updates the same piano roll `timeRange` as main-canvas wheel zoom
- [ ] Strip shift-wheel updates the same piano roll `frequencyRange` as main-canvas shift-wheel zoom
- [ ] Strip drag pans the same piano roll `timeRange` as main-canvas drag
- [ ] Main canvas, minimap, token legend, tooltip, selection, and playback behavior remain unchanged
- [ ] Smooth zoom continues to update the piano roll `data-view-start` and `data-view-end` test attributes

**Tests needed:**
- Playwright coverage for strip visibility, collapse behavior, and strip wheel or drag updating the shared viewport

---

### Task 5: Update Event Encoder Piano Roll E2E Coverage

**Files:**
- Modify: `frontend/e2e/sequence-models/event-encoder-piano-roll.spec.ts`

**Acceptance criteria:**
- [ ] Mocked region tile route records strip tile requests
- [ ] Route test asserts the spectrogram strip is visible by default
- [ ] Test asserts tile requests include `freq_min` and `freq_max`
- [ ] Test verifies strip collapse removes the strip and recovers page space
- [ ] Test verifies wheel or drag interaction on the strip changes the shared piano roll viewport
- [ ] Existing piano roll selection, legend, minimap, playback, and navigation assertions still pass

**Tests needed:**
- Updated Event Encoder piano roll Playwright tests

---

### Verification

Run in order after all tasks:

1. `cd frontend && npx vitest run src/components/sequence-models/eventEncoderSpectrogramLod.test.ts`
2. `cd frontend && npx tsc --noEmit`
3. `cd frontend && npx playwright test e2e/sequence-models/event-encoder-piano-roll.spec.ts`
4. `cd frontend && npx playwright test e2e/sequence-models/event-encoder.spec.ts`
