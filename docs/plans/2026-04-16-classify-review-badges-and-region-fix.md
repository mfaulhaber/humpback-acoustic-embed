# Classify Review — Badges, None Indicator, and Region Fix Implementation Plan

**Goal:** Fix added-event spectrograms in Classify Review, add a non-interactive "None" palette indicator, and render a 2-letter type badge on each event bar.

**Spec:** [docs/specs/2026-04-16-classify-review-badges-and-region-fix-design.md](../specs/2026-04-16-classify-review-badges-and-region-fix-design.md)

---

### Task 1: Backend — use `load_corrected_events` in the typed-events endpoint

**Files:**
- Modify: `src/humpback/api/routers/call_parsing.py`

**Acceptance criteria:**
- [ ] `get_classification_typed_events` (around line 704) builds `event_region_map` from `load_corrected_events(session, job.event_segmentation_job_id, settings.storage_root)` instead of reading `events.parquet` directly.
- [ ] The endpoint imports `load_corrected_events` from `humpback.call_parsing.segmentation.extraction`.
- [ ] The "events.parquet missing" error path remains: if `load_corrected_events` raises because `events.parquet` does not exist, the endpoint still returns HTTP 404 with a clear message.
- [ ] Response JSON shape and ordering are unchanged (`event_id`, `region_id`, `start_sec`, `end_sec`, `type_name`, `score`, `above_threshold`; sorted by `(start_sec, type_name)`).
- [ ] When `load_corrected_events` returns no row for a given typed-event `event_id` (e.g., classifier output is stale relative to corrections), the endpoint still falls back to `""` for `region_id` and does not raise.

**Tests needed:**
- Add an integration/API test alongside existing call-parsing tests: seed a segmentation job with one normal event and one `add` boundary correction, write a `typed_events.parquet` containing rows for both event_ids, create the classification job, GET the typed-events endpoint, and assert the add-event's returned `region_id` matches the correction row.
- Add a regression test: a job with no boundary corrections still returns the original `region_id` values from `events.parquet` for each typed-event row.

---

### Task 2: Frontend — "None" status chip in `TypePalette`

**Files:**
- Modify: `frontend/src/components/call-parsing/TypePalette.tsx`

**Acceptance criteria:**
- [ ] A new non-interactive chip labelled `None` renders as the leftmost element in the palette, before the `(Negative)` chip.
- [ ] The chip is present in the DOM in all states; when `activeType !== null`, it is styled `visibility: hidden` so it continues to occupy layout space (no reflow of other palette elements).
- [ ] The chip has no `onClick`, no hover state, no focus ring, and no pointer cursor. Disabled-looking style: muted foreground text, dashed muted border.
- [ ] A stable DOM attribute (`data-testid="palette-none-indicator"`) is present so Playwright can query it.
- [ ] Palette buttons (type buttons and `(Negative)`) always pass the type name to `onSelectType` — the previous "click the active button to toggle to `null`" behavior is removed.
- [ ] `handleSelectType` in `ClassifyReviewWorkspace.tsx` compares the clicked value against `currentEvent.correctedType` and skips `setPendingCorrections` when they match, so re-clicking an existing correction is idempotent (Save does not relight).
- [ ] Clicking the palette button that matches the current inference prediction promotes the prediction to a human correction (`typeSource` flips from `"inference"` to `"correction"`, workspace becomes dirty).

**Tests needed:**
- Playwright: on an event with `activeType === null`, assert the None chip is visible (computed `visibility` not `hidden`).
- Playwright: click a type button; assert the None chip's computed `visibility` is `hidden` but its bounding box still occupies space.
- Playwright: capture the `(Negative)` chip's bounding `left` with and without the None chip visible and assert the values are identical (layout-stability assertion).
- Playwright: click the palette button matching an inference-only event's predicted type; assert its badge's `data-source` flips from `inference` to `correction` and the dirty indicator ("N unsaved change(s)") appears.
- Playwright: click the same matching button a second time; assert the unsaved-change count stays at 1 (idempotent re-click).

---

### Task 3: Frontend — extend `EffectiveEvent` and populate `effectiveType` / `typeSource`

**Files:**
- Modify: `frontend/src/components/call-parsing/EventBarOverlay.tsx` (type only)
- Modify: `frontend/src/components/call-parsing/ClassifyReviewWorkspace.tsx`

**Acceptance criteria:**
- [ ] `EffectiveEvent` in `EventBarOverlay.tsx` gains two fields: `effectiveType: string | null` and `typeSource: "inference" | "correction" | "negative" | null`.
- [ ] In `ClassifyReviewWorkspace.tsx`, the `regionEffectiveEvents` memo populates both new fields for each event, including:
  - Events present in `events` list (inference/correction/negative classification per spec rules).
  - Saved `add` boundary corrections rendered as standalone events (use the corrected type if present, else inference-style if a typed_event row exists for that id, else `null`).
  - Pending `add` boundary corrections (same priority rules as above; new events will typically have `typeSource = null` until a palette click).
- [ ] `mergedCorrections` is the single source of truth for "has the user corrected this event" — no parallel bookkeeping introduced.
- [ ] No runtime behavior change visible to the user from this task alone (badge rendering arrives in Task 4).
- [ ] `npx tsc --noEmit` passes.

**Tests needed:**
- Deferred to Task 4 (badge rendering is the observable surface of this change).

---

### Task 4: Frontend — render 2-letter type badge on event bars

**Files:**
- Modify: `frontend/src/components/call-parsing/EventBarOverlay.tsx`

**Acceptance criteria:**
- [ ] Each non-deleted active bar renders a badge at `left: 0; top: 0` inside the bar, sized ~20×14 px, when `typeSource !== null`.
- [ ] Badge label is the first two characters of `effectiveType`, uppercased, for `"inference"` and `"correction"` sources.
- [ ] For `typeSource === "negative"`, badge label is `—` (em dash) on a solid red background (`hsl(0, 70%, 50%)`) with white text.
- [ ] For `typeSource === "inference"`, badge is a colored outline on a white background: `background: #fff`, 1px `typeColor(effectiveType)` border, `typeColor(effectiveType)` text. (White background keeps the outlined code legible against the purple bar fill.)
- [ ] For `typeSource === "correction"`, badge is solid: `typeColor(effectiveType)` background, white text, 1px `typeColor(effectiveType)` border.
- [ ] For `typeSource === null`, no badge is rendered.
- [ ] Badge is `pointer-events: none; overflow: visible` so it does not block drag/click handlers on the bar and can extend past a narrow bar's right edge.
- [ ] Drag-preview bars receive the same badge, pinned to the rendered left edge.
- [ ] Deleted bars have no badge (unchanged from today).
- [ ] Stable DOM attributes on the badge: `data-testid="event-badge-{eventId}"` and `data-source="{inference|correction|negative}"`.
- [ ] `npx tsc --noEmit` passes.

**Tests needed:**
- Playwright: seed a classification job with a mix of event states (inference-only, corrected positive, corrected negative, unlabeled). For each state, assert the badge element's presence, `data-source`, and visible text. For the unlabeled case, assert no `event-badge-*` element exists inside that bar.
- Playwright: select a bar and click a palette type; assert the badge's `data-source` flips from `inference` to `correction`. Click `(Negative)`; assert `data-source` becomes `negative` and text is `—`.

---

### Task 5: Manual smoke on job `728d2e47-1916-4761-bdfa-84020156731c`

**Acceptance criteria:**
- [ ] Event 14 (`add-e9921545…`) renders its spectrogram with the region shown.
- [ ] Event 23 (only below-threshold Ascending Moan) shows the "None" chip in the palette with the disabled style.
- [ ] Every classified event shows a 2-letter badge; corrections applied during review flip a bar's badge from bordered to solid.
- [ ] Palette layout does not shift when navigating between events that toggle the None chip.

**Tests needed:**
- Manual only (recorded in session notes); automated coverage is the Playwright tests above.

---

### Verification

Run in order after all tasks:

1. `uv run ruff format --check src/humpback/api/routers/call_parsing.py`
2. `uv run ruff check src/humpback/api/routers/call_parsing.py`
3. `uv run pyright src/humpback/api/routers/call_parsing.py tests/`
4. `uv run pytest tests/`
5. `cd frontend && npx tsc --noEmit`
6. `cd frontend && npx playwright test` (full suite; narrow to the new specs during development)
