# Event Encoder Timeline Viewer Implementation Plan

**Goal:** Add a read-only tokenized-event timeline viewer as the second panel on the Event Encoder job detail page.
**Spec:** [docs/specs/2026-05-08-event-encoder-timeline-viewer-design.md](../specs/2026-05-08-event-encoder-timeline-viewer-design.md)
**Primary domain:** sequence-models
**Neighbor domains:** signal-timeline, call-parsing, frontend-shell

---

### Task 1: Add Event Encoder Timeline API

**Files:**
- Modify: `src/humpback/schemas/sequence_models.py`
- Modify: `src/humpback/api/routers/sequence_models.py`
- Modify: `tests/integration/test_sequence_models_api.py`

**Acceptance criteria:**
- [x] `EventEncoderTimelineEvent` and `EventEncoderTimelineResponse` schemas expose job provenance, selected k, valid k values, region timeline bounds, and compact event token rows.
- [x] `GET /sequence-models/event-encoders/{job_id}/timeline` reads the completed job's `event_tokens.parquet`, filters to one k, and returns rows sorted by source sequence, timestamp, and event id.
- [x] The endpoint resolves `region_detection_job_id`, `job_start_timestamp`, and `job_end_timestamp` from the referenced CRNN Continuous Embedding and Pass 1 region job.
- [x] When `k` is omitted, the endpoint chooses the lowest valid k present in the token artifact.
- [x] The endpoint returns `404` for missing jobs or missing token artifacts, `409` for incomplete jobs or corrupt non-region-CRNN provenance, and `422` for unavailable requested k values.
- [x] The endpoint treats the token artifact as authoritative and does not reload current raw or effective Pass 2 events.

**Tests needed:**
- Integration tests for complete timeline response shape, k filtering, default selected k, sorted rows, invalid k, incomplete job, missing job, missing artifact, and artifact-authoritative timestamps.

---

### Task 2: Add Frontend Timeline API Types And Query Hook

**Files:**
- Modify: `frontend/src/api/sequenceModels.ts`

**Acceptance criteria:**
- [x] TypeScript interfaces mirror the backend timeline response and event-row schemas.
- [x] `fetchEventEncoderTimeline(jobId, k?)` requests the new endpoint and encodes optional k only when provided.
- [x] `useEventEncoderTimeline(jobId, k, enabled)` caches by job id and selected k and only runs when the detail page enables it.
- [x] Existing Continuous Embedding and Event Encoder create/list/detail hooks keep their current query keys and behavior.

**Tests needed:**
- TypeScript compile coverage for the new exported types, fetcher, and hook.
- Playwright mock coverage through the Event Encoder detail page.

---

### Task 3: Build Read-Only Event Encoder Timeline Panel

**Files:**
- Create: `frontend/src/components/sequence-models/EventEncoderTimelinePanel.tsx`
- Create: `frontend/src/components/sequence-models/EventEncoderTokenOverlay.tsx`
- Modify: `frontend/src/components/sequence-models/EventEncoderDetailPage.tsx`

**Acceptance criteria:**
- [x] `EventEncoderDetailPage` renders the timeline card as the second panel after the job summary and before the report card.
- [x] Complete jobs fetch and render tokenized events with `TimelineProvider`, `Spectrogram`, `regionTileUrl`, `regionAudioSliceUrl`, `REVIEW_ZOOM`, and `ZoomSelector`.
- [x] Queued, running, failed, canceled, empty, and timeline-fetch-error states render clear unavailable or empty messages without hiding the existing report and error content.
- [x] The panel toolbar includes previous event, next event, selected event counter, selected token badge, token confidence, selected k control when multiple k values exist, and a playback control.
- [x] The token overlay positions bars from absolute `start_timestamp` and `end_timestamp`, colors them deterministically with `labelColor(token_id, selected_k)`, highlights the selected event, and renders visible token-label badges such as `T17`.
- [x] Event bars are read-only: no drag handles, add mode, correction styling, or mutation affordances are exposed.
- [x] Selecting an event by click or navigation centers the viewport on that event and keeps the selected event when changing k if the event still exists.
- [x] Local keyboard shortcuts handle `A`, `D`, `Space`, `+`, `=`, `-`, `ArrowLeft`, and `ArrowRight` while ignoring input, textarea, select, contenteditable, and modified keypress contexts.
- [x] Playback uses the shared timeline playback handle rather than creating a separate audio element.

**Tests needed:**
- Component tests for overlay positioning, selected styling, and token badge rendering.
- Playwright tests for panel placement, controls, keyboard navigation, k switching, playback-triggered audio URL usage, and unavailable states.

---

### Task 4: Expand Event Encoder Frontend Test Coverage

**Files:**
- Create: `frontend/src/components/sequence-models/EventEncoderTokenOverlay.test.tsx`
- Modify: `frontend/e2e/sequence-models/event-encoder.spec.ts`

**Acceptance criteria:**
- [x] Existing Event Encoder e2e tests keep passing with the new timeline endpoint mocked.
- [x] Complete-job e2e coverage asserts the timeline panel appears between the job summary and report panel.
- [x] E2e coverage asserts token badges render, previous and next controls update selection, and `A` / `D` keyboard navigation updates selection.
- [x] E2e coverage asserts changing k requests the new k and updates the displayed token labels.
- [x] E2e coverage asserts non-complete jobs show the timeline unavailable state.
- [x] Overlay component coverage asserts bars use overlay context geometry and selected-event styling.

**Tests needed:**
- `cd frontend && npx vitest run src/components/sequence-models/EventEncoderTokenOverlay.test.tsx`
- `cd frontend && npx playwright test e2e/sequence-models/event-encoder.spec.ts`

---

### Task 5: Update Reference Docs And Agent Context

**Files:**
- Modify: `docs/reference/sequence-models-api.md`
- Modify: `docs/reference/frontend.md`
- Modify: `docs/agent-context/domains/sequence-models/README.md`
- Modify: `docs/agent-context/domains/sequence-models/references.md`
- Modify: `docs/agent-context/domains/sequence-models/tests.md`

**Acceptance criteria:**
- [x] Sequence Models API reference documents the Event Encoder timeline endpoint, response shape, selected-k behavior, and error cases.
- [x] Frontend reference describes the Event Encoder detail timeline panel as part of the active Sequence Models UI surface.
- [x] Sequence Models domain capsule references the new frontend timeline files and targeted tests.
- [x] Docs preserve Event Encoder raw/effective artifact-authoritative semantics and do not imply token labels are globally stable across jobs.

**Tests needed:**
- Documentation diff review and `git diff --check`.

---

### Verification

Run in order after all tasks:

1. `git diff --check`
2. `uv run ruff format --check src/humpback/schemas/sequence_models.py src/humpback/api/routers/sequence_models.py tests/integration/test_sequence_models_api.py`
3. `uv run ruff check src/humpback/schemas/sequence_models.py src/humpback/api/routers/sequence_models.py tests/integration/test_sequence_models_api.py`
4. `uv run pyright`
5. `uv run pytest tests/integration/test_sequence_models_api.py -q`
6. `uv run pytest tests/services/test_event_encoder_service.py tests/workers/test_event_encoder_worker.py -q`
7. `cd frontend && npx tsc --noEmit`
8. `cd frontend && npx vitest run src/components/sequence-models/EventEncoderTokenOverlay.test.tsx`
9. `cd frontend && npx playwright test e2e/sequence-models/event-encoder.spec.ts`
10. `uv run pytest tests/`
