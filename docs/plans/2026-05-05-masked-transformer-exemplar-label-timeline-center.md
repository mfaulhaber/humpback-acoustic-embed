# Masked Transformer Exemplar Label Timeline Center Implementation Plan

**Goal:** Clicking a vocal label badge in the Masked Transformer Exemplars panel recenters and reveals the Token Timeline around that exemplar.
**Spec:** [docs/specs/2026-05-05-masked-transformer-exemplar-label-timeline-center-design.md](../specs/2026-05-05-masked-transformer-exemplar-label-timeline-center-design.md)

---

### Task 1: Add Page-Level Timeline Centering Wiring

**Files:**
- Modify: `frontend/src/components/sequence-models/MaskedTransformerDetailPage.tsx`

**Acceptance criteria:**
- [ ] `MaskedTransformerDetailPage` owns a ref for the Token Timeline card or viewer container.
- [ ] The page defines a handler that computes the clicked exemplar midpoint from `start_timestamp` and `end_timestamp`.
- [ ] The handler calls the existing `TimelinePlaybackHandle.seekTo(...)` with the computed midpoint.
- [ ] The handler scrolls the Token Timeline container into view after seeking.
- [ ] The handler is safe when the timeline handle or timeline container is unavailable.

**Tests needed:**
- Covered by the Playwright interaction test in Task 3.

---

### Task 2: Make Exemplar Label Badges Center the Timeline

**Files:**
- Modify: `frontend/src/components/sequence-models/MaskedTransformerDetailPage.tsx`

**Acceptance criteria:**
- [ ] `ExemplarsSection`, `ExemplarList`, and `ExemplarEventTypeChips` accept and forward an optional `onCenterExemplar` callback.
- [ ] Non-background vocal label chips call `onCenterExemplar(exemplar)` when clicked.
- [ ] Clicking a non-background vocal label chip keeps the user on the Masked Transformer detail page.
- [ ] Non-background chips expose an accessible label and title that communicate the timeline-centering action.
- [ ] Background chips remain inert and do not seek the timeline.
- [ ] HMM exemplar behavior is unchanged because the callback is only supplied from the Masked Transformer detail page.

**Tests needed:**
- Covered by the Playwright interaction test in Task 3.
- TypeScript compile coverage for the prop-threading changes.

---

### Task 3: Add Masked Transformer Exemplar Click-to-Center Coverage

**Files:**
- Modify: `frontend/e2e/sequence-models/masked-transformer.spec.ts`

**Acceptance criteria:**
- [ ] The Masked Transformer E2E fixture includes at least one exemplar with non-background `event_types` and timestamps away from the initial timeline center.
- [ ] The test clicks a vocal label chip in the Exemplars panel.
- [ ] The test asserts the app remains on the Masked Transformer detail page.
- [ ] The test asserts the Token Timeline is visible after the click.
- [ ] The test asserts the timeline recenters around the clicked exemplar midpoint through a stable observable signal.
- [ ] The test asserts clicking a background chip does not trigger timeline movement when a background exemplar is present in the fixture.

**Tests needed:**
- `cd frontend && npx playwright test e2e/sequence-models/masked-transformer.spec.ts`

---

### Task 4: Update Frontend Reference Notes

**Files:**
- Modify: `docs/reference/frontend.md`

**Acceptance criteria:**
- [ ] The Masked Transformer detail-page notes mention that exemplar vocal label badges center the Token Timeline in-place.
- [ ] The note distinguishes this from Classify Review navigation so future UI work does not accidentally restore badge-click navigation as the primary behavior.

**Tests needed:**
- Documentation-only change; covered by review.

---

### Verification

Run in order after all tasks:

1. `cd frontend && npx tsc --noEmit`
2. `cd frontend && npm run test -- --run`
3. `cd frontend && npx playwright test e2e/sequence-models/masked-transformer.spec.ts`
4. `uv run ruff format --check src tests`
5. `uv run ruff check src tests`
6. `uv run pyright src`
7. `uv run pytest tests/`
