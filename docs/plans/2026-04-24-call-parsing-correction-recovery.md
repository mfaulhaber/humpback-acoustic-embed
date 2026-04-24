# Call Parsing Correction Recovery Implementation Plan

**Goal:** Restore recoverable legacy call-parsing correction data and fix the Classify review reload path for saved labels on boundary-added events.
**Spec:** [docs/specs/2026-04-24-call-parsing-correction-recovery-design.md](/Users/michael/development/humpback-acoustic-embed/docs/specs/2026-04-24-call-parsing-correction-recovery-design.md)

---

### Task 1: Land Recovery Tooling

**Files:**
- Create: `scripts/recover_vocalization_corrections.py`
- Create: `tests/unit/test_recover_vocalization_corrections.py`
- Create: `scripts/recover_event_boundary_corrections.py`
- Create: `tests/unit/test_recover_event_boundary_corrections.py`

**Acceptance criteria:**
- [ ] Both recovery scripts support dry-run previews and explicit `--apply` mode.
- [ ] The vocalization recovery script reconstructs unified rows from legacy DB data plus on-disk call-parsing artifacts.
- [ ] Recovery scripts verify the target DB after writes and exit non-zero on verification failure.
- [ ] Unrecoverable legacy rows are reported clearly instead of being silently skipped.

**Tests needed:**
- Unit coverage for plan generation, apply behavior, and verification behavior for the recovery scripts.

---

### Task 2: Fix Saved Label Reload For Boundary-Added Events

**Files:**
- Modify: `frontend/src/components/call-parsing/ClassifyReviewWorkspace.tsx`
- Create: `frontend/src/components/call-parsing/ClassifyReviewWorkspace.test.ts`

**Acceptance criteria:**
- [ ] Saved vocalization corrections are remapped onto boundary-added events after reload.
- [ ] Existing typed-event correction behavior remains unchanged for positive and negative corrections.
- [ ] The fix reuses the same synthetic event identity already used by the review workspace for saved adds.

**Tests needed:**
- Frontend unit coverage for saved add-event label reattachment and for existing negative correction behavior.

---

### Task 3: Verify Against Real Recovery Inputs

**Files:**
- Modify: `scripts/recover_vocalization_corrections.py`
- Modify: `frontend/src/components/call-parsing/ClassifyReviewWorkspace.tsx`

**Acceptance criteria:**
- [ ] The vocalization recovery script produces the expected dry-run summary for the known April 19 backup and live DB pair.
- [ ] The live recovery path verifies successfully after applying recovered rows.
- [ ] App-level spot checks confirm standard corrected Pass 2/Pass 3 labels render correctly, and the saved boundary-added label path is fixed.

**Tests needed:**
- Manual dry-run/apply execution against the external backup/live DB pair.
- Manual app/API spot checks against the affected Pass 2/Pass 3 jobs.

---

### Verification

Run in order after all tasks:
1. `uv run ruff format --check scripts/recover_vocalization_corrections.py scripts/recover_event_boundary_corrections.py tests/unit/test_recover_vocalization_corrections.py tests/unit/test_recover_event_boundary_corrections.py`
2. `uv run ruff check scripts/recover_vocalization_corrections.py scripts/recover_event_boundary_corrections.py tests/unit/test_recover_vocalization_corrections.py tests/unit/test_recover_event_boundary_corrections.py`
3. `uv run pytest tests/unit/test_recover_vocalization_corrections.py tests/unit/test_recover_event_boundary_corrections.py`
4. `cd frontend && npx vitest run src/components/call-parsing/ClassifyReviewWorkspace.test.ts`
5. `cd frontend && npx tsc --noEmit`
