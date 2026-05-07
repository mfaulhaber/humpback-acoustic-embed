# Agent-Local Project Structure Implementation Plan

**Goal:** Implement the Option B documentation structure so agents start from small domain-local context capsules while preserving existing reference docs and full verification gates.
**Spec:** [docs/specs/2026-05-07-agent-local-project-structure-design.md](../specs/2026-05-07-agent-local-project-structure-design.md)

---

### Task 1: Add Agent Context Root Indexes

**Files:**
- Create: `docs/agent-context/README.md`
- Create: `docs/agent-context/domain-map.md`
- Create: `docs/agent-context/global-invariants.md`
- Create: `docs/agent-context/current-state.md`
- Create: `docs/agent-context/test-map.md`

**Acceptance criteria:**
- [ ] `README.md` explains that `CLAUDE.md` stays global and `docs/agent-context/` is the domain-local loading layer.
- [ ] `domain-map.md` maps task keywords and changed file paths to one primary domain and likely neighbor domains.
- [ ] `global-invariants.md` contains only cross-domain invariants that all implementation agents should know.
- [ ] `current-state.md` moves active capability summaries out of global startup context and groups them by domain.
- [ ] `test-map.md` lists domain-oriented backend and frontend verification commands without weakening the full-suite gate.

**Tests needed:**
- Documentation review for consistency with the design spec.
- Link/path spot-checks for every referenced domain directory and workflow file.

---

### Task 2: Create Domain Capsules

**Files:**
- Create: `docs/agent-context/domains/core-platform/README.md`
- Create: `docs/agent-context/domains/core-platform/invariants.md`
- Create: `docs/agent-context/domains/core-platform/tests.md`
- Create: `docs/agent-context/domains/core-platform/references.md`
- Create: `docs/agent-context/domains/signal-timeline/README.md`
- Create: `docs/agent-context/domains/signal-timeline/invariants.md`
- Create: `docs/agent-context/domains/signal-timeline/tests.md`
- Create: `docs/agent-context/domains/signal-timeline/references.md`
- Create: `docs/agent-context/domains/ingest-detection/README.md`
- Create: `docs/agent-context/domains/ingest-detection/invariants.md`
- Create: `docs/agent-context/domains/ingest-detection/tests.md`
- Create: `docs/agent-context/domains/ingest-detection/references.md`
- Create: `docs/agent-context/domains/vocalization-clustering/README.md`
- Create: `docs/agent-context/domains/vocalization-clustering/invariants.md`
- Create: `docs/agent-context/domains/vocalization-clustering/tests.md`
- Create: `docs/agent-context/domains/vocalization-clustering/references.md`
- Create: `docs/agent-context/domains/call-parsing/README.md`
- Create: `docs/agent-context/domains/call-parsing/invariants.md`
- Create: `docs/agent-context/domains/call-parsing/tests.md`
- Create: `docs/agent-context/domains/call-parsing/references.md`
- Create: `docs/agent-context/domains/sequence-models/README.md`
- Create: `docs/agent-context/domains/sequence-models/invariants.md`
- Create: `docs/agent-context/domains/sequence-models/tests.md`
- Create: `docs/agent-context/domains/sequence-models/references.md`
- Create: `docs/agent-context/domains/frontend-shell/README.md`
- Create: `docs/agent-context/domains/frontend-shell/invariants.md`
- Create: `docs/agent-context/domains/frontend-shell/tests.md`
- Create: `docs/agent-context/domains/frontend-shell/references.md`

**Acceptance criteria:**
- [ ] Each domain `README.md` states when to load the domain, source paths, service/API/worker paths, frontend paths, artifact roots, and likely neighbor domains.
- [ ] Each `invariants.md` contains only local non-obvious rules that affect correctness or agent decision-making.
- [ ] Each `tests.md` gives a small smoke command, normal domain command, and expansion commands for API/frontend/storage changes.
- [ ] Each `references.md` points to existing `docs/reference/` files and ADR headings rather than duplicating long reference material.
- [ ] Capsules remain short enough that agents can load one primary domain and one neighbor domain without pulling in broad unrelated context.

**Tests needed:**
- Documentation review against current source, test, frontend, and reference paths.
- Run the documented domain commands only if they are added as executable shell targets; otherwise validate that paths exist.

---

### Task 3: Slim Global Agent Instructions

**Files:**
- Modify: `CLAUDE.md`
- Modify: `AGENTS.md`

**Acceptance criteria:**
- [ ] `CLAUDE.md` keeps universal development rules, universal invariants, workflow pointers, and final verification gates.
- [ ] `CLAUDE.md` routes domain-specific work to `docs/agent-context/README.md` and `docs/agent-context/domain-map.md`.
- [ ] Long current-state, schema, sensitive-component, API-surface, timeline, and sequence-model details are removed from global context or reduced to short pointers.
- [ ] `AGENTS.md` tells Codex to load the relevant domain capsule during Context and before planning/implementing.
- [ ] The rule that `CLAUDE.md` remains authoritative is preserved.

**Tests needed:**
- Documentation review to confirm no universal safety rule was removed.
- Link/path spot-checks for new `docs/agent-context/` references.

---

### Task 4: Update Session Workflows For Domain-Local Loading

**Files:**
- Modify: `docs/workflows/session-begin.md`
- Modify: `docs/workflows/session-plan.md`
- Modify: `docs/workflows/session-implement.md`
- Modify: `docs/workflows/session-review.md`

**Acceptance criteria:**
- [ ] `session-begin` summarizes project state without requiring agents to read every domain capsule.
- [ ] `session-plan` requires selecting the primary domain and likely neighbor domains before writing a plan.
- [ ] `session-implement` requires loading the selected domain capsules and using `docs/agent-context/test-map.md` before falling back to filename heuristics.
- [ ] `session-review` checks that modified docs updated the matching domain capsule or intentionally left it unchanged.
- [ ] Existing branch, commit, database-backup, and final-verification behavior remains intact.

**Tests needed:**
- Documentation review against the current workflow order.
- Verify no workflow now instructs agents to re-read `CLAUDE.md` when it is already loaded.

---

### Task 5: Make Targeted Verification Discoverable

**Files:**
- Modify: `docs/agent-context/test-map.md`
- Modify: `docs/agent-context/domains/core-platform/tests.md`
- Modify: `docs/agent-context/domains/signal-timeline/tests.md`
- Modify: `docs/agent-context/domains/ingest-detection/tests.md`
- Modify: `docs/agent-context/domains/vocalization-clustering/tests.md`
- Modify: `docs/agent-context/domains/call-parsing/tests.md`
- Modify: `docs/agent-context/domains/sequence-models/tests.md`
- Modify: `docs/agent-context/domains/frontend-shell/tests.md`

**Acceptance criteria:**
- [ ] Every domain has one fast backend command when backend files are touched.
- [ ] Every frontend-relevant domain points to the matching Playwright specs or TypeScript checks.
- [ ] Cross-domain changes have explicit expansion guidance rather than defaulting immediately to full-suite-only feedback.
- [ ] The docs clearly state that `uv run pytest tests/` remains the authoritative final backend gate.
- [ ] No pytest markers, Make targets, or helper scripts are required in this first implementation.

**Tests needed:**
- Run representative documented backend commands for at least `call-parsing`, `sequence-models`, and `signal-timeline` if time permits.
- Always run the final full backend test gate during verification.

---

### Task 6: Preserve Compatibility References

**Files:**
- Modify: `docs/reference/behavioral-constraints.md`
- Modify: `docs/reference/data-model.md`
- Modify: `docs/reference/storage-layout.md`
- Modify: `docs/reference/testing.md`

**Acceptance criteria:**
- [ ] Existing top-level reference docs remain valid and do not become broken stubs.
- [ ] Each broad reference file points agents toward the new domain capsules for task startup context.
- [ ] No historical specs, plans, or ADR links need to change.
- [ ] The design's later reference split remains documented as future work, not silently started in this implementation.

**Tests needed:**
- Documentation review for link compatibility.
- Path spot-checks for the new capsule links.

---

### Verification

Run in order after all tasks:

1. `git diff --check`
2. `uv run pytest tests/`
3. `git status --short`

No Python or frontend source files are expected. If implementation unexpectedly modifies Python files, also run `uv run ruff format --check <modified-python-files>`, `uv run ruff check <modified-python-files>`, and `uv run pyright <modified-python-files>`. If implementation unexpectedly modifies frontend files, also run `cd frontend && npx tsc --noEmit`.

