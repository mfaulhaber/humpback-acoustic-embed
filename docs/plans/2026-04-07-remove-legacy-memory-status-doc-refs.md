# Remove Legacy Memory/Status Doc References Implementation Plan

**Goal:** Remove remaining references to the retired `MEMORY.md` and `STATUS.md` files from markdown documentation while keeping the historical docs readable
**Spec:** None needed; this is a focused documentation cleanup

---

### Task 1: Identify and update legacy references in historical docs

**Files:**
- Modify: `docs/specs/2026-03-24-superpowers-workflow-adoption-design.md`
- Modify: `docs/plans/2026-03-24-superpowers-workflow-adoption.md`
- Modify: `docs/plans/2026-03-24-timeline-viewer.md`

**Acceptance criteria:**
- [ ] Literal references to `MEMORY.md` are removed from affected markdown docs
- [ ] Literal references to `STATUS.md` are removed from affected markdown docs
- [ ] Historical context remains understandable without pointing readers to retired files

**Tests needed:**
- Re-run a repo-wide markdown search for `MEMORY.md|STATUS.md` and confirm it returns no matches

---

### Task 2: Verify workflow docs and final markdown state

**Files:**
- Verify only: `docs/workflows/*.md`

**Acceptance criteria:**
- [ ] `docs/workflows/` contains no references to the retired files
- [ ] Final verification output shows no markdown references remain anywhere in the repo

**Tests needed:**
- Search `docs/workflows/*.md` specifically for `MEMORY.md|STATUS.md`
- Search all `*.md` files for `MEMORY.md|STATUS.md`

---

### Verification

Run in order after all tasks:
1. `rg -n --glob 'docs/workflows/*.md' 'MEMORY\\.md|STATUS\\.md' docs/workflows`
2. `rg -n --glob '*.md' 'MEMORY\\.md|STATUS\\.md' .`
