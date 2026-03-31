# Prune Documentation Implementation Plan

**Goal:** Reduce CLAUDE.md and DECISIONS.md context load by extracting reference material into on-demand files and removing code-derivable ADRs.
**Spec:** [docs/specs/2026-03-30-prune-documentation-design.md](../specs/2026-03-30-prune-documentation-design.md)

---

### Task 1: Create reference files from CLAUDE.md sections

**Files:**
- Create: `docs/reference/frontend.md`
- Create: `docs/reference/hydrophone-rules.md`
- Create: `docs/reference/testing.md`
- Create: `docs/reference/data-model.md`
- Create: `docs/reference/signal-processing.md`
- Create: `docs/reference/storage-layout.md`

**Acceptance criteria:**
- [ ] Each file contains the verbatim content from its CLAUDE.md source section
- [ ] Each file has a title and a one-line "when to read this" header
- [ ] No content is lost — every line from the extracted sections appears in a reference file

**Tests needed:**
- Manual diff: verify no content dropped between CLAUDE.md originals and new files

---

### Task 2: Slim down CLAUDE.md

**Files:**
- Modify: `CLAUDE.md`

**Acceptance criteria:**
- [ ] §3.7 replaced with 2-3 line summary + link to `docs/reference/frontend.md`
- [ ] §4.4-4.7 replaced with 2-3 line summary + link to `docs/reference/hydrophone-rules.md`
- [ ] §5.1-5.5 replaced with essentials (run commands, test requirement summary) + link to `docs/reference/testing.md`
- [ ] §8.3 replaced with pointer to `docs/reference/data-model.md`
- [ ] §8.4 replaced with pointer to `docs/reference/signal-processing.md`
- [ ] §8.5 replaced with pointer to `docs/reference/storage-layout.md`
- [ ] All remaining sections preserved unchanged
- [ ] Total line count is ~350-400 lines
- [ ] All relative links resolve correctly

**Tests needed:**
- Verify all links point to files that exist
- Verify no behavioral rules were removed (only reference material moved)

---

### Task 3: Triage DECISIONS.md ADRs

**Files:**
- (none modified yet — this is the triage/approval step)

**Acceptance criteria:**
- [ ] Every ADR categorized as keep or remove with one-line justification
- [ ] Keep/remove list presented to user for approval before proceeding
- [ ] Keep criteria: non-obvious *why* reasoning not derivable from code
- [ ] Remove criteria: describes current code behavior without unique insight

**Tests needed:**
- None (review step)

---

### Task 4: Prune DECISIONS.md

**Files:**
- Modify: `DECISIONS.md`

**Acceptance criteria:**
- [ ] Only user-approved ADRs remain
- [ ] Remaining ADRs retain their original numbering (no renumbering)
- [ ] File header and format unchanged
- [ ] Estimated ~400-500 lines remaining

**Tests needed:**
- Verify kept ADRs are intact and unmodified

---

### Task 5: Update cross-references

**Files:**
- Modify: `CLAUDE.md` (§3.6 Documentation section — add `docs/reference/` to the list)
- Modify: `CLAUDE.md` (§8.2 Repository Layout — add `docs/reference/` entry)

**Acceptance criteria:**
- [ ] §3.6 lists `docs/reference/` as a documentation location
- [ ] §8.2 repo layout tree includes `docs/reference/` with its files
- [ ] No stale references to removed DECISIONS.md ADR numbers elsewhere in CLAUDE.md

**Tests needed:**
- Grep CLAUDE.md for any ADR-NNN references that point to removed entries

---

### Verification

Run in order after all tasks:
1. Verify all `docs/reference/` links in CLAUDE.md resolve: `ls docs/reference/`
2. Verify CLAUDE.md line count: `wc -l CLAUDE.md` (target ~350-400)
3. Verify DECISIONS.md line count: `wc -l DECISIONS.md` (target ~400-500)
4. Verify no content was silently dropped: diff extracted sections against reference files
