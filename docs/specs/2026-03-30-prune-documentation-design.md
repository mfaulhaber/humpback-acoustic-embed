# Prune Documentation Design

**Date:** 2026-03-30

## Problem

CLAUDE.md (747 lines) and DECISIONS.md (1,256 lines) are loaded into every
conversation context. Much of their content is reference material that agents
rarely need in full, and many ADR entries simply describe current code behavior
without capturing non-obvious reasoning. The combined ~2,000 lines degrade
agent performance by burying actionable rules in reference noise and consuming
tokens unnecessarily.

## Goals

1. Reduce CLAUDE.md context load by ~50% (target ~350-400 lines)
2. Preserve all reference material in on-demand files agents read only when relevant
3. Remove DECISIONS.md entries that are derivable from code; keep non-obvious *why* reasoning
4. No behavior, code, schema, or workflow changes

## Design

### 1. CLAUDE.md Split

Extract 6 reference sections into `docs/reference/`, replace each with a 2-3
line summary and a relative link.

**Sections that stay** (behavioral rules, constraints, workflow):
- §1 Purpose
- §2 High-Level Architecture
- §3.1-3.6 Core Dev Rules (package mgmt, env commands, running code, best practices, migrations, docs)
- §3.8 UTC Standard
- §4.1-4.3 Core Design Principles (idempotent encoding, resumable workflow, async jobs)
- §6 Definition of Done
- §7 Non-Goals
- §8.1 Technology Stack table
- §8.2 Repository Layout
- §8.6 Runtime Configuration
- §8.7 Behavioral Constraints
- §9 Current State (all subsections)
- §10 Workflow (all subsections)

**Sections that move** to `docs/reference/`:
- §3.7 Frontend Stack & Development -> `docs/reference/frontend.md`
- §4.4-4.7 Hydrophone rules -> `docs/reference/hydrophone-rules.md`
- §5 Testing Requirements -> `docs/reference/testing.md`
- §8.3 Data Model Summary -> `docs/reference/data-model.md`
- §8.4 Signal Processing Parameters -> `docs/reference/signal-processing.md`
- §8.5 Storage Layout -> `docs/reference/storage-layout.md`

Each reference file gets a brief header: title, one-line description of when
an agent should read it.

CLAUDE.md pointers follow this pattern:
```
### 3.7 Frontend Stack & Development
React 18 + Vite + TypeScript + Tailwind + shadcn/ui SPA in `frontend/`.
See [docs/reference/frontend.md](docs/reference/frontend.md) for full details.
```

### 2. Reference File Structure

```
docs/reference/
  frontend.md              (§3.7 — stack, file structure, dev/build workflow, package mgmt)
  hydrophone-rules.md      (§4.4-4.7 — extraction paths, timeline assembly, TSV metadata, job lifecycle)
  testing.md               (§5 — unit test guidelines, E2E spec, Playwright patterns, model stubs)
  data-model.md            (§8.3 — full table/model summary)
  signal-processing.md     (§8.4 — params table, windowing rules, pipeline diagram)
  storage-layout.md        (§8.5 — directory tree for all artifact types)
```

Content moves verbatim from CLAUDE.md — no rewriting.

### 3. DECISIONS.md Pruning

Apply this filter to all 42 ADRs:

- **Keep** if the entry captures non-obvious *why* reasoning that cannot be
  derived by reading the current code
- **Remove** if the entry just describes what the code does today (iterative
  refinements, implementation details now baked into the source)

Present the keep/remove triage list for user approval before deleting entries.

Expected result: ~15-20 ADRs survive, reducing from ~1,256 to ~400-500 lines.
