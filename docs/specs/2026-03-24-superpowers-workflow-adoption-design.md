# Superpowers Workflow Adoption — Design Spec

**Date:** 2026-03-24
**Status:** Approved

---

## Goal

Adopt the superpowers skill workflow as the canonical development process for this project, replacing the custom session-* skills. Consolidate the 6 repo-root `.md` files down to 3. Provide a Codex-compatible workflow that mirrors the superpowers phases using only Codex-available tools.

---

## Current State

### Repo-Root Files (6)

| File | Lines | Role |
|------|-------|------|
| `CLAUDE.md` | ~285 | Behavioral rules, dev constraints, testing, Definition of Done |
| `MEMORY.md` | ~600 | Data models, workflows, signal params, storage layout |
| `STATUS.md` | ~140 | Implemented capabilities, schema version, constraints |
| `PLANS.md` | ~175 | Active/backlog/completed plan index |
| `DECISIONS.md` | ~500+ | Append-only architecture decision log |
| `AGENTS.md` | ~42 | Codex entry point |

### Session Skills (6 skills + 6 command wrappers)

Located in `.agents/skills/<name>/SKILL.md` with `.claude/commands/<name>.md` wrappers.

| Skill | Purpose |
|-------|---------|
| `session-start` | Normalize to main, read context files, summarize state |
| `session-transition` | Activate plan in PLANS.md, create feature branch |
| `session-implement` | Confirm readiness, implement, test, update docs |
| `session-review` | Ruff/Pyright/pytest gates, doc-update verification |
| `session-end` | Commit, push, PR creation, squash-merge |
| `session-debug` | Root-cause debugging workflow |

### Problems

1. **Duplication** — session skills overlap substantially with superpowers skills (brainstorming, writing-plans, executing-plans, verification-before-completion, finishing-a-development-branch, systematic-debugging)
2. **File sprawl** — 6 repo-root `.md` files with overlapping concerns (STATUS.md vs auto-memory, MEMORY.md reference vs code-derivable facts, PLANS.md vs superpowers plan files)
3. **Missing capabilities** — session skills lack brainstorming/design phase, TDD enforcement, subagent-driven execution, spec/code review subagents, git worktree isolation
4. **Codex gap** — Codex uses the same session skills but can't run superpowers tools; needs its own workflow

---

## Design

### 1. File Consolidation

**Keep (3 files):**

| File | Role | Changes |
|------|------|---------|
| `CLAUDE.md` | Authoritative project rulebook | Absorbs STATUS.md + MEMORY.md reference content; gains workflow/superpowers section |
| `DECISIONS.md` | Append-only ADR log | No changes |
| `AGENTS.md` | Codex-specific entry point | Full rewrite with Codex-compatible workflow |

**Retire (3 files):**

| File | Destination |
|------|-------------|
| `STATUS.md` | Capabilities/constraints → CLAUDE.md §9 |
| `PLANS.md` | Retired — superpowers manages `docs/plans/` |
| `MEMORY.md` | See MEMORY.md content disposition table below for per-section keep/drop decisions |

### MEMORY.md Content Disposition

| Section | Action | Destination |
|---------|--------|-------------|
| Technology Stack (table) | Absorb verbatim | CLAUDE.md §8.1 |
| Repository Layout (tree) | Absorb with updates (remove retired files from listing, update migration range) | CLAUDE.md §8.2 |
| Data Model (full field lists) | Absorb condensed (model name + key fields + one-line purpose) | CLAUDE.md §8.3 |
| Signal Processing Parameters (table) | Absorb verbatim | CLAUDE.md §8.4 |
| Processing Pipeline Diagram (Mermaid) | Absorb verbatim | CLAUDE.md §8.4 (after parameter table) |
| Storage Layout (tree) | Absorb verbatim | CLAUDE.md §8.5 |
| Runtime Configuration | Absorb condensed (keep Settings env-var prefix, .env loading, key defaults) | CLAUDE.md §8.6 |
| Worker priority order | Absorb verbatim | CLAUDE.md §8.7 "Behavioral Constraints" |
| Hydrophone prefetch semantics (time_covered_sec) | Absorb verbatim | CLAUDE.md §8.7 |
| Parquet row-store lazy upgrade | Absorb verbatim | CLAUDE.md §8.7 |
| Queue safety note (SQLite no row-level locks) | Absorb verbatim | CLAUDE.md §8.7 |
| Windowing Rules (overlap-back strategy, minimum duration, per-component enforcement) | Absorb verbatim | CLAUDE.md §8.4 (after signal params table, before Mermaid diagram) |
| Queue claim semantics (compare-and-set), status transitions, concurrency rules | Absorb verbatim | CLAUDE.md §8.7 "Behavioral Constraints" |
| Processing Workflow (step-by-step) | Drop | Derivable from worker code |
| Clustering Workflow (step-by-step) | Drop | Derivable from engine code |
| Label Processing Workflow (step-by-step) | Drop | Derivable from worker code |
| API Validation Contracts (full list) | Drop | Derivable from route handlers + schemas |
| Web UI Requirements | Drop | Derivable from frontend components |

### PLANS.md Backlog Migration

The backlog items from PLANS.md will be moved to `docs/plans/backlog.md` as a simple bullet list. This preserves the backlog without requiring PLANS.md to continue existing.

### 2. CLAUDE.md Structure

```
## 1. Purpose (existing)
## 2. High-Level Architecture (existing)
## 3. Core Development Rules
   3.1 Package Management
   3.2 Environment Commands
   3.3 Running Python Code and Tools
   3.4 Best Practices
   3.5 Database Migrations
   3.6 Documentation (UPDATED — new file list, new update triggers)
   3.7 Frontend Stack & Development
   3.8 Timezone and Timestamp Standard
## 4. Core Design Principles
   4.1–4.7 (existing)
## 5. Testing Requirements
   5.1–5.5 (existing)
## 6. Definition of Done (existing)
## 7. Non-Goals (existing)
## 8. Project Reference (NEW — from MEMORY.md)
   8.1 Technology Stack
   8.2 Repository Layout
   8.3 Data Model Summary (condensed: model names, key fields, relationships)
   8.4 Signal Processing Parameters + Processing Pipeline Diagram (Mermaid)
   8.5 Storage Layout
   8.6 Runtime Configuration (condensed)
   8.7 Behavioral Constraints (worker priority, prefetch semantics, row-store upgrade, queue safety)
## 9. Current State (NEW — from STATUS.md)
   9.1 Implemented Capabilities (condensed bullet list)
   9.2 Database Schema (engine, latest migration, table list)
   9.3 Sensitive Components
   9.4 Known Constraints
## 10. Workflow (NEW)
   10.1 Superpowers Integration
   10.2 Session Start Checklist
   10.3 Project Verification Gates
   10.4 Codex Compatibility
```

#### Section 3.6 (Documentation) — Updated

```markdown
When a change adds, removes, or modifies API endpoints, data models, configuration,
architecture, or workflows, update the relevant files:

- `CLAUDE.md` — rules, reference material, project state (this file)
- `DECISIONS.md` — append new ADR for significant architecture changes
- `README.md` — user-facing API endpoints, configuration, feature list
- `docs/specs/` — design specs (written during brainstorming phase)
- `docs/plans/` — implementation plans (written during planning phase)
```

#### Section 8.3 — Data Model (Condensed)

Include ALL models from the current MEMORY.md Data Model section. For each: model name, key fields (not exhaustive), and one-line purpose. Drop the full field-by-field enumeration that is derivable from `database.py`.

All models to include: ModelConfig, AudioFile, AudioMetadata, ProcessingJob, EmbeddingSet, SearchJob, LabelProcessingJob, ClusteringJob, Cluster, ClusterAssignment, ClassifierModel, ClassifierTrainingJob, DetectionJob.

Example format:
```
- **ModelConfig** — ML model registry entry (name, path, vector_dim, model_type, input_format)
- **AudioFile** — uploaded/imported audio (filename, folder_path, source_folder, checksum, duration)
- **EmbeddingSet** — one per audio+encoding_signature (parquet_path, model_version, vector_dim)
- **SearchJob** — ephemeral similarity search (deleted after results returned)
  ...
```

#### Section 9.1 — Capabilities (Condensed)

One bullet per capability bucket instead of multi-paragraph prose:
```
- Audio upload, folder import, metadata editing
- Processing pipeline: TFLite + TF2 SavedModel, overlap-back windowing, incremental Parquet
- Embedding similarity search (cosine/euclidean, cross-set, detection-sourced)
- Clustering: HDBSCAN/K-Means/Agglomerative, UMAP/PCA, parameter sweeps, metric learning
- Binary classifier training + local/hydrophone detection
- Hydrophone streaming: Orcasound HLS + NOAA archives, pause/resume/cancel, subprocess isolation
- Label processing: score-based + sample-builder workflows
- Vocalization labeling: type classification, active learning, sub-window annotations
- Retrain workflow: reimport → reprocess → retrain
- Web UI: routed SPA with Audio, Processing, Clustering, Classifier, Search, Label Processing, Admin
```

#### Section 10.1 — Superpowers Integration

```markdown
This project uses the superpowers skill system as its canonical development workflow.

**Canonical flow for every task:**
  brainstorming → writing-plans → subagent-driven-development → finishing-a-development-branch

**During implementation (enforced by subagent-driven-development):**
- test-driven-development (per task — write failing test first)
- requesting-code-review (per task + final review)
- verification-before-completion (before any completion claim)

**When debugging:**
- systematic-debugging (before any fix attempt)

**Artifact locations:**
- Design specs: `docs/specs/YYYY-MM-DD-<topic>-design.md`
- Implementation plans: `docs/plans/YYYY-MM-DD-<feature>.md`
- Git worktrees: `.worktrees/` (gitignored)
```

#### Section 10.2 — Session Start Checklist

```markdown
At the start of every session:
1. Normalize the repo onto local `main` (fast-forward from origin; stop if dirty or detached)
2. Read CLAUDE.md and DECISIONS.md
3. Check `docs/plans/` for active work
4. Summarize current state for the user
5. Resume active plan work, or begin superpowers brainstorming for the next task
```

#### Section 10.3 — Project Verification Gates

```markdown
Before claiming work is complete, run these in order:
1. `uv run ruff format --check` on modified Python files
2. `uv run ruff check` on modified Python files
3. `uv run pyright` on modified Python files (full run if pyproject.toml changed)
4. `uv run pytest tests/`
5. `cd frontend && npx tsc --noEmit` (if frontend files changed)

Doc-update matrix:
| Change type                  | Update                            |
|------------------------------|-----------------------------------|
| API endpoints added/changed  | CLAUDE.md §8, README.md           |
| Data model changed           | CLAUDE.md §8.3, Alembic migration |
| Signal processing changed    | CLAUDE.md §8.4, DECISIONS.md      |
| New capability               | CLAUDE.md §9.1                    |
| Constraint changed           | CLAUDE.md §9.4                    |
| Architecture decision        | DECISIONS.md                      |
| Frontend routes/components   | CLAUDE.md §3.7                    |
```

#### Section 10.4 — Codex Compatibility

```markdown
Codex follows the same phase sequence as superpowers but uses only Codex-available
tools (file read/write, bash, grep, glob). See AGENTS.md for Codex-specific
workflow instructions.
```

### 3. AGENTS.md — Codex Workflow

Full rewrite. Codex follows the same 6-phase sequence as superpowers:

```markdown
# Humpback Acoustic Embed — Codex Agent Instructions

CLAUDE.md is the authoritative project rulebook — read it first.

## Codex Workflow

Follow these phases in order for any task:

### Phase 1: Context
- Read CLAUDE.md (rules + reference)
- Read DECISIONS.md (recent ADRs)
- Check docs/plans/ for active work
- Understand what's being asked before acting

### Phase 2: Design
- For new features or significant changes:
  - Explore affected code
  - Identify 2-3 approaches with trade-offs
  - Write a design spec to docs/specs/YYYY-MM-DD-<topic>-design.md
  - Commit the spec before implementing
- For bug fixes: skip to Phase 3 after root-cause investigation

### Phase 3: Plan
- Write an implementation plan to docs/plans/YYYY-MM-DD-<feature>.md
- Break into small tasks (2-5 minutes each)
- Include affected files, test strategy, verification commands
- Commit the plan before implementing

### Phase 4: Implement
- Create a feature branch: codex/<slug>
- TDD: write failing test -> implement -> verify green -> refactor
- One task at a time, commit after each
- Follow all CLAUDE.md rules (uv, migrations, doc updates)

### Phase 5: Verify
- Run the project verification gates (CLAUDE.md §10.3)
- All must pass before proceeding

### Phase 6: Finish
- Push branch, create PR targeting main
- PR body includes: summary, test plan, verification results

## Key Constraints

- Package manager: uv only (never pip/conda/poetry)
- Frontend: npm from frontend/ directory
- DB migrations: Alembic with op.batch_alter_table() for SQLite
- Testing: every change needs tests (uv run pytest tests/)
- Idempotency: never create duplicate embedding sets for same encoding_signature
```

### 4. File Deletion

Delete all session-* skills and command wrappers:

```
.agents/skills/session-start/SKILL.md
.agents/skills/session-transition/SKILL.md
.agents/skills/session-implement/SKILL.md
.agents/skills/session-review/SKILL.md
.agents/skills/session-end/SKILL.md
.agents/skills/session-debug/SKILL.md
.claude/commands/session-start.md
.claude/commands/session-transition.md
.claude/commands/session-implement.md
.claude/commands/session-review.md
.claude/commands/session-end.md
.claude/commands/session-debug.md
```

Delete the `.agents/` directory entirely (it only contains `skills/` subdirectories and `.DS_Store` after skill deletion). Delete the `.claude/commands/` subdirectory only. Do NOT delete `.claude/` itself — it contains `hooks/`, `settings.json`, and `settings.local.json`.

Delete retired repo-root files:
```
STATUS.md
PLANS.md
MEMORY.md
```

### 5. New Directories & Gitignore

Ensure:
- `docs/specs/` exists (already created with this spec)
- `docs/plans/` exists (create it; will contain `backlog.md` migrated from PLANS.md)
- `.worktrees/` entry added to `.gitignore`

### 6. Auto-Memory Update

Update `~/.claude/projects/-Users-michael-development-humpback-acoustic-embed/memory/MEMORY.md` to:

```markdown
# Humpback Acoustic Embed — Session Memory

## Project Documentation Structure
- `CLAUDE.md` — rules, reference material, project state (auto-loaded)
- `DECISIONS.md` — append-only architecture decision log
- `AGENTS.md` — Codex entry point with phase-based workflow
- `docs/specs/` — design specs from brainstorming
- `docs/plans/` — implementation plans + backlog

## Workflow
- Canonical flow: brainstorming -> writing-plans -> subagent-driven-development -> finishing-a-development-branch
- TDD enforced for all implementation
- Git worktrees in .worktrees/ for isolation
- Codex follows same phases (see AGENTS.md)

## Key Patterns
- Package manager: `uv` only (never pip)
- Frontend: `npm` from `frontend/`
- DB migrations: Alembic with `op.batch_alter_table()` for SQLite
- Tests: `uv run pytest tests/`

## TFLite Performance
- See `project-tflite-perf-benchmark.md` — perch_v2 benchmark results, CoreML status

## humpback-hoplite Sibling Project
- Location: `~/development/humpback-hoplite/`
- Standalone CLI for Perch+Hoplite vector search/clustering experiments
- perch_v2.tflite produces **1536-d** embeddings (not 1280 as documented)
- Confirmed incompatibility: tensorflow-macos 2.16 (numpy<2.0) vs perch-hoplite (numpy>=2.0)

## Classifier Training Parameters
- `classifier_type`: `"logistic_regression"` (default) or `"mlp"` (ADR-007)
- `l2_normalize`: `False` (default), `True` prepends `Normalizer(norm="l2")` to pipeline
- `class_weight`: `"balanced"` (default) or `None`
- `C`: regularization strength for LogisticRegression (default 1.0)
```

Corrections applied during update:
- Removed stale reference to `.agents/workflows/` (directory does not exist)
- Removed stale `/project:*` command names (replaced by superpowers workflow)
- Removed stale "Latest migration: 024" (derivable from code; actual latest is 025)
- Removed references to retired files (STATUS.md, PLANS.md, MEMORY.md)

### 7. CLAUDE.md Preamble Update

The existing CLAUDE.md preamble (lines 17-28) contains a "Memory Files" table referencing STATUS.md, MEMORY.md, PLANS.md, and DECISIONS.md. This table must be removed during implementation and replaced with the updated §3.6 documentation list. The new preamble should state:

```markdown
This document defines behavioral rules, engineering constraints, project reference
material, and workflow integration. For architecture decisions, see `DECISIONS.md`.
For Codex-specific workflow, see `AGENTS.md`.
```

### 8. ADR for This Change

Append a new ADR to DECISIONS.md documenting this restructuring:

```markdown
## ADR-0XX: Adopt superpowers workflow, consolidate documentation

**Date**: 2026-03-24
**Status**: Accepted

**Context**: The project had 6 repo-root .md files with overlapping concerns and
6 custom session-* skills that duplicated superpowers functionality while missing
key capabilities (brainstorming, TDD enforcement, subagent execution, code review).

**Decision**: Adopt superpowers as the canonical workflow. Consolidate to 3 repo-root
files (CLAUDE.md, DECISIONS.md, AGENTS.md). Move specs to docs/specs/, plans to
docs/plans/. Rewrite AGENTS.md for Codex-compatible workflow.

**Consequences**:
- Single workflow system instead of two competing ones
- CLAUDE.md is larger (~450 lines) but self-contained
- Codex follows same phase sequence with its own tooling
- Session-* skills deleted; all workflow orchestration via superpowers
- Backlog items preserved in docs/plans/backlog.md
```

---

## What Does NOT Change

- `DECISIONS.md` — untouched
- `README.md` — untouched
- `CLAUDE.md` sections 1-7 — existing rules preserved exactly (only §3.6 gets a small update)
- `.gitignore` — only adds `.worktrees/`
- Frontend, backend, tests — no code changes
- Pre-commit hooks, Ruff, Pyright config — unchanged

---

## Success Criteria

1. Repo has 3 root `.md` files: CLAUDE.md, DECISIONS.md, AGENTS.md
2. No session-* skills or command wrappers exist
3. `docs/specs/` and `docs/plans/` directories exist
4. `.worktrees/` is gitignored
5. CLAUDE.md contains all non-derivable reference material from retired files
6. AGENTS.md provides a self-contained Codex workflow
7. Superpowers skills can read CLAUDE.md and find project-specific rules (verification gates, doc-update matrix, TDD requirement)
8. Auto-memory index reflects the new structure
