# Humpback Acoustic Embed — Agent Instructions

This is the entry point for Codex and other agents. `CLAUDE.md` is the authoritative spec — read it for full behavioral rules.

## Key Constraints

- **Package manager**: `uv` only (never pip/conda/poetry)
- **Frontend**: `npm` from `frontend/` directory
- **Database migrations**: Alembic with `op.batch_alter_table()` for SQLite
- **Testing**: Every change needs tests; `uv run pytest tests/`
- **Idempotency**: Never create duplicate embedding sets for the same encoding_signature
- **Documentation**: Update CLAUDE.md, MEMORY.md, README.md when changing APIs, models, or workflows

## Memory Files

| File | Purpose |
|------|---------|
| `CLAUDE.md` | Behavioral rules, development constraints, testing requirements |
| `MEMORY.md` | Data models, workflows, signal parameters, storage layout |
| `DECISIONS.md` | Architecture decision log (append-only) |
| `STATUS.md` | Current project state, implemented capabilities |
| `PLANS.md` | Active and backlog development plans |

## Session Start

Before coding, read `STATUS.md`, `PLANS.md`, and `DECISIONS.md` to understand current project state. Read `MEMORY.md` when working on data models, workflows, or signal processing. Read `CLAUDE.md` for all development rules.

## Workflows

Project skill workflows live in `.agents/skills/<name>/SKILL.md`.
Claude command wrappers in `.claude/commands/` point directly to those skill files.

For implementation work, the canonical session flow is:
`session-start -> session-transition -> session-implement -> session-review -> session-end`

- `session-start` begins by normalizing the repo onto synced local `main`
  before loading context.
- `session-transition` activates the plan and enforces feature-branch readiness
  before local changes.
- `session-end` runs only after a clean `session-review` and handles
  commit, push, PR creation, and squash-merge wrap-up.
