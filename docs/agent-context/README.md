# Agent Context

This directory is the domain-local context layer for coding agents.

`CLAUDE.md` remains the global project contract: package manager rules, database
backup rules, universal invariants, workflow, and final verification gates. Do
not add detailed API listings, schema inventories, storage trees, or product
state narratives there.

Use this directory when a task needs project detail:

1. Start with `domain-map.md`.
2. Choose one primary domain and, when needed, one neighbor domain.
3. Read that domain's `README.md`, `invariants.md`, `tests.md`, and
   `references.md`.
4. Read detailed files under `docs/reference/` only when the domain capsule
   points there for the current task.

## Files

- `domain-map.md` maps paths and task keywords to domain capsules.
- `global-invariants.md` lists cross-domain correctness rules.
- `current-state.md` summarizes active project capabilities by domain.
- `test-map.md` gives targeted verification commands by domain.
- `domains/*/` contains the local context capsules.

## Domains

- `core-platform`: database, settings, storage helpers, queue semantics,
  migrations, global models and schemas.
- `signal-timeline`: audio IO, DSP, spectrograms, PCEN, timeline cache,
  timeline API, timeline UI primitives.
- `ingest-detection`: hydrophone providers, detection jobs, detection
  embeddings, classifier training, hyperparameter tuning.
- `vocalization-clustering`: vocalization vocabulary, labels, training
  datasets, multi-label models, clustering.
- `call-parsing`: Pass 1 regions, Pass 2 segmentation, Pass 3 classification,
  corrections, feedback training, window classify.
- `sequence-models`: Continuous Embedding jobs and CRNN region embedding
  helpers.
- `frontend-shell`: app shell, shared components, query hooks, admin UI,
  navigation, frontend-only shared behavior.

## Update Rule

When a future change affects agent operating context, update the smallest
matching domain capsule first. Update `CLAUDE.md` only when every agent must
load the rule before knowing the task domain.
