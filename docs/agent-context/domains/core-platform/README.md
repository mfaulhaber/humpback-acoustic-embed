# Core Platform Domain

Load this domain for database, configuration, storage helpers, global models and
schemas, Alembic migrations, worker queue semantics, session workflows, or
agent-instruction changes.

## Primary Paths

- `src/humpback/database.py`
- `src/humpback/config.py`
- `src/humpback/storage.py`
- `src/humpback/models/`
- `src/humpback/schemas/`
- `src/humpback/workers/queue.py`
- `src/humpback/workers/runner.py`
- `alembic/`
- `pyproject.toml`
- `Makefile`
- `CLAUDE.md`
- `AGENTS.md`
- `docs/workflows/`
- `docs/agent-context/`

## Artifact Roots

- Shared helpers in `src/humpback/storage.py` define canonical roots for audio,
  detections, classifiers, clusters, training datasets, continuous embeddings,
  hyperparameter artifacts, and cleanup manifests.

## Likely Neighbors

- Load the owning feature domain when changing a domain-specific model, schema,
  storage helper, or worker claim.
- Load `frontend-shell` when workflow or API contract changes affect frontend
  expectations.

## Before Editing

1. Check whether the task touches a database schema or persisted artifact.
2. If yes, read `invariants.md` and the owning feature domain invariants.
3. If a schema change is involved, perform the database backup step before any
   migration or data modification.
