# Agent-Local Project Structure Design

**Status:** Draft for discussion
**Date:** 2026-05-07
**Track:** Documentation and agent workflow structure

---

## 1. Goal

Restructure project documentation so coding agents operate with small,
domain-local context instead of loading broad global reference material.

The reorganization should optimize for coding-agent performance rather than
traditional human SDLC layout:

- Agents should load only the rules, invariants, paths, and test commands for
  the domain they are changing.
- `CLAUDE.md` should remain a small global contract and context router, not a
  project encyclopedia.
- Detailed references should move behind domain-specific entry points.
- Targeted test verification should become a first-class workflow, while the
  full suite remains the final safety gate.

This is a documentation and structure task. It should not perform broad runtime
code refactors.

---

## 2. Current Findings

### 2.1 Global Context Load

`CLAUDE.md` is already much leaner than the historical 747-line version, but it
still carries broad project state and cross-domain pointers:

- Current size: 344 lines.
- It includes global rules, workflow, technology summary, repository layout,
  current state, schema summary, sensitive components, and verification gates.
- The "Current State" section contains long, detailed paragraphs for Sequence
  Models and Call Parsing. These are high-value references, but most tasks do
  not need them loaded upfront.
- Section 8 is a reference index, but the linked files remain organized mostly
  by reference type, not by agent work domain.

The result is an improved but still global startup context. Agents must
mentally route from global summaries to relevant references, which costs tokens
and increases the chance of loading unrelated detail.

### 2.2 Reference Docs

Current `docs/reference/` files:

- `behavioral-constraints.md`
- `call-parsing-api.md`
- `classifier-api.md`
- `data-model.md`
- `frontend.md`
- `hydrophone-rules.md`
- `runtime-config.md`
- `sequence-models-api.md`
- `signal-processing.md`
- `storage-layout.md`
- `testing.md`

These are useful, but several are cross-cutting:

- `behavioral-constraints.md` mixes job system, detection, classifier,
  timeline, call parsing, feedback training, sequence models, and frontend
  timeline architecture.
- `data-model.md` lists many tables across every product area.
- `storage-layout.md` lists all artifact roots rather than the artifact roots
  relevant to one task domain.
- `testing.md` defines broad policy but does not provide fast domain commands.

For humans, this is a sensible reference shelf. For agents, it encourages
loading broad files when a small domain capsule would be enough.

### 2.3 Source Layout

The runtime code already has strong domain seams:

- `src/humpback/call_parsing/`
- `src/humpback/classifier/`
- `src/humpback/sequence_models/`
- `src/humpback/processing/`
- `src/humpback/sample_builder/`
- `src/humpback/clustering/`
- `src/humpback/api/routers/`
- `src/humpback/services/`
- `src/humpback/workers/`
- `src/humpback/models/`
- `src/humpback/schemas/`
- `frontend/src/components/{call-parsing,classifier,sequence-models,timeline,vocalization}`

The recommended documentation structure should align with these co-change
areas without moving runtime modules.

### 2.4 Test Layout

Backend tests currently include 158 `test_*.py` files:

- 122 under `tests/unit/`
- 23 under `tests/integration/`
- smaller topical directories such as `tests/call_parsing/`,
  `tests/sequence_models/`, `tests/processing/`, `tests/services/`, and
  `tests/workers/`

Frontend has 23 Playwright spec files, mostly feature-oriented.

`pyproject.toml` has no domain markers. `session-implement.md` already
describes targeted inline testing by path convention plus background full-suite
testing, but the current test tree does not expose domain suites as named,
documented commands.

---

## 3. Non-Goals

- Do not reorganize runtime source packages.
- Do not rename Python modules, React components, API routes, database tables,
  or storage artifact roots.
- Do not rewrite historical specs, plans, or ADRs.
- Do not remove the full verification gate from `session-review`.
- Do not turn this into a CI infrastructure project unless explicitly planned
  later.

---

## 4. Agent-Efficient Domain Model

Domains should be defined by agent co-change radius: the files, invariants,
references, and tests that usually need to be loaded together for a task.

Recommended initial domains:

| Domain | Primary scope | Typical files |
|--------|---------------|---------------|
| `core-platform` | Database, settings, storage helpers, generic job status, queue semantics, migrations | `database.py`, `models/`, `schemas/`, `storage.py`, `config.py`, `alembic/`, `workers/queue.py` |
| `signal-timeline` | Audio IO, DSP, spectrograms, PCEN, timeline cache, timeline API, shared timeline UI | `processing/`, `services/timeline_*`, `api/routers/timeline.py`, `frontend/src/components/timeline/` |
| `ingest-detection` | Hydrophone providers, detection jobs, detection embeddings, classifier training, hyperparameter tuning | `classifier/`, `services/classifier_service/`, classifier routers, classifier workers |
| `vocalization-clustering` | Vocalization vocabulary, labels, training datasets, multi-label models, clustering | `services/vocalization_service.py`, `clustering/`, vocalization router, vocalization UI |
| `call-parsing` | Pass 1 regions, Pass 2 segmentation, Pass 3 classification, corrections, feedback training, window classify | `call_parsing/`, `services/call_parsing.py`, `api/routers/call_parsing.py`, relevant workers, call-parsing UI |
| `sequence-models` | Continuous embedding producer and CRNN region embedding helpers | `sequence_models/`, `services/continuous_embedding_service.py`, `workers/continuous_embedding_worker.py`, sequence models router/UI |
| `frontend-shell` | Navigation, shared UI, admin, query hooks shared across domains | `frontend/src/components/layout/`, `frontend/src/components/shared/`, `frontend/src/api/`, `frontend/src/hooks/queries/` |

Most work should load one primary domain and at most one neighbor domain. For
example, a Sequence Models label-source change likely loads `sequence-models`
and `call-parsing`; a timeline playback fix likely loads `signal-timeline` and
maybe `frontend-shell`.

---

## 5. Recommended Approach: Domain Capsules

Create a new agent-facing context layer:

```text
docs/agent-context/
  README.md
  domain-map.md
  test-map.md
  global-invariants.md
  current-state.md
  domains/
    core-platform/
      README.md
      invariants.md
      tests.md
      references.md
    signal-timeline/
      README.md
      invariants.md
      tests.md
      references.md
    ingest-detection/
      README.md
      invariants.md
      tests.md
      references.md
    vocalization-clustering/
      README.md
      invariants.md
      tests.md
      references.md
    call-parsing/
      README.md
      invariants.md
      tests.md
      references.md
    sequence-models/
      README.md
      invariants.md
      tests.md
      references.md
    frontend-shell/
      README.md
      invariants.md
      tests.md
      references.md
```

Each domain capsule should be short and operational:

- **When to load:** file path triggers and task keywords.
- **Local map:** source, API, services, workers, models, schemas, frontend,
  docs, and artifact roots for the domain.
- **Local invariants:** only the non-obvious rules that can break correctness.
- **Test commands:** the smallest useful smoke command, the normal domain
  command, and expansion commands when API/frontend/storage are touched.
- **Reference pointers:** exact files or sections to read only when needed.
- **ADR pointers:** ADR titles relevant to the domain, not full ADR text.
- **Doc update targets:** which reference or domain files must change when the
  domain behavior changes.

The capsule files should not duplicate all existing reference docs. They should
route agents to the smallest necessary detail and copy only critical local
invariants that agents must see before editing.

---

## 6. `CLAUDE.md` Changes

Convert `CLAUDE.md` from a mixed rulebook/reference/current-state document into
a small global contract:

Keep globally loaded:

- Purpose and architecture in very short form.
- Universal development rules: `uv`, frontend `npm`, database backup rule,
  migrations, UTC timestamp rule, documentation update obligation.
- Universal invariants: idempotent derived artifacts, resumable workflows,
  async observable jobs.
- Session workflow pointer.
- Final verification gates.
- Domain context router pointing to `docs/agent-context/README.md` and
  `docs/agent-context/domain-map.md`.

Move out of global load:

- Long current-state bullets into `docs/agent-context/current-state.md`, split
  by domain.
- The full database table list into `core-platform/references.md` or a
  domain-indexed data model reference.
- The sensitive components table into domain capsules, with a very short
  global note that sensitive component ownership is domain-local.
- API surface links into domain capsules.
- Timeline compound-component rules into `signal-timeline/invariants.md`.
- Sequence Models detail paragraphs into `sequence-models/README.md` and
  `sequence-models/invariants.md`.

Target size: roughly 120-180 lines, depending on how much workflow text remains
global.

---

## 7. Reference Doc Reorganization

Do this in two layers to avoid churn.

### 7.1 First Pass: Add Agent Context Without Moving Existing Reference Docs

Add `docs/agent-context/` and let capsules point to existing
`docs/reference/*.md` sections.

This gives agents an immediate small-context entry point and avoids breaking
links from historical specs, plans, and ADRs.

### 7.2 Second Pass: Split Cross-Cutting References By Domain

Once capsules prove stable, split broad reference docs into domain-local files:

```text
docs/reference/domains/
  core-platform/
    data-model.md
    storage-layout.md
    runtime-config.md
    jobs.md
  signal-timeline/
    signal-processing.md
    timeline-rendering.md
    frontend-timeline.md
  ingest-detection/
    hydrophone-rules.md
    classifier-api.md
    detection-artifacts.md
  vocalization-clustering/
    labels-training-clustering.md
  call-parsing/
    api.md
    artifacts.md
    corrections-feedback.md
  sequence-models/
    api.md
    continuous-embedding-artifacts.md
```

Keep top-level `docs/reference/*.md` as compatibility indexes that point to the
new domain files rather than deleting them immediately.

---

## 8. Targeted Test Strategy

The goal is not to weaken verification. The goal is to make agents run the
right tests early and cheaply, then keep the full suite as the final gate.

### 8.1 Add Domain Test Maps

Each capsule gets `tests.md` with commands like:

```text
# call-parsing
uv run pytest tests/call_parsing tests/unit/test_call_parsing_* tests/integration/test_call_parsing_router.py -q

# sequence-models
uv run pytest tests/sequence_models tests/unit/test_sequence_models_schemas.py tests/services/test_continuous_embedding_service.py tests/workers/test_continuous_embedding_worker.py -q

# signal-timeline
uv run pytest tests/processing tests/unit/test_timeline_* tests/unit/test_spectrogram.py tests/unit/test_pcen_rendering.py tests/integration/test_timeline_api.py -q
```

The exact lists should be verified during implementation. The important part is
that agents no longer infer commands from memory or run the full suite as the
first feedback loop.

### 8.2 Add Optional Pytest Markers Later

After the domain maps are useful, add markers:

- `core_platform`
- `signal_timeline`
- `ingest_detection`
- `vocalization_clustering`
- `call_parsing`
- `sequence_models`
- `frontend_shell`

This allows:

```bash
uv run pytest -m call_parsing -q
uv run pytest -m "call_parsing or sequence_models" -q
```

Markers can be added incrementally by file or directory. They should supplement
path commands, not replace them immediately.

### 8.3 Add Make Targets Or Script Aliases

Add stable names for agents and humans:

```text
make test-domain DOMAIN=call-parsing
make test-domain DOMAIN=sequence-models
make test-related FILES="src/humpback/call_parsing/storage.py"
```

`test-related` can start as documentation only. A later implementation could
use `docs/agent-context/test-map.md` or a small script to map changed files to
tests.

### 8.4 Preserve Full Gates

`session-implement` keeps targeted inline tests plus background full suite.
`session-review` keeps `uv run pytest tests/` as the authoritative gate.

The benefit is faster local feedback, not less coverage.

---

## 9. Alternatives Considered

### Option A: Only Slim `CLAUDE.md`

Move the long current-state and schema sections out of `CLAUDE.md`, but leave
`docs/reference/` organized as it is.

Pros:

- Smallest change.
- Low link churn.
- Reduces startup tokens quickly.

Cons:

- Agents still load broad reference docs after startup.
- Does not solve targeted test discovery.
- Keeps the burden on agents to infer the right domain context.

### Option B: Domain Capsules With Compatibility References

Add `docs/agent-context/` and slim `CLAUDE.md`; keep existing references at
first, then gradually split cross-cutting references by domain.

Pros:

- Best balance of token reduction and low runtime risk.
- No code refactor.
- Supports domain-local tests immediately.
- Allows iterative migration without breaking historical links.

Cons:

- Some duplication risk between capsules and references.
- Requires discipline to keep capsules short and operational.
- Needs a doc-update rule so new constraints land in the domain capsule, not
  only in broad references.

### Option C: Physically Reorganize Source And Tests By Domain

Move source, tests, docs, and frontend files into feature/domain packages.

Pros:

- Strongest locality in the long term.
- Natural directory-based agent context and pytest selection.

Cons:

- Broad code refactor, explicitly out of scope.
- High import churn and test churn.
- Risks distracting from actual product work.

Recommendation: choose Option B.

---

## 10. Proposed Implementation Phases

### Phase 1: Add Agent Context Layer

Files:

- `docs/agent-context/README.md`
- `docs/agent-context/domain-map.md`
- `docs/agent-context/test-map.md`
- `docs/agent-context/global-invariants.md`
- `docs/agent-context/current-state.md`
- `docs/agent-context/domains/*/{README.md,invariants.md,tests.md,references.md}`

Acceptance criteria:

- Each domain capsule is readable in under a few hundred lines total.
- Each capsule names source paths, frontend paths, relevant reference docs,
  relevant ADR headings, and test commands.
- `domain-map.md` tells an agent which capsule to load from a changed file path.

### Phase 2: Slim Global Instructions

Files:

- `CLAUDE.md`
- `AGENTS.md`
- `docs/workflows/session-begin.md`
- `docs/workflows/session-implement.md`
- `docs/workflows/session-review.md`

Acceptance criteria:

- `CLAUDE.md` contains global rules and a domain context router only.
- Workflows tell agents to load the domain capsule for the affected area before
  planning or implementing.
- Current state lives in `docs/agent-context/current-state.md` by domain.

### Phase 3: Make Targeted Tests First-Class

Files:

- `docs/agent-context/test-map.md`
- domain `tests.md` files
- optional `Makefile`
- optional `pyproject.toml`

Acceptance criteria:

- Every domain has a documented fast test command and a broader domain command.
- `session-implement` refers to `docs/agent-context/test-map.md` before using
  generic filename heuristics.
- Full-suite review remains unchanged.

### Phase 4: Split Broad References By Domain

Files:

- `docs/reference/domains/**`
- top-level `docs/reference/*.md` compatibility indexes

Acceptance criteria:

- Broad files such as `behavioral-constraints.md`, `data-model.md`, and
  `storage-layout.md` no longer force agents to read unrelated domains.
- Historical links continue to resolve through compatibility indexes.

---

## 11. Update Rules After Reorganization

When a future task changes behavior:

1. Update the domain capsule if the change affects agent operating rules,
   invariants, paths, or test commands.
2. Update domain reference docs if the change affects detailed APIs, schema,
   storage layout, runtime config, signal processing, or frontend architecture.
3. Update `docs/agent-context/current-state.md` if the change alters active
   capabilities.
4. Update `CLAUDE.md` only for universal rules that every agent must load.
5. Add an ADR only for significant architecture decisions or non-obvious
   trade-offs.

This rule is the key behavior change: global context becomes the exception.

---

## 12. Open Questions

- Should domain capsules be named `README.md` files only, or should each domain
  use `AGENTS.md` as well for tools that auto-load directory-local agent rules?
- Should `docs/plans/` and `docs/specs/` remain date-flat, or should new plans
  include a domain tag in their filename such as
  `2026-05-07-call-parsing-<topic>.md`?
- Should pytest markers be added manually, or should a lightweight collection
  hook infer them from `docs/agent-context/test-map.md`?
- Should the project add a small `scripts/test_related.py` helper, or keep
  targeted testing as documented commands only?
