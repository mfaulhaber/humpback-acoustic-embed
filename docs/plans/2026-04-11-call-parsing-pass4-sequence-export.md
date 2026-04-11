# Call Parsing — Pass 4: Sequence Export (Skeletal Plan)

**Status:** Skeletal — requires a short Pass 4 brainstorm (lightest pass in the project; brainstorm may be 15 minutes).
**Goal:** Implement the `GET /call-parsing/runs/{id}/sequence` endpoint so it returns an ordered typed-event sequence from the latest successful Pass 3 child's `typed_events.parquet`.
**Architecture inherits from:** [Phase 0 spec](../specs/2026-04-11-call-parsing-pipeline-phase0-design.md)
**Pass 4 design spec (to be written):** `docs/specs/YYYY-MM-DD-call-parsing-pass4-design.md` (optional — may be absorbed into the plan directly if the brainstorm is trivial)
**Depends on:** Pass 3 complete (need `typed_events.parquet` as input)

---

## Inherited from Phase 0 (do NOT re-derive)

- `GET /call-parsing/runs/{id}/sequence` endpoint stub (returns 501)
- `call_parsing/types.py` defines `TypedEvent` with parquet schema
- `call_parsing/storage.py` exposes `read_typed_events` with directory conventions
- Parent-run → child-pass linkage in `call_parsing_runs` row
- Reserved module `src/humpback/call_parsing/sequence.py` (not yet created)

## Brainstorm checklist — Pass 4 TBDs (short)

- [ ] **Output format.** JSON only, or also TSV/CSV for downstream spreadsheet/notebook consumption?
- [ ] **Filtering query params.** Should clients be able to filter by minimum score? By a type subset? By time range? Which are in v1 and which deferred?
- [ ] **Response shape.** Per-event top-type only (matching the design spec example) or full `type_scores` dict per event?
- [ ] **Pagination.** Raw whale recordings can produce thousands of events over a long run. Stream with cursor pagination, or accept that a single request can return everything?
- [ ] **Scope.** Endpoint only on parent runs, or also on individual `event_classification_jobs` for the standalone-runnable path? (Probably both — cheap to add.)
- [ ] **Error semantics.** What does the endpoint return if the parent run has no successful Pass 3 child yet? 404 vs 409 vs empty list with a warning.

## Tasks (skeletal — expand after brainstorm)

### Task 1: Sequence export module
**Files:**
- Create: `src/humpback/call_parsing/sequence.py`
- Create: `tests/unit/test_sequence_export.py`

**Acceptance criteria (skeletal):**
- [ ] `build_sequence(session, run_id, filters) -> SequenceResponse` — reads the run's latest successful Pass 3 child, loads `typed_events.parquet`, applies filters, returns sorted events
- [ ] Sort order: `(start_sec, end_sec, event_id)` for stable pagination
- [ ] Handles missing Pass 3 gracefully (clear error response)

### Task 2: Unstub the `/sequence` endpoint
**Files:**
- Modify: `src/humpback/api/routers/call_parsing.py`
- Modify: `src/humpback/schemas/call_parsing.py` (add response schemas)
- Modify: `tests/api/test_call_parsing_router.py`

**Acceptance criteria:**
- [ ] `GET /call-parsing/runs/{id}/sequence` returns 200 with the sorted typed-event sequence when Pass 3 is complete
- [ ] Supports the filter query params decided in the brainstorm
- [ ] Returns a descriptive error (not 501) when Pass 3 hasn't run successfully yet
- [ ] Supports the output formats decided in the brainstorm (content negotiation or a `format=` query param)
- [ ] Same endpoint added on individual `event_classification_jobs` if scoped that way in the brainstorm

### Task 3: Smoke test for end-to-end pipeline
**Files:**
- Create or modify: `tests/smoke/test_call_parsing_end_to_end.py`

**Acceptance criteria:**
- [ ] Creates a parent run on a fixture audio using mocks for all three model types (Pass 1, Pass 2, Pass 3)
- [ ] Runs all three workers to completion in sequence
- [ ] Calls `GET /call-parsing/runs/{id}/sequence`
- [ ] Asserts events returned in time-sorted order with expected types from the mock classifier

This is the first full-pipeline smoke test — worth getting right.

### Task 4: Documentation updates
- CLAUDE.md §9.1: mark Pass 4 implemented; note the full four-pass pipeline is end-to-end functional
- CLAUDE.md §8.8: finalize Pass 4 endpoint documentation with the format/filter details
- DECISIONS.md: optional short note if any non-obvious decisions were made about filtering or pagination

## Verification

1. `uv run ruff format --check` on modified files
2. `uv run ruff check` on modified files
3. `uv run pyright` on modified files
4. `uv run pytest tests/`
5. Manual: run the full pipeline (Pass 1 → 2 → 3 → 4) on fixture data end-to-end and verify the sequence endpoint returns sensible output
