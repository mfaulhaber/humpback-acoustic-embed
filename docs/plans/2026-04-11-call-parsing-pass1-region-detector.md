# Call Parsing — Pass 1: Region Detector (Skeletal Plan)

**Status:** Skeletal — requires a Pass 1 brainstorm cycle before execution.
**Goal:** Implement the Pass 1 `RegionDetectionWorker` so it produces `trace.parquet` and `regions.parquet` from a source audio file using the shared `compute_hysteresis_events` helper.
**Architecture inherits from:** [Phase 0 spec](../specs/2026-04-11-call-parsing-pipeline-phase0-design.md)
**Pass 1 design spec (to be written):** `docs/specs/YYYY-MM-DD-call-parsing-pass1-design.md` — produced by the Pass 1 brainstorm; this plan gets elaborated afterward.

---

## Inherited from Phase 0 (do NOT re-derive)

- Table `region_detection_jobs` with standard queue columns, `parent_run_id` FK, `trace_row_count`, `region_count`
- Worker shell at `src/humpback/workers/region_detection_worker.py` (claims and fails with `NotImplementedError`)
- Shared helper `compute_hysteresis_events(audio, sr, perch_model, classifier, config)` importable from `humpback.classifier.detector` (behavior-preserving extract)
- `call_parsing/types.py` defines `Region` and `WindowScore` dataclasses with parquet schemas
- `call_parsing/storage.py` exposes `write_trace` / `read_trace` / `write_regions` / `read_regions` with atomic writes and directory conventions
- Stub API endpoints at `/call-parsing/region-jobs`:
  - `POST` returns 501 (Pass 1 owns unstubbing)
  - `GET` list and detail functional
  - `DELETE` functional
  - `GET /{id}/trace` and `GET /{id}/regions` return 501
- PyTorch available via the TF extras (Pass 1 does not use it, but it's present)

## Brainstorm checklist — Pass 1 TBDs

The Pass 1 brainstorm must settle these before this plan is elaborated into executable tasks:

- [ ] **Padding semantics.** Symmetric pad (± N seconds) or asymmetric (more post-event pad)? Default value(s)? Clamp behavior at audio bounds.
- [ ] **Region overlap-merge rules.** After padding, do two regions merge if padded bounds overlap, or only if the raw-bounds gap is under a threshold? How are `max_score` / `mean_score` / `n_windows` combined after merge?
- [ ] **Temporal smoothing.** Apply a moving-average or Gaussian filter on scores before hysteresis, or rely on raw detector output? If smoothing, window size.
- [ ] **Trace storage strategy.** Write the full dense per-window trace, or decimate to a target frame rate? Impact on Pass 2 consumption.
- [ ] **Minimum region duration.** Drop regions shorter than N seconds? Default.
- [ ] **New config columns on `region_detection_jobs`.** Exact fields and whether an additional Alembic migration is required (likely yes — e.g. `padding_sec`, `merge_gap_sec`, `min_region_duration_sec`, `smoothing_window_sec`).
- [ ] **Worker resume semantics.** On crash mid-job: re-run from scratch or resume from a partial `trace.parquet`?
- [ ] **Audio source resolution.** Does Pass 1 read from uploaded audio files only, or also from hydrophone archive sources like existing detection jobs?

## Tasks (skeletal — expand after brainstorm)

### Task 1: Migration for Pass 1 config columns
New Alembic migration adding the config fields decided in the brainstorm to `region_detection_jobs`. Likely: `padding_sec`, `min_region_duration_sec`, `merge_gap_sec`, `smoothing_window_sec` and/or others.

### Task 2: Region decoder module
**Files:**
- Create: `src/humpback/call_parsing/regions.py`
- Create: `tests/unit/test_call_parsing_regions.py`

**Acceptance criteria (skeletal):**
- [ ] Pure function `decode_regions(hysteresis_events, config) -> list[Region]` implementing the padding and merge semantics from the brainstorm
- [ ] No I/O — takes events in, returns regions out
- [ ] Clamps padded bounds to `[0.0, audio_duration]`
- [ ] Returns regions sorted by `start_sec`
- [ ] Handles empty input and single-event input cleanly

**Tests:** pad/merge unit tests with synthetic event inputs; edge cases (empty, single, two adjacent, events at audio boundaries)

### Task 3: Worker implementation
**Files:**
- Modify: `src/humpback/workers/region_detection_worker.py`
- Create: `tests/integration/test_region_detection_worker.py`

**Acceptance criteria (skeletal):**
- [ ] Worker claims a queued job via atomic CAS (Phase 0 pattern preserved)
- [ ] Loads audio, Perch model, binary classifier per job config
- [ ] Calls `compute_hysteresis_events()` from the refactored detector
- [ ] Writes `trace.parquet` and `regions.parquet` to the per-job storage directory
- [ ] Updates job row with `trace_row_count`, `region_count`
- [ ] Marks `status='complete'` on success, `status='failed'` with error message on exception
- [ ] Cleans up partial artifacts on failure

**Tests:** end-to-end integration using a fixture audio + mock Perch model + mock binary classifier; assert artifacts on disk and job status transitions

### Task 4: Unstub Pass 1 API endpoints
**Files:**
- Modify: `src/humpback/api/routers/call_parsing.py`
- Modify: `src/humpback/services/call_parsing.py`
- Modify: `tests/api/test_call_parsing_router.py`

**Acceptance criteria:**
- [ ] `POST /call-parsing/region-jobs` creates a queued `RegionDetectionJob` (removes the 501 stub)
- [ ] `GET /call-parsing/region-jobs/{id}/trace` streams or JSON-serializes trace rows
- [ ] `GET /call-parsing/region-jobs/{id}/regions` returns regions sorted by start
- [ ] `DELETE` cascades the parquet directory cleanup

### Task 5: Smoke test
New smoke test that creates a region detection job via API, lets the worker process it, queries regions back, and cleans up.

### Task 6: Documentation updates
- CLAUDE.md §9.1: mark Pass 1 implemented
- CLAUDE.md §8.8: update the call parsing endpoints list
- DECISIONS.md: optional ADR if the brainstorm locks in non-obvious behavior

## Verification

1. `uv run ruff format --check` on modified files
2. `uv run ruff check` on modified files
3. `uv run pyright` on modified files
4. `uv run alembic upgrade head`
5. `uv run pytest tests/`
6. Manual: POST a region job via API, confirm worker completes it and artifacts land on disk
