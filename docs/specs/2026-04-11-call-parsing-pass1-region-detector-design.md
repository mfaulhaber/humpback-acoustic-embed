# Call Parsing — Pass 1: Region Detector

**Date:** 2026-04-11
**Status:** Approved
**Inherits from:** [Phase 0 spec](2026-04-11-call-parsing-pipeline-phase0-design.md) (ADR-048)

---

## Problem

Phase 0 shipped the four-pass call parsing scaffold: tables, worker shells, stub endpoints,
shared `call_parsing/` package, and a behavior-preserving `compute_hysteresis_events` helper
extracted from `detector.py`. The Pass 1 worker shell currently claims jobs and fails them
with `NotImplementedError`.

Pass 1 turns that shell into a real worker. It takes a single audio source — either an
uploaded audio file or a hydrophone time range — runs Perch-based dense inference through
the Phase 0 helper, shapes the hysteresis events into padded regions, and writes
`trace.parquet` + `regions.parquet` under the job's storage directory. These artifacts are
Pass 2's upstream input: Pass 2's segmentation model consumes `regions.parquet` to decide
where to crop variable-length event windows from the source audio.

Three structural constraints shape the design:

1. **Hydrophone is first-class.** Pass 1 almost always runs over 24-hour continuous
   hydrophone ranges, not short uploaded files. A 24 h buffer at 16 kHz float32 is ~5.5 GB,
   which rules out loading the full range into memory. The worker must stream audio through
   Perch inference chunk-by-chunk while producing a single concatenated trace.
2. **The Phase 0 `compute_hysteresis_events` helper is monolithic.** It takes a flat
   `audio: np.ndarray` buffer and returns both per-window scores and hysteresis events in
   one call. Pass 1 needs to run the scoring loop repeatedly across chunks while running
   hysteresis only once on the concatenated trace. A small refactor splits the helper into
   two reusable pieces without changing any existing behavior.
3. **Region shaping is a pure transformation.** Turning hysteresis events into padded,
   merged regions is a deterministic operation on event dicts + audio duration + config —
   no audio, no models, no I/O. It belongs in its own module that unit-tests cleanly.

---

## Inherited from Phase 0 (NOT re-derived here)

- `region_detection_jobs` table with standard queue columns, `parent_run_id` FK,
  `trace_row_count`, `region_count`, `config_json`, model FK columns.
- Worker shell `src/humpback/workers/region_detection_worker.py` that claims and fails.
- `compute_hysteresis_events(audio, sr, perch, classifier, config)` in
  `src/humpback/classifier/detector.py`, returning dict-shaped per-window records and
  hysteresis event dicts.
- `merge_detection_events(window_records, high_threshold, low_threshold)` public helper in
  `src/humpback/classifier/detector_utils.py`.
- `Region` and `WindowScore` frozen dataclasses in `src/humpback/call_parsing/types.py`
  with pyarrow schemas.
- Atomic parquet I/O helpers in `src/humpback/call_parsing/storage.py`
  (`write_trace` / `read_trace` / `write_regions` / `read_regions`), with the directory
  layout `storage_root/call_parsing/regions/<job_id>/`.
- Stub API router at `src/humpback/api/routers/call_parsing.py` with `POST /region-jobs`
  and the `/trace`, `/regions` GETs returning 501.
- Behavior-preserving detector refactor guarded by
  `tests/fixtures/detector_refactor_snapshot.json`.

---

## Scope

**Pass 1 ships:**

- Migration `043` dropping `audio_source_id` and adding proper source columns on both
  `call_parsing_runs` and `region_detection_jobs`.
- `src/humpback/call_parsing/regions.py` — pure `decode_regions` function with unit tests.
- `src/humpback/schemas/call_parsing.py` — Pydantic `RegionDetectionConfig` and
  `CreateRegionJobRequest` models, `exactly-one-of` source validation.
- `src/humpback/services/call_parsing.py` — new `create_region_job` service method; update
  `create_parent_run` to take the same source + model + config fields and create the
  Pass 1 child atomically.
- `src/humpback/classifier/detector.py` — small behavior-preserving refactor exposing
  `score_audio_windows` as a public helper; `compute_hysteresis_events` becomes a thin
  composition of `score_audio_windows` + `merge_detection_events`.
- `src/humpback/workers/region_detection_worker.py` — replace the stub with a real worker
  that resolves the audio source, loads models, runs the scoring loop (streaming for
  hydrophone, single-shot for uploaded files), runs hysteresis + `decode_regions`, and
  writes both parquet artifacts.
- `src/humpback/workers/queue.py` — add `RegionDetectionJob` to the stale-job recovery
  sweep.
- `src/humpback/api/routers/call_parsing.py` — unstub `POST /region-jobs`,
  `GET /{id}/trace`, `GET /{id}/regions`; extend `POST /runs` to take the same source +
  model + config fields.
- Unit, integration, and API tests (see Testing section).
- Documentation updates in `CLAUDE.md`, `docs/reference/data-model.md`, `README.md`.
- New `DECISIONS.md` ADR-049 capturing the Pass 1 algorithmic defaults and their
  rationale.

**Pass 1 does NOT ship:**

- Any Pass 2 / Pass 3 / Pass 4 logic.
- Any frontend work (UI for inspecting regions is deferred per the Phase 0 non-goals).
- A hydrophone-path worker integration test (deferred — no `ArchivePlaybackProvider`
  mock surface; correctness of the streaming loop is covered by unit tests instead).
- Smoothing (`smoothing_window_sec`), separate `merge_gap_sec`, or any config knob not
  listed in `RegionDetectionConfig` below. YAGNI until a real case argues for them.
- Partial-trace resume on worker crash. Delete-and-restart matches the rest of the
  codebase.

---

## Data flow

```
1.  Worker claims queued region_detection_job via compare-and-set.
2.  Deserialize job.config_json into RegionDetectionConfig.
3.  Resolve audio source from the job row:
        audio_file_id  → load the whole file via the existing AudioLoader
        hydrophone_id  → open an ArchivePlaybackProvider for [start_ts, end_ts]
4.  Load Perch (get_model_by_version) and the binary classifier (joblib.load).
5.  Build the dense trace:
        file path   → one call to score_audio_windows on the full buffer
        hydrophone  → streaming loop over window-aligned chunks:
                        for each chunk:
                          fetch audio into a float32 numpy buffer
                          call score_audio_windows with time_offset_sec set
                          extend the in-memory trace list
                          release the chunk buffer
6.  Run merge_detection_events(trace, high_threshold, low_threshold) — one pass
    over the full concatenated trace.
7.  Run decode_regions(events, audio_duration_sec, config) — padding, merge,
    min-duration filter; returns list[Region].
8.  Atomic writes via call_parsing.storage:
        <job_dir>/trace.parquet
        <job_dir>/regions.parquet
9.  Update job row: trace_row_count, region_count, completed_at, status='complete'.
10. On exception: clean partial artifacts, set status='failed', error_message.
```

The key architectural move is the `compute_hysteresis_events` split: scoring is the
chunk-friendly inner loop, hysteresis runs once on the concatenated trace. The file-source
path is a degenerate "one chunk" case that falls out of the same code path.

---

## Schema changes — migration `043`

File: `alembic/versions/043_call_parsing_pass1_source_columns.py`, using
`op.batch_alter_table()` for SQLite compatibility.

### `call_parsing_runs`

```
DROP  audio_source_id        (String, not null)   -- Phase 0 placeholder, unused in code
ADD   audio_file_id          (String, nullable)   -- uploaded file FK (audio_files.id)
ADD   hydrophone_id          (String, nullable)   -- hydrophone source identifier
ADD   start_timestamp        (Float,  nullable)   -- UTC epoch seconds
ADD   end_timestamp          (Float,  nullable)   -- UTC epoch seconds
```

### `region_detection_jobs`

```
DROP  audio_source_id        (String, not null)   -- Phase 0 placeholder
ADD   audio_file_id          (String, nullable)
ADD   hydrophone_id          (String, nullable)
ADD   start_timestamp        (Float,  nullable)
ADD   end_timestamp          (Float,  nullable)
```

**Constraint shape.** Exactly-one-of (`audio_file_id`) vs
(`hydrophone_id` + `start_timestamp` + `end_timestamp`) is enforced in the Pydantic
request model and the service layer, not via a DB CHECK constraint. This matches the
existing project pattern for `DetectionJob` (which also carries both source shapes in
nullable columns without a CHECK).

**Hyperparameters stay in `config_json`.** Pass 1's knobs (thresholds, padding,
min_region_duration_sec, stream_chunk_sec) live in the existing Phase 0 `config_json`
TEXT column, serialized from `RegionDetectionConfig.model_dump_json()`. No per-knob
columns — the existing `DetectionJob` table pays an ongoing migration cost every time a
new hyperparameter ships, and Pass 1 is brand-new code that will iterate on these
defaults.

**Downgrade.** The migration has a working `downgrade()` that restores the
`audio_source_id` column as nullable (not `not null`, to avoid requiring a backfill) and
drops the new source columns.

---

## Config contract

File: `src/humpback/schemas/call_parsing.py`.

```python
class RegionDetectionConfig(BaseModel):
    # Detector knobs — passed through to score_audio_windows and merge_detection_events
    window_size_seconds: float = 5.0
    hop_seconds: float = 1.0
    high_threshold: float = 0.70
    low_threshold: float = 0.45

    # Region-shaping knobs — consumed by decode_regions
    padding_sec: float = 1.0
    min_region_duration_sec: float = 0.0

    # Streaming control — hydrophone path only
    stream_chunk_sec: float = 1800.0  # 30 minutes
```

Defaults track the existing detector's hysteresis defaults exactly
(`5.0 / 1.0 / 0.70 / 0.45`) plus the Pass 1-specific additions from brainstorming
(`padding_sec=1.0` symmetric, `min_region_duration_sec=0.0` no filter, 30-minute streaming
chunks). See the ADR-049 section below for why these values.

```python
class CreateRegionJobRequest(BaseModel):
    audio_file_id: str | None = None
    hydrophone_id: str | None = None
    start_timestamp: float | None = None
    end_timestamp: float | None = None

    model_config_id: str           # Perch model
    classifier_model_id: str       # binary whale detector
    parent_run_id: str | None = None

    config: RegionDetectionConfig = Field(default_factory=RegionDetectionConfig)

    @model_validator(mode="after")
    def _exactly_one_source(self):
        has_file  = self.audio_file_id is not None
        has_hydro = all(v is not None for v in
                        (self.hydrophone_id, self.start_timestamp, self.end_timestamp))
        if has_file == has_hydro:
            raise ValueError("Provide exactly one of audio_file_id or "
                             "(hydrophone_id, start_timestamp, end_timestamp)")
        if has_hydro and self.end_timestamp <= self.start_timestamp:
            raise ValueError("end_timestamp must be after start_timestamp")
        return self
```

`POST /call-parsing/runs` takes the same source + model + config fields (as a parallel
`CreateParentRunRequest` or by embedding the `CreateRegionJobRequest` shape).

---

## Region decoder

File: `src/humpback/call_parsing/regions.py`.

```python
def decode_regions(
    events: list[dict[str, Any]],
    audio_duration_sec: float,
    config: RegionDetectionConfig,
) -> list[Region]:
    """Turn hysteresis events into padded, merged regions.

    Pure function — no I/O, no audio, no models. Accepts the dict-shaped events that
    merge_detection_events already returns: {start_sec, end_sec, avg_confidence,
    peak_confidence, n_windows}. Returns frozen Region dataclasses with padded bounds
    clamped to [0.0, audio_duration_sec].
    """
```

### Algorithm

1. Sort events by `start_sec`.
2. For each event, compute padded bounds:
   - `padded_start = max(0.0, start_sec - padding_sec)`
   - `padded_end   = min(audio_duration_sec, end_sec + padding_sec)`
3. Left-to-right merge pass. If `next.padded_start <= current.padded_end`, fuse:
   - `raw_start    = min(current.raw_start, next.raw_start)`
   - `raw_end      = max(current.raw_end, next.raw_end)`
   - `padded_start = min(current.padded_start, next.padded_start)`
   - `padded_end   = max(current.padded_end, next.padded_end)`
   - `max_score    = max(current.max_score, next.max_score)`
   - `mean_score   = sum(mean_i * n_windows_i) / sum(n_windows_i)` (window-weighted)
   - `n_windows    = current.n_windows + next.n_windows`
4. Drop regions with `(raw_end - raw_start) < min_region_duration_sec`.
5. Assign `region_id = uuid4().hex` to each survivor.
6. Return sorted by `start_sec` (already sorted after step 3).

### Unit-tested edge cases

- Empty input → empty output.
- Single event → single region.
- Two adjacent events whose padded bounds exactly touch
  (`next.padded_start == current.padded_end`) → merged (inclusive boundary).
- Event at `start_sec=0` → `padded_start=0.0` (clamp).
- Event at `end_sec=audio_duration_sec` → `padded_end=audio_duration_sec` (clamp).
- `min_region_duration_sec > 0` drops correctly.
- Three events where `(1,2)` merge but `(3)` stays standalone.
- `mean_score` weighted-average correctness on known inputs.

---

## Worker architecture

### Detector refactor

File: `src/humpback/classifier/detector.py`.

Expose a new public helper:

```python
def score_audio_windows(
    audio: np.ndarray,
    sample_rate: int,
    perch_model: EmbeddingModel,
    classifier: Pipeline,
    config: dict[str, Any],
    time_offset_sec: float = 0.0,
) -> list[dict[str, Any]]:
    """Pass 1 streaming primitive: audio → dense per-window score records.

    Returns window records (with keys offset_sec, end_sec, confidence) whose
    offset_sec / end_sec are shifted by time_offset_sec. Callers streaming audio
    in chunks set time_offset_sec to the chunk's absolute start so per-chunk
    records concatenate into a single absolute-time trace.
    """
```

`compute_hysteresis_events` is re-implemented as a two-line composition:

```python
window_records = score_audio_windows(audio, sr, perch, classifier, cfg)
events = merge_detection_events(window_records, cfg["high_threshold"],
                                 cfg["low_threshold"])
return window_records, events
```

This is behavior-preserving — the existing refactor snapshot test
(`tests/fixtures/detector_refactor_snapshot.json`) is extended with a second assertion
that the two-call composition matches `compute_hysteresis_events(...)` to float64
precision.

### Worker file

File: `src/humpback/workers/region_detection_worker.py` — replace the Phase 0 stub.

High-level structure:

1. Deserialize `job.config_json` into `RegionDetectionConfig`.
2. Resolve audio source (file path or hydrophone provider).
3. Load Perch via `get_model_by_version(session, cm.model_version, settings)` and the
   classifier via `joblib.load(cm.model_path)`, matching the existing detection worker.
4. Build the trace:
   - **File source:** load the whole buffer, call `score_audio_windows` once with
     `time_offset_sec=0.0`, get the trace.
   - **Hydrophone source:** compute aligned chunk edges, loop; per chunk fetch audio,
     call `score_audio_windows` with `time_offset_sec=(chunk_start - range_start)`,
     `.extend()` the trace list, free the chunk buffer.
5. Run `merge_detection_events(trace, high_threshold, low_threshold)`.
6. Run `decode_regions(events, audio_duration_sec, config)`.
7. `write_trace` and `write_regions` via `call_parsing.storage` (both atomic).
8. Update row counts, `completed_at`, `status='complete'`.
9. On exception: delete partial `trace.parquet` / `regions.parquet` / `.tmp` sidecars
   under the job directory, set `status='failed'`, populate `error_message`, re-raise
   if the caller expects it (matching the existing detection worker's error protocol).

### Chunk alignment rule

Every hydrophone chunk boundary is a whole multiple of `window_size_seconds` from the
range start. This guarantees that a Perch window never straddles two chunks, so
per-chunk scoring is mathematically equivalent to one scoring pass on the concatenated
buffer (proven by the chunk-concatenation unit test). A helper
`_aligned_chunk_edges(start_ts, end_ts, chunk_sec, alignment_sec)` computes the chunk
list.

With defaults (`stream_chunk_sec=1800`, `window_size_seconds=5`) and a 24 h range, the
worker processes 48 chunks, each holding ~450 MB of audio in RAM at 16 kHz float32. The
in-memory trace for a full day at 1 s hop is ~86,400 float32 pairs (~700 KB) — trivial.

### Stale recovery

`src/humpback/workers/queue.py` — add `RegionDetectionJob` to the existing stale-job
recovery sweep so a killed worker's row resets to `queued` after the 10-minute
threshold, matching the rest of the codebase.

---

## API surface

All endpoints under `/call-parsing/`. Changes on top of Phase 0:

| Method | Path | Phase 0 | After Pass 1 |
|---|---|---|---|
| POST | `/runs` | Functional (takes placeholder `audio_source_id`) | Accepts full source + model + config; creates parent row + Pass 1 child in one transaction |
| GET / DELETE | `/runs/*` | Functional | Unchanged |
| GET | `/runs/{id}/sequence` | 501 (Pass 4) | Unchanged (still 501) |
| POST | `/region-jobs` | 501 | Functional — `CreateRegionJobRequest` |
| GET | `/region-jobs` | Functional | Unchanged |
| GET / DELETE | `/region-jobs/{id}` | Functional | Unchanged |
| GET | `/region-jobs/{id}/trace` | 501 | Functional — streams `trace.parquet` as JSON, 409 if job not complete |
| GET | `/region-jobs/{id}/regions` | 501 | Functional — returns `list[Region]` sorted by start, 409 if job not complete |

**Error codes:**

- `404` — `audio_file_id` / `hydrophone_id` / `model_config_id` / `classifier_model_id`
  not found at create time.
- `409` — trace/regions endpoint called on a job with `status != 'complete'`.
- `422` — Pydantic validator error (missing source, both sources, inverted timestamps).

**Service layer.** `create_region_job(session, request) -> RegionDetectionJob` validates
foreign keys, serializes `request.config` into `config_json`, inserts a queued row,
commits, returns the model. `create_parent_run` calls through to `create_region_job`
with `parent_run_id` set inside the same transaction.

---

## Testing

### 1. `decode_regions` unit tests

File: `tests/unit/test_call_parsing_regions.py`. Pure-function tests with synthetic
event inputs. Covers every edge case listed in the Region Decoder section:
empty / single / adjacent-merge / boundary-clamp / min-duration / three-event partial
merge / weighted-mean-score correctness. ~10 focused tests, no audio, no models.

### 2. Detector refactor equivalence test

File: `tests/unit/test_detector_refactor.py` (extended). The existing refactor
snapshot test gains a second assertion:
`compute_hysteresis_events(audio, sr, perch, classifier, cfg)` must equal
`(score_audio_windows(...), merge_detection_events(score_audio_windows(...), high, low))`
to float64 precision on `tests/fixtures/detector_refactor_snapshot.json`.

### 3. Streaming chunk-concatenation test

File: `tests/unit/test_score_audio_windows_chunking.py` (new). Split a fixture audio
buffer in half at a whole-window boundary, call `score_audio_windows` twice with
appropriate `time_offset_sec`, concatenate the results, and assert they equal a single
`score_audio_windows` call on the whole buffer. Proves the streaming loop is
mathematically equivalent to buffer-and-call — the correctness guarantee behind the
hydrophone path.

### 4. Worker integration test

File: `tests/integration/test_region_detection_worker.py`. End-to-end with a short
fixture audio file (~60 s), a mock Perch embedding model (seeded-random output), and a
mock binary classifier (deterministic `predict_proba`). Test flow:

- Create an `audio_file_id`-source `RegionDetectionJob` row.
- Run one worker iteration.
- Assert: status transitions `queued → running → complete`, `trace.parquet` and
  `regions.parquet` exist under the job directory, `trace_row_count` and `region_count`
  match, region bounds are clamped to `[0.0, audio_duration]`, at least one region
  exists for the seeded signal.
- Assert cleanup on forced failure (stub the classifier to raise) — both parquet files
  absent, `status='failed'`, `error_message` populated.

### 5. API router test

File: `tests/api/test_call_parsing_router.py` (extended). `POST` a region job via the
router, run the worker synchronously, `GET /trace` and `GET /regions`, `DELETE` the
job and assert the parquet directory is gone. Additional assertions:

- `422` on bad source payloads (both source kinds, neither source kind, inverted
  timestamps).
- `409` on `GET /trace` and `GET /regions` before the job completes.
- `404` on unknown `audio_file_id` / `classifier_model_id`.

### 6. Hydrophone worker integration test — deferred

No `ArchivePlaybackProvider` mock surface currently exists. The streaming loop's
correctness is covered by test #3; an end-to-end hydrophone integration test is added
to `docs/plans/backlog.md` as a follow-up for when real-world Pass 1 hydrophone runs
surface bugs.

---

## ADR-049 — Pass 1 algorithmic defaults and streaming architecture

Appended to `DECISIONS.md`. Captures, with rationale, each non-obvious decision:

- **Symmetric `padding_sec=1.0`.** Simple, sufficient context for Pass 2 to find real
  onsets/offsets, no asymmetry evidence yet. If post-event decay tails prove
  under-covered in practice, add asymmetric padding as an opt-in knob.
- **Merge on padded-bounds overlap, no separate `merge_gap_sec`.** Padding is the
  single control. Two calls close enough that their padded regions overlap are fused
  into one region that Pass 2's segmenter can split internally.
- **No temporal smoothing on scores.** Hysteresis's 0.25-probability dead zone already
  protects against single-window dips, the Perch 5 s window / 1 s hop already smooths
  embeddings in time by construction, smoothing shifts peak positions in short
  sequences (the same issue ADR-044 called out for prominence), and `trace.parquet` is
  a data product that consumers want raw.
- **`min_region_duration_sec=0.0` default.** Pass 2 is the real noise rejector;
  Pass 1's job is high-recall. The knob exists in the config for follow-up tuning but
  defaults to off.
- **Dense raw trace, no decimation.** ~1 MB of parquet per day of audio, unlocks a
  future "re-decode without re-running Perch" endpoint, and preserves observability
  for debugging threshold choices.
- **Delete-and-restart on worker crash.** Matches the rest of the codebase.
  Partial-trace resume paths are rarely exercised and tend to rot; revisit only if a
  multi-hour Pass 1 job actually becomes a pain point.
- **`score_audio_windows` / `merge_detection_events` split and chunk-aligned
  streaming.** 24 h hydrophone ranges (~5.5 GB in memory) can't be loaded as a single
  buffer, and splitting the Phase 0 helper is the cleanest way to get a chunk-friendly
  scoring primitive. Chunk alignment on multiples of `window_size_seconds` guarantees
  the streaming path is mathematically equivalent to buffer-and-call, which the
  chunk-concatenation unit test proves.

---

## Documentation updates

- **CLAUDE.md §8.9** — mark `POST /call-parsing/region-jobs`,
  `GET /region-jobs/{id}/trace`, `GET /region-jobs/{id}/regions` as functional; remove
  their 501 callouts.
- **CLAUDE.md §8.7** — add behavioral-constraint bullets for (a) the Pass 1 source
  contract (`audio_file_id` XOR hydrophone range), (b) the chunk-alignment rule for
  hydrophone streaming, (c) delete-and-restart on crash.
- **CLAUDE.md §9.1** — append "Pass 1 region detection implemented" to the Implemented
  Capabilities list.
- **CLAUDE.md §9.2** — bump latest migration to `043_call_parsing_pass1_source_columns.py`.
- **`docs/reference/data-model.md`** — update the `region_detection_jobs` and
  `call_parsing_runs` column lists.
- **README.md** — if it lists endpoints, add the three Pass 1 endpoints; otherwise
  no-op.
- **`docs/plans/backlog.md`** (if present) — note the deferred hydrophone integration
  test.

---

## Non-goals

- Pass 2 / Pass 3 / Pass 4 logic of any kind.
- Frontend UI for inspecting regions.
- Smoothing, asymmetric padding, or a separate `merge_gap_sec` knob.
- Partial-trace resume on worker crash.
- Hydrophone-path integration test (deferred to backlog).
