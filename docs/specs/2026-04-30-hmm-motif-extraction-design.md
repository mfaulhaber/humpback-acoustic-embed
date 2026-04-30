# HMM Motif Extraction - Design Spec

**Status:** Draft
**Date:** 2026-04-30
**Track:** Sequence Models
**Builds on:** ADR-056, ADR-057, HMM Sequence jobs, HMM State Timeline Viewer

---

## 1. Goal

Add motif extraction as a first-class Sequence Models analysis. A motif
extraction job consumes a completed HMM Sequence job, converts decoded Viterbi
states into symbolic sequences, collapses consecutive repeated states, mines
recurring n-grams, ranks biologically interesting motifs, and adds a Motifs
panel to the HMM Sequence detail page with examples aligned around event
midpoints.

Conceptually:

```text
Decoded states per sequence:
[1, 1, 1, 2, 2, 3, 3, 1]

Collapsed symbolic sequence:
[1, 2, 3, 1]

Motif candidates:
2-grams through 8-grams over the collapsed sequence
```

Default extraction config:

| Field | Default |
|---|---:|
| `min_ngram` | 2 |
| `max_ngram` | 8 |
| `minimum_occurrences` | 5 |
| `minimum_event_sources` | 2 |
| `frequency_weight` | 0.40 |
| `event_source_weight` | 0.30 |
| `event_core_weight` | 0.20 |
| `low_background_weight` | 0.10 |
| `call_probability_weight` | null |

The result should answer: "Which latent-state phrases recur across events and
recordings, and appear to track event structure rather than background?"

---

## 2. Non-Goals

- Training a new model from motifs.
- Treating motifs as human call-type labels.
- Editing or correcting motifs in the UI.
- Mining motifs across multiple HMM jobs in one run.
- Mixing SurfPerch and CRNN sources in one HMM job.
- Re-decoding audio or recomputing embeddings.
- Making CRNN `call_probability` part of the default rank. It is optional in
  the schema and defaults to null for the MVP.

---

## 3. Existing Context

HMM Sequence jobs already persist source-specific `states.parquet` artifacts:

- SurfPerch source: grouped by `merged_span_id`, indexed by
  `window_index_in_span`, with `event_id`, `audio_file_id`, `is_in_pad`, and
  Viterbi state columns.
- CRNN source: grouped by `region_id`, indexed by `chunk_index_in_region`, with
  `tier`, `audio_file_id`, and Viterbi state columns.

The parent continuous embedding job persists source-specific
`embeddings.parquet` and `manifest.json` artifacts. CRNN embeddings include
`nearest_event_id`, `event_overlap_fraction`, and `call_probability`; SurfPerch
state rows already include `event_id`.

The HMM detail page already loads:

- job detail, including `source_kind`
- paginated state rows
- transition matrix, dwell histograms, label distribution, exemplars
- timeline viewer with PCEN spectrogram, HMM state bar, event/region navigation,
  and audio slice playback

Motif extraction should reuse those artifacts, but its output should be tracked
as a first-class app analysis with status, parameters, list/detail surfaces, and
stable artifact paths.

---

## 4. Approaches Considered

### Approach A: Generate motif artifacts on demand and cache by config hash

Add a deterministic pure extraction module plus API endpoints. The first request
for a given HMM job and extraction config reads `states.parquet`, mines motifs,
writes sidecar artifacts, and returns the summary. Later requests reuse the
cached artifacts.

Pros:

- No database migration.
- Supports alternate thresholds without creating many DB rows.
- Keeps motif extraction reproducible and easy to test.

Cons:

- Treats motifs like a transient visualization cache rather than a first-class
  analysis.
- First page load may do hidden work.
- Harder to compare runs, expose job history, cancel work, or show failures.

### Approach B: Generate default motifs automatically when the HMM worker completes

The HMM worker writes default motif artifacts after `states.parquet` and
`state_summary.json`.

Pros:

- The detail page can load default motifs immediately.
- Failure can be logged as a non-fatal interpretation warning.

Cons:

- Worker completion does more interpretation work even if the user never wants
  motifs.
- Alternate extraction thresholds still need another execution surface.
- Motif analysis becomes an implicit side effect of HMM training.

### Approach C: Add first-class `motif_extraction_jobs`

Create queued motif extraction jobs with status, config, artifact pointers, and
history. Motif extraction becomes a peer analysis under the Sequence Models
track, backed by its own service, worker, API endpoints, and frontend surfaces.

Pros:

- Matches the importance of motif extraction as a durable analysis in the app.
- Gives status, cancellation, failure messages, history, and repeatable configs.
- Allows users to compare alternative extraction/ranking settings.
- Keeps HMM training focused on modeling, while motif extraction remains an
  explicit downstream interpretation step.
- Provides a natural future path for cross-job motif mining.

Cons:

- Requires a database migration, queue integration, and more UI surface.
- Slightly more ceremony for the MVP than cached-on-demand artifacts.

### Decision

Use Approach C. Motif extraction is a first-class Sequence Models analysis and
should be represented by `motif_extraction_jobs`, not hidden behind the HMM
detail page as an opportunistic cache.

---

## 5. Data Model

Add `motif_extraction_jobs`.

| Column | Type | Notes |
|---|---|---|
| `id` | string PK | UUID, same pattern as other Sequence Models jobs |
| `status` | enum/string | `queued`, `running`, `complete`, `failed`, `canceled` |
| `hmm_sequence_job_id` | string FK | required, source HMM job |
| `source_kind` | string | copied from parent source for filtering/display |
| `min_ngram` | int | default 2 |
| `max_ngram` | int | default 8 |
| `minimum_occurrences` | int | default 5 |
| `minimum_event_sources` | int | default 2 |
| `frequency_weight` | float | default 0.40 |
| `event_source_weight` | float | default 0.30 |
| `event_core_weight` | float | default 0.20 |
| `low_background_weight` | float | default 0.10 |
| `call_probability_weight` | float nullable | optional, default null |
| `config_signature` | string | sha256 over source job id + extraction config |
| `total_groups` | int nullable | filled at completion |
| `total_collapsed_tokens` | int nullable | filled at completion |
| `total_candidate_occurrences` | int nullable | filled at completion |
| `total_motifs` | int nullable | filled at completion |
| `artifact_dir` | string nullable | `motif_extractions/{job_id}` |
| `error_message` | text nullable | failure reason |
| `created_at`, `updated_at` | datetime UTC | standard timestamps |

Idempotency:

- `config_signature` preserves uniqueness among completed jobs for the same
  `(hmm_sequence_job_id, extraction config)`.
- Creating an identical completed job returns the existing job.
- Creating an identical queued/running job returns that in-flight job.
- Failed/canceled jobs do not block a retry.

Validation:

- Source HMM job must exist.
- Source HMM job must be `complete` before a motif job can be queued.
- `min_ngram >= 1`.
- `max_ngram >= min_ngram`.
- `max_ngram <= 16` for MVP guardrails.
- `minimum_occurrences >= 1`.
- `minimum_event_sources >= 1`.
- Rank weights must be non-negative.
- At least one non-null rank weight must be greater than zero.

Alembic migration:

- Add `062_motif_extraction_jobs.py`.
- Use `op.batch_alter_table()` for SQLite compatibility if altering shared
  queue/status structures is needed.
- Add indexes on `status`, `hmm_sequence_job_id`, and `config_signature`.

---

## 6. Motif Semantics

### 6.1 Sequence grouping

Build one symbolic sequence per decoded source sequence:

| Source kind | Group key | Sort key |
|---|---|---|
| SurfPerch | `merged_span_id` | `window_index_in_span` |
| CRNN | `region_id` | `chunk_index_in_region` |

Within each group, sort rows by the source-specific sort key and read
`viterbi_state` as the raw state stream.

### 6.2 Collapse repeated states

Collapse consecutive equal states within each group. Each collapsed token keeps
the state id plus metadata for the run it represents:

| Token field | Meaning |
|---|---|
| `state` | Viterbi state id |
| `run_start_index` | first raw row index in the run |
| `run_end_index` | last raw row index in the run, inclusive |
| `start_timestamp` | start of first raw row |
| `end_timestamp` | end of last raw row |
| `event_source_key` | stable event-source key |
| `audio_source_key` | stable audio key for display |
| `event_core_duration` | duration considered event-core |
| `background_duration` | duration considered background |
| `mean_call_probability` | optional CRNN-only value, nullable |

For SurfPerch rows, `is_in_pad == false` is the event-core proxy and
`is_in_pad == true` is the background proxy.

For CRNN rows, `tier == "event_core"` is event-core,
`tier == "background"` is background, and `tier == "near_event"` contributes to
neither core nor background.

### 6.3 Event-source coverage

`minimum_event_sources` replaces the earlier "audio source" threshold. A motif
must appear in at least this many distinct event sources.

Event-source key:

- SurfPerch: `event_id` when present.
- CRNN: dominant non-null `nearest_event_id` from the parent CEJ
  `embeddings.parquet` joined by `(region_id, chunk_index_in_region)`.
- Fallback: group key, with `event_source_key_strategy="group_fallback"` in the
  job manifest.

The fallback keeps motif extraction usable for incomplete historical artifacts,
but the UI should show a small warning when the fallback strategy is used.

### 6.4 Candidate extraction

For each collapsed sequence, slide n-grams from `min_ngram` through
`max_ngram` over the token list. A motif key is the state tuple joined with
hyphens, for example `1-2-3`.

Each occurrence stores:

- motif key and state tuple
- source group
- token range and raw row range
- absolute start/end timestamps
- event-source key
- audio-source key for playback/display
- event midpoint anchor
- ranking support metrics

### 6.5 Minimum filters

Apply filters after aggregating occurrences across the HMM job:

- `occurrence_count >= minimum_occurrences`
- `event_source_count >= minimum_event_sources`

The default `minimum_event_sources=2` may hide all motifs for single-event test
runs; the creation form should allow lowering it for exploratory inspection.

---

## 7. Ranking

Each motif summary exposes transparent ranking components:

| Metric | Definition |
|---|---|
| `occurrence_count` | total motif occurrences after repeated-state collapse |
| `event_source_count` | distinct event sources containing the motif |
| `audio_source_count` | distinct audio sources containing the motif, display only |
| `event_core_fraction` | fraction of motif raw-row duration that is event-core |
| `background_fraction` | fraction of motif raw-row duration that is background |
| `mean_call_probability` | optional CRNN-only mean over occurrence rows, nullable |
| `mean_duration_seconds` | mean occurrence duration |
| `median_duration_seconds` | median occurrence duration |

Default rank score:

```text
rank_score =
  frequency_weight * log_frequency_norm +
  event_source_weight * event_source_recurrence_norm +
  event_core_weight * event_core_fraction +
  low_background_weight * (1 - background_fraction) +
  optional_call_probability_term
```

Where:

- `log_frequency_norm = log1p(occurrence_count) / max_log1p_count`
- `event_source_recurrence_norm = event_source_count / max_event_source_count`
- `optional_call_probability_term` is omitted when `call_probability_weight` is
  null or the source has no call probabilities

When `call_probability_weight` is non-null, use:

```text
optional_call_probability_term =
  call_probability_weight * mean_call_probability
```

Weights are user-editable in an Advanced section on the motif job creation
form. The defaults stay fixed and source-neutral.

Sort by `rank_score desc`, then `event_source_count desc`,
`occurrence_count desc`, `event_core_fraction desc`,
`background_fraction asc`, `length desc`, and `motif_key asc`.

---

## 8. Event-Time Alignment

Motif examples are aligned by event midpoint. Each occurrence gets:

| Field | Meaning |
|---|---|
| `anchor_event_id` | event id associated with the occurrence, nullable |
| `anchor_timestamp` | event midpoint timestamp used as time zero |
| `relative_start_seconds` | `occurrence_start - anchor_timestamp` |
| `relative_end_seconds` | `occurrence_end - anchor_timestamp` |
| `anchor_strategy` | how the anchor was chosen |

Anchor selection:

1. SurfPerch: use the occurrence's dominant non-empty `event_id`, then load the
   event start/end from the parent segmentation `events.parquet`; anchor to the
   event midpoint.
2. CRNN: join occurrence rows to the parent continuous embedding
   `embeddings.parquet` by `(region_id, chunk_index_in_region)` and use the
   dominant non-null `nearest_event_id`; anchor to the event midpoint.
3. If no event id is available, use the midpoint of event-core rows in the
   occurrence.
4. If no event-core rows exist, use the occurrence midpoint.

The selected strategy is persisted so analysts can distinguish true event
midpoint alignment from fallback alignment.

---

## 9. Artifact Layout

Motif extraction jobs write under their own top-level storage tree:

```text
/motif_extractions/
  {job_id}/manifest.json
  {job_id}/motifs.parquet
  {job_id}/occurrences.parquet
```

### 9.1 `manifest.json`

| Field | Meaning |
|---|---|
| `schema_version` | motif artifact schema version, initially `1` |
| `motif_extraction_job_id` | motif job id |
| `hmm_sequence_job_id` | source HMM job id |
| `continuous_embedding_job_id` | source CEJ id |
| `source_kind` | `surfperch` or `region_crnn` |
| `config` | extraction and rank config |
| `config_signature` | digest stored on the DB row |
| `generated_at` | UTC timestamp |
| `total_groups` | decoded groups scanned |
| `total_collapsed_tokens` | collapsed token count |
| `total_candidate_occurrences` | raw n-gram occurrences before filters |
| `total_motifs` | motifs after filters |
| `event_source_key_strategy` | `event_id`, `nearest_event_id`, or `group_fallback` |

### 9.2 `motifs.parquet`

One row per motif:

| Column | Type |
|---|---|
| `motif_key` | string, e.g. `1-2-3` |
| `states` | list<int16> |
| `length` | int16 |
| `occurrence_count` | int32 |
| `event_source_count` | int32 |
| `audio_source_count` | int32 |
| `group_count` | int32 |
| `event_core_fraction` | float32 |
| `background_fraction` | float32 |
| `mean_call_probability` | float32 nullable |
| `mean_duration_seconds` | float32 |
| `median_duration_seconds` | float32 |
| `rank_score` | float32 |
| `example_occurrence_ids` | list<string> |

### 9.3 `occurrences.parquet`

One row per motif occurrence:

| Column | Type |
|---|---|
| `occurrence_id` | string |
| `motif_key` | string |
| `states` | list<int16> |
| `source_kind` | string |
| `group_key` | string |
| `event_source_key` | string |
| `audio_source_key` | string nullable |
| `token_start_index` | int32 |
| `token_end_index` | int32 |
| `raw_start_index` | int32 |
| `raw_end_index` | int32 |
| `start_timestamp` | float64 |
| `end_timestamp` | float64 |
| `duration_seconds` | float32 |
| `event_core_fraction` | float32 |
| `background_fraction` | float32 |
| `mean_call_probability` | float32 nullable |
| `anchor_event_id` | string nullable |
| `anchor_timestamp` | float64 |
| `relative_start_seconds` | float32 |
| `relative_end_seconds` | float32 |
| `anchor_strategy` | string |

The raw row key columns remain source-normalized in `group_key`; the API can
include source-specific aliases for frontend convenience.

---

## 10. Backend Design

### 10.1 New module

Create `src/humpback/sequence_models/motifs.py`.

Public API:

- `MotifExtractionConfig`
- `MotifExtractionResult`
- `collapse_state_runs(rows, source_kind) -> list[CollapsedSequence]`
- `extract_motifs(states_table, *, source_kind, config, event_lookup=None, embedding_table=None) -> MotifExtractionResult`
- `write_motif_artifacts(result, output_dir) -> None`
- `read_motif_artifacts(output_dir) -> MotifExtractionResult`

The module should be pure apart from explicit artifact read/write helpers. Unit
tests can exercise collapse, counting, filtering, ranking, and anchoring without
a DB.

### 10.2 Service

Create `src/humpback/services/motif_extraction_service.py`.

Responsibilities:

- create motif extraction jobs
- compute `config_signature`
- enforce idempotency
- list jobs by status and/or source HMM job
- fetch job detail
- cancel queued/running jobs
- delete jobs and artifacts

### 10.3 Worker

Create `src/humpback/workers/motif_extraction_worker.py` and wire it into the
queue loop.

Runtime flow:

1. Claim one queued motif extraction job.
2. Mark it `running`.
3. Load the source HMM job and verify it is still `complete`.
4. Resolve parent CEJ and source kind.
5. Read `states.parquet`.
6. Read parent CEJ `embeddings.parquet` when CRNN event-source keys or
   `call_probability` values are needed.
7. Read parent segmentation `events.parquet` for event midpoint anchors.
8. Run pure motif extraction.
9. Write `manifest.json`, `motifs.parquet`, and `occurrences.parquet`
   atomically under `motif_extractions/{job_id}/`.
10. Update counters, `artifact_dir`, and status.

Cancellation:

- `queued` jobs can flip directly to `canceled`.
- `running` jobs check cancellation between major phases: after source load,
  after collapse, after candidate extraction, and before artifact rename.

### 10.4 Storage helpers

Add to `src/humpback/storage.py`:

- `motif_extraction_dir(storage_root, job_id)`
- `motif_extraction_manifest_path(storage_root, job_id)`
- `motif_extraction_motifs_path(storage_root, job_id)`
- `motif_extraction_occurrences_path(storage_root, job_id)`

### 10.5 API endpoints

Add endpoints under the Sequence Models router:

| Endpoint | Behavior |
|---|---|
| `POST /sequence-models/motif-extractions` | Create/queue a motif extraction job |
| `GET /sequence-models/motif-extractions` | List jobs, filter by status or `hmm_sequence_job_id` |
| `GET /sequence-models/motif-extractions/{id}` | Return job detail plus manifest when available |
| `POST /sequence-models/motif-extractions/{id}/cancel` | Cancel queued/running job |
| `DELETE /sequence-models/motif-extractions/{id}` | Delete job and artifacts |
| `GET /sequence-models/motif-extractions/{id}/motifs` | Return paginated motif summary rows |
| `GET /sequence-models/motif-extractions/{id}/motifs/{motif_key}/occurrences` | Return paginated occurrence rows for one motif |

Creation body:

- `hmm_sequence_job_id`
- `min_ngram`
- `max_ngram`
- `minimum_occurrences`
- `minimum_event_sources`
- `frequency_weight`
- `event_source_weight`
- `event_core_weight`
- `low_background_weight`
- `call_probability_weight`

### 10.6 Schemas

Add Pydantic models:

- `MotifExtractionJobCreate`
- `MotifExtractionJobOut`
- `MotifExtractionJobDetail`
- `MotifExtractionManifest`
- `MotifSummary`
- `MotifsResponse`
- `MotifOccurrence`
- `MotifOccurrencesResponse`

Keep `states` as an integer list in API responses, even though `motif_key` is
the stable URL identifier.

---

## 11. Frontend Design

### 11.1 Navigation and pages

Add Motif Extraction as a Sequence Models sub-surface:

- HMM Sequence detail page: Motifs panel scoped to the current HMM job.
- Optional follow-up page: `/app/sequence-models/motif-extractions` for all
  motif jobs. The MVP can start with the HMM-scoped panel if route surface needs
  to stay small, but the backend should still expose list/detail endpoints.

### 11.2 API client

Extend `frontend/src/api/sequenceModels.ts` with:

- `fetchMotifExtractionJobs(params)`
- `fetchMotifExtractionJob(jobId)`
- `createMotifExtractionJob(body)`
- `cancelMotifExtractionJob(jobId)`
- `deleteMotifExtractionJob(jobId)`
- `fetchMotifs(jobId, params)`
- `fetchMotifOccurrences(jobId, motifKey, params)`
- React Query hooks for each

Types mirror the backend schemas.

### 11.3 HMM detail page Motifs panel

Add a `Motifs` panel to `HMMSequenceDetailPage` after the HMM State Timeline
Viewer and before the existing Plotly State Timeline panel.

Panel states:

- no motif job exists for this HMM job: show create form with default config
- motif job queued/running: show status and cancel button
- motif job failed: show error and rerun controls
- motif job complete: show motif table and aligned examples

Create form:

- default fields: n-gram range, minimum occurrences, minimum event sources
- Advanced section: rank weights
- `call_probability_weight` optional, default blank/null
- show helper text only where it clarifies validation or source-specific
  behavior; avoid in-app feature narration

Motif table:

- state sequence
- occurrence count
- event-source count
- event-core fraction
- background fraction
- optional call probability
- mean duration
- rank score

### 11.4 Motif example alignment

Create `frontend/src/components/sequence-models/MotifExampleAlignment.tsx`.

Display selected occurrences as stacked rows aligned to event midpoint:

- x-axis is relative seconds around `anchor_timestamp`, with `0` marked as the
  event midpoint.
- each row renders the motif's collapsed states as colored segments using the
  existing HMM state palette.
- each row labels source, occurrence time, and anchor strategy.
- a small play control uses `regionAudioSliceUrl(regionDetectionJobId, start,
  duration)` for the occurrence window plus context padding.
- a "jump to timeline" action sets the existing timeline selected event/region
  and scroll target to the occurrence midpoint.

For large motif occurrence sets, show the top examples first:

1. highest event-core fraction
2. lowest background fraction
3. diverse event sources
4. earliest timestamp tie-breaker

### 11.5 Visual integration

Reuse existing state colors and timeline helpers. Do not render another full
spectrogram inside the Motifs panel for the MVP; the panel should provide an
aligned symbolic overview plus jump/play affordances. The existing HMM State
Timeline Viewer remains the place for detailed spectrogram inspection.

---

## 12. Testing

### Backend unit tests

Create `tests/sequence_models/test_motifs.py`:

- repeated-state collapse: `1,1,1,2,2,3 -> 1,2,3`
- n-gram extraction over collapsed tokens for lengths 2 through 8
- minimum occurrence filter
- minimum event-source filter
- ranking prefers cross-event-source recurrence over same-count single-source
  motifs
- advanced rank weights change motif ordering deterministically
- SurfPerch event-core/background fractions from `is_in_pad`
- CRNN event-core/background fractions from `tier`
- CRNN call probability is nullable by default and included only when configured
- anchor selection uses event midpoint when event ids are available
- anchor selection falls back in the documented order
- config signature changes when thresholds or weights change and remains stable
  for equivalent configs

### Service/API tests

Create `tests/integration/test_motif_extraction_api.py`:

- `POST /motif-extractions` queues a job for a completed HMM Sequence job
- identical in-flight or completed config returns the existing job
- failed/canceled jobs do not block retry
- non-complete HMM job returns `422`
- `GET /motif-extractions` filters by status and HMM job id
- `GET /motif-extractions/{id}` returns manifest after completion
- `GET /motif-extractions/{id}/motifs` returns ranked motif rows
- `minimum_event_sources=2` filters out motifs present in only one event source
- CRNN fixture returns motifs using `region_id` grouping and nearest-event
  event-source keys
- `GET /motifs/{motif_key}/occurrences` paginates occurrence rows
- cancel endpoint handles queued/running/terminal status correctly
- delete endpoint removes disk artifacts

### Worker tests

Create `tests/workers/test_motif_extraction_worker.py`:

- completed SurfPerch fixture writes manifest, motifs, and occurrences
- completed CRNN fixture writes manifest, motifs, and occurrences
- cancellation before artifact rename leaves no partial final artifacts
- failure marks job failed with a useful error message

### Frontend tests

Add Playwright coverage in `frontend/e2e/sequence-models/hmm-sequence.spec.ts`:

- HMM detail page shows Motifs panel for a completed HMM job
- default create form posts a motif extraction job
- Advanced section exposes rank weights and optional call probability weight
- running job displays status and cancel action
- complete job renders ranked motif rows from mocked API data
- selecting a motif renders aligned examples with event-midpoint zero marker
- changing/rerunning config creates or selects the corresponding motif job
- jump-to-timeline updates the existing event/region selection

---

## 13. Documentation Updates

Update these references during implementation:

- `docs/reference/storage-layout.md` - motif extraction artifact paths and schemas
- `docs/reference/sequence-models-api.md` - motif extraction endpoints and
  parameters
- `docs/reference/frontend.md` - HMM detail page Motifs panel
- `docs/reference/data-model.md` - `motif_extraction_jobs`
- `CLAUDE.md` Sequence Models section - mention HMM motif extraction if that
  section tracks active capabilities
- `DECISIONS.md` - add an ADR only if the team wants the first-class job
  boundary recorded; this design is important enough that ADR-058 would be
  reasonable

---

## 14. Resolved Questions

1. `minimum_audio_sources` is renamed to `minimum_event_sources`.
2. Motif examples align to event midpoint.
3. Rank weights are exposed in an Advanced section.
4. CRNN `call_probability` is optional for now. MVP defaults
   `call_probability_weight` to null and leaves `mean_call_probability` null
   when unavailable or unused.

