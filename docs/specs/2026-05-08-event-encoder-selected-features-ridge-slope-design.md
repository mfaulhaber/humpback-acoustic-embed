# Event Encoder Selected Features And Ridge Slope - Design

**Date:** 2026-05-08
**Status:** Draft
**Primary domain:** Sequence Models
**Neighbor domains:** Call Parsing, Signal Timeline, Frontend Shell

## 1. Goal

Add two Event Encoder improvements:

1. On the Event Encoder job detail page, add a panel below the timeline viewer
   that shows the non-CRNN-derived vector values for the currently selected
   timeline event/token in a table.
2. Replace the current `frequency_slope` acoustic descriptor with
   `ridge_log_frequency_slope`, a ridge-tracked log-frequency slope measured in
   octaves per second.

The selected-feature panel should make the Event Encoder timeline useful as an
inspection surface: a user can select a tokenized event, see its token
assignment in context, and immediately see the acoustic descriptor values that
contributed to the non-CRNN part of the event vector.

The slope replacement should make the descriptor more robust to harmonic
energy. A harmonic contour has the same log-frequency slope as its fundamental
up to a constant offset, so a tracked ridge in log frequency should avoid the
plain-Hz artifact where higher harmonics look artificially steeper.

## 2. Scope

### In scope

- Extend the Event Encoder timeline data contract with per-event acoustic
  descriptor values and standardized descriptor-vector values.
- Render a selected-event feature table directly below the Event Encoder
  timeline viewer.
- Keep the table read-only and tied to the currently selected timeline event.
- Introduce `ridge_log_frequency_slope` in the Event Encoder descriptor order
  in place of `frequency_slope` for new jobs.
- Compute ridge slope from each event crop using an STFT limited to a
  configurable vocalization band.
- Track multiple per-frame spectral ridge candidates with a smoothness penalty.
- Fit a robust line to `log2(frequency_hz)` over time and store the slope in
  octaves per second.
- Record descriptor feature names in new Event Encoder manifests so downstream
  consumers can interpret descriptor-vector indexes.
- Add backend, pure-DSP, worker, and frontend tests.

### Non-goals

- Do not store full STFT matrices in Continuous Embedding artifacts for v1.
- Do not mutate existing Event Encoder artifacts.
- Do not rewrite old `frequency_slope` jobs.
- Do not add event editing or token editing to the Event Encoder detail page.
- Do not expose CRNN pooled embedding dimensions in the selected-feature table.
- Do not change timeline spectrogram PCEN rendering or playback normalization.

## 3. Existing Context

- Event Encoder jobs already compute event-level acoustic descriptors inside
  `src/humpback/sequence_models/event_encoder.py`.
- `compute_acoustic_descriptors()` currently builds an STFT-like magnitude
  spectrogram per event crop and derives `frequency_slope` by fitting Hz over
  time to the single loudest FFT bin per frame.
- `DESCRIPTOR_ORDER` currently contains:
  `duration`, `log_energy`, `peak_frequency`, `spectral_centroid`,
  `bandwidth`, `spectral_entropy`, `frequency_slope`, and `gap_to_previous`.
- The worker writes each raw descriptor as a column in both
  `event_vectors.parquet` and `event_tokens.parquet`.
- The worker writes `descriptor_vector` as the robust-zscored descriptor block
  in `event_vectors.parquet`; that is the non-CRNN-derived part of the final
  event vector.
- The Event Encoder timeline endpoint currently reads `event_tokens.parquet`
  and returns token/timing fields, but omits descriptor fields.
- The timeline panel selection state already lives inside
  `EventEncoderTimelinePanel`, making that component the natural owner of the
  selected-feature table.

## 4. Interpretation Of "Non-CRNN-Derived Vector Values"

For this design, "non-CRNN-derived vector values" means the acoustic descriptor
block that is concatenated with the CRNN pooled embedding block before
tokenization.

The selected-event table should show both:

- the raw descriptor value written as a named parquet column; and
- the standardized descriptor-vector value used in the event vector.

Showing both avoids a common confusion: the raw acoustic value explains the
signal, while the standardized vector value explains what k-means actually saw
after descriptor preprocessing.

The table should not show CRNN pooled embedding dimensions. Those dimensions are
large and not human-readable in this view.

## 5. Approaches Considered For The Selected-Feature Panel

### Approach A: Extend The Existing Timeline Endpoint

Extend `GET /sequence-models/event-encoders/{job_id}/timeline` so each returned
event can include descriptor values and descriptor-vector values. The endpoint
continues to read `event_tokens.parquet` for k-specific token assignments and
joins one row per event from `event_vectors.parquet` for the descriptor vector.

Pros:

- Keeps the selected-feature table synchronized with the timeline selection.
- Avoids an extra request on every event selection.
- Uses completed Event Encoder artifacts as the source of truth.
- Preserves raw/effective semantics because the completed job's artifacts are
  frozen.
- The payload remains compact because the descriptor block is small.

Cons:

- Slightly increases the timeline response size.
- Requires a join between token rows and vector rows.

Verdict: recommended.

### Approach B: Add A Selected-Event Feature Endpoint

Add an endpoint such as
`GET /sequence-models/event-encoders/{job_id}/events/{event_id}/features`.

Pros:

- Keeps the timeline endpoint minimal.
- Can return richer details later without changing the timeline response.

Cons:

- Adds selection-driven fetch latency.
- Requires more frontend loading/error states.
- Makes keyboard navigation noisier because each selected event may trigger a
  new request.

Verdict: not recommended for the first pass.

### Approach C: Reuse The Report Descriptor Summary

Keep the timeline endpoint unchanged and render values from `report.json`.

Pros:

- No backend API changes.

Cons:

- The report has aggregate descriptor summaries, not selected-event values.
- It cannot show the standardized descriptor vector for the selected event.

Verdict: rejected.

## 6. Recommended Selected-Feature API Contract

Extend `EventEncoderTimelineResponse` with descriptor metadata:

| Field | Type | Notes |
|---|---|---|
| `descriptor_feature_names` | string array | Ordered descriptor names for vector-value display |
| `descriptor_feature_units` | object | Optional unit labels keyed by descriptor name |

Extend `EventEncoderTimelineEvent` with selected-feature values:

| Field | Type | Notes |
|---|---|---|
| `descriptor_values` | object | Raw acoustic descriptor values keyed by descriptor name |
| `descriptor_vector_values` | object | Standardized descriptor-vector values keyed by descriptor name |

Backend behavior:

- Continue requiring a completed Event Encoder job and a valid
  `event_tokens.parquet` artifact for timeline rendering.
- Read `event_vectors.parquet` when present and join by
  `(source_sequence_key, sequence_index, event_id)`.
- If `event_vectors.parquet` is missing or lacks a matching row, return the
  timeline rows with empty descriptor-vector values and let the frontend show a
  feature-unavailable state for the table. This keeps the timeline usable for
  partially corrupt historical artifacts.
- For new jobs, prefer `manifest.descriptor_feature_names` when available.
- For older jobs without manifest metadata, infer feature names from parquet
  columns:
  - if `ridge_log_frequency_slope` exists, use the v2 descriptor order;
  - otherwise fall back to the v1 descriptor order containing
    `frequency_slope`.
- The endpoint should not reload current raw or effective Pass 2 events.

## 7. Recommended Selected-Feature UI Design

Keep the panel inside `EventEncoderTimelinePanel`, immediately below the
spectrogram and zoom selector, so it has direct access to selected event state.

The table title should identify the selected event and token compactly:
`Selected Event Features`, with a small token badge such as `T17` and the event
counter already shown in the timeline toolbar.

Columns:

| Column | Notes |
|---|---|
| `feature` | Descriptor name, shown exactly as the artifact field name |
| `raw value` | Raw acoustic descriptor value from parquet |
| `vector value` | Standardized descriptor-vector value used for tokenization |
| `unit` | Human-readable unit when known |

Units:

| Feature | Unit |
|---|---|
| `duration` | seconds |
| `log_energy` | log power |
| `peak_frequency` | Hz |
| `spectral_centroid` | Hz |
| `bandwidth` | Hz |
| `spectral_entropy` | normalized |
| `frequency_slope` | Hz/s, old jobs only |
| `ridge_log_frequency_slope` | octaves/s |
| `gap_to_previous` | seconds |

Frontend behavior:

- Initial load selects the first timeline event, so the table should populate
  immediately when feature values are available.
- Clicking a timeline event or using previous/next navigation updates the table
  without another request.
- Changing `k` preserves selected event identity when possible and keeps the
  same feature table, because event descriptors are event-level and not k-level.
- Empty jobs show the existing timeline empty state and no feature table.
- Complete jobs whose timeline loads but feature values are unavailable show a
  small muted message in the feature panel instead of failing the timeline.

## 8. Approaches Considered For STFT/Ridge Ownership

### Approach A: Compute Ridge Slope Inside Event Encoder Descriptor Extraction

Keep STFT and ridge tracking in `compute_acoustic_descriptors()` or a pure
helper called by it. Persist only the resulting descriptor value in
`event_vectors.parquet`, `event_tokens.parquet`, `manifest.json`, and
`report.json`.

Pros:

- Matches the existing descriptor ownership model.
- Uses event crops that are already loaded by the Event Encoder worker.
- Keeps Continuous Embedding idempotency focused on region-CRNN chunks.
- Avoids coupling Continuous Embedding to raw/effective Event Encoder source
  semantics.
- Avoids storing large STFT sidecars that are not currently needed downstream.

Cons:

- If multiple future consumers need the same STFT/ridge traces, they will
  recompute them.
- Debugging ridge paths will rely on tests and optional logging unless a future
  trace sidecar is added.

Verdict: recommended for v1.

### Approach B: Add STFT Sidecars To Continuous Embedding Jobs

Extend Continuous Embedding jobs to build and persist STFT spectrograms or ridge
features for downstream consumers.

Pros:

- Potentially reusable by future analysis workflows.
- Could amortize STFT cost across multiple Event Encoder jobs.

Cons:

- Continuous Embedding region-CRNN artifacts are region-scoped, while the ridge
  slope is event-crop and raw/effective-source-mode scoped.
- Effective Event Encoder jobs can change event boundaries without changing a
  CRNN Continuous Embedding job, so a Continuous Embedding STFT sidecar would
  need its own event-source provenance layer.
- It would expand the Continuous Embedding artifact and idempotency contract for
  a descriptor that currently belongs to Event Encoder tokenization.

Verdict: rejected for this feature.

### Approach C: Store Event Encoder Ridge Trace Sidecars

Compute ridge slope inside Event Encoder and also persist per-event ridge paths
to a sidecar artifact such as `ridge_traces.parquet`.

Pros:

- Useful for debugging and visualization.
- Keeps traces aligned with Event Encoder source semantics.

Cons:

- Adds artifact volume and UI/API surface before there is a concrete consumer.
- More implementation work than needed to replace one vector feature.

Verdict: defer. Add later if ridge QA needs visual inspection.

## 9. Recommended Ridge Algorithm

Add a pure helper, for example `compute_ridge_log_frequency_slope()`, called
from `compute_acoustic_descriptors()`.

Inputs:

- one-dimensional event audio;
- sample rate;
- `n_fft`;
- `hop_length`;
- `eps`;
- vocalization band minimum frequency, default `100.0` Hz;
- vocalization band maximum frequency, default `3000.0` Hz;
- per-frame candidate count, default `5`;
- smoothness penalty, default `8.0`, applied in log-frequency space;
- peak-prominence ratio, default `0.0`, for optional local-peak filtering.

Processing:

1. Frame the event crop using the same frame/window path as the existing
   descriptor extraction.
2. Compute the magnitude STFT with a Hann window.
3. Restrict analysis bins to the configured frequency band, clipped to
   `(0, Nyquist]`.
4. For each frame, find candidate ridge bins in the band:
   - prefer local maxima from the magnitude spectrum;
   - select the strongest `ridge_candidate_count` peaks;
   - fall back to the strongest bins if a frame has no local maxima.
5. Convert candidate frequencies to `log2(frequency_hz)`.
6. Track the best path through frames with dynamic programming:
   - emission cost prefers stronger spectral magnitude;
   - transition cost penalizes squared jumps in log2 frequency;
   - optional max-jump gating can reject implausible frame-to-frame jumps.
7. Recover the minimum-cost ridge path.
8. Fit a robust line to `log2(frequency_hz)` over frame time.
9. Return the fitted slope in octaves per second.

Robust fit:

- Use Theil-Sen slope when at least three ridge frames are available.
- Fall back to a standard linear fit for two ridge frames.
- Return `0.0` when there are fewer than two usable ridge frames or the band is
  invalid.
- Never return NaN or infinity.

Why octaves per second:

- The raw fit target is `log2(frequency_hz)`.
- The slope is therefore naturally measured as octaves per second.
- A UI can display cents per second by multiplying by `1200`, but the stored
  vector feature should remain `ridge_log_frequency_slope` in octaves per
  second.

## 10. Descriptor Config Changes

Extend `EventEncoderDescriptorConfig` with ridge settings:

| Field | Default | Notes |
|---|---:|---|
| `ridge_min_frequency_hz` | `100.0` | Inclusive lower vocalization-band edge |
| `ridge_max_frequency_hz` | `3000.0` | Inclusive upper vocalization-band edge, clipped to Nyquist |
| `ridge_candidate_count` | `5` | Candidate ridge bins retained per frame |
| `ridge_smoothness_penalty` | `8.0` | Penalty per squared octave jump |
| `ridge_peak_prominence_ratio` | `0.0` | Optional prominence threshold as a fraction of the frame peak |

Validation:

- `ridge_min_frequency_hz` must be greater than zero.
- `ridge_max_frequency_hz` must be greater than
  `ridge_min_frequency_hz`.
- `ridge_candidate_count` must be positive.
- smoothness and prominence-ratio settings must be non-negative.
- `ridge_peak_prominence_ratio` should be less than or equal to `1.0`.

The descriptor config is already included in `tokenization_signature`, so these
settings naturally participate in Event Encoder idempotency.

## 11. Artifact And Versioning Contract

Because this changes the meaning and name of one vector feature, new jobs should
use a new default tokenizer version:

- old default: `crnn-event-encoder-v1`;
- new default: `crnn-event-encoder-v2`.

New v2 jobs:

- write `ridge_log_frequency_slope` instead of `frequency_slope`;
- write descriptor values in the new descriptor order;
- write `descriptor_feature_names` to `manifest.json`;
- include `descriptor_feature_names` in `report.json` or make it available via
  the timeline endpoint;
- expose `ridge_log_frequency_slope` in descriptor summaries.

Old v1 jobs:

- remain readable as completed artifacts;
- continue to show `frequency_slope` in the descriptor summary and selected
  feature table;
- are not rewritten or migrated.

Implementation detail:

- Prefer explicit descriptor orders such as `DESCRIPTOR_ORDER_V1` and
  `DESCRIPTOR_ORDER_V2`, with `DESCRIPTOR_ORDER` pointing to the active default
  order only where safe.
- Worker parquet schemas should be generated from the selected descriptor order
  rather than hard-coding one global schema if explicit v1 job creation remains
  supported.
- If the product chooses not to support creating new v1 jobs, the API should
  still read v1 artifacts but new submissions should default to v2.

## 12. Backend Implementation Outline

Likely affected files:

- `src/humpback/sequence_models/event_encoder.py`
- `src/humpback/workers/event_encoder_worker.py`
- `src/humpback/schemas/sequence_models.py`
- `src/humpback/api/routers/sequence_models.py`
- `src/humpback/services/event_encoder_service.py`
- `docs/reference/sequence-models-api.md`
- `docs/reference/storage-layout.md`

No Alembic migration is expected because these changes affect serialized
configs, parquet artifacts, JSON sidecars, API response schemas, and frontend
types, not database columns.

Steps:

1. Add descriptor order constants for v1 and v2.
2. Add ridge descriptor config fields and validators.
3. Implement pure ridge candidate selection, dynamic-programming path tracking,
   and robust log-frequency slope fitting.
4. Update `compute_acoustic_descriptors()` to emit
   `ridge_log_frequency_slope` for v2 descriptor extraction.
5. Update Event Encoder worker schema construction so descriptor columns follow
   the selected descriptor order.
6. Write `descriptor_feature_names` into `manifest.json`.
7. Extend timeline response models with descriptor metadata and per-event
   descriptor maps.
8. Extend the timeline route to read `event_vectors.parquet`, join descriptor
   vectors to selected token rows, and preserve compatibility with old v1
   artifacts.

## 13. Frontend Implementation Outline

Likely affected files:

- `frontend/src/api/sequenceModels.ts`
- `frontend/src/components/sequence-models/EventEncoderTimelinePanel.tsx`
- `frontend/e2e/sequence-models/event-encoder.spec.ts`

Add an internal component in the timeline panel, for example
`SelectedEventFeatureTable`.

The component receives:

- selected event;
- descriptor feature names;
- descriptor units;
- raw descriptor values;
- standardized descriptor-vector values.

Display behavior:

- Use compact table styling consistent with the existing report tables.
- Use the artifact field names as labels, not explanatory prose.
- Format finite numbers with the existing `formatNumber` convention or a local
  compact formatter.
- Preserve table dimensions enough that selecting another event does not cause
  the timeline area to jump dramatically.

## 14. Tests

### Pure Sequence Models tests

Add or update tests in `tests/sequence_models/test_event_encoder.py`:

- A constant-frequency sine returns ridge slope near zero.
- A log-frequency chirp returns a positive slope near the expected
  octaves-per-second value.
- A harmonic-dominant chirp returns approximately the same
  `ridge_log_frequency_slope` as the fundamental-dominant chirp.
- Low-energy, empty, too-short, or invalid-band inputs return finite zero-like
  values.
- `descriptor_vector()` follows the v2 descriptor order and has the expected
  shape.

### Worker tests

Update `tests/workers/test_event_encoder_worker.py`:

- New event vector and token parquet rows include
  `ridge_log_frequency_slope` and no v2 `frequency_slope` column.
- `manifest.json` records `descriptor_feature_names`.
- Descriptor summaries include `ridge_log_frequency_slope`.

### API tests

Update `tests/integration/test_sequence_models_api.py`:

- Timeline responses include descriptor feature metadata.
- Timeline event rows include raw descriptor values and standardized vector
  values.
- Old fixture rows with `frequency_slope` still return selected-feature values
  without breaking timeline rendering.

### Frontend tests

Update `frontend/e2e/sequence-models/event-encoder.spec.ts`:

- Mock timeline rows with descriptor values.
- Assert the selected-feature panel appears below the timeline viewer.
- Assert selecting next/previous events updates the feature table.
- Assert v2 rows display `ridge_log_frequency_slope`.
- Assert old rows can display `frequency_slope` if returned by the API.

Verification commands:

- `uv run pytest tests/sequence_models/test_event_encoder.py -q`
- `uv run pytest tests/workers/test_event_encoder_worker.py -q`
- `uv run pytest tests/integration/test_sequence_models_api.py -q`
- `cd frontend && npx playwright test e2e/sequence-models/event-encoder.spec.ts`
- `cd frontend && npx tsc --noEmit`
- Final backend gate remains `uv run pytest tests/`.

## 15. Risks And Follow-Ups

- Ridge tracking parameters may need calibration on real humpback events. The
  first implementation should keep parameters configurable and write them into
  job provenance.
- The selected-feature table is only as interpretable as the descriptor names.
  It should display concise units, but deeper explanatory text belongs in docs,
  not the operational UI.
- Returning descriptor maps for every timeline event is acceptable for the
  current descriptor count. If future non-CRNN feature blocks become large,
  move selected-feature details to a dedicated endpoint.
- If ridge path QA becomes important, add an Event Encoder
  `ridge_traces.parquet` sidecar and an optional overlay later.
