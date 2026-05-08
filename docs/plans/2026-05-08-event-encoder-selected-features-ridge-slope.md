# Event Encoder Selected Features And Ridge Slope Implementation Plan

**Goal:** Add selected-event non-CRNN feature inspection to the Event Encoder timeline and replace `frequency_slope` with ridge-tracked log-frequency slope.
**Spec:** [docs/specs/2026-05-08-event-encoder-selected-features-ridge-slope-design.md](../specs/2026-05-08-event-encoder-selected-features-ridge-slope-design.md)
**Primary domain:** sequence-models
**Neighbor domains:** call-parsing, signal-timeline, frontend-shell

---

### Task 1: Add Ridge Descriptor Configuration And Pure DSP

**Files:**
- Modify: `src/humpback/schemas/sequence_models.py`
- Modify: `src/humpback/sequence_models/event_encoder.py`
- Modify: `tests/unit/test_sequence_models_schemas.py`
- Modify: `tests/sequence_models/test_event_encoder.py`

**Acceptance criteria:**
- [x] `EventEncoderDescriptorConfig` exposes validated ridge settings for min frequency, max frequency, candidate count, smoothness penalty, and peak-prominence ratio.
- [x] The active descriptor order replaces `frequency_slope` with `ridge_log_frequency_slope` for Event Encoder jobs.
- [x] The code does not retain a `frequency_slope` artifact compatibility path because existing v1 jobs were deleted before implementation.
- [x] Ridge slope calculation frames event audio, limits candidate bins to the configured vocalization band, tracks a continuous ridge through frame candidates with a smoothness penalty, and returns a finite octaves-per-second value.
- [x] Robust line fitting uses log2 frequency over time, returns near zero for flat tones, and avoids NaN or infinity for empty, silent, short, or invalid-band inputs.
- [x] Existing acoustic descriptors other than the slope replacement keep their current semantics and units.

**Tests needed:**
- Schema validation coverage for ridge descriptor config defaults and invalid values.
- Pure Sequence Models tests for constant sine, log-frequency chirp, harmonic-dominant chirp, empty/silent/short inputs, descriptor order, and descriptor-vector shape.

---

### Task 2: Update Event Encoder Worker Artifacts And Version Contract

**Files:**
- Modify: `src/humpback/workers/event_encoder_worker.py`
- Modify: `src/humpback/services/event_encoder_service.py`
- Modify: `src/humpback/schemas/sequence_models.py`
- Modify: `src/humpback/models/sequence_models.py`
- Modify: `tests/workers/test_event_encoder_worker.py`
- Modify: `tests/services/test_event_encoder_service.py`

**Acceptance criteria:**
- [x] New Event Encoder submissions default to `crnn-event-encoder-v2`.
- [x] Event Encoder tokenization signatures include the new descriptor config values through the existing serialized descriptor config path.
- [x] New `event_vectors.parquet` and `event_tokens.parquet` artifacts include `ridge_log_frequency_slope` and do not include `frequency_slope`.
- [x] Worker parquet schema construction follows the active descriptor order rather than hard-coding stale descriptor columns.
- [x] `manifest.json` records ordered `descriptor_feature_names` and descriptor config provenance.
- [x] `report.json` descriptor summaries include `ridge_log_frequency_slope`.
- [x] No `frequency_slope` creation or read-compatibility path remains in active Event Encoder code.
- [x] No Alembic migration is added because this change affects serialized configs, parquet artifacts, JSON sidecars, API responses, and frontend types, not database columns.

**Tests needed:**
- Worker tests that inspect written parquet columns, manifest descriptor metadata, report descriptor summaries, and successful completion with v2 defaults.
- Service tests that verify idempotency still reuses matching signatures and changes when ridge descriptor settings change.

---

### Task 3: Extend Event Encoder Timeline API With Selected Feature Values

**Files:**
- Modify: `src/humpback/schemas/sequence_models.py`
- Modify: `src/humpback/api/routers/sequence_models.py`
- Modify: `tests/integration/test_sequence_models_api.py`

**Acceptance criteria:**
- [x] `EventEncoderTimelineResponse` includes ordered descriptor feature names and feature units.
- [x] `EventEncoderTimelineEvent` includes raw descriptor values and standardized descriptor-vector values keyed by descriptor name.
- [x] The timeline endpoint reads `event_tokens.parquet` for selected-k token assignments and joins matching rows from `event_vectors.parquet` by source sequence, sequence index, and event id.
- [x] The timeline endpoint keeps completed Event Encoder artifacts authoritative and does not reload current raw or effective Pass 2 events.
- [x] Missing or corrupt `event_vectors.parquet` does not break timeline token rendering; affected events return empty descriptor-vector values for the frontend unavailable state.
- [x] Descriptor feature names are read from manifest metadata when available and otherwise inferred from active artifact columns.
- [x] Existing endpoint error behavior for missing jobs, incomplete jobs, missing token artifacts, invalid k values, and non-region-CRNN provenance remains intact.

**Tests needed:**
- Integration tests for descriptor metadata, raw descriptor values, standardized vector values, ridge fields, missing vector artifact behavior, and preservation of existing timeline endpoint status codes.

---

### Task 4: Add Frontend Selected-Feature Panel Below Timeline Viewer

**Files:**
- Modify: `frontend/src/api/sequenceModels.ts`
- Modify: `frontend/src/components/sequence-models/EventEncoderTimelinePanel.tsx`
- Modify: `frontend/e2e/sequence-models/event-encoder.spec.ts`

**Acceptance criteria:**
- [x] Frontend timeline types include descriptor feature names, feature units, raw descriptor values, and standardized descriptor-vector values.
- [x] The Event Encoder timeline panel renders a read-only selected-feature table directly below the spectrogram and zoom selector.
- [x] The table updates when the user clicks an event, uses previous/next controls, uses `A` or `D`, or changes k while preserving the selected event.
- [x] The table displays feature name, raw value, vector value, and unit for the selected event.
- [x] Timeline mocks display `ridge_log_frequency_slope`.
- [x] Complete jobs with no selected feature values show a muted unavailable state without hiding the timeline.
- [x] Timeline playback, keyboard handling, k selection, and token overlay behavior keep their current contracts.

**Tests needed:**
- Playwright coverage for selected-feature panel placement, selected-event updates, ridge slope display, unavailable state, and regression coverage for existing timeline navigation and k switching.
- TypeScript compile coverage for the extended API types.

---

### Task 5: Update Reference Docs And Agent Context

**Files:**
- Modify: `docs/reference/sequence-models-api.md`
- Modify: `docs/reference/storage-layout.md`
- Modify: `docs/reference/behavioral-constraints.md`
- Modify: `docs/agent-context/domains/sequence-models/README.md`
- Modify: `docs/agent-context/domains/sequence-models/invariants.md`
- Modify: `docs/agent-context/domains/sequence-models/tests.md`

**Acceptance criteria:**
- [x] Sequence Models API reference documents the extended Event Encoder timeline response and selected-feature fields.
- [x] Storage layout reference documents Event Encoder descriptor metadata and `ridge_log_frequency_slope` artifact columns.
- [x] Behavioral constraints preserve Event Encoder raw/effective artifact-authoritative semantics and token-label job-local semantics.
- [x] Sequence Models domain capsule records the selected-feature timeline panel as part of the active Event Encoder UI surface.
- [x] Sequence Models tests capsule includes the targeted ridge descriptor, worker, API, and frontend verification commands.
- [x] Documentation clearly states that full STFT matrices are not persisted in Continuous Embedding artifacts for this feature.

**Tests needed:**
- Documentation diff review and `git diff --check`.

---

### Verification

Run in order after all tasks:

1. `git diff --check`
2. `uv run ruff format --check src/humpback/schemas/sequence_models.py src/humpback/sequence_models/event_encoder.py src/humpback/workers/event_encoder_worker.py src/humpback/services/event_encoder_service.py src/humpback/models/sequence_models.py src/humpback/api/routers/sequence_models.py tests/unit/test_sequence_models_schemas.py tests/sequence_models/test_event_encoder.py tests/workers/test_event_encoder_worker.py tests/services/test_event_encoder_service.py tests/integration/test_sequence_models_api.py`
3. `uv run ruff check src/humpback/schemas/sequence_models.py src/humpback/sequence_models/event_encoder.py src/humpback/workers/event_encoder_worker.py src/humpback/services/event_encoder_service.py src/humpback/models/sequence_models.py src/humpback/api/routers/sequence_models.py tests/unit/test_sequence_models_schemas.py tests/sequence_models/test_event_encoder.py tests/workers/test_event_encoder_worker.py tests/services/test_event_encoder_service.py tests/integration/test_sequence_models_api.py`
4. `uv run pyright`
5. `uv run pytest tests/sequence_models/test_event_encoder.py tests/unit/test_sequence_models_schemas.py -q`
6. `uv run pytest tests/services/test_event_encoder_service.py tests/workers/test_event_encoder_worker.py -q`
7. `uv run pytest tests/integration/test_sequence_models_api.py -q`
8. `cd frontend && npx tsc --noEmit`
9. `cd frontend && npx playwright test e2e/sequence-models/event-encoder.spec.ts`
10. `uv run pytest tests/`
