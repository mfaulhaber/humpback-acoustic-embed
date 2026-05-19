# Event Encoder Ridge Frequency Descriptors Implementation Plan

**Goal:** Add robust ridge-frequency summary descriptors and use them to render high-frequency Event Encoder piano roll tokens with truthful vertical placement and height.
**Spec:** `docs/specs/2026-05-19-event-encoder-ridge-frequency-descriptors-design.md`
**Primary domain:** sequence-models
**Neighbor domains:** signal-timeline, frontend-shell

---

### Task 1: Add Ridge Summary Descriptor Extraction

**Files:**
- Modify: `src/humpback/sequence_models/event_encoder.py`
- Modify: `tests/sequence_models/test_event_encoder.py`

**Acceptance criteria:**
- [ ] `DESCRIPTOR_ORDER` appends `ridge_median_frequency`, `ridge_low_frequency`, `ridge_high_frequency`, `ridge_frequency_span`, `ridge_coverage`, `ridge_energy_ratio`, `band_limited_peak_frequency`, and `high_band_energy_ratio` after the existing 14 descriptors
- [ ] Descriptor units include the eight appended ridge display descriptors
- [ ] Ridge summary extraction reuses the existing STFT ridge path work rather than adding a separate pitch estimator
- [ ] Ridge low/high frequency descriptors use configurable trimmed percentiles instead of literal path min/max
- [ ] Ridge coverage reports tracked ridge frames divided by total STFT frames
- [ ] Ridge energy ratio reports a finite normalized confidence-like value suitable for frontend trust gating
- [ ] Band-limited peak frequency excludes configurable low-frequency rumble while preserving the legacy full-spectrum `peak_frequency`
- [ ] High-band energy ratio is finite and defaults to 0.0 for silence or degenerate inputs
- [ ] Synthetic tests cover high-frequency sine/chirp, low-rumble plus high whistle, trimmed-bound outlier resistance, silence, and descriptor vector shape

**Tests needed:**
- Unit tests in `tests/sequence_models/test_event_encoder.py` for ridge medians/bounds, rumble-resistant band peak, degenerate fallbacks, and the new 22-entry descriptor vector

---

### Task 2: Version Event Encoder Defaults And Schema Validation

**Files:**
- Modify: `src/humpback/schemas/sequence_models.py`
- Modify: `src/humpback/workers/event_encoder_worker.py`
- Modify: `tests/unit/test_sequence_models_schemas.py`
- Modify: `tests/services/test_event_encoder_service.py`
- Modify: `tests/workers/test_event_encoder_worker.py`

**Acceptance criteria:**
- [ ] New Event Encoder create defaults use `crnn-event-encoder-v3`
- [ ] Default `ridge_max_frequency_hz` is raised to `6000.0` for new jobs
- [ ] Descriptor config accepts and validates `ridge_summary_low_percentile`, `ridge_summary_high_percentile`, `band_peak_min_frequency_hz`, and `band_peak_max_frequency_hz`
- [ ] Percentile validation requires `0 <= low < high <= 100`
- [ ] Band peak validation requires positive bounds with max greater than min when both are configured
- [ ] Default `descriptor_weight` is adjusted to `0.364` for new jobs while explicit preprocessing configs remain honored
- [ ] Event Encoder worker forwards the new descriptor config values into descriptor extraction
- [ ] Tokenization signatures change when ridge descriptor config values change
- [ ] Existing completed v2 artifacts remain readable because timeline endpoints derive descriptor fields from each artifact manifest

**Tests needed:**
- Schema tests for new defaults and validation errors
- Service signature tests proving new descriptor config fields affect idempotency
- Worker tests proving artifacts include the appended descriptor columns and manifest names

---

### Task 3: Add Frontend Ridge Display Band Helper

**Files:**
- Create: `frontend/src/components/sequence-models/eventEncoderDisplayBand.ts`
- Create: `frontend/src/components/sequence-models/eventEncoderDisplayBand.test.ts`

**Acceptance criteria:**
- [ ] Helper resolves a trusted ridge band from `ridge_median_frequency`, trimmed ridge bounds, `ridge_coverage`, and `ridge_energy_ratio`
- [ ] Helper falls back to voiced `median_f0` when ridge trust is weak and voicing is above threshold
- [ ] Helper falls back to `band_limited_peak_frequency` for unvoiced or failed-F0 events
- [ ] Helper preserves legacy behavior for artifacts missing v3 ridge fields
- [ ] Helper returns a compact minimum-height band for center-only fallbacks
- [ ] Helper clamps invalid or non-finite descriptor values to safe fallback behavior

**Tests needed:**
- Vitest coverage for trusted ridge bands, weak ridge fallback to F0, unvoiced fallback to band-limited peak, missing v3 fields, and invalid descriptor values

---

### Task 4: Render Ridge Bands In The Piano Roll

**Files:**
- Modify: `frontend/src/components/sequence-models/EventEncoderPianoRollPage.tsx`
- Modify: `frontend/e2e/sequence-models/event-encoder-piano-roll.spec.ts`

**Acceptance criteria:**
- [ ] Piano roll Y-mode selector includes `Ridge`
- [ ] Jobs with ridge descriptor fields default to Ridge mode; older artifacts keep the existing default behavior
- [ ] Event rectangles in Ridge mode use ridge low/high bounds for vertical extent when ridge trust gates pass
- [ ] Event rectangles still render as one token rectangle per event
- [ ] Slope line continues to use `ridge_log_frequency_slope`
- [ ] Tooltips show compact ridge summary rows when ridge descriptor fields are present
- [ ] Frequency max choices include 6000 Hz without automatically changing the user's current range
- [ ] Existing F0 and Peak Frequency modes remain available
- [ ] E2E mocks cover both legacy v2 data and v3 high-frequency ridge data

**Tests needed:**
- Updated Playwright coverage for v2 route compatibility, v3 high-frequency ridge token placement, 6000 Hz option, and tooltip ridge rows
- TypeScript coverage through `npx tsc --noEmit`

---

### Task 5: Update Documentation And Domain Context

**Files:**
- Modify: `docs/agent-context/domains/sequence-models/README.md`
- Modify: `docs/agent-context/domains/sequence-models/invariants.md`
- Modify: `docs/reference/behavioral-constraints.md`
- Modify: `docs/reference/sequence-models-api.md`
- Modify: `docs/reference/storage-layout.md`
- Modify: `docs/reference/frontend.md`

**Acceptance criteria:**
- [ ] Sequence Models context records the 22-entry v3 descriptor order
- [ ] Invariants distinguish v2 artifact readability from v3 default descriptor output
- [ ] Behavioral constraints document ridge summaries, trimmed bounds, and artifact-authoritative timeline rendering
- [ ] Sequence Models API reference documents new descriptor config fields and v3 defaults
- [ ] Storage layout documents appended ridge descriptor columns in Event Encoder parquet artifacts
- [ ] Frontend reference documents Ridge mode as the preferred piano roll display for v3 artifacts

**Tests needed:**
- Documentation review plus the targeted backend/frontend tests from prior tasks

---

### Verification

Run in order after all tasks:

1. `uv run ruff format --check src/humpback/sequence_models/event_encoder.py src/humpback/schemas/sequence_models.py src/humpback/workers/event_encoder_worker.py tests/sequence_models/test_event_encoder.py tests/unit/test_sequence_models_schemas.py tests/services/test_event_encoder_service.py tests/workers/test_event_encoder_worker.py`
2. `uv run ruff check src/humpback/sequence_models/event_encoder.py src/humpback/schemas/sequence_models.py src/humpback/workers/event_encoder_worker.py tests/sequence_models/test_event_encoder.py tests/unit/test_sequence_models_schemas.py tests/services/test_event_encoder_service.py tests/workers/test_event_encoder_worker.py`
3. `uv run pyright src/humpback/sequence_models/event_encoder.py src/humpback/schemas/sequence_models.py src/humpback/workers/event_encoder_worker.py`
4. `uv run pytest tests/sequence_models/test_event_encoder.py tests/unit/test_sequence_models_schemas.py tests/services/test_event_encoder_service.py tests/workers/test_event_encoder_worker.py -q`
5. `cd frontend && npx vitest run src/components/sequence-models/eventEncoderDisplayBand.test.ts`
6. `cd frontend && npx tsc --noEmit`
7. `cd frontend && npx playwright test e2e/sequence-models/event-encoder-piano-roll.spec.ts`
8. `cd frontend && npx playwright test e2e/sequence-models/event-encoder.spec.ts`
9. `uv run pytest tests/`

