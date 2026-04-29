# CRNN Region-Based HMM Embedding Source — Implementation Plan

**Goal:** Add a second embedding source to the Sequence Models track — Pass 2 CRNN BiGRU activations sliced into 250 ms chunks, computed per Pass 1 detection region, fed through PCA + Gaussian HMM with three training modes (full-region / event-balanced / event-only) — without disturbing the existing SurfPerch event-padded path.

**Spec:** [docs/specs/2026-04-29-crnn-region-hmm-design.md](../specs/2026-04-29-crnn-region-hmm-design.md)

**Branch:** `feature/crnn-region-hmm`

---

## Task ordering rationale

Tasks 1–3 land the Pass 2 refactor with a regression test as the first checkpoint — this is the highest-risk change and gating on it early lets the rest of the work proceed against a stable foundation. Tasks 4–6 build the producer pieces in dependency order. Task 7 lands the migration (with the mandatory backup gate). Task 8 wires the producer into the worker. Task 9 wires the HMM consumer. Tasks 10–11 land the API and schemas. Tasks 12–14 land the frontend. Task 15 lands documentation. The final verification gate runs the full project verification matrix from CLAUDE.md §10.2.

---

### Task 1: Extract `iter_inference_windows()` shared helper

**Files:**
- Create: `src/humpback/call_parsing/segmentation/window_iter.py`
- Modify: `src/humpback/call_parsing/segmentation/inference.py`

**Acceptance criteria:**
- [ ] New module exposes a single pure function `iter_inference_windows(region, audio, sample_rate, window_seconds, hop_seconds)` that yields `(window_audio: ndarray, frame_offset_in_region: int)` tuples
- [ ] Constants `_MAX_WINDOW_SEC` and `_WINDOW_HOP_SEC` (or their replacements) live in the new module
- [ ] `inference.py` imports and calls the helper instead of inlining the windowing math
- [ ] No other Pass 2 source files modified
- [ ] `_infer_windowed()` and `_infer_single()` retain their public signatures and behavior

**Tests needed:**
- `tests/call_parsing/test_window_iter.py` — golden-output tests covering: region exactly N×hop long, region with non-integer window count, region shorter than one window, region exactly one window long. Asserts produced `(audio_slice_length, frame_offset)` pairs match expected sequence for each case.

---

### Task 2: Pass 2 refactor regression test

**Files:**
- Create: `tests/call_parsing/test_pass2_refactor_regression.py`

**Acceptance criteria:**
- [ ] Loads a fixture region (synthetic or existing test fixture) and a fixture stub `SegmentationCRNN` checkpoint
- [ ] Runs `event_segmentation_worker._run_inference_pipeline()` end-to-end against this fixture
- [ ] Captures the output `events.parquet` row-by-row
- [ ] Compares against a committed golden file (`tests/call_parsing/fixtures/pass2_golden_events.parquet`) — byte-identical via parquet equality (compare row count, column dtypes, all values to within float64 precision)
- [ ] Test fails clearly if Task 1's refactor changes any output

**Tests needed:**
- (this task IS the test)

---

### Task 3: `chunk_projection.py` — projection abstraction

**Files:**
- Create: `src/humpback/sequence_models/chunk_projection.py`

**Acceptance criteria:**
- [ ] `ChunkProjection` Protocol declares `output_dim: int`, `fit(X) -> None`, `transform(X) -> ndarray`, `save(path) -> None`, classmethod `load(cls, path) -> ChunkProjection`
- [ ] `IdentityProjection(input_dim)` implements pass-through; `output_dim == input_dim`; `fit` is a no-op
- [ ] `RandomProjection(output_dim, seed)` wraps `sklearn.random_projection.GaussianRandomProjection` with deterministic output for a fixed seed
- [ ] `PCAProjection(output_dim, whiten)` wraps `sklearn.decomposition.PCA`
- [ ] All three persist via `joblib.dump`/`joblib.load`
- [ ] All three pass mypy/pyright

**Tests needed:**
- `tests/sequence_models/test_chunk_projection.py` — Identity passes through unchanged; Random produces stable output for a fixed seed and changes when seed changes; PCA fits and transforms a small fixture; round-trip save/load preserves transform output for all three.

---

### Task 4: `crnn_features.py` — chunk-embedding extractor

**Files:**
- Create: `src/humpback/sequence_models/crnn_features.py`

**Acceptance criteria:**
- [ ] Public function `extract_chunk_embeddings(checkpoint_path, audio, region, chunk_size_seconds, chunk_hop_seconds, projection, device, sample_rate)` returns a `ChunkEmbeddingResult` dataclass with fields `embeddings: ndarray[T_chunks, D_out]`, `call_probabilities: ndarray[T_chunks]`, `chunk_starts: ndarray[T_chunks]`, `chunk_ends: ndarray[T_chunks]`
- [ ] Loads `SegmentationCRNN` from checkpoint via existing checkpoint loader; checkpoint sha256 is also exposed via a separate helper
- [ ] Registers a `register_forward_hook` on the BiGRU module (no edit to `SegmentationCRNN.forward()`); hook captures activations into a buffer cleared per region
- [ ] Drives forward via `iter_inference_windows()`; stitches overlapping windows by keeping the centre half (mirroring Pass 2's stitching)
- [ ] Slices stitched activations into 8-frame chunks at the requested chunk hop; concatenates frames within each chunk; applies `projection`
- [ ] Computes `call_probability` per chunk as the mean of per-frame sigmoid probs over the 8 frames
- [ ] Load-time assertions: BiGRU `hidden_size * 2 == 128` and frame_rate is 32 fps; raise a clear error otherwise

**Tests needed:**
- `tests/sequence_models/test_crnn_features.py` — uses a tiny stub `SegmentationCRNN` (deterministic weights); asserts: hook fires once per window forward; chunk count matches `(padded_end - padded_start) / chunk_hop_seconds` ± window-edge tolerance; 8-frame slicing produces correct chunk count at 250 ms and 125 ms hops; `call_probability` equals mean of 8 known frame probs; chunk timestamps align with `padded_start_sec + chunk_index * hop`; load-time guards raise on a stub checkpoint with wrong BiGRU width.

---

### Task 5: `event_overlap_join.py` — events parquet join helper

**Files:**
- Create: `src/humpback/sequence_models/event_overlap_join.py`

**Acceptance criteria:**
- [ ] Public function `compute_chunk_event_metadata(chunks, region_id, events_for_region, event_core_overlap_threshold, near_event_window_seconds)` returns aligned arrays for `event_overlap_fraction`, `nearest_event_id`, `distance_to_nearest_event_seconds`, `tier`
- [ ] `event_overlap_fraction` = (chunk ∩ union(events)) / chunk_duration
- [ ] `tier` derivation: `event_overlap_fraction >= threshold` → `event_core`; else if `min_distance_to_event <= near_event_window_seconds` → `near_event`; else `background`
- [ ] `nearest_event_id` and `distance_to_nearest_event_seconds` are null when the chunk is `background`
- [ ] Vectorized over chunks; no Python-level loop over events × chunks

**Tests needed:**
- `tests/sequence_models/test_event_overlap_join.py` — chunk fully inside an event → `event_core`, `event_overlap_fraction == 1.0`, `distance == 0.0`; chunk 4 s from nearest event → `near_event`; chunk in a region with no events → `tier="background"`, `nearest_event_id is None`, `distance_to_nearest_event_seconds is None`; partially overlapping chunk → exact fraction.

---

### Task 6: `region_sampling.py` — training-set builder

**Files:**
- Create: `src/humpback/sequence_models/region_sampling.py`

**Acceptance criteria:**
- [ ] Public function `build_training_set(region_sequences, mode, tier_config, sampling_config)` returns a `TrainingSet` dataclass with fields `sub_sequences: list[ndarray]`, `lengths: ndarray[int]`, `was_used_for_training_per_region: dict[region_id, ndarray[bool]]`
- [ ] `mode == "full_region"`: training set = all chunks of all regions ≥ `min_sequence_length_frames`; uniform region subsample if total exceeds `target_train_chunks`
- [ ] `mode == "event_balanced"`: stratified sub-sequence extraction; `event_core` sub-sequences centred on each event_core chunk with stride `subsequence_stride_chunks`; `near_event` and `background` sub-sequences sampled from matching tier chunks; cap by `target_train_chunks` while preserving `event_balanced_proportions`
- [ ] `mode == "event_only"`: same as event_balanced but excluding `background` tier
- [ ] `was_used_for_training` mask aligned to source region order; `True` only for chunks inside a sampled sub-sequence
- [ ] Deterministic for a fixed `random_seed` config field (added to `sampling_config`)

**Tests needed:**
- `tests/sequence_models/test_region_sampling.py` — Mode A produces all chunks; Mode B produces tier-balanced sub-sequences with proportions within ±5% of `{0.4, 0.35, 0.25}` on a synthetic dataset; Mode B respects `target_train_chunks`; Mode C excludes background; `was_used_for_training` mask is consistent with sub-sequence membership; deterministic across two runs with same seed.

---

### Task 7: Alembic migration `061_crnn_region_embeddings`

**Files:**
- Create: `alembic/versions/061_crnn_region_embeddings.py`
- Modify: `src/humpback/database.py`

**Acceptance criteria:**
- [ ] **MANDATORY DB BACKUP (CLAUDE.md §3.5):** Before running this migration, read `HUMPBACK_DATABASE_URL` from `.env`, copy the database file to `<original_path>.YYYY-MM-DD-HH:mm.bak` (UTC), and confirm the backup file exists and has non-zero size. Implementation must include the explicit shell commands in the task PR description and verify them locally before running `uv run alembic upgrade head`. If the backup step fails or is skipped, **stop** — do not apply the migration.
- [ ] Adds nullable columns to `continuous_embedding_jobs`: `region_detection_job_id` (FK), `chunk_size_seconds`, `chunk_hop_seconds`, `crnn_checkpoint_sha256`, `crnn_segmentation_model_id` (FK), `projection_kind`, `projection_dim`, `total_regions`, `total_chunks`
- [ ] Adds nullable columns to `hmm_sequence_jobs`: `training_mode`, `event_core_overlap_threshold` (default 0.5), `near_event_window_seconds` (default 5.0), `event_balanced_proportions` (TEXT/JSON, default `'{"event_core":0.4,"near_event":0.35,"background":0.25}'`), `subsequence_length_chunks` (default 32), `subsequence_stride_chunks` (default 16), `target_train_chunks` (default 200000), `min_region_length_seconds` (default 2.0)
- [ ] Uses `op.batch_alter_table()` for SQLite compatibility
- [ ] `database.py` SQLAlchemy models updated with the same columns and types (nullable)
- [ ] `uv run alembic upgrade head` succeeds against the production DB (after backup)
- [ ] Downgrade step removes all added columns

**Tests needed:**
- `tests/db/test_migration_061.py` — round-trip the migration on a SQLite fixture; assert all new columns exist with correct nullability and defaults; round-trip downgrade.

---

### Task 8: Wire producer into `continuous_embedding_worker`

**Files:**
- Modify: `src/humpback/workers/continuous_embedding_worker.py`
- Modify: `src/humpback/services/continuous_embedding_service.py`

**Acceptance criteria:**
- [ ] Worker entry dispatches on `source_kind` derived from `model_version` family (`surfperch-tensorflow2*` → SurfPerch path; `crnn-call-parsing-*` → CRNN path)
- [ ] Existing SurfPerch path moves into `_run_event_padded_surfperch()` with byte-identical behavior
- [ ] New `_run_region_crnn()` orchestrates: load checkpoint via `crnn_features`, init `ChunkProjection`, iterate Pass 1 regions, skip regions shorter than `min_region_length_seconds`, call `extract_chunk_embeddings`, run `event_overlap_join` against the upstream Pass 2 events parquet, set `is_in_pad`, append rows
- [ ] Atomic write of `embeddings.parquet` via existing pattern; new schema (per spec §5)
- [ ] Atomic write of `manifest.json` summarizing: source_kind, region_detection_job_id, event_segmentation_job_id, crnn_checkpoint_sha256, projection config, chunk geometry, total_regions, total_chunks, vector_dim
- [ ] Job row counters updated: `total_regions`, `total_chunks`, `vector_dim`, `parquet_path`
- [ ] Service `compute_encoding_signature()` extended for CRNN source (per spec §5); SurfPerch signature formula unchanged for existing fixtures
- [ ] `SUPPORTED_MODEL_VERSIONS` extended with the new CRNN family
- [ ] No edits to Pass 2 source files (verified by `git diff`)

**Tests needed:**
- `tests/sequence_models/test_continuous_embedding_worker_region_crnn.py` — end-to-end stub-CRNN run on a 60 s synthetic region with one synthetic Pass 2 event. Assertions: row count matches expected chunk count, schema matches spec §5, atomic write succeeds, job row counters populated, manifest.json contents correct.
- `tests/services/test_continuous_embedding_service.py` extension — encoding signature for CRNN source includes new fields and excludes SurfPerch-only fields; SurfPerch signature unchanged for existing fixtures (regression).

---

### Task 9: Wire consumer into `hmm_sequence_worker`

**Files:**
- Modify: `src/humpback/workers/hmm_sequence_worker.py`

**Acceptance criteria:**
- [ ] Worker reads `training_mode` and tier config off the job row
- [ ] If source is CRNN: call `region_sampling.build_training_set()` to produce sub-sequences and `was_used_for_training` masks; pass sub-sequences with their `lengths` vector to existing `pca_pipeline.fit_pca` and `hmm_trainer.fit_gaussian_hmm`; decode whole regions via existing `hmm_decoder.decode_sequences`
- [ ] If source is SurfPerch: behavior is unchanged
- [ ] `states.parquet` schema for CRNN source includes `region_id`, `chunk_index_in_region`, `tier`, `was_used_for_training` per spec §5
- [ ] `summary.json` for CRNN source includes per-state tier-composition aggregates (`% event_core`, `% near_event`, `% background` per state) computed from `states.parquet`
- [ ] Atomic write of all artifacts (states.parquet, transition_matrix.npy, summary.json, hmm.joblib, pca.joblib)

**Tests needed:**
- `tests/sequence_models/test_hmm_sequence_worker_region_crnn.py` — fixture CRNN-source `embeddings.parquet` (synthetic) → run worker → assert `was_used_for_training` matches the chosen mode, decode covers all chunks (including in-pad and not-trained), `summary.json` per-state tier-composition aggregates sum to 100% per state.
- Existing SurfPerch HMM tests must pass unchanged (regression).

---

### Task 10: API router + Pydantic schemas

**Files:**
- Modify: `src/humpback/api/routers/sequence_models.py`
- Modify: `src/humpback/schemas/sequence_models.py`

**Acceptance criteria:**
- [ ] `POST /api/sequence-models/continuous-embeddings` accepts new fields: `region_detection_job_id`, `event_segmentation_job_id` (when CRNN), `crnn_segmentation_model_id`, `chunk_size_seconds`, `chunk_hop_seconds`, `projection_kind`, `projection_dim`
- [ ] Pydantic validator enforces XOR: exactly one of `event_segmentation_job_id` (alone, SurfPerch) XOR `region_detection_job_id` is the *primary* source — when `region_detection_job_id` is set, `event_segmentation_job_id` becomes a *required disambiguator* but does not change the source kind
- [ ] Service-layer validation rejects: missing/incomplete `RegionDetectionJob`; missing/incomplete `EventSegmentationJob`; mismatched parent (Pass 2 job's parent ≠ submitted Pass 1 job)
- [ ] `POST /api/sequence-models/hmm-sequence` accepts new fields: `training_mode`, `event_core_overlap_threshold`, `near_event_window_seconds`, `event_balanced_proportions`, `subsequence_length_chunks`, `subsequence_stride_chunks`, `target_train_chunks`, `min_region_length_seconds`
- [ ] Validator returns 422 when these fields are sent on a SurfPerch-source HMM job
- [ ] Validator returns 422 when `event_balanced_proportions` does not sum to 1.0 ± 1e-6
- [ ] Response payloads carry the new fields via the extended response models
- [ ] Idempotency on duplicate signature still works for CRNN-source requests

**Tests needed:**
- `tests/api/test_sequence_models_router_region_crnn.py` — happy-path `POST /continuous-embeddings` with CRNN fields succeeds; XOR validation rejects requests with both source IDs; missing disambiguator returns 422; happy-path `POST /hmm-sequence` with training mode succeeds for CRNN source; same fields on SurfPerch source return 422; non-summing proportions return 422; idempotency: two identical CRNN requests return the same job id.

---

### Task 11: Frontend new-job creation forms (source-type toggle)

**Files:**
- Modify: `frontend/src/components/sequence-models/ContinuousEmbeddingNewPage.tsx`
- Modify: `frontend/src/components/sequence-models/HMMSequenceNewPage.tsx`
- Modify: `frontend/src/api/sequenceModels.ts`

**Acceptance criteria:**
- [ ] Continuous Embedding new-job form shows a top-of-form source-type toggle: `Event-padded (SurfPerch 1 s · 5 s window)` vs `Detection-region (CRNN 250 ms chunks)`
- [ ] Toggle swaps which fields render below it; SurfPerch fields unchanged from current; CRNN fields include `region_detection_job_id` (segmentation_models picker for `crnn_segmentation_model_id`), `event_segmentation_job_id` disambiguator, `chunk_size_seconds`, `chunk_hop_seconds`, `projection_kind` (default `identity`), `projection_dim`
- [ ] HMM new-job form shows training-mode select + collapsed Advanced panel for tier config (all defaults locked per spec §8) only when the chosen source job is CRNN
- [ ] Form-level validation: proportions sum to 1.0 ± 1e-6; submit disabled otherwise
- [ ] `sequenceModels.ts` extended with new request/response types

**Tests needed:**
- `frontend/e2e/sequence_models_region_crnn.spec.ts` — Playwright: source-type toggle swaps fields; submitting a CRNN-source job round-trips the API; HMM new-job form shows training-mode block only for CRNN sources; proportion validation blocks submit when sum != 1.0.

---

### Task 12: Frontend list page badge

**Files:**
- Modify: `frontend/src/components/sequence-models/ContinuousEmbeddingListPage.tsx`

**Acceptance criteria:**
- [ ] List page renders a source badge column (`SurfPerch` / `CRNN`)
- [ ] CRNN rows show `total_regions` instead of `total_events` in the counter column
- [ ] Existing rows for SurfPerch jobs continue to render correctly (regression)

**Tests needed:**
- Playwright assertion in `sequence_models_region_crnn.spec.ts` that the badge column appears for both source kinds.

---

### Task 13: Frontend HMM detail page (`SequenceNavigator` + tier strip)

**Files:**
- Modify: `frontend/src/components/sequence-models/HMMSequenceDetailPage.tsx`
- Create or rename: `frontend/src/components/sequence-models/SequenceNavigator.tsx`

**Acceptance criteria:**
- [ ] Existing span-navigator generalized into `SequenceNavigator` accepting a `label` prop (`Span` for SurfPerch, `Region` for CRNN) and a list of `{id, start, end, audio_source}` items
- [ ] Existing keyboard shortcuts (A/D for prev/next per ADR-056) preserved
- [ ] Header badge displays source-kind + key configs: `CRNN · event_balanced · n_states=12 · pca=32` for CRNN; existing format for SurfPerch
- [ ] Per-state tier-composition stacked-bar strip rendered for CRNN-source jobs only, reading from `summary.json` aggregates
- [ ] State timeline, transition heatmap, dwell histogram, exemplars, PCEN spectrogram + state bar all render unchanged for both source kinds (regression)

**Tests needed:**
- Playwright assertion in `sequence_models_region_crnn.spec.ts` that the HMM detail page for a CRNN job shows the tier-composition strip and a `Region` label on the navigator; existing SurfPerch HMM detail e2e tests pass unchanged.

---

### Task 14: ADR-057 + reference doc updates

**Files:**
- Modify: `DECISIONS.md`
- Modify: `CLAUDE.md`
- Modify: `docs/reference/sequence-models-api.md`
- Modify: `docs/reference/data-model.md`
- Modify: `docs/reference/storage-layout.md`
- Modify: `README.md`

**Acceptance criteria:**
- [ ] `DECISIONS.md`: append `## ADR-057: CRNN region-based chunk embeddings as second Sequence Models source` covering decisions 1, 2, 3 from spec §9 (single-table dispatch, concat-and-project with IdentityProjection default, shared windowing helper)
- [ ] `CLAUDE.md` §9.1 — extend Sequence Models track entry to mention CRNN region-based embedding source and three training modes
- [ ] `CLAUDE.md` §9.2 — bump latest migration to `061_crnn_region_embeddings.py`
- [ ] `docs/reference/sequence-models-api.md` — document new request/response fields, XOR validation, training-mode rules
- [ ] `docs/reference/data-model.md` — extend `continuous_embedding_jobs` and `hmm_sequence_jobs` field listings
- [ ] `docs/reference/storage-layout.md` — extend with the new parquet schema variant
- [ ] `README.md` — surface the new capability in the user-facing feature list

**Tests needed:**
- (none — documentation only)

---

### Task 15: Stub fixture and golden parquet for tests

**Files:**
- Create: `tests/call_parsing/fixtures/pass2_golden_events.parquet` (committed binary)
- Create or modify: `tests/sequence_models/conftest.py` — shared stub-CRNN fixture and synthetic-region fixtures

**Acceptance criteria:**
- [ ] Stub `SegmentationCRNN` checkpoint with deterministic weights committed under `tests/sequence_models/fixtures/`
- [ ] Stub checkpoint validates against the load-time guards (BiGRU width 64, frame_rate 32 fps)
- [ ] Synthetic-region fixture: 60 s region with one 5 s event in the middle, fixed audio waveform
- [ ] Pass 2 golden events parquet generated by running Pass 2 against the synthetic-region fixture once and committing the output

**Tests needed:**
- (these fixtures power Tasks 2, 4, 5, 6, 8, 9 tests)

---

## Verification

Run in order after all tasks:

1. `uv run ruff format --check src/humpback/sequence_models src/humpback/workers/continuous_embedding_worker.py src/humpback/workers/hmm_sequence_worker.py src/humpback/services/continuous_embedding_service.py src/humpback/api/routers/sequence_models.py src/humpback/schemas/sequence_models.py src/humpback/database.py src/humpback/call_parsing/segmentation/inference.py src/humpback/call_parsing/segmentation/window_iter.py alembic/versions/061_crnn_region_embeddings.py tests/`
2. `uv run ruff check <same files>`
3. `uv run pyright` (full run since `pyproject.toml` is unchanged but `database.py` schema is)
4. `uv run pytest tests/`
5. `cd frontend && npx tsc --noEmit`
6. `cd frontend && npx playwright test`

Doc-update matrix per CLAUDE.md §10.2 fully covered by Task 14.

Backup gate (CLAUDE.md §3.5) covered by Task 7 acceptance criterion #1; **no migration runs without it**.
