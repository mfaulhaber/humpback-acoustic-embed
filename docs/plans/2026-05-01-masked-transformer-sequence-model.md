# Masked-Transformer Sequence Model Implementation Plan

**Goal:** Add a Masked Transformer workflow as a third Sequence Models top-level card, parallel to HMM Sequence, consuming CRNN region-based continuous embeddings and feeding the existing motif-extraction pipeline.

**Spec:** [docs/specs/2026-05-01-masked-transformer-sequence-model-design.md](../specs/2026-05-01-masked-transformer-sequence-model-design.md)

---

### Task 1: Loader normalization on `decoded.parquet`

Generalize the existing per-source HMM loader to read a normalized `decoded.parquet` schema; rename HMM worker output column `state` → `label`. This unblocks both HMM and masked-transformer to share the same loader.

**Files:**
- Modify: `src/humpback/sequence_models/loaders/__init__.py`
- Modify: `src/humpback/sequence_models/loaders/crnn_region.py`
- Modify: `src/humpback/sequence_models/loaders/surfperch.py`
- Modify: `src/humpback/workers/hmm_sequence_worker.py`
- Modify: `src/humpback/services/hmm_sequence_service.py`
- Modify: `src/humpback/api/routers/sequence_models.py` (HMM serialization layer maps `label` back to `viterbi_state` for existing API contract)
- Modify: `tests/sequence_models/test_loaders.py`
- Modify: `tests/workers/test_hmm_sequence_worker.py`
- Modify: any HMM test fixtures that assert on the old `state` column

**Acceptance criteria:**
- [ ] `CrnnRegionLoader.__init__` accepts `decoded_artifact_path: str` parameter
- [ ] `SurfPerchLoader.__init__` accepts the same parameter (parity, even though not exercised at v1)
- [ ] `get_loader(source_kind, decoded_artifact_path)` constructs loader with the path
- [ ] HMM worker writes `decoded.parquet` (renamed from `states.parquet`) with column `label` (renamed from `state`); the rest of the schema unchanged
- [ ] HMM service constructs loader with `<hmm_job_dir>/decoded.parquet`
- [ ] HMM API serialization layer continues to return `viterbi_state` field name in API responses (no frontend change required)
- [ ] All existing HMM tests pass
- [ ] Existing on-disk `states.parquet` from prior HMM jobs is read via a backwards-read shim (look for `decoded.parquet` first, fall back to `states.parquet` with column rename in memory) so existing completed HMM jobs continue to function without manual migration

**Tests needed:**
- Loader test that constructs `CrnnRegionLoader` with an explicit `decoded_artifact_path`, loads from a synthetic per-k path, and returns expected `OverlayInputs` shape
- Loader test that exercises the backwards-read shim against a fixture with the old `states.parquet` schema
- HMM worker test that verifies new `decoded.parquet` is written with column `label`

---

### Task 2: Masked transformer training module

Implement the masked-span transformer trainer (PyTorch), span-mask helper, contextual-embedding extractor, and unit tests against a synthetic dataset.

**Files:**
- Create: `src/humpback/sequence_models/masked_transformer.py`
- Create: `tests/sequence_models/test_masked_transformer.py`

**Acceptance criteria:**
- [ ] `MaskedTransformerConfig` dataclass with all fields from spec §4.1
- [ ] `_PRESETS` dict with `small`, `default`, `large` matching spec values
- [ ] `MaskedTransformer(nn.Module)` with `Linear → TransformerEncoder(norm_first=True, GELU) → Linear` topology; returns `(reconstructed, hidden_states)`
- [ ] `apply_span_mask(seq, frac, span_min, span_max, rng)` returns `(masked_seq, mask_positions)`; coverage in [frac, frac + 0.05]; spans contiguous; masked frames replaced with sequence-mean
- [ ] `train_masked_transformer(sequences, config, device)` returns `TrainResult(model, loss_curve, val_metrics, training_mask, reconstruction_error_per_chunk)`
- [ ] Train loss decreases monotonically (within epsilon) on synthetic sinusoidal sequences
- [ ] Early-stop fires when val plateaus per `early_stop_patience`
- [ ] `mask_weight_bias=True` actually weights event-adjacent positions higher in the loss (verifiable by per-position gradient magnitudes on a fixture with known tier labels)
- [ ] `extract_contextual_embeddings(model, sequences, device)` returns `(Z, lengths)` matching input ordering

**Tests needed:**
- Span mask: fraction, contiguity, span length bounds, determinism with fixed seed
- Forward shape: `(batch, T, input_dim) → reconstructed (batch, T, input_dim) + hidden (batch, T, d_model)`
- Training convergence on synthetic sinusoids; loss curve is non-increasing on average
- Early-stop triggers and reports the correct stopping epoch
- Mask-weight bias path: synthetic per-chunk tier labels result in higher loss-weight values for `event_core` than `background`
- `extract_contextual_embeddings` ordering invariance

---

### Task 3: K-means tokenization module

Implement the per-k k-means tokenizer with softmax-temperature confidence and run-length computation.

**Files:**
- Create: `src/humpback/sequence_models/tokenization.py`
- Create: `tests/sequence_models/test_tokenization.py`

**Acceptance criteria:**
- [ ] `fit_kmeans_token_model(Z, k, seed)` returns `(KMeans, tau)` with `tau > 0` equal to the median pairwise centroid distance
- [ ] `decode_tokens(Z, kmeans, tau)` returns `(labels, confidences)`; confidences are valid probabilities (max in [0,1]); labels match nearest-centroid assignment
- [ ] `compute_run_lengths(token_sequences, k)` returns `dict[str, list[int]]` keyed by token-index strings; handles empty sequences, single-token runs, and boundary chunks correctly

**Tests needed:**
- τ scales with synthetic centroid spread (multiply input by 10 → τ multiplies by 10)
- Confidences sum-bound and max-bound; deterministic with fixed seed
- Nearest-centroid agreement against `KMeans.predict`
- Run-length cases: empty, all-same-token, alternating, short runs at sequence boundaries

---

### Task 4: Database migration — `masked_transformer_jobs` table

Create the new table and the SQLAlchemy model.

**Files:**
- Create: `alembic/versions/063_masked_transformer_jobs.py`
- Modify: `src/humpback/database.py` (add `MaskedTransformerJob` model)
- Modify: `src/humpback/models/sequence_models.py` (extend if model lives there per existing conventions)

**Acceptance criteria:**
- [ ] **Production DB backup taken FIRST per CLAUDE.md §3.5:** read `HUMPBACK_DATABASE_URL` from `.env`; run `cp "$DB_PATH" "${DB_PATH}.YYYY-MM-DD-HH:mm.bak"` with UTC timestamp; verify backup exists with non-zero size BEFORE running `alembic upgrade head`. If the backup step fails or is skipped, stop.
- [ ] Migration `063_masked_transformer_jobs.py` creates the table with all columns from spec §5.1
- [ ] Uses `op.batch_alter_table()` semantics where applicable (SQLite compatibility)
- [ ] `training_signature` has a unique index
- [ ] FK to `continuous_embedding_jobs.id` with appropriate ON DELETE behavior matching the HMM table
- [ ] `uv run alembic upgrade head` applies cleanly against the production DB after backup
- [ ] `MaskedTransformerJob` SQLAlchemy model matches the schema with UTC timestamp defaults

**Tests needed:**
- Round-trip create + query of a `MaskedTransformerJob` row through the SQLAlchemy session, asserting all columns persist correctly
- Migration upgrade + downgrade against an in-memory SQLite DB

---

### Task 5: Masked-transformer service + training-signature idempotency

Implement the service layer with create / list / get / cancel / delete / extend-k-sweep / generate-interpretations.

**Files:**
- Create: `src/humpback/services/masked_transformer_service.py`
- Modify: `src/humpback/storage.py` (helpers for `masked_transformer_jobs/<id>/k<N>/` paths)
- Create: `tests/services/test_masked_transformer_service.py`

**Acceptance criteria:**
- [ ] `create_masked_transformer_job(payload)` validates `continuous_embedding_jobs.source_kind == "region_crnn"` and `status == "completed"`, otherwise raises a typed validation error
- [ ] `training_signature` computed from the spec §4.3 field set (excludes `k_values`)
- [ ] Idempotent: same signature returns the existing job (any status); does not duplicate
- [ ] `extend_k_sweep_job(job_id, additional_k)` only valid for `status="completed"`; appends k values not already present; requeues a follow-up worker pass; does not retrain transformer
- [ ] `cancel_masked_transformer_job` and `delete_masked_transformer_job` mirror HMM lifecycle behavior
- [ ] `generate_interpretations(job_id, k)` calls `compute_overlay`, `select_exemplars`, `compute_label_distribution` from existing modules unchanged

**Tests needed:**
- Create with non-CRNN upstream → validation error
- Create with non-completed upstream → validation error
- Idempotency: two creates with identical config → same job_id
- Extend-k-sweep on running/queued job → validation error
- Extend-k-sweep dedupes existing k values
- Cancel transitions status appropriately

---

### Task 6: Masked-transformer worker + atomic per-k writes + device validation

Implement the worker that trains the transformer, extracts Z, fits per-k tokenizers, decodes tokens, and triggers interpretation generation.

**Files:**
- Create: `src/humpback/workers/masked_transformer_worker.py`
- Modify: worker dispatcher / queue claim layer (wherever HMM worker is registered) to register the new worker
- Create: `tests/workers/test_masked_transformer_worker.py`
- Modify: `tests/conftest.py` (add tiny synthetic CRNN embedding fixture builder if not already present)

**Acceptance criteria:**
- [ ] Worker loads upstream `continuous_embedding_jobs` parquet via existing helpers
- [ ] Device validation: forward+backward on a fixed synthetic batch on CPU + chosen accelerator; tolerance check; persists `chosen_device` + `fallback_reason` on the job row
- [ ] On accelerator validation failure: falls back to CPU, records reason, training proceeds
- [ ] Trains transformer; persists `transformer.pt`, `loss_curve.json`, `reconstruction_error.parquet`
- [ ] Extracts Z; persists `contextual_embeddings.parquet`
- [ ] For each k in `k_values`: fits k-means, decodes tokens, writes `k<N>/decoded.parquet` (schema `(sequence_id, position, label, confidence)`), `k<N>/kmeans.joblib`, `k<N>/run_lengths.json`, then triggers `generate_interpretations(job_id, k)`
- [ ] Per-k atomic writes: stage to `k<N>.tmp/`, rename to `k<N>/` only after all per-k artifacts are written
- [ ] Failure mid-write to `k<N>.tmp/` leaves no half-written `k<N>/`
- [ ] Extend-k-sweep follow-up: only new k values are processed; transformer + Z untouched; existing k subdirs untouched
- [ ] Status transitions queued → running → completed (or failed with reason)

**Tests needed:**
- End-to-end on tiny synthetic fixture: trains, persists artifacts in correct dir layout, status flows correctly
- Atomic write semantics: simulate failure mid-write to `k100.tmp/`, verify no `k100/`
- Device fallback: mock MPS validation to fail, verify CPU fallback + `fallback_reason` set
- Idempotency: same `training_signature` → returns existing job_id, no retraining
- Extend-k-sweep: trained transformer file mtime unchanged, only new `k<N>/` written

---

### Task 7: Database migration — `motif_extraction_jobs` parent generalization

Generalize `motif_extraction_jobs` to point at either an HMM job or a masked-transformer job.

**Files:**
- Create: `alembic/versions/064_motif_extraction_jobs_generalize_parent.py`
- Modify: `src/humpback/database.py` (extend `MotifExtractionJob` model)
- Modify: `src/humpback/models/sequence_models.py` (if applicable)

**Acceptance criteria:**
- [ ] **Production DB backup taken FIRST per CLAUDE.md §3.5:** read `HUMPBACK_DATABASE_URL` from `.env`; run `cp "$DB_PATH" "${DB_PATH}.YYYY-MM-DD-HH:mm.bak"` with UTC timestamp; verify backup exists with non-zero size BEFORE running `alembic upgrade head`. If the backup step fails or is skipped, stop.
- [ ] Migration uses `op.batch_alter_table()` for SQLite compatibility
- [ ] Adds `parent_kind` (text, NOT NULL) with default `"hmm"` for backfill of existing rows
- [ ] Adds `masked_transformer_job_id` (nullable FK to `masked_transformer_jobs.id`)
- [ ] Adds `k` (nullable int)
- [ ] Makes `hmm_sequence_job_id` nullable
- [ ] CHECK constraint enforces XOR between parent FKs and consistency with `parent_kind`; `k IS NOT NULL` iff `parent_kind = "masked_transformer"`
- [ ] Existing rows backfilled with `parent_kind = "hmm"`, `masked_transformer_job_id NULL`, `k NULL`
- [ ] `uv run alembic upgrade head` applies cleanly against the production DB after backup
- [ ] `MotifExtractionJob` SQLAlchemy model reflects the new columns

**Tests needed:**
- Migration upgrade + downgrade against an in-memory SQLite DB
- Backfill correctness: existing HMM-parent rows still queryable with parent_kind="hmm"
- CHECK constraint rejects rows with both parent FKs set, neither set, or `k` set when parent_kind="hmm"

---

### Task 8: Motif extraction service + worker generalization

Wire the motif extraction service and worker to dispatch on `parent_kind`.

**Files:**
- Modify: `src/humpback/services/motif_extraction_service.py`
- Modify: `src/humpback/workers/motif_extraction_worker.py`
- Modify: `src/humpback/sequence_models/motifs.py` (if `config_signature` lives there)
- Modify: `tests/services/test_motif_extraction_service.py` (or create if absent)
- Modify: `tests/workers/test_motif_extraction_worker.py`
- Modify: `tests/sequence_models/test_motifs.py`

**Acceptance criteria:**
- [ ] `MotifExtractionConfig.config_signature()` includes `parent_kind`, parent FK, and `k`
- [ ] `create_motif_extraction_job()` validates upstream based on `parent_kind`: HMM job must be completed; masked-transformer job must be completed AND `k` must be in the job's `k_values`
- [ ] Worker constructs the loader with the appropriate `decoded_artifact_path`: HMM → `<hmm_job_dir>/decoded.parquet`; masked-transformer → `<mt_job_dir>/k<N>/decoded.parquet`
- [ ] `extract_motifs()` algorithm runs unchanged for both parents; tier-aware weighting still applies via the loader's tier metadata
- [ ] Existing HMM-parent motif tests pass unchanged (regression)
- [ ] Idempotency on `config_signature` covers both parent kinds

**Tests needed:**
- Create with `parent_kind="masked_transformer"` + valid masked-transformer job + valid k → success
- Create with `parent_kind="masked_transformer"` + k not in `k_values` → validation error
- Create with `parent_kind="masked_transformer"` + missing `k` → validation error
- Create with `parent_kind="hmm"` + `k` set → validation error (Pydantic XOR)
- Worker with masked-transformer parent reads from per-k decoded.parquet correctly
- HMM-parent regression: existing motif worker behavior preserved

---

### Task 9: API routes + Pydantic schemas

Add the masked-transformer endpoints and extend motif-extraction schemas.

**Files:**
- Modify: `src/humpback/api/routers/sequence_models.py`
- Modify: `src/humpback/schemas/sequence_models.py`
- Create: `tests/integration/test_masked_transformer_api.py`
- Modify: `tests/integration/test_motif_extraction_api.py`

**Acceptance criteria:**
- [ ] All routes from spec §6 implemented under `/sequence-models/masked-transformers/`
- [ ] `MaskedTransformerJobCreate` validates `continuous_embedding_job_id`; defaults `preset="default"`, `k_values=[100]`; advanced overrides optional
- [ ] `MaskedTransformerJobOut`, `MaskedTransformerJobDetail`, `LossCurveResponse`, `ReconstructionErrorResponse`, `ExtendKSweepRequest` defined per spec §6
- [ ] Reuse `OverlayResponse`, `ExemplarsResponse`, `LabelDistribution`, `StateTierComposition` for per-k endpoints
- [ ] Per-k endpoints: `k` query parameter; default = first entry of `k_values`; 404 on unknown k
- [ ] `generate-interpretations` body `{ k_values: list[int] | null }`; null = all configured k
- [ ] `extend-k-sweep` body `{ additional_k: list[int] }`; only valid for `status="completed"`; dedupes
- [ ] `MotifExtractionJobCreate` gains `parent_kind`, `masked_transformer_job_id`, `k` with XOR validator at the Pydantic level

**Tests needed:**
- Happy path: POST + GET list + GET detail
- Validation errors: non-CRNN upstream rejected; k_values must be non-empty list of ints ≥ 2; preset must be one of small/default/large
- Per-k endpoints with valid + invalid k
- Extend-k-sweep on non-completed job → 4xx
- Cancel + delete
- Motif POST with parent_kind="masked_transformer": happy path + XOR violations rejected

---

### Task 10: Frontend — `DiscreteSequenceBar` generalization + `RegionNavBar` extraction

Refactor existing components for shared use; verify HMM detail page continues to work.

**Files:**
- Create: `frontend/src/components/sequence-models/DiscreteSequenceBar.tsx`
- Modify: `frontend/src/components/sequence-models/HMMStateBar.tsx` (delete or thin re-export)
- Modify: `frontend/src/components/sequence-models/HMMSequenceDetailPage.tsx` (swap import; extract inline region nav)
- Create: `frontend/src/components/sequence-models/RegionNavBar.tsx`
- Modify: `frontend/src/components/sequence-models/constants.ts` (rename `STATE_COLORS` → `LABEL_COLORS`; add `labelColor(idx, total)` helper with HSL ramp above ~30 labels)
- Modify: any tests referring to `HMMStateBar` or `STATE_COLORS`

**Acceptance criteria:**
- [ ] `DiscreteSequenceBar` props: `items`, `mode: "rows" | "single-row"`, `numLabels`, `colorPalette`, `currentRegion?`, `tooltipFormatter?`
- [ ] `mode="rows"` reproduces current `HMMStateBar` behavior pixel-equivalently (same canvas, tooltip, playhead, drag-pan)
- [ ] `mode="single-row"` renders a 60px-tall canvas with full-height fillRect per chunk
- [ ] Categorical color palette: `palette[idx % palette.length]` for `total ≤ 30`, generated HSL ramp for larger `total`
- [ ] HMM detail page imports + renders correctly via `DiscreteSequenceBar mode="rows"`
- [ ] `RegionNavBar` extracted from inline HMM region nav; A/D shortcut behavior preserved; HMM detail page renders the same nav via the extracted component
- [ ] `STATE_COLORS` renamed to `LABEL_COLORS`; all imports updated
- [ ] Existing HMM Playwright spec passes unchanged

**Tests needed:**
- Component test for `DiscreteSequenceBar`: both modes render expected canvas; tooltip text uses `tooltipFormatter` when provided
- Component test for `RegionNavBar`: A/D shortcuts, region-count display, current-region highlight
- Existing Playwright HMM spec runs green (regression)

---

### Task 11: Frontend — API client + TanStack Query hooks

Add the masked-transformer fetch functions and hooks; extend motif hooks.

**Files:**
- Modify: `frontend/src/api/sequenceModels.ts`

**Acceptance criteria:**
- [ ] TS interfaces for `MaskedTransformerJob`, `MaskedTransformerJobDetail`, `MaskedTransformerJobCreate`, `LossCurveResponse`, `ReconstructionErrorResponse`, `ExtendKSweepRequest`
- [ ] Fetch functions for each endpoint in spec §6
- [ ] TanStack Query hooks: `useMaskedTransformerJobs`, `useMaskedTransformerDetail`, `useMaskedTransformerOverlay`, `useMaskedTransformerExemplars`, `useMaskedTransformerLabelDistribution`, `useMaskedTransformerRunLengths`, `useMaskedTransformerLossCurve`, `useMaskedTransformerReconstructionError`, `useExtendKSweep`, `useGenerateMaskedTransformerInterpretations`
- [ ] Per-k hooks key cache on `(jobId, k)`
- [ ] Existing motif hooks gain `parent_kind` filter parameter; create-form mutation accepts the parent-kind discriminator

**Tests needed:**
- TypeScript compiles cleanly: `cd frontend && npx tsc --noEmit`

---

### Task 12: Frontend — masked-transformer pages

Create the jobs list page, create form, and detail page.

**Files:**
- Create: `frontend/src/components/sequence-models/MaskedTransformerJobsPage.tsx`
- Create: `frontend/src/components/sequence-models/MaskedTransformerCreateForm.tsx`
- Create: `frontend/src/components/sequence-models/MaskedTransformerDetailPage.tsx`
- Create: `frontend/src/components/sequence-models/KPicker.tsx`
- Create: `frontend/src/components/sequence-models/LossCurveChart.tsx`
- Create: `frontend/src/components/sequence-models/TokenRunLengthHistograms.tsx`
- Modify: `frontend/src/App.tsx` (add routes)
- Modify: Sequence Models nav definition (add third card "Masked Transformer")

**Acceptance criteria:**
- [ ] Jobs list page renders job table with status, upstream embedding job, preset, k_values badge, chosen_device badge, row actions (cancel / delete / open detail)
- [ ] Create form: always-visible fields (upstream picker filtered to source_kind=region_crnn + status=completed, preset radio, k_values CSV input default `100`, max_epochs, mask-weight bias toggle); advanced disclosure for the rest per spec §7.2
- [ ] Form validation: k_values must parse as non-empty list of ints ≥ 2; preset radio is required; upstream picker must be selected
- [ ] Detail page renders header, tier-composition strip, KPicker (URL-synced via `useSearchParams("k")`, defaults to first `k_values`), loss curve, timeline block (spectrogram + token strip + token-confidence strip + reconstruction-error strip), token run-length histograms, overlay scatter, exemplar gallery, label distribution, motif extraction panel
- [ ] KPicker switches the `?k=` URL param without remounting the page; per-k panels react to the change via TanStack Query keys
- [ ] Motif extraction panel pre-fills `parent_kind="masked_transformer"` and the current k
- [ ] Spectrogram + timeline strips zoom/pan together via `TimelineProvider`
- [ ] Sequence Models nav shows three cards: Continuous Embedding, HMM Sequence, Masked Transformer
- [ ] `cd frontend && npx tsc --noEmit` passes

**Tests needed:**
- Component tests for `KPicker` (URL sync), `LossCurveChart` (renders trace), `TokenRunLengthHistograms` (per-token grid)
- Integration smoke (component-level): detail page mounts with mocked API responses and renders all sections without error

---

### Task 13: Frontend — Playwright E2E

End-to-end browser test for the masked-transformer workflow.

**Files:**
- Create: `frontend/e2e/sequence-models/masked-transformer.spec.ts`

**Acceptance criteria:**
- [ ] Job-create form spec: preset selection, k_values CSV parsing, advanced disclosure, validation error messages
- [ ] Job table spec: status badges, k_values chip, device badge, row actions visible
- [ ] Detail page spec: k-picker switches URL + reloads per-k panels; loss curve renders; timeline strips render; exemplar gallery renders; label distribution renders; motif panel pre-fills parent_kind + k
- [ ] Existing HMM Playwright spec continues to pass (regression)
- [ ] Spec uses the same model-stub strategy and fixture conventions as existing sequence-models specs

**Tests needed:**
- The spec itself; runs via `cd frontend && npx playwright test e2e/sequence-models/masked-transformer.spec.ts`

---

### Task 14: ADR-061 + documentation updates

Capture the architectural decisions and update reference docs.

**Files:**
- Modify: `DECISIONS.md` (append ADR-061)
- Modify: `CLAUDE.md` (§9.1 capability list, §9.2 latest migration + tables)
- Modify: `docs/reference/sequence-models-api.md`
- Modify: `docs/reference/data-model.md`
- Modify: `docs/reference/storage-layout.md`
- Modify: `docs/reference/frontend.md`
- Modify: `docs/reference/behavioral-constraints.md`
- Modify: `README.md` (feature list under Sequence Models)

**Acceptance criteria:**
- [ ] ADR-061 captures: source-agnostic loader normalization on `decoded.parquet`, per-k artifact fan-out, generalized `motif_extraction_jobs`, training-signature idempotency + extend-k-sweep, mask-weight bias, softmax-temperature confidence, MPS/CUDA training with synthetic-batch validation + CPU fallback, `DiscreteSequenceBar` generalization
- [ ] CLAUDE.md §9.1 mentions the new workflow and reuse of CRNN region-based embeddings
- [ ] CLAUDE.md §9.2 lists `masked_transformer_jobs`, notes `motif_extraction_jobs` parent generalization, bumps latest migration to `064_motif_extraction_jobs_generalize_parent.py`
- [ ] `docs/reference/sequence-models-api.md` lists all new endpoints + the generalized motif-extraction parent fields + extend-k-sweep semantics
- [ ] `docs/reference/data-model.md` adds `masked_transformer_jobs`, documents generalized `motif_extraction_jobs`
- [ ] `docs/reference/storage-layout.md` adds the per-k subdir layout
- [ ] `docs/reference/frontend.md` adds Masked Transformer route, `DiscreteSequenceBar` generalization, `RegionNavBar` extraction
- [ ] `docs/reference/behavioral-constraints.md` notes the `decoded.parquet` schema convention as a behavioral contract for future sequence-model consumers; notes training-signature / extend-k-sweep semantics
- [ ] README.md feature list updated

**Tests needed:**
- None (documentation-only)

---

### Verification

Run in order after all tasks are complete:

1. `uv run ruff format --check src/humpback tests scripts`
2. `uv run ruff check src/humpback tests scripts`
3. `uv run pyright src/humpback tests scripts`
4. `uv run pytest tests/`
5. `cd frontend && npx tsc --noEmit`
6. `cd frontend && npx playwright test`

Idempotency / data-integrity sanity checks:

7. Re-creating a masked-transformer job with identical config returns the same job_id (no row inserted)
8. Re-creating a CRNN continuous-embedding job consumed by the workflow remains idempotent on `encoding_signature` (regression)
9. Existing HMM jobs render in the UI unchanged after the `state` → `label` rename and loader generalization
10. Existing motif-extraction HMM-parent jobs render unchanged after the parent generalization
