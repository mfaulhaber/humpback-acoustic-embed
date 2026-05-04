# Sequence Models — Classify Label Source Implementation Plan

**Goal:** Replace `vocalization_labels` with `EventClassificationJob` (+ `VocalizationCorrection` / `EventBoundaryCorrection` overlays) as the only label source feeding HMM Sequence and Masked Transformer label-distribution artifacts and exemplar annotations.

**Spec:** [docs/specs/2026-05-04-sequence-models-classify-label-source-design.md](../specs/2026-05-04-sequence-models-classify-label-source-design.md)

**Branch:** `feature/sequence-models-classify-label-source`

---

### Task 1: Backup production DB and add migration 066

**Files:**
- Create: `alembic/versions/066_sequence_models_classify_binding.py`

**Acceptance criteria:**
- [x] **Production DB backed up before any change.** Read `HUMPBACK_DATABASE_URL` from `.env`, copy to `${DB_PATH}.YYYY-MM-DD-HH:MM.bak` (UTC timestamp), confirm the `.bak` exists with non-zero size. If backup fails or is skipped, stop. (CLAUDE.md §3.5.)
- [x] Migration 066 adds `event_classification_job_id` (String FK to `event_classification_jobs.id`, `ON DELETE RESTRICT`, nullable) to `hmm_sequence_jobs` and `masked_transformer_jobs`, with an index on each.
- [x] Uses `op.batch_alter_table()` for SQLite compatibility, with named FK constraints (SQLite batch mode requires named constraints).
- [x] `down_revision = "065"`; revision identifier is `"066"`. (065 was already taken by the recent `effective_event_identity` migration on main; bumped accordingly.)
- [x] `downgrade()` drops the indexes and columns in reverse order.
- [x] `uv run alembic upgrade head` runs cleanly against the production DB.
- [x] `uv run alembic downgrade -1` followed by `uv run alembic upgrade head` round-trips cleanly (verified locally; production left at `head` after).

**Tests needed:**
- An Alembic upgrade/downgrade round-trip on a temporary copy of the schema, asserting the columns appear/disappear and indexes follow them.

---

### Task 2: Event-scoped label-distribution helper module

**Files:**
- Modify: `src/humpback/sequence_models/label_distribution.py` (replaced existing detection-window-based helper)
- Modify: `tests/sequence_models/test_label_distribution.py` (replaced obsolete pure-function tests)
- Create: `tests/sequence_models/test_load_effective_event_labels.py` (loader integration tests)

**Acceptance criteria:**
- [x] `BACKGROUND_LABEL = "(background)"` exported.
- [x] `EffectiveEventLabels` dataclass (frozen): `event_id: str`, `start_utc: float`, `end_utc: float`, `types: frozenset[str]`, `confidences: dict[str, float]`. (Note: `event_id` is `str` per the existing `Event` schema, not `int` as the spec sketch suggested.)
- [x] `load_effective_event_labels(session, *, event_classification_job_id, storage_root) -> list[EffectiveEventLabels]` returns events sorted by `start_utc`, with type set computed as `(model types where above_threshold) ∪ (user-added/confirmed) − (user-deleted)` via the existing `load_effective_events()` overlay (ADR-054) and `VocalizationCorrection` rows (region-scoped, time-range overlapping).
- [x] Events with empty effective types are returned (with `types = frozenset()`); they are *not* dropped at the loader layer.
- [x] `assign_labels_to_windows(rows, events) -> list[WindowAnnotation]` returns annotations parallel to the input rows. Rows whose center time falls outside every event get `event_id=None, event_types=()`. Rows inside an empty-types event also get the background annotation (chart/exemplar invariant: `event_id` set iff at least one surviving label exists).
- [x] Algorithm is O(n_windows + n_events); single sort + single pass via two-pointer cursor.
- [x] Old detection-window center-time match against `vocalization_labels` is **removed** from this module.

**Tests needed:**
- Three disjoint events + windows straddling them; assert center-time placement, multi-label union into `event_types`, background bucket for outside windows.
- An event with all types user-deleted via `VocalizationCorrection`: its windows go to background.
- Synthetic 10k-window / 1k-event input runs in single-pass time and produces the correct counts.
- Above-threshold model type kept; below-threshold model type filtered; user-added included; user-deleted excluded; user-confirmed below-threshold included.
- `EventBoundaryCorrection` shifts an event start; window membership changes accordingly.

---

### Task 3: Rewrite HMM service interpretation + exemplar annotation

**Files:**
- Modify: `src/humpback/services/hmm_sequence_service.py` (consolidated `generate_interpretations` is now async, label distribution + exemplars + overlay all run in one call; old `generate_label_distribution` removed)
- Modify: `src/humpback/workers/hmm_sequence_worker.py` (await the now-async generator)
- Modify: `src/humpback/api/routers/sequence_models.py` (drop import of removed `generate_label_distribution`; the existing `/label-distribution` lazy-generation endpoint and `/generate-interpretations` regenerate endpoint both delegate to the consolidated function)
- Modify: `tests/services/test_hmm_sequence_service.py` (drop obsolete `generate_label_distribution` tests)

**Acceptance criteria:**
- [x] Consolidated `async generate_interpretations(session, storage_root, job, cej)` reads the bound `event_classification_job_id` from the job row, calls `load_effective_event_labels()` + `assign_labels_to_windows()`, runs `compute_label_distribution()`, and writes `label_distribution.json` + annotated `exemplars.json` + `overlay.parquet`.
- [x] Raises if `job.event_classification_job_id` is `None` (the submit-validation step in Task 6 guarantees non-NULL after job creation).
- [x] Output `label_distribution.json` shape: `{"n_states": int, "total_windows": int, "states": {"<i>": {"<label>": <count>, ...}, ...}}`. **No tier dimension.**
- [x] Per-file atomic temp-then-rename write for both JSON files and the overlay parquet.
- [x] Exemplar selection logic is unchanged; after selection, each exemplar's `extras` dict is annotated with `event_id`, `event_types`, `event_confidence`. Background exemplars get `event_id=None, event_types=[], event_confidence={}`.
- [x] CRNN's `extras.tier` is preserved on `decoded.parquet` and exemplars (untouched by this rewrite — set by the loader, never overwritten).
- [x] The detection-window/`vocalization_labels` SQL fan-out is removed from the service.

**Tests needed:**
- End-to-end against an in-memory fixture with one `EventSegmentationJob`, one `EventClassificationJob`, sample `typed_events.parquet`, synthetic `decoded.parquet`, and an HMM job row bound by FK.
- Assert produced JSON shape (no tier dimension), `(background)` bucket, and multi-label union counting (event with two types contributes `+1` to each label per overlapping window).
- Assert `exemplars.json` carries `event_id`, `event_types`, `event_confidence` for non-background exemplars and the empty-equivalents for background.

---

### Task 4: Rewrite Masked Transformer service `generate_interpretations()` + per-k caching

**Files:**
- Modify: `src/humpback/services/masked_transformer_service.py`

**Acceptance criteria:**
- [x] `generate_interpretations(session, storage_root, job, k, *, events_cache=None)` accepts an optional pre-loaded events cache to amortize the DB + parquet reads across the per-k loop.
- [x] New `generate_interpretations_all_k(session, storage_root, job, k_values)` loads effective events once via `load_effective_event_labels()` and threads the cache through each k call. Returns `{k: label_distribution_payload}`.
- [x] Each `k<N>/label_distribution.json` uses the same simplified shape as Task 3; per-state grouping uses the per-k `label` (k-means token) column.
- [x] Each `k<N>/exemplars.json` is annotated identically to Task 3 (event_id, event_types, event_confidence).
- [x] Atomic per-file temp-then-rename for each k subdir's outputs.
- [x] Detection-window/`vocalization_labels` fan-out removed; raises if `job.event_classification_job_id` is `None`.

**Tests needed:**
- Mirror of Task 3 test for two `k` values, asserting both `k<N>/` outputs have the new shape.
- Assert `load_effective_event_labels()` is called exactly once per service invocation regardless of how many `k_values` are present (verified via call counter on a spy/mock).

---

### Task 5: Loader Protocol simplification (drop tier dimension)

**Files:**
- Modify: `src/humpback/sequence_models/surfperch.py`
- Modify: `src/humpback/sequence_models/crnn_region.py`
- Modify: any shared Protocol/TypedDict declaration referenced by ADR-059 (locate during implementation)
- Modify: `tests/sequence_models/test_loader_protocol.py`

**Acceptance criteria:**
- [ ] `LabelDistribution` typed dict shape is `{n_states: int, total_windows: int, states: dict[str, dict[str, int]]}` — no tier dimension.
- [ ] SurfPerch source no longer emits a synthetic `"all"` tier in label-distribution output.
- [ ] CRNN source's `extras.tier` on `decoded.parquet` and exemplars is **untouched**.
- [ ] Both sources produce the simplified shape via the same loader Protocol.
- [ ] The `hydrophone_id → DetectionJob[]` fan-out path is removed from each loader where it served only label coverage; embedding loading paths remain.

**Tests needed:**
- Assert the Protocol shape via a typed-dict consumer test that fails on tier-dimension reintroduction.
- Assert SurfPerch source label-distribution output does not contain `"all"`.

---

### Task 6: Submit-endpoint validation, FK storage, and Classify listing endpoint

**Files:**
- Modify: `src/humpback/api/sequence_models.py` (or wherever HMM/MT submit endpoints live — locate during implementation)
- Modify: `src/humpback/services/hmm_sequence_service.py`, `masked_transformer_service.py` (submit-side)
- Modify or create: listing endpoint `GET /api/call-parsing/event-classification-jobs?event_segmentation_job_id={id}&status=completed` — verify if it already exists; if not, add it
- Modify: `tests/api/test_sequence_models_submit.py`

**Acceptance criteria:**
- [ ] HMM submit and MT submit endpoints accept optional `event_classification_job_id: int | None`.
- [ ] If omitted, server picks the most recent `EventClassificationJob` whose `event_segmentation_job_id` matches the upstream segmentation **and** `status == COMPLETED`. If zero rows, return 400 with a message naming the segmentation.
- [ ] If provided, server verifies it's `COMPLETED` **and** its `event_segmentation_job_id` matches the upstream segmentation; otherwise 400.
- [ ] On success, the FK is stored on the new column.
- [ ] Listing endpoint returns `[{id, created_at, model_name, n_events_classified}, ...]` newest first, filtered to `status=completed` and the supplied segmentation. If the endpoint already exists with a similar shape, extend rather than duplicate.

**Tests needed:**
- 400 when no Classify exists for the segmentation (no row inserted).
- Default-to-latest picks the newest completed Classify when multiple exist.
- 400 when explicit FK has mismatched segmentation.
- 400 when explicit FK is non-completed (e.g., `RUNNING`).
- Mirrored set for Masked Transformer.

---

### Task 7: Regenerate endpoints + atomic re-bind

**Files:**
- Modify: `src/humpback/api/sequence_models.py`
- Modify: `src/humpback/services/hmm_sequence_service.py`, `masked_transformer_service.py`
- Create: `tests/api/test_sequence_models_regenerate.py`

**Acceptance criteria:**
- [ ] `POST /api/sequence-models/hmm/{id}/regenerate-label-distribution` accepts optional `event_classification_job_id` body field; returns `{label_distribution, exemplars}`.
- [ ] `POST /api/sequence-models/masked-transformer/{id}/regenerate-label-distribution?k={k}` accepts the same body; rebuilds all `k<N>/label_distribution.json` files in one call (effective events loaded once); response payload returns the active `k`'s payload.
- [ ] Synchronous workers; no queueing.
- [ ] Step ordering when re-binding: (1) validate, (2) write all artifact files via per-file temp-then-rename, (3) commit FK update in a single SQL transaction. Failure during step 2 → no FK change, no artifact change. Step 3 failure may leave files paired with the old FK; recovery is by re-running regenerate (documented in spec §6.7).
- [ ] Re-bind validation: new Classify job's `event_segmentation_job_id` must match the HMM/MT job's upstream segmentation; otherwise 400 with no FK change and no artifact change.
- [ ] No `event_classification_job_id` in the body → use the existing bound FK.

**Tests needed:**
- Stale `label_distribution.json` is rebuilt to the expected new content; returned payload matches.
- Re-bind to a new Classify job updates FK and rebuilds artifacts from the new source.
- Mismatched re-bind returns 400 with no FK change and no artifact change.
- Induced write failure during step 2: FK unchanged, prior artifact intact, temp files cleaned.
- Multi-k MT regenerate updates every `k<N>/label_distribution.json`; effective events loaded exactly once per regenerate call.

---

### Task 8: Frontend submit forms + API client

**Files:**
- Modify: `frontend/src/api/sequence-models.ts` (or equivalent — locate during implementation)
- Modify: `frontend/src/components/sequence-models/HMMSequenceCreatePage.tsx`
- Modify: `frontend/src/components/sequence-models/MaskedTransformerCreatePage.tsx`

**Acceptance criteria:**
- [ ] New API client function `listEventClassificationJobsForSegmentation(segmentationJobId)` queries the listing endpoint.
- [ ] HMM and MT create pages render an "Event Classification Job" `<Select>` directly under the existing "Event Segmentation Job" select.
- [ ] Default selection is the first option (newest completed); option label is `#{id} • {model_name} • {n_events_classified} events • {created_at:relative}`.
- [ ] When the list is empty: dropdown is disabled with helper text *"Run Pass 3 Classify on this segmentation first"*; submit button is disabled.
- [ ] Changing the segmentation select clears and refetches the Classify select.
- [ ] TanStack Query key: `["event-classification-jobs", segmentationJobId]`.
- [ ] Submit POSTs include `event_classification_job_id`.
- [ ] TypeScript types updated; `cd frontend && npx tsc --noEmit` is clean.

**Tests needed:**
- Playwright (in Task 11): empty Classify dropdown → submit disabled with helper text; populated → defaults to newest and submit enabled.

---

### Task 9: Frontend detail pages — chart simplification, Regenerate button, exemplar chips

**Files:**
- Modify: `frontend/src/components/sequence-models/HMMSequenceDetailPage.tsx`
- Modify: `frontend/src/components/sequence-models/MaskedTransformerDetailPage.tsx`
- Modify: `frontend/src/components/sequence-models/LabelDistributionChart.tsx` (or wherever the chart lives)
- Modify: `frontend/src/components/sequence-models/ExemplarCard.tsx` (or equivalent)
- Modify: `frontend/src/api/sequence-models.ts`

**Acceptance criteria:**
- [ ] `LabelDistributionChart` reads `states[i]` directly as `Record<string, number>`; the tier-collapse `useMemo` is removed.
- [ ] `(background)` is rendered as a label with a reserved neutral-gray color slot; positioned in legend so it visually deprioritizes against real types.
- [ ] HMM and MT detail pages show a bound-Classify badge in the header strip with text *"Labels from Classify job #{id} ({model_name})"*; click opens Classify Review filtered to that job.
- [ ] HMM and MT detail pages show a "Regenerate label distribution" button in the chart card header. Click opens a dialog with a `<Select>` defaulting to the currently bound Classify job and listing other completed jobs for the same segmentation. Confirm POSTs to the regenerate endpoint with optional `event_classification_job_id`.
- [ ] In flight: button shows spinner + disabled. On success: TanStack Query invalidates `["hmm-job", id]`, `["hmm-label-distribution", id]`, `["hmm-exemplars", id]` (and MT equivalents); toast *"Label distribution regenerated."*. On error: toast with server message; nothing changes on disk.
- [ ] If a different Classify job is picked, the bound-Classify badge updates after refetch.
- [ ] `ExemplarCard` renders `extras.event_types` as small chips below the spectrogram, using the chart legend's color palette. Chips are wrapped in a click-target opening Classify Review filtered to `extras.event_id`.
- [ ] Background exemplars (`event_types: []`) show a single neutral *"(background)"* chip with no link.
- [ ] CRNN's `extras.tier` chip remains where it is; the new types row sits below it.
- [ ] MT regenerate dialog wording: *"Regenerate label distribution for all k values"*.
- [ ] New API client functions added: `regenerateHMMLabelDistribution(jobId, body?)`, `regenerateMTLabelDistribution(jobId, k, body?)`.
- [ ] `cd frontend && npx tsc --noEmit` is clean.

**Tests needed:**
- Playwright (Task 11): chart re-renders after regenerate; exemplar chips appear; background chip on background exemplars; re-bind updates header badge and chart contents; second regenerate without arg keeps the binding.

---

### Task 10: Wipe existing on-disk Sequence Models artifacts

**Files:**
- (Operational — no source files modified.)

**Acceptance criteria:**
- [ ] Confirm SQL rows in `hmm_sequence_jobs`, `masked_transformer_jobs`, `motif_extraction_jobs` are already empty (per user statement during brainstorming).
- [ ] `rm -rf data/sequence_models/hmm_sequence_jobs/* data/sequence_models/masked_transformer_jobs/*` (preserving the parent directories themselves).
- [ ] Verify motif extraction job artifact directory under each parent is also gone.
- [ ] Run a fresh HMM submit + MT submit smoke against a real `EventSegmentationJob` with a completed Classify; both detail pages render with the new chart shape and the new exemplar chips.

**Tests needed:**
- Manual smoke per Task 12 / spec §8.7. No automated test for the wipe itself.

---

### Task 11: Playwright tests

**Files:**
- Create: `frontend/tests/sequence-models-classify-binding.spec.ts`

**Acceptance criteria:**
- [ ] HMM create page: empty Classify dropdown disables submit and shows the helper text; populated dropdown defaults to most recent and enables submit.
- [ ] HMM detail page: Regenerate button click runs to completion; chart re-renders with new bucket counts; exemplar chips appear with type names; `(background)` chip on background exemplars.
- [ ] HMM detail page: Re-bind via dialog updates the header badge text and chart contents; second regenerate without arg keeps the binding.
- [ ] MT detail page: Regenerate works irrespective of active `?k=`; switching `k` after regenerate shows the updated chart for the new k.
- [ ] Vocalization Labeling workspace remains functional and visually unchanged (smoke).
- [ ] `cd frontend && npx playwright test` passes locally.

**Tests needed:**
- (This task IS the tests.)

---

### Task 12: Documentation updates

**Files:**
- Modify: `DECISIONS.md` (append new ADR)
- Modify: `CLAUDE.md` §9.1 and §9.2
- Modify: `docs/reference/sequence-models-api.md`
- Modify: `docs/reference/data-model.md`
- Modify: `docs/reference/behavioral-constraints.md`

**Acceptance criteria:**
- [ ] New ADR appended to `DECISIONS.md`: *"Sequence Models label source switched to Call Parsing Classify"* — explicitly supersedes ADR-060; references ADR-054 (correction overlay) and ADR-059 (loader Protocol).
- [ ] CLAUDE.md §9.1 Sequence Models bullet updated: state-to-label distribution sourced from `EventClassificationJob` + `VocalizationCorrection` overlay; tier dimension removed from label-distribution artifacts (CRNN tier metadata persists on `decoded.parquet`/exemplars).
- [ ] CLAUDE.md §9.2: latest migration bumped to `066_sequence_models_classify_binding.py`.
- [ ] `docs/reference/sequence-models-api.md`: documents the two `regenerate-label-distribution` endpoints and the new `event_classification_job_id` field on submit.
- [ ] `docs/reference/data-model.md`: notes the new FK columns on `hmm_sequence_jobs` and `masked_transformer_jobs`.
- [ ] `docs/reference/behavioral-constraints.md`: documents the submit-time precondition (Classify required) and the manual-regenerate semantics (no auto-recompute on correction writes).

**Tests needed:**
- Doc-only — no automated tests.

---

### Verification

Run in order after all tasks. **Stop at the first failing step** and resolve before proceeding.

1. `uv run ruff format --check src/ tests/`
2. `uv run ruff check src/ tests/`
3. `uv run pyright` (full run — loader Protocol changed)
4. `uv run pytest tests/`
5. `cd frontend && npx tsc --noEmit`
6. `cd frontend && npx playwright test`

After all six pass, perform the manual smoke per spec §8.7:

1. Confirm prod DB backup exists (Task 1).
2. Confirm migration is at `head`.
3. Confirm Sequence Models artifact dirs were wiped (Task 10).
4. Submit one HMM job and one MT job against an `EventSegmentationJob` with completed Classify; verify chart, exemplar chips, regenerate, and re-bind on both detail pages.
5. Open Vocalization Labeling workspace and confirm unchanged.
