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
- [x] `LabelDistribution` typed dict shape is `{n_states: int, total_windows: int, states: dict[str, dict[str, int]]}` — no tier dimension.
- [x] SurfPerch source no longer emits a synthetic `"all"` tier in label-distribution output.
- [x] CRNN source's `extras.tier` on `decoded.parquet` and exemplars is **untouched**.
- [x] Both sources produce the simplified shape via the same loader Protocol.
- [x] The `hydrophone_id → DetectionJob[]` fan-out path is removed from each loader where it served only label coverage; embedding loading paths remain.

**Tests needed:**
- Asserted the Protocol shape via `test_label_distribution_typed_dict_has_no_tier_dimension` + a regression-defensive int-not-dict check; SurfPerch synthetic `"all"` regression is blocked by `test_compute_label_distribution_omits_synthetic_all_tier` (`tests/sequence_models/test_loaders.py`).

---

### Task 6: Submit-endpoint validation, FK storage, and Classify listing endpoint

**Files:**
- Modify: `src/humpback/api/sequence_models.py` (or wherever HMM/MT submit endpoints live — locate during implementation)
- Modify: `src/humpback/services/hmm_sequence_service.py`, `masked_transformer_service.py` (submit-side)
- Modify or create: listing endpoint `GET /api/call-parsing/event-classification-jobs?event_segmentation_job_id={id}&status=completed` — verify if it already exists; if not, add it
- Modify: `tests/api/test_sequence_models_submit.py`

**Acceptance criteria:**
- [x] HMM submit and MT submit endpoints accept optional `event_classification_job_id` (the column is a string FK in this codebase, not int).
- [x] If omitted, server picks the most recent `EventClassificationJob` whose `event_segmentation_job_id` matches the upstream segmentation **and** `status == COMPLETED`. If zero rows, returns 422 with a message naming the segmentation. (FastAPI maps `ValueError` → 422 via the existing `HTTPException` plumbing; the plan's "400" was nominal.)
- [x] If provided, server verifies it's `COMPLETED` **and** its `event_segmentation_job_id` matches the upstream segmentation; otherwise 422.
- [x] On success, the FK is stored on the new column.
- [x] Listing endpoint added at `GET /call-parsing/classification-jobs/by-segmentation?event_segmentation_job_id={id}&status={status}` with `[{id, created_at, model_name, n_events_classified, status}, ...]` newest first, joined with `vocalization_models.name`.

**Tests needed:**
- Submit-time validation tests + listing tests in `tests/integration/test_sequence_models_submit.py` (HMM + MT mirror set).

---

### Task 7: Regenerate endpoints + atomic re-bind

**Files:**
- Modify: `src/humpback/api/sequence_models.py`
- Modify: `src/humpback/services/hmm_sequence_service.py`, `masked_transformer_service.py`
- Create: `tests/api/test_sequence_models_regenerate.py`

**Acceptance criteria:**
- [x] `POST /sequence-models/hmm-sequences/{id}/regenerate-label-distribution` accepts optional `event_classification_job_id` body field; returns `{status, job_id, event_classification_job_id, label_distribution}`.
- [x] `POST /sequence-models/masked-transformers/{id}/regenerate-label-distribution?k={k}` accepts the same body; rebuilds all `k<N>/label_distribution.json` files in one call (effective events loaded once); response payload returns the active `k`'s payload.
- [x] Synchronous handlers; no queueing.
- [x] Step ordering: validate → write artifacts via per-file temp-then-rename → commit FK update. Failure during step 2 leaves the FK and existing files untouched (in-memory FK swap is reverted on exception).
- [x] Re-bind validation: new Classify job's `event_segmentation_job_id` must match the HMM/MT job's upstream segmentation; otherwise `400` with no FK change and no artifact change.
- [x] No `event_classification_job_id` in the body → use the existing bound FK.

**Tests needed:**
- `tests/integration/test_sequence_models_regenerate.py`: rebuild, re-bind, mismatched-rebind 400, atomic-on-failure (service-level), multi-k regenerate loads events once.

---

### Task 8: Frontend submit forms + API client

**Files:**
- Modify: `frontend/src/api/sequence-models.ts` (or equivalent — locate during implementation)
- Modify: `frontend/src/components/sequence-models/HMMSequenceCreatePage.tsx`
- Modify: `frontend/src/components/sequence-models/MaskedTransformerCreatePage.tsx`

**Acceptance criteria:**
- [x] New API client `listEventClassificationJobsForSegmentation` + `useEventClassificationJobsForSegmentation` hook (TanStack Query key `["event-classification-jobs", segmentationJobId]`).
- [x] HMM and MT create pages render the "Event Classification Job" select directly under the source select; defaults to newest, label format `#{id8} · {model_name} · {n_events_classified} events`.
- [x] Empty list disables the select and the submit button; helper text *"Run Pass 3 Classify on this segmentation first"* renders below.
- [x] Changing the source CEJ flips the segmentation id, which re-keys the Classify query (auto-refetch).
- [x] Submit POSTs include `event_classification_job_id`.
- [x] TypeScript types updated; `cd frontend && npx tsc --noEmit` is clean.

**Tests needed:**
- Playwright spec `frontend/e2e/sequence-models/classify-binding.spec.ts` covers populated → defaults to newest + submit posts FK, and empty → submit disabled + helper text (HMM + MT).

---

### Task 9: Frontend detail pages — chart simplification, Regenerate button, exemplar chips

**Files:**
- Modify: `frontend/src/components/sequence-models/HMMSequenceDetailPage.tsx`
- Modify: `frontend/src/components/sequence-models/MaskedTransformerDetailPage.tsx`
- Modify: `frontend/src/components/sequence-models/LabelDistributionChart.tsx` (or wherever the chart lives)
- Modify: `frontend/src/components/sequence-models/ExemplarCard.tsx` (or equivalent)
- Modify: `frontend/src/api/sequence-models.ts`

**Acceptance criteria:**
- [x] `LabelDistributionChart` reads `states[i]` directly as `Record<string, number>`; the tier-collapse `useMemo` is removed.
- [x] `(background)` renders with a reserved neutral-gray color slot and is positioned last in the legend.
- [x] HMM and MT detail pages render a bound-Classify badge linking to Classify Review for the bound job.
- [x] HMM and MT detail pages render a "Regenerate label distribution" button + dialog. Dialog defaults the select to the currently bound Classify; confirm POSTs to the regenerate endpoint.
- [x] In-flight button is disabled with "Regenerating…" label; on success TanStack Query invalidates the relevant cache keys; on error the dialog surfaces the server message and nothing changes.
- [x] After successful re-bind the bound-Classify badge updates (because invalidation refetches the job detail).
- [x] Exemplar cards render `extras.event_types` chips that link to Classify Review filtered to `extras.event_id`. Background exemplars show a single neutral `(background)` chip with no link.
- [x] CRNN's `extras.tier` badge stays where it is; the new types row sits below it.
- [x] MT regenerate dialog title: *"Regenerate label distribution for all k values"*.
- [x] New API client functions: `regenerateHMMLabelDistribution`, `regenerateMTLabelDistribution`, plus `useRegenerateHMMLabelDistribution` / `useRegenerateMTLabelDistribution` hooks.
- [x] `cd frontend && npx tsc --noEmit` is clean.

**Tests needed:**
- Frontend regenerate-button + chart re-render coverage is deferred to manual smoke (spec §8.7); `tsc --noEmit` clean. The Playwright suite ships the submit-time cases (Task 11) — adding a full regenerate-flow Playwright test requires a wired-up live spectrogram fixture and is out of scope for this PR.

---

### Task 10: Wipe existing on-disk Sequence Models artifacts

**Files:**
- (Operational — no source files modified.)

**Acceptance criteria (state on `feature/sequence-models-classify-label-source`):**
- [x] Storage: `data/hmm_sequences/` and `data/masked_transformer_jobs/` are empty on disk.
- [x] SQL: `hmm_sequence_jobs` and `masked_transformer_jobs` are 0 rows.
- [ ] **Deferred to user (auto-mode guardrail)**: 16 orphaned `motif_extraction_jobs` rows + ~26 MB of `motif_extractions/{id}/` artifact dirs reference deleted MT parents and need a manual cleanup pass before the smoke test. Recommended one-liner: `sqlite3 "$DB" "DELETE FROM motif_extraction_jobs;" && rm -rf "$STORAGE/motif_extractions/"*`. The branch's code is correct independent of this cleanup; the pending rows would simply 404 in the UI when their parent MT job is missing.
- [ ] Manual smoke per spec §8.7 once the user has done the cleanup.

**Tests needed:**
- Manual smoke per Task 12 / spec §8.7. No automated test for the wipe itself.

---

### Task 11: Playwright tests

**Files:**
- Create: `frontend/tests/sequence-models-classify-binding.spec.ts`

**Acceptance criteria:**
- [x] HMM create page: populated dropdown defaults to most recent and enables submit; empty dropdown disables submit and shows helper text.
- [x] MT create page: same coverage (mirrored case).
- [ ] HMM/MT detail-page regenerate flow + chart re-render — deferred to manual smoke (spec §8.7) per Task 9 note.
- [x] Existing legacy-tier label-distribution fixtures in `hmm-sequence.spec.ts` and `masked-transformer.spec.ts` updated to the simplified shape so the existing suite remains green.
- [x] Vocalization Labeling workspace untouched; existing specs unmodified.
- [ ] `cd frontend && npx playwright test` — needs a local server; `tsc --noEmit` is clean which catches API/type drift.

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
- [x] ADR-063 appended to `DECISIONS.md` (supersedes ADR-060; references ADR-054 + ADR-059).
- [x] CLAUDE.md §9.1 Sequence Models bullet updated.
- [x] CLAUDE.md §9.2: latest migration bumped to `066_sequence_models_classify_binding.py`.
- [x] `docs/reference/sequence-models-api.md`: documents both `regenerate-label-distribution` endpoints and the new `event_classification_job_id` field on HMM/MT submit; updated `LabelDistributionResponse` shape; updated exemplar `extras` schema.
- [x] `docs/reference/call-parsing-api.md`: documents the new `classification-jobs/by-segmentation` listing endpoint.
- [x] `docs/reference/data-model.md`: notes the new FK columns on `hmm_sequence_jobs` and `masked_transformer_jobs`.
- [x] `docs/reference/behavioral-constraints.md`: documents the submit-time precondition + manual-regenerate semantics.

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
