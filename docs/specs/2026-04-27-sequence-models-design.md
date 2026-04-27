# Sequence Models — HMM Latent State Discovery on SurfPerch Embeddings

**Status:** Approved (brainstorm complete; per-PR plans to be written separately during `/session-plan`)
**Date:** 2026-04-27
**Source design input:** `/Users/michael/development/perch_hmm_sequence_model_design.md` (external author's design notes)
**Track:** Sequence Models (new top-level track, parallel to Call Parsing)

---

## 1. Goal & Scope

Build a sequence-modeling layer on top of region-bounded, 1-second-hop SurfPerch embeddings to discover temporal structure in humpback non-song vocalization timelines. The latent state sequences and transitions discovered by the model are tools for **research and post-hoc interpretation** — they are not classifications of human-defined call types and are not training data for downstream classifiers.

The system operationalizes the recommended modeling path from the source design notes:

```
Hydrophone audio → RegionDetectionJob (Pass 1)
                          ↓
                ContinuousEmbeddingJob   ← produces ordered 1s-hop embeddings
                          ↓
                 HMMSequenceJob          ← latent states + transitions + dwell + viz
                          ↓
       State timelines · Transition matrices · Motifs · Exemplars
```

### Core principles (from source design)

- Do **not** treat human multi-label window annotations as ground truth states. Human labels are interpretation evidence, not training signal.
- HMM-style models require temporal continuity. Use *full ordered 1-second-hop sequences*, not curated tiled windows.
- Multi-label windows are not a problem to remove; they are weak annotations / mixture evidence used post-hoc.
- Visual inspection and biological interpretability are primary success criteria; BIC/AIC/log-likelihood are secondary diagnostics.

### Non-goals (MVP)

- Multi-source HMM training (1:1 source linkage only).
- Sweep / grid training inside one job.
- Re-decoding existing HMM models against new sources (separate decode job type).
- GPU acceleration; pomegranate; ssm.
- Cross-track coupling beyond consuming Pass 1 region output.
- Uploaded-audio (file-based) input.
- Pass 3 / Window Classification / experimental label sources.
- Real-time streaming; model fine-tuning; multi-tenant.

---

## 2. Architectural Placement

A new top-level **Sequence Models** track in the system, with sub-tabs in the frontend nav: *Continuous Embedding* (PR 1), *HMM Sequence* (PR 2+). Future tabs (HSMM, motif explorer) slot in here.

- **Cross-track dependency**: Region Detection (Pass 1) is the only upstream call-parsing artifact consumed.
- **Independent evolution**: Sequence Models can adopt new embedding model versions, new sequence-model families, without touching the four-pass call parsing pipeline.
- **Reuse, don't fork**: existing infrastructure used as-is — `model_runners` (SurfPerch via `surfperch-tensorflow2` model_version, already used by `lr-detect-v1` and Call Parsing), audio decoder + windowing for hydrophone time ranges, the standard job/worker/queue pattern, Alembic migration conventions, the React job-page pattern with Plotly.

### Conceptual boundaries between the two new job types

- **`ContinuousEmbeddingJob`** is pure data plumbing: given a `region_detection_job_id`, model version, hop, and pad, produce one parquet of windowed embeddings. No modeling. **Idempotent on `encoding_signature`.**
- **`HMMSequenceJob`** is pure modeling: given a `continuous_embedding_job_id` (1:1) and HMM hyperparameters, fit PCA + Gaussian HMM, decode Viterbi states, output state-per-window parquet + diagnostic artifacts. No data acquisition. **Not idempotent** — training is stochastic; comparing configs requires multiple runs.

---

## 3. PR Plan (vertical-slice MVP, then iterate)

The PR sequence retires the highest-uncertainty risk first (does HMM-on-SurfPerch-embeddings actually show coherent state structure?) and matches the source design's "minimal first milestone" guidance.

### PR 1 — Continuous embedding producer

Region-bounded, hydrophone-only, 1-second-hop SurfPerch embeddings padded around detected regions.

- New `continuous_embedding_jobs` SQL table (Alembic `057_continuous_embedding_jobs.py`).
- New schema, service, worker, API router.
- New `processing/region_windowing.py` pure-function module (region merge + window iteration).
- Frontend: Sequence Models nav section + Continuous Embedding sub-tab (list, create, detail).
- Output parquet at `continuous_embeddings/{job_id}/embeddings.parquet` plus `manifest.json`.
- Idempotent on `encoding_signature = sha256(region_detection_job_id, model_version, hop_seconds, window_size_seconds, pad_seconds, target_sample_rate, feature_config)`.
- Defaults: `model_version="surfperch-tensorflow2"`, `hop_seconds=1.0`, `pad_seconds=10.0`, `window_size_seconds=5.0`.

### PR 2 — HMM training + Viterbi decode + minimum-viable viz

Vertical-slice MVP — operator can look at a real run and judge "is this learning anything?".

- New `hmm_sequence_jobs` SQL table (Alembic `058_hmm_sequence_jobs.py`).
- New `sequence_models/` package (`hmm_trainer.py`, `hmm_decoder.py`, `pca_pipeline.py`, `summary.py`).
- Service, worker, API endpoints (`/sequence-models/hmm-sequences/...`).
- HMM library: `hmmlearn` (added to `pyproject.toml`). Pomegranate / GPU deferred to future-work.
- Frontend: HMM Sequence sub-tab with three Plotly charts — state timeline (per merged span), transition matrix heatmap, dwell-time histograms grid.
- Output artifacts under `hmm_sequences/{job_id}/`: `pca_model.joblib`, `hmm_model.joblib`, `states.parquet`, `transition_matrix.npy`, `state_summary.json`, `training_log.json`.

### PR 3 — Interpretation visualizations

Lands after the manual evaluation gate (§9) confirms PR 2 baseline is producing sensible structure.

- PCA / UMAP overlay colored by HMM state (`pca_overlay.parquet`, frontend Plotly scatter).
- State-to-label distribution: vocalization-label join via center-time-in-labeled-window semantics (§5.4), output `label_distribution.json`, frontend per-state stacked-bar chart.
- State exemplars: per-state high-confidence / mean-nearest / boundary-low-confidence picks with audio/spectrogram pointers, output `exemplars/`, frontend gallery.

### PR 4 — Motif mining

- N-gram mining over collapsed-consecutive state sequences.
- Per-motif: count, recording coverage, mean duration, occurrences, associated label distribution.
- Output `motifs.parquet` and `motifs/` directory; frontend motif gallery with sortable / filterable list.

### PR 5+ — Future work (deferred until baseline empirically validated)

- GMM-HMM with mixture exploration.
- HSMM / explicit duration models.
- Cross-source HMM training (N:1 source linkage; train/val/test split *across* corpora).
- Standalone HMM decoding job (apply trained model to a new continuous-embedding source without re-training).
- Sweep / grid training inside one job.
- Pomegranate + MPS path (gated on measured scaling need).
- Future overlay visualization combining HMM states, vocal labels, and playable spectrograms on a single timeline (the union of states + labels + audio is enabled by the shared `(audio_file_id, start_time_sec, end_time_sec)` keying introduced in PR 1).

---

## 4. Architecture & Components

### 4.1 New code locations

```
src/humpback/
├── api/sequence_models.py                  # PR 1+ FastAPI router
├── database.py                             # PR 1+ ORM models added
├── schemas/sequence_models.py              # PR 1+ Pydantic schemas
├── services/
│   ├── continuous_embedding_service.py     # PR 1
│   └── hmm_sequence_service.py             # PR 2
├── workers/
│   ├── continuous_embedding_worker.py      # PR 1
│   └── hmm_sequence_worker.py              # PR 2
├── processing/region_windowing.py          # PR 1 (pure functions)
├── sequence_models/                        # PR 2 package
│   ├── hmm_trainer.py
│   ├── hmm_decoder.py
│   ├── pca_pipeline.py
│   └── summary.py
└── storage.py                              # extend with new dir helpers

alembic/versions/
├── 057_continuous_embedding_jobs.py        # PR 1
└── 058_hmm_sequence_jobs.py                # PR 2

frontend/src/
├── pages/SequenceModels/                   # PR 1 nav root + PR 2 HMM page
├── components/sequence-models/             # PR 1+ shared components
└── api/sequenceModels.ts                   # PR 1+ TanStack Query hooks
```

### 4.2 ContinuousEmbeddingJob runtime flow (PR 1)

1. Worker claims job; verifies `running`.
2. Loads `RegionDetectionJob` and its associated regions, sorted by start_time.
3. Determines hydrophone time range covering `[min(region.start) - pad, max(region.end) + pad]`. Reuses existing hydrophone audio decoder (chunked, streaming-safe, same one used by Pass 1).
4. Computes merged padded spans via `processing/region_windowing.merge_padded_regions`. Persists span manifest.
5. For each merged span:
   - Decodes audio span chunk-by-chunk (memory-bounded).
   - Runs SurfPerch via existing `model_runners` at configured hop / window-size.
   - Builds window rows (`merged_span_id`, `window_index_in_span`, times, `is_in_pad`, `source_region_ids`, `embedding`).
   - Appends rows to a temp parquet writer.
6. Atomically renames temp parquet → `embeddings.parquet`. Writes `manifest.json` similarly.
7. Updates job to `complete` with summary stats; on exception, `failed` with `error_message`.

**Resumability**: worker crash mid-run → next claim re-runs from scratch; temp parquet overwritten; the atomic rename is the only commit point.

### 4.3 HMMSequenceJob runtime flow (PR 2)

1. Worker claims job; verifies source `ContinuousEmbeddingJob.status='complete'` (else fail with clear error).
2. Reads `embeddings.parquet`. Groups rows by `merged_span_id`, sorted by `window_index_in_span` → `sequences: list[np.ndarray[T_i, D]]`.
3. Optional L2 normalization (default true; safe even if upstream embeddings already unit-norm).
4. Fits PCA on **all sequences from the single source** (single-source 1:1 has no internal train/val split; cross-source split deferred to PR 5+ when N:M lands).
5. Filters sequences below `min_sequence_length_frames` for *training* (still decoded later; output rows marked `was_used_for_training=False`).
6. Fits `hmmlearn.GaussianHMM` on the training subset, passing concatenated sequences with `lengths=[T_i, ...]`.
7. Decodes all sequences via Viterbi → state assignments + posteriors per window.
8. Computes summary stats (per-state occupancy, dwell-time histograms, transition matrix).
9. Persists artifacts atomically; updates job to `complete`.

**Determinism**: `random_seed` flows through both PCA (`random_state`) and `hmmlearn.GaussianHMM` (`random_state`).

### 4.4 Cancellation

Both job types support cooperative cancellation. `POST /cancel` flips `queued` → `canceled` directly; for `running` jobs, sets a flag the worker checks at safe points (between regions for the producer; between EM iterations for the trainer). Mirrors the pattern used by existing job types.

---

## 5. Data Model & Storage

### 5.1 `continuous_embedding_jobs` table (PR 1)

| Column | Type | Notes |
|---|---|---|
| `id` | int PK | |
| `status` | enum | queued / running / complete / failed / canceled |
| `region_detection_job_id` | int FK → `region_detection_jobs(id)` | source of regions; not nullable |
| `model_version` | str | default `"surfperch-tensorflow2"` |
| `window_size_seconds` | float | from model config, persisted for clarity |
| `hop_seconds` | float | default 1.0 |
| `pad_seconds` | float | default 10.0 |
| `target_sample_rate` | int | from model config |
| `feature_config_json` | json | feature pipeline config snapshot |
| `encoding_signature` | str (indexed; unique among `status='complete'`) | sha256 idempotency key |
| `vector_dim` | int nullable | filled at run time |
| `total_regions` | int nullable | filled at completion |
| `merged_spans` | int nullable | filled at completion |
| `total_windows` | int nullable | filled at completion |
| `parquet_path` | str nullable | path to output parquet |
| `error_message` | text nullable | failure reason |
| `created_at`, `updated_at` | datetime UTC | |

### 5.2 `hmm_sequence_jobs` table (PR 2)

| Column | Type | Notes |
|---|---|---|
| `id` | int PK | |
| `status` | enum | same enum |
| `continuous_embedding_job_id` | int FK → `continuous_embedding_jobs(id)` | 1:1 source |
| `n_states` | int | HMM state count |
| `pca_dims` | int | PCA reduction target |
| `pca_whiten` | bool | default false |
| `l2_normalize` | bool | default true |
| `covariance_type` | str | `"diag"` / `"full"` — default diag |
| `n_iter` | int | EM iterations |
| `random_seed` | int | reproducibility |
| `min_sequence_length_frames` | int | default 10 |
| `tol` | float | EM convergence tolerance |
| `library` | str | `"hmmlearn"` for MVP |
| `train_log_likelihood` | float nullable | best log-lik from EM |
| `n_train_sequences` | int nullable | count after min-length filter |
| `n_train_frames` | int nullable | total frames seen |
| `n_decoded_sequences` | int nullable | including those below filter |
| `artifact_dir` | str nullable | base path for saved artifacts |
| `error_message` | text nullable | |
| `created_at`, `updated_at` | datetime UTC | |

### 5.3 Parquet artifact schemas

**`continuous_embeddings/{job_id}/embeddings.parquet`** (PR 1):

| Column | Type | Notes |
|---|---|---|
| `merged_span_id` | int32 | groups rows belonging to one contiguous padded sequence |
| `window_index_in_span` | int32 | 0-based position within merged span; defines temporal order |
| `audio_file_id` | int32 nullable | underlying chunk file id (informational) |
| `start_time_sec` | float64 | UTC epoch seconds |
| `end_time_sec` | float64 | UTC epoch seconds |
| `is_in_pad` | bool | true if window center falls outside any source region |
| `source_region_ids` | list<int32> | region ids whose un-padded extent contains this window's center |
| `embedding` | list<float32>[vector_dim] | the SurfPerch embedding |

Sort order: `(merged_span_id, window_index_in_span)`.

A sidecar `continuous_embeddings/{job_id}/manifest.json` records: `vector_dim`, `model_version`, hop/pad/window settings, span count, total windows, span boundaries (for quick stats).

**`hmm_sequences/{job_id}/states.parquet`** (PR 2): all columns from `embeddings.parquet` minus the raw `embedding` column, plus:

| Column | Type | Notes |
|---|---|---|
| `viterbi_state` | int16 | |
| `state_posterior` | list<float32>[n_states] | |
| `max_state_probability` | float32 | argmax(state_posterior) |
| `was_used_for_training` | bool | false when span was below `min_sequence_length_frames` |

Plus sibling artifacts: `pca_model.joblib`, `hmm_model.joblib`, `transition_matrix.npy`, `state_summary.json`, `training_log.json`.

### 5.4 Vocalization-label join (PR 3 only — not in PR 1/PR 2)

For the state-to-label distribution viz, an HMM window inherits labels from any `vocalization_labels` row attached to a `detection_row` whose 5s window contains the HMM window's **center timestamp**, scoped to the same `audio_file_id` (or hydrophone time range). HMM windows in unlabeled regions get `null` labels and contribute to an "unlabeled" bucket per state.

- **Trusted source only**: `vocalization_labels` (Vocalization Labeling workspace), human-curated multi-label.
- **Excluded for MVP**: Pass 3 / Classify predictions (experimental); window-classification predictions (would create a self-referential loop with embedding-derived states).
- **PR 1 / PR 2 unaffected**: producer + HMM core just emit window rows keyed by `(audio_file_id, start_time_sec, end_time_sec)`; the join lives entirely in PR 3.

### 5.5 Storage convention

```
data/
├── continuous_embeddings/
│   └── {continuous_embedding_job_id}/
│       ├── embeddings.parquet
│       └── manifest.json
└── hmm_sequences/
    └── {hmm_sequence_job_id}/
        ├── pca_model.joblib
        ├── hmm_model.joblib
        ├── states.parquet
        ├── transition_matrix.npy
        ├── state_summary.json
        ├── training_log.json
        # PR 3:
        ├── pca_overlay.parquet
        ├── label_distribution.json
        └── exemplars/
        # PR 4:
        ├── motifs.parquet
        └── motifs/
```

Both new directory roots are added to `docs/reference/storage-layout.md` as part of PR 1 and PR 2 doc updates respectively.

### 5.6 Atomic write semantics & resumability

- Parquet/joblib/json files written to temp paths under the same job dir, then `os.rename`'d into place once the run is complete (per CLAUDE.md §4.2).
- Job status transitions to `complete` in a single SQL update *after* all artifact files are atomically in place — readers can rely on `status='complete'` meaning "all artifacts present and final".
- Worker crashes mid-run → next claim re-runs from scratch; temp parquet overwritten; the atomic rename is the only commit point.

### 5.7 Migrations & DB backup

- **PR 1 migration `057_continuous_embedding_jobs.py`** — create table + index on `encoding_signature`. Uses `op.batch_alter_table()` per SQLite convention.
- **PR 2 migration `058_hmm_sequence_jobs.py`** — create table.
- **Pre-migration backup of production DB** is the **first acceptance criterion** in each PR's plan, per CLAUDE.md §3.5. Not a parenthetical.

---

## 6. API Surface

Mounted under `/sequence-models/`:

```
POST   /sequence-models/continuous-embeddings           create job (idempotent)
GET    /sequence-models/continuous-embeddings           list jobs (filter by status)
GET    /sequence-models/continuous-embeddings/{id}      get job detail + manifest stats
POST   /sequence-models/continuous-embeddings/{id}/cancel

POST   /sequence-models/hmm-sequences                   create job
GET    /sequence-models/hmm-sequences                   list jobs (filter by status, source)
GET    /sequence-models/hmm-sequences/{id}              get job detail + summary
GET    /sequence-models/hmm-sequences/{id}/states       paginated states.parquet rows
GET    /sequence-models/hmm-sequences/{id}/transitions  transition matrix as JSON
GET    /sequence-models/hmm-sequences/{id}/dwell        dwell-time histograms as JSON
POST   /sequence-models/hmm-sequences/{id}/cancel
```

PR 3 / PR 4 add `/overlay`, `/label-distribution`, `/exemplars`, `/motifs` endpoints to the HMM job, no new top-level routers.

A new `docs/reference/sequence-models-api.md` reference doc is created in PR 1 and grown as PRs 2–4 land. Doc-update matrix in CLAUDE.md §10.2 covers this entry.

---

## 7. Frontend Surface

### PR 1 — `Sequence Models → Continuous Embedding`

- `ContinuousEmbeddingJobsPage.tsx` — list view: active vs previous, cards with status + summary stats (region count, span count, window count, vector_dim).
- `ContinuousEmbeddingCreateForm.tsx` — selector for `region_detection_job_id` (dropdown of completed Pass 1 jobs), inputs for `hop_seconds`, `pad_seconds`, model_version (defaults to surfperch-tensorflow2).
- `ContinuousEmbeddingDetail.tsx` — job manifest summary, per-span row counts, error display if failed. No charts in PR 1 — producer plumbing only.
- TanStack Query hooks polling at 3s on active jobs.

### PR 2 — `Sequence Models → HMM Sequence`

- `HMMSequenceJobsPage.tsx` — list/active/previous pattern.
- `HMMSequenceCreateForm.tsx` — source selector (dropdown of completed `ContinuousEmbeddingJob`s), inputs for `n_states`, `pca_dims`, `covariance_type`, `n_iter`, `random_seed`, `min_sequence_length_frames`.
- `HMMSequenceDetail.tsx` — three minimum-viable Plotly charts: state timeline (per merged span, with span selector), transition matrix heatmap, dwell-time histograms grid. Plus job summary stats.
- Reuses existing chart utilities and shadcn/ui components per `docs/reference/frontend.md`.

### PR 3 / PR 4 frontend additions

- PR 3: PCA/UMAP scatter colored by state; per-state stacked-bar label distribution; per-state exemplar gallery with audio playback.
- PR 4: motif gallery with sortable/filterable list; per-motif occurrence drill-down.

---

## 8. Testing

Per CLAUDE.md §5, testing is mandatory. Strategy: **stubbed model + synthetic-but-realistic-shape sequences**, no committed binary fixtures.

### 8.1 Fixtures

- **SurfPerch model stub** (`tests/fixtures/sequence_models/surfperch_stub.py`): deterministic stub mimicking the SurfPerch runner interface — same audio in → same fixed-shape embedding out, where the shape matches whatever the registered SurfPerch `model_config` reports (vector_dim is discovered at run time, not hard-coded in the stub). Wired via the existing model-stub dependency-injection / monkeypatch path.
- **Synthetic embedding sequence generator** (`tests/fixtures/sequence_models/synthetic_sequences.py`): generates sequences with planted state structure (configurable `n_states`, transition matrix, dwell distribution, sequence lengths, seed); returns embeddings + ground-truth labels.
- **Synthetic region geometry fixtures**: hand-crafted region intervals exercising non-overlap, padded-overlap merge, multi-region merge, pad-clip-at-start, pad-clip-at-end, single-region edges.

### 8.2 Unit tests — PR 1

- `tests/processing/test_region_windowing.py`: `merge_padded_regions` (all geometry cases), `iter_windows` (window-center-in-region rule, `source_region_ids` correctness).
- `tests/services/test_continuous_embedding_service.py`: idempotency (same signature → existing job), in-flight blocking (no duplicates while running), validation (rejects non-existent region_detection_job_id, unsupported model_version).
- `tests/workers/test_continuous_embedding_worker.py`: end-to-end with stubbed SurfPerch + tiny synthetic hydrophone audio (correct row count, schema, types); failure path; atomic write (no partial canonical artifact on failure); cancellation between regions.

### 8.3 Unit tests — PR 2

- `tests/sequence_models/test_pca_pipeline.py`: L2-norm + PCA fit/transform; with/without whiten; deterministic given seed.
- `tests/sequence_models/test_hmm_trainer.py`: recovers planted state structure on synthetic sequences (Hungarian-aligned state accuracy ≥ 0.85); transition matrix recovery within tolerance; `min_sequence_length_frames` filter excludes short sequences from training but still decodes them; determinism on fixed seed.
- `tests/sequence_models/test_hmm_decoder.py`: Viterbi + posterior shape correctness; `max_state_probability` matches argmax(posterior).
- `tests/sequence_models/test_summary.py`: dwell-time histogram bins; occupancy fractions; transition matrix shape.
- `tests/services/test_hmm_sequence_service.py`: source-job-must-be-complete validation.
- `tests/workers/test_hmm_sequence_worker.py`: end-to-end producing all expected artifacts; failure mode; cancellation between EM iterations.

### 8.4 Frontend Playwright tests

- **PR 1** `frontend/tests/sequence-models/continuous-embedding.spec.ts`: nav, create flow with seeded region-detection job, detail page on complete, error state on failed.
- **PR 2** `frontend/tests/sequence-models/hmm-sequence.spec.ts`: create form constrained to completed continuous-embedding-jobs; detail page renders all three charts; span selector switches between merged spans.

### 8.5 Verification gates per PR

Standard CLAUDE.md §10.2 gate, plus the **pre-migration DB backup as first acceptance criterion** (CLAUDE.md §3.5):

1. Pre-migration backup of production DB (mandatory; first acceptance criterion).
2. `uv run ruff format --check` on modified Python files.
3. `uv run ruff check` on modified Python files.
4. `uv run pyright` on modified Python files.
5. `uv run pytest tests/` — full suite green.
6. `cd frontend && npx tsc --noEmit` — green.
7. `cd frontend && npx playwright test` — green.
8. `uv run alembic upgrade head` runs cleanly against the production DB.

### 8.6 Out-of-scope for unit tests

Real SurfPerch numerical output values; cross-version `hmmlearn` numerical reproducibility (we pin a version in `pyproject.toml`); HSMM / GMM-HMM / motif mining (deferred); performance benchmarks (measured ad hoc when running against real 24hr hydrophone data).

---

## 9. Manual Evaluation Gate (between PR 2 and PR 3)

Per the source design's "Minimal First Milestone" guidance and explicit caution to not extend until baseline is evaluated:

**Before starting PR 3**, run a real `HMMSequenceJob` against a real `ContinuousEmbeddingJob` (one or more 24hr hydrophone days) and visually confirm against the source design's success criteria:

- **Temporal coherence**: states persist >1–2 frames; transitions aren't random flicker; repeated patterns appear.
- **Spatial coherence**: states occupy coherent regions in PCA space; state exemplars look acoustically related.
- **Plausible dwell-time distributions**: per-state histograms make biological sense.

If the baseline shows any of the four documented failure modes — rapid flickering, dominant state, spatially incoherent states, recording-identity states — pause PR 3 and address via the source design's documented fixes (fewer states, GMM-HMM exploration, normalization checks, per-recording normalization, train/test split by hydrophone). This is a **workflow checkpoint**, not a code gate.

---

## 10. Dependencies & Configuration

### 10.1 New Python dependency (PR 2)

- `hmmlearn>=0.3.2` added to `pyproject.toml`. PCA, UMAP, scikit-learn, scipy, numpy already present.

### 10.2 New runtime configuration

No new `HUMPBACK_`-prefixed settings required for PR 1 / PR 2. All HMM hyperparameters and producer settings are per-job parameters, not environment-level config.

### 10.3 Documentation updates (per CLAUDE.md §10.2 doc-update matrix)

- **PR 1**: add new `docs/reference/sequence-models-api.md`, update `docs/reference/storage-layout.md`, add Sequence Models to CLAUDE.md §9.1 implemented capabilities, append ADR for the new track in `DECISIONS.md`, README.md endpoint list.
- **PR 2**: extend `sequence-models-api.md` with HMM endpoints, extend `storage-layout.md` with `hmm_sequences/` tree, add HMM capability to CLAUDE.md §9.1, append ADR for the modeling-job design.
- **PR 3 / PR 4**: extend reference docs and CLAUDE.md §9.1 as features land.

---

## 11. Open Risks & Mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Real-data HMMs show one of the four failure modes | Medium | Manual evaluation gate (§9) is a hard checkpoint; source design enumerates fixes for each mode. |
| Region-bounded sequences are too short for stable transitions even with 10s pad | Medium | `min_sequence_length_frames` filter at HMM time; pad is configurable; revisit by raising default pad or by relaxing region filtering at producer time. |
| `hmmlearn` numerical instability with `vector_dim=1536` | Low (PCA reduces to 20–50) | Default `pca_dims=50`; diagonal covariance default; document in spec. |
| MPS-acceleration becomes blocking on real datasets | Low for MVP scale | Pomegranate + MPS path is a documented future-work item, gated on measured benchmark need. |
| Vocalization-label coverage is too sparse for state-to-label viz | Medium | The viz tolerates sparsity (unlabeled bucket per state); few-hundred labeled HMM windows give adequate signal; widen labeling effort if interpretation is consistently underdetermined. |

---

## 12. References

- Source design notes: `/Users/michael/development/perch_hmm_sequence_model_design.md`
- CLAUDE.md sections: §3.5 (DB migrations & backup), §3.7 (frontend stack), §3.8 (UTC standard), §4 (core design principles), §5 (testing), §10.2 (verification gates & doc-update matrix)
- ADR-048 through ADR-055 (call parsing scaffold, perch_v2 first-class, read-time correction overlay) — adjacent-but-independent context
- `docs/reference/frontend.md`, `docs/reference/storage-layout.md`, `docs/reference/testing.md`, `docs/reference/data-model.md`, `docs/reference/runtime-config.md`
- External: `hmmlearn` 0.3 (https://hmmlearn.readthedocs.io/), Perch 2.0 transfers 'whale' to underwater tasks (arXiv 2512.03219), pomegranate v1 PyTorch backend (https://pomegranate.readthedocs.io/)
