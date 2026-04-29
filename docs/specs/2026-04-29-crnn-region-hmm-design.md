# CRNN Region-Based HMM Embedding Source — Design

**Status:** Approved (brainstorming complete 2026-04-29)
**Track:** Sequence Models (parallel to Call Parsing per ADR-056)
**Source doc:** `/Users/michael/development/CRNN Embeddings HMM.md` (Phase 1, sections 1–6 only)

---

## 1. Goal

Add a second embedding source to the existing Sequence Models track: 250 ms chunk embeddings extracted from the Pass 2 segmentation CRNN, computed per Pass 1 detection region, fed through the existing PCA + Gaussian HMM pipeline. Three training modes (full-region / event-balanced / event-only) become selectable on the HMM job. The existing SurfPerch event-padded path remains untouched.

## 2. Scope

### In scope (Phase 1 of source doc, sections 1–6)

- New embedding source family: `region_crnn` (Pass 1 detection region → CRNN BiGRU activations → 8-frame chunk concat → projection → 1024-d default chunk embedding)
- Region-as-sequence semantics for the new source (one HMM sequence per Pass 1 region; SurfPerch path remains event-padded-span-as-sequence)
- Per-chunk metadata: `call_probability`, `event_overlap_fraction`, `nearest_event_id`, `distance_to_nearest_event_seconds`, `tier` ∈ {`event_core`, `near_event`, `background`}
- Three HMM training modes: `full_region`, `event_balanced`, `event_only`. Decode is unchanged across modes (always full regions).
- Tier-balanced sub-sequence sampling for Mode B (stratified, L=32 chunks, configurable proportions/cap)
- Single-table schema dispatch on `model_version` (extend `continuous_embedding_jobs` and `hmm_sequence_jobs` with nullable CRNN-only columns)
- Frontend: source-type toggle on the embedding-job creation form, source badge on the list, training-mode block on the HMM-job creation form, generalized `SequenceNavigator` on the HMM detail page, per-state tier-composition stacked-bar strip
- ADR-057 in `DECISIONS.md`; documentation updates per CLAUDE.md §10.2 doc-update matrix

### Explicit non-goals (deferred)

- Contextual Transformer over CRNN chunk embeddings (source doc §1, second source) — deferred to its own spec
- Post-hoc state labeling beyond the tier-composition strip (source doc §7) — Phase 2
- State smoothing / cleanup (source doc §8) — Phase 2
- Motif extraction (source doc §9) — Phase 2
- Multi-source baselines harness (source doc §11) — Phase 2
- Success-criteria evaluation framework (source doc §12) — Phase 2
- Modifying the Pass 2 CRNN architecture or training a new checkpoint specifically for embedding extraction
- Real-time / streaming HMM decode
- Mixing SurfPerch and CRNN embeddings in a single HMM job (source doc §1: "do not mix them initially")

## 3. Architecture

```
                 ┌─────────────────────────┐
RegionDetection──┤ NEW: CRNNRegionEmbedder │──┐
    Job (Pass 1) │  uses Pass 2 CRNN ckpt  │  │
                 └─────────────────────────┘  │
                                              ├─►  continuous_embedding_jobs
                 ┌─────────────────────────┐  │      (one table, model_version
EventSegmentation┤ EXISTING: SurfPerch     │──┘       discriminates source)
    Job (Pass 2) │  event-padded embedder  │
                 └─────────────────────────┘
                                              │
                                              ▼
                                   hmm_sequence_jobs
                                   (PCA → GaussianHMM → Viterbi)
                                   training_mode + tier config
                                   only meaningful for CRNN source
                                              │
                                              ▼
                                  states.parquet + frontend
```

**Hard insulation rule for Pass 2:** Zero edits to `SegmentationCRNN`, zero edits to its training pipeline, zero edits to the substantive behavior of `event_segmentation_worker`. The single mechanical refactor allowed is extracting the windowed-inference loop into a shared helper module; the helper's behavior is byte-for-byte unchanged, verified by Pass 2's existing test suite plus a new before/after regression test.

## 4. New and Modified Modules

### New modules

- `src/humpback/sequence_models/crnn_features.py` — sole touchpoint for Pass 2 internals. Loads a Pass 2 checkpoint, registers a non-invasive PyTorch forward hook on the BiGRU module, drives windowed inference via the shared helper, captures BiGRU activations (T_frames × 128), slices into 8-frame chunks (1024-d concat), applies a `ChunkProjection`, emits per-chunk rows with `call_probability` (mean of per-frame sigmoid over the 8 frames). Does NOT compute event-overlap metadata (delegated to a join helper — see `event_overlap_join.py`).
- `src/humpback/sequence_models/event_overlap_join.py` — pure functions joining chunk rows against a Pass 2 events parquet for the same parent run. Computes `event_overlap_fraction`, `nearest_event_id`, `distance_to_nearest_event_seconds`, derives `tier` using configured thresholds.
- `src/humpback/sequence_models/chunk_projection.py` — `ChunkProjection` Protocol + three implementations: `IdentityProjection` (default; pass-through 1024-d), `RandomProjection(dim, seed)`, `PCAProjection(dim, whiten)`. Each implements `fit / transform / save / load`.
- `src/humpback/sequence_models/region_sampling.py` — pure functions implementing Mode B stratified sub-sequence extraction (default L=32 chunks, stride=16, target cap 200_000 chunks) plus the special-case configurations for full-region (Mode A) and event-only (Mode C). Returns training sub-sequences plus a per-chunk `was_used_for_training` mask aligned to the source region order.
- `src/humpback/call_parsing/segmentation/window_iter.py` — pure helper extracted from the existing `inference.py`. Yields `(window_audio, frame_offset_in_region)` tuples for a given region/audio/window/hop. Both Pass 2's `run_inference()` and the new CRNN extractor consume it.

### Modified modules

- `src/humpback/workers/continuous_embedding_worker.py` — split entry into a strategy dispatch on `source_kind` (derived from `model_version` family). Existing SurfPerch path moves into `_run_event_padded_surfperch()` with byte-identical behavior; new `_run_region_crnn()` calls the new producer. Schema-write path shared via a `_write_chunk_rows(rows, schema, path)` helper.
- `src/humpback/services/continuous_embedding_service.py` — extend `SUPPORTED_MODEL_VERSIONS` with the new CRNN source family. Extend `compute_encoding_signature()` to fold in CRNN-only fields (region_detection_job_id, crnn_checkpoint_sha256, chunk_size_seconds, chunk_hop_seconds, projection_kind, projection_dim, event_segmentation_job_id) when source is CRNN. SurfPerch signature formula unchanged.
- `src/humpback/workers/hmm_sequence_worker.py` — read `training_mode` and tier config off the job row; if source is CRNN, run `region_sampling` to build training sub-sequences before fitting; if source is SurfPerch, behavior is unchanged. PCA + HMM trainer + decoder unchanged. Persist `was_used_for_training` and tier columns to `states.parquet`. Compute per-state tier-composition aggregates into `summary.json` for fast frontend load.
- `src/humpback/sequence_models/pca_pipeline.py` — unchanged (already source-agnostic).
- `src/humpback/api/routers/sequence_models.py` — `POST /continuous-embeddings` accepts new source-type fields with XOR validation (`event_segmentation_job_id` XOR `region_detection_job_id`). `POST /hmm-sequence` accepts training-mode + tier config; rejects them with 422 when source is SurfPerch.
- `src/humpback/schemas/sequence_models.py` — extend request/response Pydantic models.
- `src/humpback/database.py` + new Alembic migration — add nullable columns from §5.
- `src/humpback/call_parsing/segmentation/inference.py` — mechanical refactor: replace inlined windowing math with a call to `window_iter.iter_inference_windows()`. No behavior change.

### Frontend changes

- `frontend/src/components/sequence-models/ContinuousEmbeddingNewPage.tsx` — top-of-form source-type toggle (`Event-padded (SurfPerch 1 s · 5 s window)` vs `Detection-region (CRNN 250 ms chunks)`). Conditional fields render under each option.
- `frontend/src/components/sequence-models/ContinuousEmbeddingListPage.tsx` — source badge column.
- `frontend/src/components/sequence-models/HMMSequenceNewPage.tsx` — training-mode select + collapsed Advanced panel for tier config; conditional on source kind being CRNN.
- `frontend/src/components/sequence-models/HMMSequenceDetailPage.tsx` — generalize span navigator into `SequenceNavigator` (label prop: `Span` for SurfPerch jobs, `Region` for CRNN jobs); add per-state tier-composition stacked-bar strip reading from `summary.json`.
- `frontend/src/api/sequenceModels.ts` — extended request/response types and `useTierComposition()` hook.

## 5. Data Model

### Alembic migration `061_crnn_region_embeddings.py`

`continuous_embedding_jobs` — add nullable columns:

| column | type | notes |
|---|---|---|
| `region_detection_job_id` | INTEGER FK → `region_detection_jobs.id` | populated for CRNN source |
| `chunk_size_seconds` | FLOAT | populated for CRNN source |
| `chunk_hop_seconds` | FLOAT | populated for CRNN source |
| `crnn_checkpoint_sha256` | TEXT | populated for CRNN source |
| `crnn_segmentation_model_id` | INTEGER FK → `segmentation_models.id` | populated for CRNN source |
| `projection_kind` | TEXT | `identity` \| `random` \| `pca` |
| `projection_dim` | INTEGER | populated for CRNN source |
| `total_regions` | INTEGER | populated for CRNN source |
| `total_chunks` | INTEGER | populated for CRNN source |

Existing fields (`event_segmentation_job_id`, `window_size_seconds`, `hop_seconds`, `pad_seconds`, `merged_spans`, `total_events`, `total_windows`) become nullable in interpretation — populated only for SurfPerch path. `op.batch_alter_table` is required for SQLite.

Application-layer constraint (Pydantic + service): exactly one of `event_segmentation_job_id` XOR `region_detection_job_id` must be set on creation.

`hmm_sequence_jobs` — add nullable columns:

| column | type | default | notes |
|---|---|---|---|
| `training_mode` | TEXT | NULL | `full_region` \| `event_balanced` \| `event_only`; required when source is CRNN |
| `event_core_overlap_threshold` | FLOAT | 0.5 | |
| `near_event_window_seconds` | FLOAT | 5.0 | |
| `event_balanced_proportions` | TEXT (JSON) | `{"event_core":0.4,"near_event":0.35,"background":0.25}` | |
| `subsequence_length_chunks` | INTEGER | 32 | |
| `subsequence_stride_chunks` | INTEGER | 16 | |
| `target_train_chunks` | INTEGER | 200000 | |
| `min_region_length_seconds` | FLOAT | 2.0 | producer-side filter |

### Encoding signature

- SurfPerch source: `sha256(event_segmentation_job_id, model_version, hop_seconds, window_size_seconds, pad_seconds, target_sample_rate, feature_config)` — unchanged
- CRNN source: `sha256(region_detection_job_id, event_segmentation_job_id, model_version, crnn_checkpoint_sha256, chunk_size_seconds, chunk_hop_seconds, projection_kind, projection_dim, target_sample_rate, feature_config)`

### `embeddings.parquet` schema for CRNN-source jobs

| column | type | notes |
|---|---|---|
| `region_id` | string | Pass 1 region id |
| `audio_file_id` | int32, nullable | from parent `RegionDetectionJob` |
| `hydrophone_id` | string, nullable | from parent `RegionDetectionJob` |
| `chunk_index_in_region` | int32 | 0-indexed within the region |
| `start_timestamp` | float64 | UTC epoch (CLAUDE.md §3.8) |
| `end_timestamp` | float64 | UTC epoch |
| `is_in_pad` | bool | true where chunk center is outside `[start_sec, end_sec]` |
| `call_probability` | float32 | mean of CRNN sigmoid over the 8 frames |
| `event_overlap_fraction` | float32 | fraction of chunk overlapping any Pass 2 event |
| `nearest_event_id` | string, nullable | id of closest event; null if outside `near_event_window_seconds` |
| `distance_to_nearest_event_seconds` | float32, nullable | signed seconds; useful for tier debug |
| `tier` | string | `event_core` \| `near_event` \| `background` |
| `embedding` | list\<float32\> | `projection_dim`-d (1024 for IdentityProjection default) |

For SurfPerch-source jobs the existing parquet schema is preserved unchanged.

### `states.parquet` schema for CRNN-source HMM jobs

Adds to existing schema:
- `region_id` (replaces `merged_span_id` semantics for CRNN rows)
- `chunk_index_in_region` (replaces `window_index_in_span`)
- `tier` (carried through from `embeddings.parquet`)
- `was_used_for_training` (bool, set by `region_sampling`)

Existing columns (`viterbi_state`, `state_posterior`, `max_state_probability`, `start_timestamp`, `end_timestamp`, `is_in_pad`, `audio_file_id`) are unchanged.

### Storage layout

```
{storage_root}/continuous-embeddings/{job_id}/
    embeddings.parquet      ← per-chunk rows
    manifest.json           ← job-level summary, projection config
    projection/projection.joblib   ← only when projection_kind != identity
```

`{storage_root}/hmm-sequences/{job_id}/` — existing layout, with `summary.json` extended to include per-state tier-composition aggregates for CRNN-source jobs.

## 6. Workflow

### Producer (CRNN region embedding job)

1. `POST /api/sequence-models/continuous-embeddings` with `source_kind="region_crnn"`, `region_detection_job_id`, `event_segmentation_job_id` (required disambiguator when multiple Pass 2 jobs exist for the same Pass 1 job), `crnn_segmentation_model_id`, `chunk_size_seconds=0.250`, `chunk_hop_seconds=0.250` (or `0.125`), `projection_kind="identity"`, `projection_dim=1024`, plus standard fields.
2. Service validates: `RegionDetectionJob.status == "complete"`; the named `EventSegmentationJob` exists and is `complete` and has the same parent `RegionDetectionJob`. Returns 422 on violation.
3. Compute encoding signature; if a `complete` job with the same signature exists, return it (idempotency).
4. Insert row with `status=queued` and all CRNN-side config columns.
5. Worker claims the row, transitions to `running`.
6. Load `SegmentationCRNN` from the resolved checkpoint path into eval mode on the selected device (existing `select_and_validate_device`). Compute and persist checkpoint sha256. Load-time guards assert BiGRU width and frame rate.
7. Initialize the configured `ChunkProjection`. For non-identity projections, fitting happens on the first region's concat embeddings (or on a deterministic sample) — Phase 1 default is `identity` so this is a no-op.
8. Per-region loop, skipping regions where `padded_end_sec - padded_start_sec < min_region_length_seconds`:
   - Resolve audio over `[padded_start_sec, padded_end_sec]` via existing `audio_loader` / `resolve_audio_slice`.
   - Iterate windows via `window_iter.iter_inference_windows()` (shared with Pass 2).
   - Forward each window with the BiGRU forward-hook attached → BiGRU activations (T_window_frames × 128) + per-frame logits.
   - Stitch overlapping windows by keeping the centre half (mirroring Pass 2's behavior) → full-region BiGRU stream + full-region per-frame call-probability stream.
   - Slice into 8-frame chunks at `chunk_hop_seconds` stride. Concat → 1024-d → projection → `embedding`. `call_probability` = mean of the chunk's per-frame sigmoid probs.
   - Run `event_overlap_join` against the upstream `EventSegmentationJob.events.parquet` filtered to this region: compute `event_overlap_fraction`, `nearest_event_id`, `distance_to_nearest_event_seconds`, derive `tier`.
   - Set `is_in_pad`. Append row.
9. Atomic write `embeddings.parquet`, `manifest.json`, optional `projection.joblib`.
10. Update job row counters; status `complete`.

### Consumer (HMM job over a CRNN-source embedding job)

1. `POST /api/sequence-models/hmm-sequence` with `continuous_embedding_job_id`, all existing HMM hyperparameters, plus the new CRNN-only fields (`training_mode` defaulting to `event_balanced` when source is CRNN, full tier config with the documented defaults).
2. API validates: tier fields and `training_mode` rejected with 422 when source is SurfPerch; `event_balanced_proportions` must sum to 1.0 ± epsilon.
3. Worker loads `embeddings.parquet`, groups by `region_id`, sorts by `chunk_index_in_region` to produce per-region sequences `X_region`.
4. `region_sampling.build_training_set(sequences, mode, tier_config, sampling_config)`:
   - **Mode A (full_region):** training set = all chunks of all regions ≥ `min_sequence_length_frames`; `was_used_for_training=True` for all. If total exceeds `target_train_chunks`, uniformly subsample regions.
   - **Mode B (event_balanced):** walk each region; for each `event_core` chunk, extract a sub-sequence of length `L` centred on it (stride to avoid duplicates); also sample fixed-length sub-sequences from `near_event` and `background` tiers. Cap total chunks by `target_train_chunks` while preserving `event_balanced_proportions`. Sub-sequences are passed as separate sequences to hmmlearn. `was_used_for_training=True` only for chunks inside a sampled sub-sequence.
   - **Mode C (event_only):** sub-sequences from `event_core` ∪ `near_event` only; `background` excluded from training but decoded.
5. PCA fit on the L2-normalized (optional) concatenation of training sequences via existing `pca_pipeline.fit_pca`. Save `pca.joblib`.
6. Existing `hmm_trainer.fit_gaussian_hmm` consumes the (possibly fragmented) training sub-sequences with a `lengths` vector. Output: HMM model + `train_log_likelihood` + counts.
7. Existing `hmm_decoder.decode_sequences` decodes whole regions (not sub-sequences) via Viterbi + `predict_proba`. Decode covers all chunks including those marked `was_used_for_training=False` and `is_in_pad=True`.
8. Atomic write `states.parquet` (extended schema), `transition_matrix.npy`, `summary.json` (now including per-state tier-composition aggregates), PCA + HMM models to `artifact_dir`.
9. Update job row counters; status `complete`.

### Frontend

- New-job creation: source-type toggle drives conditional fields. CRNN side wires the `segmentation_models` picker; SurfPerch side keeps the existing `event_segmentation_jobs` picker.
- HMM detail page: header badge shows `CRNN · event_balanced · n_states=12 · pca=32` (or `SurfPerch · n_states=12 · pca=32` for the existing source). Generalized `SequenceNavigator` accepts a label prop (`Span` vs `Region`). State timeline, transition heatmap, dwell histogram, exemplars render unchanged. Per-state tier-composition stacked-bar strip reads from `summary.json`.

## 7. Failure Modes (must be tested)

- Region with no Pass 2 events → all chunks `tier="background"`, `nearest_event_id=null`, `distance_to_nearest_event_seconds=null`.
- Region shorter than `min_region_length_seconds` → producer skips, logs, does not appear in `embeddings.parquet`.
- CRNN checkpoint with wrong BiGRU width or frame rate → producer fails fast at load time with a clear error and a `failed` job status.
- SurfPerch-source HMM job request that includes `training_mode` or tier fields → API returns 422.
- `event_balanced_proportions` not summing to 1.0 ± epsilon → 422.
- Multiple Pass 2 jobs exist for the same Pass 1 job and the producer request omits `event_segmentation_job_id` → 422 with disambiguation message.
- Pass 2 inference refactor regression: `tests/call_parsing/test_pass2_refactor_regression.py` runs a fixture region through Pass 2 before and after the `window_iter` extraction and asserts byte-identical `events.parquet` output. Test failure blocks merge.

## 8. Tier Definitions and Sampling Configuration (locked defaults)

- `event_core_overlap_threshold` = 0.5 (`event_overlap_fraction >= 0.5` ⇒ tier `event_core`)
- `near_event_window_seconds` = 5.0 (chunk within ±5 s of any event but `event_overlap_fraction < 0.5` ⇒ tier `near_event`; otherwise `background`)
- `event_balanced_proportions` = `{event_core: 0.40, near_event: 0.35, background: 0.25}`
- `subsequence_length_chunks` (L) = 32 (≈ 8 s at 250 ms hop)
- `subsequence_stride_chunks` = 16
- `target_train_chunks` = 200_000
- `min_region_length_seconds` = 2.0

All persisted on the `hmm_sequence_jobs` row for reproducibility.

## 9. Key Decisions

1. **Single table dispatch on `model_version`.** `continuous_embedding_jobs` and `hmm_sequence_jobs` both grow nullable columns; the consumer code dispatches on source-kind. Maximum reuse of the HMM consumer; the existing SurfPerch path stays untouched. *(Q5)*
2. **Concat-and-project chunk embeddings with `IdentityProjection` default.** 8 frames × 128-d concat = 1024-d chunk vector; PCA in the existing `pca_pipeline` handles the dim reduction inside the HMM job. Projection abstraction kept for future experiments. *(Q2)*
3. **Shared windowing helper extracted from Pass 2 inference.** Both Pass 2 and the new CRNN extractor consume `iter_inference_windows()`. The Pass 2 refactor is mechanical, regression-tested, and the only edit allowed in Pass 2's source. *(Q3)*
4. **Region-as-sequence for CRNN; event-padded-span-as-sequence for SurfPerch.** Source-kind dictates sequence semantics; the HMM consumer is agnostic. *(Q3)*
5. **Padded span with `is_in_pad` filtering for training.** Mirrors the existing SurfPerch event-padded approach so per-chunk schema stays aligned across sources. *(Q3 sub-decision)*
6. **Stratified sub-sequence extraction for Mode B.** Sub-sequences of length L=32 chunks preserve local temporal context while letting the trainer hit the 40/35/25 mix. Modes A and C are special cases of the same code path. *(Q4)*
7. **Tier metadata computed at producer time.** Producer does the events-parquet join once; HMM job filters on the resulting `tier` column without redoing overlap math. Different HMM jobs can re-interpret with different thresholds via the persisted job-row config. *(Q5)*
8. **Frontend single creation form with source-type toggle.** Conditional fields under each toggle option; consistent with the abstraction goal. *(Q6)*
9. **Tier-composition strip on the HMM detail page is in scope.** Data is free (already on every row) and gives an immediate sanity-check that states are doing something tier-meaningful. Other Phase 2 visualizations remain deferred. *(Q6)*

## 10. Risks and Mitigations

1. **Pass 2 refactor regression.** *Mitigation:* before/after byte-identical regression test (`test_pass2_refactor_regression.py`). If the refactor cannot be made byte-identical, fall back to duplicating the windowing math inside `crnn_features.py` (Q3 option B as the safety net).
2. **BiGRU hidden-state discontinuity at 30-s window boundaries.** Inherited from Pass 2; acceptable for Phase 1. Flag for Phase 2 investigation if states correlate with window boundaries.
3. **PCA fit memory on large datasets.** Capped by `target_train_chunks` × `projection_dim` × 4 B (≈ 800 MB at defaults). Mode A also subsamples to the same cap to bound memory.
4. **Tier label drift on Pass 2 re-runs.** `event_segmentation_job_id` is part of the encoding signature, so different Pass 2 outputs are different sources — no silent staleness.
5. **Source-type toggle UX confusion.** Mitigation: prominent toggle with descriptive labels; conditional fields clearly grouped.

## 11. Documentation Updates (CLAUDE.md §10.2)

- `CLAUDE.md` §9.1 — extend Sequence Models track entry to mention CRNN region-based embedding source and three training modes.
- `DECISIONS.md` — append `ADR-057: CRNN region-based chunk embeddings as second Sequence Models source` covering decisions 1, 2, and 3 from §9.
- `docs/reference/sequence-models-api.md` — document new request/response fields, XOR validation rules, training-mode/tier rules.
- `docs/reference/data-model.md` — extend `continuous_embedding_jobs` and `hmm_sequence_jobs` field listings.
- `docs/reference/storage-layout.md` — extend with the new parquet schema variant.
- `README.md` — surface the new capability in the user-facing feature list.
