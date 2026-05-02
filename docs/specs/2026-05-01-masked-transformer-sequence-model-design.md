# Masked-Transformer Sequence Model — Design Spec

**Date:** 2026-05-01
**Status:** Approved
**Reference:** `/Users/michael/development/Event-Conditioned Motif Discovery Pipeline.md` (external, not in repo)

## 1. Purpose

Add a **Masked Transformer** workflow to the Sequence Models track, parallel to **HMM Sequence**. The workflow consumes existing CRNN region-based continuous embeddings (ADR-057), trains a masked-span transformer encoder, extracts contextual embeddings, fits k-means tokenizers, decodes per-chunk tokens with confidences, and feeds the existing motif-extraction pipeline.

Reference pipeline (verbatim from the source document):

```
event segments
  → CRNN embeddings (250 ms hop)        [existing: continuous_embedding_jobs / source_kind=region_crnn]
  → masked transformer                  [NEW: masked_transformer_jobs]
  → contextual embeddings (Z)           [NEW]
  → k-means tokens                      [NEW: per-k artifacts]
  → repeated n-gram motif discovery     [REUSED: motif_extraction_jobs, generalized]
```

## 2. Scope (v1)

- **Embedding source coverage:** CRNN region-based only (ADR-057). SurfPerch is deferred to a follow-up; the loader abstractions stay shaped to allow it later, but only the CRNN path is exercised at v1.
- **Job structure:** single training job per (upstream embedding, training config). Each job carries multiple k-means tokenization artifacts (one per k in the configured `k_values` sweep). The k value is selected on the detail page via a URL-persisted picker.
- **Motif extraction:** existing `motif_extraction_jobs` table is generalized to point at either an HMM job or a masked-transformer job (with k).
- **Loader normalization:** existing per-source HMM loaders are extended to read decoded sequences from a normalized `decoded.parquet` schema; masked-transformer per-k bundles use the same schema. HMM worker renames `state` column to `label`.

## 3. Architecture

### 3.1 Data flow

```
continuous_embedding_jobs (source_kind = region_crnn)
        │
        ├──→ hmm_sequence_jobs ──┐
        │                        ├──→ motif_extraction_jobs (generalized: parent_kind ∈ {hmm, masked_transformer})
        └──→ masked_transformer_jobs ──┘
                  │
                  └── per-k artifacts under k<N>/ subdirs
                       (decoded.parquet, kmeans.joblib, overlay.parquet,
                        exemplars.json, label_distribution.json, run_lengths.json)
```

### 3.2 Reuse strategy

The Sequence Models track already provides excellent abstractions. The masked-transformer workflow plugs in by:

- **Same loader Protocol** (ADR-059, `SequenceArtifactLoader`): existing `CrnnRegionLoader` gains a `decoded_artifact_path` parameter so it can read either an HMM job's `decoded.parquet` or a masked-transformer job's `k<N>/decoded.parquet`. No new loader implementations.
- **Same interpretation viz** (ADR-059): `compute_overlay`, `select_exemplars`, `compute_label_distribution` are called unchanged. The detail page reuses the overlay scatter, exemplar gallery, label distribution chart, and tier composition strip.
- **Same motif extraction** (ADR-058): generalized via `parent_kind` discriminator + nullable parent FKs + nullable `k` column. Worker constructs the loader with the appropriate `decoded_artifact_path`; the extraction algorithm is parent-agnostic.
- **Same timeline infrastructure**: spectrogram + `ConfidenceStrip` (gradient heatmap) + a new `DiscreteSequenceBar` (refactored from `HMMStateBar` with `mode: "rows" | "single-row"`). Both detail pages benefit from the generalized component.
- **Same lifecycle pattern**: queued → running → completed/failed/cancelled, async interpretation generation, per-job dirs, atomic write semantics.

### 3.3 Key new primitives

- **Masked-span transformer trainer** — PyTorch `nn.Module` with span masking, MSE (+ optional cosine) loss, MPS/CUDA training with synthetic-batch validation + CPU fallback (mirrors Pass 2/3 inference device pattern).
- **K-means tokenization with softmax-temperature confidence** — τ auto-fit per job from median pairwise centroid distance, yielding a [0,1] confidence that drops into the existing `max_state_probability` UI slots without UI changes.
- **Per-k artifact fan-out** — one transformer training, multiple k-means tokenizers and downstream interpretation bundles. Per-k subdirs `k<N>/` with atomic write via `k<N>.tmp/` → rename.
- **Extend-k-sweep endpoint** — adds new k values to a completed job without retraining the transformer.
- **Reconstruction error per chunk** — captured during training validation, persisted as a timeline-aligned strip data source.

## 4. Backend design

### 4.1 New module: `src/humpback/sequence_models/masked_transformer.py`

- `MaskedTransformerConfig` dataclass:
  - `preset: Literal["small", "default", "large"]`
  - `mask_fraction: float` (default 0.20)
  - `span_length_min: int` (default 2), `span_length_max: int` (default 6)
  - `dropout: float` (default 0.1)
  - `mask_weight_bias: bool` (default True; weights event-adjacent positions higher in the loss)
  - `cosine_loss_weight: float` (default 0.0; MSE only)
  - `max_epochs: int`, `early_stop_patience: int`, `val_split: float`, `seed: int`
- `_PRESETS: dict[str, dict]`:
  - `small`: `d_model=128, num_layers=2, num_heads=4, ff_dim=512`
  - `default`: `d_model=256, num_layers=4, num_heads=8, ff_dim=1024`
  - `large`: `d_model=384, num_layers=6, num_heads=8, ff_dim=1536`
- `MaskedTransformer(nn.Module)`:
  - `Linear(input_dim → d_model) → TransformerEncoder(norm_first=True, GELU activation) → Linear(d_model → input_dim)`
  - Returns `(reconstructed, hidden_states)`.
- `apply_span_mask(seq, frac, span_min, span_max, rng) → (masked_seq, mask_positions)`:
  - Picks contiguous spans until coverage hits `frac`, replaces masked frames with sequence-mean embeddings.
- `train_masked_transformer(sequences, config, device) → TrainResult`:
  - Returns `(model, loss_curve, val_metrics, training_mask, reconstruction_error_per_chunk)`.
  - Tracks per-epoch train + val loss; supports early stop on val plateau.
  - When `mask_weight_bias=True`, scales loss per masked position by tier weight (event_core: 1.5, near_event: 1.2, background: 0.5; weights documented in module).
- `extract_contextual_embeddings(model, sequences, device) → (Z, lengths)`:
  - Runs encoder forward (no masking), returns concatenated hidden states + per-sequence lengths.

### 4.2 New module: `src/humpback/sequence_models/tokenization.py`

- `fit_kmeans_token_model(Z, k, seed) → (KMeans, tau)`:
  - Fits sklearn KMeans, computes τ as `median(pairwise_distance(centroids))`.
- `decode_tokens(Z, kmeans, tau) → (labels, confidences)`:
  - Confidence via softmax: `confidence_t = max(softmax(-‖z_t − μ‖² / τ))`.
- `compute_run_lengths(token_sequences, k) → dict[str, list[int]]`:
  - Per-token run-length arrays; keys are token-index strings (matches existing dwell histogram convention).

### 4.3 New service: `src/humpback/services/masked_transformer_service.py`

Mirrors `hmm_sequence_service.py`:

- `create_masked_transformer_job(payload) → MaskedTransformerJob`:
  - Validates upstream `continuous_embedding_jobs.source_kind == "region_crnn"` and `status == "completed"`.
  - Computes `training_signature` from `(continuous_embedding_job_id, preset, mask_fraction, span_length_min/max, dropout, mask_weight_bias, cosine_loss_weight, max_epochs, early_stop_patience, val_split, seed)`. **Excludes `k_values`** so k-sweep is extendable.
  - Idempotent: returns existing job if signature matches.
- `extend_k_sweep_job(job_id, additional_k) → MaskedTransformerJob`:
  - Only valid for `status="completed"`. Appends k values not already present, requeues a follow-up worker pass for tokenization + interpretation only.
- `cancel_masked_transformer_job(job_id)`, `delete_masked_transformer_job(job_id)`: lifecycle parity with HMM.
- `generate_interpretations(job_id, k) → None`:
  - Per-(job, k); calls `compute_overlay`, `select_exemplars`, `compute_label_distribution` from existing modules unchanged.

### 4.4 New worker: `src/humpback/workers/masked_transformer_worker.py`

- Loads upstream `continuous_embedding_jobs` parquet via existing helpers.
- **Device validation:** runs forward+backward on a fixed synthetic batch on both CPU and chosen accelerator; asserts losses within tolerance (e.g., `abs(cpu_loss - acc_loss) / cpu_loss < 0.05`); persists `chosen_device` + `fallback_reason` on the job row.
- Trains transformer (`train_masked_transformer`); persists `transformer.pt`, `loss_curve.json`, `reconstruction_error.parquet` (per-chunk MSE on the validation pass).
- Extracts Z; persists `contextual_embeddings.parquet`.
- For each k in `k_values`:
  - Fits k-means + computes τ.
  - Decodes tokens; writes `k<N>/decoded.parquet` (schema `(sequence_id, position, label, confidence)`).
  - Persists `k<N>/kmeans.joblib`, `k<N>/run_lengths.json`.
  - Triggers `generate_interpretations(job_id, k)` to produce `k<N>/overlay.parquet`, `k<N>/exemplars.json`, `k<N>/label_distribution.json`.
- **Atomic per-k writes:** write to `k<N>.tmp/`, then rename to `k<N>/`.
- **Extend-k-sweep follow-up path:** same loop, but only for the new k values; transformer + Z untouched.

### 4.5 Loader generalization: `src/humpback/sequence_models/loaders/crnn_region.py`

- `CrnnRegionLoader.__init__` gains `decoded_artifact_path: str` parameter.
- Replaces hard-coded `states.parquet` lookup with the explicit path.
- Service-layer dispatcher `get_loader(source_kind, decoded_artifact_path)` becomes the construction entry point.
- HMM service passes `<hmm_job_dir>/decoded.parquet`; masked-transformer service passes `<mt_job_dir>/k<N>/decoded.parquet`.

### 4.6 HMM worker rename: `state` → `label`

- HMM worker's output parquet column `state` is renamed to `label` for symmetry with the new normalized schema.
- HMM service / API serialization layer maps `label` back to `viterbi_state` for the existing API contract (no frontend change required).

### 4.7 Motif extraction generalization

- `motif_extraction_jobs` table additions: `parent_kind text NOT NULL`, `masked_transformer_job_id INTEGER NULL FK`, `k INTEGER NULL`. `hmm_sequence_job_id` becomes nullable. CHECK constraint enforces XOR between the two parent FKs and consistency with `parent_kind`.
- `MotifExtractionConfig.config_signature()` gains `parent_kind`, parent FK, and `k`.
- `services/motif_extraction_service.py` `create_motif_extraction_job()` branches on `parent_kind` for upstream validation.
- `workers/motif_extraction_worker.py` constructs the loader with the appropriate `decoded_artifact_path`; the rest of `extract_motifs()` is parent-agnostic.

## 5. Database

### 5.1 New table: `masked_transformer_jobs` (migration `063_masked_transformer_jobs.py`)

```
id (PK, autoincrement)
created_at, updated_at  (UTC, per CLAUDE.md §3.8)
status  (queued | running | completed | failed | cancelled)
status_reason  (text, nullable)

continuous_embedding_job_id  (FK, NOT NULL)
training_signature  (text, indexed unique)

# Training config
preset  (text: small | default | large)
mask_fraction  (float)
span_length_min  (int)
span_length_max  (int)
dropout  (float)
mask_weight_bias  (boolean)
cosine_loss_weight  (float)
max_epochs  (int)
early_stop_patience  (int)
val_split  (float)
seed  (int)

# Tokenization config
k_values  (JSON, list[int])

# Device + outcomes
chosen_device  (text: cpu | mps | cuda)
fallback_reason  (text, nullable)
final_train_loss  (float, nullable)
final_val_loss  (float, nullable)
total_epochs  (int, nullable)

# Storage
job_dir  (text)
total_sequences  (int, nullable)
total_chunks  (int, nullable)
```

### 5.2 Migration `064_motif_extraction_jobs_generalize_parent.py`

Uses `op.batch_alter_table()` for SQLite compatibility:

- Adds `parent_kind` (text, NOT NULL, default `"hmm"` for backfill).
- Adds `masked_transformer_job_id` (nullable FK to `masked_transformer_jobs.id`).
- Adds `k` (nullable int).
- Makes `hmm_sequence_job_id` nullable.
- Adds CHECK constraint: exactly one parent FK is set, consistent with `parent_kind`, and `k IS NOT NULL` iff `parent_kind = "masked_transformer"`.
- Backfills existing rows with `parent_kind = "hmm"`.

### 5.3 Mandatory production DB backup

Per CLAUDE.md §3.5, both migrations are preceded by a backup:

```
cp "$DB_PATH" "${DB_PATH}.YYYY-MM-DD-HH:mm.bak"  # UTC timestamp
```

with `$DB_PATH` read from `HUMPBACK_DATABASE_URL` in `.env`. Backup must exist with non-zero size before `alembic upgrade head` runs.

## 6. API

All routes live in `src/humpback/api/routers/sequence_models.py`:

```
POST   /sequence-models/masked-transformers
GET    /sequence-models/masked-transformers
GET    /sequence-models/masked-transformers/:jobId
POST   /sequence-models/masked-transformers/:jobId/extend-k-sweep
POST   /sequence-models/masked-transformers/:jobId/cancel
DELETE /sequence-models/masked-transformers/:jobId
POST   /sequence-models/masked-transformers/:jobId/generate-interpretations

GET    /sequence-models/masked-transformers/:jobId/loss-curve
GET    /sequence-models/masked-transformers/:jobId/tokens?k=N
GET    /sequence-models/masked-transformers/:jobId/overlay?k=N
GET    /sequence-models/masked-transformers/:jobId/label-distribution?k=N
GET    /sequence-models/masked-transformers/:jobId/exemplars?k=N
GET    /sequence-models/masked-transformers/:jobId/run-lengths?k=N
GET    /sequence-models/masked-transformers/:jobId/reconstruction-error?k=N
```

Per-k endpoints: `k` query parameter required; default = first entry of `k_values`. 404 on unknown k.

`generate-interpretations` body: `{ k_values: list[int] | null }` (null = all configured k).

`extend-k-sweep` body: `{ additional_k: list[int] }`. Only valid for `status="completed"`. Dedupes existing k values.

Schemas in `src/humpback/schemas/sequence_models.py`:

- `MaskedTransformerJobCreate` — required `continuous_embedding_job_id`, optional `preset` (default `"default"`), `k_values` (default `[100]`), advanced overrides
- `MaskedTransformerJobOut`, `MaskedTransformerJobDetail` — mirror HMM DTOs; detail includes `tier_composition`, `k_values`, `chosen_device`, `fallback_reason`
- `ExtendKSweepRequest` — `{ additional_k: list[int] }`
- `LossCurveResponse` — `{ epochs: list[int], train_loss: list[float], val_loss: list[float|null] }`
- `ReconstructionErrorResponse` — `(sequence_id, position, score)` shape that `ConfidenceStrip` consumes
- Reused: `OverlayResponse`, `ExemplarsResponse`, `LabelDistribution`, `StateTierComposition`
- `MotifExtractionJobCreate` gains `parent_kind`, optional `masked_transformer_job_id`, optional `k` (XOR validator)

## 7. Frontend

### 7.1 Routing

- `/app/sequence-models/masked-transformer` → `MaskedTransformerJobsPage`
- `/app/sequence-models/masked-transformer/:jobId` → `MaskedTransformerDetailPage` (reads `?k=` from URL)

Sequence Models nav adds a third top-level card: **Masked Transformer**, alongside Continuous Embedding and HMM Sequence.

### 7.2 New & generalized components

- `frontend/src/components/sequence-models/DiscreteSequenceBar.tsx` — refactor of `HMMStateBar.tsx`. Props add `mode: "rows" | "single-row"`, `numLabels`, `colorPalette`, `tooltipFormatter?`.
  - `rows` mode: current `HMMStateBar` behavior.
  - `single-row` mode: 60px-tall canvas, full-height fillRect per chunk with palette-mapped color.
  - HMM detail page swaps `HMMStateBar` import → `DiscreteSequenceBar mode="rows"`.
- `frontend/src/components/sequence-models/RegionNavBar.tsx` — extracted from `HMMSequenceDetailPage`'s inline CRNN region nav (A/D shortcuts, region count). Used by both HMM and masked-transformer detail pages.
- `frontend/src/components/sequence-models/KPicker.tsx` — top-of-page tab/select component, URL-synced via `useSearchParams("k")`. Masked-transformer detail page only.
- `frontend/src/components/sequence-models/MaskedTransformerJobsPage.tsx` — list page mirroring `HMMSequenceJobsPage`.
- `frontend/src/components/sequence-models/MaskedTransformerCreateForm.tsx`:
  - Always visible: upstream embedding job picker (filtered to `source_kind=region_crnn`, `status=completed`), preset radio, k_values CSV input (default `100`), max_epochs, mask-weight bias toggle.
  - Advanced disclosure: mask_fraction (default 0.20), span_length_min/max (2/6), dropout (0.1), cosine_loss_weight (0.0), early_stop_patience (3), val_split (0.1), seed (42).
- `frontend/src/components/sequence-models/MaskedTransformerDetailPage.tsx`:
  - Header (status, chosen_device + fallback badges), tier-composition strip, KPicker, loss curve.
  - Timeline block: spectrogram + `DiscreteSequenceBar mode="single-row"` (token strip) + `ConfidenceStrip` token-confidence + `ConfidenceStrip` reconstruction-error.
  - Token run-length histograms (re-skin of `DwellHistogramsGrid`, relabeled).
  - Overlay scatter (reused `OverlayScatterChart`).
  - Exemplar gallery (reused `ExemplarGallery`, "Token N" label).
  - Label distribution (reused `LabelDistributionChart`).
  - Motif extraction panel (reused `MotifExtractionPanel`, parent_kind="masked_transformer", k pre-filled).
- `frontend/src/components/sequence-models/LossCurveChart.tsx` — Plotly line plot, train + val loss per epoch.
- `frontend/src/components/sequence-models/TokenRunLengthHistograms.tsx` — re-skin of `DwellHistogramsGrid`.

### 7.3 Constants

- `STATE_COLORS` → `LABEL_COLORS` rename in `constants.ts`.
- New helper `labelColor(idx, total)` — returns `palette[idx % palette.length]` for `total ≤ 30`, generated HSL ramp for larger `total`.

### 7.4 API client (`frontend/src/api/sequenceModels.ts`)

- New TS interfaces: `MaskedTransformerJob`, `MaskedTransformerJobDetail`, `MaskedTransformerJobCreate`, `LossCurveResponse`, `ReconstructionErrorResponse`, `ExtendKSweepRequest`.
- Reuse: `OverlayResponse`, `ExemplarsResponse`, `LabelDistribution`, `StateTierComposition`.
- New fetch functions + TanStack Query hooks for all endpoints; per-k hooks key on `(jobId, k)`.
- Motif extraction: existing hooks gain `parent_kind` filtering; create form gains parent-kind selector.

## 8. Storage layout

```
masked_transformer_jobs/
  <job_id>/
    manifest.json
    transformer.pt
    contextual_embeddings.parquet
    reconstruction_error.parquet
    loss_curve.json
    k50/
      decoded.parquet            # (sequence_id, position, label, confidence)
      kmeans.joblib
      overlay.parquet
      exemplars.json
      label_distribution.json
      run_lengths.json
    k100/
      ...
```

Atomic write: `k<N>.tmp/` → rename to `k<N>/`.

## 9. Testing

### 9.1 Backend unit

- `tests/sequence_models/test_masked_transformer.py`:
  - `apply_span_mask` — fraction in 15–30%, span lengths in 2–6, deterministic with seed.
  - `MaskedTransformer` forward shape.
  - `train_masked_transformer` — loss decreases on synthetic sinusoidal sequences, early-stop fires when val plateaus, mask-weight bias weights event-adjacent positions higher.
  - `extract_contextual_embeddings` — output shape and ordering.
- `tests/sequence_models/test_tokenization.py`:
  - `fit_kmeans_token_model` — τ positive, scales with median centroid distance.
  - `decode_tokens` — confidences are valid probabilities, assignments match nearest-centroid.
  - `compute_run_lengths` — handles empty / single-token / boundary cases.
- `tests/sequence_models/test_loaders.py` (extend):
  - `CrnnRegionLoader` with `decoded_artifact_path` — loads from masked-transformer per-k path.
  - HMM path regression after parameter addition.
- `tests/sequence_models/test_motifs.py` (extend):
  - `extract_motifs` works with `parent_kind="masked_transformer"`; tier-aware weighting still applies.

### 9.2 Backend worker

- `tests/workers/test_masked_transformer_worker.py`:
  - End-to-end on tiny synthetic CRNN embedding job: trains, persists artifacts in correct dir layout, status transitions queued → running → completed.
  - Atomic write semantics: failure mid-write to `k100.tmp/` leaves no half-written `k100/`.
  - Device validation: forced fallback (mock MPS validation failure) → CPU fallback + `fallback_reason` set.
  - Idempotency: same `training_signature` → returns existing job_id.
  - Extend-k-sweep: trained transformer untouched, only new `k<N>/` subdirs written.
- `tests/workers/test_motif_extraction_worker.py` (extend):
  - Worker accepts `parent_kind="masked_transformer"`, reads from per-k `decoded.parquet`.

### 9.3 Backend integration / API

- `tests/integration/test_masked_transformer_api.py`:
  - Happy path + validation errors (non-CRNN upstream rejected, k_values must be non-empty list of ints ≥ 2).
  - List + detail endpoints.
  - Extend-k-sweep: only completed jobs, dedupes existing k.
  - Per-k artifact endpoints; 404 on unknown k.
  - Cancel + delete.
- `tests/integration/test_motif_extraction_api.py` (extend):
  - `POST /motif-extractions` with `parent_kind="masked_transformer"`; rejects XOR violation.

### 9.4 Frontend Playwright

- `frontend/e2e/sequence-models/masked-transformer.spec.ts`:
  - Job-create form: preset selection, k_values CSV parsing, advanced disclosure, validation.
  - Job table: status badges, k_values chip, device badge, row actions.
  - Detail page: k-picker switches URL + reloads per-k panels, loss curve renders, timeline strips render, exemplar gallery + label distribution render, motif panel pre-fills (parent_kind, k).

### 9.5 Test fixtures

- Tiny synthetic CRNN embedding fixture builder (extend `tests/conftest.py`): generates a complete `continuous_embedding_jobs` row + parquet for ~5 regions × ~20 chunks at d=64.
- Synthetic transformer trainer config that converges in <5 epochs on the fixture.

## 10. Documentation

- **DECISIONS.md** — append **ADR-061: Masked-transformer sequence model parallel to HMM** capturing: source-agnostic loader normalization on `decoded.parquet`, per-k artifact fan-out, generalized `motif_extraction_jobs`, training-signature idempotency + extend-k-sweep, mask-weight bias, softmax-temperature confidence, MPS/CUDA training with synthetic-batch validation + CPU fallback, `DiscreteSequenceBar` generalization.
- **CLAUDE.md §9.1** — append the masked-transformer workflow to the Sequence Models bullet.
- **CLAUDE.md §9.2** — bump latest migration; add `masked_transformer_jobs` table; note `motif_extraction_jobs` parent generalization.
- **`docs/reference/sequence-models-api.md`** — add the masked-transformer endpoint listing, generalized motif-extraction parent fields, extend-k-sweep semantics.
- **`docs/reference/data-model.md`** — add `masked_transformer_jobs`; document generalized `motif_extraction_jobs` columns.
- **`docs/reference/storage-layout.md`** — add the per-k subdir layout.
- **`docs/reference/frontend.md`** — add Masked Transformer route, `DiscreteSequenceBar` generalization, `RegionNavBar` extraction.
- **`docs/reference/behavioral-constraints.md`** — note the `decoded.parquet` schema convention as a behavioral contract for any future sequence-model consumer; note training-signature / extend-k-sweep semantics.
- **README.md** — add Masked Transformer to the feature list under Sequence Models.

## 11. Out of scope (v1)

- SurfPerch event-padded source for masked transformer (deferred follow-up; abstractions stay shaped to allow it).
- Distributed / multi-GPU training (single-machine MVP).
- Token-bigram heatmap (dropped per Q9; can be revisited if interpretation needs it later).
- Transition matrix display (semantically wrong as Markov for k-means tokens).
- Pretrained transformer initialization or transfer learning across jobs.

## 12. Success criteria

The workflow is working end-to-end if:

- A user can create a masked-transformer job from a completed CRNN region-based continuous-embedding job, the worker trains the transformer, persists per-k tokenization bundles, and the detail page renders the timeline strips + overlay + exemplars + label distribution + motif panel correctly for any selected k.
- Extend-k-sweep adds new k values to a completed job without retraining; existing k bundles are preserved.
- Idempotency: re-creating a job with the same `training_signature` returns the existing job.
- Motif extraction with `parent_kind="masked_transformer"` produces motifs equivalent in shape to HMM-parent motifs.
- HMM detail page continues to function unchanged after `state` → `label` rename and `DiscreteSequenceBar` generalization.
- Device validation correctly falls back to CPU when accelerator validation fails; chosen device + reason surface as a UI badge.
