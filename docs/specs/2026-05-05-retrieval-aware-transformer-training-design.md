# Retrieval-Aware Transformer Training - Design

**Date:** 2026-05-05
**Status:** Approved shared roadmap
**Source:** `/Users/michael/development/Retrieval-Aware Transformer Training Design.md`

## 1. Purpose

Stage 0 nearest-neighbor diagnostics showed that the current CRNN -> masked
transformer pipeline contains useful call-shape signal, but the raw contextual
embedding space is poorly conditioned for retrieval. Removing dominant PCA
directions and PCA whitening improved cross-region same-label retrieval,
suggesting that the model has useful information but does not organize it in a
retrieval-friendly geometry.

This design adds a staged "retrieval-aware" path to the existing masked
transformer workflow. The goal is to teach the existing vanilla
`TransformerEncoder` to emit embeddings that are directly useful for
cross-region nearest-neighbor retrieval and motif discovery.

The design is intentionally split into implementation phases. Each phase can
receive its own implementation plan and feature branch while this document
remains the long-range specification.

## 2. Current Baseline

Existing masked-transformer flow:

```text
CRNN region embeddings
  -> masked-span TransformerEncoder
  -> contextual chunk embeddings
  -> k-means tokenization
  -> decoded.parquet
  -> interpretation artifacts
  -> motif extraction
```

Current implementation anchors:

- `src/humpback/sequence_models/masked_transformer.py`
  - `MaskedTransformer`
  - `MaskedTransformerConfig`
  - `train_masked_transformer`
  - `extract_contextual_embeddings`
- `src/humpback/workers/masked_transformer_worker.py`
  - trains the model
  - persists `transformer.pt`, `contextual_embeddings.parquet`,
    reconstruction error, and per-k bundles
- `src/humpback/services/masked_transformer_service.py`
  - creates idempotent training jobs
  - excludes `k_values` from `training_signature`
  - regenerates interpretation artifacts from the bound Classify job
- `scripts/masked_transformer_nn_report.py`
  - current Stage 0 nearest-neighbor diagnostic harness; Phase 0 refactors
    this logic into `src/` so reports are served through the API instead of
    living only as a standalone script

Stage 0 observed:

```text
raw_l2 cross-region same-label overlap:       24.8%
remove_pc10 cross-region same-label overlap:  31.8%
whiten_pca cross-region same-label overlap:   40.2%
```

Stage 1 should make raw learned embeddings more retrieval-friendly, not merely
improve post-hoc whitening.

## 3. Verified Label Semantics

The previous assumption that Call Parsing Classify labels are single-value is
not correct in the current code.

Verification:

- `src/humpback/call_parsing/types.py` documents `TypedEvent` as multi-label:
  a single event can produce multiple `TypedEvent` rows because Pass 3 uses
  sigmoid output.
- `src/humpback/call_parsing/event_classifier/inference.py` emits one
  `TypedEvent` per `(event, type)` whose score crosses its per-type threshold.
  If no type crosses threshold, it emits one fallback row for the best-scoring
  type with `above_threshold=False`.
- `src/humpback/sequence_models/label_distribution.py` unions all
  above-threshold model types, applies `VocalizationCorrection` add/remove
  overlays, and assigns a type set to each effective event.

Design decision for retrieval-aware training:

- Do not use model Classify labels for contrastive positives.
- Use human correction labels only.
- Treat human labels as a set per event, because corrections can add or remove
  multiple type names over the same event interval.

## 4. Goals

1. Add a retrieval projection head to the existing masked transformer.
2. Switch new masked-transformer tokenization and nearest-neighbor evaluation to
   retrieval embeddings.
3. Preserve masked-span reconstruction as the base objective.
4. Add event-level supervised contrastive training in a later phase.
5. Use human correction labels only for contrastive supervision.
6. Prefer same-label, different-region positives.
7. Keep downstream `decoded.parquet`, interpretation artifacts, and motif
   extraction source-agnostic.
8. Keep `k_values` excluded from the training signature so completed jobs can
   still extend k sweeps without retraining.

## 5. Non-Goals

- Replace the vanilla `TransformerEncoder`.
- Use a fancier architecture before testing retrieval-aware objectives.
- Train contrastive loss from model Classify labels.
- Force related or ambiguous label pairs into hard-separated islands in the
  first contrastive implementation.
- Change the public motif extraction schema.
- Require frontend UX for the first sweep workflows.
- Migrate historical masked-transformer jobs to have retrieval embeddings.

## 6. Design Decisions

### 6.1 Retrieval embeddings become the downstream embedding space

For new retrieval-aware masked-transformer jobs:

```text
Transformer hidden state
  -> retrieval projection head
  -> L2-normalized retrieval embedding
  -> k-means tokenization
  -> nearest-neighbor diagnostics
```

The current contextual hidden states remain useful for diagnostics and masked
reconstruction, but k-means tokenization should switch to retrieval embeddings
once the projection head exists.

Practical storage contract:

- Continue writing `contextual_embeddings.parquet` for compatibility and
  diagnostics.
- Add `retrieval_embeddings.parquet` for chunk-level retrieval vectors.
- Per-k `decoded.parquet` for retrieval-aware jobs is produced from
  `retrieval_embeddings.parquet`.
- Existing completed jobs without retrieval artifacts remain readable; their
  per-k bundles continue to mean "tokenized contextual embeddings."

### 6.2 Projection head

Add a small MLP after the transformer hidden state:

```text
hidden_t
  -> LayerNorm
  -> Linear
  -> GELU
  -> Linear
  -> L2 normalize
  -> retrieval_t
```

Suggested defaults:

```text
contextual dim: current preset d_model
hidden dim: 512 for default preset
retrieval dim: 128
```

The projection head is shared for chunk retrieval and event-level contrastive
training:

- Chunk embedding: `retrieval_t = head(hidden_t)`
- Event embedding: `retrieval_event = head(mean(hidden_t over event chunks))`

Mean pooling is the v1 event pooling method.

### 6.3 Human correction labels only

Add a retrieval-label loader that uses effective event boundaries but ignores
model Classify labels.

For each effective event:

```text
human_types = set()
for VocalizationCorrection overlapping event:
  if correction_type == "add":
    human_types.add(type_name)
  if correction_type == "remove":
    human_types.discard(type_name)
```

Notes:

- Effective event boundaries should come from `load_effective_events()` so
  boundary corrections are respected.
- Time bridging should match ADR-062 and ADR-063: event seconds are converted
  to absolute UTC with the upstream `RegionDetectionJob.start_timestamp`.
- Events with no surviving human types are still valid masked-modeling events
  but are excluded from supervised contrastive labels.
- `remove` rows without a corresponding surviving human add simply leave the
  event unlabeled for contrastive training.

### 6.4 Multi-label human events

Human correction labels are represented as a set.

Contrastive positive rule:

```text
positive(anchor, candidate)
  = anchor.human_types intersects candidate.human_types
```

Contrastive negative rule:

```text
negative(anchor, candidate)
  = both events have human labels
  and label sets are disjoint
  and pair is not excluded by related-label policy
```

Do not duplicate one event into separate physical training examples per label in
the first implementation. Instead, build the supervised-contrastive positive
mask from set intersection.

### 6.5 Region-aware sampling

The contrastive component should prefer batches that include:

- multiple labels
- multiple regions
- same labels across different regions
- same-region different-label hard negatives, once enabled

Default policy:

```text
labels per batch: 4
events per label: 4
minimum regions per label: 2 when available
rare labels: masked modeling only until enough events exist
```

### 6.6 Related label policy

Some label pairs may be acoustically adjacent or annotation-ambiguous. Early
contrastive training should avoid aggressively pushing them apart.

Initial related-label exclusions:

```text
Creak <-> Vibrate
Moan <-> Ascending Moan
Moan <-> Descending Moan
Growl <-> Buzz
Whup <-> Grunt
```

For Phase 3, the simplest policy is to exclude related-label pairs from the
negative mask. Down-weighting can come later.

## 7. Architecture

### 7.1 Model

Extend `MaskedTransformer` to optionally include a retrieval head.

Conceptual API:

```text
forward(x, src_key_padding_mask=None)
  -> reconstructed
  -> hidden
  -> retrieval
```

Implementation can return a dataclass or a tuple. A dataclass is preferred once
the forward result has more than two tensors.

Required behavior:

- Masked reconstruction still reads from `output_proj(hidden)`.
- Retrieval embeddings are produced from unmasked hidden states during
  extraction.
- Retrieval embeddings are L2-normalized.
- Projection-head parameters train even when contrastive loss is disabled,
  because downstream tokenization will use the head output.

### 7.2 Training objectives

Phase 1:

```text
total_loss = masked_loss
```

Phase 3:

```text
total_loss = masked_loss + contrastive_loss_weight * contrastive_loss
```

Initial contrastive defaults:

```text
contrastive_loss_weight = 0.10
contrastive_temperature = 0.07
```

Sweep values:

```text
0.05, 0.10, 0.25, 0.50
```

### 7.3 Sequence construction

Keep full-region sequences for extraction so timeline artifacts remain aligned
with the existing CRNN region chunks.

Training can later use a mixture:

```text
70% event-centered sequences
30% region-context sequences
```

Event-centered windows:

```text
pre_event_context_sec: 2.0
post_event_context_sec: 2.0
chunk_hop_sec: upstream CRNN hop, usually 0.25
```

Important distinction:

- Event-centered sequence construction can use all effective segmentation
  events.
- Contrastive labels use only human correction labels.

### 7.4 Artifacts

New artifact:

```text
retrieval_embeddings.parquet
```

Suggested schema mirrors `contextual_embeddings.parquet`:

```text
region_id
chunk_index_in_region
audio_file_id
start_timestamp
end_timestamp
tier
embedding
```

Per-k bundles are unchanged in shape:

```text
k<N>/decoded.parquet
k<N>/kmeans.joblib
k<N>/run_lengths.json
k<N>/overlay.parquet
k<N>/exemplars.json
k<N>/label_distribution.json
```

The semantic change is that `decoded.parquet` labels for retrieval-aware jobs
come from k-means over retrieval embeddings.

## 8. Database And Signature

Add retrieval-aware training config to `masked_transformer_jobs`. The exact
migration can be phased; the design expectation is that every training-affecting
field is part of `training_signature`, while `k_values` remains excluded.

Candidate new columns:

```text
retrieval_head_enabled boolean default false
retrieval_dim integer nullable
retrieval_hidden_dim integer nullable
retrieval_l2_normalize boolean default true

contrastive_loss_weight float default 0.0
contrastive_temperature float default 0.07
contrastive_label_source text default "none"
contrastive_min_events_per_label integer default 4
contrastive_min_regions_per_label integer default 2
require_cross_region_positive boolean default true
hard_negative_policy_json text nullable
related_label_policy_json text nullable

sequence_construction_mode text default "region"
event_centered_fraction float default 0.0
pre_event_context_sec float nullable
post_event_context_sec float nullable
```

Signature rules:

- Include every field above when present.
- Continue excluding `k_values`.
- Include upstream `continuous_embedding_job_id`.
- Include existing masked-model config fields.
- Include `event_classification_job_id` only if the training phase actually
  consumes it. For human-correction contrastive training it should be included
  because the available corrections are tied to the upstream segmentation and
  review state selected through the bound Classify workflow.

Backwards compatibility:

- Existing rows have `retrieval_head_enabled=false`.
- Existing jobs without `retrieval_embeddings.parquet` continue to serve their
  existing artifacts.
- NN report can accept an embedding-space argument and fall back to contextual
  embeddings when retrieval embeddings are absent.

## 9. Phased Implementation Roadmap

### Phase 0 - Retrieval Evaluation Harness

Purpose: make the Stage 0 report the official backend diagnostic surface before
changing training.

Scope:

- Move the reusable nearest-neighbor report logic out of
  `scripts/masked_transformer_nn_report.py` into a first-class source module,
  e.g. `src/humpback/sequence_models/retrieval_diagnostics.py`.
- Remove `scripts/masked_transformer_nn_report.py` after the source module and
  endpoint cover the workflow; diagnostics should not remain a standalone
  script entry point.
- Add an API endpoint in `src/humpback/api/routers/sequence_models.py`:

```text
POST /sequence-models/masked-transformers/{job_id}/nearest-neighbor-report
```

- Request body supports:
  - `k`
  - `embedding_space`: `contextual` or `retrieval`
  - `samples`
  - `topn`
  - `seed`
  - `retrieval_modes`
  - `embedding_variants`
  - `include_query_rows`
  - `include_neighbor_rows`
- Response returns structured JSON suitable for later frontend use:
  - job metadata
  - label coverage
  - aggregate metrics by retrieval mode and embedding variant
  - verdict counts
  - representative good/risky query summaries
  - optional query and neighbor detail rows
- Add embedding-space selection: contextual vs retrieval when available.
- Add event-level mean-pooled evaluation.
- Add retrieval modes:
  - unrestricted
  - exclude same event
  - exclude same event and same region
- Add human-correction-only label extraction helper.
- Report raw, centered, remove-PC, and whitened variants.
- Report label-specific same-label overlap.

Acceptance criteria:

- Nearest-neighbor diagnostics are invokable through the backend API without
  shelling out to a standalone script.
- Existing contextual report still works for old jobs.
- Report clearly distinguishes human correction labels from Classify model
  labels.
- Exclude-same-event and exclude-same-region modes are deterministic and use the
  same sampled query set across embedding variants.
- The endpoint returns 404 for missing jobs or unavailable k values, 409 for
  incomplete jobs or missing artifacts, and 422 for invalid report options.

Testing:

- Unit tests for the new `retrieval_diagnostics` module.
- API tests for the nearest-neighbor report endpoint.
- Unit tests for human-correction label extraction.
- Unit tests for neighbor exclusion modes.
- Unit tests for multi-label positive set construction.

### Phase 1 - Projection Head And Retrieval Embeddings

Purpose: change one training variable and switch downstream tokenization to the
retrieval space.

Scope:

- Add projection head to `MaskedTransformer`.
- Add config and signature fields for the retrieval head.
- Persist `retrieval_embeddings.parquet`.
- Switch per-k k-means tokenization to retrieval embeddings for new
  retrieval-head jobs.
- Keep masked loss only: `contrastive_loss_weight=0.0`.
- Update API schemas and create form fields only as needed to submit the new
  config.

Acceptance criteria:

- New retrieval-aware jobs write both contextual and retrieval embeddings.
- Per-k decoded tokens are fit from retrieval embeddings.
- Existing masked-transformer jobs remain readable.
- Stage 0/1 report can compare contextual vs retrieval embeddings for the same
  job.

Testing:

- Model forward shape includes retrieval output.
- Retrieval embeddings are L2-normalized.
- Worker persists retrieval artifact.
- Worker tokenization uses retrieval artifact for retrieval-head jobs.
- Training signature changes when retrieval-head config changes and does not
  change when only `k_values` changes.

### Phase 2 - Event-Centered Sequence Construction

Purpose: improve the masked objective's exposure to call morphology before
adding contrastive loss.

Scope:

- Add event-centered training window construction.
- Support region-only, event-centered-only, and mixed modes.
- Use effective event boundaries for window construction.
- Keep extraction over full region sequences for timeline alignment.
- Keep contrastive loss disabled.

Acceptance criteria:

- Event-centered training windows include configured pre/post context.
- Mixed mode respects `event_centered_fraction`.
- Full-region extraction still produces one retrieval embedding per upstream
  CRNN chunk.

Testing:

- Window builder handles short, long, and edge-near events.
- Mixed sampler is deterministic under seed.
- Worker artifact row counts still match upstream full-region chunk counts.

### Phase 3 - Human-Correction Supervised Contrastive Loss

Purpose: directly optimize event-level retrieval neighborhoods.

Scope:

- Add event-level mean pooling over hidden states.
- Apply projection head to pooled event hidden states.
- Build supervised-contrastive masks from human correction label sets.
- Prefer cross-region positives.
- Exclude rare labels from contrastive loss until they satisfy support
  thresholds.
- Track masked, contrastive, and total losses separately.

Acceptance criteria:

- Events without human correction labels remain in masked modeling but do not
  contribute to contrastive loss.
- Positive mask uses label-set intersection.
- Negative mask excludes related-label pairs.
- Batches with no valid positives fall back to masked loss only for that batch.

Testing:

- Contrastive loss produces finite values for multi-label events.
- Empty/rare-label batches are skipped safely.
- Same-label different-region positives are preferred when available.
- Loss curve artifact includes masked, contrastive, and total series.

### Phase 4 - Hard Negatives And Related-Label Policy

Purpose: reduce region leakage and local-context shortcuts after the basic
contrastive path is stable.

Scope:

- Add same-region different-label hard negatives.
- Add duration-matched different-label negatives.
- Add policy for related-label exclusions or down-weighting.
- Add diagnostics showing which negatives were used.

Acceptance criteria:

- Hard-negative policy is configurable and included in signature.
- Related-label pairs are not treated as full-strength negatives by default.
- Retrieval report shows whether same-region dominance decreases.

Testing:

- Hard-negative sampler returns expected same-region different-label examples.
- Duration matching obeys configured tolerance.
- Related-label exclusions remove pairs from the negative mask.

### Phase 5 - Sweep And Research Workflow

Purpose: support repeatable research comparisons without building a large UI
surface too early.

Scope:

- Add script helpers for lambda sweeps.
- Add script helpers for hard-negative policy sweeps.
- Produce a compact comparison markdown/CSV across jobs.
- Use Stage 0/1 metrics as the primary model-selection criteria.

Acceptance criteria:

- Sweep output ranks by cross-region same-human-label overlap.
- Sweep output includes raw retrieval, remove-PC, and whitened variants.
- Sweep output includes good/mixed/bad query counts.

Testing:

- Script tests can run on small synthetic or fixture jobs.
- Comparison output is deterministic under seed.

## 10. Evaluation Metrics

Primary:

```text
cross-region same human correction label overlap
```

Secondary:

```text
raw retrieval same-human-label overlap
whitened retrieval same-human-label overlap
same event rate
same region rate
similar duration rate
good / mixed / bad query counts
label-specific same-label overlap
random-pair cosine percentiles
```

Required retrieval modes:

```text
unrestricted
exclude same event
exclude same event and same region
```

Required embedding variants:

```text
raw_l2
centered_l2
remove_pc1
remove_pc3
remove_pc5
remove_pc10
whiten_pca
```

Stage 1 success should be judged by retrieval metrics, not training loss.

## 11. Risks

- Projection head may improve tokenization but not raw nearest-neighbor quality.
- Human correction labels may be sparse and uneven by label or region.
- Same-label events may still include multiple acoustic subtypes.
- Strong contrastive loss may damage motif-relevant near-miss relationships.
- Switching k-means to retrieval embeddings changes token semantics for new
  jobs, so reports must clearly record the embedding space used.
- Event-centered training windows could reduce bout-context modeling if the
  region-context mixture is too low.

## 12. Open Questions

1. What minimum human-corrected support should a label need before entering
   contrastive training?
2. Should the related-label exclusion list be hard-coded for Phase 3 or read
   from config immediately?
3. Should Phase 1 expose retrieval-head controls in the frontend, or use backend
   defaults and keep the create form simple?
4. Should retrieval-aware jobs display an explicit "embedding space:
   retrieval" badge on the detail page?
5. Should event-centered sequence construction include unlabeled effective
   events, or only human-labeled events, once contrastive training exists?

## 13. Recommended First Cut

Implement Phase 0 first, then Phase 1.

This gives a stable measurement harness before changing the model, and then
tests the smallest meaningful training change:

```text
vanilla transformer
plus projection head
masked loss only
retrieval embeddings persisted
k-means tokenization from retrieval embeddings
nearest-neighbor report compares contextual vs retrieval spaces
```

If raw retrieval embeddings move closer to whitened contextual performance,
proceed to event-centered training. If projection-only does not help, keep the
artifact/reporting work and move directly to the contrastive phase.
