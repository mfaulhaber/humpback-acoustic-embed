# Autoresearch Row-ID Detection Supervision — Design Spec

**Date:** 2026-04-03
**Status:** Approved

## Goal

Make `scripts/autoresearch` consume current production detection-job artifacts and manual detection-job supervision correctly, so experiments can use live hard negatives and positives without failing on schema drift or silently mislabeling rows.

## Problem Context

The 2026-04-01 autoresearch detection-job design assumed two things that are no longer true in production:

1. Detection embeddings still use the legacy schema `(filename, start_sec, end_sec, embedding, confidence)`.
2. Detection-job supervision can be inferred from the row-store binary columns alone (`humpback`, `orca`, `ship`, `background`).

The production audit on 2026-04-03 of detection jobs `23a1f7ca-7777-4f2b-83bf-8eb4ccc9fef3` and `2a5f51f3-b91d-470e-a92e-4900ebedb97d` showed:

- `detection_embeddings.parquet` now uses the canonical row-id schema `(row_id, embedding, confidence)`.
- `_collect_detection_examples()` currently fails on these files because it requires a `filename` column.
- Manual hard negatives added in the vocalization labeling workflow live in `vocalization_labels` as `label = "(Negative)"`, not in the row-store binary columns.
- Manual vocalization positives also exist on rows that remain binary-unlabeled in the row store.
- On job `23a1...`, a row-store-only import would misclassify dozens of manually labeled vocalization positives as unlabeled hard negatives inside the default score range.
- On job `2a5f...`, all detection embedding confidence values are null, so unlabeled score-band mining cannot recover the manual hard negatives that live only in `vocalization_labels`.

The vocalization training-dataset pipeline already treats `(Negative)` as explicit negative supervision and keys detection rows by `row_id`. Autoresearch should align with that production semantics rather than maintain a separate, stale interpretation.

## Architecture Decisions

### Canonical detection schema: row_id-based embeddings

Autoresearch will treat `(row_id, embedding, confidence)` as the primary detection-embedding schema.

Legacy detection embeddings with `(filename, start_sec, end_sec, embedding, confidence)` remain supported for backward compatibility, but live production behavior is defined by `row_id`.

### Unified detection supervision sources

Detection-job examples will be labeled by combining two sources:

1. `vocalization_labels` keyed by `(detection_job_id, row_id)`
2. `detection_rows.parquet` keyed by `row_id`

The manifest generator will query both for every requested detection job.

### Label precedence and conflict handling

For each detection row, autoresearch will classify the row using this precedence:

1. Any non-`"(Negative)"` vocalization label => positive (`label = 1`)
2. `"(Negative)"` vocalization label with no contradictory positive vocalization labels => negative (`label = 0`, `negative_group = "vocalization_negative"`)
3. Row-store `humpback=1` or `orca=1` with no contradictory vocalization-negative label => positive
4. Row-store `ship=1` or `background=1` with no contradictory vocalization-positive label => negative with semantic group
5. Rows unlabeled by both systems and with non-null confidence inside `--score-range` => negative with score-band group
6. Everything else => excluded from the manifest

Conflicting supervision is never resolved silently. If a row is positive in one system and negative in the other, the generator skips that row and records it in manifest summary counts.

This keeps manual vocalization labels authoritative without hiding data-quality issues.

### Row-id detection examples carry stable row identity

Detection-job manifest examples gain an optional `row_id` field. Row-id detection examples use `row_id` for lookup; embedding-set examples continue to use `row_index`.

Example IDs for row-id detection examples become:

`det{job_id[:8]}_{row_id}`

This keeps IDs stable across manifest regeneration and traceable back to the live row store.

### Split grouping for row-id detection jobs

Live row-id detection embeddings do not retain filename-level provenance, so the old “split by filename” rule is no longer sufficient.

For row-id detection jobs, the manifest generator will derive the split group from the row-store `start_utc`, bucketed to the UTC hour and namespaced by detection job:

`det{job_id[:8]}:{YYYY-MM-DDTHH}`

Why hourly buckets:

- Whole-job grouping would collapse a day-long hydrophone job into one split.
- Per-row grouping would leak near-duplicate temporal neighbors across splits.
- Hourly grouping preserves local temporal coherence while producing enough groups for train/val/test splits on day-scale jobs.

For legacy filename-based detection embeddings, split grouping remains filename-based.

The manifest will keep using `audio_file_id` as the split-group field for backward compatibility, but for row-id detection sources it contains the synthetic hourly group key rather than a database `audio_files.id`.

### Confidence-aware unlabeled hard negatives

Unlabeled score-band negatives are only created when the detection embedding has a non-null confidence.

Rows with null confidence remain eligible if they have explicit positive or negative supervision, but they are excluded from unlabeled hard-negative mining.

This is required for production jobs like `2a5f...`, whose detection embeddings are row-id aligned but have null confidence throughout.

### Context pooling fallback for row-id detections

Row-id detection embeddings do not preserve a trustworthy neighbor structure for autoresearch context pooling. Sync jobs can append newly generated rows, and the Parquet row order is no longer a reliable temporal adjacency contract.

For row-id detection examples:

- `context_pooling = "center"` behaves normally
- `context_pooling = "mean3"` and `"max3"` fall back to center-only for those examples

Embedding sets and legacy filename-based detection embeddings keep their current pooling behavior.

This keeps the search space stable without inventing fake neighbors.

### Manifest provenance should explain how each detection row entered the experiment

Detection-job manifest examples gain a `label_source` field with one of:

- `vocalization_positive`
- `vocalization_negative`
- `binary_positive`
- `ship`
- `background`
- `score_band`

Manifest metadata also gains per-job summary counts for:

- included positives
- included negatives by source
- unlabeled score-band negatives
- skipped conflicts
- skipped null-confidence unlabeled rows
- skipped rows missing embeddings

This makes production experiment setup auditable before any search run starts.

## Manifest Generator Changes

### Detection job processing flow

For each detection job:

1. Verify `has_positive_labels = True`
2. Read `detection_rows.parquet` and build a `row_id` index with `start_utc`, binary labels, and split-group data
3. Read `detection_embeddings.parquet`
4. If the Parquet uses `row_id`, join by `row_id`
5. If the Parquet uses legacy filename/start/end fields, continue using the legacy path
6. Query `vocalization_labels` for the job and group labels by `row_id`
7. Apply label precedence and conflict skipping
8. Emit detection-job examples plus per-job metadata counts

### Example shape

Row-id detection example:

```json
{
  "id": "det23a1f7ca_218edcaa-8b7c-47db-9c41-989da0e1fdb5",
  "split": "val",
  "label": 0,
  "source_type": "detection_job",
  "parquet_path": "/abs/path/to/detection_embeddings.parquet",
  "row_id": "218edcaa-8b7c-47db-9c41-989da0e1fdb5",
  "audio_file_id": "det23a1f7ca:2021-11-19T08",
  "negative_group": "vocalization_negative",
  "label_source": "vocalization_negative",
  "detection_confidence": 0.998105,
  "start_utc": 1637311680.0
}
```

Embedding-set examples keep the existing `row_index`-based shape.

## Train/Eval Pipeline Changes

### Parquet loading

`_load_parquet_cache()` will support three formats:

- embedding-set format: `row_index`
- canonical detection format: `row_id`
- legacy detection format: `filename`

The cache must preserve whichever row identity the file uses so `_build_embedding_lookup()` can resolve examples by `row_index` or `row_id`.

### Embedding lookup

`_build_embedding_lookup()` will:

- use `row_index` for embedding-set and legacy detection examples
- use `row_id` for canonical detection examples

Examples missing from the embedding file are skipped and counted by manifest generation whenever possible.

### Context pooling

- embedding-set examples: unchanged
- legacy detection examples: unchanged same-file neighbor pooling
- row-id detection examples: center-only fallback for all pooling modes

## Documentation Changes

Update `scripts/autoresearch/README.md` so it matches the live production behavior:

- current detection embedding schema is row-id-based
- manual `(Negative)` labels from the vocalization workflow are first-class hard negatives
- row-store labels are fallback supervision, not the only supervision
- unlabeled hard negatives require non-null confidence
- row-id detection splits are grouped by hourly UTC buckets

## Testing Strategy

### Unit tests

- row-id detection embeddings load correctly
- manifest generation pulls positives and `(Negative)` rows from `vocalization_labels`
- conflicting row-store vs vocalization labels are skipped and counted
- unlabeled rows with null confidence are excluded from score-band hard negatives
- row-id detection split groups are derived from `start_utc` hourly buckets
- legacy filename-based detection embeddings still work
- row-id detection examples use center-only pooling fallback

### Integration test

- mixed-source autoresearch run with embedding sets plus row-id detection-job examples
- includes vocalization positives, vocalization negatives, binary fallback labels, and score-band negatives
- verifies grouped metrics and output artifacts still serialize normally

## Out of Scope

- Changing the search objective
- Adding a new calibration strategy
- Implementing per-trial `hard_negative_fraction` subsampling in phase-2 replay
- Automatically transferring the winning config back into platform classifier training
