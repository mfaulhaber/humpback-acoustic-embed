# Autoresearch Detection Job Integration — Design Spec

**Date:** 2026-04-01
**Status:** Approved

## Goal

Extend the autoresearch manifest generator and train/eval pipeline to incorporate data from detection jobs — both human-labeled windows (as positives/negatives) and unlabeled high-confidence windows (as hard negatives with score-band grouping). This addresses the gap where classifier metrics look good on curated embedding sets but high-confidence false positives emerge in real detection deployments.

## Problem Context

The FP analysis report (2026-04-01) on LR-v10/Orcasound Lab shows:
- ~90% of detection windows are unlabeled (likely FPs)
- Even the 0.995–1.000 score band has only ~78% precision
- Training negatives are "easy" (acoustically distinct); deployment FPs are "hard" (overlapping frequency content)

Detection jobs contain exactly the data needed to close this gap: human-labeled true positives/negatives plus a large pool of acoustically confusing hard negatives.

## Architecture Decisions

### Data sources: labeled + unlabeled from detection jobs

Labeled detection windows (humpback=1 → positive, ship/background=1 → negative) serve as deployment-realistic ground truth. Unlabeled windows within a configurable score range serve as hard negatives. Both go into the same manifest alongside embedding set data.

### Only labeled detection jobs

Detection jobs must have `has_positive_labels=True` — a human actually did labeling work. This avoids pulling from unreviewed jobs where "unlabeled" doesn't mean "negative."

### Unified manifest with source_type field

All examples live in one `examples` array. A `source_type` field (`"embedding_set"` or `"detection_job"`) tells the pipeline which Parquet format to expect. Backward compatible — manifests without the field default to `"embedding_set"`.

### Split by filename

Detection windows are grouped by their `filename` field (the audio file they came from) and assigned to train/val/test using the same seeded shuffle as embedding set audio files. This prevents leakage and distributes detection data proportionally.

### Auto-populated negative_group from labels + score bands

- `ship=1` → `"ship"`
- `background=1` → `"background"`
- Unlabeled, score in `[0.50, 0.90)` → `"det_0.50_0.90"`
- Unlabeled, score in `[0.90, 0.95)` → `"det_0.90_0.95"`
- Unlabeled, score in `[0.95, 0.99)` → `"det_0.95_0.99"`
- Unlabeled, score in `[0.99, 0.995)` → `"det_0.99_0.995"`

Score band boundaries align with the precision breakpoints from the FP analysis report.

### Schema-aware Parquet loading

`_load_parquet_cache` auto-detects the Parquet format by checking for a `row_index` column (embedding set format) vs `filename` column (detection embeddings format). Detection files use positional row index (0, 1, 2, ...).

### Context pooling for detection windows

Context pooling falls back to center-only when neighbor rows come from a different filename. The existing graceful-degradation for missing neighbors handles this.

## Manifest Generator Changes

### New CLI flags

```bash
uv run scripts/autoresearch/generate_manifest.py \
  --job-ids <training-job-ids> \
  --detection-job-ids <detection-job-ids> \
  --score-range 0.5,0.995 \
  --output data_manifest.json
```

- `--detection-job-ids`: comma-separated detection job UUIDs
- `--score-range`: min,max confidence for unlabeled hard negative mining (default: 0.5,0.995)
- Both `--job-ids` and `--detection-job-ids` are optional but at least one must be provided

### Processing per detection job

1. Verify `has_positive_labels=True` in database
2. Read `detection_embeddings.parquet` (filename, start_sec, end_sec, embedding, confidence)
3. Read `detection_rows.parquet` (row store with label columns)
4. Match rows between the two files by positional index (both are ordered the same way — the detection embeddings Parquet is written during detection in the same file/window order as the row store)
5. Classify each window based on label columns and score range
6. Generate manifest examples with `source_type: "detection_job"`

### ID scheme

`det{job_id[:8]}_row{index}` — first 8 chars of detection job UUID + positional index.

### Example format

```json
{
  "id": "detabc12345_row17",
  "split": "train",
  "label": 0,
  "source_type": "detection_job",
  "parquet_path": "/path/to/detection_embeddings.parquet",
  "row_index": 17,
  "audio_file_id": "recording_2021-11-01_0400.flac",
  "negative_group": "det_0.95_0.99",
  "detection_confidence": 0.973
}
```

## Train/Eval Pipeline Changes

### Parquet loading

`_load_parquet_cache` checks for `row_index` column presence:
- Present → embedding set format: load `(row_indices, embeddings)` as today
- Absent → detection format: read `embedding` column, generate positional indices `[0, 1, 2, ...]`

### Context pooling

For detection-sourced windows, neighbor lookup uses positional row index ± 1 within the same Parquet file. If the neighbor row's `filename` differs from the current row's `filename`, the neighbor is skipped (falls back to center). This requires storing the `filename` column in the Parquet cache for detection files.

### Everything else unchanged

Feature transforms, classifiers, calibration, metrics, objectives, and the search loop operate on `{id: vector}` lookups and are source-agnostic.

## Metadata in manifest

```json
{
  "metadata": {
    "created_at": "...",
    "source_job_ids": ["a7f9..."],
    "positive_embedding_set_ids": [...],
    "negative_embedding_set_ids": [...],
    "detection_job_ids": ["abc1...", "def4..."],
    "score_range": [0.5, 0.995],
    "split_strategy": "by_audio_file"
  }
}
```

## Testing Strategy

### Unit tests

- Manifest generation with detection jobs: mock DB, synthetic detection Parquets, verify label classification, score-band grouping, score-range filtering
- Rejection of detection jobs without `has_positive_labels=True`
- Detection Parquet auto-detection in `_load_parquet_cache`
- Context pooling with mixed sources (cross-file fallback)

### Integration test

- End-to-end with mixed sources: embedding set Parquet + detection embeddings Parquet, unified manifest, 3 search trials, verify grouped metrics include detection score bands
