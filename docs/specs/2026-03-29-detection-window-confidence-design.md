# Detection Window Confidence in Embeddings Pipeline — Design Spec

## Problem

The vocalization labeling workspace needs to sort inference results by detection
confidence for detection job sources. Per-window classifier confidence scores are
computed during detection (`pipeline.predict_proba()`) but discarded when writing
`detection_embeddings.parquet` — only the peak window's embedding vector is kept.

An earlier attempt to recover confidence via interval joins between
`detection_rows.parquet` and `detection_embeddings.parquet` at the vocalization API
layer was fragile and coupled detection internals into the vocalization pipeline.

## Decision

Store the per-window classifier confidence directly in `detection_embeddings.parquet`.
This is the source of record for vocalization inference on detection jobs, so confidence
flows through the pipeline without cross-parquet joins.

The confidence value is the raw `predict_proba` score of the single peak-confidence
window whose embedding was selected for each detection event. This value is already
computed at the point where embedding records are built in `run_detection()`.

## Design

### Embedding Writer (detector.py)

Two changes in `run_detection()`:

1. **Capture confidence in embedding record** — At the point where `best_conf` is
   already computed during peak-window selection (~line 546), add it to the embedding
   record dict: `"confidence": best_conf`.

2. **Extend parquet schema** — `write_detection_embeddings()` adds a
   `("confidence", pa.float32())` column. Records missing the key get `None`
   (defensive, shouldn't happen for new jobs).

The hydrophone path in `classifier_worker.py` accumulates embedding records from the
same `run_detection()` call, so it picks up confidence automatically with no additional
changes.

### Vocalization Inference Pipeline

**`_load_source_embeddings` (vocalization_worker.py)** — When loading from a detection
job, read the `confidence` column if present, return it as `list[float] | None`. Return
`None` for old jobs without the column.

**`run_inference` (vocalization_inference.py)** — Accept optional
`confidences: list[float] | None` parameter. When provided, write it as a `confidence`
float32 column in the predictions parquet alongside existing columns.

**`read_predictions` (vocalization_inference.py)** — Read the `confidence` column from
predictions parquet when present, include it in the returned row dicts.

### API & Schema

**`VocalizationPredictionRow` schema (schemas/vocalization.py)** — Add optional field:
`confidence: float | None = None`. Present for detection job sources with the confidence
column, absent for embedding set sources.

**`GET /inference-jobs/{job_id}/results` endpoint (api/routers/vocalization.py)** — Add
`sort: str | None = Query(None)` parameter. When `sort=confidence_desc`, sort the full
result list by `confidence` descending (nulls last) before applying offset/limit. All
other sort modes remain client-side.

The endpoint already reads the full predictions list before slicing, so this is a
`sorted()` call inserted between read and slice.

### Frontend

**`VocClassifierPredictionRow` type (types.ts)** — Add optional `confidence?: number`.

**API client (client.ts)** — Pass `sort` query parameter when fetching inference results.

**`LabelingWorkspace.tsx`** — Add `confidence_desc` sort mode:
- Pass `sort=confidence_desc` to the API call so pagination returns
  highest-confidence items first
- Skip client-side re-sorting when using server-side sort
- Only show the option when source is a detection job
- Default sort for detection job sources becomes `confidence_desc`
- Conditionally show "Detection Confidence" in the sort dropdown when `confidence`
  is present on the first loaded row

Existing `score_desc`, `uncertainty`, and `chronological` modes remain client-side.

### Backward Compatibility

**Old detection jobs (no confidence column in parquet):**
- `_load_source_embeddings` returns `None` for confidences
- `run_inference` omits the column from predictions parquet
- `read_predictions` returns rows without `confidence`
- API returns `confidence: null` in response
- Frontend hides the confidence sort option

**Re-generated embeddings:** When `detection_embedding_worker` re-runs for an old job,
confidence will be populated going forward since `run_detection()` produces it fresh.

**Mixed sources:** Vocalization inference supports one source at a time (single detection
job or single embedding set), so there's no mixed-confidence scenario.

**`sort=confidence_desc` with no confidence data:** Server-side sort treats `None` as
lowest, effectively preserving insertion order. Frontend won't offer this mode anyway.

**`read_detection_embedding` (single-row reader for similarity search):** Does not need
confidence — leave unchanged.

### Testing

**Unit tests:**
- `write_detection_embeddings` with confidence: verify parquet schema includes
  `confidence` float32, round-trip read confirms values
- `write_detection_embeddings` backward compat: records without `confidence` key
  write `None`
- `run_inference` with and without `confidences` parameter: verify predictions parquet
  includes/excludes the column
- `read_predictions` with and without confidence column in parquet
- Server-side sort: mock prediction list with varied confidence values, verify
  `sort=confidence_desc` returns descending order with nulls last

**Integration tests:**
- API endpoint with `sort=confidence_desc`: create inference job with known confidence
  values, verify paginated results arrive in confidence order
- API endpoint with no sort param: verify existing behavior unchanged
- Old-format predictions parquet (no confidence column): verify endpoint returns
  `confidence: null` gracefully

## Files Changed

| File | Change |
|------|--------|
| `src/humpback/classifier/detector.py` | Add `confidence` to embedding record dict + parquet schema |
| `src/humpback/workers/vocalization_worker.py` | Read confidence from detection embeddings, pass to `run_inference` |
| `src/humpback/classifier/vocalization_inference.py` | Accept/write/read confidence in `run_inference` and `read_predictions` |
| `src/humpback/schemas/vocalization.py` | Add `confidence` field to `VocalizationPredictionRow` |
| `src/humpback/api/routers/vocalization.py` | Add `sort` parameter, server-side confidence sort |
| `frontend/src/api/types.ts` | Add `confidence` to `VocClassifierPredictionRow` |
| `frontend/src/api/client.ts` | Pass `sort` query parameter |
| `frontend/src/components/vocalization/LabelingWorkspace.tsx` | Add `confidence_desc` sort mode, default for detection sources |
| `tests/` | Unit + integration tests for confidence threading and sort |
