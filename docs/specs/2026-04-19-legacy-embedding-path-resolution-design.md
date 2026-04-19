# Legacy Embedding Path Resolution

**Date:** 2026-04-19
**Status:** Draft

## Problem

Hydrophone detection jobs create embeddings inline during the detection run, writing them to `detections/<id>/detection_embeddings.parquet` (the legacy path). The classifier training flow checks a different location — the model-versioned path at `detections/<id>/embeddings/<model_version>/detection_embeddings.parquet` — and requires a `detection_embedding_jobs` DB row with `status=complete`.

This means the Embedding column on the training page shows "Missing" for hydrophone jobs that already have valid embeddings, and users must click "Embed" to re-encode identical vectors to a different path. This wastes compute and is confusing.

## Embedding State Table

| State | Legacy exists? | Versioned exists? | Same model? | Status |
|-------|---------------|-------------------|-------------|--------|
| A | Yes | No | Yes | Exists — usable, copy on train |
| B | Yes | No | No | Missing — need embed for different model |
| C | No | Yes (complete) | — | Exists |
| D | No | No | — | Missing |

Key insight: relabeling detection windows in the timeline viewer modifies label fields on existing rows but does not add or remove rows from the row store. Embeddings are derived from audio, not labels, so they remain valid after relabeling.

## Design

### Embedding Status API Enhancement

Modify `GET /classifier/detection-embedding-jobs` in `embeddings.py`. When no `detection_embedding_jobs` DB row exists for a `(detection_job_id, model_version)` pair:

1. Resolve the detection job's source classifier model version
2. If the source model version matches the requested `model_version`, check if the legacy file exists at `detections/<id>/detection_embeddings.parquet`
3. If legacy file exists and model matches → return `status="complete"` with the legacy file's row count
4. Otherwise → return `status="not_started"` (unchanged behavior)

When a DB row does exist, return its status as-is (no change).

The API contract stays identical — no new status values, no schema changes. Consumers already handle `complete` and `not_started`.

### Training Job — Legacy Path Copy-on-Submit

Modify `create_training_job_from_detection_manifest()` in `training.py`. When checking for embeddings:

1. Check model-versioned path first — if exists, proceed as today
2. If not, resolve the detection job's source classifier model version
3. If source model matches the requested `model_version`, check legacy path
4. If legacy file exists, copy it to the model-versioned path, then proceed normally
5. If neither exists, raise the same ValueError as today

The copy is a single file operation on small parquet files (typically a few MB). After the copy, everything downstream is unchanged.

### File Changes

| File | Change |
|------|--------|
| `src/humpback/api/routers/classifier/embeddings.py` | Enhance `list_detection_embedding_jobs` to check legacy path when no DB row exists |
| `src/humpback/services/classifier_service/training.py` | Add legacy-path fallback with copy in `create_training_job_from_detection_manifest` |
| `tests/integration/test_classifier_api.py` | Tests for legacy-path detection in status API and copy-on-submit |

### What Doesn't Change

- No new API endpoints or response schemas
- No database migration
- No frontend changes — the Embedding column already handles `complete` correctly
- No changes to the re-embed worker or hydrophone worker
- The `generate-embeddings` endpoint still works for model-mismatch cases (state B)
