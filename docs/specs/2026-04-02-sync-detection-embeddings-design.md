# Sync Detection Embeddings Design

**Date:** 2026-04-02
**Status:** Approved

---

## Problem

Detection jobs maintain two independent data stores that can drift apart:

1. **Row store** (`detection_row_store.parquet`) — written by the timeline viewer when users add, move, or delete detection rows. Also written by the original detection run.
2. **Embeddings store** (`detection_embeddings.parquet`) — written only during the original detection run (or a full re-generation). Read by vocalization inference.

Timeline edits (add, move, delete) modify the row store but leave the embeddings store stale:

- **Added rows** have no embedding vector at all — invisible to vocalization inference.
- **Moved rows** have an embedding at the old position but none at the new position. The old embedding becomes orphaned.
- **Deleted rows** leave orphaned embeddings that vocalization inference still scores.

## Solution

A **diff-and-patch** sync operation on the timeline viewer that compares the row store against the embeddings store, generates embeddings for missing rows, and removes orphaned embeddings.

---

## Core Mechanism

### Diff Logic

Match rows by `(start_utc, end_utc)` tuples, producing three sets:

- **Missing**: rows in the row store with no matching embedding — generate embedding.
- **Orphaned**: embeddings with no matching row store entry — remove.
- **Matched**: present in both — keep unchanged.

Embedding timestamps are computed from the embeddings parquet's `filename` + `start_sec`/`end_sec` fields via `_file_base_epoch()` (which parses `YYYYMMDDTHHMMSSZ` timestamps from filenames). This works for both local detection jobs (real filenames) and hydrophone jobs (synthetic `YYYYMMDDTHHMMSSZ.wav` filenames).

Timestamp comparison uses a tolerance of 0.5 seconds to account for floating-point representation differences.

### Audio Resolution

For each missing row, the sync loads the raw audio covering the 5-second window, extracts features, and runs the embedding model.

**Local detection jobs** (`audio_folder` set): List audio files in the folder, parse timestamps via `_file_base_epoch()`, find the file whose time range covers the row's `start_utc`. Compute `offset_sec = start_utc - file_base_epoch`. Decode audio, extract the window, generate features, embed.

**Hydrophone detection jobs** (`hydrophone_id` set): Reconstruct the `ArchiveProvider` via `build_archive_detection_provider()` using the job's `hydrophone_id` and cache config. Call `iter_audio_chunks(provider, start_utc, start_utc + window_size)` to get decoded audio for the target window. With warm local caches (populated during the original detection run), this reads from disk with no network I/O. Works for both Orcasound HLS and NOAA GCS providers transparently.

**Edge case — audio unavailable**: If the covering audio file can't be found (deleted, cache evicted), the row is skipped and recorded in the sync summary with a reason.

### Embedding Generation

For each missing row's audio window:
1. Load the embedding model from the detection job's `classifier_model_id` (via `get_model_by_version` from worker model cache).
2. Extract features — logmel spectrogram or waveform pass-through based on model's `input_format`.
3. Run `model.embed()` to produce the embedding vector.
4. Record with `confidence: null` (no detector score exists for manually-added/moved rows).

### Output

Rewrite the embeddings parquet with orphaned entries removed and new entries added. Store a result summary: `{added: N, removed: M, unchanged: K, skipped: L, skipped_reasons: [...]}`.

---

## Data Model Changes

Add columns to `DetectionEmbeddingJob`:

- `mode: str | None` — `"full"` or `"sync"`. Null treated as `"full"` for legacy rows.
- `result_summary: str | None` — JSON string. For sync: `{added, removed, unchanged, skipped, skipped_reasons}`. For full: `{total}`.

Requires one Alembic migration using `op.batch_alter_table()` for SQLite compatibility.

---

## API Changes

### Modified Endpoints

**`POST /classifier/detection-jobs/{job_id}/generate-embeddings`**
- Add `mode` query parameter: `"full"` (default) or `"sync"`.
- `mode=full`: existing behavior — rejects with 409 if embeddings already exist.
- `mode=sync`: requires embeddings to already exist (rejects with 400 if not). Creates a `DetectionEmbeddingJob` with `mode="sync"`.

**`GET /classifier/detection-jobs/{job_id}/embedding-status`**
- Add `sync_needed: bool` to `EmbeddingStatusResponse` — true when the row store has entries without matching embeddings or there are orphaned embeddings.

### New Endpoints

**`GET /classifier/embedding-jobs`**
- Paginated list of all `DetectionEmbeddingJob` records, newest first.
- Includes parent detection job context: `hydrophone_name`, `audio_folder` basename.
- Query params: `offset`, `limit`.

---

## Worker Changes

Extend `detection_embedding_worker.py` to handle `mode="sync"`:

1. Load the detection job and its classifier model.
2. Run the diff logic against row store and embeddings parquet.
3. For each missing row, resolve audio and generate embedding.
4. Rewrite the embeddings parquet.
5. Store `result_summary` JSON on the job.

The sync reuses the model cache, audio decode, feature extraction, and embedding pipeline from the existing detection infrastructure. No new model loading code needed.

---

## Frontend Changes

### Timeline Viewer — Sync Embeddings Button

Add a "Sync Embeddings" button to the timeline header area. Visible when:
- The detection job has embeddings (`has_embeddings` is true).
- `sync_needed` is true.

When clicked, calls `POST .../generate-embeddings?mode=sync`. Polls for completion using the existing `useEmbeddingGenerationStatus` hook. On completion, shows the summary (e.g., "2 added, 1 removed").

### New Classifier/Embeddings Page

**Route:** `/app/classifier/embeddings`

Table view of all embedding jobs with columns:
- **Status** — badge (queued/running/complete/failed)
- **Detection Job** — linked, shows hydrophone name or folder basename
- **Mode** — Full / Sync
- **Progress** — current/total for running jobs, sync summary for completed sync jobs
- **Created** — timestamp
- **Duration** — computed from created_at to updated_at when complete

Active/running jobs show a progress indicator. Failed jobs show error message. Completed sync jobs show the added/removed/unchanged/skipped breakdown.

Navigation: add "Embeddings" link to the Classifier section in the app sidebar/nav.

---

## Testing

- **Diff logic unit tests**: Verify correct missing/orphaned/matched classification. Edge cases: empty stores, all matched, all orphaned, timestamp tolerance.
- **Audio resolution unit tests**: Local job — correct file selection and offset. Hydrophone job — provider construction and iter_audio_chunks call.
- **Sync worker integration test**: End-to-end with fake model — create detection job, add/move/delete rows, run sync, verify embeddings parquet.
- **API tests**: `mode=sync` rejects without existing embeddings. `mode=full` rejects when embeddings exist. `sync_needed` flag correctness.
- **Frontend**: Playwright test for Sync button visibility and Embeddings page table.
