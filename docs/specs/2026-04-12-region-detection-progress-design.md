# Region Detection Progress Tracking & Chunk Artifacts

**Date:** 2026-04-12
**Status:** Approved

## Problem

The region detection worker for hydrophone jobs runs end-to-end with no intermediate
progress visibility. A 24-hour detection job produces no output until completion,
cannot be resumed after a crash, and provides no timing data in logs.

## Scope

Hydrophone streaming path only. File-based region detection continues as a single
atomic operation (see ADR-051).

## Design

### 1. Chunk Artifact Layout

Each hydrophone region detection job writes intermediate artifacts to its job directory:

```
data/call_parsing/region_jobs/{job_id}/
├── manifest.json
├── chunks/
│   ├── 0000.parquet
│   ├── 0001.parquet
│   └── ...
├── trace.parquet          # Final merged (completion only)
└── regions.parquet        # Final regions (completion only)
```

### 2. Manifest

Written at job start with all chunks in `pending` status. Updated per chunk on
completion. Structure:

```json
{
  "job_id": "...",
  "config": {
    "stream_chunk_sec": 1800,
    "window_size_seconds": 5.0,
    "hop_seconds": 1.0
  },
  "chunks": [
    {
      "index": 0,
      "start_sec": 0.0,
      "end_sec": 1800.0,
      "status": "complete",
      "completed_at": "2026-04-12T03:14:22Z",
      "trace_rows": 1795,
      "elapsed_sec": 42.3
    },
    {
      "index": 1,
      "start_sec": 1800.0,
      "end_sec": 3600.0,
      "status": "pending",
      "completed_at": null,
      "trace_rows": null,
      "elapsed_sec": null
    }
  ]
}
```

### 3. Database Schema

Two new nullable integer columns on `RegionDetectionJob`:

| Column | Type | Default | Description |
|--------|------|---------|-------------|
| `chunks_total` | `Optional[int]` | `None` | Total chunks planned, set when manifest is written |
| `chunks_completed` | `Optional[int]` | `None` | Incremented after each chunk parquet is written |

Both remain `None` for file-based jobs. `region_count` and `trace_row_count`
remain as-is, populated only at completion.

Alembic migration with `batch_alter_table` for SQLite compatibility.

API response schema (`RegionDetectionJobSummary`) adds both fields.

### 4. Worker Resume Logic

**Cold start (no manifest):**
1. Compute chunk edges via `_aligned_chunk_edges`
2. Write manifest with all chunks `pending`
3. Set `chunks_total` in DB
4. Begin processing from chunk 0

**Resume (manifest exists):**
1. Read existing manifest
2. For each chunk marked `complete`, verify its parquet file exists on disk
   - If parquet missing for a "complete" chunk, reset that chunk to `pending`
3. Skip all verified-complete chunks
4. Resume from first `pending` chunk
5. Update `chunks_completed` in DB to match verified count

**Chunk completion (per iteration):**
1. Write chunk parquet to `chunks/{index:04d}.parquet.tmp`
2. Atomic rename `.tmp` to `.parquet`
3. Update manifest entry: status, completed_at, trace_rows, elapsed_sec
4. Increment `chunks_completed` in DB

**Final merge (all chunks complete):**
1. Read all chunk parquets in order, concatenate trace rows
2. Run hysteresis merge to extract regions
3. Write `trace.parquet` and `regions.parquet` (same atomic-tmp pattern)
4. Set `trace_row_count`, `region_count`, `completed_at`, status `complete`

**Crash semantics:** On exception, the worker marks the job `failed` with
`error_message`. A failed job can be re-queued and will resume from the last
completed chunk. Partial `.tmp` files from the interrupted chunk are cleaned up
on failure or ignored on resume (only `.parquet` files count as complete).

### 5. Worker Logging

Structured INFO-level log messages per chunk:

```
region_detection | job=abc123 | start | chunks=48 | range=86400.0s | hydrophone=orcasound_lab
region_detection | job=abc123 | chunk 1/48 | 0.0s-1800.0s | fetching audio
region_detection | job=abc123 | chunk 1/48 | scored 1795 windows | 42.3s (2.4s/min audio)
region_detection | job=abc123 | chunk 12/48 | resumed from manifest (11 chunks cached)
region_detection | job=abc123 | merge | 86160 trace rows -> 47 regions | 0.8s
region_detection | job=abc123 | complete | 48 chunks in 1843.2s | 47 regions
```

Per-chunk granularity only. No per-batch logging within a chunk.

### 6. Benchmark Script

`scripts/benchmark_region_detection.py` — standalone timing tool.

**Arguments:** hydrophone ID, time range (default 10 minutes), model config ID,
classifier model ID.

**Behavior:**
1. Fetch audio via the same `iter_audio_chunks` path the worker uses
2. Run `score_audio_windows` on fetched audio
3. Report timing breakdown: audio fetch, scoring, total, per-minute rate
4. Extrapolate to 24h estimate

**Example output:**
```
Benchmark: orcasound_lab | 2021-10-31T00:00Z - 2021-10-31T00:10Z (600s)
  Audio fetch:    8.2s
  Scoring:       14.1s  (1795 windows)
  Total:         22.3s  (2.23s per minute of audio)

Extrapolated 24h estimate:
  Wall time:     ~53.5 min
  Chunks (30m):  48
  Per chunk:     ~66.8s
```

Does not write DB rows or job artifacts. Imports the same scoring functions the
worker uses.

## ADR-051: Chunk Artifacts Apply to Hydrophone Path Only

**Context:** The region detection worker has two audio source paths — hydrophone
streaming (long-duration, chunked) and file-based (short, loaded in one shot).

**Decision:** The chunk artifact system (manifest, per-chunk parquets, resume
logic, progress columns) applies only to the hydrophone streaming path. File-based
region detection continues as a single atomic operation.

**Consequences:** File-based jobs cannot be paused/resumed. Progress columns
remain null for file-based jobs. If needed in the future, the file path can be
synthetically chunked without affecting the hydrophone path.
