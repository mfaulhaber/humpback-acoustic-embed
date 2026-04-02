# Vocalization Label Timestamp Drift Report

**Date:** 2026-04-02
**Detection Job:** `2a5f51f3-b91d-470e-a92e-4900ebedb97d` (Orcasound Lab, 2021-11-01 00:00 — 2021-11-02 00:00 UTC)

## Summary

Vocalization labels created from inference results can have timestamps that diverge from the current detection row store after timeline editing. This causes false-positive orphan warnings in the labeling workspace. A partial fix (tolerance-based matching in the orphan check) was applied in this session. The deeper timestamp-at-creation issue remains.

## Root Cause Analysis

### The timestamp chain

There are three independent stores of `(start_utc, end_utc)` for detection windows:

1. **Detection row store** (Parquet) — authoritative, modified by timeline editing
2. **Detection embeddings** (Parquet) — stores `(filename, start_sec, end_sec)`, UTC derived via `parse_recording_timestamp(filename) + start_sec`
3. **Vocalization inference results** (Parquet) — UTC derived from embeddings via the same formula

When timeline editing shifts a row store entry, stores 2 and 3 retain the original timestamps.

### How embeddings are generated for hydrophone jobs

During hydrophone detection, synthetic filenames are created at second precision:
```python
# hydrophone_detector.py:202
synthetic_filename = chunk_start_utc.strftime("%Y%m%dT%H%M%SZ") + ".wav"
```
Embedding records store `(filename, start_sec, end_sec)` relative to this synthetic file. The row store stores absolute `start_utc = chunk_epoch + start_sec`.

Both are consistent at detection time. They diverge only after timeline editing.

### How the embedding sync handles drift

`diff_row_store_vs_embeddings` (`detector.py:808`) uses `_SYNC_TOLERANCE_SEC = 0.5` for matching. A row store entry shifted by ≤0.5s still "matches" its original embedding. No new embedding is generated, no old one is removed.

### How vocalization inference reads timestamps

The inference worker (`vocalization_worker.py:307-313`) recomputes UTC from the embedding parquet:
```python
ts = parse_recording_timestamp(fname)
base = ts.timestamp()
start_utcs.append(base + start_secs[i])
```
This always produces the **original** (pre-edit) timestamps.

### How labels inherit wrong timestamps

The labeling workspace displays inference results with pre-edit timestamps. When the user labels a row, the API receives these timestamps:
```
POST /labeling/vocalization-labels/{job_id}?start_utc={inference_value}&end_utc={inference_value}
```
The label is stored with the inference-derived timestamp, not the current row store timestamp.

### How orphan detection worked (before this session's fix)

`refresh_preview` and `refresh_apply` in `labeling.py` used **exact** set-based matching to compare label timestamps against row store timestamps. Any fractional-second drift caused orphan detection.

## Observed Data

| Label start_utc | Nearest row store start_utc | Offset | Labels affected |
|---|---|---|---|
| 1635756141.0 | 1635756141.5 | 0.5s | 1 |
| 1635756281.0 | 1635756281.5 | 0.5s | 1 |
| 1635756394.0 | 1635756394.5 | 0.5s | 1 |
| 1635756501.0 | 1635756502.0 | 1.0s | 2 (Upsweep, Whup) |
| 1635756809.0 | 1635756809.5 | 0.5s | 2 (Upsweep, Whup) |
| 1635756814.0 | 1635756814.5 | 0.5s | 1 |
| 1635756905.0 | 1635756905.5 | 0.5s | 1 |
| 1635756914.0 | 1635756914.5 | 0.5s | 1 |
| 1635757184.0 | 1635757184.5 | 0.5s | 1 |

**Total:** 11 labels across 9 windows. All labels created 2026-04-02 with `row_store_version_at_import=56` (current). The row store had been heavily edited (`row_store_version=56`).

## Fix Applied This Session

**File:** `src/humpback/api/routers/labeling.py`

Replaced exact set-based matching in `refresh_preview` and `refresh_apply` with tolerance-based matching (`_matches_any_row`) using 0.5s tolerance, consistent with the embedding sync tolerance.

**Result:** 9 of 11 labels (0.5s offset) are no longer flagged as orphaned. 2 labels (1.0s offset) remained correctly flagged and were discarded via UI.

**Tests added:** `test_refresh_preview_tolerates_small_timestamp_shift`, `test_refresh_preview_orphans_beyond_tolerance`

## Remaining Issue

The tolerance fix is defensive — it prevents false-positive orphan warnings for small shifts. But labels are still stored with **wrong timestamps** after timeline editing. This can cause:

- Labels referencing a position 0.5s away from the actual row store window
- Shifts >0.5s still produce orphans
- The problem recurs every time a user labels inference results after timeline editing

### Reproduction steps

1. Run detection on a hydrophone job
2. Generate/sync detection embeddings
3. Run vocalization inference
4. Edit the timeline (move a label by any amount)
5. Open the vocalization labeling workspace
6. Label one of the inference result rows
7. The label is created with the pre-edit timestamp
8. If the shift was >0.5s, the orphan warning appears

## Proposed Solutions (for follow-up)

### Option A: Snap labels to row store on creation (recommended)

In `create_vocalization_label` (`labeling.py`), before storing the label, find the nearest row store entry within tolerance and use its timestamps instead of the API-provided values.

**Pros:** Small change, fixes the problem at point of creation, works regardless of how timestamps diverged.
**Cons:** Requires reading the row store on every label creation (could cache per-request). Silently corrects timestamps — the user won't know the snap happened.

**Sketch:**
```python
# In create_vocalization_label, after receiving start_utc/end_utc:
rs_path = detection_row_store_path(settings.storage_root, detection_job_id)
if rs_path.exists():
    _, rows = read_detection_row_store(rs_path)
    for r in rows:
        rs = float(r.get("start_utc", "0"))
        re = float(r.get("end_utc", "0"))
        if abs(start_utc - rs) <= 0.5 and abs(end_utc - re) <= 0.5:
            start_utc, end_utc = rs, re
            break
```

### Option B: Re-run inference after timeline edits

After timeline editing, prompt the user to re-run vocalization inference so results reflect updated row store timestamps.

**Pros:** All timestamps stay perfectly consistent.
**Cons:** Operationally heavy — inference takes time and the user may not want to wait. Existing labels from the previous inference become orphaned.

### Option C: Map inference results to row store in the labeling workspace

When the labeling workspace loads inference results, cross-reference each result's `(start_utc, end_utc)` against the current row store and substitute the row store's values for display and label creation.

**Pros:** Transparent to the user, no wrong timestamps ever stored.
**Cons:** Requires loading the row store client-side or adding an API endpoint to do the mapping. More complex frontend change.

### Option D: Propagate row store edits to embeddings

When timeline editing modifies the row store, also update the corresponding embedding parquet entries to match the new timestamps.

**Pros:** Keeps embeddings and row store in sync at all times.
**Cons:** Most invasive change. Embedding parquet stores `(filename, start_sec)` which would need recalculation. Could break other consumers of the embedding file.

## Recommendation

**Option A** is the smallest, most defensive fix and should be implemented first. It prevents wrong timestamps from entering the database regardless of the source of drift. Options B-D address upstream consistency but are larger changes that could be pursued incrementally.
