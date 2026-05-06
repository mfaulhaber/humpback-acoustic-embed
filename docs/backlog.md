# Backlog

Items identified during development that are out of scope for the current task.

---

## Hydrophone detector writes incomplete window diagnostics

**Found:** 2026-03-26 during timeline viewer enhancements

**Symptom:** The confidence heatmap in the timeline viewer shows gaps (null/dark) at time positions where high-confidence detections exist. For example, job `756882ed` has 60 detection rows spread across 24 hours, but the `window_diagnostics.parquet` only contains data for 4 one-minute audio segments (at 05:46, 11:46, 23:45, 23:46 UTC).

**Root cause:** The hydrophone detector writes detection rows for every high-confidence window it finds, but only writes window-level diagnostics (offset_sec + confidence for every processed window) for a subset of audio segments. The confidence heatmap depends on the diagnostics data to show per-window scores across the full timeline.

**Impact:** The confidence heatmap is sparse and not useful for jobs where diagnostics are incomplete. Users see detections/labels with no corresponding heatmap indication.

**Expected behavior:** The diagnostics parquet should contain one row per detection window for every audio segment the detector processed, not just the segments that yielded high-confidence detections. This would give the heatmap full timeline coverage.

---

## Frontend visualization for masked-transformer geometry diagnostics

**Found:** 2026-05-06 during projection-head geometry diagnostics.

**Symptom:** The backend nearest-neighbor report now returns per-space geometry diagnostics and sweep-gating verdicts, but there is no dedicated UI for comparing contextual/retrieval cone collapse, PCA dominance, effective rank, pre-L2 norm spread, or per-dimension standard-deviation summaries.

**Impact:** Researchers must inspect API JSON, sweep CSV/Markdown, or saved comparison artifacts to understand whether retrieval raw geometry is saturated before continuing lambda sweeps.

**Expected behavior:** Add a masked-transformer diagnostics panel that renders geometry spaces side by side, highlights red-flag thresholds, and clearly shows whether `retrieval.raw_l2` blocks further lambda sweeps.
