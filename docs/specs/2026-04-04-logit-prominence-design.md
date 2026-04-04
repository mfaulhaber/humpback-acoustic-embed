# Logit-Space Prominence for Detection Window Selection

**Date:** 2026-04-04
**Status:** Approved

## Problem

Prominence-based window selection (ADR-044) fails to detect individual
vocalizations inside sustained high-confidence regions. When the classifier
scores every 1-second window above 0.95, the dips between distinct calls are
tiny in probability space (e.g., 0.999 → 0.983 = prominence 0.017), falling
below the default `min_prominence` threshold of 0.03.

Observed in detection job `7e674dcf`:
- 12-second gap (02:16:53–02:17:05): 3 visible vocalizations, 0 detections
- 60-second gap (02:27:03–02:28:03): many visible vocalizations, 0 detections
- All windows in gaps score 0.889–0.999 — scores never drop below
  `low_threshold` (0.8), so the entire stretch is one merged hysteresis event
  whose interior peaks lack sufficient prominence in probability space.

## Root Cause

Probability scores compress meaningful variation at the extremes. A dip from
0.999 to 0.889 looks like 0.11 in probability space but represents a drop from
logit 6.9 to logit 2.1 — a prominence of 4.8 logit units. The current
algorithm operates in probability space, where this compression makes
inter-vocalization dips indistinguishable from noise.

## Solution

Transform raw probability scores to logit space (`ln(p / (1 − p))`) before
computing prominence. This is the standard mathematical fix for saturation at
the extremes — the same reason logistic regression operates in logit space.

### Score Transformation

`select_prominent_peaks_from_events()` converts raw confidence scores to logit
space before passing them to `_find_prominent_peaks()`. A clamping epsilon
(1e-7) prevents ±infinity at 0 and 1. The `min_score` threshold is also
converted to logit space so the peak-candidate filter remains consistent.

### `_find_prominent_peaks()` Stays General

The function still receives a score array and thresholds — it does not know
about the logit transform. The caller is responsible for transforming all
inputs. This keeps the function testable with arbitrary score sequences.

### Default `min_prominence` Changes

The default changes from 0.03 (probability units) to 2.0 (logit units).

Empirical basis from detection job `7e674dcf`:
- Noise-level wobbles: 0.9–1.4 logit prominence
- Genuine inter-vocalization dips: 2.8–4.8 logit prominence
- A default of 2.0 is permissive enough to catch real dips while filtering
  most score noise. The parameter remains tunable per job.

### No New API Parameters

`min_prominence` keeps the same name; its unit scale and default change.
Prominence mode shipped in the previous commit with one test job, so there is
no backward-compatibility surface.

### No Database Migration

The `min_prominence` column stores whatever the user provides. The Python
default changes. Old jobs with 0.03 stored would produce very permissive
detection (0.03 logits ≈ no filtering) if re-run, which is acceptable.

## Scope

### Changes
- `_find_prominent_peaks()` — no signature change; receives logit-transformed
  scores from caller
- `select_prominent_peaks_from_events()` — adds logit transform of raw scores
  and `min_score` before calling `_find_prominent_peaks()`
- Default `min_prominence` — 0.03 → 2.0 everywhere (detector.py,
  detector_utils.py, hydrophone_detector.py, API schemas, tests)
- Tests — update existing prominence tests for logit-scale thresholds, add
  tests for high-confidence plateau detection

### Unchanged
- Hysteresis event merging (probability-space thresholds)
- NMS window selection (separate code path)
- Fallback logic (single best window when no peaks pass)
- Deduplication and audit fields
- All downstream consumers (row store, embeddings, labeling, extraction)
- Database schema
