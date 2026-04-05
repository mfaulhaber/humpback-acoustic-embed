# Tiling Window Selection for Detection Jobs

**Date:** 2026-04-04
**Status:** Approved

## Problem

NMS and prominence-based window selection are both peak-centric: they identify
discrete peaks then (in prominence's case) retroactively fill gaps. Flat
plateaus where the classifier scores 0.99 across 20 consecutive windows have
no peaks or prominence to exploit. Gap-filling patches this but is an
after-the-fact heuristic.

## Design

A third `window_selection` mode called `"tiling"` that treats high-scoring
regions as **spans to cover** rather than peaks to find.

### Algorithm: greedy peak-then-tile

For each merged event:

1. Collect candidate windows overlapping the event, sorted by offset.
2. Compute logit scores; filter to candidates with raw score >= `min_score`.
3. Build an `uncovered` set of qualifying candidate indices.
4. Multi-pass loop while `uncovered` is non-empty:
   - **Seed:** pick the index in `uncovered` with the highest logit score.
   - **Tile left:** from seed-1, walk leftward through consecutive candidates
     while in `uncovered` and `seed_logit - candidate_logit <= max_logit_drop`.
     Stop at first violation or event edge.
   - **Tile right:** mirror of tile left from seed+1.
   - Mark seed + all tiled windows as covered; emit as detection rows.
5. Deduplicate across adjacent events (same logic as prominence mode).

### Stop criterion

Stop tiling at the first window where `seed_logit - window_logit > max_logit_drop`.
No look-ahead or recovery — if scores dip and recover, the recovered region
becomes its own seed in the next pass.

### Parameters

- `window_selection = "tiling"` — new mode value
- `max_logit_drop` — logit-unit threshold for tiling extent (default 2.0).
  A drop of 2.0 from a peak at p=0.99 stops tiling around p=0.93; from p=0.95
  around p=0.73.

### Data model

- Alembic migration 039: add `max_logit_drop` (Float, nullable) to `detection_jobs`
- `window_selection` column already free-text; no migration needed for the new value

### API / schema

- `DetectionJobCreate` and `HydrophoneDetectionCreate`: extend `window_selection`
  Literal to `"nms" | "prominence" | "tiling"`, add `max_logit_drop: Optional[float]`
  with `> 0` validation

### Frontend

- Add `"tiling"` to window selection dropdown in HydrophoneTab
- Show `max_logit_drop` input when tiling is selected (same pattern as
  `min_prominence` for prominence)

### Validation script

- Extend `scripts/validate_gap_filling.py` to accept `--mode tiling` and
  `--max-logit-drop` args, adding a third column to side-by-side output

### What stays the same

- Event formation (hysteresis merge + snap)
- Detection row format, Parquet schema, all downstream systems
- NMS and prominence modes untouched

## Testing

Unit tests in `test_detection_spans.py`:
- Basic symmetric tiling from a single peak
- Multi-pass: two peaks separated by deep dip produce two tile groups
- Plateau: flat high scores emit all windows
- Edge clipping: tiling stops at event boundary
- Drop exactly at threshold: boundary condition (<=, included)
- No qualifying windows: returns empty
- Deduplication across adjacent events

Manual validation against job `d52e03cc` at 02:10-02:34 UTC.
