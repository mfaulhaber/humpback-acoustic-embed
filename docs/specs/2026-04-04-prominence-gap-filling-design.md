# Prominence Gap-Filling Design

**Date:** 2026-04-04
**Status:** Approved

## Problem

Prominence-based window selection identifies distinct peaks in the logit-score
curve within detection events. However, strong vocalizations that don't produce
distinct peaks — because the score curve is flat between adjacent
equal-strength vocalizations — are missed. This leaves gaps in detection
coverage where spectrograms show real vocalizations but the prominence
algorithm correctly rejects the low-prominence peaks.

This is fundamentally different from the logit-compression problem (solved by
the logit transform): here the score curve is genuinely flat, and no score
transformation helps.

### Observed example

Job `eed4ffeb-0c7f-474f-9a10-c841d48dbb81`, region 02:19:02–02:19:07 UTC.
Flanking detections at 02:18:57 (conf 0.999) and 02:19:07 (conf 0.989). The
gap peak at 02:19:05 has 0.988 confidence but only 0.05 logit prominence —
correctly rejected, yet the spectrogram shows a clear vocalization.

## Solution: Recursive Gap-Filling

After prominence selects peaks (including the existing single-window fallback),
a gap-fill pass scans for uncovered regions and emits additional windows.

### Algorithm

1. Sort selected peak offsets within the event.
2. Build a gap list using the event's `start_sec`/`end_sec` as outer
   boundaries:
   - `(event.start_sec, first_peak_offset)`
   - `(peak_i_offset, peak_{i+1}_offset)` for all consecutive pairs
   - `(last_peak_offset, event.end_sec)`
3. For each gap where `(right - left) > min_gap_fill`:
   - Find the candidate window record in `(left, right)` (exclusive) with the
     highest **raw probability** score above `min_score`.
   - Emit it as an additional detection.
   - Recurse: the fill window splits the gap into two sub-gaps, each checked
     against the same threshold.
4. Recursion terminates when all sub-gaps are <= `min_gap_fill` or no
   candidates above `min_score` exist in a gap.

### Parameters

- **`min_gap_fill`**: 5.0 seconds (matching `window_size_seconds`). Hardcoded
  inside `select_prominent_peaks_from_events`. Not exposed to the API. Can be
  promoted to a parameter later if tuning is needed. Originally 3.0 seconds
  but increased after testing showed 2-second window spacing with excessive
  overlap.
- **Fill placement**: Prefer proximity to gap midpoint, with raw probability
  as tiebreaker. This spaces fills evenly across the gap rather than
  clustering next to existing peaks (where scores are naturally highest).
- **Score space for fill selection**: Raw probability (not logit). Logit
  transform matters for detecting dips (prominence computation), not for
  qualifying a candidate.
- **`min_score`**: Same `min_score` (high threshold) already passed to
  `select_prominent_peaks_from_events`.

### Activation

Always-on when `window_selection="prominence"`. No separate toggle. Gap-filling
compensates for an inherent limitation of prominence-based selection; there is
no useful case for prominence without it.

## Code Changes

### New helper: `_fill_gaps_recursive`

Location: `src/humpback/classifier/detector_utils.py`, private function.

```python
def _fill_gaps_recursive(
    candidates: list[dict],       # window records for this event
    raw_scores: list[float],       # parallel raw probability scores
    selected_offsets: set[float],  # offsets already selected as peaks
    left: float,                   # left gap boundary
    right: float,                  # right gap boundary
    min_gap_fill: float,           # minimum gap to trigger fill (3.0)
    min_score: float,              # minimum raw probability to emit
) -> list[int]:                    # indices into candidates to emit
```

Returns indices into the `candidates` list for windows to add.

### Integration point

Inside `select_prominent_peaks_from_events`, after the existing zero-peaks
fallback block and before the detection-building loop:

1. Collect sorted offsets of all selected peaks.
2. Build gap list from event boundaries and peak offsets.
3. Call `_fill_gaps_recursive` for each gap.
4. Merge returned indices into `peak_indices`.

### No caller changes

`detector.py`, `hydrophone_detector.py`, service layer, worker layer, and API
layer are unaffected. Gap-filling is an internal detail of
`select_prominent_peaks_from_events`.

## Interaction with Existing Mechanisms

### Zero-peaks fallback

Unchanged. When no peaks pass prominence, the fallback emits the single
highest-scoring window. Gap-filling then runs on that single window, finding
gaps from event edges to it. This improves coverage for broad plateaus — a
15-second plateau that currently gets 1 window could get 4-5.

### Deduplication

The existing deduplication block at the end of `select_prominent_peaks_from_events`
handles duplicate `(start_sec, end_sec)` pairs by keeping the higher-confidence
entry. Gap-fill windows pass through the same dedup — no additional logic needed.

### NMS mode

Unaffected. Gap-filling only runs inside `select_prominent_peaks_from_events`,
which is only called when `window_selection="prominence"`.

## Testing

New unit tests in `tests/unit/test_detection_spans.py` within the existing
`TestSelectProminentPeaksFromEvents` class:

1. **Basic gap fill** — two peaks 10s apart, verify a fill window appears
   between them.
2. **Recursive fill** — two peaks 15s apart, verify multiple fill windows.
3. **Edge gap fill** — single peak in the middle of a long event, verify fills
   toward event boundaries.
4. **No fill when gap is small** — peaks 2s apart, verify no extra windows.
5. **No fill when gap scores are below threshold** — gap region has low scores,
   verify no fill emitted.
6. **Fallback + gap fill compose** — zero prominent peaks on a 15s plateau,
   verify fallback window plus gap fills.
7. **Recursion terminates** — flat high-confidence scores across a long event,
   verify bounded output count.

## Documentation Updates

- **CLAUDE.md §8.7**: Update window selection modes bullet to describe
  gap-filling as part of prominence mode with the 3-second default.
- **DECISIONS.md**: Append to existing prominence ADR noting the gap-fill
  addition and rationale.

## Out of Scope

- File boundary gaps (hydrophone chunk boundaries) — pre-existing limitation
  requiring cross-chunk event merging.
- Exposing `min_gap_fill` as an API parameter — can be added later if needed.
- Changes to NMS window selection.
