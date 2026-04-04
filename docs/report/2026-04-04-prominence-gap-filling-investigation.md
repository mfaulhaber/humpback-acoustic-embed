# Prominence Gap-Filling Investigation

**Date:** 2026-04-04
**Status:** Ready for brainstorming next session
**Context:** Follow-up from logit-space prominence work (ADR-044 update)

## Background

The logit-space prominence change (this session) solved the score-compression
problem where dips between vocalizations in high-confidence plateaus (0.99+)
were invisible in probability space. After lowering the default
`min_prominence` from 2.0 to 1.0 logit units, most gaps in sustained singing
regions are filled.

However, a different class of gap remains: **strong vocalizations that don't
produce distinct peaks** in the score curve.

## Observed Gap

**Job:** `eed4ffeb-0c7f-474f-9a10-c841d48dbb81`
**Region:** 02:19:02–02:19:07 UTC (5 seconds, clear spectrogram vocalization)

Detections flanking the gap:
- 02:18:57–02:19:02 (conf 0.999)
- 02:19:07–02:19:12 (conf 0.989)

### Raw scores in the gap

```
UTC       Prob    Logit   Note
02:19:02  0.993   4.91    end of previous detection window
02:19:03  0.913   2.35    dip (still well above thresholds)
02:19:04  0.973   3.59    
02:19:05  0.988   4.44    peak — but only 0.05 logit prominence
02:19:06  0.988   4.38    
02:19:07  0.989   4.46    start of next detection window
```

## Root Cause

The peak at offset 47 (02:19:05, score 0.988) has **0.05 logit prominence**
because it is sandwiched between two nearly-equal neighbors:
- Left higher peak at offset 43 (logit 4.96)
- Right higher peak at offset 49 (logit 4.46)
- The dip between 47 and 49 (logit 4.38) is only 0.05 below the peak

The prominence algorithm correctly says "this is not a distinct peak" — but
the spectrogram shows a real vocalization. The issue is that a vocalization
can be strong without creating a prominent score peak relative to adjacent
vocalizations.

**This is fundamentally different from the logit-compression problem:**
- Logit compression: real dips exist but are invisible in probability space
  (fixed by logit transform)
- This case: the score curve is genuinely flat between adjacent vocalizations
  — no amount of score transformation helps

## Proposed Solution: Gap-Filling Fallback

After prominence selects the main peaks within an event, add a second pass
that scans for uncovered gaps between consecutive detection windows. For each
gap >= `window_size_seconds` where the best window scores above `min_score`,
emit that window as an additional detection.

### Algorithm sketch

1. Sort selected peaks by offset within each event
2. For each pair of consecutive selected peaks at offsets p_i and p_{i+1}:
   - Compute gap = p_{i+1} - p_i
   - If gap > `window_size_seconds`:
     - Find the candidate window in [p_i+1, p_{i+1}-1] with the highest raw
       score above `min_score`
     - Emit it as an additional detection
3. Also check the gap from event start to first peak, and from last peak to
   event end

### Design questions to resolve in brainstorming

- Should gap-filling use raw probability scores or logit scores for selecting
  the best fill window? (Raw makes more sense since we're just finding the
  max, not computing prominence.)
- Should the gap threshold be exactly `window_size_seconds` (5s), or
  configurable?
- Should gap-filling be recursive? (After filling a gap, the fill window
  creates two smaller sub-gaps — should those be checked too?)
- Should gap-filling be a separate parameter toggle, or always-on when
  `window_selection="prominence"`?
- How does this interact with the existing fallback (emit single best when NO
  peaks pass)? The existing fallback handles empty-event cases; gap-filling
  handles events with peaks but uncovered regions.

## Also Observed: File Boundary Gaps

Hydrophone detection processes each audio chunk independently
(`hydrophone_detector.py` lines 228–249). Events cannot span file boundaries.
This creates structural gaps of ~5 seconds at every chunk boundary where the
end of one event and the start of the next cannot merge.

Example from job `7e674dcf`: file boundary at 02:17:13–02:17:18 creates a
gap even though scores are high on both sides. This is a separate, pre-existing
limitation — not caused by prominence detection — and would require cross-chunk
event merging to fix. Out of scope for the gap-filling fallback.

## Test Data for Validation

When implementing, use these jobs for regression testing:
- `7e674dcf-749b-4f40-9f0c-4688b649aae6` — original prominence job, high-
  confidence plateaus at 02:16:48–02:17:13 and 02:27:03–02:28:03
- `eed4ffeb-0c7f-474f-9a10-c841d48dbb81` — gap at 02:19:02–02:19:07 from
  flat score region between equal-strength vocalizations
