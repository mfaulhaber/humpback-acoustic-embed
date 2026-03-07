# Architecture Decision Log

Append-only record of significant design decisions. Do not edit historical entries.

---

## ADR-001: Overlap-back windowing replaces zero-padding

**Date**: 2026-03
**Status**: Accepted
**Commit**: `7e96418`

**Context**: Zero-padded final audio windows create out-of-distribution spectrograms (silence-filled regions) that cause false positive detections in downstream classifiers.

**Decision**: Replace zero-padding with overlap-back windowing. When the last audio chunk is shorter than `window_size_seconds`, shift its start backward so it ends at the audio boundary, overlapping with the previous window. Audio shorter than one window is skipped entirely.

**Consequences**:
- Every window contains only real audio — no synthetic silence
- `WindowMetadata.is_padded` replaced by `is_overlapped`
- Files shorter than `window_size_seconds` produce 0 embeddings (logged as warning)
- Classifier false positive rate significantly reduced

---

## ADR-002: In-place folder import replaces file copy

**Date**: 2026-03
**Status**: Accepted
**Commit**: `bbc0137`

**Context**: Copying large audio files into `audio/raw/` on upload doubled disk usage and was slow for bulk imports of existing audio libraries.

**Decision**: Add `source_folder` column to `AudioFile`. When set, audio is read directly from the original location instead of from `audio/raw/`. The upload endpoint now imports folders in-place by scanning and registering files without copying.

**Consequences**:
- No disk duplication for imported audio
- Audio files must remain at their original path (user responsibility)
- `audio/raw/` still used for individually uploaded files (backward compatible)
- Added Alembic migration `009_add_source_folder.py`

---

## ADR-003: Balanced class weights for detection classifier

**Date**: 2026-03
**Status**: Accepted
**Commit**: `176b572`

**Context**: Detection spans were covering entire files because the LogisticRegression classifier was biased toward the positive class when training data was imbalanced.

**Decision**: Default to `class_weight='balanced'` in the LogisticRegression classifier so that the model automatically adjusts weights inversely proportional to class frequencies.

**Consequences**:
- Fixes false-positive-heavy detection spans
- Works automatically regardless of positive/negative ratio
- No user configuration needed (sensible default)

---

## ADR-004: Negative embedding sets for classifier training

**Date**: 2026-03
**Status**: Accepted
**Commit**: `fd22937`

**Context**: Classifier training originally required a negative audio folder path, which meant re-processing audio that might already have embedding sets. This was wasteful and inconsistent with the idempotent encoding design.

**Decision**: Replace the negative audio folder approach with negative embedding set IDs. Training jobs now accept `negative_embedding_set_ids` (JSON array) alongside `positive_embedding_set_ids`, reusing already-computed embeddings.

**Consequences**:
- No redundant audio processing for negative examples
- Consistent with the idempotent encoding principle
- Added Alembic migration `007_negative_embedding_set_ids.py`
- UI updated to allow selecting embedding sets as negative examples

---

## ADR-005: Overlapping window inference + hysteresis event detection

**Date**: 2026-03
**Status**: Accepted

**Context**: The detection pipeline used non-overlapping 5-second windows with index-based span merging. This caused poor temporal resolution (events snapped to 5s boundaries) and no ability to tune sensitivity with dual thresholds.

**Decision**: Add configurable `hop_seconds` for overlapping window inference and replace single-threshold span merging with hysteresis-based event detection using `high_threshold` (start) and `low_threshold` (continue). Extraction boundaries snap outward to `window_size` multiples for clean training samples.

**Consequences**:
- Detection defaults to 1s hop with 0.70/0.45 hysteresis thresholds (new behavior by default)
- Sub-second temporal resolution for event boundaries
- Per-event `n_windows` count in TSV output
- Added Alembic migration `010_detection_hysteresis.py`
- Legacy behavior available with `hop_seconds=5.0, high_threshold=0.5, low_threshold=0.5`
