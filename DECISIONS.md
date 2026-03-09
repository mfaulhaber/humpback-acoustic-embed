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

---

## ADR-006: Incremental detection rendering with per-file progress

**Date**: 2026-03
**Status**: Accepted

**Context**: Detection jobs on large audio folders could run for minutes with no intermediate feedback. Users had to wait for full completion before reviewing any results.

**Decision**: Refactor the detection pipeline to emit results incrementally after each audio file completes. A callback from `run_detection()` appends detections to the TSV and updates `files_processed`/`files_total` progress fields in the DB. The content API serves partial results for running jobs. The frontend polls content at 3s intervals while expanded and allows labeling (client-side) during execution, with Save Labels disabled until completion. The final `write_detections_tsv()` call still overwrites with the authoritative version on completion.

**Consequences**:
- Users can listen to and label detections while the job is still running
- Progress visible as "Processing file X/Y" in the job table
- Save Labels, Extract, Delete disabled until completion (TSV safety)
- Added Alembic migration `011_detection_progress_columns.py`
- Thread-safe progress updates via `loop.call_soon_threadsafe()` with separate session

---

## ADR-007: MLP classifier option + enhanced diagnostics for iterative training

**Date**: 2026-03
**Status**: Accepted

**Context**: Users iteratively training binary classifiers with extract-reprocess-retrain loops experienced escalating false positives. The root cause is LogisticRegression's linear decision boundary cannot curve around overlapping embedding regions. Each retrain shifts the hyperplane, creating new FPs elsewhere. Additionally, limited diagnostics (only accuracy + AUC) masked precision problems.

**Decision**: Add MLP classifier as an alternative to LogisticRegression, L2 normalization option, expanded CV metrics (precision/recall/F1), and decision boundary diagnostics (score separation, train confusion matrix). Add overlap validation to prevent same embedding set in both positive and negative lists. Add encoding signature consistency warning.

**Consequences**:
- `classifier_type` parameter: `"logistic_regression"` (default, backward-compatible) or `"mlp"` (MLPClassifier with non-linear boundary)
- `l2_normalize` parameter: opt-in Normalizer step before StandardScaler
- Training summary now includes `cv_precision`, `cv_recall`, `cv_f1`, `score_separation`, `train_confusion`, `classifier_type`, `l2_normalize`, `effective_class_weights`
- Frontend advanced options: classifier type, L2 normalize, regularization C, class weight
- Frontend model table: Precision and F1 columns, diagnostic badges, expandable detail rows
- No schema changes, no migrations required

---

## ADR-008: S3 HLS streaming detection for Orcasound hydrophones

**Date**: 2026-03
**Status**: Accepted

**Context**: Users want to run classifier detection on historic Orcasound hydrophone audio stored as HLS streams in public S3 buckets. The existing detection pipeline only works with local audio folders.

**Decision**: Extend `detection_jobs` (not a new table) with nullable hydrophone columns. Build a parallel streaming pipeline that fetches HLS `.ts` segments from S3, decodes them in-memory via ffmpeg stdin/stdout pipes, and feeds chunks through the existing inference pipeline. Anonymous S3 access (UNSIGNED signature) for the public Orcasound bucket. Audio playback re-fetches from S3 on demand. Hydrophone config is a hardcoded list in `config.py`.

**Consequences**:
- New columns on `detection_jobs`: `hydrophone_id`, `hydrophone_name`, `start_timestamp`, `end_timestamp`, `segments_processed`, `segments_total`, `time_covered_sec`, `alerts`
- `audio_folder` changed from NOT NULL to nullable (hydrophone jobs have no local folder)
- Queue claim functions split: `claim_detection_job` filters `hydrophone_id IS NULL`, `claim_hydrophone_detection_job` filters `IS NOT NULL`
- Detection job listing split: local list excludes hydrophone jobs; separate hydrophone list endpoint
- Cancel support via `threading.Event` + DB polling every 2s
- Flash alerts (JSON array) stored in DB for segment decode failures
- 7-day maximum time range per job
- Added `boto3` dependency
- Alembic migration `012_hydrophone_detection_columns.py`

---

## ADR-009: Atomic compare-and-set claims for all queue job types

**Date**: 2026-03
**Status**: Accepted

**Context**: SQLite does not provide true row-level locking semantics compatible with the prior claim flow. Under concurrent workers, selecting a queued job before status update can race and allow duplicate claims for the same job.

**Decision**: Standardize queue claiming on a compare-and-set pattern for every job type. Each claimant selects a candidate queued job ID, then performs `UPDATE ... SET status='running' WHERE id=:candidate AND status='queued'`. A claim succeeds only when exactly one row is updated; otherwise the worker retries with the next candidate.

**Consequences**:
- Eliminates duplicate claims under concurrent worker sessions on SQLite
- Provides consistent claim behavior across processing, clustering, training, detection, hydrophone detection, and extraction jobs
- Reduces reliance on database locking features that differ by backend
- Requires small retry loops in claimers but keeps queue behavior deterministic

---

## ADR-010: Shared hydrophone stream-offset resolver for playback and extraction

**Date**: 2026-03
**Status**: Accepted

**Context**: Hydrophone detection-row timestamps can be anchored differently across job generations.
Newer rows align to the first available HLS folder timestamp, while older rows are effectively
anchored to `job.start_timestamp`. Playback (`/audio-slice`) and extraction previously used
different lookup strategies, causing some late rows to fail in playback and extraction.

**Decision**: Introduce a shared stream-offset resolver in `classifier/s3_stream.py` and route both
hydrophone playback and hydrophone labeled-sample extraction through it. Resolve offsets with two
anchors in order: first available folder timestamp, then legacy `job.start_timestamp`. Decode only
nearby candidate segments and return explicit not-found behavior when no decodable slice is
resolved.

**Consequences**:
- Playback and extraction now resolve the same timeline mapping for hydrophone jobs
- Late timestamp rows that failed with per-row folder lookup are now decodable
- Legacy jobs remain backward compatible via `job.start_timestamp` fallback
- Extraction worker now passes stream start/end bounds so resolver behavior is deterministic

---

## ADR-011: Bounded hydrophone timeline assembly with numeric segment ordering

**Date**: 2026-03
**Status**: Accepted

**Context**: Hydrophone HLS segment keys can include mixed-width numeric suffixes
(e.g., `live100.ts`, `live1000.ts`, `live101.ts`). Lexicographic ordering of these keys
caused non-chronological assembly and abrupt audio jumps in both detection playback and
exported samples. Additionally, detection could process all segments in a selected folder,
which could extend past the requested `end_timestamp`.

**Decision**: Build hydrophone timelines using numeric segment ordering for `.ts` keys and
use playlist (`live.m3u8`) duration metadata when available. Treat all hydrophone consumers
(streaming detection, `/audio-slice` playback resolver, labeled-sample extraction resolver)
as readers of the same bounded timeline clipped to `[start_timestamp, end_timestamp]`.
Retain legacy playback fallback anchored to `job.start_timestamp` for older jobs.

**Consequences**:
- Eliminates lexicographic segment-order assembly errors (`...100 -> 1000 -> 101...`)
- Keeps hydrophone detection/playback/extraction within requested job time bounds
- Improves timestamp consistency between generated detection ranges and playback audio
- Preserves backward compatibility for previously generated jobs via legacy anchor fallback
