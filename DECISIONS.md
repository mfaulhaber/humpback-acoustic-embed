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

---

## ADR-012: Hydrophone detection metadata includes extraction-aligned filename

**Date**: 2026-03
**Status**: Accepted

**Context**: Hydrophone detection rows were displayed as UTC spans derived from
`filename + start_sec/end_sec`, while extracted WAV outputs use window-snapped
bounds to build final filenames. This made it harder to visually reconcile table
rows with exported sample names and encouraged brittle client-side assumptions.

**Decision**: Add an `extract_filename` column to hydrophone detection TSV rows
and surface it through the detection content API as an optional field.
Derive this value from extraction-snapped bounds using the classifier window size.
Keep local (non-hydrophone) TSV format unchanged. Preserve unknown TSV columns in
label-save rewrite logic so metadata is retained across annotation updates.
Update Hydrophone UI to keep raw UTC `Detection Range`, show `extract_filename`
in tooltip, and replace `Start/End` display with snapped `Duration`.

**Consequences**:
- Hydrophone TSV downloads now carry explicit extraction basename metadata
- UI can display extraction-consistent context without changing playback keys
- Label-save operations no longer strip non-label TSV columns
- No DB migration required; behavior is file-format and API-row augmentation only

---

## ADR-013: Local-cache-authoritative hydrophone extraction with timeline reuse

**Date**: 2026-03
**Status**: Accepted

**Context**: Hydrophone labeled-sample extraction previously used `CachingS3Client` when
`job.local_cache_path` was unset. Even with local segment cache hits for `.ts` bytes,
each labeled row rebuild could still trigger slow S3 `list_hls_folders`/`list_segments`
metadata calls, causing long extraction latency.

**Decision**: Make hydrophone extraction match playback cache authority:
- always resolve hydrophone extraction through `LocalHLSClient` using
  `job.local_cache_path` or `settings.s3_cache_path`
- do not perform S3 fallback during extraction
- build stream timeline metadata once per extraction run and reuse it across rows

**Consequences**:
- Hydrophone extraction avoids S3 metadata/object calls in normal operation
- Extraction latency is dominated by local filesystem and decode work
- Missing local-cache audio remains non-fatal: rows are skipped and counted in `n_skipped`
- No API schema or DB migration changes

---

## ADR-014: Configurable incremental hydrophone lookback + explicit no-audio failure

**Date**: 2026-03
**Status**: Accepted

**Context**: Hydrophone folder discovery previously used a fixed short lookback window,
which could miss valid overlapping audio when a request started hours after a folder
timestamp. Separately, true no-audio ranges could silently finish as `complete` with
zero windows, which obscured user-visible failure causes.

**Decision**:
- Replace fixed lookback with configurable timeline-expansion settings:
  `hydrophone_timeline_lookback_increment_hours` (default `4`) and
  `hydrophone_timeline_max_lookback_hours` (default `168`).
- Keep stream assembly clipping authoritative for requested
  `[start_timestamp, end_timestamp]` bounds.
- Propagate no-overlap timeline outcomes as `FileNotFoundError` and let the
  hydrophone worker mark jobs `failed` with a clear hydrophone/range message.

**Consequences**:
- Long-running folders that begin before the requested start can still be discovered
  when they overlap the requested interval.
- Hydrophone jobs with no overlapping audio now fail explicitly instead of silently
  completing with zero windows.
- No API schema changes and no DB migration required.

---

## ADR-015: Hydrophone job pause/resume/cancel controls and Detect tab removal

**Date**: 2026-03
**Status**: Accepted

**Context**: The Classifier/Detect tab for local audio folder detection was no longer needed
alongside the hydrophone workflow. Users running long hydrophone detection jobs had no way
to temporarily pause processing without losing all progress, and canceled jobs with partial
results were not accessible in the Previous Jobs panel.

**Decision**:
- Remove the Classifier/Detect sub-tab (frontend only; backend endpoints retained for data access).
- Add `paused` job status with `pause`/`resume` API endpoints and worker-side `threading.Event`
  pause gate that blocks the detection thread at chunk boundaries.
- Make canceled jobs fully functional in Previous Jobs (expandable, downloadable, label-editable, extractable).
- Replace `datetime-local` inputs with 24hr UTC text inputs (`YYYY-MM-DD HH:MM`).
- Display processed audio duration as hours:minutes instead of raw seconds.
- Add `hydrophone_name` column to hydrophone detection TSV output.

**Consequences**:
- Worker detection thread blocks via `pause_gate.wait()` when paused; unblocks on resume or cancel.
- Stale paused jobs are recovered to `queued` by `recover_stale_jobs()`.
- No DB migration required (`paused` is a status string value in existing column).
- Canceled jobs now support content, download, labels, and extraction endpoints.
- Hydrophone TSV includes 8 columns (added `hydrophone_name`).

---

## ADR-016: Automated retrain workflow for classifier models

**Date**: 2026-03
**Status**: Accepted

**Context**: Users iteratively improve classifiers by adding labeled audio to their positive/negative
folders, then retraining. This required manually reimporting folders, queuing processing, waiting for
completion, and creating a new training job with the same parameters — four separate operations
across multiple tabs.

**Decision**: Add a `retrain_workflows` table with a backend-orchestrated state machine
(`queued` → `importing` → `processing` → `training` → `complete`/`failed`). A single "Retrain"
button on each trained model triggers the full pipeline. The workflow traces folder roots from the
original training job's embedding set provenance, reimports those folders, queues processing for
any unprocessed audio, then collects ALL embedding sets from the folder hierarchies (not just the
original IDs) to include newly added files.

**Consequences**:
- New `retrain_workflows` table with Alembic migration `014_retrain_workflows.py`
- Worker polls retrain workflows after extraction jobs in the main loop
- Stale retrain workflows (importing/processing/training) are recovered to queued after timeout
- Frontend adds retrain sub-panel to expanded model rows in Classifier/Train tab
- 4 new API endpoints: `GET /retrain-info`, `POST /retrain`, `GET /retrain-workflows`, `GET /retrain-workflows/{id}`
- Folder tracing resolves import roots by walking `source_folder` + `folder_path` hierarchy
- Embedding set collection uses folder-path prefix matching to include all files under import roots

---

## ADR-017: Canonical exact hydrophone detection filename and preview/extraction parity

**Date**: 2026-03
**Status**: Accepted

**Context**: Hydrophone `extract_filename` values were derived from snapped extraction bounds.
Overlapping raw detections could collapse to the same snapped range, which produced duplicate
Detection Range values in the UI and ambiguity during label review. The labeling workflow requires
that what users preview is exactly what gets extracted for training.

**Decision**:
- Add canonical `detection_filename` to hydrophone detection rows using exact event bounds
  (`filename + start_sec/end_sec`) formatted as compact UTC range.
- Keep `extract_filename` for backward compatibility, but emit it as a legacy alias of
  `detection_filename` for new hydrophone rows.
- Supersede ADR-012 snapped-`extract_filename` semantics for new hydrophone jobs.
- Render a single Hydrophone Detection Range in the UI from `detection_filename`.
- Make hydrophone playback and hydrophone extraction use the same exact clip bounds as
  displayed in Detection Range.
- For legacy TSV rows without `detection_filename`, derive it at content-read time from
  `filename + start_sec/end_sec`.

**Consequences**:
- Removes snapped-range collisions in Hydrophone Detection Range display
- Guarantees preview/extraction parity for hydrophone labeling workflows
- Preserves backward compatibility for existing TSV consumers via `extract_filename`
- No database migration required (TSV/API/UI behavior change only)

---

## ADR-018: Canonical snapped clip ranges for detection labeling and extraction parity

**Date**: 2026-03
**Status**: Accepted

**Context**: Users label detections from previewed audio, then extract labeled clips for
training. Exact hysteresis event bounds (for example 7s) caused ambiguity about whether
clips should be window-aligned for model training. Applying snapping only at extraction
time created mismatch risk between what users labeled and what was exported.

**Decision**:
- Canonicalize detection rows to snapped clip bounds (`start_sec`, `end_sec`) before
  labeling for both local and hydrophone jobs.
- Preserve raw unsnapped event bounds in TSV/API metadata (`raw_start_sec`, `raw_end_sec`)
  plus `merged_event_count` when multiple raw events collapse to one snapped clip.
- Merge snapped-range collisions deterministically (weighted average confidence by
  `n_windows`, max peak confidence, summed `n_windows`).
- Align preview and extraction to the same canonical snapped clip bounds.
- Keep `detection_filename` canonical for hydrophone rows and `extract_filename` as a
  compatibility alias; for legacy rows missing `detection_filename`, normalization prefers
  valid `extract_filename`, then derives snapped canonical filename from row bounds.

**Consequences**:
- Prevents preview/label/extract mismatches for both hydrophone and local workflows
- Restores clean window-aligned clip exports without post-label widening
- Retains raw event precision for audit/debugging without changing label keys
- No database migration required (TSV/API/UI behavior change only)

---

## ADR-019: Ordered bounded S3 segment prefetch for hydrophone detection

**Date**: 2026-03
**Status**: Accepted

**Context**: Hydrophone detection decodes one `.ts` segment at a time. On cold cache
jobs, per-segment S3 `get_object` latency can dominate throughput because network fetch
and decode/inference are serialized.

**Decision**:
- Add optional concurrent segment prefetch in `iter_audio_chunks()` with bounded
  in-flight fetches and configurable worker count.
- Keep timeline-order consumption deterministic (results are consumed in segment order
  even when fetched concurrently).
- Reuse existing retry/error behavior: fetch failures still surface through warning
  alerts and cache invalidation retry logic remains intact.
- Enable prefetch for S3-backed hydrophone detection clients (`OrcasoundS3Client` and
  `CachingS3Client`) and keep local-cache-only detection behavior unchanged.
- Expose runtime controls via settings:
  `hydrophone_prefetch_enabled`, `hydrophone_prefetch_workers`,
  `hydrophone_prefetch_inflight_segments`.
- Extend hydrophone `run_summary` timing telemetry with
  `fetch_sec`, `decode_sec`, `features_sec`, `inference_sec`, and `pipeline_total_sec`.

**Consequences**:
- Improves cold-cache hydrophone detection throughput by overlapping S3 fetch with
  decode/inference work.
- Bounds extra network and memory pressure through a fixed in-flight queue.
- Preserves deterministic segment ordering and existing API/TSV/DB schemas.
- Adds only configuration + runtime behavior changes (no migration required).

---

## ADR-020: Orca detection label + species-first extraction paths + has_positive_labels rename

**Date**: 2026-03
**Status**: Accepted

**Context**: Users working with Orcasound hydrophones encounter both humpback whales and
orcas. The detection UI only supported humpback (positive), ship (negative), and background
(negative) labels. Additionally, hydrophone extraction paths placed hydrophone_id before
species/category (`{root}/{hydrophone_id}/{species}/...`), making it harder to browse all
samples of one species across multiple hydrophones.

**Decision**:
- Add `orca` as a fourth detection label, routed as a positive label alongside humpback.
- Add keyboard shortcut `o` for orca labeling.
- Reorder hydrophone extraction paths to species/category-first:
  `{root}/{species}/{hydrophone_id}/YYYY/MM/DD/*.wav`.
- Rename DB column `has_humpback_labels` → `has_positive_labels` to cover both humpback
  and orca (migration `016`).
- Flag computation: `has_positive = any(humpback == 1 or orca == 1)`.
- Backward-compatible TSV handling: old TSVs without `orca` column work fine (missing
  column reads as empty/null).

**Consequences**:
- Orca vocalizations can now be labeled and extracted for training alongside humpback
- Extraction folder structure is species-first, enabling easy cross-hydrophone browsing
- Alembic migration `016_rename_has_positive_labels.py` renames the column
- Frontend "Whale" badge now appears for jobs with any positive label (humpback or orca)
- Local (non-hydrophone) extraction paths unchanged (no hydrophone_id component)
