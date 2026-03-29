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

---

## ADR-021: FLAC labeled-sample extraction outputs + sibling conversion utility

**Date**: 2026-03
**Status**: Accepted

**Context**: Detection-job extraction was writing uncompressed WAV clips for both local
and hydrophone workflows. These clips are used as labeled training data, so lossless
compression is preferable to reduce storage without changing training semantics. Users
also need a simple way to convert existing local audio libraries to FLAC while verifying
that decoded audio remains equivalent within quantization tolerance.

**Decision**:
- Change local and hydrophone labeled-sample extraction outputs from WAV to 16-bit PCM FLAC.
- Change newly derived hydrophone `detection_filename` and `extract_filename` values to
  use `.flac` so metadata matches exported clip basenames.
- Keep raw detection source `filename` values and playback endpoints unchanged (`.wav`
  chunk identifiers and WAV preview streaming remain as-is).
- Preserve backward compatibility for legacy TSV rows that already contain explicit `.wav`
  `detection_filename` or `extract_filename` values.
- Add `scripts/convert_audio_to_flac.py` to convert `.wav`/`.mp3` files to sibling
  `.flac` files with optional decoded-sample verification.

**Consequences**:
- Extracted labeled-sample datasets use less disk space with no intentional audio loss.
- New hydrophone clip metadata now points at `.flac` basenames, while legacy `.wav`
  metadata stays readable.
- No database migration is required because the change is limited to file outputs,
  TSV/API metadata derivation, and documentation.

---

## ADR-022: Explicit platform TensorFlow extras and Python version cap

**Date**: 2026-03
**Status**: Accepted

**Context**: The project runs TensorFlow workloads on Apple Silicon macOS and on Linux
GPU servers. Keeping TensorFlow in the base dependency set forced one install contract
across incompatible platform/runtime combinations, and `uv sync --all-extras` was no
longer a safe default once Linux CPU, Linux GPU, and macOS TensorFlow variants diverged.
The supported TensorFlow wheel set also does not justify claiming Python 3.13 support.

**Decision**:
- Remove TensorFlow packages from base runtime dependencies.
- Add mutually-exclusive extras:
  - `tf-macos` for Apple Silicon (`tensorflow-macos` + `tensorflow-metal`)
  - `tf-linux-cpu` for Linux CPU (`tensorflow`)
  - `tf-linux-gpu` for Linux GPU/CUDA (`tensorflow[and-cuda]`)
- Declare the extra conflicts in `tool.uv.conflicts`.
- Cap supported Python versions at `>=3.11,<3.13`.
- Keep `soundfile` as a direct base dependency because extraction and FLAC tooling
  import it directly regardless of TensorFlow selection.

**Consequences**:
- Each environment must select exactly one TensorFlow extra; `uv sync --all-extras`
  is invalid by design.
- macOS and Linux deployments can resolve different TensorFlow stacks without
  weakening the shared base dependency set.
- The lockfile must be regenerated after TensorFlow dependency changes so platform
  forks stay explicit.
- Python 3.13 is intentionally unsupported until TensorFlow compatibility is validated.

---

## ADR-023: Env-driven deployment config and FastAPI trusted-host enforcement

**Date**: 2026-03
**Status**: Accepted

**Context**: Deployment-specific edits were being made directly to tracked files such
as `src/humpback/config.py` and `frontend/vite.config.ts` on remote hosts. That does
not survive the checked-in deployment flow because `scripts/deploy.sh` resets the repo
to `origin/main`. Production also serves the built SPA from FastAPI on port `8000`, so
Vite `allowedHosts` is not the correct control for deployed host validation.

**Decision**:
- Load deployment/runtime overrides from a repo-root `.env` file plus normal process
  environment variables, but do so explicitly in production entrypoints rather
  than in every `Settings()` construction.
- Allow `.env` to include both `HUMPBACK_*` runtime settings and deploy-time values
  like `TF_EXTRA`; the app settings loader ignores unknown keys.
- Add FastAPI bind settings `api_host` / `api_port` with defaults `0.0.0.0` / `8000`.
- Add FastAPI `TrustedHostMiddleware` driven by `HUMPBACK_ALLOWED_HOSTS`.
- Keep host validation permissive by default (`*`) so existing installs are not broken.
- Derive default extraction/cache paths from `storage_root` when the explicit path
  settings are unset.

**Consequences**:
- Deployment-specific paths and host allowlists no longer require tracked-file edits.
- Cloudflare tunnel deployments should use `HUMPBACK_API_HOST=0.0.0.0` and trusted
  host patterns like `*.trycloudflare.com`.
- Tests and library callers that instantiate `Settings()` directly remain
  hermetic and do not read deployment-local `.env` files from the cwd.
- Production host validation now lives in FastAPI; Vite `allowedHosts` remains a
  dev-server-only concern.
- No database migration is required because the change is limited to runtime
  configuration, deploy scripting, tests, and documentation.

---

## ADR-024: Promote NOAA Glacier Bay to a first-class ArchiveProvider on legacy hydrophone APIs

**Date**: 2026-03
**Status**: Accepted

**Context**: The ArchiveProvider abstraction already decouples the detection pipeline
from Orcasound HLS internals, but the user-facing worker/router/service wiring still
assumed every remote source was an Orcasound hydrophone. We also had a standalone
NOAA Glacier Bay GCS POC in `scripts/noaa_gcs_poc.py`, which proved archive access
but duplicated logic instead of participating in the production provider path.

**Decision**:
- Promote the NOAA Glacier Bay Bartlett Cove archive to `NoaaGCSProvider` in
  `src/humpback/classifier/providers/noaa_gcs.py`.
- Reuse the existing `/classifier/hydrophones` and
  `/classifier/hydrophone-detection-jobs` surfaces for this phase rather than
  renaming APIs immediately; `hydrophone_id` now semantically carries a generic
  archive `source_id`.
- Add a static archive-source registry containing the existing Orcasound HLS
  sources plus `noaa_glacier_bay`.
- Keep Orcasound behavior unchanged:
  detection uses local-cache -> S3-cache -> direct-S3 priority, while playback
  and extraction remain local-cache-authoritative.
- Make NOAA detection, playback, and extraction use direct anonymous GCS fetch;
  `local_cache_path` is rejected for NOAA sources.
- Keep the database schema unchanged; source selection is resolved from the
  static registry by `hydrophone_id`.

**Consequences**:
- NOAA Glacier Bay detection jobs now run through the same worker/router/provider
  pipeline as Orcasound jobs with no consumer-specific code paths in the
  detector/extractor core.
- `google-cloud-storage` becomes a runtime dependency because NOAA access is no
  longer POC-only.
- The legacy hydrophone API naming is now broader than its semantics; a follow-up
  backlog item is required to rename/archive-generalize those surfaces.
- No database migration is required because the change is limited to provider
  selection, runtime dependencies, tests, and documentation.

---

## ADR-025: NOAA GCS local caching provider

**Date**: 2026-03
**Status**: Accepted

**Context**: NOAA Glacier Bay detection, playback, and extraction used direct
anonymous GCS fetch for every request. This worked but was slow for repeated
access to the same time ranges (re-runs, playback, extraction) and consumed
unnecessary network bandwidth when segment data was already available locally.

**Decision**:
- Add a `CachingNoaaGCSProvider` that caches NOAA metadata manifests and `.aif`
  segment files locally with GCS fallback on cache miss.
- Add `noaa_cache_path` setting (defaults to `{storage_root}/noaa-gcs-cache`)
  to control the local cache root.
- Add factory helpers `build_noaa_detection_provider` and
  `build_noaa_playback_provider` that return `CachingNoaaGCSProvider` when
  `noaa_cache_path` is configured, or the direct `NoaaGCSProvider` when it is
  `None`.
- Route all NOAA archive consumers (detection worker, playback router,
  extraction worker) through the factory helpers so caching is applied
  uniformly.

**Consequences**:
- Repeated NOAA detection/playback/extraction on the same time ranges avoids
  redundant GCS network fetches after the first access.
- Cache layout mirrors GCS object paths under `noaa_cache_path` for
  straightforward inspection and cleanup.
- Direct `NoaaGCSProvider` remains available as a fallback when
  `noaa_cache_path` is explicitly set to `None`.
- No database migration required; the change is limited to provider selection,
  runtime configuration, and tests.

---

## ADR-026: Positive extraction windows come from stored detection diagnostics

**Date**: 2026-03
**Status**: Accepted

**Context**: Retraining labels are saved at multi-second clip granularity, but classifier
training consumes 5-second embeddings. Blindly splitting a labeled-positive clip into fixed
5-second halves creates mislabeled positives when the vocalization only occupies part of the
clip. Re-running inference during extraction would duplicate work and can drift from the exact
detection-job scores that users reviewed.

**Decision**:
- Treat persisted 1-second-hop detection diagnostics as the source of truth for positive
  extraction window selection.
- For each positive labeled row (`humpback` or `orca`), smooth candidate 5-second window
  scores with a short moving average, select the peak smoothed window, and skip the row when
  the peak is below a configurable minimum score.
- Persist hydrophone diagnostics incrementally as Parquet shards so paused/canceled jobs can
  extract positives without re-running inference.
- Store row-level selection provenance back into the detection TSV via
  `positive_selection_*` columns plus `positive_extract_filename`.
- Keep classifier rescoring only as a legacy fallback for jobs missing diagnostics.

**Consequences**:
- Positive extraction is faster and stays consistent with the original detection-job scores.
- Hydrophone jobs gain durable per-window diagnostics even before full completion.
- Detection TSVs now carry both label state and positive-window provenance, and label-save
  must preserve those extra columns.
- No database migration required; the change lives in job artifacts, extraction config, API
  parsing, tests, and documentation.

---

## ADR-027: Positive extraction can widen beyond one 5-second window

**Date**: 2026-03
**Status**: Accepted

**Context**: ADR-026 improved positive extraction by selecting the best-scoring 5-second
window from stored 1-second-hop diagnostics, but some labeled rows still contain meaningful
vocalization beyond that single window. For longer calls, exporting only the peak 5-second
clip discards adjacent high-confidence audio that should stay in the training example.

**Decision**:
- Keep the ADR-026 seed-selection rule: select the best smoothed 5-second window and skip
  the row when its peak is below `positive_selection_min_score`.
- After selecting that seed, allow the extracted positive clip to widen by exact adjacent
  5-second chunks.
- Evaluate each adjacent chunk using the smoothed score of the aligned 5-second candidate
  window at that chunk start.
- Add an adjacent chunk only when its smoothed score is at or above the new
  `positive_selection_extend_min_score` threshold.
- If both sides qualify at once, extend the higher-scoring side first and then re-evaluate.
- Continue growth until neither adjacent chunk qualifies or the labeled-row boundary is hit.

**Consequences**:
- Positive extraction can now emit 10-second, 15-second, or longer clips when the score
  support justifies it, while still keeping durations as multiples of the classifier window.
- Existing provenance fields remain sufficient; widened clips are recorded through the
  selected `positive_selection_start_sec`, `positive_selection_end_sec`, and
  `positive_extract_filename`.
- Legacy rescoring fallback follows the same widening rule, so older jobs and new jobs
  behave consistently.
- No database migration required; the change is limited to extraction logic, config/API
  defaults, tests, and documentation.

---

## ADR-028: Detection jobs use a canonical Parquet row store for editable row state

**Date**: 2026-03
**Status**: Accepted

**Context**: Detection jobs were using TSV files as both the download artifact and the mutable
source of record for labels/extraction provenance. That made downstream flows brittle: some UI
state (for example spectrogram markers and manual window edits) depended on later extraction
steps or legacy filename joins instead of the detection job itself carrying durable row state.

**Decision**:
- Add a canonical `detection_rows.parquet` artifact per detection job and store its path on the
  detection job record.
- Persist stable `row_id`, detection-time `auto_positive_selection_*`, manual override bounds,
  effective `positive_selection_*`, and `positive_extract_filename` in that row store.
- Keep `detections.tsv` as a synchronized export/download adapter rather than the editable source
  of truth.
- Populate detection-time auto-selection data during row-store creation, even before a row is
  labeled positive, so later label saves and popup edits can reuse the same stored window data.
- Add a row-level API mutation (`PUT /classifier/detection-jobs/{id}/row-state`) that atomically
  persists one row's labels plus optional manual bounds.

**Consequences**:
- Spectrogram markers no longer depend on running extraction first; completed jobs can render
  stored auto/effective bounds immediately.
- Manual window editing becomes durable and updates the same source of record extraction reads.
- Legacy TSV-only jobs can be lazily upgraded into the row store without losing download
  compatibility.
- A database migration is required only to persist `output_row_store_path`; the rest of the
  change lives in job artifacts, API behavior, workers, tests, and UI.

---

## ADR-029: TF2 hydrophone detection runs in a short-lived subprocess

**Date**: 2026-03
**Status**: Accepted

**Context**: Recent hydrophone detection profiling showed that warm-cache
Orcasound runs using the `surfperch-tensorflow2` embedding backend slowed down
substantially after the long-lived worker had processed multiple TF2 jobs.
The profiled Orcasound Lab range was already fully populated in the disk-backed
write-through cache, so repeated S3 segment downloads were not the primary
bottleneck. The stronger signal was long-lived TensorFlow/Metal memory growth in
the worker process.

**Decision**:
- Keep the existing hydrophone detection workflow, archive-provider selection,
  progress callbacks, diagnostics persistence, and pause/resume/cancel behavior.
- When a hydrophone job resolves to a TF2 SavedModel embedding backend
  (`model_type="tf2_saved_model"`, `input_format="waveform"`), execute the
  hydrophone detection loop in a spawned subprocess instead of the long-lived
  worker process.
- Load the classifier pipeline and TF2 embedding model inside that child
  process, then communicate chunk progress, diagnostics, alerts, resume
  invalidation, and final results back to the parent worker over a queue.
- Keep TFLite hydrophone detection and local-file detection on the existing
  in-process path.
- Extend hydrophone run summaries with provider/runtime metadata:
  `provider_mode`, `execution_mode`, `avg_audio_x_realtime`,
  `peak_worker_rss_mb`, and `child_pid` when subprocess mode is used.

**Consequences**:
- TF2 hydrophone jobs release TensorFlow/Metal state when the child exits,
  preventing memory buildup from degrading later jobs in the long-lived worker.
- The parent worker remains the single owner of SQL status transitions and
  artifact persistence, so UI behavior for active/paused/canceled jobs stays
  consistent with the existing hydrophone workflow.
- Hydrophone run summaries now distinguish cache/provider mode from execution
  mode, making warm-cache vs runtime-memory regressions easier to diagnose.
- No database migration is required; the change is limited to worker
  orchestration, summary metadata, tests, and documentation.

---

## ADR-030: NOAA archive sources are metadata-driven

**Date**: 2026-03
**Status**: Accepted

**Context**: NOAA support started with a single hard-coded Glacier Bay
Bartlett Cove source. That was enough for the initial provider rollout, but it
did not scale to other NOAA archive families such as SanctSound, could not
capture partitioned child-folder layouts, and made UI exposure a code change
instead of a metadata decision.

**Decision**:
- Add packaged NOAA archive metadata in
  `src/humpback/data/noaa_archive_sources.json`.
- Keep the full verified metadata set in the repo, including reference-only
  records that are not yet loadable or visible in the UI.
- Derive runtime NOAA archive sources from metadata records that have concrete
  `bucket` and `prefix` values.
- Keep the legacy Bartlett Cove runtime/source ID `noaa_glacier_bay` for API
  and job compatibility, while storing its canonical slug as
  `nps_glacier_bay_bartlettcove`.
- Add `include_in_detection_ui` so the legacy `/classifier/hydrophones`
  endpoint can expose only the currently supported NOAA sources without
  discarding the rest of the verified metadata.
- Extend the NOAA provider to use metadata-driven root prefixes,
  optional `child_folder_hints`, and broader filename parsing across Glacier
  Bay and SanctSound naming conventions.

**Consequences**:
- NOAA source expansion becomes a metadata change first, rather than another
  hard-coded config edit.
- The hydrophone detection UI now exposes two NOAA choices: `sanctsound_ci01`
  and legacy `noaa_glacier_bay`.
- Partitioned NOAA archives can avoid unnecessary root-prefix scans when
  verified child-folder hints are available, while still falling back safely
  when hints are missing or incomplete.
- No database migration is required; the change is limited to packaged
  metadata, provider/config behavior, tests, and documentation.

## ADR-031: Windowed detection mode with NMS peak selection

**Date**: 2026-03
**Status**: Accepted

**Context**: Detection jobs produce variable-length detections (10–20+ seconds)
due to hysteresis merging of overlapping 1-sec-hop windows. Users must manually
review each long detection's spectrogram and select the best 5-sec sub-window
for positive extraction. This manual positive-selection step is the primary
bottleneck in the labeling workflow. The 1-sec hop is important for detection
sensitivity but the merging creates UX problems.

**Decision**:
- Add a `detection_mode` column to `detection_jobs` (nullable; `NULL`/"merged"
  preserves existing behavior, `"windowed"` enables the new mode).
- Windowed mode keeps the full pipeline (1-sec hop → score → hysteresis merge
  → snap) for sensitivity, then applies NMS within each merged event to output
  only non-overlapping peak 5-sec windows above `high_threshold`.
- Long events with multiple distinct vocalizations produce multiple peak
  windows (NMS within each event), not just the single best.
- For windowed jobs, auto-positive-selection is trivially set to the full row
  bounds (the detection IS the positive window). The spectrogram editor's
  window-shifting controls are hidden.
- Extraction of windowed detections uses clip bounds directly — no
  `select_positive_window()` call needed.

**Consequences**:
- Labeling workflow for windowed jobs is just positive/negative — no sub-window
  selection needed.
- Each windowed detection produces exactly one training embedding (1:1 mapping
  between labeled detection and training vector).
- Existing merged-mode jobs are unaffected; `detection_mode=NULL` is treated as
  `"merged"`.
- Requires Alembic migration 018.

---

## ADR-032: Standard cosine similarity for cross-corpus embedding search

**Date**: 2026-03
**Status**: Accepted

**Context**: The existing `_cosine_similarity_matrix()` in `audio.py` uses mean-centered
cosine similarity, which removes the shared ReLU baseline direction and works well for
within-file pairwise comparison. For the new cross-corpus embedding search
(`POST /search/similar`), the corpus mean changes depending on which files are included,
making mean-centering unstable.

**Decision**: Use standard (non-mean-centered) cosine similarity for cross-corpus search.
Implement brute-force search over parquet files with an LRU cache (128 entries) for loaded
embeddings. Defer vector index (FAISS, USearch) and on-the-fly embedding to future phases.

**Alternatives considered**:
- Mean-centered cosine: unsuitable because the mean depends on corpus composition, making
  scores non-comparable across different search sets.
- Vector database (FAISS, USearch): unnecessary overhead at current scale (thousands to tens
  of thousands of embeddings); the search service is designed as a single substitution point
  if an index is needed later.
- On-the-fly embedding in the API process: conflicts with the architecture that isolates
  model loading to workers; deferred to Phase 1b.

**Consequences**:
- Search results use standard cosine similarity, which may differ from the within-file
  similarity matrix displayed in the audio detail view.
- The brute-force approach is O(N) in total embeddings; adequate at current scale but
  will need replacement if the corpus grows to millions of vectors.
- The LRU cache bounds memory usage while avoiding repeated parquet reads for hot sets.

---

## ADR-033: Detection embedding storage for similarity search

**Date**: 2026-03
**Status**: Accepted

**Context**: The Search Results UI (Phase 2) needs to search for audio similar to a
detection result. Detection rows don't map to existing embedding sets — the detector
computes embeddings per window but discards them after classification. To enable
"Search Similar" from detection rows, the system needs to persist the representative
embedding for each detection output row.

**Decision**: Store per-detection peak-window embeddings in
`{detection_dir}/detection_embeddings.parquet` during detection. The detector identifies
the peak-confidence window within each detection event's bounds and extracts its embedding
vector. The parquet schema is `(filename, start_sec, end_sec, embedding)`. A new
`POST /search/similar-by-vector` endpoint accepts a raw vector for search, reusing the
existing `_brute_force_search()` core. No database column is needed — the parquet path
is derived from the detection job's output directory.

**Alternatives considered**:
- On-the-fly re-embedding: requires loading the model in the API process, conflicting with
  the architecture that isolates model loading to workers.
- Storing embeddings in the database: vectors are large (1536 floats × 4 bytes = 6KB each)
  and parquet is the existing storage pattern for embeddings.
- Storing all window embeddings: wasteful; only the peak-confidence window per detection
  event is needed for similarity search.

**Consequences**:
- Detection jobs run after this change automatically produce `detection_embeddings.parquet`.
- Pre-existing detection jobs have no stored embeddings; the Search page shows an
  appropriate message when the embedding retrieval returns 404.
- The embedding file adds modest disk overhead (one vector per detection event).

---

## ADR-034: Worker-encoded detection search via ephemeral SearchJob

**Date**: 2026-03
**Status**: Accepted

**Context**: Detection-sourced similarity search relied on pre-stored embeddings in
`detection_embeddings.parquet` (ADR-033). Pre-existing detection jobs (created before
ADR-033) had no stored embeddings, causing a 404 error when clicking "Search Similar".
Loading the embedding model in the API process would conflict with the architecture
that isolates model loading to workers.

**Decision**: Add an async search flow using a new `SearchJob` model. The API queues
a lightweight search job (`POST /search/similar-by-audio`), the worker claims it,
resolves the detection audio (local or hydrophone), encodes it using the detection
job's classifier model, and stores the embedding vector on the search job row. The
frontend polls `GET /search/jobs/{id}` until complete, at which point the API runs the
brute-force search synchronously and returns results. The search job row is deleted
after results are returned (ephemeral cleanup). Search jobs are prioritized first in
the worker loop since encoding is sub-second interactive work.

**Alternatives considered**:
- Backfilling old detection jobs: expensive for large job histories, and doesn't help
  if the detection output directory is missing.
- Loading the model in the API process: conflicts with worker-isolated model loading
  architecture.
- Direct embedding fetch with fallback: the 404 case is the default for most existing
  jobs, so fallback would be the common path.

**Consequences**:
- "Search Similar" works on all detection jobs, including pre-existing ones.
- The `search_jobs` table is ephemeral — rows are created, processed, and deleted
  within seconds. No long-term storage impact.
- Worker poll loop checks search jobs first, before processing/clustering/detection.
- Frontend detection mode no longer uses `useDetectionEmbedding` or
  `useSearchByVector`; it uses the new mutation + poll pattern.

---

## ADR-035: Runtime path derivation for detection artifacts

**Date**: 2026-03
**Status**: Accepted

**Context**: 33 of 52 detection jobs stored relative file paths (`data/detections/...`)
in the `output_tsv_path` and `output_row_store_path` DB columns from when `storage_root`
was the default `Path("data")`. After the storage root was changed to an absolute
external drive path, these relative paths no longer resolved from the API's CWD. All
detection data was intact on disk — only the path resolution was broken.

**Decision**: Derive all detection artifact paths at runtime from
`detection_dir(settings.storage_root, job.id)` plus fixed filenames, using new
`storage.py` helpers (`detection_tsv_path()`, `detection_row_store_path()`,
`detection_diagnostics_path()`, `detection_embeddings_path()`). The worker no longer
writes `output_tsv_path` or `output_row_store_path` to the DB. The API router no
longer reads these columns; all path resolution goes through `SettingsDep`. The DB
columns are kept vestigial for now; a follow-up cleanup is in the backlog.

**Alternatives considered**:
- Backfilling relative paths to absolute: fragile if storage root changes again.
- Storing absolute paths: still redundant since paths are fully deterministic from
  `storage_root + job.id + fixed filename`.

**Consequences**:
- All 52 detection jobs are now accessible regardless of the stored path values.
- `SettingsDep` was added to 6 API endpoints that previously lacked it.
- `output_tsv_path` and `output_row_store_path` columns are vestigial; a backlog
  item tracks their removal via Alembic migration.

---

## ADR-036: LabelProcessingJob as a dedicated DB table

**Date**: 2026-03
**Status**: Accepted

**Context**: A new "Audio/Label Processing" workflow takes Raven annotation files
paired with audio recordings and uses classifier scores as a segmentation signal
to extract clean 5-second samples organized by call type and treatment category.
Unlike detection jobs which reuse the DetectionJob table with nullable hydrophone
columns, this workflow has sufficiently different inputs (annotation folders, call
type labels, treatment categories) and outputs (no TSV row store, no hydrophone
fields) that reusing DetectionJob would further widen an already-wide table.

**Decision**: Create a dedicated `label_processing_jobs` table
(`LabelProcessingJob` model) with fields specific to the annotation-based
workflow: `classifier_model_id`, `annotation_folder`, `audio_folder`,
`output_root`, `parameters` (JSON), `files_processed/total`,
`annotations_total`, `result_summary` (JSON), and `error_message`. Embeddings
generated during scoring are ephemeral — used only for classifier scoring, not
stored in the main EmbeddingSet pipeline.

**Consequences**:
- Clean separation of concerns; DetectionJob stays focused on detection.
- New migration `020_label_processing_jobs.py`.
- Worker queue extended with claim/complete/fail functions and stale recovery.
- API router at `/label-processing` with job CRUD and annotation preview.

---

## ADR-037: Annotation-guided synthesis with adaptive background threshold

**Date**: 2026-03
**Status**: Accepted

**Context**: The label processing synthesis pipeline had three related issues
causing poor training data quality:

1. **Shared-peak label contamination**: `isolate_call_segment()` centred audio
   extraction on the classifier peak's position, ignoring annotation bounds.
   When multiple nearby annotations of different call types shared a peak (64%
   of peaks in test data), they all got identical audio with different labels.

2. **No synthesis in dense recordings**: The fixed `background_threshold` (0.1)
   was too strict for recordings with elevated classifier baselines, causing
   `extract_background_regions()` to find zero qualifying runs.

3. **Repetitive backgrounds**: Even when backgrounds were found, all annotations
   in a recording cycled through the same small pool deterministically.

**Decision**: Three targeted changes to the synthesis pipeline:

- **Annotation-guided call isolation**: `isolate_call_segment()` accepts an
  optional `annotation` parameter.  When provided, the extracted segment centres
  on the annotation midpoint and uses the annotation duration (clamped to
  1–3 s), ensuring each annotation gets audio from its own labelled region.

- **Adaptive per-recording background threshold**: A new helper
  `_compute_adaptive_bg_threshold()` computes the 25th percentile of all
  smoothed scores, clamped to `[0.05, 0.5]`.  This replaces the static 0.1
  threshold when `background_threshold_auto=True` (default), allowing dense
  recordings to produce background segments from their quieter regions.
  Short runs (≥ `background_min_duration`, default 1.0 s) are tiled to fill
  the 5 s synthesis canvas with up to 3 shifted variants per run.

- **Background pool rotation**: `synthesize_variants()` accepts a `bg_offset`
  parameter; `process_recording()` increments it per annotation so successive
  annotations start at different positions in the background pool.

**Consequences**:
- Synthesised files are now annotation-specific: filenames use
  `annotation.begin_time` instead of `peak.time_sec`.
- Dense recordings that previously produced zero synthesis output now produce
  backgrounds proportional to their quiet-region count.
- Two new configurable parameters: `background_threshold_auto` (bool) and
  `background_min_duration` (float).
- No database migration required — pure algorithm change.


---

## ADR-038: Sample builder contamination screening tuned for marine recordings

**Date**: 2026-03-21
**Status**: Accepted
**Context**: The sample builder pipeline rejected 100% of annotations (1514/1514) from real marine field recordings (Emily Vierling humpback dataset). Two root causes: (1) contamination screening thresholds designed for synthetic white noise at amplitude 0.001 failed on colored (pink/red) ocean ambient noise; (2) annotation duration bounds [0.3s, 4.0s] were too restrictive for the range of humpback vocalizations.

**Decision**: Four signal-processing algorithm changes:

1. **Tonal persistence: per-bin median threshold** — Changed `_tonal_persistence` from global median (across all bins and frames) to per-bin median (each bin compared to its own baseline). Pink noise has 20-30 dB more energy at low frequencies; the global median was pulled low by quiet high-frequency bins, causing all low-frequency bins to appear "persistently active." Per-bin median normalizes for spectral shape. Added configurable `persistence_margin_db` (default 10.0 dB) to `ContaminationConfig`. Trade-off: constant tones present throughout a fragment become invisible to persistence detection; the other three features (RMS, occupancy, transient) still catch loud or sudden contamination.

2. **Spectral occupancy: raised noise floor** — Changed defaults from `-40 dB / 0.3` to `-10 dB / 0.8`. At -40 dB, ocean ambient noise activated >99% of FFT bins. At -10 dB, typical marine backgrounds (amp 0.005-0.02) score 0.06-0.24 occupancy. Spectral occupancy has inherently poor separation for tonal contamination in colored noise (a tone adds 1-2 bins to 513), so it now serves as a broadband-only backstop.

3. **Validation: relaxed splice energy ratio and averaged spectral correlation** — Raised `splice_energy_ratio_max` from 10.0 to 1000.0 because background-to-call transitions inherently have large energy ratios (25-250x) that the crossfade smooths into gradual transitions, not audible artifacts. Changed `_spectral_correlation` from single-FFT to frame-averaged Welch-style power spectrum so spectral *shape* (e.g. 1/f) is compared rather than random per-frame fluctuations in short noise segments.

4. **Widened annotation duration bounds** — `SampleBuilderConfig` defaults changed from [0.3s, 4.0s] to [0.1s, 10.0s] to accommodate brief clicks and extended songs/moans.

All contamination and annotation config parameters are now exposed through the worker's job parameters for per-job tuning.

**Consequences**:
- Marine field recordings should achieve non-zero acceptance rates with default settings.
- Contamination detection is more permissive overall; users needing stricter screening can override via job parameters.
- Per-bin persistence cannot detect constant tones — accepted trade-off since constant recording-wide tones are effectively background.
- No database migration required — pure algorithm and default-value changes.

---

## ADR-039: Retire merged detection mode from the public creation/edit surface

**Date**: 2026-03-22
**Status**: Accepted

**Context**: Windowed detection mode solved the manual positive-selection bottleneck and became the operational default, but the product still exposed merged mode in the API, Hydrophone UI, and helper scripts. Keeping both modes on the public surface increased maintenance cost, preserved edit paths that only mattered for legacy jobs, and made it harder to backfill older merged outputs into the windowed workflow.

**Decision**:
- Remove `detection_mode` from detection-job creation requests in both local and hydrophone APIs; new jobs are always persisted as `"windowed"`.
- Reject create payloads that still send `detection_mode` instead of silently ignoring the obsolete field.
- Remove the Hydrophone UI mode selector and treat legacy merged jobs (`NULL` or `"merged"`) as read-only.
- Preserve legacy merged read paths (`GET` job/list, `/download`, `/content`) temporarily so historical jobs can still be inspected during manual backfill.
- Reject label-save, row-state, and extraction operations for legacy merged jobs with a rerun-in-windowed-mode error.
- Keep the `detection_mode` DB column for now so legacy rows remain distinguishable; defer schema cleanup until after manual backfill and artifact cleanup.

**Consequences**:
- The public detection workflow is now windowed-only.
- Hydrophone spectrogram-bound editing is effectively retired because it only applied to merged jobs.
- Legacy merged jobs remain visible for audit/download purposes but can no longer be modified or extracted.

---

## ADR-040: Timeline Viewer Tile Architecture

**Status:** Accepted
**Date:** 2026-03-24

**Context:** Need a zoomable spectrogram viewer for hydrophone detection jobs spanning up to 24 hours.

**Decision:** Canvas 2D with pre-colored PNG tiles at 6 discrete zoom levels. Ocean Depth colormap baked into tiles. Coarse levels pre-rendered on job completion, fine levels rendered on demand with global FIFO cache. Timeline audio resolved from HLS local cache via `resolve_timeline_audio()`.

**Alternatives considered:**
- WebGL shader rendering (rejected: over-complex for discrete zoom levels)
- On-demand only (rejected: minimap/initial view would be slow)

**Consequences:** Simple frontend (no WebGL), human-inspectable tile cache, colormap changes require re-render.
- No Alembic migration is required in this phase because the schema is intentionally retained for compatibility.

---

## ADR-040: Split SanctSound umbrella archive IDs from site-scoped IDs

**Date**: 2026-03-24
**Status**: Accepted

**Context**: NOAA archive sources are metadata-driven, but the Hydrophone UI
surfaced `sanctsound_ci01` and `sanctsound_oc01` as if they were single-site
sources. In reality those records acted as region umbrellas and included
overlapping child-folder hints from multiple sites. That made job IDs
misleading, allowed progress to exceed the requested wall-clock range without
clear explanation, and caused deployment-specific workflows to reuse IDs with
the wrong semantics.

**Decision**:
- Add explicit umbrella source IDs `sanctsound_ci` and `sanctsound_oc` for the
  UI-visible Channel Islands and Olympic Coast archive choices.
- Keep `sanctsound_ci01` and `sanctsound_oc01` as hidden site-scoped sources
  whose child-folder hints only reference same-site deployment folders.
- Migrate historical detection jobs from legacy umbrella-in-site IDs
  (`sanctsound_ci01`, `sanctsound_oc01`) to the new umbrella IDs so playback,
  extraction, and job history keep their original semantics.
- Keep progress accounting as summed processed audio duration, but clarify the
  UI wording so umbrella sources can legitimately exceed the selected
  wall-clock range when multiple overlapping site feeds are processed.

**Consequences**:
- The Hydrophone UI keeps one visible Channel Islands source and one visible
  Olympic Coast source, but their IDs now match their umbrella behavior.
- Scripted or metadata-driven workflows can still target exact sites with
  hidden IDs such as `sanctsound_ci01` and `sanctsound_oc01`.
- Future SanctSound metadata mistakes are easier to catch because site-scoped
  records now have explicit same-site child-folder expectations.
- Added Alembic migration `025_normalize_sanctsound_source_ids.py` to preserve
  historical job semantics on existing databases.

---

## ADR-041: Adopt superpowers workflow, consolidate documentation

**Date**: 2026-03-24
**Status**: Accepted

**Context**: The project had 6 repo-root .md files with overlapping concerns and
6 custom session-* skills that duplicated superpowers functionality while missing
key capabilities (brainstorming, TDD enforcement, subagent execution, code review).

**Decision**: Adopt superpowers as the canonical workflow. Consolidate to 3 repo-root
files (CLAUDE.md, DECISIONS.md, AGENTS.md). Move specs to docs/specs/, plans to
docs/plans/. Rewrite AGENTS.md for Codex-compatible workflow.

**Consequences**:
- Single workflow system instead of two competing ones
- CLAUDE.md is larger (~450 lines) but self-contained
- Codex follows same phase sequence with its own tooling
- Session-* skills deleted; all workflow orchestration via superpowers
- Backlog items preserved in docs/plans/backlog.md

---

## ADR-042: Multi-label vocalization type classifier via binary relevance

**Date**: 2026-03-29
**Status**: Accepted

**Context**: The existing vocalization labeling system used single-label
classification on the binary classifier infrastructure. Windows can contain
multiple overlapping vocalization types (e.g., a whup during a moan), making
single-label assignment a poor fit. The annotation sub-window system added
complexity without adoption.

**Decision**: Replace the single-label vocalization classifier with a standalone
multi-label system using binary relevance (one independent sklearn pipeline per
vocalization type). Key design choices:

1. **Managed vocabulary** — `vocalization_types` table with unique names,
   importable from embedding set folder structure.
2. **Binary relevance** — N independent classifiers, one per type. A window
   labeled with types A and B is positive for both A and B pipelines, negative
   for neither. Types below `min_examples_per_type` are filtered out.
3. **Per-type threshold optimization** — each type gets an F1-maximized
   threshold from cross-validation, stored in the model and overridable at
   inference time.
4. **Dedicated tables** — `vocalization_types`, `vocalization_models`,
   `vocalization_training_jobs`, `vocalization_inference_jobs` — fully
   independent from the binary classifier tables.
5. **Three inference source types** — `detection_job` (with UTC), `embedding_set`
   (from curated sets), `rescore` (re-run previous results with a new model).
6. **Remove legacy systems** — dropped sub-window annotation system
   (`labeling_annotations` table) and the old single-label vocalization
   training endpoints from `/labeling/`.

**Consequences**:
- Windows can carry multiple type labels simultaneously
- Per-type thresholds allow precision/recall tuning per vocalization category
- Vocabulary is managed independently from training data
- Inference results are persistent parquet files, not ephemeral
- Old `/labeling/training-jobs`, `/labeling/vocalization-models`, `/labeling/predict`,
  and active learning endpoints removed; replaced by `/vocalization/` router
- `label_trainer.py` deleted; replaced by `vocalization_trainer.py`
