# Project Status

Current state of the humpback acoustic embedding and clustering platform.

---

## Implemented Capabilities

### Audio Management
- Upload individual audio files (MP3, WAV, FLAC) with copy to `audio/raw/`
- Import audio folders in-place (no copy, reads from `source_folder`)
- Audio metadata attachment and editing
- Audio detail view with processing history

### Processing Pipeline
- Configurable model registry (TFLite + TF2 SavedModel) with auto-detection of input format and vector dim (scanner + runtime safety net)
- Audio decoding, resampling (32 kHz), overlap-back windowing (5s default)
- Log-mel spectrogram extraction (128x128) for TFLite models
- Raw waveform input for TF2 SavedModel models
- Vectorized batch spectrogram extraction (`extract_logmel_batch`) via single `np.fft.rfft` call (10.9x faster than per-window)
- Batch TFLite inference via `resize_tensor_input` (batch_size=64, optimal for M-series) with automatic sequential fallback
- Multi-threaded TFLite interpreter (default `os.cpu_count()` threads); XNNPACK delegate warnings suppressed
- Processing timing instrumentation (decode, features, inference) in worker, detector, and trainer
- Incremental Parquet embedding output with atomic writes
- Idempotent encoding via `encoding_signature`
- Background worker with job queue (queued/running/complete/failed/canceled) and atomic compare-and-set claim updates
- Periodic stale job recovery every 60s in worker loop (previously startup-only); prevents jobs stuck in `running` after quick restarts
- Local detection job TSV cleanup on restart (prevents duplicate appends from stale prior runs)

### Embedding Similarity Search
- `POST /search/similar` — brute-force cosine or euclidean search across all embedding sets for the same model version
- `POST /search/similar-by-vector` — search using a raw embedding vector (for detection-sourced queries)
- `GET /classifier/detection-jobs/{id}/embedding` — retrieve stored detection embedding for a row
- `GET /audio/{id}/spectrogram-png` — PNG spectrogram for arbitrary time range (cached)
- LRU cache (128 entries) for loaded parquet embeddings avoids repeated I/O
- Standard cosine similarity (not mean-centered) for stable cross-corpus search
- Supports `top_k`, `exclude_self`, `embedding_set_ids` filter, and `metric` selection
- Returns ranked hits with audio file metadata and `window_offset_seconds`
- Detection jobs store per-detection peak-window embeddings in `detection_embeddings.parquet`
- `POST /search/similar-by-audio` — async worker-based encoding: queues a search job, worker encodes detection audio via the model, frontend polls for results
- `GET /search/jobs/{id}` — poll search job status; returns search results on completion and deletes the ephemeral job row
- Search tab UI: standalone search (pick embedding set + window) and detection-sourced search ("Search Similar" from detection rows — works on all detection jobs including pre-existing ones without stored embeddings)
- Embedding set selection: folder-tree multi-select "Embedding Sets" panel for filtering which sets to search across (defaults to all selected); detection re-search uses cached query vector for instant results
- Standalone search: dual-panel layout with "Search Query Embedding" (single-select folder tree to file level) and "Embedding Sets" (multi-select filter)
- `GET /search/jobs/{id}` returns `query_vector` and `model_version` on completion for frontend re-search without worker roundtrip
- Spectrogram thumbnail previews and inline playback in search results

### Clustering
- Model version validation: rejects clustering across different embedding models
- HDBSCAN, K-Means, Agglomerative clustering algorithms
- UMAP, PCA, or no dimensionality reduction
- Euclidean and cosine distance metrics
- Internal metrics: Silhouette, Davies-Bouldin, Calinski-Harabasz
- Supervised metrics: ARI, NMI, homogeneity, completeness, v_measure, confusion matrix
- Parameter sweep (HDBSCAN + K-Means ranges)
- Fragmentation report (entropy, Gini, noise rates)
- Opt-in: classifier baseline, stability evaluation, metric learning refinement
- Re-clustering from refined embeddings via `refined_from_job_id`
- Clustering job validation requires at least one embedding set ID

### Binary Classifier
- Train LogisticRegression (default) or MLPClassifier from positive + negative embedding sets
- Optional L2 normalization of embeddings before scaling
- Balanced class weights by default
- Cross-validation metrics (accuracy, ROC-AUC, precision, recall, F1)
- Decision boundary diagnostics: score separation (d-prime), train confusion matrix
- Overlap validation: rejects same embedding set in both positive and negative lists
- Encoding signature consistency warning when mixing different processing configs
- Detection job: scan audio folder with configurable `hop_seconds` (default 1.0s)
- Hysteresis event merging with dual thresholds (`high_threshold`/`low_threshold`)
- Per-event `n_windows` count in TSV output
- Canonical snapped detection bounds (`start_sec`/`end_sec`) before labeling/extraction
- Raw event audit metadata in TSV (`raw_start_sec`, `raw_end_sec`, `merged_event_count`)
- Windowed detection mode (`detection_mode="windowed"`): NMS peak selection within merged events produces fixed 5-sec detections, eliminating manual positive-selection in the labeling workflow
- Incremental detection results: file-by-file progress with live UI updates during job execution
- Inline audio playback and label annotation available while detection is still running
- Positive labeled extraction seeds from the best 5-second training clip in stored
  1-second-hop detection scores, then can widen in adjacent 5-second chunks when
  the neighboring smoothed score stays above the extension threshold (with classifier
  fallback for legacy jobs); negative extraction keeps canonical labeled bounds and writes
  FLAC clips with same-basename marker-free spectrogram PNG sidecars
- Inline audio playback of detected segments
- Detection labels: humpback (positive), orca (positive), ship (negative), background (negative)
- Detection label annotation with keyboard shortcuts (`h`=humpback, `o`=orca, `s`=ship, `b`=background)
- Detection label API enforces label values in `{0, 1, null}`

### Hydrophone Streaming Detection
- S3 HLS streaming from Orcasound public hydrophone network (anonymous access)
- Metadata-driven NOAA GCS archive registry packaged in
  `src/humpback/data/noaa_archive_sources.json`, including full verified NOAA
  reference records plus runtime config for loadable archive sources
- NOAA passive bioacoustic archives via anonymous GCS fetch now support the
  visible detection-UI sources `sanctsound_ci01` (Channel Islands, 4 sub-sites
  ci01–ci04), `sanctsound_oc01` (Olympic Coast, 4 sub-sites oc01–oc04), and
  legacy `noaa_glacier_bay` (Glacier Bay / Bartlett Cove), with metadata-driven
  root prefixes, mixed filename parsing, and child-folder hints for partitioned
  archive layouts; `CachingNoaaGCSProvider` caches metadata manifests + audio
  segments locally under `noaa_cache_path` with GCS fallback on cache miss
- Write-through S3 cache (`CachingS3Client`): fetches from S3 on first access, caches segments locally with atomic writes, 404 markers for missing segments
- Local HLS cache support: read pre-downloaded .ts segments from filesystem (same S3-mirrored directory structure)
- Client priority: local_cache_path > s3_cache_path > direct S3
- ArchiveProvider abstraction now spans detection, playback, extraction, and worker/router
  orchestration; upstream hydrophone consumers pass providers instead of raw clients plus
  `hydrophone_id`
- 7 UI-visible archive sources on the legacy hydrophone API: 4 Orcasound
  hydrophones plus NOAA SanctSound Channel Islands (ci01–ci04), NOAA SanctSound
  Olympic Coast (oc01–oc04), and NOAA Glacier Bay (Bartlett Cove)
- Segment fetch retry: transient S3 errors (IncompleteRead, ReadTimeoutError, ConnectionError) retried up to 3× with exponential backoff (1s/2s/4s); explicit `connect_timeout=10`, `read_timeout=30`
- Ordered concurrent segment prefetch for hydrophone detection on providers that
  support it, with NOAA source-level behavior controlled by metadata
  (for example SanctSound disabled, Glacier Bay enabled), while preserving
  timeline order and existing retry/alert behavior
- In-memory processing: segments decoded via ffmpeg stdin/stdout, no disk I/O
- NOAA long-object archives now stream ffmpeg decode in chunk-sized slices and
  opt out of raw-byte prefetch so multi-hour SanctSound files do not require a
  full in-memory decode before chunk progress begins
- Streaming detection pipeline with per-chunk progress updates
- Cancel support via threading.Event + DB polling
- TF2 SavedModel hydrophone detection now runs in a short-lived subprocess so
  TensorFlow/Metal memory is reclaimed between jobs while the parent worker
  retains progress/alert/pause/resume/cancel orchestration
- Flash alerts for segment decode failures (dismissable, color-coded)
- Hydrophone run summaries now include timing breakdown fields (`fetch_sec`, `decode_sec`, `features_sec`, `inference_sec`, `pipeline_total_sec`) plus prefetch flags/limits and runtime metadata (`provider_mode`, `execution_mode`, `avg_audio_x_realtime`, `peak_worker_rss_mb`, `child_pid`)
  where `avg_audio_x_realtime` is computed from end-to-end measured time
  (`fetch_sec + decode_sec + pipeline_total_sec`)
- Orcasound audio playback reads from local cache via LocalHLSClient (no S3 calls during playback)
- NOAA playback/extraction use `CachingNoaaGCSProvider` when `noaa_cache_path` is set,
  falling back to direct GCS fetch when it is unset
- Hydrophone timeline assembly uses numeric segment ordering plus playlist durations (when available),
  with fallback to numeric/default-duration metadata when playlists are unavailable
- Sparse local-cache segment sets preserve playlist timeline offsets (for example, cached
  mid-sequence `live6118..` ranges) so playback/spectrogram resolution stays aligned
  without S3 fallback
- Hydrophone folder selection starts at requested range and expands backward
  using configurable hour increments (default 4h) up to configurable max
  lookback (default 168h) until overlap at requested start boundary is found
  (or max lookback is reached), then clips timeline to requested bounds
- Hydrophone detection/playback/extraction are bounded to job `[start_timestamp, end_timestamp]`
- Hydrophone detection jobs fail explicitly when no overlapping stream audio exists
  in the requested time range (no silent complete-with-zero-windows)
- Hydrophone playback timestamp mapping uses stream-offset resolution against the bounded timeline
  with legacy fallback to `job.start_timestamp`
- Hydrophone TSV rows include canonical snapped `.flac` `detection_filename` plus legacy `extract_filename` alias
- Hydrophone detection persists per-window diagnostics incrementally as Parquet shards so
  paused/canceled jobs can extract positives without re-running inference
- Legacy TSV normalization for content/download prefers `extract_filename` when `detection_filename` is missing, otherwise derives snapped canonical ranges
- Detection TSV rows now persist positive-selection provenance (`positive_selection_*` +
  `positive_extract_filename`) so selected training windows survive label-save/content/download round-trips
- Detection TSV download streams normalized rows incrementally (avoids full-file in-memory buffering)
- FLAC export for hydrophone jobs: Orcasound reads from local HLS cache (no S3 calls during extraction), while NOAA uses direct anonymous GCS fetch; both write labeled samples to positive/negative folders
- Hydrophone extraction output paths: species/category before hydrophone —
  `{positive|negative}_root/{label}/{hydrophone_id}/YYYY/MM/DD/*.flac`
- Hydrophone labeled-sample extraction reuses the same stream-offset resolver as playback
- Hydrophone detection job resume after worker restart: skips already-processed segments,
  preserves prior detections, guards against timeline changes between runs
- Cache invalidation on decode failure: corrupted cached segments are deleted and re-fetched
  from S3 (single retry) instead of failing permanently
- Max 7-day time range per job
- Hydrophone job validation enforces `hop_seconds <= classifier window_size_seconds`
- `local_cache_path` is only valid for Orcasound HLS sources; NOAA sources use
  `noaa_cache_path` for local caching instead

### Web UI
- Side nav + top nav SPA with breadcrumbs (Audio, Processing, Clustering, Classifier [Training/Hydrophone Detection], Search, Label Processing, Admin)
- Model filter dropdown on Processing, Clustering, and Classifier pages
- Model version badges on processing jobs, embedding sets, and folder tree rows
- Cross-model warning banner on classifier training (prevents submission)
- Processing job delete (per-row and folder-level bulk delete)
- Model registry management and scanner
- Folder browser for audio selection
- UMAP visualization, cluster tables, evaluation panels
- Bulk delete for training/detection jobs
- Expandable detection rows with sortable TSV data
- Hydrophone Extract button enablement is based on saved labels of the expanded completed, canceled, or paused job
- Hydrophone detection table uses canonical snapped UTC Detection Range and Duration from
  `detection_filename` (no secondary raw-range row)
- Hydrophone playback and extraction use the same canonical bounds shown in Detection Range
- Hydrophone row tooltip exposes unsnapped raw audit range when it differs from canonical bounds
- Hydrophone job date range uses popover picker with dual-month calendar, month/year jump controls, and HH:MM time inputs (UTC-only)
- Hydrophone Active Jobs table shows all running/queued/paused jobs with per-row
  Pause/Resume/Cancel controls; queued jobs can be canceled before a worker claims them
- Paused hydrophone jobs support label save, TSV download, and extraction (stable TSV while worker is blocked)
- Paused hydrophone jobs with partial TSV output keep detection content available via
  `/classifier/detection-jobs/{id}/content`
- Hydrophone progress displays audio duration in hours:minutes format
- Hydrophone TSV report includes `hydrophone_name` column (short form, e.g., `rpi_north_sjc`)
- Detection row state now persists in `detection_rows.parquet` beside each job's TSV; the row store is the canonical editable/download source and carries `row_id`, detection-time `auto_positive_selection_*`, manual override bounds, effective `positive_selection_*`, and extraction artifacts, while TSV is synchronized for user download and legacy compatibility
- Detection spectrogram popup: click a detection row's Play button to open the cached STFT spectrogram alongside playback, or Alt+click the row for spectrogram-only view; positive-labeled detections render client-side black extraction markers immediately from live checkbox edits using stored auto/effective bounds before extraction runs, and completed hydrophone jobs can shift start/end markers in 5-second steps with Apply/Cancel to persist manual window overrides back into the detection job (configurable via `HUMPBACK_SPECTROGRAM_*` env vars)
- Extract Labeled Samples now writes a sibling `.png` beside each extracted `.flac`,
  using the same shared spectrogram render settings as the UI popup base image and
  the actual extracted clip window rather than the full detection span
- "Whale" badge on hydrophone jobs with confirmed positive labels — humpback or orca (`has_positive_labels` flag persisted on label save)
- 3-way audio source selector (Orcasound / NOAA / Local Cache) with hydrophone
  dropdown filtered by `provider_kind` from the `/classifier/hydrophones` API;
  the NOAA list currently exposes SanctSound Channel Islands (ci01–ci04),
  SanctSound Olympic Coast (oc01–oc04), and NOAA Glacier Bay
- Previous Jobs table: text filter (hydrophone name), sortable columns (status,
  hydrophone, date, threshold, results), client-side pagination, and preferences
  dialog (page size: 10/20/50/100, column visibility toggles)
- `DatabaseErrorBanner`: persistent top-of-page red flash bar that polls `/health`
  every 15 s and surfaces DB misconfiguration errors (dismissable; auto-redisplays
  while the API remains unhealthy)

### Retrain Workflow
- Automated retrain pipeline: reimport folders, queue processing, create training job — all from a single "Retrain" button
- Backend-orchestrated state machine: `queued` → `importing` → `processing` → `training` → `complete`
- Folder tracing: resolves import root paths from training job's embedding set provenance
- Embedding set collection: gathers ALL embedding sets from folder hierarchies (includes newly added audio)
- Worker integration: polled alongside other job types in the main worker loop
- Stale workflow recovery: importing/processing/training workflows reset to queued after timeout
- Frontend: retrain sub-panel in expanded model rows with step indicator and progress tracking
- API endpoints: retrain-info (pre-flight), create retrain, list/get workflows

### Audio/Label Processing (Phases 1–4)
- Raven selection table parser: parses TSV annotation files with call type normalization
  (handles BOM, multi-word labels, typo corrections: chrip→Chirp, whuo→Whup, piccalo→Piccolo)
- Annotation-recording pairing by filename stem (strips `.Table.N.selections` suffix)
- Score-based segmentation engine: runs classifier scoring pipeline (5s window, 1s hop)
  over full recordings, smooths scores, detects peaks, estimates onset/offset
- Annotation-to-peak mapping with overlap classification (clean, mild_overlap, heavy_overlap)
- Clean 5s window extraction centered on score peaks with FLAC + PNG spectrogram sidecars
- Fallback extraction: annotations with no peak above threshold extract at annotation
  midpoint (treatment `"fallback"`) instead of being skipped
- Re-center extraction (Option B): tries multiple crop windows around dominant peak to
  minimise overlap with neighbouring peaks; escalates to synthesis when no crop achieves
  acceptable isolation (mild_overlap annotations)
- Background region extraction: finds contiguous low-score regions (smoothed score < threshold
  for >= min_duration) and extracts non-overlapping 5s segments for synthesis use
- Synthesis extraction (Option C): isolates cleanest 1–3s call segment using onset/offset
  bounds, places into background with raised-cosine crossfade splicing, generates up to 3
  placement variants (early/centre/late) per heavy_overlap annotation
- `LabelProcessingJob` DB model with worker integration (queued → running → complete/failed)
- API endpoints: create/list/get/delete jobs, preview annotation pairing
- Output organized by treatment and call type:
  `{output_root}/{clean,fallback,recentered,synthesized}/{CallType}/*.flac`
- Configurable parameters: `enable_recentered`, `enable_synthesized`, `background_threshold`,
  `synthesis_crossfade_ms`, `synthesis_variants`, `cleanup_score_cache`
- Score cache cleanup: `{output_root}/scores/` directory removed after job completion
  by default (`cleanup_score_cache: true`); set to `false` to retain for debugging
- Web UI page (`/app/label-processing`): job creation form with classifier model selector,
  annotation/audio folder inputs, advanced parameter accordion, preview mode showing annotation
  pairing + call type distribution, active job progress tracking, and completed job result
  inspection with treatment/call-type breakdown
- E2E smoke test covers full pipeline: train classifier → create job → run worker → verify
  output FLAC durations, PNG sidecars, treatment distribution, and score cache cleanup

### Data Staging Utilities
- `scripts/convert_audio_to_flac.py` converts `.wav` and `.mp3` files to sibling `.flac`
  files and can optionally verify decoded samples after conversion.
- `scripts/stage_s3_epoch_cache.py` stages epoch-style public S3 prefixes with
  object `LastModified` overlap refinement against requested UTC ranges.
- Prefix discovery uses `s3api list-objects-v2` with `start-after` and
  end-boundary early stop to reduce startup latency for narrow windows.
- CLI uses `--dry-run` for manifest-only planning; downloads execute by default
  when `--dry-run` is omitted.
- Optional pre-count/pre-size pass (`--pre-count`, default enabled) estimates
  files/bytes totals and per-prefix breakdown.
- Download progress is script-owned (`tqdm`) using structured `s5cmd --json`
  copy events with fixed totals; planned prefix list is printed up front and
  progress postfix includes `current=<prefix>`.

### Environment & Packaging
- Platform-specific TensorFlow extras: `tf-macos`, `tf-linux-cpu`, and `tf-linux-gpu`
- Supported Python runtime versions: 3.11 and 3.12
- `google-cloud-storage` is a runtime dependency because NOAA GCS access is part of the
  production provider set
- Python code quality tooling via `uv` + pre-commit: Ruff for lint/format and
  Pyright for type checking (enforced for `src/humpback`, `scripts/`, and
  `tests/`)
- Direct runtime dependency on `soundfile` retained for extraction and FLAC conversion paths
- Repo-root `.env` support for runtime and deploy-time configuration (API/worker
  entrypoints load it explicitly; `scripts/deploy.sh` sources it for `TF_EXTRA`)
- FastAPI bind host/port are configurable via `HUMPBACK_API_HOST` / `HUMPBACK_API_PORT`
- `/health` GET endpoint: returns `{"status":"ok"}` after successful startup,
  `503 {"status":"error","detail":"..."}` when DB init failed, or
  `{"status":"starting"}` before startup completes; registered before routers so
  it does not depend on a DB session
- FastAPI trusted-host validation is configurable via comma-separated
  `HUMPBACK_ALLOWED_HOSTS` patterns (Starlette wildcard syntax, e.g.
  `*.trycloudflare.com`)
- Default extraction/cache paths derive from `storage_root` unless explicitly
  overridden by `HUMPBACK_POSITIVE_SAMPLE_PATH`,
  `HUMPBACK_NEGATIVE_SAMPLE_PATH`, `HUMPBACK_S3_CACHE_PATH`, or
  `HUMPBACK_NOAA_CACHE_PATH`

---

## Database Schema

- **Engine**: SQLite via SQLAlchemy
- **Latest migration**: `020_label_processing_jobs.py`
- **Tables**: model_configs, audio_files, audio_metadata, processing_jobs, embedding_sets, clustering_jobs, clusters, cluster_assignments, classifier_models, classifier_training_jobs, detection_jobs, retrain_workflows, label_processing_jobs

---

## Sensitive Components

Changes to these areas require extra care and testing:

| Component | Risk | Why |
|-----------|------|-----|
| `processing/windowing.py` | Signal integrity | Affects all downstream embeddings |
| `processing/features.py` | Signal integrity | Spectrogram shape must be 128x128 |
| `processing/parquet_writer.py` | Data integrity | Atomic write semantics |
| `database.py` | Schema | Must match Alembic migrations |
| `encoding_signature` computation | Idempotency | Duplicate prevention depends on this |
| `clustering/engine.py` | Correctness | Metrics and assignments must be consistent |
| `classifier/trainer.py` | Model quality | Class weight balance, CV splits |

---

## Known Constraints

- SQLite: no true row-level locking (worker claim uses UPDATE with status check)
- No real-time streaming — polling-based UI updates
- Single-machine deployment (MVP)
- Exactly one TensorFlow extra must be selected per environment; `uv sync --all-extras` is invalid
- Linux GPU installs assume a modern glibc baseline compatible with TensorFlow CUDA wheels
- Model files must be present on disk (no remote model registry)
- Pyright enforcement covers `src/humpback`, `scripts/`, and `tests/`;
  expand deliberately after cleaning any new areas
- `HUMPBACK_ALLOWED_HOSTS` uses Starlette wildcard syntax (`*.example.com`, not `.example.com`)
- Audio shorter than `window_size_seconds` (5s) is skipped entirely
- Imported audio must remain at original path (in-place reads)
- `/audio/{id}/download` returns HTTP 416 for malformed/invalid Range headers
