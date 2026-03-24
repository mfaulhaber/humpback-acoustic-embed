# Project Status

Current state of the humpback acoustic embedding and clustering platform.

---

## Implemented Capabilities

### Audio Management

- Upload individual MP3, WAV, and FLAC files into `audio/raw/`, or import audio folders in place via `source_folder` without copying files.
- Attach and edit audio metadata, and inspect per-file processing history from the audio detail view.

### Processing Pipeline

- Model registry supports TFLite and TF2 SavedModel encoders, auto-detects input format and vector dimensionality, and runs shared 32 kHz decoding plus 5-second overlap-back windowing.
- Feature extraction supports both 128x128 log-mel spectrograms and raw waveform inputs, with batched FFT extraction, batched TFLite inference, and multi-threaded interpreter defaults.
- Embeddings are written incrementally to Parquet with atomic promotion; idempotent reuse is keyed by `encoding_signature`.
- Worker execution uses queued/running/complete/failed/canceled states, compare-and-set job claims, periodic stale-job recovery, timing instrumentation, restart cleanup for stale local detection TSV outputs, and sample-precise short-audio warnings when no embeddings are produced.

### Embedding Similarity Search

- Similarity search supports brute-force cosine or euclidean lookup across same-model embedding sets with `top_k`, `exclude_self`, and embedding-set filters.
- Queries can come from stored windows, raw vectors, or async worker-encoded audio; completed audio queries return the query vector and model version for immediate re-search.
- Detection peak-window embeddings are stored in `detection_embeddings.parquet`, and search results include audio metadata, window offsets, spectrogram thumbnails, and inline playback.
- The Search UI supports both standalone search and detection-sourced "Search Similar" flows, including multi-select embedding-set filtering.

### Clustering

- Clustering supports HDBSCAN, K-Means, and Agglomerative algorithms with UMAP, PCA, or no dimensionality reduction and cosine or euclidean distance metrics.
- Evaluation includes internal metrics, supervised metrics, fragmentation reporting, parameter sweeps, and optional classifier-baseline, stability, and metric-learning refinement flows.
- Validation blocks mixed-model clustering, requires at least one embedding set, and supports reclustering from refined embeddings via `refined_from_job_id`.

### Binary Classifier

- Train LogisticRegression (default) or MLP classifiers from positive and negative embedding sets with balanced class weights, optional L2 normalization, cross-validation metrics, decision diagnostics, overlap validation, and encoding-signature consistency warnings.
- Local detection scans audio folders with configurable hop size and hysteresis thresholds, stores canonical snapped detection bounds plus raw audit metadata, and streams incremental progress back to the UI.
- New detection jobs are windowed-only: fresh jobs persist `detection_mode="windowed"` and emit fixed 5-second detections via NMS peak selection; legacy merged jobs remain readable but are read-only for labels, row-state edits, and extraction.
- Detection rows support inline playback, live labeling with keyboard shortcuts, persisted row-state and positive-selection metadata, and labeled-sample extraction to FLAC with sibling marker-free PNG spectrogram sidecars.
- Label validation remains constrained to humpback and orca positives plus ship and background negatives.

### Hydrophone Streaming Detection

- Hydrophone detection supports Orcasound HLS and metadata-driven NOAA archives, including SanctSound Channel Islands, SanctSound Olympic Coast, and legacy Glacier Bay sources.
- Cache and provider behavior is shared across detection, playback, and extraction: Orcasound prefers `local_cache_path`, then `s3_cache_path`, then direct S3; NOAA uses `noaa_cache_path` when configured and falls back to direct anonymous GCS access when needed.
- Hydrophone timelines are clipped to requested UTC ranges, use numeric segment ordering plus playlist durations when available, preserve sparse-cache offsets, and expand backward by configurable lookback windows until overlap is found.
- Worker execution supports ordered prefetch, retry handling, per-chunk progress, pause/resume/cancel, restart-safe resume, explicit no-audio failures, and TF2 subprocess isolation with runtime and timing diagnostics.
- Playback and labeled-sample extraction share the same stream-offset resolver and canonical snapped detection ranges; extraction over-fetches a small real-audio guard band, hard-trims hydrophone clips to the expected sample count when the archive contains enough audio, never zero-pads short archive clips, writes FLAC plus PNG sidecars into species/category-first output trees, and keeps partial TSV content usable while jobs are paused.
- Hydrophone UI flows include active and previous job management, UTC-only range selection, persisted detection row state, whale badges and positive-selection metadata, and guardrails such as a 7-day maximum range and `hop_seconds <= window_size_seconds`.

### Web UI

- The frontend is a routed SPA with top navigation, side navigation, breadcrumbs, and dedicated views for Audio, Processing, Clustering, Classifier, Search, Label Processing, and Admin.
- Shared UI patterns include model/version filters, folder browsing, model-registry management, processing and training bulk-delete actions, and clustering visualizations plus evaluation panels.
- Detection views support expandable rows, sortable content, cached spectrogram popups, canonical UTC range display, persisted row-state editing, and completed-job spectrogram window overrides in 5-second steps where supported.
- Hydrophone tables cover active-job pause/resume/cancel controls, paused-job content access, previous-job filtering/pagination/preferences, and source selection across Orcasound, NOAA, and local cache modes.
- `DatabaseErrorBanner` polls `/health` and surfaces backend database startup failures while the API remains unhealthy.

### Retrain Workflow

- Retrain runs reimport folders, queue processing, and create a new training job from a single frontend action.
- The backend orchestrates retrain state as `queued -> importing -> processing -> training -> complete`, with worker integration and stale-workflow recovery.
- Retrain provenance traces import roots from prior embedding sets, gathers all descendant embedding sets, and surfaces step-by-step progress in the UI.

### Audio/Label Processing (Phases 1–4)

- The score-based workflow parses Raven selection tables, normalizes call types, pairs annotations to recordings, runs classifier scoring over full recordings, and maps annotations to peaks with overlap classification.
- Outputs include clean 5-second extracts, fallback midpoint extracts when no peak exceeds threshold, and synthesized variants with annotation-guided call isolation, adaptive background thresholds, short-run tiling, and raised-cosine splicing.
- `LabelProcessingJob` is a worker-backed DB workflow with preview/create/list/get/delete support, treatment- and call-type-organized outputs, per-label score KPIs, and optional score-cache cleanup.
- `sample_builder` provides a classifier-free alternative with a 10-stage signal-processing pipeline, marine-noise-tuned contamination screening, accepted/rejected artifact reporting, and shared DSP utilities such as `raised_cosine_fade`.
- `workflow` can be `score_based` or `sample_builder`, with nullable `classifier_model_id` for classifier-free runs; the UI exposes workflow-aware forms, acceptance/rejection stats, and completed-job result inspection.
- End-to-end smoke coverage verifies training, job execution, FLAC duration correctness, PNG sidecars, treatment distribution, and score-cache cleanup.

### Vocalization Type Labeling

- The `/app/classifier/labeling` page provides a focused single-detection labeling workspace for applying extensible vocalization type labels (e.g., whup, moan, shriek) to detection rows.
- Detection neighbors endpoint performs synchronous vector search against reference embedding sets and infers vocalization types from audio folder paths.
- Vocalization labels are stored in a dedicated `vocalization_labels` SQL table with CRUD API at `/labeling/`, including vocabulary auto-complete and per-job labeling progress summary.
- Keyboard-driven navigation (j/k, space, u, 1-9, x) enables rapid labeling throughput.
- Multi-class vocalization classifiers can be trained from labeled detection embeddings via `/labeling/training-jobs`, with `classifier_purpose="vocalization"` and `job_purpose="vocalization"` columns discriminating from binary detection classifiers; training job status is pollable via `GET /labeling/training-jobs/{job_id}`.
- The labeling workspace surfaces a "Train Classifier" panel when 2+ distinct vocalization types are labeled, polls training job progress, and auto-selects the resulting model on completion for predictions and active learning.
- Trained vocalization models predict labels for all detection rows via `/labeling/predict/{job_id}`, returning per-class probabilities sorted by uncertainty for active learning.
- Sub-window annotations in `labeling_annotations` table allow marking precise call boundaries within 5-second detection windows via click-drag on the spectrogram, with CRUD at `/labeling/annotations/`.
- Active learning loop via `/labeling/active-learning-cycle` queues retrain cycles, `/labeling/uncertainty-queue/{job_id}` returns detection rows sorted by prediction uncertainty, and `/labeling/convergence/{model_id}` tracks accuracy trend and per-class metrics across training cycles.

### Data Staging Utilities

- `scripts/convert_audio_to_flac.py` converts sibling `.wav` and `.mp3` files to `.flac` and can optionally verify decoded sample fidelity.
- `scripts/repair_hydrophone_extract_lengths.py` dry-runs or repairs imported hydrophone extracts that are `1..64` samples short of the configured 5-second window, can recover legacy `.wav`-named FLAC outputs, and refreshes `audio_files` metadata when applied.
- `scripts/stage_s3_epoch_cache.py` stages epoch-style public S3 ranges with `LastModified` overlap refinement, dry-run planning, optional pre-count/pre-size estimation, and early-stop prefix discovery.
- Downloads use `s5cmd --json` plus `tqdm` for fixed-total progress reporting with per-prefix status.

### Environment & Packaging

- Python package management uses `uv` with mutually exclusive TensorFlow extras: `tf-macos`, `tf-linux-cpu`, and `tf-linux-gpu`; supported Python versions are 3.11 and 3.12.
- Runtime dependencies include `google-cloud-storage` for NOAA archive access and `soundfile` for extraction and FLAC conversion paths.
- Tooling uses Ruff, Pyright, and pre-commit; current Pyright enforcement covers `src/humpback`, `scripts/`, and `tests/`.
- API, worker, and deploy flows load the repo-root `.env`; host, port, trusted hosts, and storage-derived extraction/cache paths remain configurable via `HUMPBACK_` environment variables.
- `/health` reports `ok`, `starting`, or `error` independently of normal router DB dependencies.

---

## Database Schema

- **Engine**: SQLite via SQLAlchemy
- **Latest migration**: `024_labeling_annotations.py`
- **Tables**: model_configs, audio_files, audio_metadata, processing_jobs, embedding_sets, clustering_jobs, clusters, cluster_assignments, classifier_models, classifier_training_jobs, detection_jobs, retrain_workflows, label_processing_jobs, vocalization_labels, labeling_annotations

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

- SQLite has no true row-level locking; worker claims rely on `UPDATE` plus status checks.
- The UI remains polling-based rather than real-time.
- Deployment is still single-machine MVP infrastructure.
- Exactly one TensorFlow extra must be selected per environment; `uv sync --all-extras` is invalid.
- Linux GPU installs assume a modern glibc baseline compatible with TensorFlow CUDA wheels.
- Model files must be present on disk; there is no remote model registry.
- Pyright enforcement currently covers `src/humpback`, `scripts/`, and `tests/`; expand deliberately after cleaning new areas.
- `HUMPBACK_ALLOWED_HOSTS` uses Starlette wildcard syntax such as `*.example.com`, not `.example.com`.
- Audio shorter than `window_size_seconds` (5 seconds) is skipped entirely.
- Imported audio must remain at its original path for in-place reads.
- `/audio/{id}/download` returns HTTP 416 for malformed or invalid `Range` headers.
