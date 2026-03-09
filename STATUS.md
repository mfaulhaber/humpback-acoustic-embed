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
- Incremental detection results: file-by-file progress with live UI updates during job execution
- Inline audio playback and label annotation available while detection is still running
- Extraction boundary snapping to `window_size` multiples
- Inline audio playback of detected segments
- Detection label annotation with keyboard shortcuts
- Detection label API enforces label values in `{0, 1, null}`

### Hydrophone Streaming Detection
- S3 HLS streaming from Orcasound public hydrophone network (anonymous access)
- Write-through S3 cache (`CachingS3Client`): fetches from S3 on first access, caches segments locally with atomic writes, 404 markers for missing segments
- Local HLS cache support: read pre-downloaded .ts segments from filesystem (same S3-mirrored directory structure)
- Client priority: local_cache_path > s3_cache_path > direct S3
- 4 configured hydrophones: Orcasound Lab, North San Juan Channel, Port Townsend, Bush Point
- In-memory processing: segments decoded via ffmpeg stdin/stdout, no disk I/O
- Streaming detection pipeline with per-chunk progress updates
- Cancel support via threading.Event + DB polling
- Flash alerts for segment decode failures (dismissable, color-coded)
- Audio playback reads from local cache via LocalHLSClient (no S3 calls during playback)
- Hydrophone playback timestamp mapping uses stream-offset resolution with dual anchors:
  first available folder timestamp, then legacy `job.start_timestamp`
- UTC range display in Detection Range column (replaces raw synthetic filenames)
- WAV export for hydrophone jobs: fetches audio from HLS, writes labeled samples to positive/negative folders
- Hydrophone extraction output paths include hydrophone short label partitioning:
  `{positive|negative}_root/{hydrophone_id}/{label}/YYYY/MM/DD/*.wav`
- Hydrophone labeled-sample extraction reuses the same stream-offset resolver as playback
- Max 7-day time range per job
- Hydrophone job validation enforces `hop_seconds <= classifier window_size_seconds`

### Web UI
- Tab-based SPA (Audio, Processing, Clustering, Classifier [Train/Detect/Hydrophone], Admin)
- Model filter dropdown on Processing, Clustering, and Classifier pages
- Model version badges on processing jobs, embedding sets, and folder tree rows
- Cross-model warning banner on classifier training (prevents submission)
- Processing job delete (per-row and folder-level bulk delete)
- Model registry management and scanner
- Folder browser for audio selection
- UMAP visualization, cluster tables, evaluation panels
- Bulk delete for training/detection jobs
- Expandable detection rows with sortable TSV data
- Hydrophone Extract button enablement is based on saved labels of the expanded completed job

---

## Database Schema

- **Engine**: SQLite via SQLAlchemy
- **Latest migration**: `013_hydrophone_local_cache.py`
- **Tables**: model_configs, audio_files, audio_metadata, processing_jobs, embedding_sets, clustering_jobs, clusters, cluster_assignments, classifier_models, classifier_training_jobs, detection_jobs

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
- Model files must be present on disk (no remote model registry)
- Audio shorter than `window_size_seconds` (5s) is skipped entirely
- Imported audio must remain at original path (in-place reads)
- `/audio/{id}/download` returns HTTP 416 for malformed/invalid Range headers
