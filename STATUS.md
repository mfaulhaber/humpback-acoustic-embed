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
- Configurable model registry (TFLite + TF2 SavedModel)
- Audio decoding, resampling (32 kHz), overlap-back windowing (5s default)
- Log-mel spectrogram extraction (128x128) for TFLite models
- Raw waveform input for TF2 SavedModel models
- Incremental Parquet embedding output with atomic writes
- Idempotent encoding via `encoding_signature`
- Background worker with job queue (queued/running/complete/failed/canceled)

### Clustering
- HDBSCAN, K-Means, Agglomerative clustering algorithms
- UMAP, PCA, or no dimensionality reduction
- Euclidean and cosine distance metrics
- Internal metrics: Silhouette, Davies-Bouldin, Calinski-Harabasz
- Supervised metrics: ARI, NMI, homogeneity, completeness, v_measure, confusion matrix
- Parameter sweep (HDBSCAN + K-Means ranges)
- Fragmentation report (entropy, Gini, noise rates)
- Opt-in: classifier baseline, stability evaluation, metric learning refinement
- Re-clustering from refined embeddings via `refined_from_job_id`

### Binary Classifier
- Train LogisticRegression from positive + negative embedding sets
- Balanced class weights by default
- Cross-validation metrics (accuracy, ROC-AUC)
- Detection job: scan audio folder, produce TSV with confidence scores
- Span merging for contiguous detections
- Inline audio playback of detected segments
- Detection label annotation with keyboard shortcuts

### Web UI
- Tab-based SPA (Audio, Processing, Clustering, Classifier, Admin)
- Model registry management and scanner
- Folder browser for audio selection
- UMAP visualization, cluster tables, evaluation panels
- Bulk delete for training/detection jobs
- Expandable detection rows with sortable TSV data

---

## Database Schema

- **Engine**: SQLite via SQLAlchemy
- **Latest migration**: `009_add_source_folder.py`
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
