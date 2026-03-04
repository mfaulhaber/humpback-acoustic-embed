# Humpback Acoustic Embedding & Clustering System

## Overview

This project processes humpback whale audio recordings (MP3/WAV/FLAC) into reusable
embedding vectors using a Perch-compatible TFLite model, then performs clustering
with optional ecological/behavioral metadata.

Key features:
- Asynchronous job queue (SQL-backed, restart-safe)
- Idempotent encoding (no reprocessing for same config)
- Multi-model registry supporting TFLite and TF2 SavedModel formats
- TF2 SavedModel support for raw waveform input (e.g., SurfPerch)
- macOS GPU acceleration via tensorflow-macos/tensorflow-metal
- Embeddings stored in Parquet
- REST API for job management and inspection
- Flexible clustering pipeline: HDBSCAN, K-Means, or Agglomerative with interactive scatter plot visualization
- Dimensionality reduction: UMAP, PCA, or none; euclidean or cosine distance metric
- Quantitative cluster evaluation (Silhouette, Davies-Bouldin, Calinski-Harabasz)
- Detailed supervised metrics (ARI, NMI, homogeneity, completeness, v-measure, per-category purity, confusion matrix) from folder-path-derived category labels
- Automatic parameter sweep (HDBSCAN min_cluster_size × selection_method + K-Means k) with ARI/NMI when categories available
- Spectrogram normalization options (per-window max, global ref, standardize) via feature_config
- Fragmentation analysis: per-category and per-cluster entropy, Gini coefficient, noise rates
- Stability evaluation: re-cluster with multiple random seeds, pairwise ARI agreement
- Classifier baseline: logistic regression cross-validation with active learning priority queue
- Metric learning refinement: triplet-loss MLP projection head to optimize embedding space, base vs refined comparison

---

## Quick Start

### Install

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/), plus Node.js 18+ for the frontend.

```bash
# Backend
uv sync --all-extras

# Frontend
cd frontend && npm install
```

### Development Mode

Run the backend and frontend dev server in separate terminals:

```bash
# Terminal 1: API server
uv run humpback-api

# Terminal 2: Worker
uv run humpback-worker

# Terminal 3: Frontend dev server (hot reload, proxies API to :8000)
cd frontend && npm run dev
```

Open http://localhost:5173 for the dev UI. API docs at http://localhost:8000/docs.

### Production Build

Build the frontend and serve everything from the API server on a single port:

```bash
cd frontend && npm run build   # outputs to src/humpback/static/dist/
uv run humpback-api            # serves SPA + API on :8000
```

Open http://localhost:8000 — the FastAPI server serves the built SPA at `/` and
the API at their usual paths. When `static/dist/` is not present, it falls back
to the legacy `static/index.html`.

---

## Architecture

### Processing Workflow

```mermaid
flowchart TD
    A["Audio File<br/>(MP3/WAV/FLAC)"] --> B["Decode Audio<br/>→ float32 mono"]
    B --> C["Resample<br/>→ 32 kHz"]
    C --> D["Slice Windows<br/>5 s → 160 000 samples"]
    D --> E{input_format?}
    E -- spectrogram --> F["Log-Mel Spectrogram<br/>128 mels × 128 frames"]
    E -- waveform --> G["Raw Waveform<br/>160 000 samples"]
    F --> H["TFLite Model<br/>→ 1280-d vector"]
    G --> I["TF2 SavedModel<br/>→ N-d vector"]
    H --> J["Parquet Writer<br/>(incremental, atomic)"]
    I --> J
    J --> K["EmbeddingSet<br/>(SQL row)"]
    K --> L["UMAP<br/>→ 10-d (cluster) + 2-d (viz)"]
    L --> M["HDBSCAN<br/>→ cluster labels"]
    M --> N["Metrics + Outputs"]
```

#### Key Signal Processing Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| Sample rate | 32 kHz | Resample target |
| Window size | 5 s (160k samples) | Fixed-length, zero-padded |
| Spectrogram | 128 mels × 128 frames | n_fft=2048, hop=1252 |
| Embedding dim | 1280 | Perch default |
| UMAP cluster dims | 5 | `umap_cluster_n_components` — clustering input; viz always 2D |
| Clustering algorithm | hdbscan | `clustering_algorithm`: `"hdbscan"`, `"kmeans"`, `"agglomerative"` |
| n_clusters | 15 | For kmeans/agglomerative |
| Linkage | ward | For agglomerative: `"ward"`, `"complete"`, `"average"`, `"single"` |
| Reduction method | umap | `reduction_method`: `"umap"`, `"pca"`, `"none"` |
| Distance metric | euclidean | `distance_metric`: `"euclidean"` or `"cosine"` |
| HDBSCAN selection | leaf | `cluster_selection_method` — 'leaf' (fine-grained) or 'eom' (coarser) |
| HDBSCAN min_cluster_size | 5 | Swept 2–50 for param search |
| Normalization | per_window_max | Spectrogram normalization in feature_config: `"per_window_max"`, `"global_ref"`, `"standardize"` |
| `run_classifier` | false | Opt-in classifier baseline on category labels |
| `stability_runs` | 0 | Opt-in stability evaluation (≥ 2 to enable) |
| `enable_metric_learning` | false | Opt-in triplet-loss MLP refinement |
| `ml_output_dim` | 128 | Metric learning projection output dim |
| `ml_n_epochs` | 50 | Metric learning training epochs |
| `ml_mining_strategy` | semi-hard | Triplet mining: `"random"`, `"hard"`, `"semi-hard"` |

Encoding is associated with the audio file and configuration. Reprocessing is
skipped when an EmbeddingSet with the same encoding_signature already exists.

### Model Registry

Multiple models can be registered and managed via the Admin tab or API. Supported
model types:
- **TFLite** (`model_type="tflite"`, `input_format="spectrogram"`): Standard TFLite models
  that take log-mel spectrogram input
- **TF2 SavedModel** (`model_type="tf2_saved_model"`, `input_format="waveform"`): TensorFlow 2
  SavedModel directories that take raw audio waveform input (e.g., SurfPerch)

A default model is seeded on first startup (`multispecies_whale_fp16`). When
creating a processing job, if no `model_version` is specified, the default
registered model is used. The worker resolves the model path, vector
dimensions, model type, and input format from the registry, caching loaded
models across jobs. The scan endpoint detects both `.tflite` files and
directories containing `saved_model.pb`.

Clustering validates that all selected embedding sets share the same vector
dimensions to prevent mixing incompatible embeddings.

### Apple Silicon GPU Acceleration (Metal)

TF2 SavedModel inference runs on the Metal GPU on Apple Silicon Macs. This
required solving a compatibility issue: JAX-exported SavedModels (like
SurfPerch) embed `_XlaMustCompile` attributes in their graph, which force XLA
compilation. Apple's `tensorflow-metal` plugin has no XLA backend, so these
models fail on GPU even though every individual op (Conv2D, MatMul, RFFT, etc.)
is natively supported by Metal.

**How it works:**

1. At model load time, the system inspects the SavedModel's protobuf for
   `_XlaMustCompile` attributes
2. If found, it creates a patched copy of the model (`<model_dir>-no-xla-compile`)
   with the attribute stripped — variable shards are hard-linked to avoid
   doubling disk usage
3. The patched model is loaded on GPU and validated against a CPU baseline using
   deterministic test inputs (tolerance: atol=1e-4, rtol=1e-3)
4. If GPU validation passes, all inference runs on Metal GPU; if it fails, the
   system falls back to CPU automatically and sets `gpu_failed=True` so the UI
   can surface a warning

The patched model copy is created once and reused across worker restarts. To
force CPU-only inference (skipping GPU entirely), set `HUMPBACK_TF_FORCE_CPU=true`.

**Typical startup log (GPU success):**

```
INFO: Model at models/surfperch-tensorflow2 contains _XlaMustCompile (JAX/XLA model). Patching for Metal GPU compatibility.
INFO: Stripping _XlaMustCompile from models/surfperch-tensorflow2 → models/surfperch-tensorflow2-no-xla-compile
INFO: Stripped 3 _XlaMustCompile attribute(s)
INFO: GPU validation passed. Using GPU device: /device:GPU:0
```

### Clustering Workflow

```
selected embedding sets (must share vector_dim)
  → load from Parquet
  → dimensionality reduction (UMAP, PCA, or none)
  → clustering (HDBSCAN, K-Means, or Agglomerative)
  → persist clusters + assignments
  → compute evaluation metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz)
  → compute detailed supervised metrics (ARI, NMI, homogeneity, completeness, v-measure, per-category purity, confusion matrix)
  → run parameter sweep (HDBSCAN + K-Means, with ARI/NMI when categories available)
  → compute fragmentation report (per-category & per-cluster entropy, Gini, noise rates)
  → [opt-in] classifier baseline (logistic regression CV + active learning queue)
  → [opt-in] stability evaluation (N re-runs with different seeds, pairwise ARI)
  → [opt-in] metric learning refinement (triplet-loss MLP → re-cluster → compare)
```

### Clustering Analysis & Evaluation

After clustering completes, the system produces a suite of analysis outputs. Some
run automatically; others are opt-in via job parameters. All results are available
through the API and displayed in the Evaluation Panel in the UI.

#### Internal Metrics (always computed)

| Metric | What it measures | Higher or lower is better |
|--------|-----------------|--------------------------|
| **Silhouette Score** | How similar each point is to its own cluster vs the nearest other cluster (−1 to 1) | Higher |
| **Davies-Bouldin Index** | Average similarity between each cluster and its most similar cluster | Lower |
| **Calinski-Harabasz Score** | Ratio of between-cluster to within-cluster dispersion | Higher |

These metrics are *unsupervised* — they evaluate clustering structure without
needing ground-truth labels.

#### Supervised Metrics (when folder-path category labels exist)

Audio files organized in subfolders (e.g., `Grunt/`, `Upsweep/`, `Buzz/`) are
automatically assigned category labels derived from the deepest folder path component.
When these labels are available, the system computes:

| Metric | What it measures |
|--------|-----------------|
| **ARI** (Adjusted Rand Index) | Agreement between cluster assignments and categories, corrected for chance (−1 to 1, 1 = perfect) |
| **NMI** (Normalized Mutual Info) | Shared information between clusters and categories (0 to 1) |
| **Homogeneity** | Whether each cluster contains only members of a single category |
| **Completeness** | Whether all members of a category are assigned to the same cluster |
| **V-measure** | Harmonic mean of homogeneity and completeness |
| **Per-category purity** | Fraction of each category's samples in its dominant cluster |
| **Confusion matrix** | Full cluster × category count matrix |

**What to do with these:** ARI and NMI tell you how well the embedding space
naturally separates your call types. Low scores suggest the model may not
distinguish these call types well, or that your category labels need refinement.

#### Dendrogram Heatmap

A hierarchical clustering of the confusion matrix, showing which clusters and
categories are most similar. Helps identify cluster merges or splits that might
improve category alignment.

#### Parameter Sweep (always computed)

Automatically sweeps HDBSCAN (`min_cluster_size` 2–50 × `leaf`/`eom`) and K-Means
(`k` 2–30), recording Silhouette score, cluster count, noise fraction, and
ARI/NMI (when labels available) for each configuration.

**What to do with these:** Use the sweep results to find the parameter
configuration that maximizes your metric of interest. If ARI peaks at a different
`min_cluster_size` than Silhouette, it means the "best" clustering depends on
whether you optimize for internal cohesion or category alignment.

#### Fragmentation Report (always computed)

Measures how categories and clusters relate to each other:

- **Per-category fragmentation:** For each category, how many clusters its members
  are spread across. Metrics include normalized entropy (0 = all in one cluster,
  1 = uniformly spread), Gini coefficient, top-k cluster mass, and noise rate.
- **Per-cluster composition:** For each cluster, how mixed its members are across
  categories. Reports the dominant category, its mass fraction, and cluster entropy.
- **Global summary:** Mean entropy, mean N_eff (effective number of clusters per
  category), and overall noise rate.

**What to do with these:** High fragmentation for a category (e.g.,
`normalized_entropy > 0.5`) means that call type is acoustically diverse or the
embedding space doesn't group it tightly. This can guide data collection (get more
examples of fragmented types) or suggest those categories need subcategories.

#### Stability Evaluation (opt-in: `stability_runs ≥ 2`)

Re-runs the full clustering pipeline N times with different random seeds and
measures consistency:

- **Pairwise ARI:** Agreement between every pair of runs (mean, std, min, max).
  High mean ARI (> 0.8) indicates the clustering is robust to random initialization.
- **Aggregate metrics:** Mean/std/min/max of Silhouette, ARI, NMI, noise fraction,
  and cluster count across all runs.
- **Per-run details:** Full metrics for each individual run.

**What to do with these:** If pairwise ARI is low (< 0.5), the clustering is
unstable — results change significantly with different random seeds. Consider using
PCA instead of UMAP (PCA is deterministic), increasing `min_cluster_size`, or
switching to K-Means which is less sensitive to initialization.

#### Classifier Baseline (opt-in: `run_classifier = true`)

Trains a logistic regression classifier via stratified K-fold cross-validation on
the category labels to measure how linearly separable the categories are in the
original embedding space:

- **Overall accuracy, macro/weighted F1:** How well a simple classifier can predict
  categories from embeddings alone.
- **Per-class precision/recall/F1:** Which categories the classifier can and cannot
  distinguish.
- **Confusion matrix:** Which categories get confused with each other.
- **Active learning queue:** Every sample ranked by labeling priority — unlabeled
  samples get the highest priority, followed by high-uncertainty labeled samples
  (measured by classifier entropy and margin), boosted by the category's
  fragmentation score.

**What to do with these:** If classifier accuracy is high (> 0.85), the embedding
space already separates your categories well and clustering issues likely come from
the clustering algorithm or parameters. If accuracy is low, the embedding space
doesn't distinguish your categories — consider metric learning refinement (below),
a different model, or revising your category definitions. The active learning queue
tells you which samples to label next for maximum impact.

#### Metric Learning Refinement (opt-in: `enable_metric_learning = true`)

Trains a 2-layer MLP projection head (`Dense(512, relu) → Dense(128) → L2 normalize`)
using triplet loss on the labeled subset of embeddings, then projects *all*
embeddings through the learned projection and re-clusters:

- **Training summary:** Epochs, learning rate, margin, mining strategy, final loss,
  loss history.
- **Base vs Refined comparison:** Side-by-side table of every metric (Silhouette,
  DB, ARI, NMI, noise fraction, fragmentation index, cluster count) with deltas
  color-coded green (improvement) or red (regression), accounting for metric
  direction (higher-is-better vs lower-is-better).

**What to do with these:** If the refined metrics improve (especially ARI/NMI),
the triplet-loss projection is successfully pulling same-category embeddings closer
and pushing different categories apart. This validates that the category structure
exists in the data but wasn't fully captured by the original model. If metrics
don't improve or worsen, the original embedding space may already be near-optimal
for these categories, or more labeled data is needed.

Triplet mining strategies:
- **random:** Fast, good baseline — randomly samples anchor/positive/negative triplets
- **semi-hard** (default): Selects negatives that are closer than the positive but
  within the margin — focuses training on the most informative examples
- **hard:** Picks the closest negative for each anchor — aggressive but can cause
  training instability with noisy labels

All training runs on CPU (`tf.device('/CPU:0')`) to avoid Apple Silicon GPU
complications with TF/Keras.

### Cluster Assignment Playback Controls

When viewing a clustering job's results, expanding a cluster shows its audio
file assignments organized in a folder tree. Each audio file has a compact
playback control bar:

```
┌──────────────────────────────────────────────────────────────────────┐
│ [▶]  [0] [3] [17]  [Full]   🔈▮▮▯▯  ⋯                              │
│ 0:15 [====██████=========================] 0:20                      │
└──────────────────────────────────────────────────────────────────────┘
```

| Control | Description |
|---------|-------------|
| **Play/pause** (▶/⏸) | Plays the currently selected window or full file |
| **Row-index chips** ([0], [3], ...) | Each chip represents a window assigned to this cluster. Click to select and auto-play that window segment |
| **Full** | Switch to full-file playback mode |
| **Volume** (🔈) | Adjusts playback volume |
| **Download** (⋯) | Downloads the original audio file |
| **Time indicator** | Bottom bar shows the selected window's position within the full audio file. The blue region highlights the active segment, and a playhead tracks progress during playback |

When a file has only one window assigned to the cluster, that window is
selected by default (instead of full-file mode) so playback starts on the
relevant segment. The time indicator makes it easy to see where in the
original recording the clustered segment falls.

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/audio/upload` | Upload audio file (SHA-256 dedup) |
| GET | `/audio/` | List audio files |
| GET | `/audio/{id}` | Get audio file details |
| PUT | `/audio/{id}/metadata` | Update metadata |
| GET | `/audio/{id}/download` | Download original audio file (supports range requests) |
| GET | `/audio/{id}/window` | Get a WAV segment (`?start_seconds=&duration_seconds=`) |
| GET | `/audio/{id}/spectrogram` | Get log-mel spectrogram for a window (`?window_index=`) |
| GET | `/audio/{id}/embeddings` | Get embedding vectors (`?embedding_set_id=`) |
| POST | `/processing/jobs` | Create processing job |
| GET | `/processing/jobs` | List processing jobs |
| GET | `/processing/jobs/{id}` | Get job status |
| POST | `/processing/jobs/{id}/cancel` | Cancel job |
| GET | `/processing/embedding-sets` | List embedding sets |
| GET | `/processing/embedding-sets/{id}` | Get embedding set |
| POST | `/clustering/jobs` | Create clustering job |
| GET | `/clustering/jobs/{id}` | Get clustering job |
| DELETE | `/clustering/jobs/{id}` | Delete clustering job |
| GET | `/clustering/jobs/{id}/clusters` | List clusters |
| GET | `/clustering/jobs/{id}/visualization` | Get UMAP scatter plot data |
| GET | `/clustering/jobs/{id}/metrics` | Get cluster evaluation metrics |
| GET | `/clustering/jobs/{id}/parameter-sweep` | Get parameter sweep results |
| GET | `/clustering/jobs/{id}/dendrogram` | Get cluster × category dendrogram heatmap data |
| GET | `/clustering/jobs/{id}/fragmentation` | Get fragmentation report |
| GET | `/clustering/jobs/{id}/stability` | Get stability evaluation results |
| GET | `/clustering/jobs/{id}/classifier` | Get classifier baseline report |
| GET | `/clustering/jobs/{id}/label-queue` | Get active learning priority queue |
| GET | `/clustering/jobs/{id}/refinement` | Get metric learning refinement report |
| GET | `/clustering/clusters/{id}/assignments` | Get assignments |
| GET | `/admin/models` | List registered models |
| POST | `/admin/models` | Register a new model |
| PUT | `/admin/models/{id}` | Update model config |
| DELETE | `/admin/models/{id}` | Delete model (fails if embeddings reference it) |
| POST | `/admin/models/{id}/set-default` | Set model as default |
| GET | `/admin/models/scan` | Scan `models/` dir for unregistered models (.tflite + SavedModel dirs) |

---

## Storage Layout

```
data/
  audio/raw/{audio_file_id}/original.(wav|mp3|flac)
  embeddings/{model_version}/{audio_file_id}/{encoding_signature}.parquet
  clusters/{clustering_job_id}/clusters.json
  clusters/{clustering_job_id}/assignments.parquet
  clusters/{clustering_job_id}/umap_coords.parquet
  clusters/{clustering_job_id}/parameter_sweep.json
  clusters/{clustering_job_id}/report.json                (fragmentation)
  clusters/{clustering_job_id}/classifier_report.json     (opt-in)
  clusters/{clustering_job_id}/label_queue.json           (opt-in)
  clusters/{clustering_job_id}/stability_summary.json     (opt-in)
  clusters/{clustering_job_id}/refinement_report.json     (opt-in)
```

---

## Testing

### Run all tests

```bash
uv run pytest
```

### Run with verbose output

```bash
uv run pytest -v
```

### Run specific test categories

```bash
uv run pytest tests/unit/           # Unit tests only
uv run pytest tests/integration/    # Integration tests only
uv run pytest tests/e2e/            # E2E smoke test
uv run pytest -k <pattern>          # Pattern matching
```

### Watch mode

```bash
uv run ptw
```

---

## Configuration

Environment variables (prefix `HUMPBACK_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `HUMPBACK_DATABASE_URL` | `sqlite+aiosqlite:///data/humpback.db` | Database URL |
| `HUMPBACK_STORAGE_ROOT` | `data` | Root directory for file storage |
| `HUMPBACK_MODEL_VERSION` | `perch_v1` | Model version identifier |
| `HUMPBACK_WINDOW_SIZE_SECONDS` | `5.0` | Audio window size |
| `HUMPBACK_TARGET_SAMPLE_RATE` | `32000` | Target sample rate |
| `HUMPBACK_VECTOR_DIM` | `1280` | Embedding vector dimensions |
| `HUMPBACK_USE_REAL_MODEL` | `true` | Use real TFLite model vs fake |
| `HUMPBACK_MODEL_PATH` | `models/multispecies_whale_fp16_flex.tflite` | Path to TFLite model file (fallback) |
| `HUMPBACK_MODELS_DIR` | `models` | Directory to scan for `.tflite` model files |
| `HUMPBACK_TF_FORCE_CPU` | `false` | Force CPU for TF2 SavedModel inference (skip GPU) |

---

## Tech Stack

### Backend
- **API**: Python + FastAPI
- **Queue**: SQL-backed polling queue
- **DB**: SQLite with WAL mode (MVP)
- **Embedding**: TensorFlow Lite + TF2 SavedModel (flex delegate for custom ops; macOS GPU via tensorflow-metal; FakeTFLiteModel/FakeTF2Model for testing)
- **Clustering**: UMAP/PCA + HDBSCAN/K-Means/Agglomerative
- **Storage**: Local filesystem

### Frontend
- **Build**: Vite + TypeScript
- **UI**: React 18 + Tailwind CSS
- **Components**: shadcn/ui (Radix primitives)
- **Server State**: TanStack Query (polling, caching, mutations)
- **Charts**: react-plotly.js (spectrograms, similarity matrices, UMAP scatter, parameter sweep)
- **Icons**: lucide-react
- **API Client**: Hand-rolled typed fetch wrapper (`frontend/src/api/client.ts`)

The frontend lives in `frontend/` at the repo root. In development, Vite runs on
`:5173` and proxies API requests to `:8000`. For production, `npm run build`
outputs to `src/humpback/static/dist/` and the FastAPI server serves the SPA
alongside the API on `:8000`.

No client-side router is used — the UI is tab-based (Audio, Processing,
Clustering, Admin) with navigation managed via React state.
