# Humpback Acoustic Embedding & Clustering System

## Overview

This project processes humpback whale audio recordings (MP3/WAV) into reusable
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
- UMAP + HDBSCAN clustering pipeline

---

## Quick Start

### Install

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-extras
```

### Run the API Server

```bash
uv run humpback-api
# or
uv run python -m humpback
```

API docs available at http://localhost:8000/docs

### Run the Worker

```bash
uv run humpback-worker
```

The worker polls for queued processing and clustering jobs and executes them.

---

## Architecture

### Processing Workflow

```
audio file (MP3/WAV) + optional metadata
  → resample to target SR (default 32kHz)
  → slice into N-second windows (default 5s)
  → branch on model input_format:
      spectrogram: log-mel spectrogram (128 mel bins × 128 time frames) → TFLite inference
      waveform:    raw audio windows → TF2 SavedModel inference
  → embedding vectors (dims from model config)
  → save to Parquet (incremental, atomic)
```

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

### Clustering Workflow

```
selected embedding sets (must share vector_dim)
  → load from Parquet
  → optional UMAP dimensionality reduction
  → HDBSCAN clustering
  → persist clusters + assignments
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/audio/upload` | Upload audio file (SHA-256 dedup) |
| GET | `/audio/` | List audio files |
| GET | `/audio/{id}` | Get audio file details |
| PUT | `/audio/{id}/metadata` | Update metadata |
| POST | `/processing/jobs` | Create processing job |
| GET | `/processing/jobs` | List processing jobs |
| GET | `/processing/jobs/{id}` | Get job status |
| POST | `/processing/jobs/{id}/cancel` | Cancel job |
| GET | `/processing/embedding-sets` | List embedding sets |
| GET | `/processing/embedding-sets/{id}` | Get embedding set |
| POST | `/clustering/jobs` | Create clustering job |
| GET | `/clustering/jobs/{id}` | Get clustering job |
| GET | `/clustering/jobs/{id}/clusters` | List clusters |
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
  audio/raw/{audio_file_id}/original.(wav|mp3)
  embeddings/{model_version}/{audio_file_id}/{encoding_signature}.parquet
  clusters/{clustering_job_id}/clusters.json
  clusters/{clustering_job_id}/assignments.parquet
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

---

## Tech Stack

- **Backend**: Python + FastAPI
- **Queue**: SQL-backed polling queue
- **DB**: SQLite with WAL mode (MVP)
- **Embedding**: TensorFlow Lite + TF2 SavedModel (flex delegate for custom ops; macOS GPU via tensorflow-metal; FakeTFLiteModel/FakeTF2Model for testing)
- **Clustering**: UMAP + HDBSCAN
- **Storage**: Local filesystem
