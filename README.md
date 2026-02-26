# Humpback Acoustic Embedding & Clustering System

## Overview

This project processes humpback whale audio recordings (MP3/WAV) into reusable
embedding vectors using a Perch-compatible TFLite model, then performs clustering
with optional ecological/behavioral metadata.

Key features:
- Asynchronous job queue (SQL-backed, restart-safe)
- Idempotent encoding (no reprocessing for same config)
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
  → resample to target SR
  → slice into N-second windows (default 5s)
  → (optional) log-mel features
  → TFLite inference
  → embedding vectors (512–1024 dims)
  → save to Parquet (incremental, atomic)
```

Encoding is associated with the audio file and configuration. Reprocessing is
skipped when an EmbeddingSet with the same encoding_signature already exists.

### Clustering Workflow

```
selected embedding sets
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
| `HUMPBACK_VECTOR_DIM` | `512` | Embedding vector dimensions |
| `HUMPBACK_USE_REAL_MODEL` | `false` | Use real TFLite model vs fake |

---

## Tech Stack

- **Backend**: Python + FastAPI
- **Queue**: SQL-backed polling queue
- **DB**: SQLite with WAL mode (MVP)
- **Embedding**: TFLite runtime (or FakeTFLiteModel for testing)
- **Clustering**: UMAP + HDBSCAN
- **Storage**: Local filesystem
