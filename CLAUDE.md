# Humpback Acoustic Embedding & Clustering Platform

## 1. Purpose

This system processes humpback whale audio recordings into reusable embedding vectors
using a Perch-compatible TFLite model, then performs clustering with optional
behavioral/ecological metadata.

The system must:
- Support asynchronous, resumable workflows
- Persist workflow state in SQL
- Prevent reprocessing of already-encoded audio for the same configuration
- Allow separate queuing of processing and clustering jobs
- Provide a web UI for job management and inspection

This document defines the high-level architecture, system behavior, and engineering rules
for implementing and testing the system.

---

## 2. High-Level Architecture

Components:
1. Web UI
2. API Server
3. Workflow Queue (SQL-backed)
4. Worker Processes (processing + clustering)
5. SQL Database
6. Object/File Storage (audio + embeddings + cluster outputs)
7. Clustering Engine

All components run locally for MVP but should be designed so workers can scale horizontally.

---

## 3. Core Design Principles

### 3.1 Idempotent Encoding (No Reprocessing)
Each audio file is encoded once per (model_version, window_size, target_sample_rate, feature_config).

A ProcessingJob MUST:
- compute a stable "encoding_signature"
- check for an existing completed embedding set with that signature
- skip work if the embedding set exists

### 3.2 Resumable Workflow
All steps are recorded in SQL. Workers must be restart-safe:
- jobs can resume after crash/restart
- partial artifacts should be either:
  - safely overwritten, or
  - written to temp and atomically promoted on completion

### 3.3 Asynchronous, Observable Jobs
Jobs are queued and executed in the background by workers.
UI can monitor via polling or a push channel.

---

## 4. Data Model (Conceptual)

### AudioFile
- id
- filename
- checksum_sha256
- duration_seconds
- sample_rate_original
- created_at

### AudioMetadata (optional, editable)
- audio_file_id (FK)
- tag_data (JSON)
- visual_observations (JSON)
- group_composition (JSON)
- prey_density_proxy (JSON)

### ProcessingJob
- id
- audio_file_id (FK)
- status: queued | running | complete | failed | canceled
- encoding_signature (unique per audio+config)
- model_version
- window_size_seconds
- target_sample_rate
- feature_config (JSON)
- created_at
- updated_at
- error_message (nullable)

### EmbeddingSet (one per audio+signature)
- id
- audio_file_id (FK)
- encoding_signature (unique)
- model_version
- window_size_seconds
- target_sample_rate
- vector_dim
- parquet_path
- created_at

### ClusteringJob
- id
- status
- embedding_set_ids (JSON array or join table)
- parameters (JSON)
- created_at
- updated_at
- error_message

### Cluster
- id
- clustering_job_id
- cluster_label
- size
- metadata_summary (JSON)

### ClusterAssignment
- cluster_id
- embedding_row_id (row index in parquet OR FK to Embedding row table)

Note: do NOT store every embedding vector row in SQL in MVP; store embeddings in Parquet and
only store indexing/assignment references.

---

## 5. Processing Workflow

Input:
- audio file (MP3 or WAV)
- optional metadata

Pipeline:
1. Decode audio (MP3/WAV)
2. Resample to target sample rate (Perch target)
3. Slice into N-second windows (default: 5)
4. Feature extraction (log-mel if not built into model)
5. TFLite inference (batched)
6. Produce embedding vectors (512–1024 dims)
7. Save embeddings to Parquet (incremental)
8. Persist EmbeddingSet row in SQL
9. Mark ProcessingJob complete

Rules:
- embeddings must be written incrementally (avoid holding all vectors in RAM)
- write parquet to temp path, then atomically rename/move on completion
- job must be restart-safe: if a temp exists, worker may resume or restart cleanly
- if EmbeddingSet exists for encoding_signature and is complete → skip

---

## 6. Clustering Workflow

Input:
- selected embedding sets
- optional metadata filters (subset)

Pipeline:
1. Load embeddings from Parquet
2. Optional dimensionality reduction (UMAP)
3. Clustering (HDBSCAN default)
4. Persist clusters and assignments
5. Compute per-cluster metadata summaries
6. Mark ClusteringJob complete

---

## 7. Workflow Queue

Use SQL-backed queue semantics:
- Workers select queued jobs with row-level locking / "claim" update
- Status transitions:
  - queued → running → complete
  - queued → running → failed
  - queued → canceled

Concurrency:
- prevent two running ProcessingJobs for same encoding_signature
- allow multiple clustering jobs in parallel (configurable)

---

## 8. Storage Layout

/audio/
  raw/{audio_file_id}/original.(wav|mp3)
/embeddings/
  {model_version}/{audio_file_id}/{encoding_signature}.parquet
  {model_version}/{audio_file_id}/{encoding_signature}.tmp.parquet
/clusters/
  {clustering_job_id}/clusters.json
  {clustering_job_id}/assignments.parquet

---

## 9. Web UI Requirements

Processing:
- Upload audio
- Attach/edit metadata
- Queue processing job
- Monitor status and progress
- Audio detail: show file info, processing history, embedding set(s)

Clustering:
- Select embedding sets
- Configure clustering parameters
- Queue clustering job
- Monitor status
- Cluster detail: members and metadata distributions

Editing:
- Manage/edit AudioMetadata associated with audio file

---

## 10. Testing Requirements (MANDATORY)

Testing is not optional. Every meaningful change must include:
- unit tests for new logic
- integration tests for API endpoints
- at least one end-to-end smoke test path that exercises the real workflows

### 10.1 Unit Tests
Add unit tests for:
- encoding_signature computation (idempotency)
- audio window slicing logic
- feature extraction shape correctness
- TFLite runner batching (mock interpreter acceptable)
- Parquet writer behavior (temp + atomic promote)
- clustering pipeline (small synthetic embeddings)

Guidelines:
- prefer deterministic tests
- isolate file I/O behind temp directories
- mock external dependencies when appropriate (e.g., TFLite interpreter)

### 10.2 Running Tests Locally
The repo must include:
- `pytest` configuration
- a single command to run unit+integration tests
- a command to run tests continuously on file changes

Required commands:
- `pytest` (all tests)
- `pytest -q` (quiet)
- `pytest -k <pattern>` (focused)
- "watch mode" (choose one):
  - `pytest-watch` (`ptw`) OR
  - `watchexec -r pytest` OR
  - `entr -r pytest`

Document the chosen tool in README and add it to dev dependencies.

### 10.3 End-to-End Smoke Test (E2E)
Add a minimal E2E test that:
1. Starts API + worker (in-process for tests or via subprocess)
2. Uploads a small fixture audio file
3. Queues a ProcessingJob
4. Polls until job completes
5. Verifies EmbeddingSet exists and parquet file is readable
6. Queues a ClusteringJob on that embedding set
7. Polls until complete
8. Verifies clusters and assignments exist and are consistent

Constraints:
- E2E must run in under a few minutes locally
- Use a tiny audio fixture (e.g., 10–20 seconds)
- Use a tiny embedding model stub if needed (see below)

### 10.4 Model Stub Strategy (So Tests Are Fast)
For unit/integration/E2E tests:
- Provide a "FakeTFLiteModel" implementation that returns deterministic embeddings
  (e.g., sine/cosine transforms of window index)
- Gate real Perch model execution behind an environment flag
  - default tests use FakeTFLiteModel
  - optional manual run uses real model if available

---

## 11. Definition of Done (Engineering)
A PR/change is "done" only if:
- unit tests added/updated for changed behavior
- test suite passes locally
- E2E smoke test passes locally
- idempotency rules preserved (no duplicate embedding sets)

---

## 12. Non-Goals (MVP)
- Model fine-tuning
- Real-time streaming inference
- Multi-tenant support
- Distributed GPU execution
