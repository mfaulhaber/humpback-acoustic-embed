# Perch v2 as a First-Class Classifier Family — Design

**Date:** 2026-04-17
**Status:** Approved (pending implementation)
**ADR:** ADR-055 (to be appended on implementation)

## 1. Purpose

Make the locally-trained `perch_v2.tflite` embedding model (waveform-in, 1536-d, 5 s @ 32 kHz) a first-class citizen of the classifier pipeline so that users can:

1. **Tune** classifier-head hyperparameters on top of perch_v2 embeddings, using the same labeled detection windows that TF2-model classifiers are tuned on today.
2. **Train a deployable** `ClassifierModel` that uses perch_v2 as its embedding model.
3. **Run detection jobs** using that perch_v2-based classifier end-to-end on audio folders and hydrophone streams, with no regressions to the TF2 path.

**Scope is explicitly binary whale / not-whale classification.** Vocalization-type labels (`vocalization_labels`) are not carried across model families; they remain a TF2-side concern for the foreseeable future.

## 2. Non-Goals

- Fine-tuning the perch_v2 TFLite weights themselves. The weights are fixed; only the downstream sklearn / MLP head is tuned.
- Multi-label vocalization classifiers on perch_v2 embeddings.
- Mixing embeddings from different model families in a single manifest (this is explicitly rejected, not merely unsupported).
- Back-filling historical TF2 detection embeddings into any new artifact shape — existing data stays where it is.

## 3. Current State (what already works)

- The `model_configs` table is an authoritative registry with `model_type` (`tflite` / `tf2_saved_model`), `input_format` (`waveform` / `spectrogram`), `vector_dim`, and `path`.
- `workers/model_cache.get_model_by_version()` instantiates the right runner class from a `ModelConfig` row and returns `(model, input_format)`.
- `classifier_worker/detection.py` + `classifier/detector.py` already branch on `input_format` at inference time — for waveform-in models it passes raw windows; for spectrogram-in models it pre-computes log-mel.
- `classifier_models.model_version` is a plain string that is resolved against `model_configs.name` at runtime. No schema change needed there.
- The training worker already has a manifest-driven path (`source_mode="autoresearch_candidate"`) that trains a deployable classifier from a manifest file.

**Conclusion:** detection-side inference with perch_v2 is a zero-code-change problem once the registry row and a `ClassifierModel` row exist. The work lives on the training/tuning side.

## 4. Gap Map

| Concern | Status |
|---|---|
| `ModelConfig` row for perch_v2.tflite | **Missing.** Seed via existing admin "scan models" path or Alembic data migration. |
| Loading perch_v2 at inference time | Works via `get_model_by_version`. |
| Detection jobs using perch_v2-based classifier | Works once `ClassifierModel` row exists. |
| Re-embedding detection windows under perch_v2 | **Missing.** `detection_embedding_jobs` is 1-per-detection-job; path not keyed by model version. |
| Manifest builder over perch_v2 embeddings | Partial. Reads from `detection_embeddings_path(detection_job_id)`; must become model-aware. |
| Tuning a head on perch_v2 embeddings | Works once manifest points at perch_v2 embeddings. |
| **Deployable** classifier from perch_v2 embeddings | Partial. Manifest-based training path exists (autoresearch promotion); needs to be general. |
| TrainingTab UI accepting detection-job sources | **Missing** — Training today accepts only `embedding_sets`. |
| Progress / status of re-embedding in the UI | **Missing.** |

## 5. Decision: Approach A (Surgical Extension)

Extend existing tables and workers rather than introducing a new `labeled_manifests` artifact. Detailed in sections 6–10.

**Alternatives considered:**

- **Approach B — "promote from tuning" as the sole deployment path.** Keeps TrainingTab unchanged; detection-job-based training reachable only via the Tuning UI. Rejected: less orthogonal; forces a Tuning round-trip even when the hyperparameters are already known.
- **Approach C — unified `labeled_manifests` artifact.** Promote the manifest to a first-class table consumed by both Training and Tuning, with explicit `embedding_model_version` and heterogeneous-source validation baked into the schema. **Preferred long-term model.** Rejected *now* because it touches TrainingTab, training service, workers, and DB more broadly than this project needs, and the same label-preservation guarantees are met by A. Earmarked as a future refactor once Training and Tuning data-source handling have diverged enough to justify the consolidation. ADR-055 records this explicitly.

**Why A satisfies label preservation.** Human labels live in `vocalization_labels` (per row keyed by `(detection_job_id, row_id)`) and in the binary label columns of the detection row store. The embedding parquet only needs to carry `row_id` for the join; whether the parquet came from TF2 or perch_v2 makes no difference to label retrieval. For this project only the binary row-store labels are used.

## 6. Embedding Model Registration

**perch_v2 `ModelConfig` row.**

| Column | Value |
|---|---|
| `name` | `perch_v2` |
| `display_name` | `Perch v2 (TFLite)` |
| `path` | `models/perch_v2.tflite` |
| `model_type` | `tflite` |
| `input_format` | `waveform` |
| `vector_dim` | `1536` |
| `is_default` | `False` |

Registration path: the existing admin "scan models" flow (`services/model_registry_service.scan_model_files`) already discovers `.tflite` files and introspects dimensions. Verify it produces the expected row for perch_v2.tflite; if `input_format` or `vector_dim` are not auto-detected correctly for this particular file, allow manual override on the Models admin page (already present) and/or add an Alembic data migration that inserts the row idempotently on upgrade.

## 7. Re-Embedding Pipeline

**Schema change.** `detection_embedding_jobs` grows two dimensions:

1. **Keying by `(detection_job_id, model_version)`.** Add a `model_version: str NOT NULL` column. Replace any existing `detection_job_id` uniqueness with a composite unique `(detection_job_id, model_version)`. Backfill existing rows from `classifier_models.model_version` of the detection job's source classifier.
2. **Progress fields.** Add `rows_processed: int NOT NULL DEFAULT 0` and `rows_total: int NULL`. The worker sets `rows_total` after decoding the row store, then increments `rows_processed` per batch.

**Storage path.** `storage.detection_embeddings_path(storage_root, detection_job_id, model_version)` now includes the model version in its path. Existing files are physically moved during the Alembic upgrade so the helper has a single consistent rule.

**Worker.** The existing `detection_embedding_worker` gains a `target_model_version` parameter (today it resolves the embedding model through the detection job's source classifier; now it resolves via `get_model_by_version(target_model_version)`). It reads the detection job's row store, re-windows raw audio, runs the selected embedding model, and writes a parquet keyed by `row_id`. Idempotency is enforced by the composite unique constraint: completed rows short-circuit; queued/running ones are picked up by the normal worker loop.

**Failure handling.** On error, the row transitions to `failed` with `error_message` populated; Retry in the UI re-enqueues the same row.

## 8. Manifest Builder

**Explicit model version.** `hyperparameter_manifests` grows `embedding_model_version: str NOT NULL`. The builder reads embeddings from `detection_embeddings_path(detection_job_id, embedding_model_version)` and raises if that artifact does not exist.

**Heterogeneous source rejection.** For each `training_job_id` source, the builder verifies that the referenced training job's `model_version` matches the manifest's `embedding_model_version`; mismatched sources fail the manifest build rather than silently mixing vectors from different models.

**Label join.** For perch_v2 detection sources, labels come from the binary columns in the detection row store (`humpback`, `background`, `ship`, `orca`). The `vocalization_labels` join path is **not** exercised for perch_v2 manifests. The existing label-priority logic still applies for TF2 manifests unchanged.

## 9. Training

**New source mode.** `classifier_training_jobs.source_mode` gains the value `"detection_manifest"`. On submit with this mode, the service creates an internal manifest (same builder as tuning) and the worker runs the existing manifest-based training path. The output is a regular `ClassifierModel` row with `model_version=perch_v2`.

**Validation.**
- Every source must match the requested `embedding_model_version`.
- At least one positive and one negative binary label must be present across the selected detection jobs.
- A training job cannot mix `embedding_sets` sources with `detection_jobs` sources.

**Model artifact.** Written as today: `classifier_dir(job.id) / "model.joblib"` plus `training_summary` JSON. The new fields on the summary describe the manifest source (detection job IDs + embedding model version + split summary).

## 10. Detection

No detection-side code changes. A perch_v2-based `ClassifierModel` appears in the classifier dropdown like any other; the detection worker resolves its embedding model through `get_model_by_version` and runs the waveform-in inference path that already exists.

## 11. UI Changes

**Admin / Model Registry.** No change beyond confirming perch_v2.tflite registers with the correct `input_format` and `vector_dim`.

**`<DetectionSourcePicker>` shared component.** Extracted once, reused by TuningTab and TrainingTab. Responsibilities:

- Detection job multi-select
- Embedding-model selector (default: inferred from the first selected source)
- Inline **Re-embedding status** table: one row per `(detection_job_id, model_version)` pair, columns = Job · Status · Progress · Action. States: Not started / Queued / Running (with `rows_processed/rows_total` + %) / Complete / Failed (with error popover + retry).
- **Re-embed now** action enqueues re-embedding jobs for missing pairs.
- Polling via TanStack Query every ~2 s while any row is queued or running; stops otherwise.
- Blocks the downstream **Create Manifest** / **Create Training Job** button until every listed pair is Complete.

**TuningTab (ManifestsSection).** Uses `<DetectionSourcePicker>`. Validates all training-job sources match the selected embedding model.

**TrainingTab.** Adds a **Source mode** radio — *Embedding sets* (today) vs *Detection jobs* (new). In detection-jobs mode, `<DetectionSourcePicker>` replaces the positive/negative embedding-set selectors; classifier-head advanced options stay unchanged.

**Backend endpoint.** `GET /detection-embedding-jobs?detection_job_ids=...&model_version=...` returns the status/progress rows for the listed pairs in one query.

## 12. Data Migrations

| Migration | Summary |
|---|---|
| `049_detection_embedding_jobs_model_version.py` | Add `model_version` (nullable → backfill from source classifier → NOT NULL), add composite unique `(detection_job_id, model_version)`, physically move existing parquet files to the new path, add `rows_processed` / `rows_total`. Uses `op.batch_alter_table()`. |
| `050_hyperparameter_manifests_embedding_model_version.py` | Add `embedding_model_version` (nullable → backfill from each manifest's first resolved source → NOT NULL). |
| `051_perch_v2_model_config_seed.py` | Idempotent insert of the perch_v2 `ModelConfig` row if it is not already present. |

No changes to `vocalization_labels`, `training_datasets`, `training_dataset_labels`, `classifier_models`, or `detection_jobs`.

## 13. ADR-055

Append to `DECISIONS.md`:

- **Title:** ADR-055 — Perch v2 as first-class classifier family via surgical extension.
- **Status:** Accepted.
- **Context:** Existing model registry already loads arbitrary TFLite models; detection worker already branches on `input_format`; gap is on the training/tuning side.
- **Decision:** Extend `detection_embedding_jobs` to be per-(detection_job, embedding_model) + add re-embedding worker + extend manifest builder with explicit `embedding_model_version` + extend TrainingTab with a detection-jobs source mode. Binary whale/not-whale scope only.
- **Alternatives considered:** Approach B (promote-from-tuning only) and Approach C (unified `labeled_manifests` artifact). C is the preferred long-term model; deferred because it exceeds this project's scope and A meets the label-preservation requirement.
- **Consequences:** `detection_embedding_jobs` is now N-to-1 with detection jobs. Shared `<DetectionSourcePicker>` lives in both TuningTab and TrainingTab. Manifest builder enforces homogeneous `embedding_model_version`. Future: if a spectrogram-input embedding model with a different spectral shape is ever added, the hardcoded `n_mels=128 / hop_length=1252 / target_frames=128` in `detector.py` must move into `feature_config`.

## 14. Testing Strategy

**Backend unit tests.**
- `tests/workers/test_detection_reembedding_worker.py` — fake waveform TFLite model; verifies idempotency per `(detection_job_id, model_version)`, correct row-keyed parquet, progress updates, failure persistence.
- `tests/migrations/test_049_detection_embedding_jobs_model_version.py` — backfill correctness, composite uniqueness, file-move path.
- `tests/services/test_hyperparameter_manifest_builder.py` — heterogeneous source rejection, explicit `embedding_model_version` routing, binary-only label path.
- `tests/workers/test_classifier_training_detection_manifest.py` — `source_mode="detection_manifest"` produces a `ClassifierModel` with the requested `model_version` and `vector_dim`.
- `tests/services/test_classifier_service_validation.py` — rejection of mixed `embedding_sets` + `detection_jobs` sources; rejection of sources with mismatched `model_version`.

**E2E integration.** Uses a fake waveform TFLite model registered as `perch_v2` (`vector_dim=1536`). Run detection with an existing TF2 classifier on a small audio fixture, then: re-embed → build tuning manifest → run a tiny search → train a deployable classifier via detection-manifest mode → run a detection job with the new classifier → verify outputs.

**Frontend.**
- `frontend/tests/e2e/classifier-tuning-reembed.spec.ts` — selecting detection jobs + embedding model with missing embeddings shows the Re-embedding status table; Create Manifest is disabled; all-Complete enables it (API stubbed).
- `frontend/tests/e2e/classifier-training-detection-mode.spec.ts` — detection-jobs source mode end-to-end.
- Extend `admin_models` Playwright coverage to assert perch_v2-style registration works.

**Verification gates** (CLAUDE.md §10.2): `uv run ruff format --check`, `uv run ruff check`, `uv run pyright`, `uv run pytest tests/`, `cd frontend && npx tsc --noEmit`.

## 15. Doc Updates on Landing

- **CLAUDE.md** — §9.1 add perch_v2 as a registered embedding model family; §9.2 note latest migration number after landing.
- **DECISIONS.md** — append ADR-055.
- **README.md** — mention perch_v2 classifier support in the feature list.
- **docs/reference/data-model.md** — reflect the new `detection_embedding_jobs` shape.
- **docs/reference/classifier-api.md** — document the new `GET /detection-embedding-jobs` query params and the `detection_manifest` source mode.
- **docs/reference/storage-layout.md** — reflect the model-versioned detection-embeddings path.
