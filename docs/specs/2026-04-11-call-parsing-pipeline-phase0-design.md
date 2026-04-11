# Humpback Call Parsing Pipeline — Phase 0: Architecture & Scaffolding

**Date:** 2026-04-11
**Status:** Approved

## Problem

The existing detection pipeline (Perch binary detector → windowed classifier → NMS/prominence/tiling selection → row-store output) is effective for coarse whale-presence detection and per-5s-window multi-label call-type tagging, but it has four structural limitations for downstream vocalization analysis:

1. Detection windows can clip or miss individual vocalizations — events are forced onto 5s window alignment.
2. A single 5s region can contain multiple distinct call types, but labels are clip-level, not event-level.
3. The current multi-label vocalization classifier (ADR-042) runs on fixed 5s Perch embeddings; it cannot classify variable-length event crops.
4. Inference does not recover event **order** within a whale-active region — there is no motif/transition-ready ordered sequence output.

The fix is a four-pass pipeline that treats Perch as a high-recall proposal stage, moves segmentation into a learned framewise model, moves classification onto event crops, and preserves temporal ordering end-to-end. This document specifies **Phase 0** — the architecture contract and scaffolding that binds the four passes together. Pass 1, Pass 2, Pass 3, and Pass 4 ship as their own subsequent feature branches, each brainstormed and planned independently but against this contract.

## Approach — four-pass pipeline

```
Long audio recording
  → Pass 1: Perch-based whale activity detector    (preserves dense trace, outputs padded regions)
  → Pass 2: spectrogram event segmentation         (learned PyTorch CRNN/transformer, framewise head)
  → Pass 3: event-level call-type classifier       (PyTorch CNN on event crops, reuses vocabulary)
  → Pass 4: ordered typed-event sequence           (read-only export endpoint)
```

- Pass 1 is ~70% built today: `detector.py` already computes dense per-window confidence and `detector_utils.merge_detection_events` already produces first-class hysteresis events. The gap is that events are immediately collapsed into NMS/prominence/tiling rows instead of being surfaced as padded continuous regions.
- Pass 2 is net-new: learned framewise onset/offset detection in PyTorch. No PyTorch training infrastructure exists in the project today.
- Pass 3 extends the existing multi-label vocalization classifier (ADR-042): the `vocalization_models` and `vocalization_training_jobs` tables grow a `model_family` / `input_mode` axis, vocabulary and thresholds carry over unchanged, and a new `pytorch_event_cnn` family runs on variable-length event crops.
- Pass 4 is a read-only API endpoint, not a queue job.

## Phase 0 scope

Phase 0 is a real implementation cycle that ships a working-but-empty scaffold. After Phase 0, the project compiles, tests pass, the migration applies cleanly, a parent `CallParsingRun` can be created and its status queried, and each subsequent Pass spec starts from a ready-made skeleton rather than scaffolding from scratch.

**Phase 0 ships:**

- Alembic migration `042` creating new tables (`call_parsing_runs`, `region_detection_jobs`, `event_segmentation_jobs`, `event_classification_jobs`, `segmentation_models`) and extending existing tables (`vocalization_models`, `vocalization_training_jobs`) with `model_family` / `input_mode` columns.
- SQLAlchemy model classes for the new tables.
- `src/humpback/call_parsing/` package with `types.py` (dataclasses), `storage.py` (parquet I/O helpers).
- `src/humpback/ml/` shared PyTorch harness (device selection, training loop, checkpointing) with unit tests that train a tiny toy torch module on synthetic data.
- Empty worker shells for the three new workers (claim → fail with `NotImplementedError` → status `failed`), wired into the claim priority order.
- API router `src/humpback/api/routers/call_parsing.py` with functional parent-run CRUD and stub pass endpoints (returning `501 Not Implemented`).
- Pydantic schemas in `src/humpback/schemas/call_parsing.py`.
- Service layer in `src/humpback/services/call_parsing.py` for parent/child orchestration.
- Behavior-preserving refactor of `detector.py`: extract the dense-inference-to-hysteresis-events path into a shared helper `compute_hysteresis_events()` so both the existing window-selecting detector and the forthcoming Pass 1 `RegionDetectionWorker` can consume it.
- PyTorch added to each TF extra in `pyproject.toml` (`tf-macos`, `tf-linux-cpu`, `tf-linux-gpu`).
- Documentation: `CLAUDE.md` worker-order and API-surface updates, `DECISIONS.md` ADR-048 for the four-pass architecture.

**Phase 0 does NOT ship:**

- Pass 1 region-generation logic (worker fail-fast, stub endpoint)
- Pass 2 CRNN/transformer model or training
- Pass 3 BirdNET-like CNN or inference
- Pass 4 sequence-export logic
- Any model training or label curation
- Any frontend work

## Execution model

### Parent/child job layout

A new parent table `call_parsing_runs` threads one end-to-end pipeline run on an audio source:

```
call_parsing_runs
  id                                   (uuid4 string)
  audio_source_id                      (FK to audio_files or hydrophone source record)
  status                               (queued | running | complete | failed | partial | canceled)
  config_snapshot                      (JSON: thresholds, model refs, padding, etc.)
  region_detection_job_id              (nullable FK)
  event_segmentation_job_id            (nullable FK)
  event_classification_job_id          (nullable FK)
  error                                (nullable TEXT)
  created_at, updated_at, completed_at
```

Four new child job tables, one per pass, each with the standard queue fields (`status`, `error`, counts, timestamps) plus a nullable `parent_run_id` FK to `call_parsing_runs` and an upstream-pass FK for individual runnability:

| Job table                    | Pass | Upstream FK                   | Model FK                                    |
|------------------------------|------|-------------------------------|---------------------------------------------|
| `region_detection_jobs`      | 1    | —                             | `model_configs.id`, `classifier_models.id`  |
| `event_segmentation_jobs`    | 2    | `region_detection_job_id`     | `segmentation_models.id`                    |
| `event_classification_jobs`  | 3    | `event_segmentation_job_id`   | `vocalization_models.id`                    |

**Individual runnability:** creating a child pass job standalone leaves `parent_run_id` NULL; the `CallParsingRun` service sets it when orchestrating a parent run. Each child job also stores an explicit upstream-pass FK, so Pass 2 can point at any completed Pass 1 and Pass 3 at any completed Pass 2 without a parent run existing. Re-running a downstream pass against a different model is a first-class operation.

**Pass 4 is not a job.** It is a read-only API endpoint (`GET /call-parsing/runs/{id}/sequence`) that streams the sorted typed events from the latest successful Pass 3 child's parquet.

### Worker priority

The three new workers slot into the existing claim order (CLAUDE.md §8.7) between `vocalization inference` and `manifest generation`:

```
search → processing → clustering → classifier training → detection →
extraction → detection embedding generation → label processing →
retrain → vocalization training → vocalization inference →
region detection → event segmentation → event classification →
manifest generation → hyperparameter search
```

Priority increases left-to-right in the existing convention (later categories drain first when contention exists). Placing the three new workers at the end of the vocalization cluster means a ready Pass 3 job drains before a new Pass 1 starts — depth-first on the pipeline chain.

## Data contract

### Region / Event / TypedEvent schemas

The three domain types live as frozen dataclasses in `src/humpback/call_parsing/types.py`:

```
Region
  region_id            (UUID4 string)
  start_sec            (float, seconds from audio start)
  end_sec              (float)
  padded_start_sec     (float, clamped to audio bounds)
  padded_end_sec       (float, clamped to audio bounds)
  max_score            (float, peak detector confidence in region)
  mean_score           (float, mean detector confidence in region)
  n_windows            (int, number of merged windows)

Event
  event_id                 (UUID4 string)
  region_id                (UUID4 string, cross-file reference)
  start_sec                (float, absolute in source audio)
  end_sec                  (float)
  center_sec               (float, convenience)
  segmentation_confidence  (float, 0–1)

TypedEvent
  event_id           (UUID4 string, cross-file reference)
  start_sec          (float, copied from Event for sort-ability)
  end_sec            (float, copied from Event)
  type_name          (str, from vocalization_types vocabulary)
  score              (float, per-type probability)
  above_threshold    (bool, applies per-type threshold at inference time)
```

These are the Phase 0 minimum fields. Per-pass specs may add columns (e.g., Pass 2 may add `onset_uncertainty`, Pass 3 may add `secondary_types`) without breaking the contract.

### Parquet artifact layout

Each pass writes per-job Parquet files under its storage directory, following the existing detection-row pattern:

| Pass | Parquet files under `storage_root/call_parsing/<pass>/<job_id>/` |
|---|---|
| 1 | `trace.parquet` (dense per-window scores: `time_sec`, `score`), `regions.parquet` (one row per padded region) |
| 2 | `events.parquet` (one row per event), optional `frame_probs.parquet` when framewise persistence is flagged on |
| 3 | `typed_events.parquet` (one row per `(event_id, type_name)` prediction) |

Cross-pass linkage uses UUID `event_id` / `region_id` values embedded in the parquet — the same pattern `vocalization_labels` already uses to reference detection `row_id`. No row-level SQL tables, no row-level foreign keys.

### Model registries

- **`classifier_models`** (existing): used by Pass 1 unchanged — the binary detector.
- **`segmentation_models`** (new): Pass 2 PyTorch checkpoint registry. Minimum fields: `id`, `name`, `model_family` (starts with `"pytorch_crnn"`), `model_path`, `config_json`, `training_job_id`, `created_at`. Extended as needed in the Pass 2 spec.
- **`vocalization_models`** (extended): two new columns.
  - `model_family` — `"sklearn_perch_embedding"` (default, existing rows migrate to this) | `"pytorch_event_cnn"` (Pass 3)
  - `input_mode` — `"detection_row"` (default) | `"segmented_event"` (Pass 3)
  Existing sklearn rows default cleanly; zero breaking change.
- **`vocalization_training_jobs`** (extended): same two columns (`model_family`, `input_mode`) with the same defaults. Pass 3 training reuses this table with the new family rather than introducing a parallel table.
- **`segmentation_training_jobs`** (new, added in Pass 2 spec, not Phase 0): Pass 2 training jobs. Phase 0 does NOT create this table — it waits for the Pass 2 spec, where the training config fields will be known.

## Framework strategy

### PyTorch integration

This is the project's first PyTorch workload. Current stack: TensorFlow (Perch TFLite + Keras metric learning) + scikit-learn, with mutually exclusive TF extras (`tf-macos`, `tf-linux-cpu`, `tf-linux-gpu`). PyTorch is bundled into each of the three TF extras:

- `tf-macos` → adds `torch` (macOS CPU/MPS wheel)
- `tf-linux-cpu` → adds `torch` + `--index-url` marker selecting CPU wheel
- `tf-linux-gpu` → adds `torch` + `--index-url` marker selecting CUDA wheel

Rationale: the TF extras already represent "pick your platform"; bundling torch adds zero new user-facing knobs and keeps install complexity constant. A torch-only install path is not a future requirement.

### Shared `src/humpback/ml/` harness

Pass 2 and Pass 3 both train PyTorch models. A shared harness prevents duplicating the training loop:

- `ml/device.py` — one helper for MPS / CUDA / CPU selection, used by both trainers and inference workers.
- `ml/training_loop.py` — generic `fit(model, optimizer, train_loader, val_loader, callbacks, epochs, ...)` implementing gradient accumulation, LR schedule hook, early stopping, checkpoint-every-N-epochs. Metric callbacks plug in for Pass 2's framewise F1 and Pass 3's per-type F1 without the loop knowing the task type.
- `ml/checkpointing.py` — save/load helpers: `save_checkpoint(path, model, optimizer, config)` writes a `.pt` file with `model_state_dict`, `optimizer_state_dict`, and `config_json`; `load_checkpoint(path, model, optimizer=None)` restores.

Model-specific code (architecture, loss, dataset) lives in each pass's own subpackage and imports `humpback.ml`. Phase 0 ships the harness plus unit tests that train a tiny toy module on synthetic data.

## Repo layout

```
src/humpback/
├── call_parsing/                    # NEW — four-pass pipeline
│   ├── __init__.py
│   ├── types.py                     # Region / Event / TypedEvent dataclasses + schemas
│   ├── storage.py                   # parquet I/O helpers for all pass artifacts
│   ├── regions.py                   # (Pass 1) dense-trace → padded regions logic — stub in Phase 0
│   ├── segmentation/                # (Pass 2) — not created in Phase 0
│   ├── event_classifier/            # (Pass 3) — not created in Phase 0
│   └── sequence.py                  # (Pass 4) — not created in Phase 0
├── ml/                              # NEW — shared PyTorch infrastructure
│   ├── __init__.py
│   ├── device.py
│   ├── training_loop.py
│   └── checkpointing.py
├── workers/
│   ├── region_detection_worker.py       # NEW — empty shell
│   ├── event_segmentation_worker.py     # NEW — empty shell
│   └── event_classification_worker.py   # NEW — empty shell
├── api/routers/
│   └── call_parsing.py              # NEW — parent-run CRUD functional, pass endpoints 501
├── schemas/
│   └── call_parsing.py              # NEW — Pydantic request/response
└── services/
    └── call_parsing.py              # NEW — parent/child orchestration
```

Directories not yet created in Phase 0 (`segmentation/`, `event_classifier/`, `sequence.py`) are reserved by the design — the Pass specs that follow will create them.

## API surface

New router mounted at `/call-parsing/`:

### Parent runs
- `POST /call-parsing/runs` — create a parent run; creates a Pass 1 job and optionally pre-queues Pass 2/3 with upstream FKs resolved when each pass completes.
- `GET /call-parsing/runs` — list
- `GET /call-parsing/runs/{id}` — detail including nested pass statuses
- `DELETE /call-parsing/runs/{id}` — cascade delete all child jobs and artifacts
- `GET /call-parsing/runs/{id}/sequence` — **Pass 4 output** (returns 501 in Phase 0)

### Individual pass jobs (each standalone-runnable)
- `POST /call-parsing/region-jobs`, `GET`, `GET {id}`, `DELETE {id}` — Pass 1 (returns 501 on POST in Phase 0)
- `POST /call-parsing/segmentation-jobs`, `GET`, `GET {id}`, `DELETE {id}` — Pass 2 (POST body requires `region_detection_job_id`)
- `POST /call-parsing/classification-jobs`, `GET`, `GET {id}`, `DELETE {id}` — Pass 3 (POST body requires `event_segmentation_job_id`)

### Artifact access
- `GET /call-parsing/region-jobs/{id}/trace` — dense per-window scores
- `GET /call-parsing/region-jobs/{id}/regions`
- `GET /call-parsing/segmentation-jobs/{id}/events`
- `GET /call-parsing/classification-jobs/{id}/typed-events`

Phase 0 implements parent-run CRUD as real functionality (the metadata doesn't need pass logic). All pass-level POST and artifact endpoints return `501 Not Implemented` with a message naming the pass that owns it.

## Pass 1 code reuse — `detector.py` refactor

Pass 1 shares two pieces with the existing detector: dense Perch inference plus per-window classifier scoring, and hysteresis event merging (`detector_utils.merge_detection_events`). The difference is what happens downstream: today the detector calls snap-merge + NMS/prominence/tiling and writes window-aligned rows; Pass 1 wants to skip window selection, pad the hysteresis events into regions, and write `regions.parquet` + `trace.parquet`.

Phase 0 extracts the "audio → dense per-window scores → hysteresis events" path from `detector.py` into a shared helper:

```
compute_hysteresis_events(
    audio, sample_rate, perch_model, binary_classifier, config
) -> tuple[list[WindowScore], list[HysteresisEvent]]
```

Both consumers call the helper, then diverge:

- **Existing `run_detection`** takes the raw hysteresis events, runs snap-merge + configured window selection (NMS/prominence/tiling), writes detection rows.
- **New Pass 1 `RegionDetectionWorker`** (stub in Phase 0, implemented in Pass 1 spec) takes the same raw hysteresis events, pads and merges them into regions, writes `trace.parquet` and `regions.parquet`.

The refactor is behavior-preserving for the existing detector. Phase 0's test suite verifies: (a) the refactored `run_detection` produces bit-identical detection rows on a fixture audio versus the pre-refactor baseline, (b) the new helper returns events matching the existing internal shape.

This is the **only** "touch existing code" work in Phase 0. Everything else is additive.

## Storage layout

```
storage_root/
├── call_parsing/                           # NEW
│   ├── regions/<job_id>/
│   │   ├── trace.parquet
│   │   ├── regions.parquet
│   │   └── job_meta.json
│   ├── segmentation/<job_id>/
│   │   └── events.parquet
│   └── classification/<job_id>/
│       └── typed_events.parquet
├── segmentation_models/<model_id>/         # NEW — Pass 2 PyTorch checkpoints
│   ├── checkpoint.pt
│   └── config.json
└── vocalization_models/<model_id>/         # existing — Pass 3 checkpoints for pytorch_event_cnn
    ├── checkpoint.pt                       # new file for pytorch family
    ├── config.json                         # new file for pytorch family
    └── (existing sklearn artifacts unchanged for sklearn family)
```

## Testing strategy

- **Unit:** parquet I/O roundtrip for all three artifact types, `ml/training_loop` training a tiny toy model on synthetic data, `ml/checkpointing` save/load roundtrip, worker claim semantics (compare-and-set against `status='queued'`), detector refactor bit-identical output test.
- **Migration:** fresh-DB upgrade + snapshot-DB upgrade (existing `vocalization_models` and `vocalization_training_jobs` rows default cleanly to new columns).
- **Integration:** parent-run CRUD lifecycle — create run, assert Pass 1 job created with `parent_run_id` set, query run status, cascade-delete run, assert child gone.
- **Smoke placeholder:** the full 4-pass end-to-end integration test is scaffolded (fixture audio + mock models structure) but not yet runnable — each Pass spec fills in its own assertions as it lands.

Existing smoke tests and detection tests must continue to pass — the `detector.py` refactor is validated against them.

## Non-goals for Phase 0

Deferred to per-pass specs:

- **Pass 1:** exact padding / smoothing / merging parameters, region overlap-merge semantics, UI for inspecting regions
- **Pass 2:** CRNN vs transformer architecture, loss formulation, onset/offset label format, training-dataset assembly from existing vocalization labels, `segmentation_training_jobs` table fields
- **Pass 3:** BirdNET-like CNN architecture, event crop padding/cropping strategy, inference batching, how the `vocalization_training_jobs` existing rows coexist with new `pytorch_event_cnn` rows during training
- **Pass 4:** output format beyond sorted JSON, query parameters for type filtering, pagination
- **Frontend:** all UI work is deferred until Pass 3 is end-to-end working

## Future phases

After Phase 0 ships, the remaining four passes each get their own feature branch, spec, plan, and implementation cycle:

| Phase    | Feature branch                                 | Scope |
|----------|------------------------------------------------|-------|
| Phase 1  | `feature/call-parsing-pass1-region-detector`   | `RegionDetectionWorker` logic, region padding/merging, `trace.parquet` / `regions.parquet` writers, functional Pass 1 endpoints |
| Phase 2  | `feature/call-parsing-pass2-segmentation`      | PyTorch CRNN/transformer, `segmentation_training_jobs` table (new migration), training + inference workers, `events.parquet` writer, functional Pass 2 endpoints |
| Phase 3  | `feature/call-parsing-pass3-event-classifier`  | PyTorch event-CNN, `vocalization_training_jobs` extensions, training + inference workers, `typed_events.parquet` writer, functional Pass 3 endpoints |
| Phase 4  | `feature/call-parsing-pass4-sequence-export`   | `/call-parsing/runs/{id}/sequence` endpoint implementation, optional format/filter query params |

Each subsequent phase inherits the Phase 0 contract — tables, parquet schemas, repo layout, worker placement, API shape — and focuses purely on its own logic.
