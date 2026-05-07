# Domain Test Map

Use this map for targeted feedback during implementation. These commands do not
replace the final backend gate: `uv run pytest tests/`.

## Core Platform

- Smoke: `uv run pytest tests/unit/test_config.py tests/unit/test_storage.py tests/unit/test_queue.py -q`
- Database/migration: `uv run pytest tests/unit/test_migration_* tests/db -q`
- Worker queue/runner: `uv run pytest tests/unit/test_worker_runner.py tests/unit/test_queue.py -q`

## Signal Timeline

- Smoke: `uv run pytest tests/unit/test_timeline_tiles.py tests/unit/test_timeline_audio.py tests/unit/test_pcen_rendering.py -q`
- Domain: `uv run pytest tests/processing tests/unit/test_timeline_* tests/unit/test_spectrogram.py tests/unit/test_pcen_rendering.py tests/integration/test_timeline_api.py -q`
- Frontend: `cd frontend && npx playwright test e2e/timeline.spec.ts e2e/timeline-labeling.spec.ts`

## Ingest Detection

- Smoke: `uv run pytest tests/unit/test_detector.py tests/unit/test_detection_rows.py tests/unit/test_detection_embedding_service.py -q`
- Domain: `uv run pytest tests/unit/test_detector*.py tests/unit/test_classifier*.py tests/unit/test_trainer.py tests/integration/test_classifier_api.py tests/integration/test_detection_embedding_api.py tests/integration/test_hydrophone_api.py -q`
- Frontend: `cd frontend && npx playwright test e2e/detection-*.spec.ts e2e/hydrophone-*.spec.ts e2e/classifier-training.spec.ts`

## Vocalization Clustering

- Smoke: `uv run pytest tests/unit/test_vocalization_service.py tests/unit/test_vocalization_trainer.py tests/unit/test_clustering_metrics.py -q`
- Domain: `uv run pytest tests/unit/test_vocalization_* tests/unit/test_training_dataset.py tests/unit/test_clustering_* tests/integration/test_vocalization_api.py tests/integration/test_training_dataset_api.py -q`
- Frontend: `cd frontend && npx playwright test e2e/vocalization-labeling.spec.ts`

## Call Parsing

- Smoke: `uv run pytest tests/call_parsing tests/unit/test_call_parsing_types.py tests/unit/test_call_parsing_storage.py -q`
- Domain: `uv run pytest tests/call_parsing tests/unit/test_call_parsing_* tests/unit/test_segmentation_* tests/unit/test_event_classifier_* tests/unit/test_window_classification_worker.py tests/integration/test_call_parsing_router.py -q`
- Frontend: `cd frontend && npx playwright test e2e/call-parsing-detection.spec.ts e2e/call-parsing-segment.spec.ts e2e/call-parsing-classify-review.spec.ts`

## Sequence Models

- Smoke: `uv run pytest tests/sequence_models tests/unit/test_sequence_models_schemas.py tests/services/test_continuous_embedding_service.py -q`
- Domain: `uv run pytest tests/sequence_models tests/unit/test_sequence_models_schemas.py tests/services/test_continuous_embedding_service.py tests/workers/test_continuous_embedding_worker.py tests/integration/test_sequence_models_api.py -q`
- Frontend: `cd frontend && npx playwright test e2e/sequence-models/continuous-embedding.spec.ts`

## Frontend Shell

- Typecheck: `cd frontend && npx tsc --noEmit`
- Navigation smoke: `cd frontend && npx playwright test e2e/navigation-retired-workflows.spec.ts e2e/compute-device-badge.spec.ts`
- Shared timeline components often also need the Signal Timeline frontend command.

## Expansion Rules

- API route changes: include the matching `tests/integration/test_*_api.py` or
  router test.
- Worker changes: include the worker test plus `tests/unit/test_queue.py` when
  claim semantics change.
- Storage helper changes: include `tests/unit/test_storage.py` and the owning
  domain's artifact tests.
- Frontend query-hook or API-client changes: run `cd frontend && npx tsc --noEmit`
  plus the owning domain's Playwright smoke.
- Cross-domain behavior: run the union of affected domain smoke commands before
  the full suite.
