# Ingest Detection Tests

Use these commands for targeted feedback. The final backend gate remains
`uv run pytest tests/`.

## Smoke

- `uv run pytest tests/unit/test_detector.py tests/unit/test_detection_rows.py tests/unit/test_detection_embedding_service.py -q`

## Backend Domain

- `uv run pytest tests/unit/test_detector*.py tests/unit/test_classifier*.py tests/unit/test_trainer.py tests/unit/test_detection_* tests/integration/test_classifier_api.py tests/integration/test_detection_embedding_api.py tests/integration/test_detection_embedding_jobs_router.py tests/integration/test_hydrophone_api.py -q`

## Hydrophone And E2E Frontend

- `cd frontend && npx playwright test e2e/detection-hysteresis.spec.ts e2e/detection-incremental.spec.ts e2e/detection-labels.spec.ts e2e/detection-playback.spec.ts e2e/detection-spectrogram.spec.ts`
- `cd frontend && npx playwright test e2e/hydrophone-active-queue.spec.ts e2e/hydrophone-canceled-job.spec.ts e2e/hydrophone-extract.spec.ts e2e/hydrophone-pause-resume.spec.ts e2e/hydrophone-progress-format.spec.ts e2e/hydrophone-utc-timezone.spec.ts`
- `cd frontend && npx playwright test e2e/classifier-training.spec.ts`

## Expansion

- Hyperparameter changes: include `tests/integration/test_hyperparameter_api.py`
  and `tests/unit/test_hyperparameter_*`.
- Autoresearch changes: include `tests/integration/test_autoresearch.py` and
  `tests/unit/test_autoresearch.py`.
