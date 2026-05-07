# Vocalization Clustering Tests

Use these commands for targeted feedback. The final backend gate remains
`uv run pytest tests/`.

## Smoke

- `uv run pytest tests/unit/test_vocalization_service.py tests/unit/test_vocalization_trainer.py tests/unit/test_clustering_metrics.py -q`

## Backend Domain

- `uv run pytest tests/unit/test_vocalization_* tests/unit/test_training_dataset.py tests/unit/test_clustering_* tests/integration/test_vocalization_api.py tests/integration/test_training_dataset_api.py tests/integration/test_labeling_api.py -q`

## Frontend Domain

- `cd frontend && npx playwright test e2e/vocalization-labeling.spec.ts`
- `cd frontend && npx tsc --noEmit`

## Expansion

- Detection-label consistency changes: also run the Ingest Detection smoke.
- Call Parsing correction changes: also run the Call Parsing smoke.
