# Call Parsing Tests

Use these commands for targeted feedback. The final backend gate remains
`uv run pytest tests/`.

## Smoke

- `uv run pytest tests/call_parsing tests/unit/test_call_parsing_types.py tests/unit/test_call_parsing_storage.py -q`

## Backend Domain

- `uv run pytest tests/call_parsing tests/unit/test_call_parsing_* tests/unit/test_segmentation_* tests/unit/test_event_classifier_* tests/unit/test_window_classification_worker.py tests/integration/test_call_parsing_router.py -q`

## Worker Expansion

- Region detection: add `tests/integration/test_region_detection_worker.py`
- Event segmentation: add `tests/integration/test_event_segmentation_worker.py`
- Event classifier smoke: add `tests/integration/test_event_classifier_smoke.py`

## Frontend Domain

- `cd frontend && npx playwright test e2e/call-parsing-detection.spec.ts e2e/call-parsing-segment.spec.ts e2e/call-parsing-classify-review.spec.ts`
- `cd frontend && npx tsc --noEmit`
