# Sequence Models Tests

Use these commands for targeted feedback. The final backend gate remains
`uv run pytest tests/`.

## Smoke

- `uv run pytest tests/sequence_models tests/unit/test_sequence_models_schemas.py tests/services/test_continuous_embedding_service.py tests/services/test_event_encoder_service.py -q`

## Backend Domain

- `uv run pytest tests/sequence_models tests/unit/test_sequence_models_schemas.py tests/unit/test_storage.py tests/services/test_continuous_embedding_service.py tests/services/test_event_encoder_service.py tests/workers/test_continuous_embedding_worker.py tests/workers/test_event_encoder_worker.py tests/integration/test_sequence_models_api.py -q`

## Event Encoder Ridge And Timeline Features

- `uv run pytest tests/sequence_models/test_event_encoder.py tests/unit/test_sequence_models_schemas.py -q`
- `uv run pytest tests/services/test_event_encoder_service.py tests/workers/test_event_encoder_worker.py -q`
- `uv run pytest tests/integration/test_sequence_models_api.py -q`
- `cd frontend && npx vitest run src/components/sequence-models/EventEncoderTokenOverlay.test.tsx`
- `cd frontend && npx tsc --noEmit`
- `cd frontend && npx playwright test e2e/sequence-models/event-encoder.spec.ts`

## Frontend Domain

- `cd frontend && npx playwright test e2e/sequence-models/continuous-embedding.spec.ts`
- `cd frontend && npx playwright test e2e/sequence-models/event-encoder.spec.ts`
- `cd frontend && npx vitest run src/components/sequence-models/EventEncoderTokenOverlay.test.tsx`
- `cd frontend && npx tsc --noEmit`

## Expansion

- Upstream validation changes: also run the Call Parsing smoke.
- Audio chunk/window changes: also run the Signal Timeline smoke.
- Storage helper changes: also run `tests/unit/test_storage.py`.
