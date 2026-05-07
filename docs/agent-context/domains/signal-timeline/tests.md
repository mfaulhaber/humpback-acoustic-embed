# Signal Timeline Tests

Use these commands for targeted feedback. The final backend gate remains
`uv run pytest tests/`.

## Smoke

- `uv run pytest tests/unit/test_timeline_tiles.py tests/unit/test_timeline_audio.py tests/unit/test_pcen_rendering.py -q`

## Backend Domain

- `uv run pytest tests/processing tests/unit/test_timeline_* tests/unit/test_spectrogram.py tests/unit/test_pcen_rendering.py tests/integration/test_timeline_api.py -q`

## Frontend Domain

- `cd frontend && npx playwright test e2e/timeline.spec.ts e2e/timeline-labeling.spec.ts`
- `cd frontend && npx tsc --noEmit`

## Expansion

- Overlay changes in a workspace: also run that workspace domain's frontend
  smoke tests.
- Audio source changes: also run the owning source domain tests.
