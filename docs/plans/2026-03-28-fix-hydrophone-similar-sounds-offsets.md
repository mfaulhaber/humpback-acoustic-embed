# Fix Hydrophone Similar Sounds Offsets

## Goal

Fix hydrophone-origin Similar Sounds searches so Search and labeling flows work when the UI sends job-relative detection offsets returned by `GET /classifier/detection-jobs/{id}/content`.

## Root Cause

Hydrophone detection content is normalized for the UI with job-relative `start_sec` and `end_sec`, but several downstream backend paths still consume those values as if they were file-relative offsets within the chunk filename. That breaks stored embedding lookup and archive audio re-embedding for hydrophone rows launched from the UI.

## Scope

- Add a shared helper for converting hydrophone job-relative offsets back to file-relative offsets.
- Apply that conversion in detection embedding lookup and labeling Similar Sounds lookup.
- Apply that conversion in the search worker before resolving hydrophone archive audio.
- Add regression tests for the affected hydrophone flows.

## Tasks

### Task 1: Add shared offset conversion helper

Files:
- `src/humpback/classifier/detection_rows.py`
- `src/humpback/api/routers/classifier.py`

Acceptance criteria:
- Shared helper converts a hydrophone UI offset back to the stored file-relative offset using the chunk timestamp encoded in `filename`.
- Existing label-save behavior continues to use the shared helper rather than a router-local copy.

Test requirements:
- Covered indirectly by integration/unit tests added in later tasks.

### Task 2: Fix stored embedding lookups for hydrophone rows

Files:
- `src/humpback/api/routers/classifier.py`
- `src/humpback/api/routers/labeling.py`
- `tests/integration/test_classifier_api.py`
- `tests/integration/test_labeling_api.py`

Acceptance criteria:
- Hydrophone detection embedding lookup accepts the job-relative offsets returned by `GET /content`.
- Hydrophone labeling neighbor lookup accepts the same UI offsets and returns results when a matching stored embedding exists.
- Non-hydrophone behavior remains unchanged.

Test requirements:
- Add integration coverage for hydrophone embedding lookup using a timestamped chunk filename and job-relative offsets.
- Add integration coverage for hydrophone detection-neighbor lookup using the same offset translation.

### Task 3: Fix hydrophone Search page audio re-embedding

Files:
- `src/humpback/workers/search_worker.py`
- `tests/unit/test_search_worker.py`

Acceptance criteria:
- Hydrophone audio search converts UI job-relative offsets back to file-relative offsets before archive audio resolution.
- Local audio search behavior remains unchanged.

Test requirements:
- Add a unit test that verifies the hydrophone search worker passes file-relative offsets into archive slice resolution when given job-relative offsets from the UI.

## Verification

- `uv run pytest tests/integration/test_classifier_api.py tests/integration/test_labeling_api.py tests/unit/test_search_worker.py`
- `uv run pytest tests/`

## Ready For Review

- Ready for session-end: no
