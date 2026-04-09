# Tuning Page Improvements — Implementation Plan

**Spec**: `docs/specs/2026-04-09-tuning-page-improvements-design.md`
**Branch**: `feature/tuning-page-improvements`

---

## Task 1: Add positive/negative counts to ManifestSummary API response

**Files**: `src/humpback/schemas/hyperparameter.py`, `src/humpback/api/routers/classifier/hyperparameter.py`

1. Add `positive_count: Optional[int] = None` and `negative_count: Optional[int] = None` to `ManifestSummary` schema
2. Update `_manifest_to_summary()` helper to compute totals from `split_summary` JSON:
   - Parse `m.split_summary` (JSON string with `{train: {positive, negative}, val: {...}, test: {...}}`)
   - Sum `positive` across all splits -> `positive_count`
   - Sum `negative` across all splits -> `negative_count`
   - Leave as `None` if `split_summary` is null (incomplete manifests)

**Acceptance**: `GET /classifier/hyperparameter/manifests` returns `positive_count` and `negative_count` fields.

---

## Task 2: Show Positives/Negatives columns in manifest table (frontend)

**Files**: `frontend/src/api/types.ts`, `frontend/src/components/classifier/TuningTab.tsx`

1. Add `positive_count: number | null` and `negative_count: number | null` to `HyperparameterManifestSummary` type
2. Add two `<th>` columns ("Positives", "Negatives") after "Examples" in manifest table header
3. Add two `<td>` cells in each manifest row: `{m.positive_count ?? "—"}` and `{m.negative_count ?? "—"}`

**Acceptance**: Manifest table shows Positives and Negatives columns with integer counts or "—".

---

## Task 3: Show detection job list in manifest detail

**Files**: `frontend/src/components/classifier/TuningTab.tsx`

1. Update `ManifestDetail` component to resolve detection job labels:
   - The parent `ManifestsSection` already fetches `detectionJobs` — pass it as a prop
   - For each `detection_job_id` in the manifest, look up the job and format using `fmtDetectionJobLabel()`
   - Display as a list of strings under the existing detail section
2. Replace the current generic `sourceSummary()` in detail with the formatted detection job list (keep source summary in the table row column)

**Acceptance**: Expanding a manifest row shows detection jobs formatted as "Orcasound Lab 2021-10-29 00:00 UTC — 2021-10-30 00:00 UTC".

---

## Task 4: Add candidate delete endpoint (backend)

**Files**: `src/humpback/api/routers/classifier/hyperparameter.py`

1. Add `DELETE /classifier/hyperparameter/candidates/{candidate_id}` endpoint:
   - Look up candidate; 404 if not found
   - Delete DB record (no disk artifacts to clean — candidates reference shared search/manifest files)
   - Return `{"status": "deleted"}`
2. Import `AutoresearchCandidate` model in the router

**Acceptance**: `DELETE /classifier/hyperparameter/candidates/{id}` deletes the record and returns 200.

---

## Task 5: Add candidate delete to frontend with promotion warning

**Files**: `frontend/src/api/client.ts`, `frontend/src/api/types.ts`, `frontend/src/hooks/queries/useClassifier.ts`, `frontend/src/components/classifier/AutoresearchCandidatesSection.tsx`

1. Add `deleteCandidate(id: string)` to API client
2. Add `useDeleteCandidate()` hook with query invalidation for `["autoresearchCandidates"]`
3. In `AutoresearchCandidatesSection`:
   - Add `Trash2` icon import
   - Add trash button to each candidate row
   - On click: check if `candidate.new_model_id` is non-null (= promoted)
   - If promoted: show confirmation dialog with warning "This candidate has been promoted to a classifier model. Delete anyway?"
   - If not promoted: show standard confirmation
   - On confirm: call delete mutation

**Acceptance**: Candidate rows have delete button; promoted candidates show warning before delete.

---

## Task 6: Tests

**Files**: `tests/api/test_hyperparameter.py` (or create if needed)

1. Test `ManifestSummary` includes `positive_count`/`negative_count` (null for queued, populated for complete)
2. Test `DELETE /classifier/hyperparameter/candidates/{id}` returns 200 and removes record
3. Test candidate delete returns 404 for nonexistent ID

**Acceptance**: All new tests pass.

---

## Verification

- [ ] `uv run ruff format --check` on modified Python files
- [ ] `uv run ruff check` on modified Python files
- [ ] `uv run pyright` on modified Python files
- [ ] `uv run pytest tests/`
- [ ] `cd frontend && npx tsc --noEmit`
