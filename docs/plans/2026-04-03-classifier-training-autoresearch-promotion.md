# Classifier Training Autoresearch Promotion — Implementation Plan

**Goal:** Let the Classifier Training UI import autoresearch comparison artifacts, review them against production models such as `LR-v12`, and launch reproducible candidate-backed training jobs.
**Spec:** [docs/specs/2026-04-03-classifier-training-autoresearch-promotion-design.md](../specs/2026-04-03-classifier-training-autoresearch-promotion-design.md)

---

### Task 1: Add a persisted autoresearch-candidate backend model and import API

**Files:**
- Modify: `src/humpback/models/classifier.py` or add a dedicated candidate model file if preferred
- Modify: `src/humpback/schemas/classifier.py`
- Modify: `src/humpback/services/classifier_service.py`
- Modify: `src/humpback/api/routers/classifier.py`
- Create: `alembic/versions/<next>_autoresearch_candidates.py`

**Acceptance criteria:**
- [x] The backend can persist an imported autoresearch candidate with artifact paths, source model metadata, promoted config, split metrics, replay summary, status, and warnings
- [x] A new import endpoint accepts server-side artifact paths and validates `manifest.json`, `best_run.json`, and optional comparison/top-false-positive files
- [x] Candidate detail and list endpoints return summarized comparison data suitable for the Training UI
- [x] Candidate status distinguishes at least promotable, blocked, training, complete, and failed states
- [x] The import flow records whether the candidate is exactly reproducible by the current production trainer

**Tests needed:**
- [x] API tests for import success, missing artifact failure, malformed JSON failure, and candidate listing/detail
- Migration test coverage if project conventions require it

---

### Task 2: Support manifest-backed classifier training jobs sourced from imported candidates

**Files:**
- Modify: `src/humpback/models/classifier.py`
- Modify: `src/humpback/schemas/classifier.py`
- Modify: `src/humpback/services/classifier_service.py`
- Modify: `src/humpback/workers/classifier_worker.py`
- Modify: `src/humpback/classifier/trainer.py`
- Modify: `src/humpback/api/routers/classifier.py`

**Acceptance criteria:**
- [x] Classifier training jobs can represent either embedding-set training or autoresearch-candidate training without overloading the old contract ambiguously
- [x] A new promotion endpoint can create a training job from an imported candidate and persist source candidate provenance
- [x] The classifier worker can load manifest-backed training examples from the candidate artifact instead of only embedding-set IDs
- [x] Promotion is blocked when the current trainer cannot faithfully reproduce the candidate config
- [x] Completed models expose candidate provenance and source comparison context through the existing model APIs

**Tests needed:**
- Unit tests for candidate-to-training-job creation and reproducibility checks
- Worker tests for manifest-backed candidate training jobs
- API tests proving blocked candidates cannot be promoted

---

### Task 3: Extend the Classifier Training frontend with candidate review and promotion UX

**Files:**
- Modify: `frontend/src/components/classifier/TrainingTab.tsx`
- Modify: `frontend/src/api/client.ts`
- Modify: `frontend/src/api/types.ts`
- Modify: `frontend/src/hooks/queries/useClassifier.ts`
- Create: any focused candidate UI subcomponents extracted from `TrainingTab.tsx` if needed

**Acceptance criteria:**
- [x] The Training page renders an `Autoresearch Candidates` section populated from the new API
- [x] Users can import a candidate from server-side artifact paths from the UI
- [x] Candidate rows show source model, key metric deltas, phase/replay summary, and promotable vs blocked status
- [x] Expanded candidate detail shows promoted config, comparison metrics, disagreement preview, and warnings
- [x] Users can start a candidate-backed training job with a new model name when the candidate is promotable
- [x] Existing embedding-set training and folder-root retrain UX remain intact

**Tests needed:**
- Frontend unit/component tests for candidate rendering and promotion-state handling if the project already uses them in this area
- Playwright coverage for import, review, and promote flow in the Training tab

---

### Task 4: Vendor stable autoresearch output fixtures for UI and API development

**Files:**
- Create: `scripts/autoresearch/output/README.md`
- Create: `scripts/autoresearch/output/explicit-negatives/manifest.json`
- Create: `scripts/autoresearch/output/explicit-negatives/comparison_summary.json`
- Create: `scripts/autoresearch/output/explicit-negatives/phase1/best_run.json`
- Create: `scripts/autoresearch/output/explicit-negatives/phase1/search_history.json`
- Create: `scripts/autoresearch/output/explicit-negatives/phase1/top_false_positives.json`
- Create: `scripts/autoresearch/output/explicit-negatives/phase1/lr-v12-comparison.json`
- Create: `scripts/autoresearch/output/explicit-negatives/phase2/best_run.json`
- Create: `scripts/autoresearch/output/explicit-negatives/phase2/search_history.json`
- Create: `scripts/autoresearch/output/explicit-negatives/phase2/top_false_positives.json`

**Acceptance criteria:**
- [x] The repo contains a stable explicit-negative fixture bundle copied from the latest production-backed run
- [x] The fixture README explains provenance, intended use, and how it relates to `LR-v12`
- [x] Frontend and API tests can read these fixtures without depending on `/tmp`

**Tests needed:**
- Add or update tests to read at least one vendored fixture file during candidate import/render coverage

---

### Task 5: Document the promotion workflow and reproducibility limits

**Files:**
- Modify: `scripts/autoresearch/README.md`
- Modify: `README.md`
- Modify: `CLAUDE.md` if workflow/reference expectations change materially

**Acceptance criteria:**
- [x] Docs explain how to import comparison artifacts and promote a reviewed candidate
- [x] Docs distinguish legacy retrain-from-folders from candidate-backed promotion
- [x] Docs clearly state which autoresearch config features are currently promotable and which block promotion
- [x] Docs point developers at the vendored fixture bundle for UI/API testing

**Tests needed:**
- Documentation task only; no new standalone tests required beyond coverage added in earlier tasks

---

### Verification

Run in order after all tasks:
1. `uv run ruff format --check src/humpback/api/routers/classifier.py src/humpback/services/classifier_service.py src/humpback/classifier/trainer.py src/humpback/workers/classifier_worker.py frontend/src/components/classifier/TrainingTab.tsx frontend/src/api/client.ts frontend/src/api/types.ts frontend/src/hooks/queries/useClassifier.ts tests/`
2. `uv run ruff check src/humpback/ tests/`
3. `uv run pyright src/humpback scripts/autoresearch tests/`
4. `uv run pytest tests/`
5. `cd frontend && npx tsc --noEmit`
6. `cd frontend && npx playwright test`
