# Classifier Training Provenance Implementation Plan

**Goal:** Fill in empty training data and metrics for detection-manifest and promoted classifier models via backend normalization.
**Spec:** [docs/specs/2026-04-19-classifier-training-provenance-design.md](../specs/2026-04-19-classifier-training-provenance-design.md)

---

### Task 1: Add DetectionSourceInfo schema and extend TrainingDataSummaryResponse

**Files:**
- Modify: `src/humpback/schemas/classifier.py`
- Modify: `frontend/src/api/types.ts`

**Acceptance criteria:**
- [ ] `DetectionSourceInfo` Pydantic model with fields: detection_job_id, hydrophone_name (optional), start_timestamp (optional float), end_timestamp (optional float), positive_count (optional int), negative_count (optional int)
- [ ] `TrainingDataSummaryResponse` gains optional `detection_sources: list[DetectionSourceInfo] | None` field defaulting to None
- [ ] Frontend `TrainingDataSummaryResponse` type updated with matching `detection_sources` field

**Tests needed:**
- Schema serialization round-trip for DetectionSourceInfo
- TrainingDataSummaryResponse with and without detection_sources

---

### Task 2: Add detection_manifest branch to get_training_data_summary

**Files:**
- Modify: `src/humpback/services/classifier_service/training.py`

**Acceptance criteria:**
- [ ] New `elif tj.source_mode == "detection_manifest"` branch before the embedding-set fallthrough
- [ ] Reads `detection_job_ids` from model's training_summary JSON
- [ ] Loads DetectionJob records from DB for hydrophone_name, start_timestamp, end_timestamp
- [ ] Reads per-job counts from `training_data_source.per_job_counts` when available, falls back to totals-only when not
- [ ] Returns response with `detection_sources` populated and `positive_sources`/`negative_sources` as empty lists
- [ ] Total positive/negative counts come from training_summary's n_positive/n_negative

**Tests needed:**
- Unit test with detection_manifest model that has per_job_counts in training_summary
- Unit test with detection_manifest model missing per_job_counts (fallback path)
- Verify detection_sources contains correct hydrophone info from DetectionJob records

---

### Task 3: Normalize promoted model metrics in classifier training worker

**Files:**
- Modify: `src/humpback/workers/classifier_worker/training.py`

**Acceptance criteria:**
- [ ] After building the replay summary for autoresearch_candidate mode, merge standard fields: n_positive, n_negative, balance_ratio, cv_accuracy, cv_precision, cv_recall, cv_f1, classifier_type, class_weight_strategy, effective_class_weights, train_confusion
- [ ] Metrics computed from best_run_metrics (tp/fp/fn/tn/precision/recall) and training_data_source (positive_count/negative_count)
- [ ] classifier_type and class weights derived from trainer_parameters in promotion_provenance
- [ ] Fields that aren't available (cv_roc_auc, score_separation, _std variants, n_cv_folds) are omitted, not set to None

**Tests needed:**
- Unit test that builds a mock autoresearch summary and verifies standard fields are present and correct
- Verify cv_f1 computed correctly from precision and recall
- Verify train_confusion maps tp/fp/fn/tn correctly

---

### Task 4: Add per-job label breakdown to detection_manifest training path

**Files:**
- Modify: `src/humpback/workers/classifier_worker/training.py`
- Modify: `src/humpback/classifier/trainer.py`

**Acceptance criteria:**
- [ ] `load_manifest_split_embeddings` returns per-job breakdown in source_summary: each entry has detection_job_id, positive_count, negative_count
- [ ] Per-job counts derived by grouping manifest examples by their parquet_path → detection job mapping
- [ ] Worker stores the per_job_counts in training_data_source within training_summary
- [ ] Existing return signature preserved (no breaking changes)

**Tests needed:**
- Unit test for load_manifest_split_embeddings with a mock manifest containing multiple detection jobs
- Verify per_job_counts sums match total positive/negative counts

---

### Task 5: Frontend — detection_manifest training data rendering

**Files:**
- Modify: `frontend/src/components/classifier/TrainingTab.tsx`

**Acceptance criteria:**
- [ ] New rendering branch when `model.training_source_mode === "detection_manifest"` in the Training Data section
- [ ] Two-column layout: left shows Positive/Negative totals with vector counts and balance ratio; right shows "Detection Jobs" header with each job as "Hydrophone — YYYY-MM-DD to YYYY-MM-DD UTC"
- [ ] Timestamps formatted from epoch seconds to date strings using UTC
- [ ] Falls back gracefully when detection_sources is empty or null (shows totals only)

**Tests needed:**
- Visual verification in browser with detection-manifest model expanded

---

### Task 6: Frontend — training source label for detection_manifest

**Files:**
- Modify: `frontend/src/components/classifier/TrainingTab.tsx`

**Acceptance criteria:**
- [ ] Training Parameters section shows "Detection Jobs" as Training Source when `training_source_mode === "detection_manifest"`
- [ ] Existing "Embedding Sets" and "Candidate: ..." labels unchanged

**Tests needed:**
- Visual verification in browser

---

### Task 7: Backfill script for existing models

**Files:**
- Create: `scripts/backfill_training_summary.py`

**Acceptance criteria:**
- [ ] Finds all classifier_models with training_source_mode = 'autoresearch_candidate', loads promotion_provenance, computes and merges standard metric fields into training_summary
- [ ] Finds all classifier_models with training_source_mode = 'detection_manifest', reads manifest file from training_summary.manifest_path, computes per-job breakdowns, patches training_data_source.per_job_counts
- [ ] Uses the production DB from HUMPBACK_DATABASE_URL / .env
- [ ] Dry-run mode by default (prints what would change), --apply flag to actually write
- [ ] Prints a summary of models patched and fields added

**Tests needed:**
- Run with --apply against production DB after verifying dry-run output
- Verify patched models display correctly in the UI

---

### Verification

Run in order after all tasks:
1. `uv run ruff format --check src/humpback/schemas/classifier.py src/humpback/services/classifier_service/training.py src/humpback/workers/classifier_worker/training.py src/humpback/classifier/trainer.py scripts/backfill_training_summary.py`
2. `uv run ruff check src/humpback/schemas/classifier.py src/humpback/services/classifier_service/training.py src/humpback/workers/classifier_worker/training.py src/humpback/classifier/trainer.py scripts/backfill_training_summary.py`
3. `uv run pyright src/humpback/schemas/classifier.py src/humpback/services/classifier_service/training.py src/humpback/workers/classifier_worker/training.py src/humpback/classifier/trainer.py scripts/backfill_training_summary.py`
4. `uv run pytest tests/`
5. `cd frontend && npx tsc --noEmit`
