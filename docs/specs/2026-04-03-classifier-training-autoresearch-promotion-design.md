# Classifier Training Autoresearch Promotion — Design Spec

**Date:** 2026-04-03
**Status:** Approved

## Goal

Extend the Classifier Training frontend and API so a user can take a reviewed autoresearch comparison against a production model such as `LR-v12`, inspect the evidence inside the app, and launch a new training job that faithfully reproduces the promoted autoresearch candidate.

## Problem Context

The current Classifier Training page supports two workflows:

1. Manual training from selected positive and negative embedding sets
2. Folder-root retrain from an existing classifier model's original import roots

Those flows are not enough to operationalize the findings from the latest autoresearch work:

- The winning candidates now come from manifest-backed experiments, not only embedding-set IDs.
- The experiments can include explicit negatives mined from detection jobs and vocalization `"(Negative)"` labels.
- The strongest evidence for promotion is the comparison artifact produced by `scripts/autoresearch/compare_classifiers.py`, not just the raw `best_run.json`.
- The current production trainer only understands a subset of the autoresearch config space, so blindly copying parameters would silently change the trained model.

The last production-backed explicit-negative run demonstrated this gap clearly:

- the phase-1 autoresearch winner materially outperformed production `LR-v12` on the explicit-negative test split
- the evidence lives in JSON artifacts under `/tmp/humpback-autoresearch-2026-04-03-dj-2a5f-23a1-explicit-negatives/`
- none of that comparison context is visible or actionable from the existing Training UI

We need a first-class “promotion candidate” flow that turns those artifacts into reviewable application state and a reproducible training job.

## Non-Goals

- Running full autoresearch from the web UI
- Replacing the existing embedding-set training form
- Replacing the existing folder-root retrain workflow
- Building deployment/inference rollout controls in this feature
- Automatically approving or publishing promoted models without user review

## Current Constraints

### Existing Training API shape is embedding-set centric

`POST /classifier/training-jobs` currently accepts only:

- `name`
- `positive_embedding_set_ids`
- `negative_embedding_set_ids`
- `parameters`

This contract cannot describe manifest-backed training data from detection jobs.

### Existing retrain flow rebuilds from folder roots, not from reviewed artifacts

`GET /classifier/models/{model_id}/retrain-info` and `POST /classifier/retrain` trace the original embedding-set folder roots from a source model. That flow does not preserve:

- manifest split membership
- detection-job explicit negatives
- comparison metrics against production
- replay metadata from phase-2 experiments

### Production trainer does not yet match the full autoresearch config space

The current binary trainer supports:

- classifier type: logistic regression or MLP
- `l2_normalize`
- `class_weight`
- logistic `C`

Autoresearch candidates can also depend on:

- PCA
- probability calibration
- asymmetric class weights
- context pooling
- manifest-backed detection examples
- hard-negative replay provenance

Promotion must therefore be explicit about which candidates are faithfully reproducible today and which require trainer enhancements first.

## Approaches Considered

### Option A: Prefill existing retrain form from comparison JSON

Use the comparison artifact only as a recommendation layer and push users back into the existing folder-root retrain flow.

Pros:

- smallest API surface change
- reuses current Training tab layout

Cons:

- loses manifest-backed examples and explicit detection negatives
- silently deviates from the reviewed candidate
- cannot preserve split metrics or replay provenance

### Option B: Import autoresearch artifacts as a first-class promotion candidate

Create a durable server-side object representing one reviewed autoresearch candidate. Surface it in the Training tab and allow promotion into a manifest-backed training job.

Pros:

- faithful handoff from experiment to training
- reviewable evidence in the UI
- supports detection-job negatives and comparison artifacts directly
- creates stable provenance for future audits

Cons:

- requires new API and model shapes
- requires a new training job mode

### Option C: Run compare-and-promote fully client-side from arbitrary JSON uploads

Let the frontend parse local JSON and submit the resulting training request without durable server-side candidate storage.

Pros:

- faster initial UX

Cons:

- brittle provenance
- hard to share between users
- harder to test and audit

## Decision

Choose Option B.

Introduce a first-class imported `AutoresearchCandidate` record in the backend, expose it through the Classifier Training API, and render it in the Training tab as a new promotion-oriented workflow adjacent to the existing model list and retrain panel.

## User Experience

### 1. Import a candidate

From the Training page, the user opens a new `Autoresearch Candidates` panel and imports a results bundle from server-managed JSON artifacts.

The imported bundle includes:

- `manifest.json`
- `best_run.json`
- optional `top_false_positives.json`
- optional comparison JSON such as `lr-v12-comparison.json`

The server validates and stores a summarized candidate record.

### 2. Review the evidence

The candidate row shows a concise comparison summary:

- candidate name
- source production model
- objective delta
- recall delta
- false-positive delta
- high-confidence false-positive delta
- train/val/test counts
- phase and replay summary
- reproducibility status

Expanding the row reveals:

- promoted config
- comparison metrics by split
- top false positives for both models
- prediction disagreements
- manifest source composition
- warnings when the candidate depends on unsupported training/runtime behavior

### 3. Promote to training

If the candidate is reproducible, the user clicks `Create Model From Candidate`, provides a new model name, and starts a new training job.

The resulting training job:

- references the candidate record
- reads its training data from the imported manifest's `train` split
- stores promotion provenance on both the training job and the resulting model

### 4. Review the resulting model

When training completes, the model detail row shows:

- source candidate name and id
- source production model used for comparison
- promoted config snapshot
- link or path to the imported comparison artifact
- any deployment caveats

## Backend Design

### New domain object: `AutoresearchCandidate`

Add a new persisted record with fields covering:

- identity:
  - `id`
  - `name`
  - `status`
  - `created_at`
  - `updated_at`
- artifact paths:
  - `manifest_path`
  - `best_run_path`
  - `top_false_positives_path`
  - `comparison_path`
- comparison provenance:
  - `source_model_id`
  - `source_model_name`
  - `comparison_target`
- promoted candidate summary:
  - `phase`
  - `config`
  - `objective_name`
  - `threshold`
  - `replay_summary`
  - `split_metrics`
  - `metric_deltas`
  - `source_counts`
  - `warnings`
- promotion linkage:
  - `training_job_id`
  - `new_model_id`

`status` should support at least:

- `imported`
- `promotable`
- `blocked`
- `training`
- `complete`
- `failed`

### New API endpoints

Add:

- `POST /classifier/autoresearch-candidates/import`
  - input: explicit server-side artifact paths, optional display name, optional source model override
  - behavior: validate files, parse summaries, persist candidate
- `GET /classifier/autoresearch-candidates`
- `GET /classifier/autoresearch-candidates/{candidate_id}`
- `POST /classifier/autoresearch-candidates/{candidate_id}/training-jobs`
  - input: `new_model_name`, optional notes
  - behavior: create a manifest-backed classifier training job

Keep existing endpoints unchanged:

- `/classifier/training-jobs`
- `/classifier/models/{id}/retrain-info`
- `/classifier/retrain`

This feature adds a new promotion flow rather than overloading the legacy retrain API.

### Training job source modes

Extend classifier training jobs with a source mode:

- `embedding_sets`
- `autoresearch_candidate`

For `autoresearch_candidate` jobs, persist:

- `source_candidate_id`
- `source_model_id`
- `manifest_path`
- `promoted_config`
- `training_split_name` (always `train` initially)
- `comparison_summary`

### Trainer parity requirement

The promotion API must not accept a candidate unless the server can reproduce its training behavior without silent downgrades.

Initial reproducibility rules:

- allow only candidates whose config can be mapped exactly to production training
- block candidates requiring unsupported features

Today, likely blockers include:

- PCA dimensions
- calibration
- context pooling other than deployment-supported behavior
- asymmetric class weights if the trainer cannot express them exactly
- any manifest-only feature not yet handled by the classifier worker

Blocked candidates still import and render, but the UI must show `Not yet promotable`.

## Frontend Design

### Training page layout

Add a new section to [TrainingTab.tsx](/Users/michael/development/humpback-acoustic-embed/frontend/src/components/classifier/TrainingTab.tsx):

- `Autoresearch Candidates`

This section sits near the existing model list because it is a review-and-promote workflow, not raw dataset selection.

### Candidate list row

Each row should show:

- candidate display name
- source model badge, e.g. `Compared to LR-v12`
- promoted phase badge, e.g. `Phase 1`
- status badge: `Promotable`, `Blocked`, `Training`, `Complete`
- core deltas on the default comparison split

### Candidate detail panel

The expanded state should include:

- promoted config
- validation/test metric tables
- delta table versus production
- replay summary
- top false positives
- prediction disagreement preview
- manifest source counts by label source and negative group
- artifact path list for debugging
- promotion caveats

### Promotion modal

The modal needs only:

- new model name
- optional notes
- confirmation that training will use the imported manifest's `train` split exactly

It should not expose raw parameter editing in v1. The point is faithful promotion, not ad hoc tweaking.

### Frontend data additions

Add new client/types/hooks for:

- candidate import
- candidate listing/detail
- candidate promotion

The existing model row can link back to its source candidate once promotion is complete.

## Artifact Fixture Strategy

Vendor a stable fixture bundle into the repo for UI development under:

- [scripts/autoresearch/output](/Users/michael/development/humpback-acoustic-embed/scripts/autoresearch/output)

Initial fixture set:

- `explicit-negatives/manifest.json`
- `explicit-negatives/comparison_summary.json`
- `explicit-negatives/phase1/best_run.json`
- `explicit-negatives/phase1/search_history.json`
- `explicit-negatives/phase1/top_false_positives.json`
- `explicit-negatives/phase1/lr-v12-comparison.json`
- `explicit-negatives/phase2/best_run.json`
- `explicit-negatives/phase2/search_history.json`
- `explicit-negatives/phase2/top_false_positives.json`

Why vendor these files:

- the UI can render real candidate cards before the import API is finished
- API tests can use real comparison payloads without hitting `/tmp`
- the explicit-negative fixture captures the production-backed `LR-v12` delta that motivated this feature

## Risks and Mitigations

### Risk: silent training mismatch

Mitigation:

- promotion requires explicit reproducibility checks
- blocked candidates remain viewable but not trainable
- persisted warnings are shown in both API and UI

### Risk: artifact paths outside managed storage

Mitigation:

- import copies or rehomes artifacts into managed app storage
- fixture files in `scripts/autoresearch/output` are development/test fixtures only

### Risk: manifest-backed training path diverges from existing worker assumptions

Mitigation:

- create an explicit source mode instead of trying to fake embedding-set IDs
- add worker tests for manifest-backed jobs

## Testing Strategy

### Backend tests

- candidate import validates required artifacts and rejects malformed bundles
- comparison JSON is parsed into stable summary fields
- blocked/promotable status is derived correctly from candidate config
- promotion creates manifest-backed training jobs with correct provenance
- resulting models retain source candidate linkage

### Frontend tests

- Training page renders fixture-backed candidates
- blocked candidates show warnings and disable promotion
- promotable candidates open the promotion modal and submit the correct payload
- promoted models show candidate provenance after completion

### End-to-end tests

- import fixture bundle
- review comparison against `LR-v12`
- create a candidate-backed training job
- observe completed model with linked provenance

