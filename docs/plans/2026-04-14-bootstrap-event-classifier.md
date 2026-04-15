# Bootstrap Event Classifier Implementation Plan

**Goal:** Create a CLI script that trains a bootstrap event classifier from segmentation training data with random type labels, breaking the cold-start circular dependency in the feedback training loop.
**Spec:** [docs/specs/2026-04-14-bootstrap-event-classifier-design.md](../specs/2026-04-14-bootstrap-event-classifier-design.md)

---

### Task 1: Create bootstrap script

**Files:**
- Create: `scripts/bootstrap_classifier.py`

**Acceptance criteria:**
- [ ] Script accepts `dataset_id` as a required CLI argument (argparse)
- [ ] Connects to the database using `HUMPBACK_DATABASE_URL` from settings/env
- [ ] Queries `SegmentationTrainingSample` rows for the given dataset ID
- [ ] Validates the dataset exists and has samples; exits with clear error if not
- [ ] Queries all `VocalizationType` names for the vocabulary
- [ ] Validates at least one vocalization type exists; exits with clear error if not
- [ ] Parses each sample's `events_json` and flattens into individual events
- [ ] Offsets event times by `crop_start_sec` to convert from crop-relative to region-relative coordinates
- [ ] Assigns random type labels using round-robin first (guaranteeing `min_examples_per_type` per type) then uniform random for the remainder
- [ ] Builds sample objects with `.start_sec`, `.end_sec`, `.type_index`, `.hydrophone_id`, `.start_timestamp`, `.end_timestamp`
- [ ] Builds audio loader using `resolve_timeline_audio()` with context padding (same pattern as feedback worker)
- [ ] Calls `train_event_classifier()` with default `EventClassifierTrainingConfig` and full vocabulary
- [ ] Model directory is `storage_root/vocalization_models/<model_id>/`
- [ ] Registers `VocalizationClassifierModel` row with `model_family='pytorch_event_cnn'`, `input_mode='segmented_event'`, vocabulary snapshot, thresholds, and metrics
- [ ] Prints the registered model ID to stdout on success
- [ ] Handles errors with clear messages (dataset not found, no samples, no types, training failure)

**Tests needed:**
- Unit test for the random label assignment logic (round-robin + uniform distribution guarantees minimum coverage per type)
- Unit test for the event flattening and coordinate offset logic
- Unit test for the full script flow using mocked DB and trainer (verify sample construction, trainer invocation, model registration)

---

### Task 2: Add unit tests

**Files:**
- Create: `tests/unit/test_bootstrap_classifier.py`

**Acceptance criteria:**
- [ ] Tests the event flattening logic: given sample rows with `events_json` and `crop_start_sec`, verifies correct region-relative event coordinates
- [ ] Tests the random label assignment: verifies every type gets at least `min_examples_per_type` events when there are enough total events
- [ ] Tests the edge case where total events < `min_examples_per_type * n_types` (should still assign all events without error)
- [ ] Tests that the sample objects have the correct attributes for the trainer interface
- [ ] Tests error handling: missing dataset, empty dataset, no vocalization types

**Tests needed:**
- These ARE the tests

---

### Verification

Run in order after all tasks:
1. `uv run ruff format --check scripts/bootstrap_classifier.py tests/unit/test_bootstrap_classifier.py`
2. `uv run ruff check scripts/bootstrap_classifier.py tests/unit/test_bootstrap_classifier.py`
3. `uv run pyright scripts/bootstrap_classifier.py tests/unit/test_bootstrap_classifier.py`
4. `uv run pytest tests/unit/test_bootstrap_classifier.py -v`
5. `uv run pytest tests/`
