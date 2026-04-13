# Bootstrap Hydrophone Support Implementation Plan

**Goal:** Make both bootstrap scripts and the segmentation training worker work with hydrophone-sourced detection jobs, removing file-based audio resolution code from the scripts since call parsing is hydrophone-only.
**Spec:** [docs/specs/2026-04-12-bootstrap-hydrophone-support-design.md](../specs/2026-04-12-bootstrap-hydrophone-support-design.md)

---

### Task 1: Refactor `bootstrap_segmentation_dataset.py` to hydrophone-only

**Files:**
- Modify: `scripts/bootstrap_segmentation_dataset.py`

**Acceptance criteria:**
- [ ] File-based helpers removed: `_file_base_epoch`, `_build_audio_index`, `_resolve_file_for_row`, `AudioFile` lookups
- [ ] Detection jobs without `hydrophone_id` are skipped with warning
- [ ] Detection job's `hydrophone_id`, `start_timestamp`, `end_timestamp` are read for each job
- [ ] Crop window computed from detection row's absolute `start_utc`/`end_utc`, clamped to detection job's `[start_timestamp, end_timestamp]` range
- [ ] Samples stored with `hydrophone_id=dj.hydrophone_id`, `start_timestamp=crop_start_utc`, `end_timestamp=crop_end_utc`, `crop_start_sec=0.0`, `crop_end_sec=crop_duration`, `audio_file_id=None`
- [ ] `events_json` contains event bounds relative to crop start
- [ ] Idempotency check unchanged (uses `source_ref=row_id`)
- [ ] Dry-run mode still works
- [ ] `_discover_row_ids_from_jobs` no longer skips hydrophone jobs

**Tests needed:**
- Covered in Task 2

---

### Task 2: Update tests for `bootstrap_segmentation_dataset.py`

**Files:**
- Modify: `tests/unit/test_bootstrap_segmentation_dataset.py`

**Acceptance criteria:**
- [ ] `_seed_fixture` creates hydrophone-sourced detection jobs (sets `hydrophone_id`, `start_timestamp`, `end_timestamp`, no `audio_folder`)
- [ ] Happy path test asserts sample has `hydrophone_id` set, `audio_file_id` is None, `start_timestamp`/`end_timestamp` are correct crop bounds, `crop_start_sec=0.0`
- [ ] No-label, multi-label, idempotency, dry-run, unknown-row, crop-too-short tests all pass with hydrophone fixtures
- [ ] `_discover_row_ids_from_jobs` test uses hydrophone detection jobs
- [ ] File-based test helpers removed (`_write_sine_wav`, `AudioFile` fixtures)

**Tests needed:**
- Self-referential — this IS the test task

---

### Task 3: Add hydrophone support to segmentation training worker

**Files:**
- Modify: `src/humpback/workers/segmentation_training_worker.py`

**Acceptance criteria:**
- [ ] `_build_audio_loader` handles hydrophone samples: when `sample.hydrophone_id` is set and `sample.audio_file_id` is None, fetches audio via `resolve_timeline_audio()`
- [ ] `resolve_timeline_audio` called with `hydrophone_id=sample.hydrophone_id`, `local_cache_path` from settings, `job_start_timestamp=sample.start_timestamp`, `job_end_timestamp=sample.end_timestamp`, `start_sec=sample.start_timestamp`, `duration_sec=crop_end - crop_start`, `target_sr` from feature config
- [ ] File-based audio loading path preserved (worker is generic infrastructure)
- [ ] `_load_audio_files` still works for mixed datasets (skips samples without `audio_file_id`)
- [ ] Settings passed to `_build_audio_loader` (needs `s3_cache_path`, `noaa_cache_path`)

**Tests needed:**
- Covered in Task 4

---

### Task 4: Add hydrophone training worker test

**Files:**
- Modify: `tests/integration/test_segmentation_training_worker.py`

**Acceptance criteria:**
- [ ] New test seeds hydrophone-sourced `SegmentationTrainingSample` rows (hydrophone_id set, audio_file_id=None, start_timestamp/end_timestamp as crop bounds)
- [ ] `resolve_timeline_audio` is mocked to return synthetic audio (sine wave / silence) matching the requested duration
- [ ] Worker completes training, produces a `SegmentationModel` row and checkpoint on disk
- [ ] Result summary contains valid metrics

**Tests needed:**
- Self-referential — this IS the test task

---

### Task 5: Refactor `bootstrap_event_classifier_dataset.py` to hydrophone-only

**Files:**
- Modify: `scripts/bootstrap_event_classifier_dataset.py`

**Acceptance criteria:**
- [ ] File-based helpers removed: `_file_base_epoch`, `_build_audio_index`, `_resolve_file_for_row`, `AudioFile` lookups, per-file audio decode/cache
- [ ] Detection jobs without `hydrophone_id` are skipped with warning
- [ ] Audio for each window fetched via `resolve_timeline_audio()` using detection job's `hydrophone_id` and window's `start_utc`/`end_utc`
- [ ] Output samples contain `hydrophone_id` and absolute UTC `start_sec`/`end_sec` instead of `audio_file_id` and file-relative offsets
- [ ] Segmentation inference on hydrophone audio produces correct events
- [ ] Idempotency via `source_row_id` in output JSON unchanged
- [ ] Dry-run mode still works

**Tests needed:**
- Covered in Task 6

---

### Task 6: Update tests for `bootstrap_event_classifier_dataset.py`

**Files:**
- Modify: `tests/unit/test_bootstrap_event_classifier.py`

**Acceptance criteria:**
- [ ] Fixtures create hydrophone-sourced detection jobs (no `audio_folder`, has `hydrophone_id`/`start_timestamp`/`end_timestamp`)
- [ ] `resolve_timeline_audio` is mocked to return synthetic audio buffers
- [ ] Segmentation inference still mocked via `_mock_segmentation_events`
- [ ] Output samples assert `hydrophone_id` present, `audio_file_id` absent
- [ ] All existing test scenarios (single-label, multi-label, negative, events-outside-window) pass with hydrophone fixtures
- [ ] File-based test helpers removed (`_write_fake_audio`, `AudioFile` fixtures)

**Tests needed:**
- Self-referential — this IS the test task

---

### Verification

Run in order after all tasks:
1. `uv run ruff format --check scripts/bootstrap_segmentation_dataset.py scripts/bootstrap_event_classifier_dataset.py src/humpback/workers/segmentation_training_worker.py tests/unit/test_bootstrap_segmentation_dataset.py tests/unit/test_bootstrap_event_classifier.py tests/integration/test_segmentation_training_worker.py`
2. `uv run ruff check scripts/bootstrap_segmentation_dataset.py scripts/bootstrap_event_classifier_dataset.py src/humpback/workers/segmentation_training_worker.py tests/unit/test_bootstrap_segmentation_dataset.py tests/unit/test_bootstrap_event_classifier.py tests/integration/test_segmentation_training_worker.py`
3. `uv run pyright scripts/bootstrap_segmentation_dataset.py scripts/bootstrap_event_classifier_dataset.py src/humpback/workers/segmentation_training_worker.py`
4. `uv run pytest tests/unit/test_bootstrap_segmentation_dataset.py tests/unit/test_bootstrap_event_classifier.py tests/integration/test_segmentation_training_worker.py -v`
5. `uv run pytest tests/`
