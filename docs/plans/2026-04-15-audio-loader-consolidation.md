# Audio Loader Consolidation Implementation Plan

**Goal:** Replace six duplicated private audio loader implementations with a shared module that centralizes coordinate conversion, caching, and the two consumer protocol families.
**Spec:** `docs/specs/2026-04-15-audio-loader-consolidation-design.md`

---

### Task 1: Create shared audio loader module

**Files:**
- Create: `src/humpback/call_parsing/audio_loader.py`

**Acceptance criteria:**
- [ ] `CachedAudioSource` class with `from_file` and `from_hydrophone` classmethods
- [ ] `from_file` decodes audio once, caches waveform, returns `(audio, 0.0)` from `get_audio`
- [ ] `from_hydrophone` accepts optional `preload_span`; when set, calls `resolve_timeline_audio` once at construction
- [ ] `get_audio(rel_start_sec, duration_sec)` performs `job_start_ts + rel_start_sec` conversion for hydrophone sources
- [ ] `build_event_audio_loader` factory returns `Callable[[Any], tuple[np.ndarray, float]]`
- [ ] `build_event_audio_loader` computes bounding span with context padding from `preload_events` when provided
- [ ] Context padding uses `max(10.0, event_duration)`, symmetric, clamped to job bounds
- [ ] `build_region_audio_loader` factory returns `Callable[[Any], np.ndarray]`
- [ ] Region factory slices the returned buffer using `padded_start_sec`/`padded_end_sec` and `target_sr`
- [ ] File-based region loader slices from cached waveform using sample indices
- [ ] Module passes Pyright type checking

**Tests needed:**
- Coordinate conversion tests (spec test cases 1-6)
- Pre-load span caching tests (spec test cases 7-10)
- Boundary/degenerate input tests (spec test cases 11-14)
- Protocol contract tests (spec test cases 15-18)
- Integration tests verifying `resolve_timeline_audio` call counts (spec test cases 19-20)

---

### Task 2: Create test suite for the shared module

**Files:**
- Create: `tests/test_audio_loader.py`

**Acceptance criteria:**
- [ ] All 20 test cases from the spec are implemented
- [ ] Hydrophone tests mock `resolve_timeline_audio` to verify arguments and call counts
- [ ] File-based tests use a real temporary WAV file (no mocking of decode)
- [ ] All tests pass

**Tests needed:**
- This task IS the test suite

---

### Task 3: Migrate event classification worker

**Files:**
- Modify: `src/humpback/workers/event_classification_worker.py`

**Acceptance criteria:**
- [ ] Delete `_build_audio_loader` (file-based, lines ~66-78)
- [ ] Delete `_build_hydrophone_audio_loader` (lines ~81-114)
- [ ] Replace call sites with `build_event_audio_loader(audio_file=...)` and `build_event_audio_loader(hydrophone_id=..., preload_events=events)`
- [ ] Remove now-unused imports (`decode_audio`, `resample`, `resolve_audio_path` if no other usage)
- [ ] Existing tests still pass

**Tests needed:**
- Existing worker tests cover this; no new tests needed

---

### Task 4: Migrate event classifier feedback worker

**Files:**
- Modify: `src/humpback/workers/event_classifier_feedback_worker.py`

**Acceptance criteria:**
- [ ] Delete `_build_audio_loader` (lines ~207-246)
- [ ] Replace call site with `build_event_audio_loader(hydrophone_id=..., preload_events=samples)`
- [ ] Remove now-unused imports
- [ ] Existing tests still pass
- [ ] Training now benefits from span pre-loading (was per-sample before)

**Tests needed:**
- Existing worker tests cover this; no new tests needed

---

### Task 5: Migrate segmentation training worker

**Files:**
- Modify: `src/humpback/workers/segmentation_training_worker.py`

**Acceptance criteria:**
- [ ] Delete `_build_audio_loader` (lines ~52-81)
- [ ] Replace call site with `build_region_audio_loader(hydrophone_id=..., preload_span=...)` where span is computed from sample bounds
- [ ] Remove now-unused imports
- [ ] Existing tests still pass
- [ ] Training now benefits from span pre-loading (was per-sample before)

**Tests needed:**
- Existing worker tests cover this; no new tests needed

---

### Task 6: Migrate event segmentation worker

**Files:**
- Modify: `src/humpback/workers/event_segmentation_worker.py`

**Acceptance criteria:**
- [ ] Delete `_build_file_audio_loader` (lines ~93-111)
- [ ] Delete `_build_hydrophone_audio_loader` (lines ~114-140)
- [ ] Replace call sites with `build_region_audio_loader(audio_file=...)` and `build_region_audio_loader(hydrophone_id=...)`
- [ ] Remove now-unused imports (`decode_audio`, `resample`, `resolve_audio_path` if no other usage)
- [ ] Existing tests still pass

**Tests needed:**
- Existing worker tests cover this; no new tests needed

---

### Task 7: Migrate bootstrap classifier script

**Files:**
- Modify: `scripts/bootstrap_classifier.py`

**Acceptance criteria:**
- [ ] Delete `_build_audio_loader` (lines ~137-168)
- [ ] Replace call site with `build_event_audio_loader(hydrophone_id=..., preload_events=samples)`
- [ ] Remove now-unused imports
- [ ] Script still passes Pyright

**Tests needed:**
- No dedicated test; verified by Pyright and manual review

---

### Verification

Run in order after all tasks:
1. `uv run ruff format --check src/humpback/call_parsing/audio_loader.py src/humpback/workers/event_classification_worker.py src/humpback/workers/event_classifier_feedback_worker.py src/humpback/workers/segmentation_training_worker.py src/humpback/workers/event_segmentation_worker.py scripts/bootstrap_classifier.py tests/test_audio_loader.py`
2. `uv run ruff check src/humpback/call_parsing/audio_loader.py src/humpback/workers/event_classification_worker.py src/humpback/workers/event_classifier_feedback_worker.py src/humpback/workers/segmentation_training_worker.py src/humpback/workers/event_segmentation_worker.py scripts/bootstrap_classifier.py tests/test_audio_loader.py`
3. `uv run pyright src/humpback/call_parsing/audio_loader.py src/humpback/workers/event_classification_worker.py src/humpback/workers/event_classifier_feedback_worker.py src/humpback/workers/segmentation_training_worker.py src/humpback/workers/event_segmentation_worker.py scripts/bootstrap_classifier.py tests/test_audio_loader.py`
4. `uv run pytest tests/`
