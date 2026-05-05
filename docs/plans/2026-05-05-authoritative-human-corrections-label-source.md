# Authoritative Human Corrections Label Source Implementation Plan

**Goal:** Make Sequence Models label-distribution and exemplar annotations treat human Classify corrections as authoritative replacement labels for corrected events.
**Spec:** `docs/specs/2026-05-05-authoritative-human-corrections-label-source-design.md`

---

### Task 1: Replace Union Semantics in Effective Event Label Loading

**Files:**
- Modify: `src/humpback/sequence_models/label_distribution.py`

**Acceptance criteria:**
- [ ] `load_effective_event_labels()` groups overlapping `VocalizationCorrection` rows into added and removed type sets per effective event.
- [ ] Events with at least one overlapping `add` correction use the added type set as the base labels instead of unioning with model above-threshold labels.
- [ ] Events with no overlapping `add` correction preserve the existing model above-threshold base labels.
- [ ] Overlapping `remove` corrections subtract from whichever base set was selected.
- [ ] `event_confidence` retains model confidence only for surviving model-origin labels and does not fabricate confidence for user-added labels.
- [ ] Empty final type sets continue to produce background window annotations through the existing `assign_labels_to_windows()` behavior.

**Tests needed:**
- Unit coverage proving human adds replace collapsed model labels.
- Unit coverage proving model labels remain active when there is no human add.
- Unit coverage proving removes subtract from both model-only and human-replacement paths.
- Unit coverage proving empty final labels are still handled as background.

---

### Task 2: Extend Label Loader Regression Tests

**Files:**
- Modify: `tests/sequence_models/test_load_effective_event_labels.py`
- Modify: `tests/sequence_models/test_label_distribution.py`

**Acceptance criteria:**
- [ ] Loader tests cover an event whose model emits multiple above-threshold labels and whose human add selects one label.
- [ ] Loader tests cover multi-label human additions as intentional multi-label replacements.
- [ ] Loader tests cover a human add plus remove on the same effective event.
- [ ] Existing tests that intentionally verify additive correction behavior are updated to the new authoritative semantics.
- [ ] Pure label-distribution tests remain focused on window assignment and background behavior, with docstrings updated if they imply union semantics.

**Tests needed:**
- `uv run pytest tests/sequence_models/test_load_effective_event_labels.py tests/sequence_models/test_label_distribution.py`

---

### Task 3: Add Sequence Models Artifact Regression Coverage

**Files:**
- Modify: `tests/workers/test_masked_transformer_worker.py`
- Modify: `tests/services/test_hmm_sequence_service.py` or the current HMM label-distribution service test file if coverage has moved.
- Modify: `tests/services/test_masked_transformer_service.py` or the current MT label-distribution service test file if coverage has moved.

**Acceptance criteria:**
- [ ] A Masked Transformer interpretation test constructs collapsed `typed_events.parquet` model labels plus varied `VocalizationCorrection(add)` rows and asserts `exemplars.json` reflects human labels.
- [ ] The same MT test or a nearby service test asserts `label_distribution.json` reflects varied human labels after regeneration.
- [ ] HMM coverage exercises the same effective-label helper path so both Sequence Models consumers are protected.
- [ ] Existing artifact shape remains unchanged: `event_id`, `event_types`, and `event_confidence` keys are still present in exemplar extras.

**Tests needed:**
- Targeted MT worker/service tests for regenerated exemplars and label distribution.
- Targeted HMM service tests for regenerated exemplars and label distribution.

---

### Task 4: Update ADR and Design Documentation

**Files:**
- Modify: `DECISIONS.md`
- Modify: `docs/specs/2026-05-04-sequence-models-classify-label-source-design.md`
- Keep: `docs/specs/2026-05-05-authoritative-human-corrections-label-source-design.md`

**Acceptance criteria:**
- [ ] ADR-063 or a new follow-up ADR states the authoritative-human-corrections formula.
- [ ] The older union formula is clearly marked superseded for Sequence Models interpretation artifacts.
- [ ] The May 4 label-source spec is aligned with the new replacement semantics or points to the May 5 spec as an amendment.
- [ ] Documentation notes that existing artifacts require manual regeneration.

**Tests needed:**
- Documentation-only review for consistency with implemented semantics.

---

### Task 5: Local Diagnostic Check Against Investigated Data

**Files:**
- No committed diagnostic script unless the implementation naturally exposes a reusable test helper.

**Acceptance criteria:**
- [ ] Before/after diagnostic confirms the investigated Classify job no longer resolves most corrected events to the collapsed 11-label set.
- [ ] Corrected events resolve mostly to the single human-added label distribution observed in the database.
- [ ] Uncorrected events still fall back to model above-threshold labels.
- [ ] The diagnostic result is summarized in the implementation session notes or final response.

**Tests needed:**
- Run a small local diagnostic using the `.env` database and storage root after the code change.

---

### Verification

Run in order after all tasks:

1. `uv run ruff format --check src/humpback/sequence_models/label_distribution.py tests/sequence_models/test_load_effective_event_labels.py tests/sequence_models/test_label_distribution.py tests/workers/test_masked_transformer_worker.py tests/services/test_hmm_sequence_service.py tests/services/test_masked_transformer_service.py`
2. `uv run ruff check src/humpback/sequence_models/label_distribution.py tests/sequence_models/test_load_effective_event_labels.py tests/sequence_models/test_label_distribution.py tests/workers/test_masked_transformer_worker.py tests/services/test_hmm_sequence_service.py tests/services/test_masked_transformer_service.py`
3. `uv run pyright src/humpback/sequence_models/label_distribution.py`
4. `uv run pytest tests/sequence_models/test_load_effective_event_labels.py tests/sequence_models/test_label_distribution.py`
5. `uv run pytest tests/workers/test_masked_transformer_worker.py tests/services/test_hmm_sequence_service.py tests/services/test_masked_transformer_service.py`
6. `uv run pytest tests/`
