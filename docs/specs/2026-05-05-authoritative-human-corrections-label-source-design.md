# Sequence Models — Authoritative Human Corrections for Event Labels

**Status:** design
**Date:** 2026-05-05
**Related:** ADR-054, ADR-063, `docs/specs/2026-05-04-sequence-models-classify-label-source-design.md`

## 1. Problem

ADR-063 switched HMM and Masked Transformer label-distribution artifacts and
exemplar chips from legacy `vocalization_labels` to the bound Call Parsing
Classify job:

```
effective labels =
  model above-threshold labels
  ∪ VocalizationCorrection(add)
  − VocalizationCorrection(remove)
```

That union semantics is correct when a user adds an extra type to a mostly
trustworthy model output. It breaks down when human corrections are intended to
override a collapsed or over-broad Classify result.

The current production data shows this failure mode:

- Bound Classify job `2070352e-e459-40ee-955f-0b013563dd41` emits one identical
  11-label above-threshold set for all 1,257 typed event ids.
- The associated region has human `VocalizationCorrection` rows overlapping
  1,236 of 1,273 effective events.
- Most corrected events have exactly one human `add` label.
- Because the loader unions model labels and human adds, 1,183 events still
  resolve to the same 11-label set after overlay.
- Masked Transformer exemplars therefore appear to share the same label set
  across jobs and parameters even when the human-corrected labels are varied.

This is not a Masked Transformer time-alignment issue. The window timestamps,
decoded rows, and contextual rows align. The wrong label variety is introduced
by effective-label semantics.

## 2. Goal

Make human type corrections authoritative for Sequence Models label
distribution and exemplar annotation:

```
if an effective event has any overlapping VocalizationCorrection(add):
    base labels = all overlapping add type names
else:
    base labels = model above-threshold labels

effective labels = base labels − overlapping VocalizationCorrection(remove)
```

In words: once a user has assigned one or more labels to an event in Classify
Review, those human labels replace the model's type set for Sequence Models
interpretation artifacts.

## 3. Non-Goals

- Do not change Pass 3 inference or `typed_events.parquet`.
- Do not change Classify Review correction storage.
- Do not change boundary correction semantics.
- Do not change contrastive training's existing human-correction-only label
  path unless a test proves it shares the same bug.
- Do not automatically regenerate existing HMM or Masked Transformer artifacts
  on correction writes.
- Do not introduce a user-facing toggle for replacement versus union semantics.

## 4. Design

### 4.1 Effective Label Rule

Update `load_effective_event_labels()` in
`src/humpback/sequence_models/label_distribution.py`.

For each effective event:

1. Load model labels from `typed_events.parquet` where `above_threshold == True`.
2. Find overlapping `VocalizationCorrection` rows for the region detection job.
3. Split overlaps into `added_types` and `removed_types`.
4. If `added_types` is non-empty, set `types = added_types`.
5. Otherwise set `types = model_types`.
6. Remove all `removed_types`.
7. Keep model confidences only for surviving model-origin labels.

User-added labels do not have classifier confidence, so they remain absent from
`event_confidence`. This matches current behavior for added labels and avoids
inventing confidence values.

### 4.2 Empty Labels

Events whose final type set is empty keep the current behavior:

- They are returned by `load_effective_event_labels()` as real intervals.
- `assign_labels_to_windows()` maps their windows to background:
  `event_id=None`, `event_types=[]`, `event_confidence={}`.

This preserves the invariant used by exemplar cards: `event_id` is present only
when there is at least one surviving label to link.

### 4.3 Overlap Semantics

Keep the current overlap predicate:

```
vc.start_sec < event.end_sec and vc.end_sec > event.start_sec
```

This is consistent with existing region-scoped correction semantics. The fix is
about replacement versus union, not boundary matching.

### 4.4 Provenance Semantics

ADR-063 should be amended to say:

```
Effective type set per event =
  human-added types − human-removed types, when any human add overlaps the event;
  otherwise model above-threshold types − human-removed types.
```

The previous union rule should be documented as superseded for Sequence Models
interpretation artifacts.

## 5. Affected Surfaces

Backend:

- `src/humpback/sequence_models/label_distribution.py`
- `tests/sequence_models/test_load_effective_event_labels.py`
- `tests/sequence_models/test_label_distribution.py` if any docstring or
  behavioral expectation references union semantics.
- HMM/MT service tests that assert exemplar `event_types` or
  `label_distribution.json` counts.
- `DECISIONS.md` ADR-063 amendment or new ADR entry.

Frontend:

- No direct behavior change. The frontend already renders whatever
  `exemplars.json` and `label_distribution.json` contain.
- Existing artifacts must be regenerated manually from the detail-page
  regenerate action to reflect the new semantics.

Docs:

- Update the ADR-063 effective type-set formula.
- Update the May 4 Classify label-source spec if implementation keeps specs
  aligned after design acceptance.

## 6. Migration and Regeneration

No database migration is required.

Existing artifact files remain stale until regenerated:

- HMM: `POST /sequence-models/hmm-sequences/{id}/regenerate-label-distribution`
- MT: `POST /sequence-models/masked-transformer/{id}/regenerate-label-distribution`

The MT endpoint already rebuilds all `k<N>/label_distribution.json` files and
the per-k `exemplars.json` files. After this fix lands, users should regenerate
the affected jobs bound to collapsed or over-broad Classify outputs.

## 7. Acceptance Criteria

- An event with model labels `{"A", "B", "C"}` and one overlapping human
  `add("B")` resolves to `{"B"}`, not `{"A", "B", "C"}`.
- An event with model labels `{"A", "B"}` and no human adds resolves to
  `{"A", "B"}`.
- An event with model labels `{"A", "B"}`, no human adds, and
  `remove("B")` resolves to `{"A"}`.
- An event with human adds `{"A", "C"}` and `remove("C")` resolves to `{"A"}`.
- An event whose human add labels are all removed resolves to background in
  window annotations.
- Model confidences are retained only for surviving model-origin labels.
- User-added labels appear in `event_types` without fabricated confidence.
- HMM and Masked Transformer regenerated exemplar chips show varied
  human-corrected labels for the investigated Classify job.

## 8. Test Plan

Unit tests:

- Extend `tests/sequence_models/test_load_effective_event_labels.py` with:
  - human add replaces collapsed model labels;
  - no human add preserves model labels;
  - remove still subtracts from model-only labels;
  - remove subtracts from human replacement labels;
  - empty final labels remain representable as background.

Service tests:

- Add or update an HMM/MT label-distribution fixture where
  `typed_events.parquet` gives every event the same model labels but
  `VocalizationCorrection(add)` differs by event. Assert regenerated
  `label_distribution.json` and exemplar `extras.event_types` reflect the
  human labels.

Regression data check:

- Against the local investigated data, run a small diagnostic script before and
  after the change:
  - before: most events resolve to the same 11-label set;
  - after: corrected events resolve mostly to single human labels with a varied
    distribution.

Verification gates:

- `uv run pytest tests/sequence_models/test_load_effective_event_labels.py`
- Targeted HMM/MT label-distribution service tests.
- Full project verification per CLAUDE.md §10.2 before session end.

## 9. Risks and Trade-Offs

- A user may intend to add an extra type rather than replace model labels.
  Current correction UI/data does not distinguish "confirm/additive" from
  "authoritative replacement". Given Classify Review usage and the observed
  collapsed-model failure, replacement is the safer Sequence Models
  interpretation default.
- If a genuinely multi-label event is corrected with multiple add rows, the new
  rule preserves that multi-label event.
- If a user only removes one incorrect model label and adds no replacement
  label, model labels continue to supply the base set. This preserves existing
  subtractive cleanup behavior.

## 10. Open Questions

- Should Classify Review eventually write explicit removes for all unselected
  model labels when a user chooses a replacement label? That would make the
  stored correction set self-describing across all consumers, but it is not
  required for this Sequence Models fix.
- Should model-quality guardrails prevent collapsed Classify jobs from being
  offered as Sequence Models label sources? This is complementary but separate:
  authoritative human corrections fix corrected events; guardrails would help
  uncorrected events and future jobs.
