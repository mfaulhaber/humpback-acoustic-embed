# Fix Comparison Logic Defaults — Design Spec

## Problem

`compare_classifiers` in `comparison.py` evaluates production models with incorrect
hardcoded defaults:

- `production_context_pooling` defaults to `"center"` — should use the model's actual
  training config (e.g., `"mean3"`)
- `production_threshold` defaults to `0.5` — should use the model's actual threshold
  (e.g., `0.5` from `promoted_config`)

Both values are available in `production_classifier["training_summary"]` under
`promoted_config` (or `replay_effective_config`), which `resolve_production_classifier`
already loads. The result is that comparison metrics for the production model are
computed under mismatched conditions — wrong context pooling and potentially wrong
threshold — leading to misleading precision/recall numbers.

Additionally, the `scripts/autoresearch/` directory contains deprecated CLI wrappers
that have been fully superseded by the hyperparameter service layer. These should be
removed.

## Solution

### Auto-detect production model settings

Change `compare_classifiers` to accept `None` for both `production_context_pooling`
and `production_threshold`. When `None`, extract the correct values from the
production classifier's `training_summary` JSON:

1. Check `promoted_config` first (present on candidate-promoted models)
2. Fall back to `replay_effective_config`
3. Fall back to legacy defaults (`"center"` / `0.5`) for older models without
   training summary data

Add a helper `_resolve_production_defaults()` to encapsulate this extraction logic.

### Worker cleanup

The hyperparameter worker currently passes `production_threshold=job.comparison_threshold or 0.5`.
Change to only pass `production_threshold` when `job.comparison_threshold` is explicitly set,
letting auto-detect handle the default case.

### Remove deprecated scripts

Delete `scripts/autoresearch/` entirely and clean up references in `CLAUDE.md` and
`README.md`. Test files import from the service layer, not the scripts, so they are
unaffected.

## Out of scope

- API/DB fields for explicit context_pooling override
- Re-running past comparisons with corrected settings
- Changes to the autoresearch candidate import flow
