# Autoresearch Output Fixtures

This directory vendors a small set of real autoresearch JSON artifacts into the repo so the Classifier Training UI and API can be developed and tested without depending on `/tmp`.

## Fixture Set

- `explicit-negatives/`

This bundle comes from the 2026-04-03 production-backed explicit-negative experiment using detection jobs:

- `2a5f51f3-b91d-470e-a92e-4900ebedb97d`
- `23a1f7ca-7777-4f2b-83bf-8eb4ccc9fef3`

It includes:

- `manifest.json`
- `comparison_summary.json` (lightweight bundle summary; importable, but without split-level production deltas)
- phase 1 search outputs, including `lr-v12-comparison.json`
- phase 2 search outputs

## LR-v12 Relationship

`phase1/lr-v12-comparison.json` captures the production comparison that the promotion workflow is built around in this milestone. It compares the imported autoresearch candidate against the production classifier `LR-v12`, including split metrics, deltas, disagreement previews, and top false positives that the UI surfaces during review.

## Intended Use

- frontend fixture data for candidate import/review UIs
- API tests for artifact import and candidate summarization
- regression fixtures for promotion workflows built on top of `scripts/autoresearch/compare_classifiers.py`

## Notes

- These files are development fixtures, not a canonical storage location for production experiment outputs.
- Paths and model IDs inside the JSON reflect the original run provenance and are intentionally preserved for realism.
