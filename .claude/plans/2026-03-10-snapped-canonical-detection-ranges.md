# Plan: Snapped Canonical Detection Ranges With Preview/Extract Parity

## Summary

- Canonical user-facing detection ranges will be snapped to `window_size_seconds` multiples before labeling.
- Preview, labeling, and extraction will operate on the same snapped clip bounds.
- Original unsnapped event bounds will be retained as audit metadata.
- Legacy TSV rows will be normalized at read/download time without rewriting source files.

## Key Changes

1. Detection generation (local + hydrophone)
   - Snap merged event bounds to window multiples.
   - Merge snapped-range collisions deterministically.
   - Persist audit fields (`raw_start_sec`, `raw_end_sec`) plus merged-count metadata.
2. Canonical filenames and extraction
   - Use canonical snapped filename bounds for hydrophone detection/extraction.
   - Remove extraction-time widening for local extraction; trust canonical row bounds.
   - Legacy hydrophone fallback order: `detection_filename` -> `extract_filename` -> snapped-from-range.
3. API and TSV normalization
   - Normalize legacy rows in content/download responses.
   - Expose raw audit metadata in detection content.

## Tests

- Unit tests for snapping + collision merge logic.
- Unit tests for extraction precedence and no-extra-snapping behavior.
- Integration tests for content/download normalization and preview/extract parity.
- Regression tests for label save compatibility with added metadata columns.
