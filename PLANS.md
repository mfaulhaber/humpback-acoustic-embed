# Development Plans

---

## Active

- None currently active.

---

## Recently Completed

### Compact PLANS.md and STATUS.md Safely

[Full plan](/Users/michael/.claude/plans/compact-plans-status-docs.md)

- Compacted `STATUS.md` into a shorter session-start snapshot while preserving capability buckets, schema state, sensitive components, known constraints, and the highest-signal current behavior changes.
- Compacted `PLANS.md` into one active-plan pointer, six recent-completed cards, a concise backlog, and dated links to older plan files.
- Kept the session-skill workflow intact by preserving top-level sections, active-plan link semantics, and direct links to full plan files.
- Verification: `uv run pytest tests/` passed (`974 passed, 1 skipped`).

### Retire Merged Detection Mode

[Full plan](/Users/michael/.claude/plans/retire-merged-detection-mode.md)

- New detection jobs are always created as `windowed`; merged-mode creation was removed from the API, Hydrophone UI, and NOAA metadata helper tooling.
- Legacy merged jobs remain readable in the UI and API, but are now read-only for label saves, row-state edits, and extraction.
- ADR-039 plus API, UI, test, and documentation updates landed for the rollout.
- Verification: Ruff, Pyright, `cd frontend && npx tsc --noEmit`, and `uv run pytest tests/` all passed (`974 passed, 1 skipped`).

### Sample Builder — Fix 100% Rejection Rate on Marine Recordings

[Full plan](/Users/michael/.claude/plans/buzzing-petting-zephyr.md)

- Retuned contamination screening for real marine noise with per-bin tonal persistence, a raised spectral occupancy floor, and a much looser splice energy ratio.
- Widened default annotation duration bounds to `[0.1s, 10.0s]` and exposed the contamination knobs through worker job parameters.
- ADR-038 records the signal-processing rationale and trade-offs.
- Verification: Ruff, Pyright, `cd frontend && npx tsc --noEmit`, and `uv run pytest tests/` all passed (`968 passed, 1 skipped`).

### 5-Second Sample Builder — Alternative Label Processing Workflow

[Full plan](/Users/michael/.claude/plans/structured-baking-mist.md)

- Added `sample_builder`, a classifier-free 10-stage label-processing workflow covering normalization, contamination screening, assembly, validation, and orchestration.
- Added migration `021_label_processing_workflow.py`, made `classifier_model_id` nullable, and dispatched worker behavior by workflow type.
- Frontend now exposes workflow selection plus sample-builder acceptance and rejection stats.
- Verification: Ruff, Pyright, `cd frontend && npx tsc --noEmit`, and `uv run pytest tests/` all passed (`968 passed, 1 skipped`).

### Fix Synthesis Call Isolation to Use Annotation Bounds

[Full plan](/Users/michael/.claude/plans/humble-marinating-hamming.md)

- `isolate_call_segment()` now centers on annotation bounds instead of a shared classifier peak, preventing label contamination across nearby calls.
- Added adaptive per-recording background thresholds, short-run tiling, and rotated background offsets for denser and less repetitive synthesis output.
- ADR-037 documents the synthesis changes and new `background_threshold_auto` / `background_min_duration` defaults.
- Verification: Ruff, Pyright, and `uv run pytest tests/` all passed (`833 passed, 1 skipped`).

### Synthesize All Annotations + Score KPIs

[Full plan](/Users/michael/.claude/plans/foamy-sauteeing-willow.md)

- Removed the old recentering treatment; annotations with matched peaks now route to synthesis, and clean annotations get both clean and synthesized outputs.
- Added per-label classifier score KPIs to job results and fixed impulsive background spikes with a crest-factor limiter.
- Updated the related workflow documentation.
- Verification: Ruff, Pyright, `cd frontend && npx tsc --noEmit`, and `uv run pytest tests/` all passed (`822 passed, 1 skipped`).

## Backlog

- Agile Modeling Phase 1b: search by uploaded audio clip by embedding the clip on the fly with a selected model, then searching existing embedding sets.
- Agile Modeling Phase 3: connect search-result labeling into classifier training and the retrain loop.
- Agile Modeling Phase 4: prioritize labeling suggestions using model uncertainty signals such as entropy or margin.
- Smoke-test `tf-linux-gpu` on a real Ubuntu/NVIDIA host, including `uv sync --extra tf-linux-gpu`, TensorFlow import, and GPU device visibility.
- Generalize legacy hydrophone API and frontend naming toward archive-source terminology now that NOAA Glacier Bay shares the same backend surfaces.
- Explore GPU-accelerated batch processing for large audio libraries.
- Add WebSocket push for real-time job status updates to replace polling.
- Investigate multi-model ensemble clustering.
- Optimize `/audio/{id}/spectrogram` to avoid materializing all windows when only one index is requested.
- Optimize hydrophone incremental lookback discovery to avoid repeated full S3 folder scans during startup.
- Add an integration and performance harness for hydrophone S3 prefetch so worker defaults can be tuned on real S3-backed runs.
- Investigate a lower-overhead Orcasound decode path, likely chunk-level or persistent-stream decode, and treat it as a signal-processing/runtime change that needs validation plus an ADR.
- Make `hydrophone_id` optional for local-cache detection jobs in the backend API, service layer, and worker.
- Remove vestigial `output_tsv_path` and `output_row_store_path` fields from the detection model, schema, and database via migration.

---

## Completed

### Older Plan Files

- 2026-03-20 — [Audio/Label Processing — Classifier Score as Segmentation Signal](/Users/michael/.claude/plans/valiant-crunching-moonbeam.md)
- 2026-03-19 — [Search Embedding Selection Enhancements](/Users/michael/.claude/plans/streamed-crafting-turtle.md)
- 2026-03-19 — [Fix NOAA Channel Islands Detection Playback/Spectrogram 500 Error](/Users/michael/.claude/plans/serene-giggling-hamming.md)
- 2026-03-19 — [Side + Top Navigation with Breadcrumbs](/Users/michael/.claude/plans/effervescent-soaring-comet.md)
- 2026-03-19 — [Derive Detection Output Paths from Storage Root, Not DB](/Users/michael/.claude/plans/sparkling-jingling-axolotl.md)
- 2026-03-19 — [Search by Audio — Worker-Encoded Detection Search](/Users/michael/.claude/plans/greedy-tickling-orbit.md)
- 2026-03-18 — [Agile Modeling Phase 2 — Search Results UI](/Users/michael/.claude/plans/polished-strolling-hare.md)
- 2026-03-18 — [Agile Modeling Phase 1 — Embedding Similarity Search](/Users/michael/.claude/plans/replicated-snacking-goblet.md)
- 2026-03-18 — [DB Load Logging + UI Error Flash Bar](/Users/michael/.claude/plans/staged-greeting-waterfall.md)
- 2026-03-18 — [Local Dev Stack Startup](/Users/michael/.claude/plans/playful-scribbling-eclipse.md)
- 2026-03-17 — [Fix Slow NOAA SanctSound Playback — Process-Level Provider Registry](/Users/michael/.claude/plans/imperative-wiggling-pascal.md)
- 2026-03-17 — [Refactor noaa_detection_metadata.py — CSV URL Input](/Users/michael/.claude/plans/sleepy-swinging-wren.md)
- 2026-03-17 — [Enable Multi-Site NOAA SanctSound Sources (Channel Islands + Olympic Coast)](/Users/michael/.claude/plans/shimmying-napping-avalanche.md)
- 2026-03-16 — [Fix Slow NOAA SanctSound Audio Preview](/Users/michael/.claude/plans/misty-dreaming-pascal.md)
- 2026-03-16 — [NOAA Hydrophone Detection Metadata Job Generator](/Users/michael/.claude/plans/parsed-pondering-curry.md)
- 2026-03-16 — [Windowed Detection Mode (Fixed 5-Second Detections)](/Users/michael/.claude/plans/adaptive-wiggling-firefly.md)
- 2026-03-16 — [NOAA Hydrophone Metadata](/Users/michael/.claude/plans/noaa-hydrophone-metadata.md)
- 2026-03-16 — [Isolate TF2 Hydrophone Detection in a Subprocess](/Users/michael/.claude/plans/tf2-hydrophone-subprocess-isolation.md)

### Earlier Completed Items

- Rewrite README Overview for Research-First Positioning
- Add Year-Jump Buttons to the Hydrophone UTC Date Picker
- Sidecar Spectrogram PNGs for Extracted Detection Clips
- Classifier/Detection Data Flow + Spectrogram Windowing
- UI Changes for Classifier/Detection Spectrogram
- Relax Positive Extraction Length With 5-Second Chunk Growth
- Positive Window Selection From Stored Detection Scores
- UI Changes for Classifier/Detection Page
- UI Refactor for Classifier/Detection Page
- Fix NOAA GCS Playback/Spectrogram — Interval Estimation Bug
