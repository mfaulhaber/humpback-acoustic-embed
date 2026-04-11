# Architecture Decision Log

Record of significant design decisions with non-obvious reasoning.
Original numbering preserved; removed entries described current code behavior
without unique insight and were pruned in the documentation consolidation (2026-03-30).

---

## ADR-001: Overlap-back windowing replaces zero-padding

**Date**: 2026-03
**Status**: Accepted
**Commit**: `7e96418`

**Context**: Zero-padded final audio windows create out-of-distribution spectrograms (silence-filled regions) that cause false positive detections in downstream classifiers.

**Decision**: Replace zero-padding with overlap-back windowing. When the last audio chunk is shorter than `window_size_seconds`, shift its start backward so it ends at the audio boundary, overlapping with the previous window. Audio shorter than one window is skipped entirely.

**Consequences**:
- Every window contains only real audio — no synthetic silence
- `WindowMetadata.is_padded` replaced by `is_overlapped`
- Files shorter than `window_size_seconds` produce 0 embeddings (logged as warning)
- Classifier false positive rate significantly reduced

---

## ADR-003: Balanced class weights for detection classifier

**Date**: 2026-03
**Status**: Accepted
**Commit**: `176b572`

**Context**: Detection spans were covering entire files because the LogisticRegression classifier was biased toward the positive class when training data was imbalanced.

**Decision**: Default to `class_weight='balanced'` in the LogisticRegression classifier so that the model automatically adjusts weights inversely proportional to class frequencies.

**Consequences**:
- Fixes false-positive-heavy detection spans
- Works automatically regardless of positive/negative ratio
- No user configuration needed (sensible default)

---

## ADR-005: Overlapping window inference + hysteresis event detection

**Date**: 2026-03
**Status**: Accepted

**Context**: The detection pipeline used non-overlapping 5-second windows with index-based span merging. This caused poor temporal resolution (events snapped to 5s boundaries) and no ability to tune sensitivity with dual thresholds.

**Decision**: Add configurable `hop_seconds` for overlapping window inference and replace single-threshold span merging with hysteresis-based event detection using `high_threshold` (start) and `low_threshold` (continue). Extraction boundaries snap outward to `window_size` multiples for clean training samples.

**Consequences**:
- Detection defaults to 1s hop with 0.70/0.45 hysteresis thresholds (new behavior by default)
- Sub-second temporal resolution for event boundaries
- Per-event `n_windows` count in TSV output
- Added Alembic migration `010_detection_hysteresis.py`
- Legacy behavior available with `hop_seconds=5.0, high_threshold=0.5, low_threshold=0.5`

---

## ADR-007: MLP classifier option + enhanced diagnostics for iterative training

**Date**: 2026-03
**Status**: Accepted

**Context**: Users iteratively training binary classifiers with extract-reprocess-retrain loops experienced escalating false positives. The root cause is LogisticRegression's linear decision boundary cannot curve around overlapping embedding regions. Each retrain shifts the hyperplane, creating new FPs elsewhere. Additionally, limited diagnostics (only accuracy + AUC) masked precision problems.

**Decision**: Add MLP classifier as an alternative to LogisticRegression, L2 normalization option, expanded CV metrics (precision/recall/F1), and decision boundary diagnostics (score separation, train confusion matrix). Add overlap validation to prevent same embedding set in both positive and negative lists. Add encoding signature consistency warning.

**Consequences**:
- `classifier_type` parameter: `"logistic_regression"` (default, backward-compatible) or `"mlp"` (MLPClassifier with non-linear boundary)
- `l2_normalize` parameter: opt-in Normalizer step before StandardScaler
- Training summary now includes `cv_precision`, `cv_recall`, `cv_f1`, `score_separation`, `train_confusion`, `classifier_type`, `l2_normalize`, `effective_class_weights`
- Frontend advanced options: classifier type, L2 normalize, regularization C, class weight
- Frontend model table: Precision and F1 columns, diagnostic badges, expandable detail rows
- No schema changes, no migrations required

---

## ADR-009: Atomic compare-and-set claims for all queue job types

**Date**: 2026-03
**Status**: Accepted

**Context**: SQLite does not provide true row-level locking semantics compatible with the prior claim flow. Under concurrent workers, selecting a queued job before status update can race and allow duplicate claims for the same job.

**Decision**: Standardize queue claiming on a compare-and-set pattern for every job type. Each claimant selects a candidate queued job ID, then performs `UPDATE ... SET status='running' WHERE id=:candidate AND status='queued'`. A claim succeeds only when exactly one row is updated; otherwise the worker retries with the next candidate.

**Consequences**:
- Eliminates duplicate claims under concurrent worker sessions on SQLite
- Provides consistent claim behavior across processing, clustering, training, detection, hydrophone detection, and extraction jobs
- Reduces reliance on database locking features that differ by backend
- Requires small retry loops in claimers but keeps queue behavior deterministic

---

## ADR-022: Explicit platform TensorFlow extras and Python version cap

**Date**: 2026-03
**Status**: Accepted

**Context**: The project runs TensorFlow workloads on Apple Silicon macOS and on Linux
GPU servers. Keeping TensorFlow in the base dependency set forced one install contract
across incompatible platform/runtime combinations, and `uv sync --all-extras` was no
longer a safe default once Linux CPU, Linux GPU, and macOS TensorFlow variants diverged.
The supported TensorFlow wheel set also does not justify claiming Python 3.13 support.

**Decision**:
- Remove TensorFlow packages from base runtime dependencies.
- Add mutually-exclusive extras:
  - `tf-macos` for Apple Silicon (`tensorflow-macos` + `tensorflow-metal`)
  - `tf-linux-cpu` for Linux CPU (`tensorflow`)
  - `tf-linux-gpu` for Linux GPU/CUDA (`tensorflow[and-cuda]`)
- Declare the extra conflicts in `tool.uv.conflicts`.
- Cap supported Python versions at `>=3.11,<3.13`.
- Keep `soundfile` as a direct base dependency because extraction and FLAC tooling
  import it directly regardless of TensorFlow selection.

**Consequences**:
- Each environment must select exactly one TensorFlow extra; `uv sync --all-extras`
  is invalid by design.
- macOS and Linux deployments can resolve different TensorFlow stacks without
  weakening the shared base dependency set.
- The lockfile must be regenerated after TensorFlow dependency changes so platform
  forks stay explicit.
- Python 3.13 is intentionally unsupported until TensorFlow compatibility is validated.

---

## ADR-023: Env-driven deployment config and FastAPI trusted-host enforcement

**Date**: 2026-03
**Status**: Accepted

**Context**: Deployment-specific edits were being made directly to tracked files such
as `src/humpback/config.py` and `frontend/vite.config.ts` on remote hosts. That does
not survive the checked-in deployment flow because `scripts/deploy.sh` resets the repo
to `origin/main`. Production also serves the built SPA from FastAPI on port `8000`, so
Vite `allowedHosts` is not the correct control for deployed host validation.

**Decision**:
- Load deployment/runtime overrides from a repo-root `.env` file plus normal process
  environment variables, but do so explicitly in production entrypoints rather
  than in every `Settings()` construction.
- Allow `.env` to include both `HUMPBACK_*` runtime settings and deploy-time values
  like `TF_EXTRA`; the app settings loader ignores unknown keys.
- Add FastAPI bind settings `api_host` / `api_port` with defaults `0.0.0.0` / `8000`.
- Add FastAPI `TrustedHostMiddleware` driven by `HUMPBACK_ALLOWED_HOSTS`.
- Keep host validation permissive by default (`*`) so existing installs are not broken.
- Derive default extraction/cache paths from `storage_root` when the explicit path
  settings are unset.

**Consequences**:
- Deployment-specific paths and host allowlists no longer require tracked-file edits.
- Cloudflare tunnel deployments should use `HUMPBACK_API_HOST=0.0.0.0` and trusted
  host patterns like `*.trycloudflare.com`.
- Tests and library callers that instantiate `Settings()` directly remain
  hermetic and do not read deployment-local `.env` files from the cwd.
- Production host validation now lives in FastAPI; Vite `allowedHosts` remains a
  dev-server-only concern.
- No database migration is required because the change is limited to runtime
  configuration, deploy scripting, tests, and documentation.

---

## ADR-026: Positive extraction windows come from stored detection diagnostics

**Date**: 2026-03
**Status**: Accepted

**Context**: Retraining labels are saved at multi-second clip granularity, but classifier
training consumes 5-second embeddings. Blindly splitting a labeled-positive clip into fixed
5-second halves creates mislabeled positives when the vocalization only occupies part of the
clip. Re-running inference during extraction would duplicate work and can drift from the exact
detection-job scores that users reviewed.

**Decision**:
- Treat persisted 1-second-hop detection diagnostics as the source of truth for positive
  extraction window selection.
- For each positive labeled row (`humpback` or `orca`), smooth candidate 5-second window
  scores with a short moving average, select the peak smoothed window, and skip the row when
  the peak is below a configurable minimum score.
- Persist hydrophone diagnostics incrementally as Parquet shards so paused/canceled jobs can
  extract positives without re-running inference.
- Store row-level selection provenance back into the detection TSV via
  `positive_selection_*` columns plus `positive_extract_filename`.
- Keep classifier rescoring only as a legacy fallback for jobs missing diagnostics.

**Consequences**:
- Positive extraction is faster and stays consistent with the original detection-job scores.
- Hydrophone jobs gain durable per-window diagnostics even before full completion.
- Detection TSVs now carry both label state and positive-window provenance, and label-save
  must preserve those extra columns.
- No database migration required; the change lives in job artifacts, extraction config, API
  parsing, tests, and documentation.

---

## ADR-027: Positive extraction can widen beyond one 5-second window

**Date**: 2026-03
**Status**: Accepted

**Context**: ADR-026 improved positive extraction by selecting the best-scoring 5-second
window from stored 1-second-hop diagnostics, but some labeled rows still contain meaningful
vocalization beyond that single window. For longer calls, exporting only the peak 5-second
clip discards adjacent high-confidence audio that should stay in the training example.

**Decision**:
- Keep the ADR-026 seed-selection rule: select the best smoothed 5-second window and skip
  the row when its peak is below `positive_selection_min_score`.
- After selecting that seed, allow the extracted positive clip to widen by exact adjacent
  5-second chunks.
- Evaluate each adjacent chunk using the smoothed score of the aligned 5-second candidate
  window at that chunk start.
- Add an adjacent chunk only when its smoothed score is at or above the new
  `positive_selection_extend_min_score` threshold.
- If both sides qualify at once, extend the higher-scoring side first and then re-evaluate.
- Continue growth until neither adjacent chunk qualifies or the labeled-row boundary is hit.

**Consequences**:
- Positive extraction can now emit 10-second, 15-second, or longer clips when the score
  support justifies it, while still keeping durations as multiples of the classifier window.
- Existing provenance fields remain sufficient; widened clips are recorded through the
  selected `positive_selection_start_sec`, `positive_selection_end_sec`, and
  `positive_extract_filename`.
- Legacy rescoring fallback follows the same widening rule, so older jobs and new jobs
  behave consistently.
- No database migration required; the change is limited to extraction logic, config/API
  defaults, tests, and documentation.

---

## ADR-029: TF2 hydrophone detection runs in a short-lived subprocess

**Date**: 2026-03
**Status**: Accepted

**Context**: Recent hydrophone detection profiling showed that warm-cache
Orcasound runs using the `surfperch-tensorflow2` embedding backend slowed down
substantially after the long-lived worker had processed multiple TF2 jobs.
The profiled Orcasound Lab range was already fully populated in the disk-backed
write-through cache, so repeated S3 segment downloads were not the primary
bottleneck. The stronger signal was long-lived TensorFlow/Metal memory growth in
the worker process.

**Decision**:
- Keep the existing hydrophone detection workflow, archive-provider selection,
  progress callbacks, diagnostics persistence, and pause/resume/cancel behavior.
- When a hydrophone job resolves to a TF2 SavedModel embedding backend
  (`model_type="tf2_saved_model"`, `input_format="waveform"`), execute the
  hydrophone detection loop in a spawned subprocess instead of the long-lived
  worker process.
- Load the classifier pipeline and TF2 embedding model inside that child
  process, then communicate chunk progress, diagnostics, alerts, resume
  invalidation, and final results back to the parent worker over a queue.
- Keep TFLite hydrophone detection and local-file detection on the existing
  in-process path.
- Extend hydrophone run summaries with provider/runtime metadata:
  `provider_mode`, `execution_mode`, `avg_audio_x_realtime`,
  `peak_worker_rss_mb`, and `child_pid` when subprocess mode is used.

**Consequences**:
- TF2 hydrophone jobs release TensorFlow/Metal state when the child exits,
  preventing memory buildup from degrading later jobs in the long-lived worker.
- The parent worker remains the single owner of SQL status transitions and
  artifact persistence, so UI behavior for active/paused/canceled jobs stays
  consistent with the existing hydrophone workflow.
- Hydrophone run summaries now distinguish cache/provider mode from execution
  mode, making warm-cache vs runtime-memory regressions easier to diagnose.
- No database migration is required; the change is limited to worker
  orchestration, summary metadata, tests, and documentation.

---

## ADR-031: Windowed detection mode with NMS peak selection

**Date**: 2026-03
**Status**: Accepted

**Context**: Detection jobs produce variable-length detections (10-20+ seconds)
due to hysteresis merging of overlapping 1-sec-hop windows. Users must manually
review each long detection's spectrogram and select the best 5-sec sub-window
for positive extraction. This manual positive-selection step is the primary
bottleneck in the labeling workflow. The 1-sec hop is important for detection
sensitivity but the merging creates UX problems.

**Decision**:
- Add a `detection_mode` column to `detection_jobs` (nullable; `NULL`/"merged"
  preserves existing behavior, `"windowed"` enables the new mode).
- Windowed mode keeps the full pipeline (1-sec hop -> score -> hysteresis merge
  -> snap) for sensitivity, then applies NMS within each merged event to output
  only non-overlapping peak 5-sec windows above `high_threshold`.
- Long events with multiple distinct vocalizations produce multiple peak
  windows (NMS within each event), not just the single best.
- For windowed jobs, auto-positive-selection is trivially set to the full row
  bounds (the detection IS the positive window). The spectrogram editor's
  window-shifting controls are hidden.
- Extraction of windowed detections uses clip bounds directly — no
  `select_positive_window()` call needed.

**Consequences**:
- Labeling workflow for windowed jobs is just positive/negative — no sub-window
  selection needed.
- Each windowed detection produces exactly one training embedding (1:1 mapping
  between labeled detection and training vector).
- Existing merged-mode jobs are unaffected; `detection_mode=NULL` is treated as
  `"merged"`.
- Requires Alembic migration 018.

---

## ADR-032: Standard cosine similarity for cross-corpus embedding search

**Date**: 2026-03
**Status**: Accepted

**Context**: The existing `_cosine_similarity_matrix()` in `audio.py` uses mean-centered
cosine similarity, which removes the shared ReLU baseline direction and works well for
within-file pairwise comparison. For the new cross-corpus embedding search
(`POST /search/similar`), the corpus mean changes depending on which files are included,
making mean-centering unstable.

**Decision**: Use standard (non-mean-centered) cosine similarity for cross-corpus search.
Implement brute-force search over parquet files with an LRU cache (128 entries) for loaded
embeddings. Defer vector index (FAISS, USearch) and on-the-fly embedding to future phases.

**Alternatives considered**:
- Mean-centered cosine: unsuitable because the mean depends on corpus composition, making
  scores non-comparable across different search sets.
- Vector database (FAISS, USearch): unnecessary overhead at current scale (thousands to tens
  of thousands of embeddings); the search service is designed as a single substitution point
  if an index is needed later.
- On-the-fly embedding in the API process: conflicts with the architecture that isolates
  model loading to workers; deferred to Phase 1b.

**Consequences**:
- Search results use standard cosine similarity, which may differ from the within-file
  similarity matrix displayed in the audio detail view.
- The brute-force approach is O(N) in total embeddings; adequate at current scale but
  will need replacement if the corpus grows to millions of vectors.
- The LRU cache bounds memory usage while avoiding repeated parquet reads for hot sets.

---

## ADR-037: Annotation-guided synthesis with adaptive background threshold

**Date**: 2026-03
**Status**: Accepted

**Context**: The label processing synthesis pipeline had three related issues
causing poor training data quality:

1. **Shared-peak label contamination**: `isolate_call_segment()` centred audio
   extraction on the classifier peak's position, ignoring annotation bounds.
   When multiple nearby annotations of different call types shared a peak (64%
   of peaks in test data), they all got identical audio with different labels.

2. **No synthesis in dense recordings**: The fixed `background_threshold` (0.1)
   was too strict for recordings with elevated classifier baselines, causing
   `extract_background_regions()` to find zero qualifying runs.

3. **Repetitive backgrounds**: Even when backgrounds were found, all annotations
   in a recording cycled through the same small pool deterministically.

**Decision**: Three targeted changes to the synthesis pipeline:

- **Annotation-guided call isolation**: `isolate_call_segment()` accepts an
  optional `annotation` parameter.  When provided, the extracted segment centres
  on the annotation midpoint and uses the annotation duration (clamped to
  1-3 s), ensuring each annotation gets audio from its own labelled region.

- **Adaptive per-recording background threshold**: A new helper
  `_compute_adaptive_bg_threshold()` computes the 25th percentile of all
  smoothed scores, clamped to `[0.05, 0.5]`.  This replaces the static 0.1
  threshold when `background_threshold_auto=True` (default), allowing dense
  recordings to produce background segments from their quieter regions.
  Short runs (>= `background_min_duration`, default 1.0 s) are tiled to fill
  the 5 s synthesis canvas with up to 3 shifted variants per run.

- **Background pool rotation**: `synthesize_variants()` accepts a `bg_offset`
  parameter; `process_recording()` increments it per annotation so successive
  annotations start at different positions in the background pool.

**Consequences**:
- Synthesised files are now annotation-specific: filenames use
  `annotation.begin_time` instead of `peak.time_sec`.
- Dense recordings that previously produced zero synthesis output now produce
  backgrounds proportional to their quiet-region count.
- Two new configurable parameters: `background_threshold_auto` (bool) and
  `background_min_duration` (float).
- No database migration required — pure algorithm change.

---

## ADR-038: Sample builder contamination screening tuned for marine recordings

**Date**: 2026-03-21
**Status**: Accepted
**Context**: The sample builder pipeline rejected 100% of annotations (1514/1514) from real marine field recordings (Emily Vierling humpback dataset). Two root causes: (1) contamination screening thresholds designed for synthetic white noise at amplitude 0.001 failed on colored (pink/red) ocean ambient noise; (2) annotation duration bounds [0.3s, 4.0s] were too restrictive for the range of humpback vocalizations.

**Decision**: Four signal-processing algorithm changes:

1. **Tonal persistence: per-bin median threshold** — Changed `_tonal_persistence` from global median (across all bins and frames) to per-bin median (each bin compared to its own baseline). Pink noise has 20-30 dB more energy at low frequencies; the global median was pulled low by quiet high-frequency bins, causing all low-frequency bins to appear "persistently active." Per-bin median normalizes for spectral shape. Added configurable `persistence_margin_db` (default 10.0 dB) to `ContaminationConfig`. Trade-off: constant tones present throughout a fragment become invisible to persistence detection; the other three features (RMS, occupancy, transient) still catch loud or sudden contamination.

2. **Spectral occupancy: raised noise floor** — Changed defaults from `-40 dB / 0.3` to `-10 dB / 0.8`. At -40 dB, ocean ambient noise activated >99% of FFT bins. At -10 dB, typical marine backgrounds (amp 0.005-0.02) score 0.06-0.24 occupancy. Spectral occupancy has inherently poor separation for tonal contamination in colored noise (a tone adds 1-2 bins to 513), so it now serves as a broadband-only backstop.

3. **Validation: relaxed splice energy ratio and averaged spectral correlation** — Raised `splice_energy_ratio_max` from 10.0 to 1000.0 because background-to-call transitions inherently have large energy ratios (25-250x) that the crossfade smooths into gradual transitions, not audible artifacts. Changed `_spectral_correlation` from single-FFT to frame-averaged Welch-style power spectrum so spectral *shape* (e.g. 1/f) is compared rather than random per-frame fluctuations in short noise segments.

4. **Widened annotation duration bounds** — `SampleBuilderConfig` defaults changed from [0.3s, 4.0s] to [0.1s, 10.0s] to accommodate brief clicks and extended songs/moans.

All contamination and annotation config parameters are now exposed through the worker's job parameters for per-job tuning.

**Consequences**:
- Marine field recordings should achieve non-zero acceptance rates with default settings.
- Contamination detection is more permissive overall; users needing stricter screening can override via job parameters.
- Per-bin persistence cannot detect constant tones — accepted trade-off since constant recording-wide tones are effectively background.
- No database migration required — pure algorithm and default-value changes.

---

## ADR-039: Retire merged detection mode from the public creation/edit surface

**Date**: 2026-03-22
**Status**: Accepted

**Context**: Windowed detection mode solved the manual positive-selection bottleneck and became the operational default, but the product still exposed merged mode in the API, Hydrophone UI, and helper scripts. Keeping both modes on the public surface increased maintenance cost, preserved edit paths that only mattered for legacy jobs, and made it harder to backfill older merged outputs into the windowed workflow.

**Decision**:
- Remove `detection_mode` from detection-job creation requests in both local and hydrophone APIs; new jobs are always persisted as `"windowed"`.
- Reject create payloads that still send `detection_mode` instead of silently ignoring the obsolete field.
- Remove the Hydrophone UI mode selector and treat legacy merged jobs (`NULL` or `"merged"`) as read-only.
- Preserve legacy merged read paths (`GET` job/list, `/download`, `/content`) temporarily so historical jobs can still be inspected during manual backfill.
- Reject label-save, row-state, and extraction operations for legacy merged jobs with a rerun-in-windowed-mode error.
- Keep the `detection_mode` DB column for now so legacy rows remain distinguishable; defer schema cleanup until after manual backfill and artifact cleanup.

**Consequences**:
- The public detection workflow is now windowed-only.
- Hydrophone spectrogram-bound editing is effectively retired because it only applied to merged jobs.
- Legacy merged jobs remain visible for audit/download purposes but can no longer be modified or extracted.

---

## ADR-041: Adopt superpowers workflow, consolidate documentation

**Date**: 2026-03-24
**Status**: Accepted

**Context**: The project had 6 repo-root .md files with overlapping concerns and
6 custom session-* skills that duplicated superpowers functionality while missing
key capabilities (brainstorming, TDD enforcement, subagent execution, code review).

**Decision**: Adopt superpowers as the canonical workflow. Consolidate to 3 repo-root
files (CLAUDE.md, DECISIONS.md, AGENTS.md). Move specs to docs/specs/, plans to
docs/plans/. Rewrite AGENTS.md for Codex-compatible workflow.

**Consequences**:
- Single workflow system instead of two competing ones
- CLAUDE.md is larger (~450 lines) but self-contained
- Codex follows same phase sequence with its own tooling
- Session-* skills deleted; all workflow orchestration via superpowers
- Backlog items preserved in docs/plans/backlog.md

---

## ADR-042: Multi-label vocalization type classifier via binary relevance

**Date**: 2026-03-29
**Status**: Accepted

**Context**: The existing vocalization labeling system used single-label
classification on the binary classifier infrastructure. Windows can contain
multiple overlapping vocalization types (e.g., a whup during a moan), making
single-label assignment a poor fit. The annotation sub-window system added
complexity without adoption.

**Decision**: Replace the single-label vocalization classifier with a standalone
multi-label system using binary relevance (one independent sklearn pipeline per
vocalization type). Key design choices:

1. **Managed vocabulary** — `vocalization_types` table with unique names,
   importable from embedding set folder structure.
2. **Binary relevance** — N independent classifiers, one per type. A window
   labeled with types A and B is positive for both A and B pipelines, negative
   for neither. Types below `min_examples_per_type` are filtered out.
3. **Per-type threshold optimization** — each type gets an F1-maximized
   threshold from cross-validation, stored in the model and overridable at
   inference time.
4. **Dedicated tables** — `vocalization_types`, `vocalization_models`,
   `vocalization_training_jobs`, `vocalization_inference_jobs` — fully
   independent from the binary classifier tables.
5. **Three inference source types** — `detection_job` (with UTC), `embedding_set`
   (from curated sets), `rescore` (re-run previous results with a new model).
6. **Remove legacy systems** — dropped sub-window annotation system
   (`labeling_annotations` table) and the old single-label vocalization
   training endpoints from `/labeling/`.

**Consequences**:
- Windows can carry multiple type labels simultaneously
- Per-type thresholds allow precision/recall tuning per vocalization category
- Vocabulary is managed independently from training data
- Inference results are persistent parquet files, not ephemeral
- Old `/labeling/training-jobs`, `/labeling/vocalization-models`, `/labeling/predict`,
  and active learning endpoints removed; replaced by `/vocalization/` router
- `label_trainer.py` deleted; replaced by `vocalization_trainer.py`

---

## ADR-043: Exact replay promotion for AR-v1 autoresearch candidates

**Date**: 2026-04
**Status**: Accepted

**Context**: The candidate-backed promotion path blocked AR-v1 winners because the production trainer could not replay PCA (`pca_dim=128`), probability calibration (`prob_calibration="platt"`), or context pooling (`context_pooling="mean3"`). The existing promotion reduced autoresearch configs into a smaller parameter shape designed for manual embedding-set training, silently changing the candidate's learned behavior.

**Decision**: Extend candidate-backed promotion with an exact-replay path. A shared replay module (`src/humpback/classifier/replay.py`) extracted from `scripts/autoresearch/train_eval.py` owns the full training pipeline (context pooling, feature normalization, PCA, classifier, calibration). Both autoresearch scripts and the production worker import from this shared module. Candidate-backed training uses `promoted_config` directly through the replay module, bypassing the legacy `train_binary_classifier` path. Replay verification compares produced metrics against imported candidate metrics with configurable tolerance.

**Consequences**:
- PCA, probability calibration (platt/isotonic), and non-center context pooling (mean3/max3) are now promotable
- Autoresearch and production cannot drift — both call the same shared code
- Replay verification proves parity with the reviewed candidate on val/test splits
- Verification mismatch does not fail the job — the model is saved but flagged
- Calibration is baked into the sklearn Pipeline; `predict_proba()` returns calibrated probabilities directly
- `linear_svm` and `hard_negative_fraction > 0` remain blocked; MLP with explicit class weights is now supported via `sample_weight`

---

## ADR-044: Prominence-based window selection for detection

**Date**: 2026-04-04
**Status**: Accepted

**Context**: In windowed detection mode, NMS selects non-overlapping 5-second peak windows from merged events by suppressing all windows within `window_size_seconds` of each selected peak. This creates 2–4 second gaps between detection items, even in regions where the classifier scores every window above 0.9. Distinct vocalizations falling in these gaps are missed for both labeling and training data. Investigation of job `d52e03cc` confirmed the classifier scores 0.95–0.997 at 02:16:53–56 UTC, but no detection item is emitted because it falls in the NMS gap between adjacent selected peaks.

**Decision**: Add a `window_selection` parameter to detection jobs with two modes:
- `"nms"` (default) — existing greedy NMS with `window_size_seconds` suppression zone, non-overlapping output.
- `"prominence"` — peak prominence detection in logit (log-odds) space. Raw confidence scores are transformed to logits (`ln(p/(1-p))`) before peak finding and prominence computation. This amplifies meaningful dips in high-confidence regions where probability scores saturate near 1.0 (e.g., a dip from 0.999 to 0.983 is 0.016 in probability but 2.15 in logit units). Peaks passing `min_prominence` (default 1.0, in logit units) and `min_score` emit 5-second windows that may overlap. A fallback emits the single highest window when no peaks pass prominence (e.g., true plateau regions).

Raw scores are used (no smoothing) to preserve the true dip depth between vocalizations. Smoothing was found to shift peak positions in short sequences, creating inconsistencies between smoothed peak locations and raw prominence values.

**Consequences**:
- Every detected vocalization event gets at least one detection item; dense singing regions produce overlapping windows covering each distinct call
- Overlapping detection windows are compatible with all downstream systems (row store, embeddings, labeling, extraction, training datasets) because they are keyed by `row_id`, not time-range uniqueness
- Requires Alembic migration 038 (`window_selection`, `min_prominence` columns on `detection_jobs`)
- NMS remains the default; prominence mode is opt-in via UI toggle or API parameter
- Side-by-side comparison is possible by running the same time range with each mode

**Update (2026-04-04): Recursive gap-filling**

Prominence correctly rejects peaks with low prominence, but this misses strong vocalizations in flat score regions between adjacent equal-strength detections (e.g., confidence 0.988 with only 0.05 logit prominence). The score curve is genuinely flat — no score transformation helps.

After prominence peak selection, a recursive gap-filling pass scans for gaps > 5.0 seconds between consecutive selected peaks (and from event edges to the nearest peak). For each gap, the candidate closest to the gap midpoint (with score as tiebreaker) above `min_score` is emitted, splitting the gap into two sub-gaps that are checked recursively. Midpoint-first placement prevents fills from clustering next to existing peaks where scores are naturally highest, producing evenly spaced coverage instead. The 5.0-second threshold (matching `window_size_seconds`) was chosen after testing showed that 3.0 seconds produced 2-second spacing between windows — 60% overlap with minimal new coverage. This is always-on in prominence mode; the threshold is hardcoded (not an API parameter).

---

## ADR-045: Tiling window selection for detection

**Date**: 2026-04-04
**Status**: Accepted

**Context**: Both NMS and prominence-based window selection are peak-centric — they identify discrete score peaks then (in prominence's case) retroactively fill gaps between them. Flat plateau regions where the classifier scores 0.99 across many consecutive windows have no peaks or prominence to exploit. Gap-filling patches this with midpoint insertion, but it is an after-the-fact heuristic bolted onto a peak-finding algorithm.

**Decision**: Add a third `window_selection` mode called `"tiling"` that treats high-scoring regions as spans to cover rather than peaks to find:

1. Within each event, collect candidates above `min_score` and compute logit scores.
2. **Multi-pass**: while uncovered candidates remain, pick the highest-scoring uncovered window as seed, tile left and right through consecutive candidates while `seed_logit - candidate_logit <= max_logit_drop`, mark tiled windows as covered.
3. Stop tiling at the first window where the drop exceeds the threshold — no look-ahead. Recovered regions become their own seeds in subsequent passes.

`max_logit_drop` (default 2.0, in logit units) controls tiling extent. Requires Alembic migration 039.

**Consequences**:
- Plateau regions get full contiguous coverage without gap-filling heuristics
- Multi-pass naturally segments distinct vocalizations separated by score drops without prominence computation
- Every above-threshold window is either tiled from a seed or becomes a seed itself — exhaustive coverage
- Overlapping output windows are compatible with all downstream systems (same as prominence mode)
- NMS remains the default; tiling and prominence are opt-in via UI or API parameter

---

## ADR-046: Promote `linear_svm` candidates from hyperparameter tuning

**Date**: 2026-04-10
**Status**: Accepted
**Supersedes aspect of**: ADR-043

**Context**: ADR-043 shipped exact-replay promotion for AR-v1 candidates but left `linear_svm` on the blocked list because no reviewed research candidate justified the added surface area at the time. The hyperparameter tuning page search space (`DEFAULT_SEARCH_SPACE`) already includes `linear_svm`, so searches run end-to-end and produce winning trials. One such trial (candidate `s-v2 (imported)`) is now worth promoting, and the error message `classifier='linear_svm' is not supported by the production trainer` blocks it at import time.

The runtime replay path already supports `linear_svm`: `replay.build_classifier` wraps `LinearSVC` in `CalibratedClassifierCV(cv=3, method="sigmoid")` to expose `predict_proba`, `apply_calibration` short-circuits for SVMs (already calibrated), and the candidate-backed training worker uses `build_replay_pipeline` directly via `source_mode == "autoresearch_candidate"`. The only thing preventing promotion is a two-line allowlist in `_assess_reproducibility` plus a matching branch in `map_autoresearch_config_to_training_parameters` (used for display metadata only).

**Decision**: Add `linear_svm` to the promotable-classifier allowlist. Detection code (`detector.py`, `hydrophone_detector.py`) is already classifier-agnostic because it calls `predict_proba` on the saved sklearn pipeline. `hard_negative_fraction > 0` remains blocked — that path would require new data-resampling logic the replay module does not own.

**Consequences**:
- Linear SVM candidates from the tuning page are now promotable via the existing candidate import → promotion flow
- No schema change, no migration, no worker change, no detector change
- The Classifier → Training tab still offers only Logistic Regression and MLP for direct training; linear_svm remains tuning-only
- A new frontend label ("SVM" / "Linear SVM") surfaces on promoted models
- Replay verification still gates promotion quality; the cv=3 `CalibratedClassifierCV` wrapping is deterministic under a fixed seed, which a new `test_linear_svm_pipeline_deterministic` unit test explicitly guards
- `hard_negative_fraction > 0` remains deferred until a research candidate justifies the data-resampling work

---

## ADR-047: Replace heuristic gain-step detection with PCEN timeline normalization

**Date**: 2026-04-10
**Status**: Accepted
**Supersedes**: the gain-normalization approach shipped in PR #82

**Context**: The gain normalization feature shipped as a ~350-line heuristic step detector that walked the job's audio in 1-second RMS windows, grouped high-gain regions into segments via a median-plus-threshold rule, and attenuated them before STFT and before MP3 encoding. It was wired in tandem with a per-job `ref_db` pre-pass that used `max()` of sampled tile power stats to pick the colormap ceiling. In practice the result had several persistent pain points:

- Two per-job pre-passes and two cache sidecars (`.gain_profile.json`, `.ref_db.json`) that had to be managed and invalidated.
- Step detection is fragile — thresholds, min-duration filtering, internal-drop splitting, and boundary refinement all need tuning.
- It only handles discrete gain jumps; in-band noise that buries calls is never addressed.
- Even after gain correction, residual bright content pulled `ref_db` up via `max()` and the 80 dB dynamic range painted most of the spectrum near black — the rendered tiles were consistently "too dark".
- Visual and audio were forced through the same correction pipeline even though they have different ideal normalizations.

**Decision**: Replace the entire gain-normalization + `ref_db` pipeline with two independent, well-understood normalizations, one per output path:

- **Spectrogram rendering**: Apply `librosa.pcen` (Per-Channel Energy Normalization, Lostanlen et al.) to the STFT magnitude in `generate_timeline_tile`. PCEN is the field-standard bioacoustic AGC (Google Perch, BirdNET, whale pipelines). Each tile renders independently from its own audio fetch plus a configurable warm-up prefix (`pcen_warmup_sec`, default 2 s) so the filter can settle; the warm-up frames are trimmed off the output. PCEN output is bounded, so tiles use a fixed colormap range (`pcen_vmin=0.0`, `pcen_vmax=1.0`) — no `ref_db`, no per-job pre-pass.
- **Audio playback**: A new `normalize_for_playback` helper in `audio_encoding.py` scales each chunk's RMS to `playback_target_rms_dbfs` (default −20 dBFS) and soft-clips with `tanh` at `playback_ceiling`. Used by both the `/audio` endpoint and the timeline export audio chunk loop.

Scope is deliberately limited to the timeline viewer and export. The detection / classifier feature extraction in `features.py` is intentionally untouched; extending PCEN to classifier training is a separate investigation.

**Consequences**:
- `src/humpback/processing/gain_normalization.py` (~350 lines) and its test file deleted.
- `_compute_job_ref_db`, `_apply_gain_correction`, `_compute_job_gain_profile`, and the Pass 0 of `_prepare_tiles_sync` all removed from `src/humpback/api/routers/timeline.py`.
- `.gain_profile.json` and `.ref_db.json` no longer produced; a new `.cache_version` marker plus `TimelineTileCache.ensure_job_cache_current` handle migration of existing job caches on first access (one-shot delete of stale tiles and legacy sidecars).
- New config knobs: `pcen_time_constant_sec`, `pcen_gain`, `pcen_bias`, `pcen_power`, `pcen_eps`, `pcen_warmup_sec`, `pcen_vmin`, `pcen_vmax`, `playback_target_rms_dbfs`, `playback_ceiling`.
- `gain_norm_threshold_db`, `gain_norm_min_duration_sec`, and `timeline_dynamic_range_db` removed from `Settings`.
- Fixes a latent inconsistency in `timeline_export.py`: previously the exported MP3s bypassed gain correction entirely (shipping raw audio while exported tiles were gain-corrected). Post-refactor both viewer and export route through the same `normalize_for_playback` helper.
- Visual and audio are no longer bit-exact matched, which is intentional — PCEN is wrong for listening and a compressor is wrong for visualization.
- PCEN uses magnitude input (not power); librosa's docs are ambiguous on this, but power input produces runaway values with gain<1 and is empirically unsuitable. See inline comment in `pcen_rendering.py`.
- Path open to extend PCEN to the detection pipeline in a future session as an opt-in `features.py` normalization mode, should classifier SNR benefit materialize.

**Update (2026-04-10 — cold-start fix and parameter retune)**: The initial shipped values (`time_constant=0.5`, `bias=10.0`, `power=0.25`, `vmax=0.15`) produced three user-visible defects: (a) a dark vertical strip at every tile boundary, (b) a grainy appearance at all zoom levels, and (c) insufficient feature contrast at coarse zoom levels where no whale calls were visible. Root cause: `librosa.pcen`'s default initial filter state is scaled by `lfilter_zi(b, [1, b-1])`, which corresponds to the steady-state response for a unit-amplitude DC input. Real hydrophone STFT magnitudes are orders of magnitude smaller, so the per-bin low-pass filter decays exponentially from ~1 toward the actual signal level over many frames. At coarse zoom levels the 2-second warm-up covers only a handful of frames, so the first several columns of every tile sit well below the steady-state PCEN output and render nearly black. Fix: `pcen_rendering.py` now pre-scales `scipy.signal.lfilter_zi` by the first STFT frame's magnitude (per frequency bin) before passing it as the `zi` argument to `librosa.pcen`. This places each bin's filter at its own settled level from the first frame, removing the cold-start transient entirely. Parameters were simultaneously retuned to librosa-aligned defaults (`bias=2.0`, `power=0.5`, `time_constant=2.0`) with `vmax=1.0`, which gives a useful dynamic range at all zoom levels where a 1–3 s whale call produces a visible brighter spike above the noise floor.
