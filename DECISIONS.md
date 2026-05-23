# Architecture Decision Log

Record of significant design decisions with non-obvious reasoning.
Original numbering preserved; removed entries described current code behavior
without unique insight and were pruned in the documentation consolidation (2026-03-30).

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
   originally importable from embedding-set folder structure. Update
   2026-04-29: the embedding-set import path is retired; vocabulary is now
   managed directly through retained vocalization workflows.
2. **Binary relevance** — N independent classifiers, one per type. A window
   labeled with types A and B is positive for both A and B pipelines, negative
   for neither. Types below `min_examples_per_type` are filtered out.
3. **Per-type threshold optimization** — each type gets an F1-maximized
   threshold from cross-validation, stored in the model and overridable at
   inference time.
4. **Dedicated tables** — `vocalization_types`, `vocalization_models`,
   `vocalization_training_jobs`, `vocalization_inference_jobs` — fully
   independent from the binary classifier tables.
5. **Three inference source types** — originally `detection_job` (with UTC),
   `embedding_set` (from curated sets), and `rescore` (re-run previous results
   with a new model). Update 2026-04-29: retained sources are now
   `detection_job` and `rescore`; `embedding_set` inference was retired in the
   legacy workflow removal.
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

---

## ADR-048: Four-pass call parsing pipeline (Phase 0 scaffold)

**Date**: 2026-04-11
**Status**: Accepted

**Context**: The existing detection pipeline produces fixed-size 5-second windows tagged with a binary whale/non-whale score. Downstream research increasingly wants event-level parsing — individual vocalizations with precise onset/offset bounds, per-event multi-label call-type classification, and time-ordered sequence export for sequence/grammar studies. None of that is derivable from 5-second window scores, and bolting it onto `run_detection` would entangle three very different tasks (dense scoring, framewise segmentation, per-event classification) behind a single job type.

**Decision**: Introduce a four-pass chained pipeline with one parent run and three independently-queueable child job types, each owning one pass:

1. **Pass 1 — Region detection** (`region_detection_jobs`): dense Perch inference + existing hysteresis merge + padding. Reuses the new `compute_hysteresis_events` helper factored out of `detector.py`. Produces `trace.parquet` + `regions.parquet`.
2. **Pass 2 — Event segmentation** (`event_segmentation_jobs`): framewise PyTorch segmentation model (CRNN or transformer, TBD) decodes onset/offset per event inside each region. Produces `events.parquet`.
3. **Pass 3 — Event classification** (`event_classification_jobs`): per-event multi-label PyTorch CNN operating on variable-length event crops. Lives under the existing `vocalization_models` table via a new `model_family="pytorch_event_cnn"` + `input_mode="segmented_event"` axis so the legacy sklearn family coexists. Produces `typed_events.parquet`.
4. **Pass 4 — Sequence export**: read Pass 3 output and emit a sorted, merged sequence via `GET /call-parsing/runs/{id}/sequence`. No new table; pure derivation.

A parent `CallParsingRun` row threads the four passes, and each child job can also be queued directly so researchers can rerun a single pass with alternate parameters without re-running the full chain (individual-runnable contract).

Phase 0 ships the full scaffold only — tables, worker shells that claim-and-fail, stub endpoints that return 501 for anything requiring pass logic — plus the behavior-preserving detector refactor that extracts `compute_hysteresis_events` for reuse by Pass 1. Passes 1–4 each brainstorm their own internal design (model architecture, labels, training data, decoding) against this fixed contract.

**Consequences**:
- Migration 042 adds five tables: `call_parsing_runs`, `region_detection_jobs`, `event_segmentation_jobs`, `event_classification_jobs`, `segmentation_models`. Extends `vocalization_models` and `vocalization_training_jobs` with `model_family` / `input_mode` columns (backfilled to `sklearn_perch_embedding` / `detection_row`).
- PyTorch added to every `tf-*` extra in `pyproject.toml`. `uv sync --group dev --extra tf-macos` now installs torch alongside tensorflow-macos.
- New `src/humpback/ml/` harness (`device`, `training_loop`, `checkpointing`) shared by Passes 2 and 3 so each pass only defines its model + data loaders.
- New `src/humpback/call_parsing/` package holds the cross-pass dataclasses (`Region`, `Event`, `TypedEvent`, `WindowScore`), pyarrow schemas, and atomic parquet I/O helpers. Each pass writes its own per-job parquet files and references previous-pass rows by stable UUID (`region_id`, `event_id`) — the same indirection pattern vocalization labels already use against detection `row_id`.
- `detector.py` is refactored to route its inner loop through `_run_window_pipeline`, which is the backbone for both `run_detection` (unchanged public API) and the new `compute_hysteresis_events` public helper. A committed snapshot fixture (`tests/fixtures/detector_refactor_snapshot.json`) guards bit-identical output on a deterministic audio + classifier combination.
- Three new worker types registered in the main dispatcher between `vocalization_inference` and `manifest_generation`. Phase 0 workers claim atomically and mark the job `failed` with a message naming the owning Pass.
- Phase 0 API: `POST/GET/DELETE /call-parsing/runs*` functional (DB-backed). Per-pass creation endpoints and every artifact access endpoint return 501 with a detail message naming the owning Pass.
- Per-pass specs and implementation plans deferred to dedicated sessions: `docs/plans/2026-04-11-call-parsing-pass1-region-detector.md`, `pass2-segmentation`, `pass3-event-classifier`, `pass4-sequence-export`.

---

## ADR-049: Call parsing Pass 1 — algorithmic defaults and streaming architecture

**Date**: 2026-04-11
**Status**: Accepted
**Builds on**: ADR-048 (four-pass call parsing scaffold)

**Context**: Phase 0 shipped the Pass 1 worker shell as a claim-and-fail stub with a `compute_hysteresis_events` helper extracted from `detector.py`. Pass 1 turns that shell into a real worker that accepts one audio source (uploaded file or hydrophone time range), runs dense Perch inference, shapes the hysteresis events into padded whale-active regions, and writes `trace.parquet` + `regions.parquet` under the per-job storage directory. Pass 2's segmentation model consumes `regions.parquet` to decide where to crop variable-length event windows.

Three structural constraints shape the design:

- **Hydrophone is first-class.** Pass 1 will routinely run over 24-hour continuous hydrophone ranges. A 24 h buffer at 16 kHz float32 is ~5.5 GB — loading the full range into memory is not an option, so the worker must stream audio through Perch inference chunk-by-chunk while producing a single concatenated trace.
- **The Phase 0 `compute_hysteresis_events` helper is monolithic.** It takes a flat `audio: np.ndarray` and returns both per-window scores and hysteresis events in one call. Pass 1 needs to run the scoring loop repeatedly across chunks but run hysteresis only once on the concatenated trace.
- **Region shaping is a pure transformation.** Turning hysteresis events into padded, merged regions is deterministic on event dicts + audio duration + config — no audio, no models, no I/O. It belongs in its own module.

**Decision**:

- **Symmetric `padding_sec=1.0` default.** One second on each side of a hysteresis-merged event gives Pass 2 enough context to find real onset/offset without guessing an asymmetric distribution ahead of data. If post-event decay tails prove under-covered in practice, add asymmetric padding as an opt-in knob later.
- **Merge on padded-bounds overlap, no separate `merge_gap_sec`.** Padding is the single control. Two calls close enough that their padded bounds overlap are fused into one region that Pass 2's segmenter can split internally; farther-apart calls stay separate. This removes a redundant knob the prominence/tiling detectors had to grow.
- **No temporal smoothing on scores.** Hysteresis's 0.25-probability dead zone (`0.70` / `0.45` defaults) already protects against single-window dips, the Perch 5 s window / 1 s hop already smooths embeddings in time by construction, smoothing would shift peak positions in short sequences (the same issue ADR-044 called out for prominence), and `trace.parquet` is a data product that consumers want raw.
- **`min_region_duration_sec=0.0` default.** Pass 2 is the real noise rejector; Pass 1's job is high-recall. The knob exists in `RegionDetectionConfig` for follow-up tuning but defaults to off.
- **Dense raw trace with no decimation.** ~1 MB of parquet per day of audio is cheap, unlocks a future "re-decode without re-running Perch" endpoint, and preserves observability for debugging threshold choices. Decimation is a premature optimization that throws away information needed for exactly the kind of post-hoc threshold sweep this pipeline is being built to enable.
- **Delete-and-restart crash semantics.** On worker exception the partial `trace.parquet` / `regions.parquet` / `.tmp` sidecars are deleted and the row flips to `failed`. Matches the rest of the codebase. Partial-trace resume paths are rarely exercised and tend to rot; revisit only if a multi-hour Pass 1 job actually becomes a pain point.
- **`score_audio_windows` / `merge_detection_events` split and chunk-aligned streaming.** `compute_hysteresis_events` is re-implemented as a thin two-call composition: `score_audio_windows(audio, sr, perch, classifier, config, time_offset_sec=0.0)` emits dict-shaped per-window records with shifted `offset_sec` / `end_sec`, and `merge_detection_events` runs once over the concatenated trace. The hydrophone path computes chunk edges as whole multiples of `window_size_seconds` via `_aligned_chunk_edges` so no Perch window ever straddles a chunk boundary. The file-source path falls out as the degenerate "one chunk" case that runs `score_audio_windows` once on the full buffer. A chunk-concatenation unit test asserts that the streaming path is bit-identical to a single-buffer call on the same audio, and the existing `detector_refactor_snapshot.json` fixture guards that `compute_hysteresis_events` still returns what it did before the split.

**Consequences**:

- Migration `043_call_parsing_pass1_source_columns.py` drops the Phase 0 `audio_source_id` placeholder on both `call_parsing_runs` and `region_detection_jobs` and adds `audio_file_id` / `hydrophone_id` / `start_timestamp` / `end_timestamp`. `downgrade()` restores `audio_source_id` as nullable (not NOT NULL) to avoid requiring a backfill. Exactly-one-of (`audio_file_id`) vs the hydrophone triple is enforced in the Pydantic request model and service layer, not via a DB CHECK — matching the existing `DetectionJob` pattern.
- `src/humpback/call_parsing/regions.py` holds the pure `decode_regions(events, audio_duration_sec, config)` function. Pure — no I/O, no audio, no models — so it unit-tests exhaustively against fused-pair / gap / boundary-clamp / weighted-mean-score / min-duration-filter cases.
- `src/humpback/classifier/detector.py` exposes `score_audio_windows` as a public helper; `compute_hysteresis_events` becomes a two-call composition `score_audio_windows(...) + merge_detection_events(records, high, low)` with public signature and return shape unchanged.
- `src/humpback/workers/region_detection_worker.py` claims the job via the shared compare-and-set pattern, resolves the audio source (file path via `AudioLoader`, hydrophone via the archive playback provider + `iter_audio_chunks`), runs the scoring loop once (file) or across `_aligned_chunk_edges` chunks (hydrophone) with `time_offset_sec = chunk_start - range_start`, runs hysteresis + `decode_regions` once, writes both parquet files via the atomic helpers in `call_parsing.storage`, and updates `trace_row_count` / `region_count` / `completed_at` / `status`. On exception it deletes partial artifacts and marks the row `failed` with the exception message.
- `src/humpback/workers/queue.py` — `RegionDetectionJob` joins the stale-job recovery sweep so a killed worker's row resets to `queued` after the existing timeout.
- `src/humpback/api/routers/call_parsing.py` — `POST /call-parsing/region-jobs`, `GET /region-jobs/{id}/trace`, and `GET /region-jobs/{id}/regions` are now functional. Trace and regions endpoints return 409 while the job is not `complete`, 404 when the parquet file is missing. `POST /call-parsing/runs` accepts the same `CreateRegionJobRequest` shape and creates the parent run + Pass 1 child atomically via `create_parent_run`.
- No Pass 2 / Pass 3 / Pass 4 logic ships here. No frontend UI. No hydrophone-path worker integration test (deferred — no `ArchivePlaybackProvider` mock surface yet; correctness of the streaming loop is covered by the chunk-concatenation unit test instead, and the deferred test lives in `docs/plans/backlog.md`).
- `RegionDetectionConfig` hyperparameters live in the existing Phase 0 `config_json` TEXT column serialized from `model_dump_json()`. No per-knob columns — Pass 1 is brand-new code that will iterate on these defaults and the existing `DetectionJob` table's per-knob columns pay an ongoing migration cost every time a new hyperparameter ships.

## ADR-050: Call parsing Pass 2 — bootstrap-era design decisions

**Date**: 2026-04-11
**Status**: Accepted
**Builds on**: ADR-047 (PCEN timeline normalization), ADR-048 (four-pass scaffold), ADR-049 (Pass 1 region detector)

**Context**: Pass 2 turns the Phase 0 event segmentation stub into a working training + inference pipeline: a PyTorch CRNN that learns framewise humpback call presence from labeled detection rows, with a hysteresis decoder that produces `events.parquet` from `regions.parquet`. The bootstrap script is the first data pathway into the training dataset tables — future pathways include a UI timeline-editor that produces the same `SegmentationTrainingSample` rows from a visual annotation workflow. Six design decisions need to be locked now because they affect the data contract, the model architecture, and the table schema.

**Decision**:

- **Framewise α supervision target.** Binary presence inside/outside event bounds, masked `BCEWithLogitsLoss` with auto `pos_weight`. Bootstrap row bounds are too loose for onset/offset point targets, and α is the simplest thing that gives the hysteresis decoder a clean frame probability vector to consume. γ (ignore band at edges) is a cost-free loss-swap upgrade if α's edge calibration proves bad.
- **No PCEN for Pass 2 features.** Per-region z-score is the default normalization for the Pass 2 log-mel spectrogram. ADR-047 already opened the door for extending PCEN to the classifier feature pipeline if SNR demands; Pass 2 deliberately does not walk through that door yet — simpler first, upgrade when data demands it.
- **Per-audio-file train/val split.** Per-sample random splits leak background noise signature in bioacoustic ML. The mandatory audio-source-disjoint split in `split_by_audio_source` groups samples by `audio_file_id` (or `hydrophone_id` for future hydrophone-sourced samples) and ensures no audio source appears in both train and val. This rule is unconditional even on small bootstrap datasets.
- **Persistent training dataset contract.** `segmentation_training_datasets` + `segmentation_training_samples` are designed for both the one-shot bootstrap script and the future UI timeline-editor workflow. The bootstrap is the first writer, not the only writer. The schema requires no migration when the UI extension lands — `SegmentationTrainingSample` already carries the generic `source` / `source_ref` columns the editor needs.
- **CRNN at ~300k parameters as the first shipping architecture.** Field-standard inductive bias for bioacoustic framewise segmentation (Conv2d stack for spectrotemporal features, bidirectional GRU for temporal context, frame-head Linear for per-frame logits), data-efficient relative to a transformer, produces a clean frame head for the hysteresis decoder. The `segmentation_models.model_family` column is the extension hook for when a different architecture earns its place.
- **Separate `segmentation_training_jobs` table.** Framewise regression is a different task from Pass 3's per-event multi-label classification; keeping the training-job table separate lets the trainer code stay cleanly isolated and avoids growing yet another `model_family` branch on `vocalization_training_jobs`. The table lives in migration `044_segmentation_training_tables.py` alongside the dataset and sample tables.

**Consequences**:

- Migration `044_segmentation_training_tables.py` creates `segmentation_training_datasets`, `segmentation_training_samples`, and `segmentation_training_jobs` with indexes on `training_dataset_id` and a composite index on `(training_dataset_id, source_ref)` for the bootstrap script's idempotency check. `downgrade()` drops all three tables and their indexes.
- `src/humpback/call_parsing/segmentation/` is a new subpackage with five modules: `features.py` (log-mel + z-score), `model.py` (SegmentationCRNN), `decoder.py` (hysteresis events), `dataset.py` (framewise target builder + PyTorch Dataset), `trainer.py` (train driver + event matching + final eval), and `inference.py` (per-region forward pass).
- `src/humpback/workers/segmentation_training_worker.py` claims `segmentation_training_jobs` rows, runs `train_model` on a background thread, saves the checkpoint under `storage_root/segmentation_models/<model_id>/`, and registers a `segmentation_models` row.
- `src/humpback/workers/event_segmentation_worker.py` claims `event_segmentation_jobs` rows, loads the model checkpoint, resolves the audio source from the upstream Pass 1 job's columns (not its own), runs per-region inference, and writes `events.parquet` atomically.
- `scripts/bootstrap_segmentation_dataset.py` reads row IDs from a file, resolves vocalization labels and audio sources, computes audio-relative crop windows, and inserts `SegmentationTrainingSample` rows with idempotency checking.
- The worker priority order in `queue.py` places `segmentation_training` between `vocalization_inference` and `region_detection`, and `event_segmentation` after `region_detection` — matching the pipeline's natural flow.
- No hydrophone-sourced training samples or inference in this pass. The training worker and event segmentation worker raise `NotImplementedError` on the hydrophone path; the deferred integration test lives in `docs/plans/backlog.md`.

## ADR-051: Call parsing Pass 3 — event classifier architecture

**Date**: 2026-04-11
**Status**: Accepted
**Builds on**: ADR-048 (four-pass scaffold), ADR-050 (Pass 2 segmentation)

**Context**: Pass 3 turns the Phase 0 event classification stub into a working training + inference pipeline. A PyTorch CNN classifies variable-length event crops (produced by Pass 2) into vocalization types from the existing vocabulary. The model coexists with the sklearn per-window vocalization classifier via the `model_family` / `input_mode` columns on `vocalization_models`. Five design decisions lock the architecture, data strategy, and coexistence model.

**Decision**:

- **Frequency-only pooling in the CNN.** `MaxPool2d((2,1))` after each of the four Conv2d/BN/ReLU blocks pools the mel-frequency axis by 2× per block (64→4 after 4 blocks) while preserving the time dimension entirely. This is critical because humpback events span 0.2–5 seconds (~6–156 time frames), and standard 2×2 pooling would collapse short events to a single frame or zero before reaching the classifier head. `AdaptiveAvgPool2d((1,1))` handles the final variable-length collapse.
- **Variable-length crops with batch padding.** Events are cropped at their exact Pass 2 boundaries — no fixed-length windowing, no context padding. The custom `collate_fn` pads spectrograms to max-T within each batch. Zero-padded frames contribute near-zero values post z-score normalization, diluting slightly but not corrupting the adaptive-average-pooled representation. This is simpler than bucketed sampling and adequate for the small batch sizes typical of bootstrapped bioacoustic datasets.
- **Model family coexistence via `vocalization_models`.** Pass 3 reuses the `vocalization_models` table with `model_family='pytorch_event_cnn'` and `input_mode='segmented_event'`, rather than creating a separate model registry. The vocalization training worker dispatches on `model_family` at the top of its training path. Existing sklearn endpoints only accept `sklearn_perch_embedding` models; the `pytorch_event_cnn` family is only usable through the call parsing classification job flow. No new migration needed — migration 042 already added the `model_family` and `input_mode` columns.
- **Bootstrap data strategy: single-label windows only.** The bootstrap script transfers vocalization type labels from detection windows to Pass 2 events, but only for windows with exactly one vocalization type label (excluding `(Negative)`). Multi-label windows are excluded because a 5-second detection window may contain multiple events of different types — assigning all types to all contained events would inject label noise. This conservative filter trades dataset size for label accuracy, which is the right trade-off for a bootstrap-era model.
- **Per-type threshold optimization on validation set.** After training, per-type classification thresholds are swept on the validation set to maximize per-type F1. Thresholds are stored in `thresholds.json` and applied at inference time to populate `above_threshold` on each `TypedEvent` row. This matches the existing sklearn vocalization model pattern (ADR-042) and decouples the score calibration from the model weights.

**Consequences**:

- `src/humpback/call_parsing/event_classifier/` is a new subpackage with four modules: `model.py` (EventClassifierCNN), `dataset.py` (EventCropDataset + collate_fn), `trainer.py` (training driver + threshold optimization), and `inference.py` (load + classify).
- `src/humpback/workers/event_classification_worker.py` replaces the Phase 0 stub with full inference: loads model from `vocalization_models` row, reads `events.parquet` from upstream Pass 2 job, resolves audio transitively from Pass 1, runs batch CNN inference, writes `typed_events.parquet` atomically.
- `src/humpback/workers/vocalization_worker.py` gains a `model_family` dispatcher: `sklearn_perch_embedding` (unchanged) vs `pytorch_event_cnn` (new event classifier trainer). Worker priority order is unchanged.
- `scripts/bootstrap_event_classifier_dataset.py` reads detection job IDs, filters to single-label vocalization-labeled windows, runs Pass 2 segmentation on each window to discover events, transfers labels, and writes a JSON file of training samples. Idempotent via existing-output deduplication.
- API endpoints `POST /classification-jobs` and `GET /classification-jobs/{id}/typed-events` are fully functional (previously 501 stubs). POST validates model family (422) and upstream completion (409).
- No new database migration — Pass 3 uses existing tables from migrations 042 (call parsing scaffold) and the `vocalization_models` extensions.

## ADR-052: Chunk artifact system applies to hydrophone path only

**Date**: 2026-04-12
**Status**: Accepted
**Supersedes**: ADR-049 "delete-and-restart crash semantics" for hydrophone jobs

**Context**: Multi-hour hydrophone region detection jobs (24h range ≈ 48 chunks at 30 min each) produce no intermediate output, cannot be resumed after a crash, and provide no progress visibility. ADR-049 noted "revisit only if a multi-hour Pass 1 job actually becomes a pain point" — it has.

**Decision**: The chunk artifact system (per-chunk parquet files, manifest.json, DB progress columns, resume logic) applies only to the hydrophone streaming path. File-based region detection continues as a single atomic operation with no intermediate artifacts.

**Consequences**:

- Migration `045_region_detection_progress_columns.py` adds `chunks_total` and `chunks_completed` nullable integer columns to `region_detection_jobs`. Both remain NULL for file-based jobs.
- The hydrophone worker writes `manifest.json` at job start with all chunks in `pending` status, then writes `chunks/{0000..N}.parquet` per completed chunk with atomic tmp-rename. DB `chunks_completed` is incremented after each chunk commit.
- On resume (re-queued failed job), the worker reads the existing manifest, verifies each "complete" chunk has a parquet file on disk, resets any without, and continues from the first pending chunk.
- `_cleanup_partial_artifacts` preserves completed chunk parquets and manifest for resume; only final artifacts (`trace.parquet`, `regions.parquet`) and `.tmp` files are deleted on failure.
- File-based jobs cannot be paused/resumed. If needed in the future, the file path can be synthetically chunked without affecting the hydrophone path.

## ADR-053: Feedback training architecture — correction tables and bootstrap cleanup

**Date**: 2026-04-12
**Status**: Accepted

**Context**: The call parsing pipeline (Pass 2 segmentation, Pass 3 classification) shipped with bootstrap-only training paths: CLI scripts that call trainer functions directly to produce initial models. The production workflow needs human-in-the-loop correction → retraining, and the bootstrap training worker + API endpoints created confusion about the intended workflow boundary.

**Decision**: Store human corrections in separate SQL tables rather than amending parquet output. Correction tables (`event_boundary_corrections`, `event_type_corrections`) reference events by `event_id` from the immutable inference parquet. Dedicated feedback training workers assemble training data by merging corrections with uncorrected (implicitly approved) events, then call the same trainer functions as bootstrap. Bootstrap worker (`segmentation_training_worker.py`) and its API endpoints are removed; bootstrap scripts continue to call trainers directly without queueing.

**Alternatives considered**:
- Amending parquet files in-place: rejected because it loses the original inference output, complicates diffing, and breaks idempotency (re-running inference would overwrite corrections).
- Storing corrections in a separate parquet: rejected because SQL provides better query semantics for upsert-by-event-id and join with job metadata.
- Keeping bootstrap training worker alongside feedback workers: rejected because two training paths for the same model type creates ambiguity about which to use and doubles the maintenance surface.

**Consequences**:
- Migration `046_feedback_training_tables.py` adds four tables: `event_boundary_corrections`, `event_type_corrections`, `event_segmentation_training_jobs`, `event_classifier_training_jobs`.
- Two new workers (`event_segmentation_feedback_worker.py`, `event_classifier_feedback_worker.py`) join the worker priority order between event classification and manifest generation.
- `vocalization_worker.py` rejects `pytorch_event_cnn` model family — only handles `sklearn_perch_embedding`.
- `segmentation_training_worker.py` deleted; its API endpoints (`POST/GET/GET/{id}/DELETE /call-parsing/segmentation-training-jobs`) removed. The `segmentation_training_jobs` table remains for any existing bootstrap rows but is no longer written to by the application.
- Implicit approval means the training data volume grows with the number of source jobs even if the user only corrects a few events. This is intentional — uncorrected regions provide context and prevent catastrophic forgetting.

## ADR-054: Read-time correction overlay for downstream consumers

**Date**: 2026-04-14
**Status**: Accepted

**Context**: ADR-053 established that human boundary corrections are stored in SQL correction tables while parquet artifacts remain immutable. However, downstream consumers (Pass 3 classification inference, Pass 3 classifier feedback training) read `events.parquet` directly without applying corrections. This means corrected boundaries only benefited segmentation retraining, not classification inference or classifier feedback training.

**Decision**: Introduce a shared `load_corrected_events()` utility in `call_parsing/segmentation/extraction.py` that reads `events.parquet`, queries `event_boundary_corrections` for the job, and merges them via the existing `apply_corrections()` function, returning `list[Event]`. All downstream consumers — classification inference worker and classifier feedback training worker — must use this utility instead of reading parquet directly. Parquet files remain immutable inference snapshots; corrections are applied as read-time overlays.

**Alternatives considered**:
- Materializing corrected events into a new parquet file: rejected because it introduces mutable state, risks stale files, and breaks the immutability contract from ADR-053.
- Having each consumer independently query and apply corrections: rejected because it duplicates logic across consumers and increases the risk of inconsistency.

**Consequences**:
- `event_classification_worker.py` calls `load_corrected_events()` instead of `read_events()` during inference. Added events are classified; deleted events are excluded; adjusted events use corrected boundaries for audio cropping.
- `event_classifier_feedback_worker.py` calls `load_corrected_events()` to get corrected boundaries when assembling training data. Boundary-deleted events are excluded before type resolution; added events need a type correction to be included in training.
- The classify review UI allows boundary editing (adjust, add, delete) with corrections stored against the upstream segmentation job via existing correction endpoints.
- The `with-correction-counts` endpoint now includes `has_new_corrections` to surface when corrections exist that haven't been consumed by a training dataset.

## ADR-055: Perch v2 as first-class classifier family (Approach A — surgical extension)

**Date**: 2026-04-17
**Status**: Accepted

**Context**: The `perch_v2.tflite` model produces 1536-d waveform-input embeddings that are incompatible with the existing perch_v1 (TF2 spectrogram, 1280-d) pipeline. We need to support training classifiers against detection rows embedded with perch_v2, running hyperparameter tuning, and re-embedding detection jobs with an arbitrary model version — all without breaking existing TF2 workflows.

**Decision**: Extend the existing tables and workers surgically (Approach A) rather than introducing a new unified labeled-manifest artifact (Approach B) or a model-agnostic embedding pipeline (Approach C).

Key changes:
- `detection_embedding_jobs` gains `model_version` (composite unique with `detection_job_id`) and progress fields (`rows_processed`, `rows_total`).
- `hyperparameter_manifests` gains `embedding_model_version` — manifest generation now requires an explicit model version and validates all sources against it.
- Storage paths for detection embeddings become model-versioned: `detections/{job_id}/embeddings/{model_version}/`.
- Training jobs accept a new `detection_manifest` source mode with `detection_job_ids` + `embedding_model_version` as an alternative to embedding-set IDs.
- The training worker builds a manifest at execution time via the existing manifest builder, then trains a standard binary classifier.
- For perch_v2 detection sources, only binary row-store labels (`humpback`, `background`, `ship`, `orca`) are used — the vocalization_labels join is skipped since those are a TF2-era concern.
- A `perch_v2` ModelConfig seed row is added via migration 051.

**Alternatives considered**:
- **Approach B (unified labeled-manifest artifact)**: Introduces a new `labeled_manifests` table to unify all training data sources. Rejected because it adds a new abstraction layer and migration complexity without near-term payoff.
- **Approach C (model-agnostic embedding pipeline)**: Refactors the entire embedding pipeline to be model-version-parametric. Rejected because the scope is too large for the current goal and TF2 paths must remain untouched.

**Consequences**:
- Perch v2 is a registered embedding model family with its own ModelConfig row.
- Detection embedding parquet files now live under a model-version subdirectory; migration 049 relocates existing files.
- The TuningTab and TrainingTab in the frontend gain a DetectionSourcePicker component with inline re-embedding status.
- Future work: the hardcoded spectrogram feature parameters in `detector.py` should eventually be resolved from the ModelConfig rather than assumed.

## ADR-056: Sequence Models track parallel to Call Parsing pipeline

**Date**: 2026-04-27
**Status**: Accepted; updated 2026-05-06 after downstream retirement

**Context**: We need a sequence-modeling-adjacent data layer that can consume
Call Parsing outputs without becoming a fifth pass of the four-pass pipeline.
The retained scope is the Continuous Embedding producer: durable, idempotent
embedding artifacts that other analysis workflows can consume later.

**Decision**: Keep **Sequence Models** as a top-level track parallel to Call
Parsing, with Continuous Embedding as its active runtime surface. The retained
implementation includes:
- `continuous_embedding_jobs` with idempotency keyed on `encoding_signature`.
- `processing/region_windowing.py` pure functions for padded event span merging
  and window iteration.
- A producer service and worker that write
  `continuous_embeddings/{job_id}/embeddings.parquet` plus `manifest.json`.
- `/sequence-models/continuous-embeddings` API endpoints and the matching
  frontend Sequence Models navigation entry.

The SurfPerch model invocation remains structured behind an injected
`EmbedderProtocol` so the worker is testable without depending on hydrophone
audio decoding or model loading.

**Alternatives considered**:
- Folding the producer into the Call Parsing pipeline: rejected because
  embeddings have independent model/version/idempotency semantics.
- Storing producer parameters on `region_detection_jobs`: rejected because
  hop, padding, source mode, and model version are orthogonal to detection.
- Reusing cached Pass 1 embeddings directly: rejected because the producer
  needs source-specific coverage and artifact schemas.

**Consequences**:
- `continuous_embedding_jobs` and `continuous_embeddings/{job_id}/` remain part
  of the active storage, API, worker, and UI contract.
- The retired downstream runtime surfaces were removed by the 2026-05-06
  cleanup and schema migration 075.
- Producer idempotency lookup is the service layer's responsibility; workers
  assume their job row is canonical.

## ADR-057: CRNN region-based chunk embeddings as second Sequence Models source

**Date**: 2026-04-29
**Status**: Accepted; updated 2026-05-06 after downstream retirement
**Builds on**: ADR-050 (Pass 2 segmentation), ADR-056 (Sequence Models track)

**Context**: SurfPerch event-padded embeddings cover the immediate neighborhood
of Pass 2 events. We also need a region-scoped Continuous Embedding source that
uses the segmentation CRNN's learned representations over full Pass 1 detection
regions.

**Decision**:
1. **Single-table dispatch on `model_version`**. `continuous_embedding_jobs`
   holds nullable CRNN-only columns. SurfPerch rows leave those columns null;
   CRNN rows leave SurfPerch-only configuration null. Per-source tables were
   rejected because they duplicate the same job lifecycle and idempotency
   behavior.
2. **Concat-and-project chunk embeddings**. Chunk vectors concatenate the
   checkpoint-derived number of feature frames for the requested chunk duration
   by 128 BiGRU channels. The default 16 kHz / 512-hop checkpoint still yields
   eight frames and a 1024-d identity vector; checkpoints trained with other
   feature configs derive their identity dimension from the persisted
   `feature_config`. `IdentityProjection` is the default; random and PCA
   projection remain available for retained experiments.
3. **Shared inference windowing**. Pass 2 inference and the CRNN extractor share
   `iter_inference_windows()`, keeping frame/chunk geometry consistent.

**Consequences**:
- CRNN Continuous Embedding rows include `region_detection_job_id`,
  `event_segmentation_job_id`, `crnn_segmentation_model_id`,
  `crnn_checkpoint_sha256`, chunk geometry, projection config, and CRNN summary
  counters.
- The CRNN `embeddings.parquet` schema records `region_id`,
  `chunk_index_in_region`, `tier`, `event_overlap_fraction`,
  `nearest_event_id`, `distance_to_nearest_event_seconds`, `call_probability`,
  and `embedding`.
- Encoding signatures differ by source kind: CRNN signatures fold in the region
  job, Pass 2 disambiguator, checkpoint, checkpoint feature config, chunk
  geometry, and projection config; SurfPerch signatures include the event
  source mode and SurfPerch geometry.
- The Continuous Embedding API rejects CRNN-only fields on SurfPerch requests
  and SurfPerch-only fields on CRNN requests.
- The frontend Continuous Embedding create form exposes both retained sources
  behind a source-type toggle.

## ADR-062: Segmentation-scoped effective event identity

**Date**: 2026-05-03

**Status**: Accepted

**Supersedes**: ADR-054 for event boundary correction ownership and
downstream event loading.

**Context**: ADR-054 introduced a shared read-time correction overlay, but
boundary corrections were still anchored only to `region_detection_job_id`.
That made corrections ambiguous when multiple Pass 2 segmentation jobs shared
one Pass 1 region job, and it allowed stale saved `add` corrections to appear
beside adjusted raw events. A retained classify-review job also exposed a
region lookup failure: a typed event produced before a later boundary adjustment
could no longer resolve a `region_id` from the corrected event list, leaving the
timeline spectrogram empty.

**Decision**: Boundary corrections are now scoped to the Pass 2 artifact they
edit via nullable `event_segmentation_job_id` and `source_event_id` columns on
`event_boundary_corrections`. New review clients populate those fields. A
canonical `load_effective_events()` utility reads immutable `events.parquet`
rows and overlays only corrections for the selected segmentation job. Adjusted
events preserve their source `event_id`; added events use stable synthetic IDs
derived from the correction row ID. The correction write service rejects add or
adjust batches whose final effective event set would overlap within the same
segmentation job and region.

Classification typed-event APIs still return persisted `typed_events.parquet`
rows, but resolve `region_id` from raw segmentation events first and effective
events second. This keeps older classification artifacts renderable after later
boundary edits while still supporting effective added-event IDs. Sequence Models
continuous embedding jobs now carry `event_source_mode`: `raw` preserves the
old `events.parquet` behavior, while `effective` consumes
`load_effective_events()` and includes a correction revision fingerprint in the
encoding signature.

**Consequences**:
- Historical correction intent is not automatically backfilled or repaired.
  Existing ambiguous rows are cleaned manually.
- `events.parquet` and `typed_events.parquet` remain immutable inference
  artifacts; reviewed event state is a read-time contract.
- Region-scoped vocalization corrections remain time-range labels and are
  resolved by overlap against the effective event set.
- Future event-aware consumers must explicitly choose raw versus effective
  event semantics and include that choice in provenance or idempotency keys.

## ADR-063: Event Encoder v3 ridge frequency descriptors for piano-roll display

**Date**: 2026-05-20

**Status**: Accepted

**Builds on**: ADR-056 (Sequence Models track), ADR-057 (CRNN region-based
chunk embeddings)

**Context**: Event Encoder piano-roll views previously placed events primarily
from `median_f0` with fallback to full-spectrum `peak_frequency`. That was
misleading for high whistles, shrieks, and harmonic moans: `pyin` can miss or
lock to low subharmonics, and full-spectrum peaks can be dominated by rumble
even when the spectrogram contains strong higher ridges. The existing STFT
ridge tracker already recovers a dominant ridge path, but only persisted slope
and inflection summaries, so the UI had no artifact-backed way to render the
ridge's frequency band.

**Decision**: Introduce Event Encoder tokenizer/default contract
`crnn-event-encoder-v3`. V3 appends eight scalar DSP descriptors to the
existing 14-entry descriptor block:
`ridge_median_frequency`, `ridge_low_frequency`, `ridge_high_frequency`,
`ridge_frequency_span`, `ridge_coverage`, `ridge_energy_ratio`,
`band_limited_peak_frequency`, and `high_band_energy_ratio`.

Ridge low/high values are trimmed percentile summaries of the tracked ridge
path, not literal frame min/max values. `band_limited_peak_frequency` keeps a
rumble-resistant display peak while preserving legacy `peak_frequency`.
`high_band_energy_ratio` records how much mean-spectrum energy lies above the
configured high-band floor. New v3 defaults raise `ridge_max_frequency_hz` to
6000 Hz and lower the default descriptor block weight to 0.364 so the appended
display descriptors do not increase aggregate descriptor influence solely by
adding columns.

The piano roll defaults v3 artifacts to Ridge mode. It still renders one
rectangle per event: trusted ridge summaries provide the low/center/high
frequency band, with F0 and peak fallbacks for older artifacts or weak ridge
summaries. For broad harmonic events, the UI may conservatively expand the
rendered upper bound to the spectral centroid when high-band energy,
bandwidth, and low spectral entropy indicate a tonal high-band envelope.

**Alternatives considered**:
- Persisting full frame-level ridge contours: rejected for this feature because
  the immediate display problem only needs scalar summaries, and contour
  sidecars would create a larger artifact/versioning contract.
- Drawing multiple ridges inside each token rectangle: rejected because token
  identity remains one event token, and the first UI improvement should make
  the existing rectangle's frequency extent truthful.
- Replacing the existing ridge tracker with a new pitch estimator: rejected
  because the STFT ridge path already captured the high-frequency examples and
  avoids adding a second DSP pipeline.

**Consequences**:
- Completed v2 Event Encoder artifacts remain readable through each artifact's
  manifest-recorded descriptor names.
- New v3 tokenization signatures differ when ridge descriptor settings change.
- Frontend display is artifact-authoritative: it uses persisted descriptor
  values from the timeline endpoint and does not recompute audio descriptors.
- Full STFT matrices, F0 contours, and ridge contours are still not persisted
  for Event Encoder artifacts.

## ADR-064: Piano Roll Notes sidecar worker

**Status**: Accepted
**Date**: 2026-05-20
**Builds on**: ADR-056 (Sequence Models track), ADR-057 (CRNN region-based
chunk embeddings), ADR-063 (v3 ridge frequency descriptors).

**Context**: The Event Encoder piano roll renders one rectangle per
tokenized event, which conveys token identity and rough frequency envelope
but not the per-partial pitch content of each call. We want a MIDI-style
view that resolves the F0 and visible harmonics inside each event without
recomputing during render, without entangling the descriptor block, and
without ever modifying immutable Event Encoder outputs.

**Decision**: Introduce a Piano Roll Notes worker that runs independently
and idempotently after each Event Encoder job. The worker reads the
encoder's source audio plus its descriptor / token artifacts and writes a
per-job parquet sidecar of per-event MIDI notes alongside the existing
Event Encoder artifacts.

- **Idempotent key**: `(event_encoder_job_id, extractor_version)`, mirroring
  the encoder-level signature pattern from ADR-056. Re-submitting an
  in-flight or completed key is a no-op; failed and canceled keys reset.
- **Sidecar path**: `event_encoders/{job_id}/event_notes_{version}.parquet`.
  One row per note (not per event), sorted by `(start_utc, midi_pitch)`.
- **Auto-enqueue**: completing an Event Encoder job auto-enqueues a notes
  job at the current default `extractor_version`. The hook swallows
  conflicts so the encoder transition cannot be blocked.
- **Algorithm**: CQT + per-frame peak picking + greedy nearest-neighbor
  cross-frame tracking + harmonic-prior labeling + MIDI quantization, with
  job-level velocity calibration from per-frame log-magnitude percentiles.
  Defaults are persisted in `params_json` for reproducibility.
- **Versioning**: `extractor_version` starts at `"v1"`. Future tracker /
  range / quantizer changes bump the version. The notes-jobs table allows
  multiple completed runs per encoder job at different versions; the UI
  serves the latest completed row.
- **UI**: a `Notes` view mode renders the sidecar on a log-frequency 88-key
  Y axis (MIDI 21–108) with semitone gridlines, octave labels (C0…C8), and
  black-key shading. It defaults when sidecar is available, falls back to
  the prior rectangle mode on fetch failure, and exposes a `Generate notes`
  / `Re-run` action backed by `POST .../notes-jobs`.

**Alternatives considered**:
- Extending the descriptor block with note-level fields: rejected because
  it would couple per-partial MIDI quantization to the embedding /
  tokenization signature and force a new encoder run for every algorithm
  change.
- Computing notes lazily in the frontend: rejected because CQT + tracking
  is too expensive for interactive rendering at full-job scale, and any
  later MIDI/MPE export needs the same per-job artifact.
- Re-using the existing F0 contour pipeline: rejected because the
  descriptor block intentionally persists only scalar summaries and a
  contour sidecar would change the artifact contract from ADR-063.

**Consequences**:
- Existing Event Encoder artifacts remain untouched; piano roll notes can
  be regenerated independently.
- A new SQL table `piano_roll_notes_jobs` tracks lifecycle and per-version
  history; canonical artifact paths follow the `event_encoder_dir` layout.
- Failure or re-run of the notes worker never invalidates the descriptor
  block. Encoder jobs without a notes sidecar render with the existing
  rectangle modes.

## ADR-065: Extended Piano Roll Notes pitch range (placeholder)

**Status**: Deferred
**Date**: 2026-05-20

**Context**: Humpback social sounds extend above the standard 88-key piano
range (above C8 ≈ 4186 Hz) and very low moans extend below A0 (27.5 Hz).
The first Piano Roll Notes release intentionally clamps to MIDI 21–108 so
the canvas matches the familiar 88-key piano metaphor and `extractor_version`
stays scoped to a single tractable range.

**Decision (placeholder)**: extend the pitch range in a future
`extractor_version` (e.g. `"v2"`). Open questions: does the UI use a wider
piano-key band (e.g. MIDI 12–127) or switch to a log-frequency continuum,
how to keep `tokenColor(event_token)` legible across denser pitches, and
whether to expose per-job overrides of the quantizer's
`[min_pitch, max_pitch]` parameters.

**Consequences (anticipated)**: the existing v1 sidecar stays readable;
extending the range bumps `extractor_version` and triggers a re-run on
demand without invalidating older runs.

## ADR-066: User-initiated async MIDI export for Piano Roll Notes

**Status**: Accepted
**Date**: 2026-05-20

**Context**: The Piano Roll Notes worker (ADR-064) emits one parquet row
per detected MIDI note. Users have no built-in path to export this data as
a Standard MIDI File for use in DAWs or MIDI viewers. A persisted `.mid`
artifact is also useful as a portable, browseable representation of the
notes payload outside the app.

**Decision**: Add a user-initiated asynchronous MIDI export pipeline
parallel to (but independent of) the Piano Roll Notes worker.

- Storage: a new top-level `<storage_root>/exports/` directory holds export
  artifacts; MIDI exports specifically live under
  `exports/event_encoders/{job_id}/notes_{extractor_version}.mid`.
- Persistence: a new `piano_roll_midi_exports` table tracks each export
  job, idempotent on `(event_encoder_job_id, extractor_version)`. The
  unique key allows `force=true` to reset a `complete` row back to
  `queued` so users can refresh exports.
- Worker: `piano_roll_midi_export_worker.run_piano_roll_midi_export` mirrors
  the Piano Roll Notes worker lifecycle (queued → running → complete /
  failed, atomic write with `.tmp` rename, partial-file cleanup on
  exception). The queue claim and runner dispatch follow the same pattern.
- Synthesis: `humpback.processing.midi_synthesis.notes_table_to_midi_bytes`
  is a pure, deterministic function that takes a pyarrow Table matching
  the Piano Roll Notes schema and returns SMF Type 1 bytes. Conventions:
  480 ticks-per-quarter, constant 120 BPM tempo (written once at tick 0),
  all partials stacked on MIDI channel 1, time origin shifted to the
  earliest note's `start_utc`, velocity used verbatim, pitches outside
  `[0, 127]` clamped silently, zero-duration notes dropped.
- API: `GET /event-encoders/{id}/midi-export-status`,
  `POST /event-encoders/{id}/midi-exports`, and
  `GET /event-encoders/{id}/midi-export` mirror the notes endpoint shape.
- UI: an "Export MIDI" button to the left of the Notes status pill drives
  the lifecycle (disabled until notes are `complete`; transitions through
  "Exporting…" to "Download MIDI"). A small overflow menu next to the
  download button exposes a "Re-export" item that submits `force=true`.
- Library: `mido` (pure Python, no native deps). Chosen because it is the
  de-facto MIDI standard, supports SMF read/write, and exposes every
  primitive the deferred MPE pitch-bend extension will need (multi-channel
  pitch bend, RPN sequences for the MPE Configuration Message and bend
  range setup, CC 74 timbre, channel pressure). No library change will be
  required when pitch-bend lands.

Alternatives considered:
- *Folding MIDI generation into the notes worker.* Rejected because it
  ties the export rules to the notes lifecycle, forces every notes run to
  pay the synthesis cost, and complicates re-export.
- *Frontend-side MIDI synthesis.* Rejected because the user-stated
  requirement is a persisted artifact under the root app path, which
  client-side synthesis cannot satisfy.
- *Lazy backend endpoint that synthesizes on demand.* Considered as a
  fallback, but the user wanted an async worker so the UI can signal
  completion status, matching the existing Notes pill pattern.

**Consequences**:
- One new table, model, schema, service, worker, and three API endpoints
  added under the sequence-models domain.
- `mido` becomes a base dependency (small, pure Python, no platform
  considerations).
- The export action is per-job and per-extractor-version. Users who want a
  fresh `.mid` after editing notes parquet must POST with `force=true`.
- Per-frame pitch contours and pitch-bend rendering remain explicitly out
  of scope. When added, they require a parquet schema extension (notes
  `v2`) and an MPE-aware synthesizer; both can be implemented inside this
  module without library or API redesign.



## ADR-067: Per-frame harmonic labeling and channelized MIDI export

**Date**: 2026-05-20
**Status**: Accepted
**Spec**: [docs/specs/2026-05-20-event-encoder-midi-channelized-design.md](docs/specs/2026-05-20-event-encoder-midi-channelized-design.md)

**Context**:
Inspecting a representative Piano Roll Notes export
(`event_encoder_job_id = b759d8bf-0ecf-469a-b169-333b36c60906`,
80,561 notes across 1,672 events) revealed that **82.8% of notes carried
`partial_index = -1`**. A deeper analysis of those `-1` tracks showed three
upstream causes inside `label_harmonics()` plus one fundamental limitation:
the `max_harmonic = 8` cap excluded 55% of overlapping content (real 9th
through Nth harmonics of low-pitch F0s); the F0 anchor sort key
`(start_frame, median_bin)` let higher-frequency earlier-starting tracks
win over the actual lowest-bin F0; the consume-on-overlap step was
unconditional, so any track that overlapped an F0 candidate but failed
the harmonic check was permanently locked out from anchoring its own
cluster; and the median-bin ratio metric failed on sweeping pitches that
maintain a clean per-frame harmonic relationship even when their medians
do not align.

Independently, every note in the v1 export landed on a single MIDI
channel. Users wanted to audition the harmonic stack separately from the
fundamental in a DAW, which a single channel does not support.

**Decision**:
Bundle the upstream labeler rewrite and the downstream channelization
into one shipped change, and explicitly skip backward compatibility —
existing v1 artifacts will be deleted manually via the UI:

1. **Labeler rewrite** in `src/humpback/processing/piano_roll_tracker.py`.
   `label_harmonics()` now sorts F0 anchors by `median_bin` ascending
   (with `track_id` as a deterministic tiebreaker), uses per-frame
   ratios at every shared frame, summarizes them with
   `statistics.median_low` over the nearest-integer harmonic and the
   median absolute cents deviation against that integer multiple, and
   leaves tracks that fail the check unprocessed so they remain eligible
   to anchor their own clusters on later iterations. New
   `HarmonicParams` defaults: `max_harmonic = 16`, `cents_tolerance =
   75.0`, new `min_overlap_frames = 3`. `Track` gains a `frames:
   list[int]` field populated by `build_tracks()`; legacy fixtures with
   empty `frames` lists synthesize them contiguously from `start_frame`.

2. **Channelized MIDI synthesis** in
   `src/humpback/processing/midi_synthesis.py`. The single-channel
   `MIDI_CHANNEL = 0` constant is removed; a new `CHANNEL_LAYOUT` tuple
   pins seven `ChannelSpec(channel, program, name)` entries for F0,
   2nd–5th harmonics (one channel each), a combined
   `CHANNEL_HARMONIC_HIGH` for partial_index ≥ 5, and `CHANNEL_UNMATCHED`
   for `partial_index = -1`. The GM drum channel (channel 9, 1-indexed
   10) is intentionally absent so playback engines do not re-map pitched
   humpback content to drum sounds. The SMF Type 1 file now contains one
   tempo track plus one channel track per layout entry; each channel
   track starts at tick 0 with `track_name` + `program_change` meta
   events so DAWs render named, distinctly-voiced lanes out of the box.
   Empty parquet still produces a tempo-only SMF.

3. **Extractor version bump** `v1 → v2` in
   `src/humpback/models/piano_roll_notes.py`. The parquet schema is
   unchanged — only the distribution of `partial_index` values changes —
   but the version increment marks the labeling-semantics shift in the
   filename. No Alembic migration, no batch rebuild: v1 artifacts on
   disk and the corresponding job rows stay where they are until the
   user deletes them via the UI.

Alternatives considered:
- *Bug-fix-only labeler change.* Rejected because median-bin ratios
  fundamentally cannot represent sweeping pitches, which is the dominant
  motion in humpback song.
- *Full multi-cluster labeling (concurrent F0s).* Deferred because the
  single-cluster model already shrinks `-1` from 83% to a projected
  17–25%, and multi-cluster handling requires a cluster-graph algorithm
  and new validation methodology.
- *MPE / pitch-bend mode.* Still deferred — requires a future parquet
  `v3` with sub-semitone pitch contours.
- *User-configurable channel layout via `params_json`.* YAGNI. The slim
  layout is hard-coded; users who want a different layout can convert
  the file in their DAW.
- *Keep `-1` for genuinely non-harmonic tracks.* Implemented as
  "anchor own cluster" instead — under v2, every track with valid
  frequency data eventually anchors as F0 of some cluster, so `-1`
  becomes vanishingly rare in practice. This is a stronger outcome than
  the original spec predicted and the unmatched channel will mostly
  hold tracks rejected for sub-overlap or pre-iteration reasons.

**Consequences**:
- The per-frame algorithm needs `Track.frames` populated by
  `build_tracks()`. Test fixtures that hand-construct a `Track` with
  fewer `bins` entries than `end_frame - start_frame + 1` are
  reinterpreted as "track active for `len(bins)` contiguous frames
  starting at `start_frame`" so legacy fixtures keep working.
- Existing v1 Piano Roll Notes and MIDI Export artifacts are not
  automatically rebuilt. Users who want the new labels and channelized
  exports must delete the v1 rows and re-enqueue from the UI.
- The Playwright fixtures and integration tests now expect `v2` filenames
  and the multi-track MIDI structure. One pinning test in
  `test_get_notes_pins_to_explicit_extractor_version` still passes "v1"
  explicitly to demonstrate that older versions remain reachable when
  pinned.
- `partial_index ≥ 5` collapses onto a single channel in the export but
  remains distinguishable in the parquet, so future export-side reshuffles
  do not require a notes re-run.

## ADR-068: Piano Roll windowed bundled export (MIDI + FLAC)

**Date**: 2026-05-21

**Status**: Accepted

**Context**: ADR-066 introduced a user-initiated MIDI export that wrote the
entire Event Encoder job's Piano Roll Notes parquet as one canonical SMF
under `<storage_root>/exports/event_encoders/{job_id}/notes_{version}.mid`.
ADR-067 reshaped the SMF as a slim seven-channel layout. In practice users
work in DAWs (Logic Pro) and want to drop both the notes and the source
audio for a *specific* zoomed/panned region of a job onto the same project
position. Two gaps drove this change:

1. The export shipped the full job, not the viewer's current `timeRange`.
2. There was no co-exported audio clip aligned to the same window.

**Decision**:

1. **Windowed bundled export.** The `POST` create endpoint now requires
   `window_start_utc` and `window_end_utc`; the worker filters Piano Roll
   Notes by `[window_start_utc, window_end_utc)`, clips partially-overlapping
   notes, drops sub-millisecond residuals, and synthesizes the MIDI with a
   new `time_origin_utc=window_start_utc` argument on
   `notes_table_to_midi_bytes()` so that a note at `window_start_utc` lands
   at tick 0.
2. **Co-exported `.flac`.** The same worker resolves the source audio for
   the same window via `resolve_timeline_audio()` (32 kHz mono, gap-filled
   with silence) and writes a 16-bit PCM FLAC alongside the MIDI at
   `<storage_root>/exports/event_encoders/{job_id}/audio_{version}.flac`.
   FLAC samples are NOT loudness-normalized so the clip matches what the
   piano-roll player rendered.
3. **One rolling artifact per `(job, version)`.** The existing
   uniqueness key is preserved; re-exporting overwrites both the MIDI and
   the FLAC and updates the persisted window bounds. The new columns
   (`window_start_utc`, `window_end_utc`, `audio_path`, `audio_size_bytes`,
   `audio_sample_rate`, `audio_duration_s`) are all NOT NULL, established
   by Alembic revision `079`. The migration drops legacy non-windowed rows
   and their `.mid` files because they no longer represent something the
   UI would surface.
4. **30-minute soft cap.** The schema validator and the API layer reject
   windows whose duration exceeds 1800 s. The UI disables the export
   button with a tooltip when the current viewport exceeds the cap.
5. **Atomic dual-write.** The worker writes both files to `*.tmp` paths;
   on success it renames both. If the FLAC write fails after the MIDI
   rename, the MIDI is rolled back so the on-disk pair stays consistent.
6. **DAW workflow.** No in-UI hint. Standard PPQN (480) + 120 BPM is
   preserved; users who want the MIDI and audio to line up in Logic Pro
   import the MIDI with "use MIDI file tempo" accepted, drop the FLAC at
   the same project position, and alignment holds.

Alternatives considered:

- *Two independent exports (separate MIDI and FLAC rows, services, and
  endpoints).* Rejected — two state machines and two failure paths for
  one user action.
- *SMPTE-timecode MIDI for tempo-independent alignment.* Rejected — DAW
  behavior with SMPTE-format SMF is less predictable than the tempo-event
  path, and the current approach already works as long as project tempo
  matches the MIDI's 120 BPM.
- *Window-keyed multiple cached exports.* Rejected — the user request
  was an "export current view / re-export current view" affordance, not
  a history of windowed exports. A single rolling artifact matches the
  intent and avoids cleanup complexity.

**Consequences**:

- Legacy `piano_roll_midi_exports` rows from the ADR-066 era are removed
  by migration `079`; users must re-export the windows they care about.
- The audio resolution chain at export time is `EventEncoderJob →
  EventSegmentationJob → RegionDetectionJob → (hydrophone_id,
  start_timestamp, end_timestamp)`, identical to the chain used by the
  Piano Roll Notes worker. Tests rely on monkey-patching
  `_resolve_window_audio` so they don't need a real hydrophone provider.
- The frontend `MidiExportButton` now requires `windowStartUtc` /
  `windowEndUtc` props. The Piano Roll page threads them from its
  existing `timeRange` state. The button's "Re-export view" affordance
  is emphasized when the current viewport differs from the persisted
  window by more than 50 ms.

## ADR-069: Ridge-aligned F0 + harmonics extractor and MPE Piano Roll MIDI export

**Date**: 2026-05-22

**Status**: Accepted

**Context**: The Piano Roll Notes pipeline ran an independent CQT peak tracker
that produced one MIDI note per semitone-quantized track. Three structural
problems followed:

1. A continuous frequency sweep crossing semitone boundaries fragmented into a
   staircase of short fixed-pitch notes — the visual track did not match the
   spectrogram ridge.
2. The Event Encoder already runs its own STFT ridge tracker for descriptor
   computation, so the system had two DSP paths computing F0 independently
   from the same audio.
3. ADR-067's slim 7-channel MIDI layout encoded partial identity by channel
   but offered no per-note expressive pitch — DAWs could not audition the
   actual humpback pitch trajectory.

Production data on job `2679ab0d` showed 66.2 % `partial_index = 0` (F0
dominance) versus ADR-067's predicted 17–25 % unmatched rate. Root cause: a
stale-params bug in `_resolve_params()` silently downgraded harmonic prior
thresholds for every auto-enqueued v2 job, so the v2 algorithm ran with v1
tolerances. The bug was real, but the fix path was a clean rewrite rather
than a patch.

**Decision**:

1. **STFT ridge as canonical F0 source.** Extract `compute_ridge_path()` into
   `src/humpback/processing/ridge_path.py` (shared module). The Event Encoder
   worker persists per-event ridge contours to
   `event_encoders/{job_id}/event_ridges_{tokenizer_version}.parquet`
   (one row per frame per event with `event_id`, `frame_index`,
   `frame_time_offset_s`, `log_frequency`, `strength`, `energy_ratio`). The
   Piano Roll Notes v3 worker reads this sidecar; if absent, it recomputes
   in-process.
2. **Coherent-contour note model.** The v3 extractor segments the refined F0
   contour into notes only at energy gaps (≥ 3 frames below the per-frame
   amplitude floor) or surviving octave jumps from subharmonic refinement.
   A sweep from C5 to E5 emits one note, not three. Harmonic siblings are
   derived structurally at `n · f₀(t)` for `n ∈ {2..16}` with ±75¢ tolerance;
   harmonic bend streams in cents equal their parent F0's bend stream in
   cents (cents conservation). Subharmonic refinement gates each candidate
   on two tests: the per-frame `k_sub · MAD` noise-floor check from spec
   §5.2 *and* a `min_relative_log_magnitude = -2.5` ceiling that rejects
   CQT leakage at `f₀/2` for pure tones — without the second gate, clean
   sinusoids could be demoted by spectral skirts that pass the bare
   noise-floor test.
3. **MPE Lower Zone replaces slim 7-channel.** Per-voice channel rotation
   (deterministic on `(start_utc, note_uid)` with longest-idle pick and FIFO
   voice steal across the 15 member channels) plus per-member ±24-semitone
   pitch bend. Partial identity is preserved through per-note
   `program_change` (F0→0, H2→11, H3→12, H4→10, H5→8, H6..H16→88), CC 74
   (= `partial_index * 16`), and master-track `MetaMessage("text", "pN")`
   events. DAW-side per-partial mute/solo by routing is structurally lost —
   documented regression.
4. **MIDI pitch range MIDI 12–120.** Notes view Y-axis becomes C0…G9; the
   88-key band (MIDI 21–108) keeps normal shading and extended bands render
   with a desaturated tint. Cents are clamped to ±9600 for safety even
   though the bend range is ±2400 cents.
5. **Two new sidecars.** `event_encoders/{job_id}/event_notes_v3.parquet`
   gains `note_uid` (deterministic UUID v5 of
   `(job_id, event_id, partial_index, track_id, start_utc_rounded_ms)`),
   `f0_track_id`, and `contour_frame_count` columns. A new
   `event_encoders/{job_id}/event_note_contours_v3.parquet` carries one row
   per frame per note with `cents_from_pitch`, `harmonic_strength`, and
   `subharmonic_octave`. Both files write via `.tmp` + atomic rename with
   paired cleanup on exception.
6. **Curved-ribbon rendering by default.** The Piano Roll page batches
   `/notes/contours` requests via React Query keyed on `note_uid`; cached
   uids survive viewport pans. Notes without a fetched contour render as
   flat bars and hydrate into ribbons when their contour arrives. A 500 on
   `/notes/contours` triggers a single non-blocking toast.
7. **No backward compatibility for v1/v2.** Existing artifacts on disk stay
   readable but are not regenerated. The export worker resolves the highest
   `complete` notes-job version (`max("v1", "v2", "v3")`), so a job with a
   complete v3 row gets v3 exports automatically. Users delete legacy rows
   via the existing job-admin UI when they want to free space.
8. **`HarmonicParams` retired.** The dataclass and the per-frame harmonic
   labeling pass from ADR-067 (`label_harmonics` in `piano_roll_tracker.py`)
   leave the active code path. The stale-params bug ceases to exist by
   construction. Users wanting "correct v2" quality re-run jobs at v3.
9. **`DEFAULT_EXTRACTOR_VERSION = "v3"`.** Auto-enqueue creates v3 notes
   jobs for any newly-completing encoder job.

Alternatives considered:

- *Patch the stale-params bug and keep ADR-067's slim 7-channel layout.*
  Rejected — fixes one defect but leaves the staircase artifact, the dual
  DSP paths, and the lack of per-note expressive pitch.
- *Keep MIDI pitch range at MIDI 21–108.* Rejected — humpback fundamentals
  routinely fall below C2 and harmonics reach above C8; clipping outside the
  88-key band hides real content. The extended bands are tinted in the
  renderer so users still recognize them as "outside piano range" at a
  glance.
- *Use one MIDI channel per partial (extend ADR-067 with bend on a small
  channel pool).* Rejected — pitch bend is channel-wide, so two simultaneous
  F0 notes at different pitches collide on channel 1. MPE Lower Zone solves
  this by design.
- *Multi-F0 per event.* Rejected as out of scope. The single-cluster-per-event
  simplification from ADR-067 is preserved; tracks that don't fit the
  dominant F0 cluster become their own F0 anchors.
- *Pre-populate v1/v2 → v3 migration via batch job.* Rejected — bumping the
  default version and letting users opt in via the existing UI matches the
  pattern in ADR-066 and ADR-067.

**Consequences**:

- The Piano Roll Notes spec from ADR-064 / ADR-067 is superseded in part:
  §6 of [2026-05-20-piano-roll-midi-notes-design.md](docs/specs/2026-05-20-piano-roll-midi-notes-design.md)
  (extraction algorithm) and §8.1 (Notes view rectangle rendering); §5–§6 of
  [2026-05-20-event-encoder-midi-channelized-design.md](docs/specs/2026-05-20-event-encoder-midi-channelized-design.md)
  (harmonic labeler + slim 7-channel layout); §10 of
  [2026-05-20-piano-roll-midi-export-design.md](docs/specs/2026-05-20-piano-roll-midi-export-design.md)
  (single-channel MIDI synthesis).
- `notes_table_to_midi_bytes()` detects v3 input by the presence of
  `note_uid` in the parquet and switches to MPE synthesis. v2-shape callers
  remain on the legacy slim 7-channel path during the dual-version window
  for regression-test stability; the production worker resolves to v3 once
  any v3 job is complete.
- `partial_index = -1` is no longer reachable; the v3 architecture has no
  "unmatched tracks" concept — every ridge segment is either an F0 anchor or
  a derived harmonic. Older parquets keeping `-1` rows remain readable.
- New API endpoint `POST /sequence-models/event-encoders/{id}/notes/contours`
  takes `{note_uids: [str], extractor_version?: str}` in the request body
  and is bounded to 2000 `note_uids` per request (413 above cap). 422 when
  the resolved job has no v3 contour sidecar. Unknown `note_uid` in a
  valid request is silently dropped (partial misses aren't errors). The
  endpoint uses POST (not GET) because a viewport-scale uid list at
  ~48 bytes per UUID overruns the Vite dev server's ~8 KB header limit
  when sent as repeated query parameters — caught in production after
  Phase 4 shipped GET; commit ee9467c migrated both sides to POST.
- Encoder runs not produced under the new sidecar-writing code fall through
  to in-process ridge recomputation in the notes worker. Output is
  identical; cost is ~50 s extra on a 1672-event job.
- No Alembic migration — sidecar paths live in `params_json` on the existing
  `piano_roll_notes_jobs` row.
- Frontend ribbon hit-testing uses ≤ 6 px polyline distance. The
  `rafBudgetMs` cap protects against worst-case redraw cost. A new
  Playwright perf spec asserts ≥ 30 fps median during a pan gesture on a
  synthetic 10k-notes × 10-frames-each fixture.

## ADR-070: Piano Roll Notes v4 — HPS F0 selection with extended low band

**Date**: 2026-05-23

**Status**: Accepted

**Spec**: [docs/specs/2026-05-23-piano-roll-notes-v4-hps-f0-design.md](docs/specs/2026-05-23-piano-roll-notes-v4-hps-f0-design.md)

**Context**: The v3 ridge-aligned extractor (ADR-069) refined the per-frame
Viterbi ridge by trying to halve F0 one octave at a time and accepting the
candidate only when its CQT magnitude was within ≈22 dB of the current
ridge magnitude (`min_relative_log_magnitude = -2.5`). On Event Encoder job
`690580c5-7804-43c9-bd8d-690691b5d6d4` (1635 events, 56 352 notes, 463 963
contour rows) this produced a severely top-heavy pitch distribution: only
10 % of notes were F0 (`partial_index = 0`), median F0 sat at MIDI 69
(440 Hz), and the ridge tracker emitted **zero frames below 100 Hz**
because `STFTParams.min_frequency_hz = 100.0` capped every candidate. 39 %
of F0 frames already triggered subharmonic refinement, yet the post-refine
median was still 440 Hz — the relative-magnitude gate rejected true F0s
that were more than ~22 dB weaker than their H2/H3, which is the textbook
humpback song case.

Wide-CQT noise characterization on the same audio: 3.8 % of sub-100 Hz
frame-bins exceed the 200–1500 Hz band's 90th percentile, so genuine
moan content does exist in that band, but the median sub-100 Hz energy
is ~6 dB below the mid band — single-bin energy thresholds cannot
separate signal from noise. A multi-bin harmonic-support test can.

**Decision**:

1. **HPS-style F0 selection.** Replace `_refine_subharmonic` with
   `_score_f0_candidates` (new function in
   `src/humpback/processing/note_extractor_v4.py`). For each ridge frame
   at log₂-frequency `L`, score candidates `f_c = ridge / d` for
   `d ∈ {1, 2, 3, 4, 5, 6}` by total harmonic-stack support across the
   first 8 partials in the CQT column. Pick the candidate with the
   highest score, tie-break toward the smallest `d` (Occam). The
   STFT ridge tracker still seeds frame presence; HPS chooses which
   sub-divisor represents the true F0.
2. **Per-harmonic gates.** A harmonic counts toward `count_present`
   only when (a) it is a strict local maximum in the CQT column,
   (b) it clears the per-frame noise floor by `k_noise · MAD`, and
   (c) it sits at least `min_above_floor = 1.0` log units above the
   noise floor. After collection, drop harmonics more than
   `max_harmonic_dynamic_range_log = 3.0` (~26 dB) below the candidate's
   strongest harmonic so CQT filter-skirt artifacts at subharmonic
   positions of a strong tone (~30 dB below the parent) cannot inflate
   the count.
3. **Sub-100 Hz tightening.** Candidates below
   `low_band_threshold_hz = 100.0` need ≥ 3 surviving harmonics; above
   100 Hz they need ≥ 2. A flat `low_band_penalty = 0.5` log units
   acts as a tie-breaker when sub-100 and ≥ 100 Hz scores are within
   ~4 dB; strong moans outscore noise by far more than 0.5.
4. **Lower the STFT band floor to 30 Hz.** `STFTParams.min_frequency_hz`
   default drops from `100.0` to `30.0` so HPS candidates can actually
   descend into the humpback moan band. v3 jobs keep the historical
   100 Hz default so re-running an old `params_json` reproduces v3
   bytes.
5. **Sidecars and contracts.** Outputs land at
   `event_encoders/{job_id}/event_notes_v4.parquet` and
   `event_note_contours_v4.parquet`. The schemas match v3 exactly; only
   the `subharmonic_octave` column changes semantics — in v4 it stores
   `chosen_divisor − 1` (0 = ridge is F0, 5 = ridge is H6). Renderers
   treating it as a diagnostic "ridge-shift indicator" work unchanged.
6. **`DEFAULT_EXTRACTOR_VERSION = "v4"`.** New encoder jobs auto-enqueue
   v4. The MIDI export resolver already orders by lex-sorted
   `extractor_version` desc, so a complete v4 row wins over a complete
   v3 row automatically; the MPE Lower Zone synthesizer detects MPE by
   the presence of `note_uid` (unchanged in v4), so the windowed bundled
   export path needs no changes.
7. **No auto-backfill of v4 for completed v3 jobs.** Mirrors the v3
   launch pattern from ADR-069. Users delete legacy rows via the
   existing job-admin UI when they want to free space and regenerate.

Alternatives considered:

- *Patch the v3 relative-magnitude gate.* Cheaper but still assumes the
  true F0 sits at an octave subdivision of the ridge; misses H3 lock.
  Rejected.
- *Add explicit `f/3` and `f×2/3` candidates without full HPS.* Catches
  H3 specifically but is ad-hoc and doesn't generalize to H5/H6.
  Rejected.
- *Full HPS scan over a coarse F0 grid (no ridge dependency).* Cleaner
  but loses the ridge tracker's smoothness prior and pays per-frame
  compute. Rejected for v4 — the ridge is a good seed and the divisor
  set covers realistic lock cases.
- *Multi-F0 per event.* Out of scope; one F0 contour per event remains
  the contract.

**Consequences**:

- No DB migration. The `piano_roll_notes_jobs` table already supports
  arbitrary `extractor_version` strings.
- v3 sidecars on disk stay readable; nothing is deleted. Mixed-version
  coexistence is tested.
- Renderer, MPE synthesizer, MIDI export worker, and the frontend Notes
  view need no changes — v4 emits the same `note_uid` keyed contour
  shape as v3.
- The `subharmonic_octave` column semantics change for v4 rows. Any
  downstream code interpreting the field as "octave halvings" needs to
  switch to "divisor − 1" for v4. Today only the renderer's diagnostic
  tooltip touches the field, and it treats it as opaque.
