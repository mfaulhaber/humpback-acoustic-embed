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
**Status**: Accepted

**Context**: We need to add a sequence-modeling layer (HMM latent state discovery on SurfPerch embeddings) that consumes Pass-1 region detections without coupling to the four-pass call parsing pipeline. The first PR lands the data-plumbing producer (continuous 1-second-hop embeddings padded around regions); subsequent PRs add HMM training, interpretation visualizations, and motif mining.

**Decision**: Introduce a new top-level **Sequence Models** track parallel to Call Parsing rather than extending the four-pass pipeline. PR 1 adds:
- `continuous_embedding_jobs` SQL table (Alembic 057) with idempotency keyed on `encoding_signature = sha256(region_detection_job_id, model_version, hop_seconds, window_size_seconds, pad_seconds, target_sample_rate, feature_config)`.
- A new `processing/region_windowing.py` pure-function module: `merge_padded_regions` and `iter_windows`, deterministic and side-effect free.
- A producer service + worker pair that reads regions from a completed `RegionDetectionJob` and writes `continuous_embeddings/{job_id}/embeddings.parquet` + `manifest.json` atomically.
- A new FastAPI router under `/sequence-models/` and a corresponding frontend Sequence Models nav section.
- 1:1 source linkage (one `region_detection_job_id` per producer job); cross-source training and stored-model decode are explicit non-goals for this PR.

The SurfPerch model invocation is structured behind an injected `EmbedderProtocol` so the worker is testable without depending on hydrophone audio decoding or model loading.

**Alternatives considered**:
- Folding the producer into the Call Parsing pipeline (e.g., as a fifth pass): rejected because Sequence Models has independent evolution needs (different embedding model versions, different sequence-model families, no human-correction flow).
- Stuffing the producer parameters into the existing `region_detection_jobs` row: rejected because the producer is parameterized by hop / pad / model version that are orthogonal to detection.
- Re-using the cached Pass-1 embeddings parquet as-is: rejected because Pass-1 caches at the detector hop, while the producer needs an arbitrary hop and full SurfPerch coverage of the padded region (not just within-region windows).

**Consequences**:
- New migration 057 adds `continuous_embedding_jobs` with indices on `encoding_signature` and `status`.
- New ORM model in `src/humpback/models/sequence_models.py`; reuses the existing `JobStatus` enum from `models/processing.py`.
- New `/sequence-models/continuous-embeddings` API surface; documented in `docs/reference/sequence-models-api.md`.
- New `continuous_embeddings/{job_id}/` storage tree with atomic temp-rename writes for both `embeddings.parquet` and `manifest.json`.
- The default production embedder is intentionally a stub that raises until the SurfPerch + hydrophone-streaming integration lands; the worker contract is finalized today behind the injected `EmbedderProtocol`.
- Producer idempotency lookup is the service layer's responsibility — the worker never re-checks `encoding_signature` and assumes its job row is canonical.
- `processing/region_windowing.py` is added to the sensitive-components list because every downstream HMM sequence consumer relies on its merge/window-center semantics.

## ADR-057: CRNN region-based chunk embeddings as second Sequence Models source

**Date**: 2026-04-29
**Status**: Accepted
**Builds on**: ADR-050 (Pass 2 segmentation), ADR-056 (Sequence Models track)

**Context**: SurfPerch event-padded embeddings (ADR-056) cover only the immediate
neighborhood of Pass 2 events. To explore latent-state structure spanning
whole detection regions — and to leverage the segmentation CRNN's
already-learned representations — we want a second Sequence Models embedding
source: the Pass 2 BiGRU activations sliced into 250 ms chunks, computed per
Pass 1 detection region, with per-chunk metadata joining against Pass 2
events. The HMM consumer needs to keep working unchanged for SurfPerch jobs
while gaining three training-mode options (full-region, event-balanced,
event-only) for the new source.

**Decision**:
1. **Single-table dispatch on `model_version`**. `continuous_embedding_jobs`
   and `hmm_sequence_jobs` grow nullable columns (Alembic 061) covering the
   CRNN-only configuration; the worker, service, and API all dispatch on
   the `model_version` family. SurfPerch rows leave the new columns null;
   CRNN rows leave the SurfPerch-only columns null. Alternatives considered:
   per-source tables (rejected — duplicates HMM consumer code), source-kind
   discriminator column (rejected — redundant with `model_version`).
2. **Concat-and-project chunk embeddings with `IdentityProjection` default**.
   Eight 32 fps frames × 128-d BiGRU width = 1024-d chunk vector, passed
   through a pluggable `ChunkProjection`. Phase 1 ships `IdentityProjection`
   as the default (PCA + HMM consumer handles dim-reduction); the
   abstraction also covers `RandomProjection` and `PCAProjection` for future
   experiments. Alternative considered: average-pooling 8 frames into a
   128-d chunk vector — rejected because it discards within-chunk temporal
   structure that the BiGRU specifically encodes.
3. **Shared windowing helper extracted from Pass 2 inference**. Both Pass 2
   and the new CRNN extractor consume `iter_inference_windows()`. The
   Pass 2 refactor is the only edit to Pass 2 source files and is
   regression-tested for byte-identical `events.parquet` output before/after.

**Consequences**:
- Migration 061 adds nullable columns to `continuous_embedding_jobs` and
  `hmm_sequence_jobs`; SurfPerch-only configuration columns are relaxed to
  nullable so CRNN-source rows can leave them unset.
- Producer worker (`_run_region_crnn`) and HMM worker (`_run_region_crnn_hmm`)
  share the public entry points but dispatch internally on source kind.
- New `embeddings.parquet` schema variant for CRNN source includes
  `region_id`, `chunk_index_in_region`, `tier`, `event_overlap_fraction`,
  `nearest_event_id`, `distance_to_nearest_event_seconds`, and
  `call_probability` (mean per-frame sigmoid over 8 frames).
- New `states.parquet` schema variant for CRNN-source HMM jobs adds `tier`
  and `was_used_for_training`. `summary.json` adds per-state
  `tier_composition` aggregates so the frontend can render a stacked-bar
  strip without re-aggregating from `states.parquet`.
- Encoding signature differs between source kinds (CRNN signature folds in
  `region_detection_job_id`, `event_segmentation_job_id`,
  `crnn_checkpoint_sha256`, chunk geometry, projection config); SurfPerch
  signature is unchanged.
- API: `POST /sequence-models/continuous-embeddings` and
  `POST /sequence-models/hmm-sequences` return 422 when CRNN-only fields
  are sent on a SurfPerch source (and vice versa) or when
  `event_balanced_proportions` does not sum to 1.0 ± 1e-6.
- Frontend new-job creation forms gain a source-type toggle; the list page
  gains a `Source` badge column; the HMM detail page renders a per-state
  `Tier Composition` stacked-bar strip for CRNN-source jobs and uses the
  generalized `SpanNavBar` `itemLabel` prop (`Region` for CRNN source,
  `Event` for SurfPerch).
- Mixing SurfPerch and CRNN sources in a single HMM job is explicitly out
  of scope for Phase 1.

## ADR-058: First-class HMM motif extraction jobs

**Date**: 2026-04-30
**Status**: Accepted
**Builds on**: ADR-056, ADR-057

**Context**: ADR-056 identified motif mining as a downstream Sequence Models
interpretation stage. After the HMM Sequence workflow gained both SurfPerch
event-padded and CRNN region-based sources, motif extraction became important
enough to compare, rerun, cancel, and inspect as durable analysis output rather
than as a hidden cache behind the HMM detail page.

**Decision**: Add `motif_extraction_jobs` as a first-class Sequence Models job
type. Each job consumes one completed `hmm_sequence_jobs` row, collapses
consecutive repeated Viterbi states into symbolic sequences, mines n-grams,
filters by `minimum_occurrences` and `minimum_event_sources`, ranks motifs with
configurable weights, and writes `manifest.json`, `motifs.parquet`, and
`occurrences.parquet` under `motif_extractions/{job_id}/`.

Alternatives considered:
- On-demand motif cache keyed by config hash: rejected because motifs are a
  durable research artifact and need status, history, cancellation, and failure
  visibility.
- Automatic default motif generation inside the HMM worker: rejected because
  motif analysis should be an explicit downstream interpretation step, not an
  implicit side effect of HMM training.

**Consequences**:
- Migration 062 adds `motif_extraction_jobs` with idempotent
  `config_signature` lookup among queued/running/complete jobs.
- The queue and worker runner claim motif extraction jobs after HMM Sequence
  jobs.
- The API exposes `/sequence-models/motif-extractions` plus motif summary and
  occurrence endpoints.
- The HMM detail page gains a Motifs panel with create/status controls, advanced
  rank weights, ranked motif table, and event-midpoint aligned examples.
- CRNN `call_probability` remains optional in ranking; the default weight is
  null so ranking stays source-neutral unless explicitly enabled.

## ADR-059: Source-agnostic HMM interpretation loader Protocol

**Date**: 2026-05-01
**Status**: Accepted
**Builds on**: ADR-056, ADR-057

**Context**: ADR-057 deferred PCA/UMAP overlay, exemplars, and label
distribution for CRNN-source HMM jobs as Phase 2 of that source's spec. The
deferred path left `_load_overlay_inputs()` hardcoded to the SurfPerch
parquet schema, the worker's `_run_region_crnn_hmm()` skipping the
interpretation call, and the `OverlayPoint` / `ExemplarRecord` schemas
requiring SurfPerch's `merged_span_id` / `window_index_in_span` int fields —
none of which fit CRNN region UUIDs and chunk indices. Adding a third
source family later would compound the dispatch.

**Decision**: Introduce a `SequenceArtifactLoader` Protocol with a single
implementation per embedding source family (SurfPerch event-padded, CRNN
region-based). `generate_interpretations()` resolves the right loader at
runtime via `get_loader(source_kind)`; downstream pure functions
(`compute_overlay`, `select_exemplars`) consume a generic `OverlayInputs`
shape and never branch on source kind. The identifier model unifies on
`(sequence_id: str, position_in_sequence: int)` — SurfPerch stringifies its
int span id; CRNN passes its region UUID. Source-specific metadata flows
through `extras: dict[str, str | int | float | None]` on `WindowMeta` and
`ExemplarRecord` (CRNN populates `extras["tier"]`).

Alternatives considered:
- Schema-dispatch inside a single `_load_overlay_inputs()`: rejected because
  the conditional grows with each new source family and bleeds source
  knowledge into downstream code.
- Discriminated-union typing of `extras` per source kind: rejected as YAGNI
  for two source families; the dict shape is documented and extensible
  without a type tax.

**Consequences**:
- New module `src/humpback/sequence_models/loaders/` with `__init__.py`
  (Protocol + registry), `surfperch.py`, and `crnn_region.py`. Adding a
  new source family is one new loader file plus one registry entry.
- `overlay.parquet` and `exemplars.json` written by new jobs use the
  unified column / key names (`sequence_id`, `position_in_sequence`, plus
  `extras` on exemplars). Pre-PR SurfPerch artifacts on disk remain in the
  legacy shape; a transitional read-time adapter on `GET /overlay` and
  `GET /exemplars` translates legacy column / key names in-memory before
  serializing the response. Disk files are not rewritten by the adapter;
  the existing Refresh button rewrites them in unified form on demand.
- `_run_region_crnn_hmm()` now calls `generate_interpretations()` after
  `summary.json` is written, with the same `try/except` + `logger.warning`
  swallow pattern the SurfPerch path uses. Interpretation failure is
  non-fatal; HMM job status remains `complete`.
- Frontend `ExemplarRecord` interface gains `extras`; the gallery row
  renders a small tier badge when `record.extras?.tier` is a non-null
  string. SurfPerch records leave `extras` empty and render no badge.
  `PcaUmapScatter` is unchanged — it never read identifier columns.
- Label distribution remains SurfPerch-only pending Phase 2; the loader
  Protocol does not yet cover the label-distribution flow.

## ADR-060: Source-agnostic HMM label distribution with tier-aware storage

**Date**: 2026-05-01
**Status**: Accepted
**Builds on**: ADR-057, ADR-059

**Context**: ADR-059 unified PCA/UMAP overlay and exemplar generation
across SurfPerch and CRNN HMM jobs but explicitly deferred label
distribution as Phase 2. The deferred surfaces were:
`generate_label_distribution()` traversing `cej → EventSegmentationJob →
RegionDetectionJob → hydrophone_id` (a path that does not exist for CRNN
CEJs, which carry `region_detection_job_id` directly), the CRNN skip in
`POST /generate-interpretations/{id}`, and a flat
`states[state] = {label: count}` JSON shape that left no room to surface
the per-chunk tier classification CRNN already records on its
`states.parquet`.

**Decision**: Extend the existing `SequenceArtifactLoader` Protocol with
`load_label_distribution_inputs()` returning a generic
`LabelDistributionInputs` (`hydrophone_id`, `state_rows`,
`tier_per_row`). Each loader owns its source-specific traversal — the
service keeps the shared DetectionJob + VocalizationLabel SQL fetch and
the pure compute call. The on-disk shape becomes always-tiered:
SurfPerch buckets every row to a synthetic `"all"` tier key, CRNN
buckets each row to its own `event_core` / `near_event` / `background`
tier. The frontend chart collapses the tier dimension in `useMemo`
before building Plotly traces — identical visual to today, with the
richer tier-stratified data preserved on disk for a future tier-aware
UI.

Alternatives considered:
- A sibling `LabelDistributionLoader` Protocol: rejected — the same
  source-resolution knowledge already lives in the Phase 1 loader files,
  and a parallel Protocol would split that knowledge across two
  registries.
- Source-discriminated payload (flat for SurfPerch, nested for CRNN):
  rejected — two readers, two schemas, two TS types, and frontend code
  that must branch on source kind. The synthetic `"all"` bucket pays a
  tiny structural tax for a single reader on both sides.
- Forced one-shot regeneration of pre-PR SurfPerch files: rejected —
  Phase 1 already established the read-time legacy adapter pattern; the
  Refresh button rewrites in unified form on demand.

**Consequences**:
- New `LabelDistributionInputs` dataclass and Protocol method in
  `src/humpback/sequence_models/loaders/__init__.py`. Each loader file
  gains an async method that owns its source's hydrophone-traversal +
  tier-extraction logic.
- `compute_label_distribution()` gains a `tier_per_row: list[str] |
  None` parameter and emits the unified nested shape unconditionally.
  Length mismatch with `states` raises `ValueError`.
- `LabelDistributionResponse` becomes
  `dict[str, dict[str, dict[str, int]]]` and the frontend
  `LabelDistribution` TS type matches.
- The CRNN skip in `POST /generate-interpretations/{id}` is removed;
  `label_distribution_generated` becomes a constant `true` in the
  response. CRNN HMM jobs now produce label-distribution artifacts via
  Refresh and on first GET, identical to SurfPerch.
- A read-time legacy adapter on `GET /label-distribution` projects
  pre-PR flat-shape SurfPerch JSON to the unified nested shape
  in-memory; disk files are unchanged until the Refresh button rewrites
  them. Mirrors the Phase 1 overlay/exemplar adapter pattern; the
  comment names the adapter as transitional.
- Frontend `LabelDistributionChart` collapses the tier dimension in
  `useMemo`. SurfPerch jobs render unchanged; CRNN jobs render the same
  single-stack chart with bars summed across tiers. The
  `data-testid="hmm-label-distribution"` attribute and Plotly layout are
  preserved.
- The `"all"` tier key is reserved for sources without a tier dimension;
  future sources with a real tier dimension must pick distinct keys.

---

## ADR-061: Masked-transformer sequence model parallel to HMM

**Date**: 2026-05-01
**Status**: Accepted
**Builds on**: ADR-056, ADR-057, ADR-058, ADR-059, ADR-060

**Context**: The Sequence Models track shipped HMM-based latent-state
discovery over CRNN region-based and SurfPerch event-padded continuous
embeddings, plus motif extraction over collapsed Viterbi-state n-grams
(ADR-056 through ADR-060). The Event-Conditioned Motif Discovery
reference pipeline calls for a second consumer parallel to the HMM: a
masked-span transformer encoder trained on the same continuous
embeddings, producing contextualized embeddings (Z) that are then
tokenized via k-means with a softmax-temperature confidence and fed to
the same motif-extraction pipeline. The k-sweep is intrinsic to the
workflow — researchers want to explore vocabulary sizes — and one
training pass should support multiple k-means tokenizers without
retraining the transformer. The HMM "state" naming is HMM-specific; a
generalized "label" naming makes the loader/interpretation/motif stack
source-agnostic across HMM and masked-transformer parents.

**Decision**: Add a `masked_transformer_jobs` table (one row per
upstream + training config), persist multiple per-k tokenization
artifacts under `k<N>/` subdirs of the job dir, generalize the existing
HMM loader stack to read a normalized `decoded.parquet` with column
`label` (renamed from `state`), and generalize `motif_extraction_jobs`
to point at either parent kind via a `parent_kind` discriminator + XOR
parent FKs + a parent-kind-conditional `k` column. Training-signature
idempotency excludes `k_values` so a completed job can extend its
k-sweep without retraining. Mask-weight bias scales the loss higher on
event-adjacent positions when the upstream source carries tier metadata.
K-means confidence uses softmax with τ auto-fit per job from the median
pairwise centroid distance, dropping into the existing
`max_state_probability` UI slots without bespoke chart code. Training
runs on MPS/CUDA with synthetic-batch validation against CPU and a
fallback to CPU on validation failure, mirroring Pass 2/3 inference.
The frontend extracts a generic `DiscreteSequenceBar` (modes `rows` |
`single-row`) from the existing `HMMStateBar` so both detail pages share
a single timeline strip implementation.

Alternatives considered:
- **Separate motif-extraction tables per parent kind**: rejected —
  duplicates the entire ranking/storage/UI surface for a workflow that
  is genuinely parent-agnostic at the algorithm level. The XOR + check
  constraint adds a few lines of schema for one shared codepath.
- **Per-k separate jobs (one row per (upstream, training, k))**:
  rejected — forces a full transformer retrain per k, prevents the
  extend-k-sweep follow-up flow, and clutters the jobs list with many
  near-duplicate rows.
- **Custom transformer-token-only motif consumer**: rejected — the
  motif extraction algorithm is already loader-driven and needs only
  per-chunk `(label, confidence)` rows + tier metadata. Sharing the
  loader is the smaller change.
- **Keep the `state` column name**: rejected — the column is now
  consumed by both HMM Viterbi states and k-means tokens, and "label"
  is the right cross-source noun. The HMM API serialization layer maps
  `label` → `viterbi_state` so the existing HMM API contract and
  frontend are unchanged.
- **Token-bigram heatmap on the masked-transformer detail page**:
  rejected for v1 — k-means tokens are not a Markov chain, so transition
  matrices are semantically wrong; the existing motif n-gram view is the
  right semantic surface for token co-occurrence.

**Consequences**:
- New `masked_transformer_jobs` table (migration 063); new
  `MaskedTransformer` PyTorch module + `train_masked_transformer` +
  `extract_contextual_embeddings` (`src/humpback/sequence_models/masked_transformer.py`).
- New `tokenization.py` module with `fit_kmeans_token_model`,
  `decode_tokens`, `compute_run_lengths`. τ auto-fit per job; confidence
  is softmax over `−‖z − μ‖² / τ` and lives in `max_state_probability`
  on disk for UI parity.
- Per-k artifact fan-out under `k<N>/` (decoded.parquet, kmeans.joblib,
  overlay.parquet, exemplars.json, label_distribution.json,
  run_lengths.json). Atomic write via `k<N>.tmp/` → rename so a failed
  per-k pass leaves no half-written subdir.
- Training-signature idempotency excludes `k_values`; a completed job
  exposes `POST /extend-k-sweep` to add new k values without retraining.
  The follow-up worker pass writes only the new `k<N>/` subdirs and
  leaves `transformer.pt` + `contextual_embeddings.parquet` untouched.
- HMM worker writes `decoded.parquet` with column `label` (renamed from
  the old `states.parquet` / `state` column). A read-time legacy adapter
  in the loader stack falls back to the old filename + column on
  pre-migration HMM job dirs so existing completed jobs remain
  functional without manual migration. The HMM API serialization layer
  continues to expose `viterbi_state` field names, and the frontend is
  unchanged.
- `motif_extraction_jobs` generalization (migration 064) adds
  `parent_kind text NOT NULL`, nullable `masked_transformer_job_id` FK,
  nullable `k`, and makes `hmm_sequence_job_id` nullable. CHECK
  constraint enforces XOR between parent FKs + consistency with
  `parent_kind`, and `k IS NOT NULL` iff `parent_kind =
  "masked_transformer"`. Existing HMM-parent rows backfill with
  `parent_kind = "hmm"`.
- `MotifExtractionConfig.config_signature()` gains `parent_kind`, parent
  FK, and `k`; the worker constructs the loader with the appropriate
  `decoded_artifact_path` (HMM → `<hmm_job_dir>/decoded.parquet`;
  masked-transformer → `<mt_job_dir>/k<N>/decoded.parquet`). The
  extraction algorithm is parent-agnostic.
- Mask-weight bias scales per-position loss by tier weight (event_core
  1.5, near_event 1.2, background 0.5) when the source carries tier
  metadata. Default on for CRNN sources; SurfPerch sources without tier
  metadata fall back to uniform weighting.
- Device validation runs forward+backward on a fixed synthetic batch on
  both CPU and the chosen accelerator before training; outside a
  tolerance band the worker records `fallback_reason` and proceeds on
  CPU. `chosen_device` and `fallback_reason` persist on the job row and
  surface as a UI badge, mirroring Pass 2/3 inference.
- Frontend extracts a generic `DiscreteSequenceBar` (props add `mode:
  "rows" | "single-row"`, `numLabels`, `colorPalette`,
  `tooltipFormatter?`) and a `RegionNavBar` from the existing
  `HMMSequenceDetailPage`. HMM detail page swaps to
  `DiscreteSequenceBar mode="rows"` and the extracted region nav, no
  visual change. `STATE_COLORS` renames to `LABEL_COLORS`; new
  `labelColor(idx, total)` helper covers `total > 30` via an HSL ramp.
- A third Sequence Models nav card "Masked Transformer" routes to
  `/app/sequence-models/masked-transformer` (jobs list) and
  `/app/sequence-models/masked-transformer/:jobId` (detail). The detail
  page reads `?k=` from the URL via `useSearchParams`; per-k panels key
  cache on `(jobId, k)` so KPicker switches don't remount the page.
- `decoded.parquet` with column `label` becomes the behavioral contract
  for any future sequence-model consumer — the loader Protocol
  (ADR-059) reads it from an explicit path so new sources can plug in
  without touching downstream interpretation or motif code.
- SurfPerch source for the masked transformer is deferred to a
  follow-up; the loader and config abstractions stay shaped to allow
  it. Multi-GPU / distributed training, pretrained transformer
  initialization, and token-bigram heatmaps are out of scope for v1.

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
