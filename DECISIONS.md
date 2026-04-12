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
