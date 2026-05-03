# Storage Layout

> Read this when working on file paths, artifact storage, or storage-related configuration.

```
/clusters/
  {clustering_job_id}/clusters.json              (retained Vocalization / Clustering jobs only)
  {clustering_job_id}/assignments.parquet
  {clustering_job_id}/umap_coords.parquet
  {clustering_job_id}/parameter_sweep.json
  {clustering_job_id}/report.json                (fragmentation report)
  {clustering_job_id}/classifier_report.json     (opt-in classifier baseline)
  {clustering_job_id}/label_queue.json           (opt-in active learning queue)
  {clustering_job_id}/stability_summary.json     (opt-in stability evaluation)
  {clustering_job_id}/refinement_report.json     (opt-in metric learning refinement)
  {clustering_job_id}/refined_embeddings.parquet (opt-in refined embedding vectors for re-clustering)
/classifiers/
  {classifier_model_id}/model.joblib              (StandardScaler + LogisticRegression pipeline)
  {classifier_model_id}/training_summary.json
/vocalization_models/
  {vocalization_model_id}/{type_name}.joblib       (per-type sklearn pipeline)
  {vocalization_model_id}/metadata.json            (vocabulary, thresholds, metrics)
/vocalization_inference/
  {inference_job_id}/predictions.parquet           (row_id + per-type score columns; legacy: filename/start_sec/end_sec or start_utc/end_utc)
/detections/
  {detection_job_id}/detection_rows.parquet       (canonical editable row store; rows keyed by stable row_id UUID)
  {detection_job_id}/embeddings/{model_version}/detection_embeddings.parquet (row_id, embedding, confidence; model-versioned since ADR-055)
  {detection_job_id}/detections.tsv               (generated on-the-fly for download; not persisted)
  {detection_job_id}/window_diagnostics.parquet   (local: single file; hydrophone: shard directory)
  {detection_job_id}/run_summary.json
/training_datasets/
  {dataset_id}/embeddings.parquet                 (unified training embeddings: row_index, embedding, source_type, source_id, filename, start_sec, end_sec, confidence)
/hyperparameter/
  manifests/{manifest_id}/manifest.json              (generated training data manifest with split assignments)
  searches/{search_id}/search_history.json           (per-trial config, metrics, objective)
  searches/{search_id}/best_run.json                 (best trial config + metrics)
  searches/{search_id}/top_false_positives.json      (highest-scoring false positives from best run)
/call_parsing/
  regions/{job_id}/trace.parquet               (Pass 1 scalar confidence trace)
  regions/{job_id}/embeddings.parquet          (Pass 1 cached 1536-d Perch embedding vectors: time_sec, embedding)
  regions/{job_id}/regions.parquet             (detected regions with padded bounds)
  segmentation/{job_id}/events.parquet         (Pass 2 framewise segmentation events)
  classification/{job_id}/typed_events.parquet (Pass 3 per-event classification scores)
  window_classification/{job_id}/window_scores.parquet (wide format: time_sec, region_id, one float64 column per vocalization type)
/continuous_embeddings/
  {job_id}/embeddings.parquet  (Sequence Models producer; schema is source-specific)
                                 SurfPerch source: merged_span_id, event_id, window_index_in_span, audio_file_id, start_timestamp, end_timestamp, is_in_pad, embedding
                                 CRNN source (ADR-057): region_id, audio_file_id, hydrophone_id, chunk_index_in_region, start_timestamp, end_timestamp, is_in_pad, call_probability, event_overlap_fraction, nearest_event_id (nullable), distance_to_nearest_event_seconds (nullable), tier ∈ {event_core, near_event, background}, embedding
  {job_id}/manifest.json       (source_kind discriminator + per-source counters)
                                 SurfPerch: vector_dim, hop/pad/window settings, total_events, merged_spans, total_windows, per-span summaries
                                 CRNN: vector_dim, region_detection_job_id, event_segmentation_job_id, crnn_checkpoint_sha256, chunk_size/hop, projection_kind/dim, total_regions, total_chunks, per-region summaries
/hmm_sequences/
  {job_id}/pca_model.joblib        (fitted PCA model)
  {job_id}/hmm_model.joblib        (fitted GaussianHMM model)
  {job_id}/decoded.parquet         (decoded sequences; renamed from `states.parquet` and column `state` → `label` in migration 063 per ADR-061. A read-time loader shim falls back to legacy `states.parquet`/`state` for pre-migration job dirs. Schema is source-specific):
                                     SurfPerch: merged_span_id, window_index_in_span, audio_file_id, start_timestamp, end_timestamp, is_in_pad, event_id, label, state_posterior, max_state_probability, was_used_for_training
                                     CRNN (ADR-057): region_id, chunk_index_in_region, audio_file_id (nullable), start_timestamp, end_timestamp, is_in_pad, tier, label, state_posterior, max_state_probability, was_used_for_training
  {job_id}/transition_matrix.npy   (n_states × n_states row-normalized transition matrix)
  {job_id}/state_summary.json      (per-state occupancy, mean_dwell_frames, dwell_histogram; CRNN-source jobs additionally include source_kind, training_mode, tier_composition[])
  {job_id}/training_log.json       (training hyperparameters and result stats; CRNN-source jobs include training_mode + tier_proportions + sub-sequence config)
  {job_id}/pca_overlay.parquet     (PR 3: 2-D PCA + UMAP projections colored by label)
  {job_id}/label_distribution.json (PR 3: per-state vocalization-label counts from center-time join; ADR-060 nested `state[tier][label] = count` shape, SurfPerch buckets to synthetic `"all"` tier)
  {job_id}/exemplars/exemplars.json (PR 3: per-state high-confidence, nearest-centroid, boundary exemplar windows; CRNN records carry `extras.tier`)
/masked_transformer_jobs/
  {job_id}/transformer.pt              (PyTorch state_dict for the trained MaskedTransformer module)
  {job_id}/contextual_embeddings.parquet (Z extracted from the trained encoder; per-chunk contextual embedding vectors)
  {job_id}/loss_curve.json             (per-epoch train + val loss history)
  {job_id}/reconstruction_error.parquet (per-chunk reconstruction MSE; consumed by the timeline ConfidenceStrip)
  {job_id}/k<N>/decoded.parquet        (per-chunk k-means tokens: sequence_id, position, label, confidence; confidence is softmax-temperature, drops into the same `max_state_probability` UI slot as HMM)
  {job_id}/k<N>/kmeans.joblib          (fitted KMeans + τ for that k)
  {job_id}/k<N>/overlay.parquet        (2-D PCA + UMAP projections colored by token label; OverlayPoint shape)
  {job_id}/k<N>/exemplars.json         (per-token high-confidence, nearest-centroid, boundary exemplars)
  {job_id}/k<N>/label_distribution.json (per-token vocalization-label counts; ADR-060 nested shape)
  {job_id}/k<N>/run_lengths.json       (per-token run-length arrays)
                                       Per-k subdirs are written atomically: the worker stages to `k<N>.tmp/` and renames to `k<N>/` only after every per-k artifact is written.
/motif_extractions/
  {job_id}/manifest.json           (schema/version, parent_kind ∈ {hmm, masked_transformer}, parent FK + k for masked-transformer parent, source HMM/CEJ ids, source_kind, extraction config, config_signature, counters, event_source_key_strategy)
  {job_id}/motifs.parquet          (motif_key, states, length, occurrence_count, event_source_count, audio_source_count, group_count, event_core_fraction, background_fraction, mean_call_probability nullable, duration stats, rank_score, example_occurrence_ids)
  {job_id}/occurrences.parquet     (occurrence_id, motif_key, states, source_kind, group_key, event_source_key, audio_source_key nullable, token/raw ranges, absolute timestamps, event/background fractions, mean_call_probability nullable, event-midpoint alignment fields)
/timeline_cache/
  spans/{span_key}/.source.json            (hydrophone id, source identity, start/end timestamps, deterministic span key)
  spans/{span_key}/.audio_manifest.json    (optional persisted HLS segment manifest shared by compatible timeline consumers)
  spans/{span_key}/.prepare_plan.json      (active prepare scope used by /prepare-status)
  spans/{span_key}/.last_access            (mtime sentinel for span access)
  spans/{span_key}/{renderer_id}/v{renderer_version}/{zoom_level}/f{min}-{max}/w{width}_h{height}/tile_{NNNN}.png
                                           (PCEN-normalized spectrogram tiles; renderer id/version and tile geometry are cache identity)
  {job_id}/.cache_version                  (legacy per-job cache marker; current is 2. Migrations run on first legacy access)
  {job_id}/.prepare.lock                   (advisory flock for exclusive classifier prepare ownership)
/cleanup-manifests/
  {timestamp}-legacy-workflow-removal.json (manifest written by scripts/cleanup_legacy_workflows.py)
```

Timeline tile image writes now use the shared `spans/{span_key}/...`
repository. The `span_key` is derived from hydrophone id, source identity
(`local_cache_path` or configured archive/cache root), and the job start/end
timestamps. Classifier timelines, region timelines, HMM detail pages, masked
transformer detail pages, and review workspaces therefore share one tile set
when they point at the same hydrophone span. The legacy `{job_id}/...` layout is
kept only for migration/version-marker compatibility.

Legacy roots removed by archive-backed cleanup:

- `/audio/raw/`
- `/embeddings/`
- `/label_processing/`

Those paths may still exist in historical environments until
`scripts/cleanup_legacy_workflows.py --apply --archive-root ...` is run, but
they are no longer part of the active storage contract.

Timeline audio endpoint supports `format=mp3` for compressed playback (128kbps mono, up to 600s segments). Playback audio is RMS-scaled to `playback_target_rms_dbfs` with a `tanh` soft-clip before encoding.

## Extraction Output

- Positive labels: `{positive_sample_path}/{humpback|orca}/{hydrophone_id}/YYYY/MM/DD/{start}_{end}.flac`
- Negative labels: `{negative_sample_path}/{ship|background}/{hydrophone_id}/YYYY/MM/DD/{start}_{end}.flac`
- Local extraction: same structure without `{hydrophone_id}/` level
- Every `.flac` also gets a same-basename `.png` spectrogram sidecar
