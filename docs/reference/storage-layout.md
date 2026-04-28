# Storage Layout

> Read this when working on file paths, artifact storage, or storage-related configuration.

```
/audio/
  raw/{audio_file_id}/original.(wav|mp3|flac)    (uploaded files only; imported files are read from source_folder)
/embeddings/
  {model_version}/{audio_file_id}/{encoding_signature}.parquet
  {model_version}/{audio_file_id}/{encoding_signature}.tmp.parquet
/clusters/
  {clustering_job_id}/clusters.json
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
  {job_id}/embeddings.parquet  (Sequence Models PR 1 producer: merged_span_id, window_index_in_span, audio_file_id, start_timestamp, end_timestamp, is_in_pad, source_region_ids, embedding)
  {job_id}/manifest.json       (vector_dim, hop/pad/window settings, total_regions, merged_spans, total_windows, per-span summaries)
/hmm_sequences/
  {job_id}/pca_model.joblib        (fitted PCA model)
  {job_id}/hmm_model.joblib        (fitted GaussianHMM model)
  {job_id}/states.parquet          (decoded windows: merged_span_id, window_index_in_span, audio_file_id, start_timestamp, end_timestamp, is_in_pad, source_region_ids, viterbi_state, state_posterior, max_state_probability, was_used_for_training)
  {job_id}/transition_matrix.npy   (n_states × n_states row-normalized transition matrix)
  {job_id}/state_summary.json      (per-state occupancy, mean_dwell_frames, dwell_histogram)
  {job_id}/training_log.json       (training hyperparameters and result stats)
  {job_id}/pca_overlay.parquet     (PR 3: 2-D PCA + UMAP projections colored by viterbi_state)
  {job_id}/label_distribution.json (PR 3: per-state vocalization-label counts from center-time join)
  {job_id}/exemplars/exemplars.json (PR 3: per-state high-confidence, nearest-centroid, boundary exemplar windows)
/timeline_cache/
  {job_id}/{zoom_level}/tile_{NNNN}.png   (PCEN-normalized spectrogram tiles, LRU-evicted per job)
  {job_id}/.cache_version                  (integer; current is 2. Migrations run on first access when missing or lower)
  {job_id}/.audio_manifest.json            (optional persisted HLS segment manifest for reuse across resampling rates)
  {job_id}/.prepare_plan.json              (active prepare scope used by /prepare-status)
  {job_id}/.last_access                    (mtime sentinel for per-job LRU eviction)
  {job_id}/.prepare.lock                   (advisory flock for exclusive prepare ownership)
```

Timeline audio endpoint supports `format=mp3` for compressed playback (128kbps mono, up to 600s segments). Playback audio is RMS-scaled to `playback_target_rms_dbfs` with a `tanh` soft-clip before encoding.

## Extraction Output

- Positive labels: `{positive_sample_path}/{humpback|orca}/{hydrophone_id}/YYYY/MM/DD/{start}_{end}.flac`
- Negative labels: `{negative_sample_path}/{ship|background}/{hydrophone_id}/YYYY/MM/DD/{start}_{end}.flac`
- Local extraction: same structure without `{hydrophone_id}/` level
- Every `.flac` also gets a same-basename `.png` spectrogram sidecar
