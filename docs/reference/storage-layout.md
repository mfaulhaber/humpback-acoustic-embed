# Storage Layout

> Agent startup context lives in `docs/agent-context/`. Read the relevant
> domain capsule first. Use this file when working on file paths, artifact
> storage, or storage-related configuration.

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
/event_encoders/
  {job_id}/event_vectors.parquet    (event_id, timing/provenance, descriptor columns including ridge_log_frequency_slope, pooled embedding block, descriptor_vector, final event_vector)
  {job_id}/event_tokens.parquet     (one row per valid k/event: token_id, token_label, distances, token_confidence, descriptor columns including ridge_log_frequency_slope)
  {job_id}/token_sequences.parquet  (one ordered token sequence per valid k/source_sequence_key)
  {job_id}/manifest.json            (source job ids, tokenizer/provenance signatures, configs, descriptor_feature_names, valid/invalid k values, counters)
  {job_id}/report.json              (summary, token distributions, exemplar event ids, descriptor_feature_names, descriptor summaries, sequence preview)
  {job_id}/preprocess.joblib        (PCA model when used plus descriptor robust-scaling state)
  {job_id}/kmeans_k{k}.joblib       (one k-means model per valid k)
/timeline_cache/
  spans/{span_key}/.source.json            (hydrophone id, source identity, start/end timestamps, deterministic span key)
  spans/{span_key}/.audio_manifest.json    (optional persisted HLS segment manifest shared by compatible timeline consumers)
  spans/{span_key}/.prepare_plan.json      (active prepare scope used by /prepare-status)
  spans/{span_key}/.last_access            (mtime sentinel for span access)
  spans/{span_key}/{renderer_id}/v{renderer_version}/{zoom_level}/f{min}-{max}/w{width}_h{height}/tile_{NNNN}.png
                                           (PCEN-normalized spectrogram tiles; renderer id/version and tile geometry are cache identity)
  {job_id}/.cache_version                  (legacy per-job cache marker; current is 2. Migrations run on first legacy access)
  {job_id}/.prepare.lock                   (advisory flock for exclusive classifier prepare ownership)
```

Timeline tile image writes now use the shared `spans/{span_key}/...`
repository. The `span_key` is derived from hydrophone id, source identity
(`local_cache_path` or configured archive/cache root), and the job start/end
timestamps. Classifier timelines, region timelines, and review workspaces share
one tile set when they point at the same hydrophone span. The legacy
`{job_id}/...` layout is kept only for migration/version-marker compatibility.

Legacy roots retired from the active storage contract:

- `/audio/raw/`
- `/embeddings/`
- `/label_processing/`
- `/hmm_sequences/`
- `/masked_transformer_jobs/`
- `/motif_extractions/`

Those paths may still exist in historical environments, but current runtime
code should not create or depend on them.

Timeline audio endpoint supports `format=mp3` for compressed playback (128kbps mono, up to 600s segments). Playback audio is RMS-scaled to `playback_target_rms_dbfs` with a `tanh` soft-clip before encoding.

Event Encoder ridge slope uses per-event STFT analysis during descriptor
extraction and persists only the scalar `ridge_log_frequency_slope` descriptor.
Continuous Embedding artifacts do not store full STFT matrices or ridge trace
sidecars for this feature.

## Extraction Output

- Positive labels: `{positive_sample_path}/{humpback|orca}/{hydrophone_id}/YYYY/MM/DD/{start}_{end}.flac`
- Negative labels: `{negative_sample_path}/{ship|background}/{hydrophone_id}/YYYY/MM/DD/{start}_{end}.flac`
- Local extraction: same structure without `{hydrophone_id}/` level
- Every `.flac` also gets a same-basename `.png` spectrogram sidecar
