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
  {detection_job_id}/detection_embeddings.parquet (row_id, embedding, confidence)
  {detection_job_id}/detections.tsv               (generated on-the-fly for download; not persisted)
  {detection_job_id}/window_diagnostics.parquet   (local: single file; hydrophone: shard directory)
  {detection_job_id}/run_summary.json
/training_datasets/
  {dataset_id}/embeddings.parquet                 (unified training embeddings: row_index, embedding, source_type, source_id, filename, start_sec, end_sec, confidence)
/timeline_cache/
  {job_id}/{zoom_level}/tile_{NNNN}.png   (per-job LRU-evicted spectrogram tiles)
```

Timeline audio endpoint supports `format=mp3` for compressed playback (128kbps mono, up to 600s segments).

## Extraction Output

- Positive labels: `{positive_sample_path}/{humpback|orca}/{hydrophone_id}/YYYY/MM/DD/{start}_{end}.flac`
- Negative labels: `{negative_sample_path}/{ship|background}/{hydrophone_id}/YYYY/MM/DD/{start}_{end}.flac`
- Local extraction: same structure without `{hydrophone_id}/` level
- Every `.flac` also gets a same-basename `.png` spectrogram sidecar
