# Data Model Summary

> Read this when working on database models, migrations, or API schemas.

Condensed model reference. For full field lists, see `src/humpback/database.py`.

- **ModelConfig** (`model_configs`) — ML model registry entry (name, path, vector_dim, model_type, input_format, is_default). `TFLiteModelConfig` is a backward-compatible alias.
- **AudioFile** (`audio_files`) — uploaded/imported audio (filename, folder_path, source_folder, checksum_sha256, duration_seconds, sample_rate_original)
- **AudioMetadata** (`audio_metadata`) — optional editable metadata per audio file (tag_data, visual_observations, group_composition, prey_density_proxy — all JSON)
- **ProcessingJob** (`processing_jobs`) — encoding job (audio_file_id FK, encoding_signature, model_version, window_size_seconds, target_sample_rate, feature_config JSON, status, warning_message)
- **EmbeddingSet** (`embedding_sets`) — one per audio+encoding_signature (parquet_path, model_version, vector_dim). Embeddings stored in Parquet, not SQL.
- **SearchJob** (`search_jobs`) — ephemeral similarity search, deleted after results returned (detection_job_id, top_k, metric, embedding_set_ids, embedding_vector)
- **ClusteringJob** (`clustering_jobs`) — clustering run (embedding_set_ids JSON, parameters JSON, metrics_json, refined_from_job_id)
- **Cluster** (`clusters`) — one per cluster label per job (clustering_job_id FK, cluster_label, size, metadata_summary JSON)
- **ClusterAssignment** (`cluster_assignments`) — links cluster to embedding row index (cluster_id FK, embedding_row_id)
- **ClassifierModel** (`classifier_models`) — binary classifier artifact (name, model_path .joblib, model_version, vector_dim, training_summary JSON, `training_source_mode`, optional `source_candidate_id`/`source_model_id`, `promotion_provenance` JSON for candidate-backed models)
- **ClassifierTrainingJob** (`classifier_training_jobs`) — training run (positive/negative_embedding_set_ids JSON, classifier_model_id set on completion; candidate-backed jobs also persist `source_mode`, `source_candidate_id`, `source_model_id`, `manifest_path`, `training_split_name`, `promoted_config`, `source_comparison_context`)
- **AutoresearchCandidate** (`autoresearch_candidates`) — imported autoresearch bundle for review/promotion (artifact paths, promoted config JSON, split metrics/deltas, source counts, warnings, source model metadata, exact-replay status, optional linked training/model ids)
- **DetectionJob** (`detection_jobs`) — local or hydrophone detection scan (classifier_model_id FK, audio_folder, confidence/hop/threshold params, detection_mode, window_selection, min_prominence, output_tsv_path, result_summary JSON, extract_* columns)
- **LabelProcessingJob** (`label_processing_jobs`) — score-based audio sample extraction (classifier_model_id, annotation_folder, audio_folder, output_root, parameters JSON, result_summary JSON)
- **VocalizationLabel** (`vocalization_labels`) — per-detection vocalization type label (detection_job_id, row_id, label, source). Linked to detection rows by stable UUID `row_id`.
- **RetrainWorkflow** (`retrain_workflows`) — orchestrated reimport+reprocess+retrain (status, step, provenance)
- **VocalizationType** (`vocalization_types`) — managed vocabulary entry for vocalization type classification (name, description, unique name constraint)
- **VocalizationClassifierModel** (`vocalization_models`) — multi-label vocalization model artifact (name, model_dir_path, vocabulary_snapshot JSON, per_class_thresholds JSON, per_class_metrics JSON, is_active)
- **VocalizationTrainingJob** (`vocalization_training_jobs`) — multi-label training run (source_config JSON with embedding_set_ids + detection_job_ids, parameters JSON, vocalization_model_id set on completion)
- **VocalizationInferenceJob** (`vocalization_inference_jobs`) — scoring run (vocalization_model_id FK, source_type, source_id, output_path to predictions parquet)
- **DetectionEmbeddingJob** (`detection_embedding_jobs`) — post-hoc detection embedding generation and sync (detection_job_id, mode [full/sync], progress_current/total, result_summary JSON, status)
- **TrainingDataset** (`training_datasets`) — unified editable snapshot of training embeddings and labels (name, source_config JSON, parquet_path, total_rows, vocabulary JSON)
- **TrainingDatasetLabel** (`training_dataset_labels`) — per-row label on a training dataset (training_dataset_id FK, row_index, label, source). Indexed on (training_dataset_id, row_index).
