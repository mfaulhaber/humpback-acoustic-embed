// ---- Audio ----

export interface AudioMetadata {
  id: string;
  audio_file_id: string;
  tag_data: Record<string, unknown> | null;
  visual_observations: Record<string, unknown> | null;
  group_composition: Record<string, unknown> | null;
  prey_density_proxy: Record<string, unknown> | null;
}

export interface AudioFile {
  id: string;
  filename: string;
  folder_path: string;
  source_folder: string | null;
  checksum_sha256: string;
  duration_seconds: number | null;
  sample_rate_original: number | null;
  created_at: string;
  metadata: AudioMetadata | null;
}

export interface FolderImportResult {
  folder_path: string;
  imported: number;
  skipped: number;
  errors: string[];
}

export interface SpectrogramData {
  window_index: number;
  sample_rate: number;
  window_size_seconds: number;
  shape: number[];
  data: number[][];
  total_windows: number;
  min_db: number;
  max_db: number;
  y_axis_hz: number[];
  x_axis_seconds: number[];
}

export interface EmbeddingSimilarity {
  embedding_set_id: string;
  vector_dim: number;
  num_windows: number;
  row_indices: number[];
  similarity_matrix: number[][];
}

// ---- Folder Delete ----

export interface AffectedClusteringJob {
  id: string;
  status: string;
  overlapping_embedding_set_ids: string[];
}

export interface FolderDeletePreview {
  folder_path: string;
  audio_file_count: number;
  embedding_set_count: number;
  processing_job_count: number;
  affected_clustering_jobs: AffectedClusteringJob[];
  has_clustering_conflicts: boolean;
}

export interface FolderDeleteResult {
  folder_path: string;
  deleted_audio_files: number;
  deleted_embedding_sets: number;
  deleted_processing_jobs: number;
  deleted_clustering_jobs: number;
}

// ---- Processing ----

export interface ProcessingJobCreate {
  audio_file_id: string;
  model_version?: string | null;
  window_size_seconds?: number;
  target_sample_rate?: number;
  feature_config?: Record<string, unknown> | null;
}

export interface ProcessingJob {
  id: string;
  audio_file_id: string;
  status: "queued" | "running" | "complete" | "failed" | "canceled";
  encoding_signature: string;
  model_version: string;
  window_size_seconds: number;
  target_sample_rate: number;
  feature_config: Record<string, unknown> | null;
  error_message: string | null;
  warning_message: string | null;
  created_at: string;
  updated_at: string;
  skipped: boolean;
}

export interface EmbeddingSet {
  id: string;
  audio_file_id: string;
  encoding_signature: string;
  model_version: string;
  window_size_seconds: number;
  target_sample_rate: number;
  vector_dim: number;
  parquet_path: string;
  created_at: string;
}

// ---- Clustering ----

export interface ClusteringJobCreate {
  embedding_set_ids: string[];
  parameters?: Record<string, unknown> | null;
  refined_from_job_id?: string | null;
}

export interface ClusteringJob {
  id: string;
  status: "queued" | "running" | "complete" | "failed" | "canceled";
  embedding_set_ids: string[];
  parameters: Record<string, unknown> | null;
  error_message: string | null;
  metrics: Record<string, unknown> | null;
  refined_from_job_id: string | null;
  created_at: string;
  updated_at: string;
}

export interface ClusterOut {
  id: string;
  clustering_job_id: string;
  cluster_label: number;
  size: number;
  metadata_summary: Record<string, unknown> | null;
}

export interface ClusterAssignment {
  id: string;
  cluster_id: string;
  embedding_set_id: string;
  embedding_row_index: number;
}

export interface VisualizationData {
  x: number[];
  y: number[];
  cluster_label: number[];
  embedding_set_id: string[];
  embedding_row_index: number[];
  audio_filename: string[];
  audio_file_id: string[];
  window_size_seconds: number[];
  category: string[];
}

export interface ClusteringMetrics {
  silhouette_score?: number;
  davies_bouldin_index?: number;
  calinski_harabasz_score?: number;
  n_clusters?: number;
  noise_count?: number;
  adjusted_rand_index?: number;
  normalized_mutual_info?: number;
  n_categories?: number;
  category_metrics?: Record<string, unknown>;
  [key: string]: unknown;
}

export interface ParameterSweepPoint {
  min_cluster_size?: number;
  k?: number;
  silhouette_score: number | null;
  n_clusters: number;
  algorithm?: string;
  selection_method?: string;
  adjusted_rand_index?: number | null;
  normalized_mutual_info?: number | null;
}

export interface DendrogramCoords {
  icoord: number[][];
  dcoord: number[][];
}

export interface DendrogramData {
  categories: string[];
  cluster_labels: string[];
  values: number[][];
  raw_counts: number[][];
  row_dendrogram: DendrogramCoords;
  col_dendrogram: DendrogramCoords;
}

// ---- Fragmentation ----

export interface CategoryFragmentation {
  n_total: number;
  n_non_noise: number;
  n_noise: number;
  noise_rate: number;
  top1_mass: number;
  top2_mass: number;
  top3_mass: number;
  entropy: number;
  normalized_entropy: number;
  neff: number;
  gini: number;
}

export interface ClusterFragmentation {
  size: number;
  dominant_category: string;
  dominant_mass: number;
  cluster_entropy: number;
  cluster_entropy_norm: number;
}

export interface GlobalFragmentation {
  mean_entropy_norm: number;
  mean_neff: number;
  mean_noise_rate: number;
  mean_cluster_entropy_norm: number;
}

export interface FragmentationSummary {
  n_categories: number;
  n_clusters: number;
  n_total: number;
  n_noise_total: number;
  overall_noise_rate: number;
}

export interface FragmentationReport {
  job_id: string;
  category_fragmentation: Record<string, CategoryFragmentation>;
  cluster_fragmentation: Record<string, ClusterFragmentation>;
  global_fragmentation: GlobalFragmentation;
  summary: FragmentationSummary;
}

// ---- Stability ----

export interface StabilityRunMetrics {
  run_index: number;
  seed: number;
  n_clusters: number;
  noise_fraction: number;
  silhouette_score: number | null;
  adjusted_rand_index: number | null;
  normalized_mutual_info: number | null;
  fragmentation_index: number | null;
}

export interface PairwiseLabelAgreement {
  mean_pairwise_ari: number | null;
  std_pairwise_ari: number | null;
  min_pairwise_ari: number | null;
  max_pairwise_ari: number | null;
}

export interface StabilitySummary {
  n_runs: number;
  seeds: number[];
  pairwise_label_agreement: PairwiseLabelAgreement;
  aggregate_metrics: Record<string, number | null>;
  per_run: StabilityRunMetrics[];
}

// ---- Classifier Baseline ----

export interface PerClassMetrics {
  precision: number;
  recall: number;
  f1_score: number;
  support: number;
}

export interface ClassifierReport {
  n_samples: number;
  n_categories: number;
  n_folds: number;
  categories_excluded: string[];
  overall_accuracy: number;
  per_class: Record<string, PerClassMetrics>;
  macro_avg: PerClassMetrics;
  weighted_avg: PerClassMetrics;
  confusion_matrix: Record<string, Record<string, number>>;
}

export interface LabelQueueEntry {
  rank: number;
  global_index: number;
  embedding_set_id: string;
  embedding_row_index: number;
  current_category: string | null;
  predicted_category: string | null;
  entropy: number | null;
  margin: number | null;
  max_prob: number | null;
  fragmentation_boost: number;
  priority: number;
}

// ---- Metric Learning Refinement ----

export interface RefinementTrainingParams {
  output_dim: number;
  hidden_dim: number;
  n_epochs: number;
  lr: number;
  margin: number;
  batch_size: number;
  mining_strategy: string;
}

export interface RefinementMetricComparison {
  metric: string;
  key: string;
  base: number | null;
  refined: number | null;
  delta: number | null;
  improved: boolean | null;
}

export interface RefinementReport {
  training_params: RefinementTrainingParams;
  n_labeled_samples: number;
  n_categories: number;
  n_total_samples: number;
  categories_used: string[];
  loss_history: number[];
  final_loss: number;
  comparison: RefinementMetricComparison[];
  base_summary: Record<string, number | null>;
  refined_summary: Record<string, number | null>;
}

// ---- Binary Classifier ----

export interface ClassifierTrainingJobCreate {
  name: string;
  positive_embedding_set_ids: string[];
  negative_embedding_set_ids: string[];
  parameters?: Record<string, unknown> | null;
}

export interface ClassifierTrainingJob {
  id: string;
  status: "queued" | "running" | "complete" | "failed" | "canceled";
  name: string;
  positive_embedding_set_ids: string[];
  negative_embedding_set_ids: string[];
  model_version: string;
  window_size_seconds: number;
  target_sample_rate: number;
  feature_config: Record<string, unknown> | null;
  parameters: Record<string, unknown> | null;
  classifier_model_id: string | null;
  error_message: string | null;
  created_at: string;
  updated_at: string;
}

export interface ClassifierModelInfo {
  id: string;
  name: string;
  model_path: string;
  model_version: string;
  vector_dim: number;
  window_size_seconds: number;
  target_sample_rate: number;
  feature_config: Record<string, unknown> | null;
  training_summary: Record<string, unknown> | null;
  training_job_id: string | null;
  created_at: string;
  updated_at: string;
}

export interface DetectionJob {
  id: string;
  status: "queued" | "running" | "complete" | "failed" | "canceled" | "paused";
  classifier_model_id: string;
  audio_folder: string | null;
  confidence_threshold: number;
  hop_seconds: number;
  high_threshold: number;
  low_threshold: number;
  detection_mode: "merged" | "windowed" | null;
  output_tsv_path: string | null;
  output_row_store_path: string | null;
  result_summary: Record<string, unknown> | null;
  error_message: string | null;
  files_processed: number | null;
  files_total: number | null;
  extract_status: string | null;
  extract_error: string | null;
  extract_summary: Record<string, unknown> | null;
  // Hydrophone fields
  hydrophone_id: string | null;
  hydrophone_name: string | null;
  start_timestamp: number | null;
  end_timestamp: number | null;
  segments_processed: number | null;
  segments_total: number | null;
  time_covered_sec: number | null;
  alerts: FlashAlert[] | null;
  local_cache_path: string | null;
  has_positive_labels: boolean | null;
  row_store_version: number;
  created_at: string;
  updated_at: string;
}

export interface FlashAlert {
  type: "error" | "warning" | "info";
  message: string;
  timestamp: string;
}

export interface HydrophoneInfo {
  id: string;
  name: string;
  location: string;
  provider_kind: string;
}

export interface HydrophoneDetectionJobCreate {
  classifier_model_id: string;
  hydrophone_id: string;
  start_timestamp: number;
  end_timestamp: number;
  confidence_threshold?: number;
  hop_seconds?: number;
  high_threshold?: number;
  low_threshold?: number;
  local_cache_path?: string;
}

export interface ExtractionSettings {
  positive_output_path: string;
  negative_output_path: string;
  positive_selection_smoothing_window: number;
  positive_selection_min_score: number;
  positive_selection_extend_min_score: number;
}

// ---- Directory Browsing ----

export interface DirectoryEntry {
  name: string;
  path: string;
}

export interface DirectoryListing {
  path: string;
  subdirectories: DirectoryEntry[];
}

// ---- Detection Content ----

export interface DetectionRow {
  start_utc: number;
  end_utc: number;
  avg_confidence: number | null;
  peak_confidence: number | null;
  n_windows: number | null;
  hydrophone_name?: string | null;
  raw_start_utc?: number | null;
  raw_end_utc?: number | null;
  merged_event_count?: number | null;
  auto_positive_selection_score_source?: string | null;
  auto_positive_selection_decision?: "positive" | "skip" | null;
  auto_positive_selection_offsets?: number[] | null;
  auto_positive_selection_raw_scores?: number[] | null;
  auto_positive_selection_smoothed_scores?: number[] | null;
  auto_positive_selection_start_utc?: number | null;
  auto_positive_selection_end_utc?: number | null;
  auto_positive_selection_peak_score?: number | null;
  manual_positive_selection_start_utc?: number | null;
  manual_positive_selection_end_utc?: number | null;
  positive_selection_origin?: string | null;
  positive_selection_score_source?: string | null;
  positive_selection_decision?: "positive" | "skip" | null;
  positive_selection_offsets?: number[] | null;
  positive_selection_raw_scores?: number[] | null;
  positive_selection_smoothed_scores?: number[] | null;
  positive_selection_start_utc?: number | null;
  positive_selection_end_utc?: number | null;
  positive_selection_peak_score?: number | null;
  positive_extract_filename?: string | null;
  humpback: number | null;
  orca: number | null;
  ship: number | null;
  background: number | null;
}

export interface DetectionLabelRow {
  start_utc: number;
  end_utc: number;
  humpback: number | null;
  orca: number | null;
  ship: number | null;
  background: number | null;
}

export interface DetectionRowStateUpdate {
  start_utc: number;
  end_utc: number;
  humpback: number | null;
  orca: number | null;
  ship: number | null;
  background: number | null;
  manual_positive_selection_start_utc: number | null;
  manual_positive_selection_end_utc: number | null;
}

export interface DetectionRowStateResponse {
  status: string;
  row: DetectionRow;
}

export interface TrainingSourceInfo {
  embedding_set_id: string;
  audio_file_id: string | null;
  filename: string | null;
  folder_path: string | null;
  n_vectors: number;
  duration_represented_sec: number | null;
}

export interface TrainingDataSummaryResponse {
  model_id: string;
  model_name: string;
  positive_sources: TrainingSourceInfo[];
  negative_sources: TrainingSourceInfo[];
  total_positive: number;
  total_negative: number;
  balance_ratio: number;
  window_size_seconds: number;
  positive_duration_sec: number | null;
  negative_duration_sec: number | null;
}

// ---- Retrain Workflows ----

export interface RetrainFolderInfo {
  model_id: string;
  model_name: string;
  model_version: string;
  window_size_seconds: number;
  target_sample_rate: number;
  feature_config: Record<string, unknown> | null;
  positive_folder_roots: string[];
  negative_folder_roots: string[];
  parameters: Record<string, unknown>;
}

export interface RetrainWorkflowCreate {
  source_model_id: string;
  new_model_name: string;
  parameters?: Record<string, unknown> | null;
}

export interface RetrainWorkflow {
  id: string;
  status: string;
  source_model_id: string;
  new_model_name: string;
  model_version: string;
  window_size_seconds: number;
  target_sample_rate: number;
  feature_config: Record<string, unknown> | null;
  parameters: Record<string, unknown> | null;
  positive_folder_roots: string[];
  negative_folder_roots: string[];
  import_summary: Record<string, unknown> | null;
  processing_job_ids: string[] | null;
  processing_total: number | null;
  processing_complete: number | null;
  training_job_id: string | null;
  new_model_id: string | null;
  error_message: string | null;
  created_at: string;
  updated_at: string;
}

// ---- Admin ----

export interface ModelConfig {
  id: string;
  name: string;
  display_name: string;
  path: string;
  vector_dim: number;
  description: string | null;
  is_default: boolean;
  model_type: string;
  input_format: string;
  created_at: string;
}

export interface ModelConfigCreate {
  name: string;
  display_name: string;
  path: string;
  vector_dim?: number;
  description?: string | null;
  is_default?: boolean;
  model_type?: string;
  input_format?: string;
}

export interface AvailableModelFile {
  filename: string;
  path: string;
  size_bytes: number;
  registered: boolean;
  model_type: string;
  input_format: string;
  detected_vector_dim?: number;
}

export interface TableInfo {
  table: string;
  count: number;
}

// ---- Search ----

export interface ScoreHistogramBin {
  bin_start: number;
  bin_end: number;
  count: number;
}

export interface ScoreDistribution {
  mean: number;
  std: number;
  min: number;
  max: number;
  p25: number;
  p50: number;
  p75: number;
  histogram: ScoreHistogramBin[];
}

export interface SimilaritySearchHit {
  score: number;
  percentile_rank: number;
  embedding_set_id: string;
  row_index: number;
  audio_file_id: string;
  audio_filename: string;
  audio_folder_path: string | null;
  window_offset_seconds: number;
}

export interface SimilaritySearchResponse {
  query_embedding_set_id: string;
  query_row_index: number;
  model_version: string;
  metric: string;
  total_candidates: number;
  results: SimilaritySearchHit[];
  score_distribution: ScoreDistribution;
}

export interface DetectionEmbeddingResponse {
  vector: number[];
  model_version: string;
  vector_dim: number;
}

export interface AudioSearchRequest {
  detection_job_id: string;
  start_utc: number;
  end_utc: number;
  top_k?: number;
  metric?: string;
  embedding_set_ids?: string[];
  search_mode?: "raw" | "projected";
  classifier_model_id?: string | null;
}

export interface SearchJobResponse {
  id: string;
  status: string;
  error?: string | null;
  results?: SimilaritySearchResponse | null;
  query_vector?: number[] | null;
  model_version?: string | null;
}

// ---- Label Processing ----

export interface LabelProcessingJobCreate {
  workflow?: "score_based" | "sample_builder";
  classifier_model_id?: string | null;
  annotation_folder: string;
  audio_folder: string;
  output_root: string;
  parameters?: Record<string, unknown> | null;
}

export interface LabelProcessingJob {
  id: string;
  status: "queued" | "running" | "complete" | "failed";
  workflow: "score_based" | "sample_builder";
  classifier_model_id: string | null;
  annotation_folder: string;
  audio_folder: string;
  output_root: string;
  parameters: Record<string, unknown> | null;
  files_processed: number | null;
  files_total: number | null;
  annotations_total: number | null;
  result_summary: Record<string, unknown> | null;
  error_message: string | null;
  created_at: string;
  updated_at: string;
}

export interface LabelProcessingPairedFile {
  annotation_file: string;
  audio_file: string;
  annotation_count: number;
}

export interface LabelProcessingPreview {
  paired_files: LabelProcessingPairedFile[];
  total_annotations: number;
  call_type_distribution: Record<string, number>;
  unpaired_annotations: string[];
  unpaired_audio: string[];
}

// ---- Labeling ----

export interface VocalizationLabel {
  id: string;
  detection_job_id: string;
  start_utc: number;
  end_utc: number;
  label: string;
  confidence: number | null;
  source: string;
  notes: string | null;
  row_store_version_at_import: number | null;
  created_at: string;
  updated_at: string;
}

export interface TimelineVocalizationLabel {
  start_utc: number;
  end_utc: number;
  label: string;
  confidence: number | null;
  source: string;
}

export interface OrphanedLabelDetail {
  id: string;
  start_utc: number;
  end_utc: number;
  label: string;
}

export interface RefreshPreviewResponse {
  matched_count: number;
  orphaned_count: number;
  orphaned_labels: OrphanedLabelDetail[];
  current_version: number;
}

export interface RefreshApplyResponse {
  deleted_count: number;
  surviving_count: number;
  current_version: number;
}

export interface NeighborHit {
  score: number;
  embedding_set_id: string;
  row_index: number;
  audio_file_id: string;
  audio_filename: string;
  audio_folder_path: string | null;
  window_offset_seconds: number;
  inferred_label: string | null;
}

export interface DetectionNeighborsResponse {
  hits: NeighborHit[];
  total_candidates: number;
}

export interface LabelingSummary {
  total_rows: number;
  labeled_rows: number;
  unlabeled_rows: number;
  label_distribution: Record<string, number>;
}

export interface TrainingSummary {
  labeled_job_ids: string[];
  labeled_rows: number;
  label_distribution: Record<string, number>;
}

// ---- Multi-Label Vocalization Classifier ----

export interface VocalizationType {
  id: string;
  name: string;
  description: string | null;
  created_at: string;
  updated_at: string;
}

export interface VocalizationTypeCreate {
  name: string;
  description?: string | null;
}

export interface VocalizationTypeUpdate {
  name?: string | null;
  description?: string | null;
}

export interface VocalizationTypeImportRequest {
  embedding_set_ids: string[];
}

export interface VocalizationTypeImportResponse {
  added: string[];
  skipped: string[];
}

export interface VocalizationTrainingSourceConfig {
  embedding_set_ids: string[];
  detection_job_ids: string[];
}

export interface VocClassifierTrainingJobCreate {
  source_config?: VocalizationTrainingSourceConfig | null;
  training_dataset_id?: string | null;
  parameters?: Record<string, unknown> | null;
}

export interface VocClassifierTrainingJob {
  id: string;
  status: string;
  source_config: Record<string, unknown>;
  parameters: Record<string, unknown> | null;
  vocalization_model_id: string | null;
  result_summary: Record<string, unknown> | null;
  error_message: string | null;
  created_at: string;
  updated_at: string;
}

export interface VocClassifierModel {
  id: string;
  name: string;
  model_dir_path: string;
  vocabulary_snapshot: string[];
  per_class_thresholds: Record<string, number>;
  per_class_metrics: Record<string, unknown> | null;
  training_summary: Record<string, unknown> | null;
  is_active: boolean;
  training_dataset_id: string | null;
  created_at: string;
}

export interface VocClassifierInferenceJobCreate {
  vocalization_model_id: string;
  source_type: "detection_job" | "embedding_set" | "rescore";
  source_id: string;
}

export interface VocClassifierInferenceJob {
  id: string;
  status: string;
  vocalization_model_id: string;
  source_type: string;
  source_id: string;
  output_path: string | null;
  result_summary: Record<string, unknown> | null;
  error_message: string | null;
  created_at: string;
  updated_at: string;
}

export interface VocClassifierPredictionRow {
  filename: string;
  start_sec: number;
  end_sec: number;
  start_utc: number | null;
  end_utc: number | null;
  confidence: number | null;
  scores: Record<string, number>;
  tags: string[];
}

// ---- Detection Embeddings ----

export interface EmbeddingStatus {
  has_embeddings: boolean;
  count: number | null;
  sync_needed: boolean | null;
}

export interface DetectionEmbeddingJob {
  id: string;
  status: string;
  detection_job_id: string;
  mode: string | null;
  progress_current: number | null;
  progress_total: number | null;
  error_message: string | null;
  result_summary: string | null;
  created_at: string;
  updated_at: string;
}

export interface EmbeddingJobListItem extends DetectionEmbeddingJob {
  hydrophone_name: string | null;
  audio_folder: string | null;
}

export interface VocalizationTrainingSource {
  source_config: VocalizationTrainingSourceConfig | null;
  parameters: Record<string, unknown> | null;
}

// ---- Folder Embedding Set ----

export interface FolderEmbeddingSetResponse {
  folder_path: string;
  embedding_set_ids: string[];
  total_files: number;
  processed_files: number;
  pending_files: number;
  status: "ready" | "processing" | "queued";
}

// ---- Labeling Source ----

export type LabelingSource =
  | { type: "detection_job"; jobId: string }
  | { type: "embedding_set"; embeddingSetId: string }
  | { type: "local"; folderPath: string };

// ---- Training Datasets ----

export interface TrainingDataset {
  id: string;
  name: string;
  source_config: Record<string, unknown>;
  total_rows: number;
  vocabulary: string[];
  created_at: string;
  updated_at: string;
}

export interface TrainingDatasetRow {
  row_index: number;
  filename: string;
  start_sec: number;
  end_sec: number;
  source_type: string;
  source_id: string;
  confidence: number | null;
  labels: string[];
}

export interface TrainingDatasetRowsResponse {
  total: number;
  rows: TrainingDatasetRow[];
}

export interface TrainingDatasetLabel {
  id: string;
  training_dataset_id: string;
  row_index: number;
  label: string;
  source: string;
  created_at: string;
  updated_at: string;
}

export interface TrainingDatasetExtendRequest {
  embedding_set_ids?: string[];
  detection_job_ids?: string[];
}

// ---- Health ----

export interface HealthStatus {
  status: "ok" | "error" | "starting";
  db: string;
  detail?: string;
}

// ---- Timeline viewer ----

export type ZoomLevel = "24h" | "6h" | "1h" | "15m" | "5m" | "1m";

export interface TimelineConfidenceResponse {
  window_sec: number;
  scores: (number | null)[];
  start_timestamp: number;
  end_timestamp: number;
}

export interface ZoomProgress {
  total: number;
  rendered: number;
}

export type PrepareStatusResponse = Record<string, ZoomProgress>;

export interface LabelEditItem {
  action: "add" | "move" | "delete" | "change_type";
  start_utc?: number;
  end_utc?: number;
  new_start_utc?: number;
  new_end_utc?: number;
  label?: "humpback" | "orca" | "ship" | "background";
}

export interface LabelEditRequest {
  edits: LabelEditItem[];
}
