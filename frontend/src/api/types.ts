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
  checksum_sha256: string;
  duration_seconds: number | null;
  sample_rate_original: number | null;
  created_at: string;
  metadata: AudioMetadata | null;
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
}

export interface ClusteringJob {
  id: string;
  status: "queued" | "running" | "complete" | "failed" | "canceled";
  embedding_set_ids: string[];
  parameters: Record<string, unknown> | null;
  error_message: string | null;
  metrics: Record<string, unknown> | null;
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
}

export interface TableInfo {
  table: string;
  count: number;
}
