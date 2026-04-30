import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

export type ContinuousEmbeddingSourceKind = "surfperch" | "region_crnn";

export interface ContinuousEmbeddingJob {
  id: string;
  status: string;
  event_segmentation_job_id: string | null;
  model_version: string;
  window_size_seconds: number | null;
  hop_seconds: number | null;
  pad_seconds: number | null;
  target_sample_rate: number;
  feature_config_json: string | null;
  encoding_signature: string;
  vector_dim: number | null;
  total_events: number | null;
  merged_spans: number | null;
  total_windows: number | null;
  parquet_path: string | null;
  error_message: string | null;
  // CRNN region-based source columns
  region_detection_job_id: string | null;
  chunk_size_seconds: number | null;
  chunk_hop_seconds: number | null;
  crnn_checkpoint_sha256: string | null;
  crnn_segmentation_model_id: string | null;
  projection_kind: string | null;
  projection_dim: number | null;
  total_regions: number | null;
  total_chunks: number | null;
  created_at: string;
  updated_at: string;
}

export interface ContinuousEmbeddingSpanSummary {
  merged_span_id: number;
  event_id: string;
  region_id: string;
  start_timestamp: number;
  end_timestamp: number;
  window_count: number;
}

export interface ContinuousEmbeddingRegionSummary {
  region_id: string;
  start_timestamp: number;
  end_timestamp: number;
  chunk_count: number;
}

export interface ContinuousEmbeddingJobManifest {
  job_id: string;
  model_version: string;
  source_kind: ContinuousEmbeddingSourceKind;
  vector_dim: number;
  target_sample_rate: number;
  // SurfPerch-only counters
  window_size_seconds?: number | null;
  hop_seconds?: number | null;
  pad_seconds?: number | null;
  total_events?: number | null;
  merged_spans?: number | null;
  total_windows?: number | null;
  spans?: ContinuousEmbeddingSpanSummary[];
  // CRNN-only counters
  region_detection_job_id?: string | null;
  event_segmentation_job_id?: string | null;
  crnn_checkpoint_sha256?: string | null;
  chunk_size_seconds?: number | null;
  chunk_hop_seconds?: number | null;
  projection_kind?: string | null;
  projection_dim?: number | null;
  total_regions?: number | null;
  total_chunks?: number | null;
  regions?: ContinuousEmbeddingRegionSummary[];
}

export interface ContinuousEmbeddingJobDetail {
  job: ContinuousEmbeddingJob;
  manifest: ContinuousEmbeddingJobManifest | null;
}

export interface CreateContinuousEmbeddingJobRequest {
  event_segmentation_job_id?: string;
  model_version?: string;
  hop_seconds?: number;
  pad_seconds?: number;
  // CRNN region-based source fields
  region_detection_job_id?: string;
  crnn_segmentation_model_id?: string;
  chunk_size_seconds?: number;
  chunk_hop_seconds?: number;
  projection_kind?: "identity" | "random" | "pca";
  projection_dim?: number;
}

export function continuousEmbeddingSourceKind(
  job: Pick<ContinuousEmbeddingJob, "model_version" | "region_detection_job_id">,
): ContinuousEmbeddingSourceKind {
  if (job.region_detection_job_id) return "region_crnn";
  if (job.model_version.startsWith("crnn-")) return "region_crnn";
  return "surfperch";
}

const ROOT = "/sequence-models/continuous-embeddings";

class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, init);
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new ApiError(res.status, text);
  }
  if (res.status === 204) return undefined as T;
  return res.json();
}

export function fetchContinuousEmbeddingJobs(
  status?: string,
): Promise<ContinuousEmbeddingJob[]> {
  const q = status ? `?status=${encodeURIComponent(status)}` : "";
  return request<ContinuousEmbeddingJob[]>(`${ROOT}${q}`);
}

export function fetchContinuousEmbeddingJob(
  jobId: string,
): Promise<ContinuousEmbeddingJobDetail> {
  return request<ContinuousEmbeddingJobDetail>(`${ROOT}/${jobId}`);
}

export function createContinuousEmbeddingJob(
  body: CreateContinuousEmbeddingJobRequest,
): Promise<ContinuousEmbeddingJob> {
  return request<ContinuousEmbeddingJob>(ROOT, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function cancelContinuousEmbeddingJob(
  jobId: string,
): Promise<ContinuousEmbeddingJob> {
  return request<ContinuousEmbeddingJob>(`${ROOT}/${jobId}/cancel`, {
    method: "POST",
  });
}

export function deleteContinuousEmbeddingJob(
  jobId: string,
): Promise<void> {
  return request<void>(`${ROOT}/${jobId}`, { method: "DELETE" });
}

const ACTIVE_STATUSES = new Set(["queued", "running"]);

export function isContinuousEmbeddingJobActive(
  job: Pick<ContinuousEmbeddingJob, "status">,
): boolean {
  return ACTIVE_STATUSES.has(job.status);
}

export function useContinuousEmbeddingJobs(refetchInterval = 3000) {
  return useQuery({
    queryKey: ["continuous-embedding-jobs"],
    queryFn: () => fetchContinuousEmbeddingJobs(),
    refetchInterval,
  });
}

export function useContinuousEmbeddingJob(jobId: string | null) {
  return useQuery({
    queryKey: ["continuous-embedding-job", jobId],
    queryFn: () => fetchContinuousEmbeddingJob(jobId as string),
    enabled: jobId != null,
    refetchInterval: (query) => {
      const data = query.state.data as
        | ContinuousEmbeddingJobDetail
        | undefined;
      if (!data) return 3000;
      return isContinuousEmbeddingJobActive(data.job) ? 3000 : false;
    },
  });
}

export function useCreateContinuousEmbeddingJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: CreateContinuousEmbeddingJobRequest) =>
      createContinuousEmbeddingJob(body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["continuous-embedding-jobs"] });
    },
  });
}

export function useCancelContinuousEmbeddingJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (jobId: string) => cancelContinuousEmbeddingJob(jobId),
    onSuccess: (_data, jobId) => {
      qc.invalidateQueries({ queryKey: ["continuous-embedding-jobs"] });
      qc.invalidateQueries({ queryKey: ["continuous-embedding-job", jobId] });
    },
  });
}

export function useDeleteContinuousEmbeddingJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (jobId: string) => deleteContinuousEmbeddingJob(jobId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["continuous-embedding-jobs"] });
    },
  });
}

// ---------------------------------------------------------------------------
// HMM Sequence Jobs
// ---------------------------------------------------------------------------

const HMM_ROOT = "/sequence-models/hmm-sequences";

export interface HMMSequenceJob {
  id: string;
  status: string;
  continuous_embedding_job_id: string;
  n_states: number;
  pca_dims: number;
  pca_whiten: boolean;
  l2_normalize: boolean;
  covariance_type: string;
  n_iter: number;
  random_seed: number;
  min_sequence_length_frames: number;
  tol: number;
  library: string;
  train_log_likelihood: number | null;
  n_train_sequences: number | null;
  n_train_frames: number | null;
  n_decoded_sequences: number | null;
  artifact_dir: string | null;
  error_message: string | null;
  // CRNN-only training-mode + tier configuration
  training_mode: string | null;
  event_core_overlap_threshold: number | null;
  near_event_window_seconds: number | null;
  event_balanced_proportions: string | null;
  subsequence_length_chunks: number | null;
  subsequence_stride_chunks: number | null;
  target_train_chunks: number | null;
  min_region_length_seconds: number | null;
  created_at: string;
  updated_at: string;
}

export interface HMMStateSummary {
  state: number;
  occupancy: number;
  mean_dwell_frames: number;
  dwell_histogram: number[];
}

export interface StateTierComposition {
  state: number;
  event_core: number;
  near_event: number;
  background: number;
}

export interface HMMSequenceJobDetail {
  job: HMMSequenceJob;
  region_detection_job_id: string;
  region_start_timestamp: number | null;
  region_end_timestamp: number | null;
  summary: HMMStateSummary[] | null;
  tier_composition?: StateTierComposition[] | null;
  source_kind?: ContinuousEmbeddingSourceKind;
}

export interface CreateHMMSequenceJobRequest {
  continuous_embedding_job_id: string;
  n_states: number;
  pca_dims?: number;
  pca_whiten?: boolean;
  l2_normalize?: boolean;
  covariance_type?: "diag" | "full";
  n_iter?: number;
  random_seed?: number;
  min_sequence_length_frames?: number;
  tol?: number;
  // CRNN-only training-mode + tier configuration
  training_mode?: "full_region" | "event_balanced" | "event_only";
  event_core_overlap_threshold?: number;
  near_event_window_seconds?: number;
  event_balanced_proportions?: Record<string, number>;
  subsequence_length_chunks?: number;
  subsequence_stride_chunks?: number;
  target_train_chunks?: number;
  min_region_length_seconds?: number;
}

export interface TransitionMatrix {
  n_states: number;
  matrix: number[][];
}

export interface DwellHistograms {
  n_states: number;
  histograms: Record<string, number[]>;
}

export function fetchHMMSequenceJobs(
  status?: string,
  continuousEmbeddingJobId?: string,
): Promise<HMMSequenceJob[]> {
  const params = new URLSearchParams();
  if (status) params.set("status", status);
  if (continuousEmbeddingJobId)
    params.set("continuous_embedding_job_id", continuousEmbeddingJobId);
  const q = params.toString() ? `?${params.toString()}` : "";
  return request<HMMSequenceJob[]>(`${HMM_ROOT}${q}`);
}

export function fetchHMMSequenceJob(
  jobId: string,
): Promise<HMMSequenceJobDetail> {
  return request<HMMSequenceJobDetail>(`${HMM_ROOT}/${jobId}`);
}

export function createHMMSequenceJob(
  body: CreateHMMSequenceJobRequest,
): Promise<HMMSequenceJob> {
  return request<HMMSequenceJob>(HMM_ROOT, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function cancelHMMSequenceJob(
  jobId: string,
): Promise<HMMSequenceJob> {
  return request<HMMSequenceJob>(`${HMM_ROOT}/${jobId}/cancel`, {
    method: "POST",
  });
}

export function deleteHMMSequenceJob(jobId: string): Promise<void> {
  return request<void>(`${HMM_ROOT}/${jobId}`, { method: "DELETE" });
}

export function fetchHMMTransitions(
  jobId: string,
): Promise<TransitionMatrix> {
  return request<TransitionMatrix>(`${HMM_ROOT}/${jobId}/transitions`);
}

export function fetchHMMDwell(jobId: string): Promise<DwellHistograms> {
  return request<DwellHistograms>(`${HMM_ROOT}/${jobId}/dwell`);
}

export function fetchHMMStates(
  jobId: string,
  offset = 0,
  limit = 50000,
): Promise<{
  total: number;
  offset: number;
  limit: number;
  items: Record<string, unknown>[];
}> {
  return request(`${HMM_ROOT}/${jobId}/states?offset=${offset}&limit=${limit}`);
}

export function isHMMSequenceJobActive(
  job: Pick<HMMSequenceJob, "status">,
): boolean {
  return ACTIVE_STATUSES.has(job.status);
}

export function useHMMSequenceJobs(refetchInterval = 3000) {
  return useQuery({
    queryKey: ["hmm-sequence-jobs"],
    queryFn: () => fetchHMMSequenceJobs(),
    refetchInterval,
  });
}

export function useHMMSequenceJob(jobId: string | null) {
  return useQuery({
    queryKey: ["hmm-sequence-job", jobId],
    queryFn: () => fetchHMMSequenceJob(jobId as string),
    enabled: jobId != null,
    refetchInterval: (query) => {
      const data = query.state.data as HMMSequenceJobDetail | undefined;
      if (!data) return 3000;
      return isHMMSequenceJobActive(data.job) ? 3000 : false;
    },
  });
}

export function useCreateHMMSequenceJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: CreateHMMSequenceJobRequest) =>
      createHMMSequenceJob(body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["hmm-sequence-jobs"] });
    },
  });
}

export function useCancelHMMSequenceJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (jobId: string) => cancelHMMSequenceJob(jobId),
    onSuccess: (_data, jobId) => {
      qc.invalidateQueries({ queryKey: ["hmm-sequence-jobs"] });
      qc.invalidateQueries({ queryKey: ["hmm-sequence-job", jobId] });
    },
  });
}

export function useDeleteHMMSequenceJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (jobId: string) => deleteHMMSequenceJob(jobId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["hmm-sequence-jobs"] });
    },
  });
}

export function useHMMTransitions(jobId: string | null, enabled = true) {
  return useQuery({
    queryKey: ["hmm-transitions", jobId],
    queryFn: () => fetchHMMTransitions(jobId as string),
    enabled: enabled && jobId != null,
  });
}

export function useHMMDwell(jobId: string | null, enabled = true) {
  return useQuery({
    queryKey: ["hmm-dwell", jobId],
    queryFn: () => fetchHMMDwell(jobId as string),
    enabled: enabled && jobId != null,
  });
}

export function useHMMStates(
  jobId: string | null,
  offset = 0,
  limit = 50000,
  enabled = true,
) {
  return useQuery({
    queryKey: ["hmm-states", jobId, offset, limit],
    queryFn: () => fetchHMMStates(jobId as string, offset, limit),
    enabled: enabled && jobId != null,
  });
}

// ---------------------------------------------------------------------------
// Interpretation visualizations (PR 3)
// ---------------------------------------------------------------------------

export interface OverlayPoint {
  merged_span_id: number;
  window_index_in_span: number;
  start_timestamp: number;
  end_timestamp: number;
  pca_x: number;
  pca_y: number;
  umap_x: number;
  umap_y: number;
  viterbi_state: number;
  max_state_probability: number;
}

export interface OverlayResponse {
  total: number;
  items: OverlayPoint[];
}

export interface LabelDistribution {
  n_states: number;
  total_windows: number;
  states: Record<string, Record<string, number>>;
}

export interface ExemplarRecord {
  merged_span_id: number;
  window_index_in_span: number;
  audio_file_id: number | null;
  start_timestamp: number;
  end_timestamp: number;
  max_state_probability: number;
  exemplar_type: string;
}

export interface ExemplarsResponse {
  n_states: number;
  states: Record<string, ExemplarRecord[]>;
}

export function fetchHMMOverlay(
  jobId: string,
  offset = 0,
  limit = 50000,
): Promise<OverlayResponse> {
  return request(
    `${HMM_ROOT}/${jobId}/overlay?offset=${offset}&limit=${limit}`,
  );
}

export function fetchHMMLabelDistribution(
  jobId: string,
): Promise<LabelDistribution> {
  return request(`${HMM_ROOT}/${jobId}/label-distribution`);
}

export function fetchHMMExemplars(
  jobId: string,
): Promise<ExemplarsResponse> {
  return request(`${HMM_ROOT}/${jobId}/exemplars`);
}

export function postGenerateInterpretations(
  jobId: string,
): Promise<{ status: string; job_id: string }> {
  return request(`${HMM_ROOT}/${jobId}/generate-interpretations`, {
    method: "POST",
  });
}

export function useHMMOverlay(jobId: string | null, enabled = true) {
  return useQuery({
    queryKey: ["hmm-overlay", jobId],
    queryFn: () => fetchHMMOverlay(jobId as string),
    enabled: enabled && jobId != null,
  });
}

export function useHMMLabelDistribution(
  jobId: string | null,
  enabled = true,
) {
  return useQuery({
    queryKey: ["hmm-label-distribution", jobId],
    queryFn: () => fetchHMMLabelDistribution(jobId as string),
    enabled: enabled && jobId != null,
  });
}

export function useHMMExemplars(jobId: string | null, enabled = true) {
  return useQuery({
    queryKey: ["hmm-exemplars", jobId],
    queryFn: () => fetchHMMExemplars(jobId as string),
    enabled: enabled && jobId != null,
  });
}

export function useGenerateInterpretations() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (jobId: string) => postGenerateInterpretations(jobId),
    onSuccess: (_data, jobId) => {
      qc.invalidateQueries({ queryKey: ["hmm-overlay", jobId] });
      qc.invalidateQueries({ queryKey: ["hmm-label-distribution", jobId] });
      qc.invalidateQueries({ queryKey: ["hmm-exemplars", jobId] });
    },
  });
}
