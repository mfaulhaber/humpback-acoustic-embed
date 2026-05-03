import { useMemo } from "react";
import { useMutation, useQueries, useQuery, useQueryClient } from "@tanstack/react-query";

export type ContinuousEmbeddingSourceKind = "surfperch" | "region_crnn";

export interface ContinuousEmbeddingJob {
  id: string;
  status: string;
  event_segmentation_job_id: string | null;
  event_source_mode: "raw" | "effective";
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
  event_source_mode?: "raw" | "effective";
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
  event_source_mode?: "raw" | "effective";
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
// Motif Extraction Jobs
// ---------------------------------------------------------------------------

const MOTIF_ROOT = "/sequence-models/motif-extractions";

export type MotifParentKind = "hmm" | "masked_transformer";

export interface MotifExtractionJob {
  id: string;
  status: string;
  parent_kind: MotifParentKind;
  hmm_sequence_job_id: string | null;
  masked_transformer_job_id: string | null;
  k: number | null;
  source_kind: ContinuousEmbeddingSourceKind;
  min_ngram: number;
  max_ngram: number;
  minimum_occurrences: number;
  minimum_event_sources: number;
  frequency_weight: number;
  event_source_weight: number;
  event_core_weight: number;
  low_background_weight: number;
  call_probability_weight: number | null;
  config_signature: string;
  total_groups: number | null;
  total_collapsed_tokens: number | null;
  total_candidate_occurrences: number | null;
  total_motifs: number | null;
  artifact_dir: string | null;
  error_message: string | null;
  created_at: string;
  updated_at: string;
}

export interface CreateMotifExtractionJobRequest {
  parent_kind?: MotifParentKind;
  hmm_sequence_job_id?: string | null;
  masked_transformer_job_id?: string | null;
  k?: number | null;
  min_ngram?: number;
  max_ngram?: number;
  minimum_occurrences?: number;
  minimum_event_sources?: number;
  frequency_weight?: number;
  event_source_weight?: number;
  event_core_weight?: number;
  low_background_weight?: number;
  call_probability_weight?: number | null;
}

export interface MotifExtractionManifest {
  schema_version: number;
  motif_extraction_job_id: string;
  parent_kind: MotifParentKind;
  hmm_sequence_job_id: string | null;
  masked_transformer_job_id: string | null;
  k: number | null;
  continuous_embedding_job_id: string;
  source_kind: ContinuousEmbeddingSourceKind;
  config: Record<string, unknown>;
  config_signature: string;
  generated_at: string;
  total_groups: number;
  total_collapsed_tokens: number;
  total_candidate_occurrences: number;
  total_motifs: number;
  event_source_key_strategy: string;
}

export interface MotifExtractionJobDetail {
  job: MotifExtractionJob;
  manifest: MotifExtractionManifest | null;
}

export interface MotifSummary {
  motif_key: string;
  states: number[];
  length: number;
  occurrence_count: number;
  event_source_count: number;
  audio_source_count: number;
  group_count: number;
  event_core_fraction: number;
  background_fraction: number;
  mean_call_probability: number | null;
  mean_duration_seconds: number;
  median_duration_seconds: number;
  rank_score: number;
  example_occurrence_ids: string[];
}

export interface MotifsResponse {
  total: number;
  offset: number;
  limit: number;
  items: MotifSummary[];
}

export interface MotifOccurrence {
  occurrence_id: string;
  motif_key: string;
  states: number[];
  source_kind: ContinuousEmbeddingSourceKind;
  group_key: string;
  event_source_key: string;
  audio_source_key: string | null;
  token_start_index: number;
  token_end_index: number;
  raw_start_index: number;
  raw_end_index: number;
  start_timestamp: number;
  end_timestamp: number;
  duration_seconds: number;
  event_core_fraction: number;
  background_fraction: number;
  mean_call_probability: number | null;
  anchor_event_id: string | null;
  anchor_timestamp: number;
  relative_start_seconds: number;
  relative_end_seconds: number;
  anchor_strategy: string;
}

export interface MotifOccurrencesResponse {
  total: number;
  offset: number;
  limit: number;
  items: MotifOccurrence[];
}

export function fetchMotifExtractionJobs(params?: {
  status?: string;
  hmm_sequence_job_id?: string;
  masked_transformer_job_id?: string;
  parent_kind?: MotifParentKind;
  k?: number;
}): Promise<MotifExtractionJob[]> {
  const search = new URLSearchParams();
  if (params?.status) search.set("status", params.status);
  if (params?.hmm_sequence_job_id)
    search.set("hmm_sequence_job_id", params.hmm_sequence_job_id);
  if (params?.masked_transformer_job_id)
    search.set("masked_transformer_job_id", params.masked_transformer_job_id);
  if (params?.parent_kind) search.set("parent_kind", params.parent_kind);
  if (params?.k != null) search.set("k", String(params.k));
  const q = search.toString() ? `?${search.toString()}` : "";
  return request<MotifExtractionJob[]>(`${MOTIF_ROOT}${q}`);
}

export function fetchMotifExtractionJob(
  jobId: string,
): Promise<MotifExtractionJobDetail> {
  return request<MotifExtractionJobDetail>(`${MOTIF_ROOT}/${jobId}`);
}

export function createMotifExtractionJob(
  body: CreateMotifExtractionJobRequest,
): Promise<MotifExtractionJob> {
  return request<MotifExtractionJob>(MOTIF_ROOT, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function cancelMotifExtractionJob(
  jobId: string,
): Promise<MotifExtractionJob> {
  return request<MotifExtractionJob>(`${MOTIF_ROOT}/${jobId}/cancel`, {
    method: "POST",
  });
}

export function deleteMotifExtractionJob(jobId: string): Promise<void> {
  return request<void>(`${MOTIF_ROOT}/${jobId}`, { method: "DELETE" });
}

export function fetchMotifs(
  jobId: string,
  offset = 0,
  limit = 100,
): Promise<MotifsResponse> {
  return request<MotifsResponse>(
    `${MOTIF_ROOT}/${jobId}/motifs?offset=${offset}&limit=${limit}`,
  );
}

export function fetchMotifOccurrences(
  jobId: string,
  motifKey: string,
  offset = 0,
  limit = 100,
): Promise<MotifOccurrencesResponse> {
  return request<MotifOccurrencesResponse>(
    `${MOTIF_ROOT}/${jobId}/motifs/${encodeURIComponent(
      motifKey,
    )}/occurrences?offset=${offset}&limit=${limit}`,
  );
}

export function useMotifExtractionJobs(
  params?: {
    status?: string;
    hmm_sequence_job_id?: string;
    masked_transformer_job_id?: string;
    parent_kind?: MotifParentKind;
    k?: number;
  },
  enabled = true,
  refetchInterval: number | false = 3000,
) {
  return useQuery({
    queryKey: ["motif-extraction-jobs", params],
    queryFn: () => fetchMotifExtractionJobs(params),
    enabled,
    refetchInterval,
  });
}

export function useMotifExtractionJob(jobId: string | null, enabled = true) {
  return useQuery({
    queryKey: ["motif-extraction-job", jobId],
    queryFn: () => fetchMotifExtractionJob(jobId as string),
    enabled: enabled && jobId != null,
    refetchInterval: (query) => {
      const data = query.state.data as MotifExtractionJobDetail | undefined;
      if (!data) return 3000;
      return ACTIVE_STATUSES.has(data.job.status) ? 3000 : false;
    },
  });
}

export function useCreateMotifExtractionJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: CreateMotifExtractionJobRequest) =>
      createMotifExtractionJob(body),
    onSuccess: (job) => {
      qc.invalidateQueries({ queryKey: ["motif-extraction-jobs"] });
      if (job.hmm_sequence_job_id) {
        qc.invalidateQueries({
          queryKey: [
            "motif-extraction-jobs",
            { hmm_sequence_job_id: job.hmm_sequence_job_id },
          ],
        });
      }
      if (job.masked_transformer_job_id) {
        qc.invalidateQueries({
          queryKey: [
            "motif-extraction-jobs",
            { masked_transformer_job_id: job.masked_transformer_job_id },
          ],
        });
      }
      qc.invalidateQueries({ queryKey: ["motif-extraction-job", job.id] });
    },
  });
}

export function useCancelMotifExtractionJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (jobId: string) => cancelMotifExtractionJob(jobId),
    onSuccess: (job) => {
      qc.invalidateQueries({ queryKey: ["motif-extraction-jobs"] });
      qc.invalidateQueries({ queryKey: ["motif-extraction-job", job.id] });
    },
  });
}

export function useDeleteMotifExtractionJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (jobId: string) => deleteMotifExtractionJob(jobId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["motif-extraction-jobs"] });
    },
  });
}

export function useMotifs(
  jobId: string | null,
  offset = 0,
  limit = 100,
  enabled = true,
) {
  return useQuery({
    queryKey: ["motifs", jobId, offset, limit],
    queryFn: () => fetchMotifs(jobId as string, offset, limit),
    enabled: enabled && jobId != null,
  });
}

export function useMotifOccurrences(
  jobId: string | null,
  motifKey: string | null,
  offset = 0,
  limit = 100,
  enabled = true,
) {
  return useQuery({
    queryKey: ["motif-occurrences", jobId, motifKey, offset, limit],
    queryFn: () =>
      fetchMotifOccurrences(jobId as string, motifKey as string, offset, limit),
    enabled: enabled && jobId != null && motifKey != null,
  });
}

/**
 * Pure helper: filter the motif list to the requested ``length``. Returns
 * an empty array when ``length`` is null. Exported for unit testing.
 */
export function filterMotifsByLength(
  motifs: MotifSummary[],
  length: number | null,
): MotifSummary[] {
  if (length == null) return [];
  return motifs.filter((m) => m.length === length);
}

/**
 * Pure helper: flatten per-motif occurrence arrays into a single
 * time-sorted array. Undefined entries (queries still pending) are
 * skipped. Exported for unit testing.
 */
export function mergeOccurrencesByStart(
  perMotif: Array<MotifOccurrence[] | undefined>,
): MotifOccurrence[] {
  const merged: MotifOccurrence[] = [];
  for (const items of perMotif) {
    if (items) merged.push(...items);
  }
  merged.sort((a, b) => a.start_timestamp - b.start_timestamp);
  return merged;
}

/**
 * Filter the cached motif list to the requested ``length`` and fan out
 * per-motif occurrence fetches. Returns a flat, time-sorted array of
 * occurrences across every length-N motif. Reuses the same query keys as
 * ``useMotifOccurrences`` so single-motif mode and byLength mode share
 * the React Query cache (no redundant fetches when toggling between
 * modes for an already-loaded motif).
 */
export function useMotifsByLength(
  motifJobId: string | null,
  motifs: MotifSummary[],
  length: number | null,
  enabled = true,
): {
  motifs: MotifSummary[];
  occurrences: MotifOccurrence[];
  isLoading: boolean;
} {
  const motifsOfLength = useMemo(
    () => filterMotifsByLength(motifs, length),
    [motifs, length],
  );

  const queries = useQueries({
    queries: motifsOfLength.map((m) => ({
      queryKey: ["motif-occurrences", motifJobId, m.motif_key, 0, 100],
      queryFn: () =>
        fetchMotifOccurrences(motifJobId as string, m.motif_key, 0, 100),
      enabled: enabled && motifJobId != null && length != null,
    })),
  });

  const isLoading = queries.some((q) => q.isLoading || q.isFetching);

  // Memo key derived from per-query ``dataUpdatedAt`` (a stable timestamp
  // React Query bumps only when the underlying data changes). This keeps
  // the merged array's reference stable between renders so downstream
  // consumers (e.g. ``visibleByLengthOccurrences`` in TimelineBody) don't
  // recompute on every render.
  const dataKey = queries.map((q) => q.dataUpdatedAt).join("|");
  const occurrences = useMemo(
    () => mergeOccurrencesByStart(queries.map((q) => q.data?.items)),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [dataKey],
  );

  return { motifs: motifsOfLength, occurrences, isLoading };
}

// ---------------------------------------------------------------------------
// Interpretation visualizations (PR 3)
// ---------------------------------------------------------------------------

export interface OverlayPoint {
  sequence_id: string;
  position_in_sequence: number;
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
  // Unified nested shape (ADR-060): outer = state, middle = tier bucket,
  // inner = label. SurfPerch jobs use the synthetic "all" tier; CRNN jobs
  // use real tier keys ("event_core" / "near_event" / "background").
  states: Record<string, Record<string, Record<string, number>>>;
}

export interface ExemplarRecord {
  sequence_id: string;
  position_in_sequence: number;
  audio_file_id: number | null;
  start_timestamp: number;
  end_timestamp: number;
  max_state_probability: number;
  exemplar_type: string;
  extras: Record<string, string | number | null>;
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

// ---------------------------------------------------------------------------
// Masked Transformer Jobs (ADR-061)
// ---------------------------------------------------------------------------

const MT_ROOT = "/sequence-models/masked-transformers";

export type MaskedTransformerPreset = "small" | "default" | "large";

export interface MaskedTransformerJob {
  id: string;
  status: string;
  status_reason: string | null;
  continuous_embedding_job_id: string;
  training_signature: string;
  preset: MaskedTransformerPreset;
  mask_fraction: number;
  span_length_min: number;
  span_length_max: number;
  dropout: number;
  mask_weight_bias: boolean;
  cosine_loss_weight: number;
  max_epochs: number;
  early_stop_patience: number;
  val_split: number;
  seed: number;
  k_values: number[];
  chosen_device: string | null;
  fallback_reason: string | null;
  final_train_loss: number | null;
  final_val_loss: number | null;
  total_epochs: number | null;
  job_dir: string | null;
  total_sequences: number | null;
  total_chunks: number | null;
  error_message: string | null;
  created_at: string;
  updated_at: string;
}

export interface MaskedTransformerJobDetail {
  job: MaskedTransformerJob;
  region_detection_job_id: string | null;
  region_start_timestamp: number | null;
  region_end_timestamp: number | null;
  tier_composition: StateTierComposition[] | null;
  source_kind: ContinuousEmbeddingSourceKind;
}

export interface MaskedTransformerJobCreate {
  continuous_embedding_job_id: string;
  preset?: MaskedTransformerPreset;
  k_values?: number[];
  mask_fraction?: number;
  span_length_min?: number;
  span_length_max?: number;
  dropout?: number;
  mask_weight_bias?: boolean;
  cosine_loss_weight?: number;
  max_epochs?: number;
  early_stop_patience?: number;
  val_split?: number;
  seed?: number;
}

export interface ExtendKSweepRequest {
  additional_k: number[];
}

export interface LossCurveResponse {
  epochs: number[];
  train_loss: number[];
  val_loss: (number | null)[];
  val_metrics: Record<string, unknown>;
}

export interface ReconstructionErrorRow {
  sequence_id: string;
  position: number;
  score: number;
  start_timestamp: number;
  end_timestamp: number;
}

export interface ReconstructionErrorResponse {
  total: number;
  offset: number;
  limit: number;
  items: ReconstructionErrorRow[];
}

export interface TokenRow {
  sequence_id: string;
  position: number;
  label: number;
  confidence: number;
  start_timestamp: number;
  end_timestamp: number;
  tier: string | null;
  audio_file_id: number | null;
}

export interface TokensResponse {
  total: number;
  offset: number;
  limit: number;
  items: TokenRow[];
}

export interface RunLengthsResponse {
  k: number;
  tau: number;
  run_lengths: Record<string, number[]>;
}

export interface GenerateMaskedTransformerInterpretationsRequest {
  k_values?: number[] | null;
}

export function fetchMaskedTransformerJobs(params?: {
  status?: string;
  continuous_embedding_job_id?: string;
}): Promise<MaskedTransformerJob[]> {
  const search = new URLSearchParams();
  if (params?.status) search.set("status", params.status);
  if (params?.continuous_embedding_job_id)
    search.set("continuous_embedding_job_id", params.continuous_embedding_job_id);
  const q = search.toString() ? `?${search.toString()}` : "";
  return request<MaskedTransformerJob[]>(`${MT_ROOT}${q}`);
}

export function fetchMaskedTransformerJob(
  jobId: string,
): Promise<MaskedTransformerJobDetail> {
  return request<MaskedTransformerJobDetail>(`${MT_ROOT}/${jobId}`);
}

export function createMaskedTransformerJob(
  body: MaskedTransformerJobCreate,
): Promise<MaskedTransformerJob> {
  return request<MaskedTransformerJob>(MT_ROOT, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function cancelMaskedTransformerJob(
  jobId: string,
): Promise<MaskedTransformerJob> {
  return request<MaskedTransformerJob>(`${MT_ROOT}/${jobId}/cancel`, {
    method: "POST",
  });
}

export function deleteMaskedTransformerJob(jobId: string): Promise<void> {
  return request<void>(`${MT_ROOT}/${jobId}`, { method: "DELETE" });
}

export function postExtendKSweep(
  jobId: string,
  body: ExtendKSweepRequest,
): Promise<MaskedTransformerJob> {
  return request<MaskedTransformerJob>(`${MT_ROOT}/${jobId}/extend-k-sweep`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function postGenerateMaskedTransformerInterpretations(
  jobId: string,
  body: GenerateMaskedTransformerInterpretationsRequest,
): Promise<{ status: string; job_id: string; k_values: number[] }> {
  return request(`${MT_ROOT}/${jobId}/generate-interpretations`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

export function fetchMaskedTransformerLossCurve(
  jobId: string,
): Promise<LossCurveResponse> {
  return request(`${MT_ROOT}/${jobId}/loss-curve`);
}

export function fetchMaskedTransformerReconstructionError(
  jobId: string,
  offset = 0,
  limit = 5000,
): Promise<ReconstructionErrorResponse> {
  return request(
    `${MT_ROOT}/${jobId}/reconstruction-error?offset=${offset}&limit=${limit}`,
  );
}

export function fetchMaskedTransformerTokens(
  jobId: string,
  k: number | null,
  offset = 0,
  limit = 5000,
): Promise<TokensResponse> {
  const params = new URLSearchParams();
  if (k != null) params.set("k", String(k));
  params.set("offset", String(offset));
  params.set("limit", String(limit));
  return request(`${MT_ROOT}/${jobId}/tokens?${params.toString()}`);
}

export function fetchMaskedTransformerOverlay(
  jobId: string,
  k: number | null,
  offset = 0,
  limit = 5000,
): Promise<OverlayResponse> {
  const params = new URLSearchParams();
  if (k != null) params.set("k", String(k));
  params.set("offset", String(offset));
  params.set("limit", String(limit));
  return request(`${MT_ROOT}/${jobId}/overlay?${params.toString()}`);
}

export function fetchMaskedTransformerExemplars(
  jobId: string,
  k: number | null,
): Promise<ExemplarsResponse> {
  const params = new URLSearchParams();
  if (k != null) params.set("k", String(k));
  return request(`${MT_ROOT}/${jobId}/exemplars?${params.toString()}`);
}

export function fetchMaskedTransformerLabelDistribution(
  jobId: string,
  k: number | null,
): Promise<LabelDistribution> {
  const params = new URLSearchParams();
  if (k != null) params.set("k", String(k));
  return request(`${MT_ROOT}/${jobId}/label-distribution?${params.toString()}`);
}

export function fetchMaskedTransformerRunLengths(
  jobId: string,
  k: number | null,
): Promise<RunLengthsResponse> {
  const params = new URLSearchParams();
  if (k != null) params.set("k", String(k));
  return request(`${MT_ROOT}/${jobId}/run-lengths?${params.toString()}`);
}

export function isMaskedTransformerJobActive(
  job: Pick<MaskedTransformerJob, "status">,
): boolean {
  return ACTIVE_STATUSES.has(job.status);
}

export function useMaskedTransformerJobs(
  params?: { status?: string; continuous_embedding_job_id?: string },
  enabled = true,
  refetchInterval: number | false = 3000,
) {
  return useQuery({
    queryKey: ["masked-transformer-jobs", params],
    queryFn: () => fetchMaskedTransformerJobs(params),
    enabled,
    refetchInterval,
  });
}

export function useMaskedTransformerJob(jobId: string | null) {
  return useQuery({
    queryKey: ["masked-transformer-job", jobId],
    queryFn: () => fetchMaskedTransformerJob(jobId as string),
    enabled: jobId != null,
    refetchInterval: (query) => {
      const data = query.state.data as MaskedTransformerJobDetail | undefined;
      if (!data) return 3000;
      return isMaskedTransformerJobActive(data.job) ? 3000 : false;
    },
  });
}

export function useCreateMaskedTransformerJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: MaskedTransformerJobCreate) =>
      createMaskedTransformerJob(body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["masked-transformer-jobs"] });
    },
  });
}

export function useCancelMaskedTransformerJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (jobId: string) => cancelMaskedTransformerJob(jobId),
    onSuccess: (_data, jobId) => {
      qc.invalidateQueries({ queryKey: ["masked-transformer-jobs"] });
      qc.invalidateQueries({ queryKey: ["masked-transformer-job", jobId] });
    },
  });
}

export function useDeleteMaskedTransformerJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (jobId: string) => deleteMaskedTransformerJob(jobId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["masked-transformer-jobs"] });
    },
  });
}

export function useExtendKSweep() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ jobId, body }: { jobId: string; body: ExtendKSweepRequest }) =>
      postExtendKSweep(jobId, body),
    onSuccess: (_data, vars) => {
      qc.invalidateQueries({
        queryKey: ["masked-transformer-job", vars.jobId],
      });
      qc.invalidateQueries({ queryKey: ["masked-transformer-jobs"] });
    },
  });
}

export function useGenerateMaskedTransformerInterpretations() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({
      jobId,
      body,
    }: {
      jobId: string;
      body: GenerateMaskedTransformerInterpretationsRequest;
    }) => postGenerateMaskedTransformerInterpretations(jobId, body),
    onSuccess: (_data, vars) => {
      qc.invalidateQueries({
        queryKey: ["masked-transformer-overlay", vars.jobId],
      });
      qc.invalidateQueries({
        queryKey: ["masked-transformer-exemplars", vars.jobId],
      });
      qc.invalidateQueries({
        queryKey: ["masked-transformer-label-distribution", vars.jobId],
      });
    },
  });
}

export function useMaskedTransformerLossCurve(
  jobId: string | null,
  enabled = true,
) {
  return useQuery({
    queryKey: ["masked-transformer-loss-curve", jobId],
    queryFn: () => fetchMaskedTransformerLossCurve(jobId as string),
    enabled: enabled && jobId != null,
  });
}

export function useMaskedTransformerReconstructionError(
  jobId: string | null,
  offset = 0,
  limit = 5000,
  enabled = true,
) {
  return useQuery({
    queryKey: ["masked-transformer-reconstruction-error", jobId, offset, limit],
    queryFn: () =>
      fetchMaskedTransformerReconstructionError(jobId as string, offset, limit),
    enabled: enabled && jobId != null,
  });
}

export function useMaskedTransformerTokens(
  jobId: string | null,
  k: number | null,
  offset = 0,
  limit = 5000,
  enabled = true,
) {
  return useQuery({
    queryKey: ["masked-transformer-tokens", jobId, k, offset, limit],
    queryFn: () =>
      fetchMaskedTransformerTokens(jobId as string, k, offset, limit),
    enabled: enabled && jobId != null,
  });
}

export function useMaskedTransformerOverlay(
  jobId: string | null,
  k: number | null,
  offset = 0,
  limit = 5000,
  enabled = true,
) {
  return useQuery({
    queryKey: ["masked-transformer-overlay", jobId, k, offset, limit],
    queryFn: () =>
      fetchMaskedTransformerOverlay(jobId as string, k, offset, limit),
    enabled: enabled && jobId != null,
  });
}

export function useMaskedTransformerExemplars(
  jobId: string | null,
  k: number | null,
  enabled = true,
) {
  return useQuery({
    queryKey: ["masked-transformer-exemplars", jobId, k],
    queryFn: () => fetchMaskedTransformerExemplars(jobId as string, k),
    enabled: enabled && jobId != null,
  });
}

export function useMaskedTransformerLabelDistribution(
  jobId: string | null,
  k: number | null,
  enabled = true,
) {
  return useQuery({
    queryKey: ["masked-transformer-label-distribution", jobId, k],
    queryFn: () => fetchMaskedTransformerLabelDistribution(jobId as string, k),
    enabled: enabled && jobId != null,
  });
}

export function useMaskedTransformerRunLengths(
  jobId: string | null,
  k: number | null,
  enabled = true,
) {
  return useQuery({
    queryKey: ["masked-transformer-run-lengths", jobId, k],
    queryFn: () => fetchMaskedTransformerRunLengths(jobId as string, k),
    enabled: enabled && jobId != null,
  });
}
