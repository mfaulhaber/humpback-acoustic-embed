import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

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
  window_size_seconds?: number | null;
  hop_seconds?: number | null;
  pad_seconds?: number | null;
  total_events?: number | null;
  merged_spans?: number | null;
  total_windows?: number | null;
  spans?: ContinuousEmbeddingSpanSummary[];
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
  extra?: Record<string, unknown> | null;
}

export interface CreateContinuousEmbeddingJobRequest {
  event_segmentation_job_id?: string;
  event_source_mode?: "raw" | "effective";
  model_version?: string;
  hop_seconds?: number;
  pad_seconds?: number;
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

export function deleteContinuousEmbeddingJob(jobId: string): Promise<void> {
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
      const data = query.state.data as ContinuousEmbeddingJobDetail | undefined;
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
