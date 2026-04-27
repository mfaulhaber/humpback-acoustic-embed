import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

export interface ContinuousEmbeddingJob {
  id: string;
  status: string;
  region_detection_job_id: string;
  model_version: string;
  window_size_seconds: number;
  hop_seconds: number;
  pad_seconds: number;
  target_sample_rate: number;
  feature_config_json: string | null;
  encoding_signature: string;
  vector_dim: number | null;
  total_regions: number | null;
  merged_spans: number | null;
  total_windows: number | null;
  parquet_path: string | null;
  error_message: string | null;
  created_at: string;
  updated_at: string;
}

export interface ContinuousEmbeddingSpanSummary {
  merged_span_id: number;
  start_time_sec: number;
  end_time_sec: number;
  window_count: number;
  source_region_ids: string[];
}

export interface ContinuousEmbeddingJobManifest {
  job_id: string;
  model_version: string;
  vector_dim: number;
  window_size_seconds: number;
  hop_seconds: number;
  pad_seconds: number;
  target_sample_rate: number;
  total_regions: number;
  merged_spans: number;
  total_windows: number;
  spans: ContinuousEmbeddingSpanSummary[];
}

export interface ContinuousEmbeddingJobDetail {
  job: ContinuousEmbeddingJob;
  manifest: ContinuousEmbeddingJobManifest | null;
}

export interface CreateContinuousEmbeddingJobRequest {
  region_detection_job_id: string;
  model_version?: string;
  hop_seconds?: number;
  pad_seconds?: number;
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
