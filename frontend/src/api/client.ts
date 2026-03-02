import type {
  AudioFile,
  AvailableModelFile,
  ClusterAssignment,
  ClusteringJob,
  ClusteringJobCreate,
  ClusteringMetrics,
  ClusterOut,
  EmbeddingSet,
  EmbeddingSimilarity,
  FolderDeletePreview,
  FolderDeleteResult,
  ModelConfig,
  ModelConfigCreate,
  ParameterSweepPoint,
  ProcessingJob,
  ProcessingJobCreate,
  SpectrogramData,
  TableInfo,
  VisualizationData,
} from "./types";

class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, init);
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new ApiError(res.status, text);
  }
  return res.json();
}

function post<T>(path: string, body: unknown): Promise<T> {
  return api<T>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

// ---- Audio ----

export async function uploadAudio(file: File, folderPath?: string): Promise<AudioFile> {
  const form = new FormData();
  form.append("file", file);
  if (folderPath) form.append("folder_path", folderPath);
  const res = await fetch("/audio/upload", { method: "POST", body: form });
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new ApiError(res.status, text);
  }
  return res.json();
}

export const fetchAudioFiles = () => api<AudioFile[]>("/audio/");

export const fetchAudioFile = (id: string) => api<AudioFile>(`/audio/${id}`);

export const fetchSpectrogram = (
  audioId: string,
  windowIndex: number,
  windowSizeSeconds = 5,
  targetSampleRate = 32000,
) =>
  api<SpectrogramData>(
    `/audio/${audioId}/spectrogram?window_index=${windowIndex}&window_size_seconds=${windowSizeSeconds}&target_sample_rate=${targetSampleRate}`,
  );

export const fetchEmbeddings = (audioId: string, embeddingSetId: string) =>
  api<EmbeddingSimilarity>(`/audio/${audioId}/embeddings?embedding_set_id=${embeddingSetId}`);

export function audioDownloadUrl(audioId: string) {
  return `/audio/${audioId}/download`;
}

export function audioWindowUrl(audioId: string, startSeconds: number, durationSeconds: number) {
  return `/audio/${audioId}/window?start_seconds=${startSeconds}&duration_seconds=${durationSeconds}`;
}

// ---- Folder Delete ----

export const fetchFolderDeletePreview = (folderPath: string) =>
  api<FolderDeletePreview>(`/audio/folders/delete-preview?folder_path=${encodeURIComponent(folderPath)}`);

export const deleteFolder = (folderPath: string, confirmClusteringDelete: boolean) =>
  api<FolderDeleteResult>(
    `/audio/folders?folder_path=${encodeURIComponent(folderPath)}&confirm_clustering_delete=${confirmClusteringDelete}`,
    { method: "DELETE" },
  );

// ---- Processing ----

export const fetchProcessingJobs = () => api<ProcessingJob[]>("/processing/jobs");

export const createProcessingJob = (body: ProcessingJobCreate) =>
  post<ProcessingJob>("/processing/jobs", body);

export const cancelProcessingJob = (jobId: string) =>
  api<ProcessingJob>(`/processing/jobs/${jobId}/cancel`, { method: "POST" });

export const fetchEmbeddingSets = () => api<EmbeddingSet[]>("/processing/embedding-sets");

// ---- Clustering ----

export const fetchClusteringJobs = () => api<ClusteringJob[]>("/clustering/jobs");

export const createClusteringJob = (body: ClusteringJobCreate) =>
  post<ClusteringJob>("/clustering/jobs", body);

export const fetchClusters = (jobId: string) =>
  api<ClusterOut[]>(`/clustering/jobs/${jobId}/clusters`);

export const fetchVisualization = (jobId: string) =>
  api<VisualizationData>(`/clustering/jobs/${jobId}/visualization`);

export const fetchMetrics = (jobId: string) =>
  api<ClusteringMetrics>(`/clustering/jobs/${jobId}/metrics`);

export const fetchParameterSweep = (jobId: string) =>
  api<ParameterSweepPoint[]>(`/clustering/jobs/${jobId}/parameter-sweep`);

export const fetchAssignments = (clusterId: string) =>
  api<ClusterAssignment[]>(`/clustering/clusters/${clusterId}/assignments`);

// ---- Admin ----

export const fetchModels = () => api<ModelConfig[]>("/admin/models");

export const createModel = (body: ModelConfigCreate) =>
  post<ModelConfig>("/admin/models", body);

export const deleteModel = (modelId: string) =>
  api<{ status: string }>(`/admin/models/${modelId}`, { method: "DELETE" });

export const setDefaultModel = (modelId: string) =>
  api<ModelConfig>(`/admin/models/${modelId}/set-default`, { method: "POST" });

export const scanModels = () => api<AvailableModelFile[]>("/admin/models/scan");

export const fetchTables = () => api<TableInfo[]>("/admin/tables");

export const deleteAllRecords = () =>
  api<{ status: string; message: string }>("/admin/tables", { method: "DELETE" });

export { ApiError };
