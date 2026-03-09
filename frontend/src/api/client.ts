import type {
  AudioFile,
  AvailableModelFile,
  FolderImportResult,
  ClassifierModelInfo,
  ClassifierReport,
  ClassifierTrainingJob,
  ClassifierTrainingJobCreate,
  ClusterAssignment,
  ClusteringJob,
  ClusteringJobCreate,
  ClusteringMetrics,
  ClusterOut,
  DendrogramData,
  DetectionJob,
  DetectionLabelRow,
  DetectionRow,
  DirectoryListing,
  ExtractionSettings,
  EmbeddingSet,
  EmbeddingSimilarity,
  FolderDeletePreview,
  FolderDeleteResult,
  FragmentationReport,
  HydrophoneDetectionJobCreate,
  HydrophoneInfo,
  LabelQueueEntry,
  RefinementReport,
  StabilitySummary,
  ModelConfig,
  ModelConfigCreate,
  ParameterSweepPoint,
  ProcessingJob,
  ProcessingJobCreate,
  SpectrogramData,
  TableInfo,
  TrainingDataSummaryResponse,
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

function put<T>(path: string, body: unknown): Promise<T> {
  return api<T>(path, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

// ---- Audio ----

export const importFolder = (folderPath: string) =>
  api<FolderImportResult>(`/audio/import-folder?folder_path=${encodeURIComponent(folderPath)}`, {
    method: "POST",
  });

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

export const deleteProcessingJob = (jobId: string) =>
  api<{ status: string }>(`/processing/jobs/${jobId}`, { method: "DELETE" });

export const bulkDeleteProcessingJobs = (ids: string[]) =>
  post<{ status: string; count: number }>("/processing/jobs/bulk-delete", { ids });

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

export const fetchDendrogram = (jobId: string) =>
  api<DendrogramData>(`/clustering/jobs/${jobId}/dendrogram`);

export const fetchFragmentation = (jobId: string) =>
  api<FragmentationReport>(`/clustering/jobs/${jobId}/fragmentation`);

export const fetchStability = (jobId: string) =>
  api<StabilitySummary>(`/clustering/jobs/${jobId}/stability`);

export const fetchClassifier = (jobId: string) =>
  api<ClassifierReport>(`/clustering/jobs/${jobId}/classifier`);

export const fetchLabelQueue = (jobId: string) =>
  api<LabelQueueEntry[]>(`/clustering/jobs/${jobId}/label-queue`);

export const fetchRefinement = (jobId: string) =>
  api<RefinementReport>(`/clustering/jobs/${jobId}/refinement`);

export const fetchAssignments = (clusterId: string) =>
  api<ClusterAssignment[]>(`/clustering/clusters/${clusterId}/assignments`);

export const deleteClusteringJob = (jobId: string) =>
  api<{ status: string }>(`/clustering/jobs/${jobId}`, { method: "DELETE" });

// ---- Binary Classifier ----

export const fetchTrainingJobs = () =>
  api<ClassifierTrainingJob[]>("/classifier/training-jobs");

export const createTrainingJob = (body: ClassifierTrainingJobCreate) =>
  post<ClassifierTrainingJob>("/classifier/training-jobs", body);

export const fetchTrainingJob = (jobId: string) =>
  api<ClassifierTrainingJob>(`/classifier/training-jobs/${jobId}`);

export const fetchClassifierModels = () =>
  api<ClassifierModelInfo[]>("/classifier/models");

export const fetchClassifierModel = (modelId: string) =>
  api<ClassifierModelInfo>(`/classifier/models/${modelId}`);

export const deleteClassifierModel = (modelId: string) =>
  api<{ status: string }>(`/classifier/models/${modelId}`, { method: "DELETE" });

export const fetchDetectionJob = (jobId: string) =>
  api<DetectionJob>(`/classifier/detection-jobs/${jobId}`);

export function detectionTsvUrl(jobId: string) {
  return `/classifier/detection-jobs/${jobId}/download`;
}

export const browseDirectories = (root: string) =>
  api<DirectoryListing>(`/classifier/browse-directories?root=${encodeURIComponent(root)}`);

export const deleteTrainingJob = (jobId: string) =>
  api<{ status: string }>(`/classifier/training-jobs/${jobId}`, { method: "DELETE" });

export const bulkDeleteTrainingJobs = (ids: string[]) =>
  post<{ status: string; count: number }>("/classifier/training-jobs/bulk-delete", { ids });

export const bulkDeleteClassifierModels = (ids: string[]) =>
  post<{ status: string; count: number }>("/classifier/models/bulk-delete", { ids });

export const deleteDetectionJob = (jobId: string) =>
  api<{ status: string }>(`/classifier/detection-jobs/${jobId}`, { method: "DELETE" });

export const bulkDeleteDetectionJobs = (ids: string[]) =>
  post<{ status: string; count: number }>("/classifier/detection-jobs/bulk-delete", { ids });

export const fetchDetectionContent = (jobId: string) =>
  api<DetectionRow[]>(`/classifier/detection-jobs/${jobId}/content`);

export const saveDetectionLabels = (jobId: string, rows: DetectionLabelRow[]) =>
  put<{ status: string; updated: number }>(
    `/classifier/detection-jobs/${jobId}/labels`,
    rows,
  );

export const fetchExtractionSettings = () =>
  api<ExtractionSettings>("/classifier/extraction-settings");

export const extractLabeledSamples = (
  jobIds: string[],
  positiveOutputPath?: string,
  negativeOutputPath?: string,
) =>
  post<{ status: string; count: number }>("/classifier/detection-jobs/extract", {
    job_ids: jobIds,
    positive_output_path: positiveOutputPath,
    negative_output_path: negativeOutputPath,
  });

export function detectionAudioSliceUrl(
  jobId: string,
  filename: string,
  startSec: number,
  durationSec: number,
) {
  return `/classifier/detection-jobs/${jobId}/audio-slice?filename=${encodeURIComponent(filename)}&start_sec=${startSec}&duration_sec=${durationSec}`;
}

// ---- Hydrophone Detection ----

export const fetchHydrophones = () =>
  api<HydrophoneInfo[]>("/classifier/hydrophones");

export const fetchHydrophoneDetectionJobs = () =>
  api<DetectionJob[]>("/classifier/hydrophone-detection-jobs");

export const createHydrophoneDetectionJob = (body: HydrophoneDetectionJobCreate) =>
  post<DetectionJob>("/classifier/hydrophone-detection-jobs", body);

export const cancelHydrophoneDetectionJob = (jobId: string) =>
  api<{ status: string }>(`/classifier/hydrophone-detection-jobs/${jobId}/cancel`, {
    method: "POST",
  });

export const pauseHydrophoneDetectionJob = (jobId: string) =>
  api<{ status: string }>(`/classifier/hydrophone-detection-jobs/${jobId}/pause`, {
    method: "POST",
  });

export const resumeHydrophoneDetectionJob = (jobId: string) =>
  api<{ status: string }>(`/classifier/hydrophone-detection-jobs/${jobId}/resume`, {
    method: "POST",
  });

export const fetchTrainingDataSummary = (modelId: string) =>
  api<TrainingDataSummaryResponse>(`/classifier/models/${modelId}/training-summary`);

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
