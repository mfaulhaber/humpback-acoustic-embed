import type {
  AudioFile,
  AutoresearchCandidateDetail,
  AutoresearchCandidateImport,
  AutoresearchCandidateSummary,
  AutoresearchCandidateTrainingJobCreate,
  HyperparameterManifestDetail,
  HyperparameterManifestSummary,
  HyperparameterSearchDetail,
  HyperparameterSearchSummary,
  ManifestCreateRequest,
  SearchCreateRequest,
  SearchHistoryEntry,
  SearchSpaceDefaults,
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
  DetectionEmbeddingResponse,
  DetectionJob,
  DetectionLabelRow,
  DetectionRow,
  DetectionRowStateResponse,
  DetectionRowStateUpdate,
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
  SimilaritySearchResponse,
  AudioSearchRequest,
  SearchJobResponse,
  StabilitySummary,
  ModelConfig,
  ModelConfigCreate,
  ParameterSweepPoint,
  ProcessingJob,
  ProcessingJobCreate,
  SpectrogramData,
  TableInfo,
  RetrainFolderInfo,
  RetrainWorkflow,
  RetrainWorkflowCreate,
  TrainingDataSummaryResponse,
  VisualizationData,
  HealthStatus,
  LabelProcessingJob,
  LabelProcessingJobCreate,
  LabelProcessingPreview,
  VocalizationLabel,
  TimelineVocalizationLabel,
  VocalizationLabelBatchRequest,
  DetectionNeighborsResponse,
  LabelingSummary,
  TrainingSummary,
  TimelineConfidenceResponse,
  LabelEditItem,
  VocalizationType,
  VocalizationTypeCreate,
  VocalizationTypeUpdate,
  VocalizationTypeImportRequest,
  VocalizationTypeImportResponse,
  VocClassifierTrainingJobCreate,
  VocClassifierTrainingJob,
  VocClassifierModel,
  VocClassifierInferenceJobCreate,
  VocClassifierInferenceJob,
  VocClassifierPredictionRow,
  EmbeddingStatus,
  DetectionEmbeddingJob,
  EmbeddingJobListItem,
  VocalizationTrainingSource,
  FolderEmbeddingSetResponse,
  TrainingDataset,
  TrainingDatasetRowsResponse,
  TrainingDatasetLabel,
  TrainingDatasetExtendRequest,
  RegionDetectionJob,
  CreateRegionJobRequest,
  EventSegmentationJob,
  CreateSegmentationJobRequest,
  SegmentationEvent,
  SegmentationModel,
  SegmentationTrainingJob,
  CreateSegmentationTrainingJobRequest,
  SegmentationTrainingDatasetSummary,
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

function patch<T>(path: string, body: unknown): Promise<T> {
  return api<T>(path, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

const HP = "/classifier/hyperparameter";

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

export const fetchFolderEmbeddingSet = (folderPath: string) =>
  post<FolderEmbeddingSetResponse>("/processing/folder-embedding-set", {
    folder_path: folderPath,
  });

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

export const importAutoresearchCandidate = (body: AutoresearchCandidateImport) =>
  post<AutoresearchCandidateDetail>(`${HP}/candidates/import`, body);

export const fetchAutoresearchCandidates = () =>
  api<AutoresearchCandidateSummary[]>(`${HP}/candidates`);

export const fetchAutoresearchCandidate = (candidateId: string) =>
  api<AutoresearchCandidateDetail>(`${HP}/candidates/${candidateId}`);

export const createAutoresearchCandidateTrainingJob = (
  candidateId: string,
  body: AutoresearchCandidateTrainingJobCreate,
) =>
  post<ClassifierTrainingJob>(
    `${HP}/candidates/${candidateId}/training-jobs`,
    body,
  );

export const fetchClassifierModels = () =>
  api<ClassifierModelInfo[]>("/classifier/models");

export const fetchClassifierModel = (modelId: string) =>
  api<ClassifierModelInfo>(`/classifier/models/${modelId}`);

export const deleteClassifierModel = (modelId: string) =>
  api<{ status: string }>(`/classifier/models/${modelId}`, { method: "DELETE" });

export const fetchDetectionJobs = () =>
  api<DetectionJob[]>("/classifier/detection-jobs");

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

export const saveDetectionRowState = (
  jobId: string,
  body: DetectionRowStateUpdate,
) =>
  put<DetectionRowStateResponse>(
    `/classifier/detection-jobs/${jobId}/row-state`,
    body,
  );

export const fetchExtractionSettings = () =>
  api<ExtractionSettings>("/classifier/extraction-settings");

export const extractLabeledSamples = (
  jobIds: string[],
  positiveOutputPath?: string,
  negativeOutputPath?: string,
  options?: {
    positiveSelectionSmoothingWindow?: number;
    positiveSelectionMinScore?: number;
    positiveSelectionExtendMinScore?: number;
  },
) =>
  post<{ status: string; count: number }>("/classifier/detection-jobs/extract", {
    job_ids: jobIds,
    positive_output_path: positiveOutputPath,
    negative_output_path: negativeOutputPath,
    positive_selection_smoothing_window: options?.positiveSelectionSmoothingWindow,
    positive_selection_min_score: options?.positiveSelectionMinScore,
    positive_selection_extend_min_score: options?.positiveSelectionExtendMinScore,
  });

export function detectionAudioSliceUrl(
  jobId: string,
  startUtc: number,
  durationSec: number,
) {
  return `/classifier/detection-jobs/${jobId}/audio-slice?start_utc=${startUtc}&duration_sec=${durationSec}`;
}

export function detectionSpectrogramUrl(
  jobId: string,
  startUtc: number,
  durationSec: number,
) {
  return `/classifier/detection-jobs/${jobId}/spectrogram?start_utc=${startUtc}&duration_sec=${durationSec}`;
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

// ---- Retrain Workflows ----

export const fetchRetrainInfo = (modelId: string) =>
  api<RetrainFolderInfo>(`/classifier/models/${modelId}/retrain-info`);

export const createRetrainWorkflow = (body: RetrainWorkflowCreate) =>
  post<RetrainWorkflow>("/classifier/retrain", body);

export const fetchRetrainWorkflows = () =>
  api<RetrainWorkflow[]>("/classifier/retrain-workflows");

// ---- Search ----

export const searchSimilar = (body: {
  embedding_set_id: string;
  row_index: number;
  top_k?: number;
  metric?: string;
  exclude_self?: boolean;
  embedding_set_ids?: string[];
}) => post<SimilaritySearchResponse>("/search/similar", body);

export const searchSimilarByVector = (body: {
  vector: number[];
  model_version: string;
  top_k?: number;
  metric?: string;
  embedding_set_ids?: string[];
}) => post<SimilaritySearchResponse>("/search/similar-by-vector", body);

export const fetchDetectionEmbedding = (
  jobId: string,
  rowId: string,
) =>
  api<DetectionEmbeddingResponse>(
    `/classifier/detection-jobs/${jobId}/embedding?row_id=${encodeURIComponent(rowId)}`,
  );

export const createAudioSearch = (body: AudioSearchRequest) =>
  post<SearchJobResponse>("/search/similar-by-audio", body);

export const pollSearchJob = (jobId: string) =>
  api<SearchJobResponse>(`/search/jobs/${jobId}`);

export function audioSpectrogramPngUrl(
  audioId: string,
  startSeconds: number,
  durationSeconds: number,
) {
  return `/audio/${audioId}/spectrogram-png?start_seconds=${startSeconds}&duration_seconds=${durationSeconds}`;
}

// ---- Label Processing ----

export const fetchLabelProcessingJobs = () =>
  api<LabelProcessingJob[]>("/label-processing/jobs");

export const createLabelProcessingJob = (body: LabelProcessingJobCreate) =>
  post<LabelProcessingJob>("/label-processing/jobs", body);

export const deleteLabelProcessingJob = (jobId: string) =>
  api<{ status: string }>(`/label-processing/jobs/${jobId}`, { method: "DELETE" });

export const fetchLabelProcessingPreview = (annotationFolder: string, audioFolder: string) =>
  api<LabelProcessingPreview>(
    `/label-processing/preview?annotation_folder=${encodeURIComponent(annotationFolder)}&audio_folder=${encodeURIComponent(audioFolder)}`,
  );

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

// ---- Labeling ----

export const fetchVocalizationLabels = (
  detectionJobId: string,
  rowId: string,
) =>
  api<VocalizationLabel[]>(
    `/labeling/vocalization-labels/${detectionJobId}?row_id=${encodeURIComponent(rowId)}`,
  );

export const fetchAllVocalizationLabels = (detectionJobId: string) =>
  api<TimelineVocalizationLabel[]>(
    `/labeling/vocalization-labels/${detectionJobId}/all`,
  );

export const createVocalizationLabel = (
  detectionJobId: string,
  rowId: string,
  body: { label: string; confidence?: number; source?: string; notes?: string },
) =>
  post<VocalizationLabel>(
    `/labeling/vocalization-labels/${detectionJobId}?row_id=${encodeURIComponent(rowId)}`,
    body,
  );

export const updateVocalizationLabel = (
  labelId: string,
  body: { label?: string; confidence?: number; notes?: string },
) => put<VocalizationLabel>(`/labeling/vocalization-labels/${labelId}`, body);

export const deleteVocalizationLabel = async (labelId: string) => {
  const res = await fetch(`/labeling/vocalization-labels/${labelId}`, {
    method: "DELETE",
  });
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new ApiError(res.status, text);
  }
};

export const patchVocalizationLabels = (
  detectionJobId: string,
  body: VocalizationLabelBatchRequest,
) =>
  patch<TimelineVocalizationLabel[]>(
    `/labeling/vocalization-labels/${detectionJobId}/batch`,
    body,
  );

export const fetchLabelVocabulary = () =>
  api<string[]>("/labeling/label-vocabulary");

export const fetchLabelingSummary = (detectionJobId: string) =>
  api<LabelingSummary>(`/labeling/summary/${detectionJobId}`);

export const fetchTrainingSummary = () =>
  api<TrainingSummary>("/labeling/training-summary");

export const fetchDetectionNeighbors = (
  detectionJobId: string,
  params: {
    row_id: string;
    top_k?: number;
    metric?: string;
    embedding_set_ids?: string[];
  },
) =>
  post<DetectionNeighborsResponse>(
    `/labeling/detection-neighbors/${detectionJobId}`,
    params,
  );

// ---- Timeline viewer API ----

export function timelineTileUrl(
  jobId: string,
  zoomLevel: string,
  tileIndex: number,
  freqMin = 0,
  freqMax = 3000,
): string {
  return `/classifier/detection-jobs/${jobId}/timeline/tile?zoom_level=${zoomLevel}&tile_index=${tileIndex}&freq_min=${freqMin}&freq_max=${freqMax}`;
}

export function timelineAudioUrl(
  jobId: string,
  startSec: number,
  durationSec = 300,
  format = "mp3",
): string {
  const params = new URLSearchParams({
    start_sec: startSec.toString(),
    duration_sec: durationSec.toString(),
    format,
  });
  return `/classifier/detection-jobs/${jobId}/timeline/audio?${params}`;
}

export const fetchTimelineConfidence = (jobId: string) =>
  api<TimelineConfidenceResponse>(
    `/classifier/detection-jobs/${jobId}/timeline/confidence`,
  );

export const patchDetectionLabels = (jobId: string, edits: LabelEditItem[]) =>
  patch<DetectionRow[]>(`/classifier/detection-jobs/${jobId}/labels`, { edits });

export interface PrepareTimelineTilesRequest {
  scope?: "startup" | "full";
  zoomLevel?: string;
  centerTimestamp?: number;
  radiusTiles?: number;
}

export const prepareTimelineTiles = (
  jobId: string,
  request: PrepareTimelineTilesRequest = {},
) =>
  post<{ status: string }>(
    `/classifier/detection-jobs/${jobId}/timeline/prepare`,
    {
      scope: request.scope,
      zoom_level: request.zoomLevel,
      center_timestamp: request.centerTimestamp,
      radius_tiles: request.radiusTiles,
    },
  );

export async function fetchPrepareStatus(
  jobId: number | string,
): Promise<Record<string, { total: number; rendered: number }>> {
  const resp = await fetch(
    `/classifier/detection-jobs/${jobId}/timeline/prepare-status`,
  );
  if (!resp.ok) throw new Error(`prepare-status ${resp.status}`);
  return resp.json();
}

// ---- Multi-Label Vocalization Classifier ----

// Vocabulary
export const fetchVocalizationTypes = () =>
  api<VocalizationType[]>("/vocalization/types");

export const createVocalizationType = (body: VocalizationTypeCreate) =>
  post<VocalizationType>("/vocalization/types", body);

export const updateVocalizationType = (typeId: string, body: VocalizationTypeUpdate) =>
  put<VocalizationType>(`/vocalization/types/${typeId}`, body);

export const deleteVocalizationType = async (typeId: string) => {
  const res = await fetch(`/vocalization/types/${typeId}`, { method: "DELETE" });
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new ApiError(res.status, text);
  }
};

export const importVocalizationTypes = (body: VocalizationTypeImportRequest) =>
  post<VocalizationTypeImportResponse>("/vocalization/types/import", body);

// Models
export const fetchVocClassifierModels = () =>
  api<VocClassifierModel[]>("/vocalization/models");

export const fetchVocClassifierModel = (modelId: string) =>
  api<VocClassifierModel>(`/vocalization/models/${modelId}`);

export const activateVocClassifierModel = (modelId: string) =>
  put<VocClassifierModel>(`/vocalization/models/${modelId}/activate`, {});

// Training Jobs
export const createVocClassifierTrainingJob = (body: VocClassifierTrainingJobCreate) =>
  post<VocClassifierTrainingJob>("/vocalization/training-jobs", body);

export const fetchVocClassifierTrainingJobs = () =>
  api<VocClassifierTrainingJob[]>("/vocalization/training-jobs");

export const fetchVocClassifierTrainingJob = (jobId: string) =>
  api<VocClassifierTrainingJob>(`/vocalization/training-jobs/${jobId}`);

// Inference Jobs
export const createVocClassifierInferenceJob = (body: VocClassifierInferenceJobCreate) =>
  post<VocClassifierInferenceJob>("/vocalization/inference-jobs", body);

export const fetchVocClassifierInferenceJobs = () =>
  api<VocClassifierInferenceJob[]>("/vocalization/inference-jobs");

export const fetchVocClassifierInferenceJob = (jobId: string) =>
  api<VocClassifierInferenceJob>(`/vocalization/inference-jobs/${jobId}`);

export const fetchVocClassifierInferenceResults = (
  jobId: string,
  params?: {
    offset?: number;
    limit?: number;
    thresholds?: Record<string, number>;
    sort?: string;
  },
) => {
  const qs = new URLSearchParams();
  if (params?.offset !== undefined) qs.set("offset", String(params.offset));
  if (params?.limit !== undefined) qs.set("limit", String(params.limit));
  if (params?.thresholds) qs.set("thresholds", JSON.stringify(params.thresholds));
  if (params?.sort) qs.set("sort", params.sort);
  const suffix = qs.toString() ? `?${qs}` : "";
  return api<VocClassifierPredictionRow[]>(
    `/vocalization/inference-jobs/${jobId}/results${suffix}`,
  );
};

export function vocClassifierInferenceExportUrl(
  jobId: string,
  thresholds?: Record<string, number>,
): string {
  const qs = new URLSearchParams();
  if (thresholds) qs.set("thresholds", JSON.stringify(thresholds));
  const suffix = qs.toString() ? `?${qs}` : "";
  return `/vocalization/inference-jobs/${jobId}/export${suffix}`;
}

// ---- Detection Embeddings ----

export const fetchEmbeddingStatus = (jobId: string) =>
  api<EmbeddingStatus>(`/classifier/detection-jobs/${jobId}/embedding-status`);

export const generateEmbeddings = (jobId: string, mode: "full" | "sync" = "full") =>
  post<DetectionEmbeddingJob>(`/classifier/detection-jobs/${jobId}/generate-embeddings?mode=${mode}`, {});

export const fetchEmbeddingGenerationStatus = (jobId: string) =>
  api<DetectionEmbeddingJob | null>(
    `/classifier/detection-jobs/${jobId}/embedding-generation-status`,
  );

export const fetchEmbeddingJobs = (offset = 0, limit = 50) =>
  api<EmbeddingJobListItem[]>(`/classifier/embedding-jobs?offset=${offset}&limit=${limit}`);

// ---- Vocalization Training Source ----

export const fetchVocModelTrainingSource = (modelId: string) =>
  api<VocalizationTrainingSource>(`/vocalization/models/${modelId}/training-source`);

// ---- Training Datasets ----

export const fetchTrainingDatasets = () =>
  api<TrainingDataset[]>("/vocalization/training-datasets");

export const fetchTrainingDataset = (datasetId: string) =>
  api<TrainingDataset>(`/vocalization/training-datasets/${datasetId}`);

export const fetchTrainingDatasetRows = (
  datasetId: string,
  params?: {
    type?: string;
    group?: string;
    source_type?: string;
    offset?: number;
    limit?: number;
  },
) => {
  const qs = new URLSearchParams();
  if (params?.type) qs.set("type", params.type);
  if (params?.group) qs.set("group", params.group);
  if (params?.source_type) qs.set("source_type", params.source_type);
  if (params?.offset !== undefined) qs.set("offset", String(params.offset));
  if (params?.limit !== undefined) qs.set("limit", String(params.limit));
  const suffix = qs.toString() ? `?${qs}` : "";
  return api<TrainingDatasetRowsResponse>(
    `/vocalization/training-datasets/${datasetId}/rows${suffix}`,
  );
};

export function trainingDatasetSpectrogramUrl(
  datasetId: string,
  rowIndex: number,
): string {
  return `/vocalization/training-datasets/${datasetId}/spectrogram?row_index=${rowIndex}`;
}

export function trainingDatasetAudioSliceUrl(
  datasetId: string,
  rowIndex: number,
): string {
  return `/vocalization/training-datasets/${datasetId}/audio-slice?row_index=${rowIndex}`;
}

export const extendTrainingDataset = (
  datasetId: string,
  body: TrainingDatasetExtendRequest,
) => post<TrainingDataset>(`/vocalization/training-datasets/${datasetId}/extend`, body);

export const createTrainingDatasetLabel = (
  datasetId: string,
  body: { row_index: number; label: string },
) =>
  post<TrainingDatasetLabel>(
    `/vocalization/training-datasets/${datasetId}/labels`,
    body,
  );

export const deleteTrainingDatasetLabel = async (
  datasetId: string,
  labelId: string,
) => {
  const res = await fetch(
    `/vocalization/training-datasets/${datasetId}/labels/${labelId}`,
    { method: "DELETE" },
  );
  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    throw new ApiError(res.status, text);
  }
};

// ---- Health ----

export function getHealth(): Promise<HealthStatus> {
  return fetch("/health")
    .then((r) => r.json() as Promise<HealthStatus>)
    .catch(() => ({
      status: "error" as const,
      db: "unreachable",
      detail: "Could not reach the API server.",
    }));
}

// ---- Hyperparameter Tuning ----

export const createManifest = (body: ManifestCreateRequest) =>
  post<HyperparameterManifestSummary>(`${HP}/manifests`, body);

export const listManifests = () =>
  api<HyperparameterManifestSummary[]>(`${HP}/manifests`);

export const getManifest = (id: string) =>
  api<HyperparameterManifestDetail>(`${HP}/manifests/${id}`);

export const deleteManifest = (id: string) =>
  api<{ status: string }>(`${HP}/manifests/${id}`, { method: "DELETE" });

export const createSearch = (body: SearchCreateRequest) =>
  post<HyperparameterSearchSummary>(`${HP}/searches`, body);

export const listSearches = () =>
  api<HyperparameterSearchSummary[]>(`${HP}/searches`);

export const getSearch = (id: string) =>
  api<HyperparameterSearchDetail>(`${HP}/searches/${id}`);

export const getSearchHistory = (id: string) =>
  api<SearchHistoryEntry[]>(`${HP}/searches/${id}/history`);

export const deleteSearch = (id: string) =>
  api<{ status: string }>(`${HP}/searches/${id}`, { method: "DELETE" });

export const getSearchSpaceDefaults = () =>
  api<SearchSpaceDefaults>(`${HP}/search-space-defaults`);

export const importCandidateFromSearch = (searchId: string) =>
  post<AutoresearchCandidateDetail>(`${HP}/searches/${searchId}/import-candidate`, {});

export const deleteCandidate = (id: string) =>
  api<{ status: string }>(`${HP}/candidates/${id}`, { method: "DELETE" });

// ---- Call Parsing (Region Detection) ----

export const fetchRegionDetectionJobs = () =>
  api<RegionDetectionJob[]>("/call-parsing/region-jobs");

export const createRegionDetectionJob = (body: CreateRegionJobRequest) =>
  post<RegionDetectionJob>("/call-parsing/region-jobs", body);

export const deleteRegionDetectionJob = (jobId: string) =>
  api<{ status: string }>(`/call-parsing/region-jobs/${jobId}`, {
    method: "DELETE",
  });

// ---- Call Parsing (Event Segmentation) ----

export const fetchSegmentationJobs = () =>
  api<EventSegmentationJob[]>("/call-parsing/segmentation-jobs");

export const createSegmentationJob = (body: CreateSegmentationJobRequest) =>
  post<EventSegmentationJob>("/call-parsing/segmentation-jobs", body);

export const deleteSegmentationJob = (jobId: string) =>
  api<{ status: string }>(`/call-parsing/segmentation-jobs/${jobId}`, {
    method: "DELETE",
  });

export const fetchSegmentationJobEvents = (jobId: string) =>
  api<SegmentationEvent[]>(`/call-parsing/segmentation-jobs/${jobId}/events`);

// ---- Call Parsing (Segmentation Models & Training) ----

export const fetchSegmentationModels = () =>
  api<SegmentationModel[]>("/call-parsing/segmentation-models");

export const deleteSegmentationModel = (modelId: string) =>
  api<{ status: string }>(`/call-parsing/segmentation-models/${modelId}`, {
    method: "DELETE",
  });

export const fetchSegmentationTrainingJobs = () =>
  api<SegmentationTrainingJob[]>("/call-parsing/segmentation-training-jobs");

export const createSegmentationTrainingJob = (
  body: CreateSegmentationTrainingJobRequest,
) => post<SegmentationTrainingJob>("/call-parsing/segmentation-training-jobs", body);

export const deleteSegmentationTrainingJob = (jobId: string) =>
  api<{ status: string }>(`/call-parsing/segmentation-training-jobs/${jobId}`, {
    method: "DELETE",
  });

export const fetchSegmentationTrainingDatasets = () =>
  api<SegmentationTrainingDatasetSummary[]>(
    "/call-parsing/segmentation-training-datasets",
  );
