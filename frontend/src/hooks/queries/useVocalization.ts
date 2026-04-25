import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import type {
  FolderEmbeddingSetResponse,
  TrainingDatasetExtendRequest,
  VocalizationClusteringJobCreate,
} from "@/api/types";
import {
  fetchVocalizationTypes,
  createVocalizationType,
  updateVocalizationType,
  deleteVocalizationType,
  importVocalizationTypes,
  fetchVocClassifierModels,
  fetchVocClassifierModel,
  activateVocClassifierModel,
  createVocClassifierTrainingJob,
  fetchVocClassifierTrainingJobs,
  fetchVocClassifierTrainingJob,
  createVocClassifierInferenceJob,
  fetchVocClassifierInferenceJobs,
  fetchVocClassifierInferenceJob,
  fetchVocClassifierInferenceResults,
  fetchEmbeddingStatus,
  generateEmbeddings,
  fetchEmbeddingGenerationStatus,
  fetchEmbeddingJobs,
  fetchVocModelTrainingSource,
  fetchFolderEmbeddingSet,
  fetchTrainingDataset,
  fetchTrainingDatasetRows,
  extendTrainingDataset,
  createTrainingDatasetLabel,
  deleteTrainingDatasetLabel,
  fetchClusteringEligibleJobs,
  fetchVocalizationClusteringJobs,
  fetchVocalizationClusteringJob,
  createVocalizationClusteringJob,
  deleteVocalizationClusteringJob,
  fetchVocClusteringClusters,
  fetchVocClusteringVisualization,
  fetchVocClusteringMetrics,
  fetchVocClusteringStability,
} from "@/api/client";
import type {
  VocalizationTypeCreate,
  VocalizationTypeUpdate,
  VocalizationTypeImportRequest,
  VocClassifierTrainingJobCreate,
  VocClassifierInferenceJobCreate,
} from "@/api/types";

// ---- Vocabulary ----

export function useVocalizationTypes() {
  return useQuery({
    queryKey: ["vocalization", "types"],
    queryFn: fetchVocalizationTypes,
    staleTime: 10_000,
  });
}

export function useCreateVocalizationType() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: VocalizationTypeCreate) => createVocalizationType(body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["vocalization", "types"] });
    },
  });
}

export function useUpdateVocalizationType() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (params: { typeId: string; body: VocalizationTypeUpdate }) =>
      updateVocalizationType(params.typeId, params.body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["vocalization", "types"] });
    },
  });
}

export function useDeleteVocalizationType() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (typeId: string) => deleteVocalizationType(typeId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["vocalization", "types"] });
    },
  });
}

export function useImportVocalizationTypes() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: VocalizationTypeImportRequest) =>
      importVocalizationTypes(body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["vocalization", "types"] });
    },
  });
}

// ---- Models ----

export function useVocClassifierModels() {
  return useQuery({
    queryKey: ["vocalization", "models"],
    queryFn: fetchVocClassifierModels,
    staleTime: 10_000,
  });
}

export function useVocClassifierModel(modelId: string | null) {
  return useQuery({
    queryKey: ["vocalization", "models", modelId],
    queryFn: () => fetchVocClassifierModel(modelId!),
    enabled: modelId !== null,
  });
}

export function useActivateVocClassifierModel() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (modelId: string) => activateVocClassifierModel(modelId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["vocalization", "models"] });
    },
  });
}

// ---- Training Jobs ----

export function useVocClassifierTrainingJobs() {
  return useQuery({
    queryKey: ["vocalization", "training-jobs"],
    queryFn: fetchVocClassifierTrainingJobs,
    refetchInterval: 5000,
  });
}

export function useVocClassifierTrainingJob(jobId: string | null) {
  return useQuery({
    queryKey: ["vocalization", "training-jobs", jobId],
    queryFn: () => fetchVocClassifierTrainingJob(jobId!),
    enabled: jobId !== null,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      if (status === "complete" || status === "failed") return false;
      return 2000;
    },
  });
}

export function useCreateVocClassifierTrainingJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: VocClassifierTrainingJobCreate) =>
      createVocClassifierTrainingJob(body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["vocalization", "training-jobs"] });
    },
  });
}

// ---- Inference Jobs ----

export function useVocClassifierInferenceJobs() {
  return useQuery({
    queryKey: ["vocalization", "inference-jobs"],
    queryFn: fetchVocClassifierInferenceJobs,
    refetchInterval: 5000,
  });
}

export function useVocClassifierInferenceJob(jobId: string | null) {
  return useQuery({
    queryKey: ["vocalization", "inference-jobs", jobId],
    queryFn: () => fetchVocClassifierInferenceJob(jobId!),
    enabled: jobId !== null,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      if (status === "complete" || status === "failed") return false;
      return 2000;
    },
  });
}

export function useCreateVocClassifierInferenceJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: VocClassifierInferenceJobCreate) =>
      createVocClassifierInferenceJob(body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["vocalization", "inference-jobs"] });
    },
  });
}

// ---- Results ----

export function useVocClassifierInferenceResults(
  jobId: string | null,
  params?: {
    offset?: number;
    limit?: number;
    thresholds?: Record<string, number>;
    sort?: string;
  },
) {
  return useQuery({
    queryKey: ["vocalization", "inference-results", jobId, params],
    queryFn: () => fetchVocClassifierInferenceResults(jobId!, params),
    enabled: jobId !== null,
  });
}

// ---- Detection Embeddings ----

export function useEmbeddingStatus(detectionJobId: string | null) {
  return useQuery({
    queryKey: ["embedding-status", detectionJobId],
    queryFn: () => fetchEmbeddingStatus(detectionJobId!),
    enabled: detectionJobId !== null,
  });
}

export function useGenerateEmbeddings() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (jobId: string) => generateEmbeddings(jobId, "full"),
    onSuccess: (_data, jobId) => {
      qc.invalidateQueries({ queryKey: ["embedding-status", jobId] });
      qc.invalidateQueries({ queryKey: ["embedding-generation", jobId] });
    },
  });
}

export function useSyncEmbeddings() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (jobId: string) => generateEmbeddings(jobId, "sync"),
    onSuccess: (_data, jobId) => {
      qc.invalidateQueries({ queryKey: ["embedding-status", jobId] });
      qc.invalidateQueries({ queryKey: ["embedding-generation", jobId] });
    },
  });
}

export function useEmbeddingGenerationStatus(detectionJobId: string | null) {
  return useQuery({
    queryKey: ["embedding-generation", detectionJobId],
    queryFn: () => fetchEmbeddingGenerationStatus(detectionJobId!),
    enabled: detectionJobId !== null,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      if (!status || status === "complete" || status === "failed") return false;
      return 2000;
    },
  });
}

export function useEmbeddingJobs(offset = 0, limit = 50) {
  return useQuery({
    queryKey: ["embedding-jobs", offset, limit],
    queryFn: () => fetchEmbeddingJobs(offset, limit),
    refetchInterval: (query) => {
      const jobs = query.state.data;
      if (!jobs) return false;
      const hasActive = jobs.some((j) => j.status === "queued" || j.status === "running");
      return hasActive ? 3000 : false;
    },
  });
}

// ---- Folder Embedding Set ----

export function useFolderEmbeddingSet(folderPath: string | null) {
  return useQuery({
    queryKey: ["folder-embedding-set", folderPath],
    queryFn: () => fetchFolderEmbeddingSet(folderPath!),
    enabled: folderPath !== null && folderPath.length > 0,
    refetchInterval: (query) => {
      const status = (query.state.data as FolderEmbeddingSetResponse | undefined)?.status;
      if (status === "ready") return false;
      return 3000;
    },
  });
}

// ---- Training Source ----

export function useVocModelTrainingSource(modelId: string | null) {
  return useQuery({
    queryKey: ["vocalization", "training-source", modelId],
    queryFn: () => fetchVocModelTrainingSource(modelId!),
    enabled: modelId !== null,
  });
}

// ---- Training Datasets ----

export function useTrainingDataset(datasetId: string | null) {
  return useQuery({
    queryKey: ["vocalization", "training-dataset", datasetId],
    queryFn: () => fetchTrainingDataset(datasetId!),
    enabled: datasetId !== null,
  });
}

export function useTrainingDatasetRows(
  datasetId: string | null,
  params?: { type?: string; group?: string; source_type?: string; offset?: number; limit?: number },
) {
  return useQuery({
    queryKey: ["vocalization", "training-dataset-rows", datasetId, params],
    queryFn: () => fetchTrainingDatasetRows(datasetId!, params),
    enabled: datasetId !== null,
  });
}

export function useExtendTrainingDataset() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (params: { datasetId: string; body: TrainingDatasetExtendRequest }) =>
      extendTrainingDataset(params.datasetId, params.body),
    onSuccess: (_data, params) => {
      qc.invalidateQueries({
        queryKey: ["vocalization", "training-dataset", params.datasetId],
      });
      qc.invalidateQueries({
        queryKey: ["vocalization", "training-dataset-rows", params.datasetId],
      });
    },
  });
}

export function useCreateTrainingDatasetLabel() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (params: {
      datasetId: string;
      body: { row_index: number; label: string };
    }) => createTrainingDatasetLabel(params.datasetId, params.body),
    onSuccess: (_data, params) => {
      qc.invalidateQueries({
        queryKey: ["vocalization", "training-dataset-rows", params.datasetId],
      });
    },
  });
}

export function useDeleteTrainingDatasetLabel() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (params: { datasetId: string; labelId: string }) =>
      deleteTrainingDatasetLabel(params.datasetId, params.labelId),
    onSuccess: (_data, params) => {
      qc.invalidateQueries({
        queryKey: ["vocalization", "training-dataset-rows", params.datasetId],
      });
    },
  });
}

// ---- Vocalization Clustering ----

export function useClusteringEligibleJobs() {
  return useQuery({
    queryKey: ["vocalization", "clustering-eligible-jobs"],
    queryFn: fetchClusteringEligibleJobs,
    staleTime: 10_000,
  });
}

export function useVocalizationClusteringJobs(pollInterval?: number) {
  return useQuery({
    queryKey: ["vocalization", "clustering-jobs"],
    queryFn: fetchVocalizationClusteringJobs,
    refetchInterval: pollInterval || false,
  });
}

export function useVocalizationClusteringJob(jobId: string | null) {
  return useQuery({
    queryKey: ["vocalization", "clustering-jobs", jobId],
    queryFn: () => fetchVocalizationClusteringJob(jobId!),
    enabled: jobId !== null,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      if (status === "complete" || status === "failed") return false;
      return 2000;
    },
  });
}

export function useCreateVocalizationClusteringJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: VocalizationClusteringJobCreate) =>
      createVocalizationClusteringJob(body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["vocalization", "clustering-jobs"] });
    },
  });
}

export function useDeleteVocalizationClusteringJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (jobId: string) => deleteVocalizationClusteringJob(jobId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["vocalization", "clustering-jobs"] });
    },
  });
}

export function useVocClusteringClusters(jobId: string | null) {
  return useQuery({
    queryKey: ["vocalization", "clustering-clusters", jobId],
    queryFn: () => fetchVocClusteringClusters(jobId!),
    enabled: jobId !== null,
  });
}

export function useVocClusteringVisualization(jobId: string | null) {
  return useQuery({
    queryKey: ["vocalization", "clustering-visualization", jobId],
    queryFn: () => fetchVocClusteringVisualization(jobId!),
    enabled: jobId !== null,
  });
}

export function useVocClusteringMetrics(jobId: string | null) {
  return useQuery({
    queryKey: ["vocalization", "clustering-metrics", jobId],
    queryFn: () => fetchVocClusteringMetrics(jobId!),
    enabled: jobId !== null,
  });
}

export function useVocClusteringStability(jobId: string | null) {
  return useQuery({
    queryKey: ["vocalization", "clustering-stability", jobId],
    queryFn: () => fetchVocClusteringStability(jobId!),
    enabled: jobId !== null,
  });
}
