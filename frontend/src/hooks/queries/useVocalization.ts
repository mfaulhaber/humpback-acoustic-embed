import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
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
  fetchVocModelTrainingSource,
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
  params?: { offset?: number; limit?: number; thresholds?: Record<string, number> },
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
    mutationFn: (jobId: string) => generateEmbeddings(jobId),
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

// ---- Training Source ----

export function useVocModelTrainingSource(modelId: string | null) {
  return useQuery({
    queryKey: ["vocalization", "training-source", modelId],
    queryFn: () => fetchVocModelTrainingSource(modelId!),
    enabled: modelId !== null,
  });
}
