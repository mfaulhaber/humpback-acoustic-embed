import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  fetchTrainingJobs,
  createTrainingJob,
  fetchClassifierModels,
  deleteClassifierModel,
  browseDirectories,
  bulkDeleteTrainingJobs,
  bulkDeleteClassifierModels,
  bulkDeleteDetectionJobs,
  fetchDetectionContent,
  saveDetectionLabels,
  fetchExtractionSettings,
  extractLabeledSamples,
  fetchTrainingDataSummary,
  fetchHydrophones,
  fetchHydrophoneDetectionJobs,
  createHydrophoneDetectionJob,
  cancelHydrophoneDetectionJob,
  pauseHydrophoneDetectionJob,
  resumeHydrophoneDetectionJob,
  fetchRetrainInfo,
  createRetrainWorkflow,
  fetchRetrainWorkflows,
} from "@/api/client";
import type {
  ClassifierTrainingJobCreate,
  DetectionLabelRow,
  HydrophoneDetectionJobCreate,
  RetrainWorkflowCreate,
} from "@/api/types";

export function useTrainingJobs(refetchInterval?: number) {
  return useQuery({
    queryKey: ["trainingJobs"],
    queryFn: fetchTrainingJobs,
    refetchInterval,
  });
}

export function useClassifierModels() {
  return useQuery({
    queryKey: ["classifierModels"],
    queryFn: fetchClassifierModels,
  });
}

export function useCreateTrainingJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: ClassifierTrainingJobCreate) => createTrainingJob(body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["trainingJobs"] });
    },
  });
}

export function useDeleteClassifierModel() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (modelId: string) => deleteClassifierModel(modelId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["classifierModels"] });
    },
  });
}

export function useBrowseDirectories(root: string | null) {
  return useQuery({
    queryKey: ["browseDirectories", root],
    queryFn: () => browseDirectories(root!),
    enabled: root !== null,
  });
}

export function useBulkDeleteTrainingJobs() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (ids: string[]) => bulkDeleteTrainingJobs(ids),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["trainingJobs"] });
      qc.invalidateQueries({ queryKey: ["classifierModels"] });
    },
  });
}

export function useBulkDeleteClassifierModels() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (ids: string[]) => bulkDeleteClassifierModels(ids),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["classifierModels"] });
    },
  });
}

export function useBulkDeleteDetectionJobs() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (ids: string[]) => bulkDeleteDetectionJobs(ids),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["detectionJobs"] });
    },
  });
}

export function useDetectionContent(jobId: string | null, refetchInterval?: number) {
  return useQuery({
    queryKey: ["detectionContent", jobId],
    queryFn: () => fetchDetectionContent(jobId!),
    enabled: jobId !== null,
    refetchInterval,
  });
}

export function useExtractionSettings() {
  return useQuery({
    queryKey: ["extractionSettings"],
    queryFn: fetchExtractionSettings,
  });
}

export function useExtractLabeledSamples() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({
      jobIds,
      positiveOutputPath,
      negativeOutputPath,
    }: {
      jobIds: string[];
      positiveOutputPath?: string;
      negativeOutputPath?: string;
    }) => extractLabeledSamples(jobIds, positiveOutputPath, negativeOutputPath),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["detectionJobs"] });
    },
  });
}

export function useSaveDetectionLabels() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ jobId, rows }: { jobId: string; rows: DetectionLabelRow[] }) =>
      saveDetectionLabels(jobId, rows),
    onSuccess: (_data, variables) => {
      qc.invalidateQueries({ queryKey: ["detectionContent", variables.jobId] });
      qc.invalidateQueries({ queryKey: ["hydrophoneDetectionJobs"] });
    },
  });
}

export function useTrainingDataSummary(modelId: string | null) {
  return useQuery({
    queryKey: ["trainingDataSummary", modelId],
    queryFn: () => fetchTrainingDataSummary(modelId!),
    enabled: modelId !== null,
  });
}

// ---- Hydrophone Detection ----

export function useHydrophones() {
  return useQuery({
    queryKey: ["hydrophones"],
    queryFn: fetchHydrophones,
  });
}

export function useHydrophoneDetectionJobs(refetchInterval?: number) {
  return useQuery({
    queryKey: ["hydrophoneDetectionJobs"],
    queryFn: fetchHydrophoneDetectionJobs,
    refetchInterval,
  });
}

export function useCreateHydrophoneDetectionJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: HydrophoneDetectionJobCreate) =>
      createHydrophoneDetectionJob(body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["hydrophoneDetectionJobs"] });
    },
  });
}

export function useCancelHydrophoneDetectionJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (jobId: string) => cancelHydrophoneDetectionJob(jobId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["hydrophoneDetectionJobs"] });
    },
  });
}

export function usePauseHydrophoneDetectionJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (jobId: string) => pauseHydrophoneDetectionJob(jobId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["hydrophoneDetectionJobs"] });
    },
  });
}

export function useResumeHydrophoneDetectionJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (jobId: string) => resumeHydrophoneDetectionJob(jobId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["hydrophoneDetectionJobs"] });
    },
  });
}

// ---- Retrain Workflows ----

export function useRetrainInfo(modelId: string | null) {
  return useQuery({
    queryKey: ["retrainInfo", modelId],
    queryFn: () => fetchRetrainInfo(modelId!),
    enabled: modelId !== null,
  });
}

export function useRetrainWorkflows(refetchInterval?: number) {
  return useQuery({
    queryKey: ["retrainWorkflows"],
    queryFn: fetchRetrainWorkflows,
    refetchInterval,
  });
}

export function useCreateRetrainWorkflow() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: RetrainWorkflowCreate) => createRetrainWorkflow(body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["retrainWorkflows"] });
      qc.invalidateQueries({ queryKey: ["classifierModels"] });
    },
  });
}
