import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  fetchTrainingJobs,
  createTrainingJob,
  fetchClassifierModels,
  deleteClassifierModel,
  fetchDetectionJobs,
  createDetectionJob,
  browseDirectories,
  bulkDeleteTrainingJobs,
  bulkDeleteClassifierModels,
  bulkDeleteDetectionJobs,
  fetchDetectionContent,
} from "@/api/client";
import type { ClassifierTrainingJobCreate, DetectionJobCreate } from "@/api/types";

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

export function useDetectionJobs(refetchInterval?: number) {
  return useQuery({
    queryKey: ["detectionJobs"],
    queryFn: fetchDetectionJobs,
    refetchInterval,
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

export function useCreateDetectionJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: DetectionJobCreate) => createDetectionJob(body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["detectionJobs"] });
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

export function useDetectionContent(jobId: string | null) {
  return useQuery({
    queryKey: ["detectionContent", jobId],
    queryFn: () => fetchDetectionContent(jobId!),
    enabled: jobId !== null,
  });
}
