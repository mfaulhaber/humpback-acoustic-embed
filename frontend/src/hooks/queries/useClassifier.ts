import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  fetchTrainingJobs,
  createTrainingJob,
  fetchClassifierModels,
  deleteClassifierModel,
  fetchDetectionJobs,
  createDetectionJob,
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
