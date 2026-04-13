import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  fetchRegionDetectionJobs,
  createRegionDetectionJob,
  deleteRegionDetectionJob,
  fetchSegmentationJobs,
  createSegmentationJob,
  deleteSegmentationJob,
  fetchSegmentationJobEvents,
  fetchSegmentationModels,
  deleteSegmentationModel,
  fetchSegmentationTrainingJobs,
  createSegmentationTrainingJob,
  deleteSegmentationTrainingJob,
  fetchSegmentationTrainingDatasets,
} from "@/api/client";
import type {
  CreateRegionJobRequest,
  CreateSegmentationJobRequest,
  CreateSegmentationTrainingJobRequest,
  ModelConfig,
} from "@/api/types";

const REGION_JOBS_KEY = ["regionDetectionJobs"];

export function useRegionDetectionJobs(refetchInterval?: number) {
  return useQuery({
    queryKey: REGION_JOBS_KEY,
    queryFn: fetchRegionDetectionJobs,
    refetchInterval,
  });
}

export function useCreateRegionJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: CreateRegionJobRequest) =>
      createRegionDetectionJob(body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: REGION_JOBS_KEY });
    },
  });
}

export function useDeleteRegionJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (jobId: string) => deleteRegionDetectionJob(jobId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: REGION_JOBS_KEY });
    },
  });
}

export function resolveModelConfigId(
  modelVersion: string,
  modelConfigs: ModelConfig[],
): string | null {
  const match = modelConfigs.find((mc) => mc.name === modelVersion);
  return match?.id ?? null;
}

// ---- Event Segmentation Jobs ----

const SEG_JOBS_KEY = ["segmentation-jobs"];

export function useSegmentationJobs(refetchInterval = 3000) {
  return useQuery({
    queryKey: SEG_JOBS_KEY,
    queryFn: fetchSegmentationJobs,
    refetchInterval,
  });
}

export function useCreateSegmentationJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: CreateSegmentationJobRequest) =>
      createSegmentationJob(body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: SEG_JOBS_KEY });
    },
  });
}

export function useDeleteSegmentationJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (jobId: string) => deleteSegmentationJob(jobId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: SEG_JOBS_KEY });
    },
  });
}

export function useSegmentationJobEvents(jobId: string | null) {
  return useQuery({
    queryKey: ["segmentation-job-events", jobId],
    queryFn: () => fetchSegmentationJobEvents(jobId!),
    enabled: !!jobId,
  });
}

// ---- Segmentation Models ----

const SEG_MODELS_KEY = ["segmentation-models"];

export function useSegmentationModels() {
  return useQuery({
    queryKey: SEG_MODELS_KEY,
    queryFn: fetchSegmentationModels,
  });
}

export function useDeleteSegmentationModel() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (modelId: string) => deleteSegmentationModel(modelId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: SEG_MODELS_KEY });
    },
  });
}

// ---- Segmentation Training Jobs ----

const SEG_TRAINING_JOBS_KEY = ["segmentation-training-jobs"];

export function useSegmentationTrainingJobs(refetchInterval = 3000) {
  return useQuery({
    queryKey: SEG_TRAINING_JOBS_KEY,
    queryFn: fetchSegmentationTrainingJobs,
    refetchInterval,
  });
}

export function useCreateSegmentationTrainingJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: CreateSegmentationTrainingJobRequest) =>
      createSegmentationTrainingJob(body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: SEG_TRAINING_JOBS_KEY });
      qc.invalidateQueries({ queryKey: SEG_MODELS_KEY });
    },
  });
}

export function useDeleteSegmentationTrainingJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (jobId: string) => deleteSegmentationTrainingJob(jobId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: SEG_TRAINING_JOBS_KEY });
    },
  });
}

// ---- Segmentation Training Datasets ----

export function useSegmentationTrainingDatasets() {
  return useQuery({
    queryKey: ["segmentation-training-datasets"],
    queryFn: fetchSegmentationTrainingDatasets,
  });
}
