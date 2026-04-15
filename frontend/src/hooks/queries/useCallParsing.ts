import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  fetchRegionDetectionJobs,
  createRegionDetectionJob,
  deleteRegionDetectionJob,
  fetchRegionJobRegions,
  fetchSegmentationJobs,
  createSegmentationJob,
  deleteSegmentationJob,
  fetchSegmentationJobEvents,
  fetchBoundaryCorrections,
  saveBoundaryCorrections,
  clearBoundaryCorrections,
  fetchSegmentationModels,
  deleteSegmentationModel,
  fetchSegmentationJobsWithCorrectionCounts,
  fetchSegmentationTrainingDatasets,
  createDatasetFromCorrections,
  createSegmentationTrainingJob,
  quickRetrain,
  fetchClassificationJobs,
  createClassificationJob,
  deleteClassificationJob,
  fetchTypedEvents,
  fetchTypeCorrections,
  saveTypeCorrections,
  clearTypeCorrections,
  fetchClassificationJobsWithCorrectionCounts,
  fetchClassifierTrainingJobs,
  createClassifierTrainingJob,
  deleteClassifierTrainingJob,
  fetchEventClassifierModels,
  deleteEventClassifierModel,
} from "@/api/client";
import type {
  CreateRegionJobRequest,
  CreateSegmentationJobRequest,
  CreateDatasetFromCorrectionsRequest,
  CreateSegmentationTrainingJobRequest,
  QuickRetrainRequest,
  BoundaryCorrectionRequest,
  ModelConfig,
  CreateEventClassificationJobRequest,
  TypeCorrectionItem,
  CreateEventClassifierTrainingJobRequest,
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

export function useRegionJobRegions(jobId: string | null) {
  return useQuery({
    queryKey: ["region-job-regions", jobId],
    queryFn: () => fetchRegionJobRegions(jobId!),
    enabled: !!jobId,
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

// ---- Boundary Corrections ----

const CORRECTIONS_KEY = "boundary-corrections";

export function useBoundaryCorrections(jobId: string | null) {
  return useQuery({
    queryKey: [CORRECTIONS_KEY, jobId],
    queryFn: () => fetchBoundaryCorrections(jobId!),
    enabled: !!jobId,
  });
}

export function useSaveBoundaryCorrections() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({
      jobId,
      body,
    }: {
      jobId: string;
      body: BoundaryCorrectionRequest;
    }) => saveBoundaryCorrections(jobId, body),
    onSuccess: (_data, variables) => {
      qc.invalidateQueries({
        queryKey: [CORRECTIONS_KEY, variables.jobId],
      });
    },
  });
}

export function useClearBoundaryCorrections() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (jobId: string) => clearBoundaryCorrections(jobId),
    onSuccess: (_data, jobId) => {
      qc.invalidateQueries({ queryKey: [CORRECTIONS_KEY, jobId] });
    },
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

// ---- Multi-job segmentation training ----

const SEG_CORRECTION_COUNTS_KEY = ["segmentation-jobs-correction-counts"];
const SEG_TRAINING_DATASETS_KEY = ["segmentation-training-datasets"];
const SEG_TRAINING_JOBS_KEY = ["segmentation-training-jobs"];

export function useSegmentationJobsWithCorrectionCounts() {
  return useQuery({
    queryKey: SEG_CORRECTION_COUNTS_KEY,
    queryFn: fetchSegmentationJobsWithCorrectionCounts,
  });
}

export function useCreateSegmentationTrainingDataset() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: CreateDatasetFromCorrectionsRequest) =>
      createDatasetFromCorrections(body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: SEG_TRAINING_DATASETS_KEY });
    },
  });
}

export function useSegmentationTrainingDatasets() {
  return useQuery({
    queryKey: SEG_TRAINING_DATASETS_KEY,
    queryFn: fetchSegmentationTrainingDatasets,
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

export function useQuickRetrain() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: QuickRetrainRequest) => quickRetrain(body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: SEG_TRAINING_DATASETS_KEY });
      qc.invalidateQueries({ queryKey: SEG_TRAINING_JOBS_KEY });
      qc.invalidateQueries({ queryKey: SEG_MODELS_KEY });
    },
  });
}

// ---- Event Classification Jobs (Pass 3) ----

const CLASSIFY_JOBS_KEY = ["classification-jobs"];
const TYPED_EVENTS_KEY = "typed-events";
const TYPE_CORRECTIONS_KEY = "type-corrections";
const CLASSIFY_CORRECTION_COUNTS_KEY = ["classification-jobs-correction-counts"];
const CLASSIFIER_TRAINING_JOBS_KEY = ["classifier-training-jobs"];
const CLASSIFIER_MODELS_KEY = ["classifier-models"];

export function useClassificationJobs(refetchInterval = 3000) {
  return useQuery({
    queryKey: CLASSIFY_JOBS_KEY,
    queryFn: fetchClassificationJobs,
    refetchInterval,
  });
}

export function useCreateClassificationJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: CreateEventClassificationJobRequest) =>
      createClassificationJob(body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: CLASSIFY_JOBS_KEY });
    },
  });
}

export function useDeleteClassificationJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (jobId: string) => deleteClassificationJob(jobId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: CLASSIFY_JOBS_KEY });
    },
  });
}

export function useTypedEvents(jobId: string | null) {
  return useQuery({
    queryKey: [TYPED_EVENTS_KEY, jobId],
    queryFn: () => fetchTypedEvents(jobId!),
    enabled: !!jobId,
  });
}

export function useTypeCorrections(jobId: string | null) {
  return useQuery({
    queryKey: [TYPE_CORRECTIONS_KEY, jobId],
    queryFn: () => fetchTypeCorrections(jobId!),
    enabled: !!jobId,
  });
}

export function useUpsertTypeCorrections() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({
      jobId,
      corrections,
    }: {
      jobId: string;
      corrections: TypeCorrectionItem[];
    }) => saveTypeCorrections(jobId, corrections),
    onSuccess: (_data, variables) => {
      qc.invalidateQueries({
        queryKey: [TYPE_CORRECTIONS_KEY, variables.jobId],
      });
    },
  });
}

export function useClearTypeCorrections() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (jobId: string) => clearTypeCorrections(jobId),
    onSuccess: (_data, jobId) => {
      qc.invalidateQueries({ queryKey: [TYPE_CORRECTIONS_KEY, jobId] });
    },
  });
}

export function useClassificationJobsWithCorrectionCounts() {
  return useQuery({
    queryKey: CLASSIFY_CORRECTION_COUNTS_KEY,
    queryFn: fetchClassificationJobsWithCorrectionCounts,
  });
}

export function useClassifierTrainingJobs(refetchInterval?: number) {
  return useQuery({
    queryKey: CLASSIFIER_TRAINING_JOBS_KEY,
    queryFn: fetchClassifierTrainingJobs,
    refetchInterval,
  });
}

export function useCreateClassifierTrainingJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: CreateEventClassifierTrainingJobRequest) =>
      createClassifierTrainingJob(body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: CLASSIFIER_TRAINING_JOBS_KEY });
      qc.invalidateQueries({ queryKey: CLASSIFIER_MODELS_KEY });
    },
  });
}

export function useDeleteClassifierTrainingJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (jobId: string) => deleteClassifierTrainingJob(jobId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: CLASSIFIER_TRAINING_JOBS_KEY });
    },
  });
}

export function useEventClassifierModels() {
  return useQuery({
    queryKey: CLASSIFIER_MODELS_KEY,
    queryFn: fetchEventClassifierModels,
  });
}

export function useDeleteEventClassifierModel() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (modelId: string) => deleteEventClassifierModel(modelId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: CLASSIFIER_MODELS_KEY });
    },
  });
}

