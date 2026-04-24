import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  fetchRegionDetectionJobs,
  createRegionDetectionJob,
  deleteRegionDetectionJob,
  fetchRegionJobRegions,
  fetchRegionJobConfidence,
  fetchRegionCorrections,
  saveRegionCorrections,
  fetchSegmentationJobs,
  createSegmentationJob,
  deleteSegmentationJob,
  fetchSegmentationJobEvents,
  fetchEventBoundaryCorrections,
  upsertEventBoundaryCorrections,
  clearEventBoundaryCorrections,
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
  fetchVocalizationCorrections,
  upsertVocalizationCorrections,
  clearVocalizationCorrections,
  fetchClassificationJobsWithCorrectionCounts,
  fetchClassifierTrainingJobs,
  createClassifierTrainingJob,
  deleteClassifierTrainingJob,
  fetchEventClassifierModels,
  deleteEventClassifierModel,
  createWindowClassificationJob,
  fetchWindowClassificationJobs,
  deleteWindowClassificationJob,
  fetchWindowScores,
} from "@/api/client";
import type {
  CreateRegionJobRequest,
  RegionCorrection,
  CreateSegmentationJobRequest,
  CreateDatasetFromCorrectionsRequest,
  CreateSegmentationTrainingJobRequest,
  QuickRetrainRequest,
  EventBoundaryCorrectionItem,
  ModelConfig,
  CreateEventClassificationJobRequest,
  VocalizationCorrectionItem,
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

export function useRegionJobConfidence(jobId: string | null) {
  return useQuery({
    queryKey: ["region-job-confidence", jobId],
    queryFn: () => fetchRegionJobConfidence(jobId!),
    enabled: !!jobId,
    staleTime: Infinity,
  });
}

// ---- Region Boundary Corrections ----

const regionCorrectionsKey = (jobId: string) => [
  "region-corrections",
  jobId,
];

export function useRegionCorrections(jobId: string | null) {
  return useQuery({
    queryKey: regionCorrectionsKey(jobId ?? ""),
    queryFn: () => fetchRegionCorrections(jobId!),
    enabled: !!jobId,
  });
}

export function useSaveRegionCorrections() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({
      jobId,
      corrections,
    }: {
      jobId: string;
      corrections: RegionCorrection[];
    }) => saveRegionCorrections(jobId, corrections),
    onSuccess: (_data, { jobId }) => {
      qc.invalidateQueries({ queryKey: regionCorrectionsKey(jobId) });
      qc.invalidateQueries({ queryKey: ["region-job-regions", jobId] });
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

// ---- Event Boundary Corrections ----

const EVENT_BOUNDARY_CORRECTIONS_KEY = "event-boundary-corrections";

export function useEventBoundaryCorrections(
  regionDetectionJobId: string | null,
) {
  return useQuery({
    queryKey: [EVENT_BOUNDARY_CORRECTIONS_KEY, regionDetectionJobId],
    queryFn: () => fetchEventBoundaryCorrections(regionDetectionJobId!),
    enabled: !!regionDetectionJobId,
  });
}

export function useUpsertEventBoundaryCorrections() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({
      regionDetectionJobId,
      corrections,
    }: {
      regionDetectionJobId: string;
      corrections: EventBoundaryCorrectionItem[];
    }) => upsertEventBoundaryCorrections(regionDetectionJobId, corrections),
    onSuccess: (_data, variables) => {
      qc.invalidateQueries({
        queryKey: [
          EVENT_BOUNDARY_CORRECTIONS_KEY,
          variables.regionDetectionJobId,
        ],
      });
    },
  });
}

export function useClearEventBoundaryCorrections() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (regionDetectionJobId: string) =>
      clearEventBoundaryCorrections(regionDetectionJobId),
    onSuccess: (_data, regionDetectionJobId) => {
      qc.invalidateQueries({
        queryKey: [EVENT_BOUNDARY_CORRECTIONS_KEY, regionDetectionJobId],
      });
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
const VOCALIZATION_CORRECTIONS_KEY = "vocalizationCorrections";
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

export function useVocalizationCorrections(
  regionDetectionJobId: string | null,
) {
  return useQuery({
    queryKey: [VOCALIZATION_CORRECTIONS_KEY, regionDetectionJobId],
    queryFn: () => fetchVocalizationCorrections(regionDetectionJobId!),
    enabled: !!regionDetectionJobId,
  });
}

export function useUpsertVocalizationCorrections() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({
      regionDetectionJobId,
      corrections,
    }: {
      regionDetectionJobId: string;
      corrections: VocalizationCorrectionItem[];
    }) => upsertVocalizationCorrections(regionDetectionJobId, corrections),
    onSuccess: (_data, variables) => {
      qc.invalidateQueries({
        queryKey: [
          VOCALIZATION_CORRECTIONS_KEY,
          variables.regionDetectionJobId,
        ],
      });
    },
  });
}

export function useClearVocalizationCorrections() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (regionDetectionJobId: string) =>
      clearVocalizationCorrections(regionDetectionJobId),
    onSuccess: (_data, regionDetectionJobId) => {
      qc.invalidateQueries({
        queryKey: [VOCALIZATION_CORRECTIONS_KEY, regionDetectionJobId],
      });
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

// ---- Window Classification Sidecar ----

const WINDOW_CLASSIFICATION_JOBS_KEY = ["windowClassificationJobs"];

export function useWindowClassificationJobs(refetchInterval?: number) {
  return useQuery({
    queryKey: WINDOW_CLASSIFICATION_JOBS_KEY,
    queryFn: fetchWindowClassificationJobs,
    refetchInterval,
  });
}

export function useCreateWindowClassificationJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: {
      region_detection_job_id: string;
      vocalization_model_id: string;
    }) => createWindowClassificationJob(body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: WINDOW_CLASSIFICATION_JOBS_KEY });
    },
  });
}

export function useDeleteWindowClassificationJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (jobId: string) => deleteWindowClassificationJob(jobId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: WINDOW_CLASSIFICATION_JOBS_KEY });
    },
  });
}

export function useWindowScores(
  jobId: string | undefined,
  params?: { region_id?: string; min_score?: number; type_name?: string },
) {
  return useQuery({
    queryKey: ["windowScores", jobId, params],
    queryFn: () => fetchWindowScores(jobId!, params),
    enabled: !!jobId,
  });
}


