import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  createAutoresearchCandidateTrainingJob,
  fetchTrainingJobs,
  createTrainingJob,
  fetchAutoresearchCandidate,
  fetchAutoresearchCandidates,
  fetchClassifierModels,
  importAutoresearchCandidate,
  deleteClassifierModel,
  browseDirectories,
  bulkDeleteTrainingJobs,
  bulkDeleteClassifierModels,
  bulkDeleteDetectionJobs,
  fetchDetectionContent,
  saveDetectionLabels,
  saveDetectionRowState,
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
import { toast } from "@/components/ui/use-toast";
import type {
  AutoresearchCandidateImport,
  AutoresearchCandidateTrainingJobCreate,
  ClassifierTrainingJobCreate,
  DetectionLabelRow,
  DetectionRowStateUpdate,
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

export function useAutoresearchCandidates(refetchInterval?: number) {
  return useQuery({
    queryKey: ["autoresearchCandidates"],
    queryFn: fetchAutoresearchCandidates,
    refetchInterval,
  });
}

export function useAutoresearchCandidate(candidateId: string | null) {
  return useQuery({
    queryKey: ["autoresearchCandidate", candidateId],
    queryFn: () => fetchAutoresearchCandidate(candidateId!),
    enabled: candidateId !== null,
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

export function useImportAutoresearchCandidate() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: AutoresearchCandidateImport) =>
      importAutoresearchCandidate(body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["autoresearchCandidates"] });
      qc.invalidateQueries({ queryKey: ["autoresearchCandidate"] });
      toast({
        title: "Candidate imported",
        description: "The autoresearch candidate is ready for review.",
      });
    },
    onError: (err: Error) => {
      toast({
        title: "Import failed",
        description: err.message,
        variant: "destructive",
        duration: 8000,
      });
    },
  });
}

export function useCreateAutoresearchCandidateTrainingJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({
      candidateId,
      body,
    }: {
      candidateId: string;
      body: AutoresearchCandidateTrainingJobCreate;
    }) => createAutoresearchCandidateTrainingJob(candidateId, body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["autoresearchCandidates"] });
      qc.invalidateQueries({ queryKey: ["autoresearchCandidate"] });
      qc.invalidateQueries({ queryKey: ["trainingJobs"] });
      toast({
        title: "Candidate training queued",
        description: "The candidate-backed training job has been created.",
      });
    },
    onError: (err: Error) => {
      toast({
        title: "Promotion failed",
        description: err.message,
        variant: "destructive",
        duration: 8000,
      });
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
    mutationFn: (ids: string[]) =>
      bulkDeleteDetectionJobs(ids) as Promise<{
        status: string;
        count: number;
        blocked?: { job_id: string; detail: string }[];
      }>,
    onSuccess: (data) => {
      qc.invalidateQueries({ queryKey: ["detectionJobs"] });
      if (data.blocked && data.blocked.length > 0) {
        const msgs = data.blocked.map((b) => b.detail);
        toast({
          title: "Some detection jobs could not be deleted",
          description: msgs.join("\n"),
          variant: "destructive",
          duration: 8000,
        });
      }
    },
    onError: (err: Error) => {
      toast({
        title: "Cannot delete detection job",
        description: err.message,
        variant: "destructive",
        duration: 8000,
      });
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
      positiveSelectionSmoothingWindow,
      positiveSelectionMinScore,
      positiveSelectionExtendMinScore,
    }: {
      jobIds: string[];
      positiveOutputPath?: string;
      negativeOutputPath?: string;
      positiveSelectionSmoothingWindow?: number;
      positiveSelectionMinScore?: number;
      positiveSelectionExtendMinScore?: number;
    }) =>
      extractLabeledSamples(jobIds, positiveOutputPath, negativeOutputPath, {
        positiveSelectionSmoothingWindow,
        positiveSelectionMinScore,
        positiveSelectionExtendMinScore,
      }),
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

export function useSaveDetectionRowState() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({
      jobId,
      body,
    }: {
      jobId: string;
      body: DetectionRowStateUpdate;
    }) => saveDetectionRowState(jobId, body),
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
