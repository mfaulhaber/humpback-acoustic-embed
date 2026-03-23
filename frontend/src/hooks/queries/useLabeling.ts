import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  fetchVocalizationLabels,
  createVocalizationLabel,
  deleteVocalizationLabel,
  fetchLabelVocabulary,
  fetchLabelingSummary,
  fetchDetectionNeighbors,
  createVocalizationTrainingJob,
  fetchVocalizationModels,
  predictVocalizationLabels,
  fetchAnnotations,
  createAnnotation,
  updateAnnotation,
  deleteAnnotation,
  startActiveLearningCycle,
  fetchUncertaintyQueue,
  fetchConvergenceMetrics,
} from "@/api/client";
import type { VocalizationTrainingJobCreate } from "@/api/types";

export function useLabelingSummary(jobId: string | null) {
  return useQuery({
    queryKey: ["labeling", "summary", jobId],
    queryFn: () => fetchLabelingSummary(jobId!),
    enabled: jobId !== null,
    refetchInterval: 5000,
  });
}

export function useVocalizationLabels(
  detectionJobId: string | null,
  rowId: string | null,
) {
  return useQuery({
    queryKey: ["labeling", "vocalization-labels", detectionJobId, rowId],
    queryFn: () => fetchVocalizationLabels(detectionJobId!, rowId!),
    enabled: detectionJobId !== null && rowId !== null,
  });
}

export function useAddVocalizationLabel() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (params: {
      detectionJobId: string;
      rowId: string;
      label: string;
      confidence?: number;
      source?: string;
    }) =>
      createVocalizationLabel(params.detectionJobId, params.rowId, {
        label: params.label,
        confidence: params.confidence,
        source: params.source ?? "manual",
      }),
    onSuccess: (_data, vars) => {
      qc.invalidateQueries({
        queryKey: [
          "labeling",
          "vocalization-labels",
          vars.detectionJobId,
          vars.rowId,
        ],
      });
      qc.invalidateQueries({
        queryKey: ["labeling", "summary", vars.detectionJobId],
      });
      qc.invalidateQueries({ queryKey: ["labeling", "label-vocabulary"] });
    },
  });
}

export function useDeleteVocalizationLabel() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (params: {
      labelId: string;
      detectionJobId: string;
      rowId: string;
    }) => deleteVocalizationLabel(params.labelId),
    onSuccess: (_data, vars) => {
      qc.invalidateQueries({
        queryKey: [
          "labeling",
          "vocalization-labels",
          vars.detectionJobId,
          vars.rowId,
        ],
      });
      qc.invalidateQueries({
        queryKey: ["labeling", "summary", vars.detectionJobId],
      });
    },
  });
}

export function useLabelVocabulary() {
  return useQuery({
    queryKey: ["labeling", "label-vocabulary"],
    queryFn: fetchLabelVocabulary,
    staleTime: 30_000,
  });
}

export function useDetectionNeighbors(
  detectionJobId: string | null,
  filename: string | null,
  startSec: number | null,
  endSec: number | null,
  embeddingSetIds?: string[],
) {
  return useQuery({
    queryKey: [
      "labeling",
      "neighbors",
      detectionJobId,
      filename,
      startSec,
      endSec,
      embeddingSetIds,
    ],
    queryFn: () =>
      fetchDetectionNeighbors(detectionJobId!, {
        filename: filename!,
        start_sec: startSec!,
        end_sec: endSec!,
        embedding_set_ids: embeddingSetIds,
      }),
    enabled:
      detectionJobId !== null &&
      filename !== null &&
      startSec !== null &&
      endSec !== null,
    staleTime: Infinity,
  });
}

// ---- Vocalization Classifier ----

export function useVocalizationModels() {
  return useQuery({
    queryKey: ["labeling", "vocalization-models"],
    queryFn: fetchVocalizationModels,
    staleTime: 10_000,
  });
}

export function useCreateVocalizationTrainingJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: VocalizationTrainingJobCreate) =>
      createVocalizationTrainingJob(body),
    onSuccess: () => {
      qc.invalidateQueries({
        queryKey: ["labeling", "vocalization-models"],
      });
    },
  });
}

export function usePredictVocalizationLabels(
  detectionJobId: string | null,
  vocalizationModelId: string | null,
) {
  return useQuery({
    queryKey: [
      "labeling",
      "predictions",
      detectionJobId,
      vocalizationModelId,
    ],
    queryFn: () =>
      predictVocalizationLabels(detectionJobId!, vocalizationModelId!),
    enabled: detectionJobId !== null && vocalizationModelId !== null,
    staleTime: 60_000,
  });
}

// ---- Annotations ----

export function useAnnotations(
  detectionJobId: string | null,
  rowId: string | null,
) {
  return useQuery({
    queryKey: ["labeling", "annotations", detectionJobId, rowId],
    queryFn: () => fetchAnnotations(detectionJobId!, rowId!),
    enabled: detectionJobId !== null && rowId !== null,
  });
}

export function useCreateAnnotation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (params: {
      detectionJobId: string;
      rowId: string;
      start_offset_sec: number;
      end_offset_sec: number;
      label: string;
      notes?: string;
    }) =>
      createAnnotation(params.detectionJobId, params.rowId, {
        start_offset_sec: params.start_offset_sec,
        end_offset_sec: params.end_offset_sec,
        label: params.label,
        notes: params.notes,
      }),
    onSuccess: (_data, vars) => {
      qc.invalidateQueries({
        queryKey: [
          "labeling",
          "annotations",
          vars.detectionJobId,
          vars.rowId,
        ],
      });
    },
  });
}

export function useUpdateAnnotation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (params: {
      annotationId: string;
      detectionJobId: string;
      rowId: string;
      start_offset_sec?: number;
      end_offset_sec?: number;
      label?: string;
    }) =>
      updateAnnotation(params.annotationId, {
        start_offset_sec: params.start_offset_sec,
        end_offset_sec: params.end_offset_sec,
        label: params.label,
      }),
    onSuccess: (_data, vars) => {
      qc.invalidateQueries({
        queryKey: [
          "labeling",
          "annotations",
          vars.detectionJobId,
          vars.rowId,
        ],
      });
    },
  });
}

export function useDeleteAnnotation() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (params: {
      annotationId: string;
      detectionJobId: string;
      rowId: string;
    }) => deleteAnnotation(params.annotationId),
    onSuccess: (_data, vars) => {
      qc.invalidateQueries({
        queryKey: [
          "labeling",
          "annotations",
          vars.detectionJobId,
          vars.rowId,
        ],
      });
    },
  });
}

// ---- Active Learning ----

export function useStartActiveLearningCycle() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: {
      vocalization_model_id: string;
      detection_job_ids: string[];
      name: string;
    }) => startActiveLearningCycle(body),
    onSuccess: () => {
      qc.invalidateQueries({
        queryKey: ["labeling", "vocalization-models"],
      });
    },
  });
}

export function useUncertaintyQueue(
  detectionJobId: string | null,
  vocalizationModelId: string | null,
) {
  return useQuery({
    queryKey: [
      "labeling",
      "uncertainty-queue",
      detectionJobId,
      vocalizationModelId,
    ],
    queryFn: () =>
      fetchUncertaintyQueue(detectionJobId!, vocalizationModelId!),
    enabled: detectionJobId !== null && vocalizationModelId !== null,
    staleTime: 60_000,
  });
}

export function useConvergenceMetrics(vocalizationModelId: string | null) {
  return useQuery({
    queryKey: ["labeling", "convergence", vocalizationModelId],
    queryFn: () => fetchConvergenceMetrics(vocalizationModelId!),
    enabled: vocalizationModelId !== null,
    staleTime: 30_000,
  });
}
