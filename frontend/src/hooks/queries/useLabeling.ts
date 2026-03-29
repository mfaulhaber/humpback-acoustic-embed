import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  fetchVocalizationLabels,
  createVocalizationLabel,
  deleteVocalizationLabel,
  fetchLabelVocabulary,
  fetchLabelingSummary,
  fetchTrainingSummary,
  fetchDetectionNeighbors,
} from "@/api/client";

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
  startUtc: number | null,
  endUtc: number | null,
) {
  return useQuery({
    queryKey: ["labeling", "vocalization-labels", detectionJobId, startUtc, endUtc],
    queryFn: () => fetchVocalizationLabels(detectionJobId!, startUtc!, endUtc!),
    enabled: detectionJobId !== null && startUtc !== null && endUtc !== null,
  });
}

export function useAddVocalizationLabel() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (params: {
      detectionJobId: string;
      startUtc: number;
      endUtc: number;
      label: string;
      confidence?: number;
      source?: string;
    }) =>
      createVocalizationLabel(params.detectionJobId, params.startUtc, params.endUtc, {
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
          vars.startUtc,
          vars.endUtc,
        ],
      });
      qc.invalidateQueries({
        queryKey: ["labeling", "summary", vars.detectionJobId],
      });
      qc.invalidateQueries({ queryKey: ["labeling", "training-summary"] });
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
      startUtc: number;
      endUtc: number;
    }) => deleteVocalizationLabel(params.labelId),
    onSuccess: (_data, vars) => {
      qc.invalidateQueries({
        queryKey: [
          "labeling",
          "vocalization-labels",
          vars.detectionJobId,
          vars.startUtc,
          vars.endUtc,
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
  startUtc: number | null,
  endUtc: number | null,
  embeddingSetIds?: string[],
) {
  return useQuery({
    queryKey: [
      "labeling",
      "neighbors",
      detectionJobId,
      startUtc,
      endUtc,
      embeddingSetIds,
    ],
    queryFn: () =>
      fetchDetectionNeighbors(detectionJobId!, {
        start_utc: startUtc!,
        end_utc: endUtc!,
        embedding_set_ids: embeddingSetIds,
      }),
    enabled:
      detectionJobId !== null &&
      startUtc !== null &&
      endUtc !== null,
    staleTime: Infinity,
  });
}

// ---- Training Summary ----

export function useTrainingSummary() {
  return useQuery({
    queryKey: ["labeling", "training-summary"],
    queryFn: fetchTrainingSummary,
    refetchInterval: 5000,
  });
}
