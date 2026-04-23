import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  fetchVocalizationLabels,
  createVocalizationLabel,
  deleteVocalizationLabel,
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
      qc.invalidateQueries({ queryKey: ["labeling", "training-summary"] });
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

export function useDetectionNeighbors(
  detectionJobId: string | null,
  rowId: string | null,
  embeddingSetIds?: string[],
) {
  return useQuery({
    queryKey: [
      "labeling",
      "neighbors",
      detectionJobId,
      rowId,
      embeddingSetIds,
    ],
    queryFn: () =>
      fetchDetectionNeighbors(detectionJobId!, {
        row_id: rowId!,
        embedding_set_ids: embeddingSetIds,
      }),
    enabled: detectionJobId !== null && rowId !== null,
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
