import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  fetchTimelineConfidence,
  fetchDetectionContent,
  prepareTimelineTiles,
  fetchPrepareStatus,
  patchDetectionLabels,
  type PrepareTimelineTilesRequest,
} from "@/api/client";
import type { LabelEditItem } from "@/api/types";

export function useTimelineConfidence(jobId: string) {
  return useQuery({
    queryKey: ["timelineConfidence", jobId],
    queryFn: () => fetchTimelineConfidence(jobId),
    staleTime: Infinity,
  });
}

export function useTimelineDetections(jobId: string) {
  return useQuery({
    queryKey: ["timelineDetections", jobId],
    queryFn: () => fetchDetectionContent(jobId),
    staleTime: Infinity,
  });
}

export function usePrepareTimeline() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({
      jobId,
      request,
    }: {
      jobId: string;
      request?: PrepareTimelineTilesRequest;
    }) => prepareTimelineTiles(jobId, request),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["hydrophoneDetectionJobs"] });
    },
  });
}

export function usePrepareStatus(jobId: number | string, enabled: boolean) {
  return useQuery({
    queryKey: ["timelinePrepareStatus", jobId],
    queryFn: () => fetchPrepareStatus(jobId),
    enabled,
    refetchInterval: enabled ? 3000 : false,
  });
}

export function useSaveLabels(jobId: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (edits: LabelEditItem[]) => patchDetectionLabels(jobId, edits),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["timelineDetections", jobId] });
      qc.invalidateQueries({ queryKey: ["embedding-status", jobId] });
    },
  });
}
