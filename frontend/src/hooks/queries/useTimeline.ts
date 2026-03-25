import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  fetchTimelineConfidence,
  fetchDetectionContent,
  prepareTimelineTiles,
  fetchPrepareStatus,
} from "@/api/client";

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
    mutationFn: (jobId: string) => prepareTimelineTiles(jobId),
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
