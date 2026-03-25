import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  fetchTimelineConfidence,
  fetchDetectionContent,
  prepareTimelineTiles,
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
