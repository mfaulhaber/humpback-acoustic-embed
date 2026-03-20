import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  fetchLabelProcessingJobs,
  createLabelProcessingJob,
  deleteLabelProcessingJob,
  fetchLabelProcessingPreview,
} from "@/api/client";
import type { LabelProcessingJobCreate } from "@/api/types";

export function useLabelProcessingJobs(refetchInterval?: number) {
  return useQuery({
    queryKey: ["labelProcessingJobs"],
    queryFn: fetchLabelProcessingJobs,
    refetchInterval,
  });
}

export function useCreateLabelProcessingJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: LabelProcessingJobCreate) => createLabelProcessingJob(body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["labelProcessingJobs"] });
    },
  });
}

export function useDeleteLabelProcessingJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (jobId: string) => deleteLabelProcessingJob(jobId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["labelProcessingJobs"] });
    },
  });
}

export function useLabelProcessingPreview(
  annotationFolder: string,
  audioFolder: string,
) {
  return useQuery({
    queryKey: ["labelProcessingPreview", annotationFolder, audioFolder],
    queryFn: () => fetchLabelProcessingPreview(annotationFolder, audioFolder),
    enabled: annotationFolder.length > 0 && audioFolder.length > 0,
  });
}
