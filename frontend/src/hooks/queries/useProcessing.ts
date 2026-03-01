import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  fetchProcessingJobs,
  fetchEmbeddingSets,
  createProcessingJob,
  cancelProcessingJob,
} from "@/api/client";
import type { ProcessingJobCreate } from "@/api/types";

export function useProcessingJobs(refetchInterval?: number) {
  return useQuery({
    queryKey: ["processingJobs"],
    queryFn: fetchProcessingJobs,
    refetchInterval,
  });
}

export function useEmbeddingSets(refetchInterval?: number) {
  return useQuery({
    queryKey: ["embeddingSets"],
    queryFn: fetchEmbeddingSets,
    refetchInterval,
  });
}

export function useCreateProcessingJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: ProcessingJobCreate) => createProcessingJob(body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["processingJobs"] });
    },
  });
}

export function useCancelProcessingJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (jobId: string) => cancelProcessingJob(jobId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["processingJobs"] });
    },
  });
}
