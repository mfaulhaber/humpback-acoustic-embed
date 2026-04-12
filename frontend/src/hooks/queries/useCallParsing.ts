import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  fetchRegionDetectionJobs,
  createRegionDetectionJob,
  deleteRegionDetectionJob,
} from "@/api/client";
import type { CreateRegionJobRequest, ModelConfig } from "@/api/types";

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

export function resolveModelConfigId(
  modelVersion: string,
  modelConfigs: ModelConfig[],
): string | null {
  const match = modelConfigs.find((mc) => mc.name === modelVersion);
  return match?.id ?? null;
}
