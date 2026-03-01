import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  fetchClusteringJobs,
  fetchClusters,
  fetchVisualization,
  fetchMetrics,
  fetchParameterSweep,
  fetchAssignments,
  createClusteringJob,
} from "@/api/client";
import type { ClusteringJobCreate } from "@/api/types";

export function useClusteringJobs(refetchInterval?: number) {
  return useQuery({
    queryKey: ["clusteringJobs"],
    queryFn: fetchClusteringJobs,
    refetchInterval,
  });
}

export function useClusters(jobId: string | null) {
  return useQuery({
    queryKey: ["clusters", jobId],
    queryFn: () => fetchClusters(jobId!),
    enabled: !!jobId,
  });
}

export function useVisualization(jobId: string | null) {
  return useQuery({
    queryKey: ["visualization", jobId],
    queryFn: () => fetchVisualization(jobId!),
    enabled: !!jobId,
    staleTime: Infinity,
  });
}

export function useMetrics(jobId: string | null) {
  return useQuery({
    queryKey: ["metrics", jobId],
    queryFn: () => fetchMetrics(jobId!),
    enabled: !!jobId,
    staleTime: Infinity,
  });
}

export function useParameterSweep(jobId: string | null) {
  return useQuery({
    queryKey: ["parameterSweep", jobId],
    queryFn: () => fetchParameterSweep(jobId!),
    enabled: !!jobId,
    staleTime: Infinity,
  });
}

export function useAssignments(clusterId: string | null) {
  return useQuery({
    queryKey: ["assignments", clusterId],
    queryFn: () => fetchAssignments(clusterId!),
    enabled: !!clusterId,
  });
}

export function useCreateClusteringJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: ClusteringJobCreate) => createClusteringJob(body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["clusteringJobs"] });
    },
  });
}
