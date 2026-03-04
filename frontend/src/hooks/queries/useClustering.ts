import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  fetchClusteringJobs,
  fetchClusters,
  fetchVisualization,
  fetchMetrics,
  fetchDendrogram,
  fetchFragmentation,
  fetchStability,
  fetchClassifier,
  fetchLabelQueue,
  fetchRefinement,
  fetchParameterSweep,
  fetchAssignments,
  createClusteringJob,
  deleteClusteringJob,
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

export function useDendrogram(jobId: string | null) {
  return useQuery({
    queryKey: ["dendrogram", jobId],
    queryFn: () => fetchDendrogram(jobId!),
    enabled: !!jobId,
    staleTime: Infinity,
  });
}

export function useFragmentation(jobId: string | null) {
  return useQuery({
    queryKey: ["fragmentation", jobId],
    queryFn: () => fetchFragmentation(jobId!),
    enabled: !!jobId,
    staleTime: Infinity,
  });
}

export function useStability(jobId: string | null) {
  return useQuery({
    queryKey: ["stability", jobId],
    queryFn: () => fetchStability(jobId!),
    enabled: !!jobId,
    staleTime: Infinity,
  });
}

export function useClassifier(jobId: string | null) {
  return useQuery({
    queryKey: ["classifier", jobId],
    queryFn: () => fetchClassifier(jobId!),
    enabled: !!jobId,
    staleTime: Infinity,
  });
}

export function useLabelQueue(jobId: string | null) {
  return useQuery({
    queryKey: ["labelQueue", jobId],
    queryFn: () => fetchLabelQueue(jobId!),
    enabled: !!jobId,
    staleTime: Infinity,
  });
}

export function useRefinement(jobId: string | null) {
  return useQuery({
    queryKey: ["refinement", jobId],
    queryFn: () => fetchRefinement(jobId!),
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

export function useDeleteClusteringJob() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (jobId: string) => deleteClusteringJob(jobId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["clusteringJobs"] });
    },
  });
}
