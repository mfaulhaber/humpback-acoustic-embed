import { useQuery } from "@tanstack/react-query";
import {
  fetchAllVocalizationLabels,
  fetchVocClassifierInferenceJobs,
} from "@/api/client";

export function useVocalizationOverlay(jobId: string) {
  const labelsQuery = useQuery({
    queryKey: ["vocalizationLabelsAll", jobId],
    queryFn: () => fetchAllVocalizationLabels(jobId),
    staleTime: 30_000,
  });

  const inferenceQuery = useQuery({
    queryKey: ["vocClassifierInferenceJobs"],
    queryFn: fetchVocClassifierInferenceJobs,
    staleTime: 60_000,
  });

  const hasCompletedInference =
    inferenceQuery.data?.some(
      (j) =>
        j.source_type === "detection_job" &&
        j.source_id === jobId &&
        j.status === "complete",
    ) ?? false;

  const labels = labelsQuery.data ?? [];
  const hasVocalizationData =
    hasCompletedInference || labels.length > 0;

  return {
    labels,
    hasVocalizationData,
    isLoading: labelsQuery.isLoading || inferenceQuery.isLoading,
  };
}
