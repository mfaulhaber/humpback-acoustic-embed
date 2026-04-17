import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { toast } from "@/components/ui/use-toast";
import {
  fetchReembeddingStatus,
  enqueueReembedding,
} from "@/api/client";

export function useReembeddingStatus(
  detectionJobIds: string[],
  modelVersion: string,
  enabled: boolean,
  refetchInterval?: number | false,
) {
  return useQuery({
    queryKey: ["reembeddingStatus", detectionJobIds, modelVersion],
    queryFn: () => fetchReembeddingStatus(detectionJobIds, modelVersion),
    enabled: enabled && detectionJobIds.length > 0 && !!modelVersion,
    refetchInterval,
  });
}

export function useEnqueueReembedding() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({
      detectionJobId,
      mode,
      modelVersion,
    }: {
      detectionJobId: string;
      mode?: "full" | "sync";
      modelVersion?: string;
    }) => enqueueReembedding(detectionJobId, mode, modelVersion),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["reembeddingStatus"] });
      qc.invalidateQueries({ queryKey: ["embedding-jobs"] });
    },
    onError: (err: Error) => {
      toast({
        title: "Re-embed failed",
        description: err.message,
        variant: "destructive",
      });
    },
  });
}
