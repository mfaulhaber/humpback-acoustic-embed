import { useQuery } from "@tanstack/react-query";
import type { EmbeddingSet } from "@/api/types";

export function useEmbeddingSets(refetchInterval?: number) {
  return useQuery({
    queryKey: ["embeddingSets"],
    queryFn: async (): Promise<EmbeddingSet[]> => [],
    refetchInterval,
  });
}
