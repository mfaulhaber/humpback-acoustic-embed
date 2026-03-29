import { useMutation, useQuery } from "@tanstack/react-query";
import {
  searchSimilar,
  searchSimilarByVector,
  fetchDetectionEmbedding,
  createAudioSearch,
  pollSearchJob,
} from "@/api/client";
import type { AudioSearchRequest, SearchJobResponse } from "@/api/types";

export function useSearchSimilar(
  body: {
    embedding_set_id: string;
    row_index: number;
    top_k?: number;
    metric?: string;
    exclude_self?: boolean;
    embedding_set_ids?: string[];
  } | null,
) {
  return useQuery({
    queryKey: ["search", "similar", body],
    queryFn: () => searchSimilar(body!),
    enabled: body !== null,
  });
}

export function useSearchByVector(
  body: {
    vector: number[];
    model_version: string;
    top_k?: number;
    metric?: string;
    embedding_set_ids?: string[];
  } | null,
) {
  return useQuery({
    queryKey: ["search", "byVector", body?.model_version, body?.top_k, body?.metric],
    queryFn: () => searchSimilarByVector(body!),
    enabled: body !== null,
  });
}

export function useDetectionEmbedding(
  jobId: string | null,
  startUtc: number | null,
  endUtc: number | null,
) {
  return useQuery({
    queryKey: ["detectionEmbedding", jobId, startUtc, endUtc],
    queryFn: () => fetchDetectionEmbedding(jobId!, startUtc!, endUtc!),
    enabled: jobId !== null && startUtc !== null && endUtc !== null,
    retry: false,
  });
}

export function useCreateAudioSearch() {
  return useMutation({
    mutationFn: (body: AudioSearchRequest) => createAudioSearch(body),
  });
}

export function useSearchJobPoll(jobId: string | null) {
  return useQuery<SearchJobResponse>({
    queryKey: ["search", "job", jobId],
    queryFn: () => pollSearchJob(jobId!),
    enabled: jobId !== null,
    retry: 2,
    refetchInterval: (query) => {
      if (query.state.error) return false;
      const status = query.state.data?.status;
      if (status === "complete" || status === "failed") return false;
      return 500;
    },
  });
}
