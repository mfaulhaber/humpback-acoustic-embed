import { useQuery } from "@tanstack/react-query";
import { fetchAudioFiles, fetchAudioFile, fetchSpectrogram, fetchEmbeddings } from "@/api/client";

export function useAudioFiles() {
  return useQuery({
    queryKey: ["audioFiles"],
    queryFn: fetchAudioFiles,
  });
}

export function useAudioFile(id: string | null) {
  return useQuery({
    queryKey: ["audioFile", id],
    queryFn: () => fetchAudioFile(id!),
    enabled: !!id,
  });
}

export function useSpectrogram(
  audioId: string | null,
  windowIndex: number,
  windowSizeSeconds?: number,
  targetSampleRate?: number,
) {
  return useQuery({
    queryKey: ["spectrogram", audioId, windowIndex, windowSizeSeconds, targetSampleRate],
    queryFn: () => fetchSpectrogram(audioId!, windowIndex, windowSizeSeconds, targetSampleRate),
    enabled: !!audioId,
  });
}

export function useEmbeddingSimilarity(audioId: string | null, embeddingSetId: string | null) {
  return useQuery({
    queryKey: ["embeddingSimilarity", audioId, embeddingSetId],
    queryFn: () => fetchEmbeddings(audioId!, embeddingSetId!),
    enabled: !!audioId && !!embeddingSetId,
  });
}
