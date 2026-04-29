import { useQuery } from "@tanstack/react-query";
import type { AudioFile } from "@/api/types";

export function useAudioFiles() {
  return useQuery({
    queryKey: ["audioFiles"],
    queryFn: async (): Promise<AudioFile[]> => [],
  });
}

export function useAudioFile(id: string | null) {
  return useQuery({
    queryKey: ["audioFile", id],
    queryFn: async (): Promise<AudioFile | null> => null,
    enabled: false,
  });
}
