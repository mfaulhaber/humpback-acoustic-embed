import { useState } from "react";
import { useAudioFiles } from "@/hooks/queries/useAudioFiles";
import { useEmbeddingSets } from "@/hooks/queries/useProcessing";
import { AudioUpload } from "./AudioUpload";
import { AudioList } from "./AudioList";
import { AudioDetail } from "./AudioDetail";
import type { AudioFile } from "@/api/types";

export function AudioTab() {
  const [selectedAudioId, setSelectedAudioId] = useState<string | null>(null);
  const { data: audioFiles = [], refetch } = useAudioFiles();
  const { data: embeddingSets = [] } = useEmbeddingSets();

  const selectedFile = audioFiles.find((f) => f.id === selectedAudioId) ?? null;

  // For prev/next navigation
  const sortedFiles = [...audioFiles].sort((a, b) => {
    const pa = a.folder_path || "";
    const pb = b.folder_path || "";
    if (pa !== pb) return pa.localeCompare(pb);
    return a.filename.localeCompare(b.filename);
  });
  const currentIndex = sortedFiles.findIndex((f) => f.id === selectedAudioId);

  const handlePrev = () => {
    if (currentIndex > 0) setSelectedAudioId(sortedFiles[currentIndex - 1].id);
  };
  const handleNext = () => {
    if (currentIndex < sortedFiles.length - 1) setSelectedAudioId(sortedFiles[currentIndex + 1].id);
  };

  if (selectedFile) {
    return (
      <AudioDetail
        file={selectedFile}
        embeddingSets={embeddingSets.filter((es) => es.audio_file_id === selectedFile.id)}
        onBack={() => setSelectedAudioId(null)}
        onPrev={currentIndex > 0 ? handlePrev : undefined}
        onNext={currentIndex < sortedFiles.length - 1 ? handleNext : undefined}
      />
    );
  }

  return (
    <div className="space-y-4">
      <AudioUpload onUploadComplete={() => refetch()} />
      <AudioList audioFiles={audioFiles} onSelect={(f: AudioFile) => setSelectedAudioId(f.id)} />
    </div>
  );
}
