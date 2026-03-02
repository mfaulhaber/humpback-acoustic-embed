import { useParams, useNavigate } from "react-router-dom";
import { useAudioFiles } from "@/hooks/queries/useAudioFiles";
import { useEmbeddingSets } from "@/hooks/queries/useProcessing";
import { AudioUpload } from "./AudioUpload";
import { AudioList } from "./AudioList";
import { AudioDetail } from "./AudioDetail";
import type { AudioFile } from "@/api/types";

export function AudioTab() {
  const { audioId } = useParams<{ audioId?: string }>();
  const navigate = useNavigate();
  const { data: audioFiles = [], refetch } = useAudioFiles();
  const { data: embeddingSets = [] } = useEmbeddingSets();

  const selectedFile = audioId ? (audioFiles.find((f) => f.id === audioId) ?? null) : null;

  // For prev/next navigation
  const sortedFiles = [...audioFiles].sort((a, b) => {
    const pa = a.folder_path || "";
    const pb = b.folder_path || "";
    if (pa !== pb) return pa.localeCompare(pb);
    return a.filename.localeCompare(b.filename);
  });
  const currentIndex = sortedFiles.findIndex((f) => f.id === audioId);

  const handlePrev = () => {
    if (currentIndex > 0) navigate(`/app/audio/${sortedFiles[currentIndex - 1].id}`);
  };
  const handleNext = () => {
    if (currentIndex < sortedFiles.length - 1) navigate(`/app/audio/${sortedFiles[currentIndex + 1].id}`);
  };

  if (selectedFile) {
    return (
      <AudioDetail
        file={selectedFile}
        embeddingSets={embeddingSets.filter((es) => es.audio_file_id === selectedFile.id)}
        onBack={() => navigate("/app/audio")}
        onPrev={currentIndex > 0 ? handlePrev : undefined}
        onNext={currentIndex < sortedFiles.length - 1 ? handleNext : undefined}
      />
    );
  }

  return (
    <div className="space-y-4">
      <AudioUpload onUploadComplete={() => refetch()} />
      <AudioList audioFiles={audioFiles} onSelect={(f: AudioFile) => navigate(`/app/audio/${f.id}`)} />
    </div>
  );
}
