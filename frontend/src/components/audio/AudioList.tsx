import type { AudioFile } from "@/api/types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { FolderTree } from "@/components/shared/FolderTree";
import { fmtDate, formatTime } from "@/utils/format";

interface AudioListProps {
  audioFiles: AudioFile[];
  onSelect: (file: AudioFile) => void;
}

export function AudioList({ audioFiles, onSelect }: AudioListProps) {
  if (audioFiles.length === 0) {
    return (
      <Card>
        <CardContent className="p-6 text-center text-muted-foreground text-sm">
          No audio files uploaded yet.
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Audio Files ({audioFiles.length})</CardTitle>
      </CardHeader>
      <CardContent>
        <FolderTree
          items={audioFiles}
          getPath={(f) => f.folder_path}
          stateKey="audioTree"
          renderLeaf={(file) => (
            <button
              onClick={() => onSelect(file)}
              className="flex items-center gap-3 w-full text-left py-1.5 px-2 rounded hover:bg-accent text-sm"
            >
              <span className="flex-1 truncate font-medium">{file.filename}</span>
              <span className="text-muted-foreground text-xs shrink-0">
                {file.duration_seconds != null ? formatTime(file.duration_seconds) : "—"}
              </span>
              <span className="text-muted-foreground text-xs shrink-0">
                {file.sample_rate_original != null ? `${file.sample_rate_original} Hz` : ""}
              </span>
              <span className="text-muted-foreground text-xs shrink-0">{fmtDate(file.created_at)}</span>
            </button>
          )}
        />
      </CardContent>
    </Card>
  );
}
