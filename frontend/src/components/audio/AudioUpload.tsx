import { useState, useCallback } from "react";
import { FolderInput } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { importFolder } from "@/api/client";
import { showMsg } from "@/components/shared/MessageToast";
import { FolderBrowser } from "@/components/shared/FolderBrowser";

interface AudioUploadProps {
  onUploadComplete: () => void;
}

export function AudioUpload({ onUploadComplete }: AudioUploadProps) {
  const [folderPath, setFolderPath] = useState("");
  const [folderBrowserOpen, setFolderBrowserOpen] = useState(false);
  const [importing, setImporting] = useState(false);

  const handleImport = useCallback(
    async (path: string) => {
      if (!path.trim()) return;
      setImporting(true);
      try {
        const result = await importFolder(path.trim());
        if (result.errors.length > 0) {
          showMsg("warning", `Imported ${result.imported}, skipped ${result.skipped}, ${result.errors.length} error(s)`);
        } else {
          showMsg("success", `Imported ${result.imported} file(s), skipped ${result.skipped}`);
        }
        onUploadComplete();
      } catch (e) {
        showMsg("error", `Import failed: ${e instanceof Error ? e.message : String(e)}`);
      } finally {
        setImporting(false);
      }
    },
    [onUploadComplete],
  );

  const handleBrowseSelect = useCallback(
    (path: string) => {
      setFolderPath(path);
      handleImport(path);
    },
    [handleImport],
  );

  return (
    <Card>
      <CardContent className="p-4">
        <div className="flex items-center gap-2">
          <Input
            placeholder="Paste folder path, e.g. /Users/michael/data/whales"
            value={folderPath}
            onChange={(e) => setFolderPath(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") handleImport(folderPath);
            }}
            disabled={importing}
            className="flex-1"
          />
          <Button
            onClick={() => handleImport(folderPath)}
            disabled={importing || !folderPath.trim()}
          >
            {importing ? "Importing…" : "Import"}
          </Button>
          <Button
            variant="outline"
            onClick={() => setFolderBrowserOpen(true)}
            disabled={importing}
          >
            <FolderInput className="h-4 w-4 mr-2" />
            Browse
          </Button>
        </div>
        <p className="text-xs text-muted-foreground mt-2">
          Scans folder for WAV, MP3, FLAC files and references them in-place (no copy)
        </p>

        <FolderBrowser
          open={folderBrowserOpen}
          onOpenChange={setFolderBrowserOpen}
          onSelect={handleBrowseSelect}
        />
      </CardContent>
    </Card>
  );
}
