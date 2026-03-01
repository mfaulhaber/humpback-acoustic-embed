import { useRef, useState, useCallback } from "react";
import { Upload } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { uploadAudio } from "@/api/client";
import { isAudioFile } from "@/utils/audio";
import { showMsg } from "@/components/shared/MessageToast";

interface AudioUploadProps {
  onUploadComplete: () => void;
}

interface FileWithPath {
  file: File;
  folderPath: string;
}

async function readEntriesRecursive(
  dirEntry: FileSystemDirectoryEntry,
  basePath: string,
): Promise<FileWithPath[]> {
  const reader = dirEntry.createReader();
  const results: FileWithPath[] = [];

  const readBatch = (): Promise<FileSystemEntry[]> =>
    new Promise((resolve, reject) => reader.readEntries(resolve, reject));

  let batch: FileSystemEntry[];
  do {
    batch = await readBatch();
    for (const entry of batch) {
      if (entry.isFile) {
        const fileEntry = entry as FileSystemFileEntry;
        const file = await new Promise<File>((resolve, reject) =>
          fileEntry.file(resolve, reject),
        );
        if (isAudioFile(file.name)) {
          results.push({ file, folderPath: basePath });
        }
      } else if (entry.isDirectory) {
        const subResults = await readEntriesRecursive(
          entry as FileSystemDirectoryEntry,
          basePath ? `${basePath}/${entry.name}` : entry.name,
        );
        results.push(...subResults);
      }
    }
  } while (batch.length > 0);

  return results;
}

export function AudioUpload({ onUploadComplete }: AudioUploadProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState({ done: 0, total: 0, failed: 0 });
  const [dragOver, setDragOver] = useState(false);

  const doUpload = useCallback(
    async (files: FileWithPath[]) => {
      if (files.length === 0) return;
      setUploading(true);
      setProgress({ done: 0, total: files.length, failed: 0 });
      let done = 0;
      let failed = 0;

      for (const { file, folderPath } of files) {
        try {
          await uploadAudio(file, folderPath || undefined);
        } catch {
          failed++;
        }
        done++;
        setProgress({ done, total: files.length, failed });
      }

      setUploading(false);
      if (failed === 0) {
        showMsg("success", `Uploaded ${done} file(s)`);
      } else {
        showMsg("warning", `Uploaded ${done - failed}/${done} file(s), ${failed} failed`);
      }
      onUploadComplete();
    },
    [onUploadComplete],
  );

  const handleFiles = useCallback(
    (fileList: FileList) => {
      const files: FileWithPath[] = [];
      for (let i = 0; i < fileList.length; i++) {
        const f = fileList[i];
        if (!isAudioFile(f.name)) continue;
        // Extract folder path from webkitRelativePath
        let folderPath = "";
        if (f.webkitRelativePath) {
          const parts = f.webkitRelativePath.split("/");
          if (parts.length > 1) {
            folderPath = parts.slice(0, -1).join("/");
          }
        }
        files.push({ file: f, folderPath });
      }
      doUpload(files);
    },
    [doUpload],
  );

  const handleDrop = useCallback(
    async (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);

      const items = e.dataTransfer.items;
      const files: FileWithPath[] = [];

      if (items) {
        const entries: FileSystemEntry[] = [];
        for (let i = 0; i < items.length; i++) {
          const entry = items[i].webkitGetAsEntry?.();
          if (entry) entries.push(entry);
        }

        for (const entry of entries) {
          if (entry.isFile) {
            const fileEntry = entry as FileSystemFileEntry;
            const file = await new Promise<File>((resolve, reject) =>
              fileEntry.file(resolve, reject),
            );
            if (isAudioFile(file.name)) {
              files.push({ file, folderPath: "" });
            }
          } else if (entry.isDirectory) {
            const subFiles = await readEntriesRecursive(
              entry as FileSystemDirectoryEntry,
              entry.name,
            );
            files.push(...subFiles);
          }
        }
      } else {
        // Fallback for browsers without DataTransferItem
        const fl = e.dataTransfer.files;
        for (let i = 0; i < fl.length; i++) {
          if (isAudioFile(fl[i].name)) {
            files.push({ file: fl[i], folderPath: "" });
          }
        }
      }

      doUpload(files);
    },
    [doUpload],
  );

  const pct = progress.total > 0 ? Math.round((progress.done / progress.total) * 100) : 0;

  return (
    <Card>
      <CardContent className="p-4">
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
            dragOver ? "border-primary bg-primary/5" : "border-muted-foreground/25"
          }`}
        >
          <Upload className="h-8 w-8 mx-auto mb-2 text-muted-foreground" />
          <p className="text-sm text-muted-foreground">
            Drag & drop audio files or folders here, or{" "}
            <button
              onClick={() => fileInputRef.current?.click()}
              className="text-primary underline"
              disabled={uploading}
            >
              browse files
            </button>
            {" / "}
            <button
              onClick={() => folderInputRef.current?.click()}
              className="text-primary underline"
              disabled={uploading}
            >
              browse folder
            </button>
          </p>
          <p className="text-xs text-muted-foreground mt-1">Supports WAV, MP3, FLAC</p>
        </div>

        {uploading && (
          <div className="mt-3">
            <div className="w-full bg-secondary rounded-full h-2">
              <div className="bg-primary h-2 rounded-full transition-all" style={{ width: `${pct}%` }} />
            </div>
            <p className="text-xs text-muted-foreground mt-1">
              {progress.done} / {progress.total}
              {progress.failed > 0 && ` (${progress.failed} failed)`}
            </p>
          </div>
        )}

        <input
          ref={fileInputRef}
          type="file"
          accept=".wav,.mp3,.flac"
          multiple
          className="hidden"
          onChange={(e) => e.target.files && handleFiles(e.target.files)}
        />
        <input
          ref={folderInputRef}
          type="file"
          // @ts-expect-error webkitdirectory is non-standard
          webkitdirectory=""
          multiple
          className="hidden"
          onChange={(e) => e.target.files && handleFiles(e.target.files)}
        />
      </CardContent>
    </Card>
  );
}
