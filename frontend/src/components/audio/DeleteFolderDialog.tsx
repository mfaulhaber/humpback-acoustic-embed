import { useState, useEffect } from "react";
import { AlertTriangle, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { fetchFolderDeletePreview, deleteFolder } from "@/api/client";
import type { FolderDeletePreview } from "@/api/types";

interface DeleteFolderDialogProps {
  folderPath: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onDeleted: () => void;
}

export function DeleteFolderDialog({ folderPath, open, onOpenChange, onDeleted }: DeleteFolderDialogProps) {
  const [preview, setPreview] = useState<FolderDeletePreview | null>(null);
  const [loading, setLoading] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!open) {
      setPreview(null);
      setError(null);
      return;
    }
    setLoading(true);
    fetchFolderDeletePreview(folderPath)
      .then(setPreview)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, [open, folderPath]);

  async function handleDelete() {
    setDeleting(true);
    setError(null);
    try {
      await deleteFolder(folderPath, true);
      onOpenChange(false);
      onDeleted();
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Delete failed");
    } finally {
      setDeleting(false);
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Delete folder &ldquo;{folderPath}&rdquo;</DialogTitle>
          <DialogDescription>
            This will permanently delete the folder and all related data.
          </DialogDescription>
        </DialogHeader>

        {loading && (
          <div className="flex items-center justify-center py-6">
            <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
          </div>
        )}

        {error && (
          <div className="rounded bg-destructive/10 text-destructive text-sm p-3">
            {error}
          </div>
        )}

        {preview && (
          <div className="space-y-3 text-sm">
            <div className="grid grid-cols-2 gap-2">
              <div className="text-muted-foreground">Audio files</div>
              <div className="font-medium">{preview.audio_file_count}</div>
              <div className="text-muted-foreground">Embedding sets</div>
              <div className="font-medium">{preview.embedding_set_count}</div>
              <div className="text-muted-foreground">Processing jobs</div>
              <div className="font-medium">{preview.processing_job_count}</div>
            </div>

            {preview.has_clustering_conflicts && (
              <div className="flex items-start gap-2 rounded border border-yellow-500/30 bg-yellow-500/10 p-3">
                <AlertTriangle className="h-4 w-4 text-yellow-600 shrink-0 mt-0.5" />
                <div>
                  <p className="font-medium text-yellow-700">
                    {preview.affected_clustering_jobs.length} clustering job(s) will also be deleted
                  </p>
                  <p className="text-muted-foreground text-xs mt-1">
                    These jobs reference embedding sets from this folder.
                  </p>
                </div>
              </div>
            )}
          </div>
        )}

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)} disabled={deleting}>
            Cancel
          </Button>
          <Button
            variant="destructive"
            onClick={handleDelete}
            disabled={loading || deleting || !preview}
          >
            {deleting ? (
              <>
                <Loader2 className="h-4 w-4 animate-spin mr-1" />
                Deleting...
              </>
            ) : (
              "Delete"
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
