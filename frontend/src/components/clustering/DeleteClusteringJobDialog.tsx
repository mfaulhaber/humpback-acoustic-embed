import { useState } from "react";
import { Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { useDeleteClusteringJob } from "@/hooks/queries/useClustering";
import { shortId } from "@/utils/format";

interface DeleteClusteringJobDialogProps {
  jobId: string;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onDeleted: () => void;
}

export function DeleteClusteringJobDialog({
  jobId,
  open,
  onOpenChange,
  onDeleted,
}: DeleteClusteringJobDialogProps) {
  const [error, setError] = useState<string | null>(null);
  const deleteMutation = useDeleteClusteringJob();

  async function handleDelete() {
    setError(null);
    try {
      await deleteMutation.mutateAsync(jobId);
      onOpenChange(false);
      onDeleted();
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Delete failed");
    }
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Delete clustering job</DialogTitle>
          <DialogDescription>
            This will permanently delete clustering job{" "}
            <span className="font-mono font-medium">{shortId(jobId)}</span> and
            all its results.
          </DialogDescription>
        </DialogHeader>

        {error && (
          <div className="rounded bg-destructive/10 text-destructive text-sm p-3">
            {error}
          </div>
        )}

        <DialogFooter>
          <Button
            variant="outline"
            onClick={() => onOpenChange(false)}
            disabled={deleteMutation.isPending}
          >
            Cancel
          </Button>
          <Button
            variant="destructive"
            onClick={handleDelete}
            disabled={deleteMutation.isPending}
          >
            {deleteMutation.isPending ? (
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
