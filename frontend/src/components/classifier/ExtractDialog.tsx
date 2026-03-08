import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogFooter,
  DialogDescription,
} from "@/components/ui/dialog";
import { FolderOpen } from "lucide-react";
import { FolderBrowser } from "@/components/shared/FolderBrowser";
import {
  useExtractionSettings,
  useExtractLabeledSamples,
} from "@/hooks/queries/useClassifier";

export function ExtractDialog({
  open,
  onOpenChange,
  selectedIds,
  extractMutation,
  onSuccess,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  selectedIds: Set<string>;
  extractMutation: ReturnType<typeof useExtractLabeledSamples>;
  onSuccess: () => void;
}) {
  const { data: defaults } = useExtractionSettings();
  const [posPath, setPosPath] = useState("");
  const [negPath, setNegPath] = useState("");
  const [posBrowserOpen, setPosBrowserOpen] = useState(false);
  const [negBrowserOpen, setNegBrowserOpen] = useState(false);

  // Initialize paths from defaults when loaded
  useEffect(() => {
    if (defaults && !posPath) setPosPath(defaults.positive_output_path);
    if (defaults && !negPath) setNegPath(defaults.negative_output_path);
  }, [defaults]);

  const handleConfirm = () => {
    extractMutation.mutate(
      {
        jobIds: [...selectedIds],
        positiveOutputPath: posPath || undefined,
        negativeOutputPath: negPath || undefined,
      },
      {
        onSuccess: () => {
          onOpenChange(false);
          onSuccess();
        },
      },
    );
  };

  return (
    <>
      <Dialog open={open} onOpenChange={onOpenChange}>
        <DialogContent className="sm:max-w-lg">
          <DialogHeader>
            <DialogTitle>Extract Labeled Samples</DialogTitle>
            <DialogDescription>
              Extract labeled audio segments from {selectedIds.size} selected detection job
              {selectedIds.size !== 1 ? "s" : ""} as WAV files.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-3 py-2">
            <div>
              <label className="text-sm font-medium">Positive Output Path (humpback)</label>
              <div className="flex gap-2 mt-1">
                <Input
                  value={posPath}
                  onChange={(e) => setPosPath(e.target.value)}
                  placeholder="data/samples/positive"
                />
                <Button
                  variant="outline"
                  size="icon"
                  onClick={() => setPosBrowserOpen(true)}
                  title="Browse folders"
                >
                  <FolderOpen className="h-4 w-4" />
                </Button>
              </div>
            </div>
            <div>
              <label className="text-sm font-medium">Negative Output Path (ship/background)</label>
              <div className="flex gap-2 mt-1">
                <Input
                  value={negPath}
                  onChange={(e) => setNegPath(e.target.value)}
                  placeholder="data/samples/negative"
                />
                <Button
                  variant="outline"
                  size="icon"
                  onClick={() => setNegBrowserOpen(true)}
                  title="Browse folders"
                >
                  <FolderOpen className="h-4 w-4" />
                </Button>
              </div>
            </div>
          </div>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => onOpenChange(false)}
              disabled={extractMutation.isPending}
            >
              Cancel
            </Button>
            <Button onClick={handleConfirm} disabled={extractMutation.isPending}>
              {extractMutation.isPending ? "Queuing…" : "Extract"}
            </Button>
          </DialogFooter>
          {extractMutation.isError && (
            <p className="text-sm text-red-600">
              {(extractMutation.error as Error).message}
            </p>
          )}
        </DialogContent>
      </Dialog>

      <FolderBrowser
        open={posBrowserOpen}
        onOpenChange={setPosBrowserOpen}
        onSelect={setPosPath}
        initialPath={posPath || "/"}
      />
      <FolderBrowser
        open={negBrowserOpen}
        onOpenChange={setNegBrowserOpen}
        onSelect={setNegPath}
        initialPath={negPath || "/"}
      />
    </>
  );
}
