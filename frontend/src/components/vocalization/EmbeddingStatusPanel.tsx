import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { CheckCircle, AlertTriangle, Loader2, RefreshCw } from "lucide-react";
import {
  useEmbeddingStatus,
  useGenerateEmbeddings,
  useEmbeddingGenerationStatus,
} from "@/hooks/queries/useVocalization";
import { useQueryClient } from "@tanstack/react-query";
import { useEffect } from "react";

interface Props {
  detectionJobId: string | null;
}

export function EmbeddingStatusPanel({ detectionJobId }: Props) {
  const { data: status } = useEmbeddingStatus(detectionJobId);
  const { data: genJob } = useEmbeddingGenerationStatus(detectionJobId);
  const generateMut = useGenerateEmbeddings();
  const qc = useQueryClient();

  const isGenerating =
    genJob?.status === "queued" || genJob?.status === "running";

  // Re-check embedding status when generation completes
  useEffect(() => {
    if (genJob?.status === "complete") {
      qc.invalidateQueries({ queryKey: ["embedding-status", detectionJobId] });
    }
  }, [genJob?.status, detectionJobId, qc]);

  if (!detectionJobId) return null;

  // Embeddings exist — collapsed success state
  if (status?.has_embeddings) {
    return (
      <Card>
        <CardHeader className="pb-2 pt-3">
          <CardTitle className="text-sm flex items-center gap-2 text-muted-foreground">
            <CheckCircle className="h-4 w-4 text-green-600" />
            Embeddings — Ready ({status.count} vectors)
          </CardTitle>
        </CardHeader>
      </Card>
    );
  }

  // Generating
  if (isGenerating) {
    const current = genJob?.progress_current ?? 0;
    const total = genJob?.progress_total ?? 0;
    return (
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Embeddings</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" />
            Generating embeddings...{" "}
            {total > 0 && `${current} / ${total} files`}
          </div>
        </CardContent>
      </Card>
    );
  }

  // Failed
  if (genJob?.status === "failed") {
    return (
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Embeddings</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2">
          <div className="flex items-center gap-2 text-sm text-destructive">
            <AlertTriangle className="h-4 w-4" />
            Generation failed: {genJob.error_message}
          </div>
          <Button
            size="sm"
            variant="outline"
            onClick={() => {
              if (detectionJobId) generateMut.mutate(detectionJobId);
            }}
            disabled={generateMut.isPending}
          >
            <RefreshCw className="h-3.5 w-3.5 mr-1" />
            Retry
          </Button>
        </CardContent>
      </Card>
    );
  }

  // Missing — prompt to generate
  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Embeddings</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <p className="text-sm text-muted-foreground">
          This detection job has no stored embeddings. Embeddings are needed for
          vocalization inference.
        </p>
        <Button
          size="sm"
          onClick={() => {
            if (detectionJobId) generateMut.mutate(detectionJobId);
          }}
          disabled={generateMut.isPending}
        >
          {generateMut.isPending ? (
            <Loader2 className="h-3.5 w-3.5 mr-1 animate-spin" />
          ) : null}
          Generate Embeddings
        </Button>
      </CardContent>
    </Card>
  );
}
