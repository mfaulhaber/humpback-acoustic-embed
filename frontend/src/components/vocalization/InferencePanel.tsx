import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { CheckCircle, Loader2, Play, RotateCcw } from "lucide-react";
import {
  useVocClassifierModels,
  useVocClassifierInferenceJobs,
  useCreateVocClassifierInferenceJob,
  useVocClassifierInferenceJob,
} from "@/hooks/queries/useVocalization";
import { shortId } from "@/utils/format";
import type { LabelingSource } from "@/api/types";

interface Props {
  source: LabelingSource;
  embeddingsReady: boolean;
  /** For local sources, the resolved embedding set ID from folder processing. */
  localEmbeddingSetId?: string | null;
  onInferenceReady: (inferenceJobId: string) => void;
}

/** Derive the inference source_type and source_id from the LabelingSource. */
function inferenceSourceParams(
  source: LabelingSource,
  localEmbeddingSetId?: string | null,
): {
  source_type: "detection_job" | "embedding_set";
  source_id: string;
} | null {
  switch (source.type) {
    case "detection_job":
      return { source_type: "detection_job", source_id: source.jobId };
    case "embedding_set":
      return { source_type: "embedding_set", source_id: source.embeddingSetId };
    case "local":
      if (!localEmbeddingSetId) return null;
      return { source_type: "embedding_set", source_id: localEmbeddingSetId };
  }
}

export function InferencePanel({
  source,
  embeddingsReady,
  localEmbeddingSetId,
  onInferenceReady,
}: Props) {
  const { data: models = [] } = useVocClassifierModels();
  const { data: allInferenceJobs = [] } = useVocClassifierInferenceJobs();
  const createMut = useCreateVocClassifierInferenceJob();

  const [modelId, setModelId] = useState<string>("");
  const [activeJobId, setActiveJobId] = useState<string | null>(null);

  const activeModel = models.find((m) => m.is_active);
  const effectiveModelId = modelId || activeModel?.id || "";

  const sourceParams = inferenceSourceParams(source, localEmbeddingSetId);
  const source_type = sourceParams?.source_type ?? "detection_job";
  const source_id = sourceParams?.source_id ?? "";

  // Auto-detect existing inference job for this source
  const existingJob = allInferenceJobs.find(
    (j) =>
      j.source_type === source_type &&
      j.source_id === source_id &&
      j.status === "complete",
  );

  // Poll active job status
  const { data: polledJob } = useVocClassifierInferenceJob(activeJobId);

  // Auto-select existing job or completed polled job
  useEffect(() => {
    if (existingJob && !activeJobId) {
      onInferenceReady(existingJob.id);
    }
  }, [existingJob, activeJobId, onInferenceReady]);

  useEffect(() => {
    if (polledJob?.status === "complete") {
      onInferenceReady(polledJob.id);
    }
  }, [polledJob?.status, polledJob?.id, onInferenceReady]);

  // Reset when source changes
  useEffect(() => {
    setActiveJobId(null);
  }, [source_id]);

  if (!embeddingsReady) {
    return (
      <Card>
        <CardHeader className="pb-2 pt-3">
          <CardTitle className="text-sm text-muted-foreground">
            Inference — waiting for embeddings...
          </CardTitle>
        </CardHeader>
      </Card>
    );
  }

  const isRunning =
    polledJob?.status === "queued" || polledJob?.status === "running";

  // Inference already complete (existing or just finished)
  const completedJob =
    existingJob ?? (polledJob?.status === "complete" ? polledJob : null);
  if (completedJob && !isRunning) {
    const modelName = models.find(
      (m) => m.id === completedJob.vocalization_model_id,
    )?.name;
    const resultCount =
      (completedJob.result_summary as Record<string, unknown> | null)
        ?.n_predictions ?? "";
    const differentModel =
      activeModel && completedJob.vocalization_model_id !== activeModel.id;

    return (
      <Card>
        <CardHeader className="pb-2 pt-3">
          <CardTitle className="text-sm flex items-center gap-2 text-muted-foreground">
            <CheckCircle className="h-4 w-4 text-green-600" />
            Inference — Complete
            {modelName && ` (${modelName})`}
            {resultCount && ` — ${resultCount} scored`}
            {differentModel && (
              <Button
                size="sm"
                variant="ghost"
                className="h-6 px-2 ml-2"
                onClick={() => {
                  if (!effectiveModelId) return;
                  createMut.mutate(
                    {
                      vocalization_model_id: effectiveModelId,
                      source_type,
                      source_id,
                    },
                    { onSuccess: (job) => setActiveJobId(job.id) },
                  );
                }}
              >
                <RotateCcw className="h-3 w-3 mr-1" />
                Rescore with {activeModel?.name ?? "active model"}
              </Button>
            )}
          </CardTitle>
        </CardHeader>
      </Card>
    );
  }

  // Running
  if (isRunning) {
    return (
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">Inference</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" />
            Running inference...
          </div>
        </CardContent>
      </Card>
    );
  }

  // No inference yet — show form
  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Inference</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="space-y-1">
          <label className="text-sm font-medium">Model</label>
          <Select value={effectiveModelId} onValueChange={setModelId}>
            <SelectTrigger className="h-8">
              <SelectValue placeholder="Select model..." />
            </SelectTrigger>
            <SelectContent>
              {models.map((m) => (
                <SelectItem key={m.id} value={m.id}>
                  {m.name || shortId(m.id)}
                  {m.is_active && " (active)"}
                  {" — "}
                  {m.vocabulary_snapshot.length} types
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
        <Button
          size="sm"
          onClick={() => {
            if (!effectiveModelId) return;
            createMut.mutate(
              {
                vocalization_model_id: effectiveModelId,
                source_type,
                source_id,
              },
              { onSuccess: (job) => setActiveJobId(job.id) },
            );
          }}
          disabled={!effectiveModelId || createMut.isPending}
        >
          {createMut.isPending ? (
            <Loader2 className="h-3.5 w-3.5 mr-1 animate-spin" />
          ) : (
            <Play className="h-3.5 w-3.5 mr-1" />
          )}
          Run Inference
        </Button>
      </CardContent>
    </Card>
  );
}
