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

interface Props {
  detectionJobId: string | null;
  embeddingsReady: boolean;
  onInferenceReady: (inferenceJobId: string) => void;
}

export function InferencePanel({
  detectionJobId,
  embeddingsReady,
  onInferenceReady,
}: Props) {
  const { data: models = [] } = useVocClassifierModels();
  const { data: allInferenceJobs = [] } = useVocClassifierInferenceJobs();
  const createMut = useCreateVocClassifierInferenceJob();

  const [modelId, setModelId] = useState<string>("");
  const [activeJobId, setActiveJobId] = useState<string | null>(null);

  const activeModel = models.find((m) => m.is_active);
  const effectiveModelId = modelId || activeModel?.id || "";

  // Auto-detect existing inference job for this detection job
  const existingJob = allInferenceJobs.find(
    (j) =>
      j.source_type === "detection_job" &&
      j.source_id === detectionJobId &&
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

  // Reset when detection job changes
  useEffect(() => {
    setActiveJobId(null);
  }, [detectionJobId]);

  if (!detectionJobId || !embeddingsReady) return null;

  const isRunning =
    polledJob?.status === "queued" || polledJob?.status === "running";

  // Inference already complete (existing or just finished)
  const completedJob = existingJob ?? (polledJob?.status === "complete" ? polledJob : null);
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
                  if (!effectiveModelId || !detectionJobId) return;
                  createMut.mutate(
                    {
                      vocalization_model_id: effectiveModelId,
                      source_type: "detection_job",
                      source_id: detectionJobId,
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
            if (!effectiveModelId || !detectionJobId) return;
            createMut.mutate(
              {
                vocalization_model_id: effectiveModelId,
                source_type: "detection_job",
                source_id: detectionJobId,
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
