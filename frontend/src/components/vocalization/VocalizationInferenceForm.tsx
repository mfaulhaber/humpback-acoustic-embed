import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { ChevronRight, Loader2, Play, RotateCcw } from "lucide-react";
import {
  useVocClassifierModels,
  useVocClassifierInferenceJobs,
  useCreateVocClassifierInferenceJob,
} from "@/hooks/queries/useVocalization";
import { fetchDetectionJobs, fetchEmbeddingSets } from "@/api/client";
import { useQuery } from "@tanstack/react-query";
import type { VocClassifierInferenceJob } from "@/api/types";
import { fmtDate, shortId } from "@/utils/format";

interface Props {
  selectedJobId: string | null;
  onSelectJob: (jobId: string | null) => void;
}

export function VocalizationInferenceForm({ selectedJobId, onSelectJob }: Props) {
  const { data: models = [] } = useVocClassifierModels();
  const { data: inferenceJobs = [] } = useVocClassifierInferenceJobs();
  const createMut = useCreateVocClassifierInferenceJob();

  const { data: detectionJobs = [] } = useQuery({
    queryKey: ["detectionJobs"],
    queryFn: fetchDetectionJobs,
  });
  const { data: embeddingSets = [] } = useQuery({
    queryKey: ["embeddingSets"],
    queryFn: fetchEmbeddingSets,
  });

  const [modelId, setModelId] = useState<string>("");
  const [sourceType, setSourceType] = useState<"detection_job" | "embedding_set">("detection_job");
  const [sourceId, setSourceId] = useState<string>("");

  const activeModel = models.find((m) => m.is_active);
  const effectiveModelId = modelId || activeModel?.id || "";

  const completedDetJobs = detectionJobs.filter((j) => j.status === "complete");

  function handleQueue() {
    if (!effectiveModelId || !sourceId) return;
    createMut.mutate(
      {
        vocalization_model_id: effectiveModelId,
        source_type: sourceType,
        source_id: sourceId,
      },
      {
        onSuccess: (job) => onSelectJob(job.id),
      },
    );
  }

  function handleRescore(job: VocClassifierInferenceJob) {
    if (!effectiveModelId) return;
    createMut.mutate(
      {
        vocalization_model_id: effectiveModelId,
        source_type: "rescore",
        source_id: job.id,
      },
      {
        onSuccess: (newJob) => onSelectJob(newJob.id),
      },
    );
  }

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Inference</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Model selector */}
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

        {/* Source type */}
        <div className="space-y-1">
          <label className="text-sm font-medium">Source Type</label>
          <Select
            value={sourceType}
            onValueChange={(v) => {
              setSourceType(v as "detection_job" | "embedding_set");
              setSourceId("");
            }}
          >
            <SelectTrigger className="h-8">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="detection_job">Detection Job</SelectItem>
              <SelectItem value="embedding_set">Embedding Set</SelectItem>
            </SelectContent>
          </Select>
        </div>

        {/* Source selector */}
        <div className="space-y-1">
          <label className="text-sm font-medium">Source</label>
          <Select value={sourceId} onValueChange={setSourceId}>
            <SelectTrigger className="h-8">
              <SelectValue placeholder="Select source..." />
            </SelectTrigger>
            <SelectContent>
              {sourceType === "detection_job"
                ? completedDetJobs.map((dj) => (
                    <SelectItem key={dj.id} value={dj.id}>
                      {shortId(dj.id)}
                      {dj.audio_folder && ` — ${dj.audio_folder}`}
                      {dj.hydrophone_name && ` — ${dj.hydrophone_name}`}
                    </SelectItem>
                  ))
                : embeddingSets.map((es) => (
                    <SelectItem key={es.id} value={es.id}>
                      {es.model_version} — {shortId(es.id)}
                    </SelectItem>
                  ))}
            </SelectContent>
          </Select>
        </div>

        {/* Queue button */}
        <Button
          onClick={handleQueue}
          disabled={!effectiveModelId || !sourceId || createMut.isPending}
        >
          {createMut.isPending ? (
            <Loader2 className="h-3.5 w-3.5 mr-1 animate-spin" />
          ) : (
            <Play className="h-3.5 w-3.5 mr-1" />
          )}
          Queue Inference
        </Button>

        {/* Inference job list */}
        {inferenceJobs.length > 0 && (
          <div className="space-y-1">
            <h4 className="text-sm font-medium">Inference Jobs</h4>
            <div className="border rounded-md divide-y text-sm">
              {inferenceJobs.map((j) => (
                <InferenceJobRow
                  key={j.id}
                  job={j}
                  isSelected={j.id === selectedJobId}
                  onSelect={() => onSelectJob(j.id)}
                  onRescore={() => handleRescore(j)}
                  canRescore={!!effectiveModelId}
                  models={models}
                />
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function InferenceJobRow({
  job,
  isSelected,
  onSelect,
  onRescore,
  canRescore,
  models,
}: {
  job: VocClassifierInferenceJob;
  isSelected: boolean;
  onSelect: () => void;
  onRescore: () => void;
  canRescore: boolean;
  models: { id: string; name: string }[];
}) {
  const [open, setOpen] = useState(false);
  const modelName = models.find((m) => m.id === job.vocalization_model_id)?.name;
  const summary = job.result_summary as Record<string, unknown> | null;

  return (
    <Collapsible open={open} onOpenChange={setOpen}>
      <div
        className={`px-3 py-2 cursor-pointer transition-colors ${
          isSelected ? "bg-blue-50" : "hover:bg-muted/50"
        }`}
        onClick={job.status === "complete" ? onSelect : undefined}
      >
        <CollapsibleTrigger
          className="flex items-center justify-between w-full"
          onClick={(e) => e.stopPropagation()}
        >
          <div className="flex items-center gap-2 min-w-0">
            <ChevronRight
              className={`h-3.5 w-3.5 shrink-0 transition-transform ${open ? "rotate-90" : ""}`}
            />
            <span className="text-muted-foreground">{shortId(job.id)}</span>
            <StatusBadge status={job.status} />
            <span className="truncate text-xs text-muted-foreground">
              {job.source_type}: {shortId(job.source_id)}
            </span>
          </div>
          <div className="flex items-center gap-2 shrink-0">
            {job.status === "complete" && canRescore && (
              <Button
                size="sm"
                variant="ghost"
                className="h-6 px-2"
                onClick={(e) => {
                  e.stopPropagation();
                  onRescore();
                }}
                title="Rescore with current model"
              >
                <RotateCcw className="h-3 w-3" />
              </Button>
            )}
            <span className="text-xs text-muted-foreground">{fmtDate(job.created_at)}</span>
          </div>
        </CollapsibleTrigger>

        <CollapsibleContent className="mt-2 pl-5 space-y-1">
          {modelName && (
            <p className="text-xs text-muted-foreground">Model: {modelName}</p>
          )}
          {job.error_message && (
            <p className="text-xs text-destructive">{job.error_message}</p>
          )}
          {summary && (
            <div className="flex flex-wrap gap-1">
              {Object.entries(summary).map(([key, val]) =>
                typeof val === "number" ? (
                  <Badge key={key} variant="outline" className="text-xs">
                    {key}: {val}
                  </Badge>
                ) : null,
              )}
            </div>
          )}
        </CollapsibleContent>
      </div>
    </Collapsible>
  );
}

function StatusBadge({ status }: { status: string }) {
  const variant =
    status === "complete"
      ? "default"
      : status === "failed"
        ? "destructive"
        : "secondary";
  return <Badge variant={variant}>{status}</Badge>;
}
