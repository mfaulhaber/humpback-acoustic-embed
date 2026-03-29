import { useState, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useQuery } from "@tanstack/react-query";
import {
  fetchDetectionJobs,
  fetchHydrophoneDetectionJobs,
  fetchHydrophones,
  fetchEmbeddingSets,
} from "@/api/client";
import type {
  DetectionJob,
  HydrophoneInfo,
  EmbeddingSet,
  LabelingSource,
} from "@/api/types";

type SourceType = "detection_job" | "embedding_set" | "local";

interface Props {
  source: LabelingSource | null;
  onSourceChange: (source: LabelingSource | null) => void;
}

function formatUtcDate(epoch: number): string {
  const d = new Date(epoch * 1000);
  return d.toISOString().replace("T", " ").slice(0, 16) + " UTC";
}

function formatJobLabel(
  job: DetectionJob,
  hydrophoneMap: Map<string, HydrophoneInfo>,
): string {
  if (job.hydrophone_id && job.start_timestamp && job.end_timestamp) {
    const info = hydrophoneMap.get(job.hydrophone_id);
    const displayName = info
      ? `${info.name} (${info.location})`
      : (job.hydrophone_name ?? job.hydrophone_id);
    return `${displayName}    ${formatUtcDate(job.start_timestamp)} — ${formatUtcDate(job.end_timestamp)}`;
  }
  const windowCount =
    (job.result_summary as Record<string, unknown> | null)?.n_total_windows ??
    (job.result_summary as Record<string, unknown> | null)?.n_detections ??
    null;
  const suffix = windowCount !== null ? `    ${windowCount} windows` : "";
  return `${job.audio_folder ?? "unknown"}${suffix}`;
}

function topLevelFolder(parquetPath: string): string {
  // parquet_path like "embeddings/model/audioId/sig.parquet"
  // or a path relative to storage — extract the audio folder from it
  const parts = parquetPath.replace(/\\/g, "/").split("/");
  // Try to find a meaningful top-level name
  // If it starts with "embeddings/", skip that
  const start = parts[0] === "embeddings" ? 1 : 0;
  return parts[start] ?? parquetPath;
}

function embeddingSetDisplayName(es: EmbeddingSet): string {
  return topLevelFolder(es.parquet_path);
}

export function SourceSelector({ source, onSourceChange }: Props) {
  const [sourceType, setSourceType] = useState<SourceType>(
    source?.type ?? "detection_job",
  );
  const [localFolder, setLocalFolder] = useState("");

  // Detection jobs data
  const { data: localJobs = [] } = useQuery({
    queryKey: ["detectionJobs"],
    queryFn: fetchDetectionJobs,
  });
  const { data: hydrophoneJobs = [] } = useQuery({
    queryKey: ["hydrophoneDetectionJobs"],
    queryFn: fetchHydrophoneDetectionJobs,
  });
  const { data: hydrophones = [] } = useQuery({
    queryKey: ["hydrophones"],
    queryFn: fetchHydrophones,
    staleTime: 60_000,
  });

  // Embedding sets data
  const { data: embeddingSets = [] } = useQuery({
    queryKey: ["embeddingSets"],
    queryFn: fetchEmbeddingSets,
    staleTime: 30_000,
  });

  const hydrophoneMap = useMemo(() => {
    const m = new Map<string, HydrophoneInfo>();
    for (const h of hydrophones) m.set(h.id, h);
    return m;
  }, [hydrophones]);

  const allJobs = [...localJobs, ...hydrophoneJobs];
  const completedJobs = allJobs.filter((j) => j.status === "complete");
  const hydrophoneCompleted = completedJobs.filter((j) => j.hydrophone_id);
  const localCompleted = completedJobs.filter((j) => !j.hydrophone_id);

  const selectedJobId =
    source?.type === "detection_job" ? source.jobId : "";
  const selectedEmbeddingSetId =
    source?.type === "embedding_set" ? source.embeddingSetId : "";

  function handleTypeChange(newType: SourceType) {
    setSourceType(newType);
    onSourceChange(null);
  }

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base">Source</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {/* Segmented toggle */}
        <div className="flex rounded-md border overflow-hidden">
          {(
            [
              ["detection_job", "Detection Jobs"],
              ["embedding_set", "Embedding Set"],
              ["local", "Local"],
            ] as const
          ).map(([type, label]) => (
            <button
              key={type}
              className={`flex-1 px-3 py-1.5 text-sm font-medium transition-colors ${
                sourceType === type
                  ? "bg-primary text-primary-foreground"
                  : "bg-background hover:bg-muted text-muted-foreground"
              }`}
              onClick={() => handleTypeChange(type)}
            >
              {label}
            </button>
          ))}
        </div>

        {/* Source-specific input */}
        {sourceType === "detection_job" && (
          <Select
            value={selectedJobId}
            onValueChange={(v) =>
              onSourceChange(v ? { type: "detection_job", jobId: v } : null)
            }
          >
            <SelectTrigger className="h-9">
              <SelectValue placeholder="Select a detection job..." />
            </SelectTrigger>
            <SelectContent>
              {hydrophoneCompleted.length > 0 && (
                <SelectGroup>
                  <SelectLabel>Hydrophone</SelectLabel>
                  {hydrophoneCompleted.map((j) => (
                    <SelectItem key={j.id} value={j.id}>
                      {formatJobLabel(j, hydrophoneMap)}
                    </SelectItem>
                  ))}
                </SelectGroup>
              )}
              {localCompleted.length > 0 && (
                <SelectGroup>
                  <SelectLabel>Local</SelectLabel>
                  {localCompleted.map((j) => (
                    <SelectItem key={j.id} value={j.id}>
                      {formatJobLabel(j, hydrophoneMap)}
                    </SelectItem>
                  ))}
                </SelectGroup>
              )}
            </SelectContent>
          </Select>
        )}

        {sourceType === "embedding_set" && (
          <Select
            value={selectedEmbeddingSetId}
            onValueChange={(v) =>
              onSourceChange(
                v ? { type: "embedding_set", embeddingSetId: v } : null,
              )
            }
          >
            <SelectTrigger className="h-9">
              <SelectValue placeholder="Select an embedding set..." />
            </SelectTrigger>
            <SelectContent>
              {embeddingSets.map((es) => (
                <SelectItem key={es.id} value={es.id}>
                  {embeddingSetDisplayName(es)} — {es.vector_dim}d,{" "}
                  {es.model_version}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        )}

        {sourceType === "local" && (
          <div className="flex gap-2">
            <Input
              placeholder="/path/to/audio/folder"
              value={localFolder}
              onChange={(e) => setLocalFolder(e.target.value)}
              className="h-9"
            />
            <Button
              size="sm"
              className="h-9 shrink-0"
              disabled={!localFolder.trim()}
              onClick={() =>
                onSourceChange({
                  type: "local",
                  folderPath: localFolder.trim(),
                })
              }
            >
              Load
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
