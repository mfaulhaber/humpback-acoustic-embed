import { useState, useCallback } from "react";
import { SourceSelector } from "./SourceSelector";
import { EmbeddingStatusPanel } from "./EmbeddingStatusPanel";
import { InferencePanel } from "./InferencePanel";
import { LabelingWorkspace } from "./LabelingWorkspace";
import { RetrainFooter } from "./RetrainFooter";
import {
  useEmbeddingStatus,
  useFolderEmbeddingSet,
} from "@/hooks/queries/useVocalization";
import type { LabelingSource } from "@/api/types";

export function VocalizationLabelingTab() {
  const [source, setSource] = useState<LabelingSource | null>(null);
  const [inferenceJobId, setInferenceJobId] = useState<string | null>(null);
  const [labelCount, setLabelCount] = useState(0);

  // Local and embedding_set sources are readonly (no detection_job_id for labels)
  const isReadonly =
    source?.type === "embedding_set" || source?.type === "local";

  // Embedding status for detection_job source
  const detectionJobId =
    source?.type === "detection_job" ? source.jobId : null;
  const { data: embeddingStatus } = useEmbeddingStatus(detectionJobId);

  // Folder embedding status for local source
  const localFolderPath =
    source?.type === "local" ? source.folderPath : null;
  const { data: folderStatus } = useFolderEmbeddingSet(localFolderPath);

  const embeddingsReady =
    source?.type === "embedding_set" ||
    embeddingStatus?.has_embeddings === true ||
    folderStatus?.status === "ready";

  const handleSourceChange = useCallback(
    (newSource: LabelingSource | null) => {
      setSource(newSource);
      setInferenceJobId(null);
      setLabelCount(0);
    },
    [],
  );

  const handleInferenceReady = useCallback((jobId: string) => {
    setInferenceJobId(jobId);
  }, []);

  const handleLabelCountChange = useCallback((count: number) => {
    setLabelCount((prev) => prev + count);
  }, []);

  return (
    <div className="space-y-4 max-w-5xl pb-16">
      <h2 className="text-lg font-semibold">Vocalization Labeling</h2>

      <SourceSelector source={source} onSourceChange={handleSourceChange} />

      {/* Embedding status for detection jobs */}
      {source?.type === "detection_job" && detectionJobId && (
        <EmbeddingStatusPanel detectionJobId={detectionJobId} />
      )}

      {/* Folder processing status for local sources */}
      {source?.type === "local" && folderStatus && folderStatus.status !== "ready" && (
        <div className="rounded-md border px-4 py-3 text-sm text-muted-foreground">
          Processing folder... {folderStatus.processed_files} / {folderStatus.total_files} files
        </div>
      )}

      {source && (
        <InferencePanel
          source={source}
          embeddingsReady={embeddingsReady}
          localEmbeddingSetId={folderStatus?.embedding_set_ids?.[0] ?? null}
          onInferenceReady={handleInferenceReady}
        />
      )}

      {inferenceJobId && source && (
        <LabelingWorkspace
          inferenceJobId={inferenceJobId}
          source={source}
          readonly={isReadonly}
          onLabelCountChange={handleLabelCountChange}
        />
      )}

      {inferenceJobId && source && !isReadonly && (
        <RetrainFooter
          source={source}
          inferenceJobId={inferenceJobId}
          labelCount={labelCount}
        />
      )}
    </div>
  );
}
