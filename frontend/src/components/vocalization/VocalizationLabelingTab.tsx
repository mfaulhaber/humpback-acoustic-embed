import { useState, useCallback } from "react";
import { SourceSelector } from "./SourceSelector";
import { EmbeddingStatusPanel } from "./EmbeddingStatusPanel";
import { InferencePanel } from "./InferencePanel";
import { LabelingWorkspace } from "./LabelingWorkspace";
import { RetrainFooter } from "./RetrainFooter";
import { useEmbeddingStatus } from "@/hooks/queries/useVocalization";
import type { LabelingSource } from "@/api/types";

export function VocalizationLabelingTab() {
  const [source, setSource] = useState<LabelingSource | null>(null);
  const [inferenceJobId, setInferenceJobId] = useState<string | null>(null);
  const [labelCount, setLabelCount] = useState(0);

  const isReadonly = source?.type === "embedding_set";

  // Embedding status for detection_job source
  const detectionJobId =
    source?.type === "detection_job" ? source.jobId : null;
  const { data: embeddingStatus } = useEmbeddingStatus(detectionJobId);
  const embeddingsReady =
    source?.type === "embedding_set" ||
    embeddingStatus?.has_embeddings === true;

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
    setLabelCount(count);
  }, []);

  // Show embedding status for detection_job and local sources (not embedding_set)
  const showEmbeddingStatus =
    source !== null && source.type !== "embedding_set";

  return (
    <div className="space-y-4 max-w-5xl pb-16">
      <h2 className="text-lg font-semibold">Vocalization Labeling</h2>

      <SourceSelector source={source} onSourceChange={handleSourceChange} />

      {showEmbeddingStatus && detectionJobId && (
        <EmbeddingStatusPanel detectionJobId={detectionJobId} />
      )}

      {source && (
        <InferencePanel
          source={source}
          embeddingsReady={embeddingsReady}
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
          labelCount={labelCount}
        />
      )}
    </div>
  );
}
