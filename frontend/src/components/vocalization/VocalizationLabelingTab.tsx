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

  const detectionJobId =
    source?.type === "detection_job" ? source.jobId : null;
  const { data: embeddingStatus } = useEmbeddingStatus(detectionJobId);
  const embeddingsReady = embeddingStatus?.has_embeddings === true;

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
          readonly={false}
          onLabelCountChange={handleLabelCountChange}
        />
      )}

      {inferenceJobId && source && (
        <RetrainFooter
          source={source}
          inferenceJobId={inferenceJobId}
          labelCount={labelCount}
        />
      )}
    </div>
  );
}
