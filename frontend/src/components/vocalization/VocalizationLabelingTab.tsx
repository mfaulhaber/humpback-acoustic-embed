import { useState, useCallback } from "react";
import { DetectionJobPicker } from "./DetectionJobPicker";
import { EmbeddingStatusPanel } from "./EmbeddingStatusPanel";
import { InferencePanel } from "./InferencePanel";
import { LabelingWorkspace } from "./LabelingWorkspace";
import { RetrainFooter } from "./RetrainFooter";
import { useEmbeddingStatus } from "@/hooks/queries/useVocalization";

export function VocalizationLabelingTab() {
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [inferenceJobId, setInferenceJobId] = useState<string | null>(null);
  const [labelCount, setLabelCount] = useState(0);

  // Embedding status is query-driven, not callback-driven
  const { data: embeddingStatus } = useEmbeddingStatus(selectedJobId);
  const embeddingsReady = embeddingStatus?.has_embeddings === true;

  const handleSelectJob = useCallback((jobId: string | null) => {
    setSelectedJobId(jobId);
    setInferenceJobId(null);
    setLabelCount(0);
  }, []);

  const handleInferenceReady = useCallback((jobId: string) => {
    setInferenceJobId(jobId);
  }, []);

  const handleLabelCountChange = useCallback((count: number) => {
    setLabelCount(count);
  }, []);

  return (
    <div className="space-y-4 max-w-5xl pb-16">
      <h2 className="text-lg font-semibold">Vocalization Labeling</h2>

      <DetectionJobPicker
        selectedJobId={selectedJobId}
        onSelect={handleSelectJob}
      />

      {selectedJobId && (
        <EmbeddingStatusPanel
          detectionJobId={selectedJobId}
        />
      )}

      {selectedJobId && (
        <InferencePanel
          detectionJobId={selectedJobId}
          embeddingsReady={embeddingsReady}
          onInferenceReady={handleInferenceReady}
        />
      )}

      {inferenceJobId && selectedJobId && (
        <LabelingWorkspace
          inferenceJobId={inferenceJobId}
          detectionJobId={selectedJobId}
          onLabelCountChange={handleLabelCountChange}
        />
      )}

      {inferenceJobId && selectedJobId && (
        <RetrainFooter
          detectionJobId={selectedJobId}
          labelCount={labelCount}
        />
      )}
    </div>
  );
}
