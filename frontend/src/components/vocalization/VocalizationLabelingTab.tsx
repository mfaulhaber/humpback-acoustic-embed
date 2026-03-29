import { useState, useCallback } from "react";
import { DetectionJobPicker } from "./DetectionJobPicker";
import { EmbeddingStatusPanel } from "./EmbeddingStatusPanel";
import { InferencePanel } from "./InferencePanel";
import { LabelingWorkspace } from "./LabelingWorkspace";
import { RetrainFooter } from "./RetrainFooter";

export function VocalizationLabelingTab() {
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [embeddingsReady, setEmbeddingsReady] = useState(false);
  const [inferenceJobId, setInferenceJobId] = useState<string | null>(null);
  const [labelCount, setLabelCount] = useState(0);

  const handleSelectJob = useCallback((jobId: string | null) => {
    setSelectedJobId(jobId);
    setEmbeddingsReady(false);
    setInferenceJobId(null);
    setLabelCount(0);
  }, []);

  const handleEmbeddingsReady = useCallback(() => {
    setEmbeddingsReady(true);
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
          onReady={handleEmbeddingsReady}
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
