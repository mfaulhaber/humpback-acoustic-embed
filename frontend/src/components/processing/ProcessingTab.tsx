import { useProcessingJobs, useEmbeddingSets } from "@/hooks/queries/useProcessing";
import { QueueJobForm } from "./QueueJobForm";
import { ProcessingJobsList } from "./ProcessingJobsList";
import { EmbeddingSetsList } from "./EmbeddingSetsList";

export function ProcessingTab() {
  const { data: jobs = [] } = useProcessingJobs(3000);
  const { data: embeddingSets = [] } = useEmbeddingSets(3000);

  return (
    <div className="space-y-4">
      <QueueJobForm jobs={jobs} />
      <ProcessingJobsList jobs={jobs} />
      <EmbeddingSetsList embeddingSets={embeddingSets} />
    </div>
  );
}
