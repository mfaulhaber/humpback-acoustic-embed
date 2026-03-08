import { useState, useMemo } from "react";
import { useProcessingJobs, useEmbeddingSets } from "@/hooks/queries/useProcessing";
import { QueueJobForm } from "./QueueJobForm";
import { ProcessingJobsList } from "./ProcessingJobsList";
import { EmbeddingSetsList } from "./EmbeddingSetsList";
import { ModelFilter } from "@/components/shared/ModelFilter";

export function ProcessingTab() {
  const { data: jobs = [] } = useProcessingJobs(3000);
  const { data: embeddingSets = [] } = useEmbeddingSets(3000);
  const [modelFilter, setModelFilter] = useState("__all__");

  const allItems = useMemo(() => {
    const jobModels = jobs.filter((j) => j.model_version).map((j) => ({ model_version: j.model_version! }));
    const esModels = embeddingSets.map((es) => ({ model_version: es.model_version }));
    return [...jobModels, ...esModels];
  }, [jobs, embeddingSets]);

  const filteredJobs = useMemo(
    () => modelFilter === "__all__" ? jobs : jobs.filter((j) => j.model_version === modelFilter),
    [jobs, modelFilter],
  );
  const filteredSets = useMemo(
    () => modelFilter === "__all__" ? embeddingSets : embeddingSets.filter((es) => es.model_version === modelFilter),
    [embeddingSets, modelFilter],
  );

  return (
    <div className="space-y-4">
      <QueueJobForm jobs={jobs} onModelUsed={setModelFilter} />
      <ModelFilter items={allItems} value={modelFilter} onChange={setModelFilter} />
      <ProcessingJobsList jobs={filteredJobs} />
      <EmbeddingSetsList embeddingSets={filteredSets} />
    </div>
  );
}
