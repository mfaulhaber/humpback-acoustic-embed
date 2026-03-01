import { useMemo } from "react";
import { useClusteringJobs } from "@/hooks/queries/useClustering";
import { useEmbeddingSets } from "@/hooks/queries/useProcessing";
import { EmbeddingSetSelector } from "./EmbeddingSetSelector";
import { ClusteringJobCard } from "./ClusteringJobCard";

export function ClusteringTab() {
  const { data: embeddingSets = [] } = useEmbeddingSets();
  const { data: jobs = [] } = useClusteringJobs(3000);

  // Sort newest first
  const sortedJobs = useMemo(
    () => [...jobs].sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime()),
    [jobs],
  );

  return (
    <div className="space-y-4">
      <EmbeddingSetSelector embeddingSets={embeddingSets} />
      {sortedJobs.map((job) => (
        <ClusteringJobCard key={job.id} job={job} />
      ))}
      {sortedJobs.length === 0 && (
        <p className="text-sm text-muted-foreground text-center py-8">No clustering jobs yet.</p>
      )}
    </div>
  );
}
