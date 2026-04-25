import { useMemo } from "react";
import { useVocalizationClusteringJobs } from "@/hooks/queries/useVocalization";
import { VocalizationClusteringForm } from "./VocalizationClusteringForm";
import { VocalizationClusteringJobTablePanel } from "./VocalizationClusteringJobTable";

export function VocalizationClusteringPage() {
  const { data: jobs = [] } = useVocalizationClusteringJobs(3000);

  const activeJobs = useMemo(
    () => jobs.filter((j) => j.status === "queued" || j.status === "running"),
    [jobs],
  );

  const previousJobs = useMemo(
    () => jobs.filter((j) => j.status === "complete" || j.status === "failed"),
    [jobs],
  );

  return (
    <div className="space-y-6">
      <VocalizationClusteringForm />

      <VocalizationClusteringJobTablePanel
        title="Active Jobs"
        jobs={activeJobs}
        mode="active"
      />

      <VocalizationClusteringJobTablePanel
        title="Previous Jobs"
        jobs={previousJobs}
        mode="previous"
      />
    </div>
  );
}
