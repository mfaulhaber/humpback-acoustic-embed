import { useMemo } from "react";
import { useRegionDetectionJobs } from "@/hooks/queries/useCallParsing";
import { useHydrophones } from "@/hooks/queries/useClassifier";
import { RegionJobForm } from "./RegionJobForm";
import { RegionJobTablePanel } from "./RegionJobTable";

export function DetectionPage() {
  const { data: hydrophones = [] } = useHydrophones();
  const { data: jobs = [] } = useRegionDetectionJobs(3000);

  const activeJobs = useMemo(
    () => jobs.filter((j) => j.status === "queued" || j.status === "running"),
    [jobs],
  );

  const previousJobs = useMemo(
    () =>
      jobs.filter(
        (j) =>
          j.status === "complete" ||
          j.status === "failed",
      ),
    [jobs],
  );

  return (
    <div className="space-y-6">
      <RegionJobForm />

      <RegionJobTablePanel
        title="Active Jobs"
        jobs={activeJobs}
        hydrophones={hydrophones}
        mode="active"
      />

      <RegionJobTablePanel
        title="Previous Jobs"
        jobs={previousJobs}
        hydrophones={hydrophones}
        mode="previous"
      />
    </div>
  );
}
