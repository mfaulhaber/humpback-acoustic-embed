import { useMemo } from "react";
import { useSearchParams } from "react-router-dom";
import {
  useSegmentationJobs,
  useSegmentationModels,
  useRegionDetectionJobs,
} from "@/hooks/queries/useCallParsing";
import { useHydrophones } from "@/hooks/queries/useClassifier";
import { SegmentJobForm } from "./SegmentJobForm";
import { SegmentJobTablePanel } from "./SegmentJobTable";

export function SegmentPage() {
  const [searchParams] = useSearchParams();
  const initialRegionJobId = searchParams.get("regionJobId");

  const { data: hydrophones = [] } = useHydrophones();
  const { data: regionJobs = [] } = useRegionDetectionJobs(3000);
  const { data: segJobs = [] } = useSegmentationJobs(3000);
  const { data: models = [] } = useSegmentationModels();

  const activeJobs = useMemo(
    () => segJobs.filter((j) => j.status === "queued" || j.status === "running"),
    [segJobs],
  );

  const previousJobs = useMemo(
    () => segJobs.filter((j) => j.status === "complete" || j.status === "failed"),
    [segJobs],
  );

  return (
    <div className="space-y-6">
      <SegmentJobForm initialRegionJobId={initialRegionJobId} />

      <SegmentJobTablePanel
        title="Active Jobs"
        jobs={activeJobs}
        regionJobs={regionJobs}
        hydrophones={hydrophones}
        models={models}
        mode="active"
      />

      <SegmentJobTablePanel
        title="Previous Jobs"
        jobs={previousJobs}
        regionJobs={regionJobs}
        hydrophones={hydrophones}
        models={models}
        mode="previous"
      />
    </div>
  );
}
