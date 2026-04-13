import { useCallback, useMemo } from "react";
import { useSearchParams } from "react-router-dom";
import {
  useSegmentationJobs,
  useSegmentationModels,
  useRegionDetectionJobs,
} from "@/hooks/queries/useCallParsing";
import { useHydrophones } from "@/hooks/queries/useClassifier";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { SegmentJobForm } from "./SegmentJobForm";
import { SegmentJobTablePanel } from "./SegmentJobTable";
import { SegmentReviewWorkspace } from "./SegmentReviewWorkspace";

export function SegmentPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const activeTab = searchParams.get("tab") ?? "jobs";
  const initialRegionJobId = searchParams.get("regionJobId");
  const reviewJobId = searchParams.get("reviewJobId");

  const onTabChange = useCallback(
    (value: string) => {
      setSearchParams((prev) => {
        const next = new URLSearchParams(prev);
        next.set("tab", value);
        return next;
      });
    },
    [setSearchParams],
  );

  return (
    <Tabs value={activeTab} onValueChange={onTabChange}>
      <TabsList>
        <TabsTrigger value="jobs">Jobs</TabsTrigger>
        <TabsTrigger value="review">Review</TabsTrigger>
      </TabsList>

      <TabsContent value="jobs">
        <SegmentJobsTab initialRegionJobId={initialRegionJobId} />
      </TabsContent>

      <TabsContent value="review">
        <SegmentReviewWorkspace initialJobId={reviewJobId ?? undefined} />
      </TabsContent>
    </Tabs>
  );
}

function SegmentJobsTab({
  initialRegionJobId,
}: {
  initialRegionJobId: string | null;
}) {
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
