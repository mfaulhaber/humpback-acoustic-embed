import { useCallback, useMemo } from "react";
import { useSearchParams } from "react-router-dom";
import {
  useClassificationJobs,
  useSegmentationJobs,
  useRegionDetectionJobs,
  useEventClassifierModels,
} from "@/hooks/queries/useCallParsing";
import { useHydrophones } from "@/hooks/queries/useClassifier";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ClassifyJobForm } from "./ClassifyJobForm";
import { ClassifyJobTablePanel } from "./ClassifyJobTable";
import { ClassifyReviewWorkspace } from "./ClassifyReviewWorkspace";

export function ClassifyPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const activeTab = searchParams.get("tab") ?? "jobs";
  const initialSegJobId = searchParams.get("segmentJobId");
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

  const onReview = useCallback(
    (jobId: string) => {
      setSearchParams((prev) => {
        const next = new URLSearchParams(prev);
        next.set("tab", "review");
        next.set("reviewJobId", jobId);
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
        <ClassifyJobsTab
          initialSegJobId={initialSegJobId}
          onReview={onReview}
        />
      </TabsContent>

      <TabsContent value="review">
        <ClassifyReviewWorkspace initialJobId={reviewJobId ?? undefined} />
      </TabsContent>
    </Tabs>
  );
}

function ClassifyJobsTab({
  initialSegJobId,
  onReview,
}: {
  initialSegJobId: string | null;
  onReview: (jobId: string) => void;
}) {
  const { data: classifyJobs = [] } = useClassificationJobs(3000);
  const { data: segJobs = [] } = useSegmentationJobs(3000);
  const { data: regionJobs = [] } = useRegionDetectionJobs();
  const { data: hydrophones = [] } = useHydrophones();
  const { data: models = [] } = useEventClassifierModels();

  const activeJobs = useMemo(
    () =>
      classifyJobs.filter(
        (j) => j.status === "queued" || j.status === "running",
      ),
    [classifyJobs],
  );

  const previousJobs = useMemo(
    () =>
      classifyJobs.filter(
        (j) => j.status === "complete" || j.status === "failed",
      ),
    [classifyJobs],
  );

  return (
    <div className="space-y-6">
      <ClassifyJobForm initialSegmentJobId={initialSegJobId} />

      <ClassifyJobTablePanel
        title="Active Jobs"
        jobs={activeJobs}
        segJobs={segJobs}
        regionJobs={regionJobs}
        hydrophones={hydrophones}
        models={models}
        mode="active"
      />

      <ClassifyJobTablePanel
        title="Previous Jobs"
        jobs={previousJobs}
        segJobs={segJobs}
        regionJobs={regionJobs}
        hydrophones={hydrophones}
        models={models}
        mode="previous"
        onReview={onReview}
      />
    </div>
  );
}
