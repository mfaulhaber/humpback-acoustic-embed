import { useCallback, useMemo } from "react";
import { useSearchParams } from "react-router-dom";
import {
  useWindowClassificationJobs,
  useRegionDetectionJobs,
} from "@/hooks/queries/useCallParsing";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { WindowClassifyJobForm } from "./WindowClassifyJobForm";
import { WindowClassifyJobTable } from "./WindowClassifyJobTable";
import { WindowClassifyReviewWorkspace } from "./WindowClassifyReviewWorkspace";

export function WindowClassifyPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const activeTab = searchParams.get("tab") ?? "jobs";
  const reviewJobId = searchParams.get("jobId");

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
        next.set("jobId", jobId);
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
        <WindowClassifyJobsTab onReview={onReview} />
      </TabsContent>

      <TabsContent value="review">
        <WindowClassifyReviewWorkspace
          initialJobId={reviewJobId ?? undefined}
        />
      </TabsContent>
    </Tabs>
  );
}

function WindowClassifyJobsTab({
  onReview,
}: {
  onReview: (jobId: string) => void;
}) {
  const { data: wcJobs = [] } = useWindowClassificationJobs(3000);
  const { data: regionJobs = [] } = useRegionDetectionJobs();

  const activeJobs = useMemo(
    () =>
      wcJobs.filter(
        (j) => j.status === "queued" || j.status === "running",
      ),
    [wcJobs],
  );
  const previousJobs = useMemo(
    () =>
      wcJobs.filter(
        (j) => j.status === "complete" || j.status === "failed",
      ),
    [wcJobs],
  );

  return (
    <div className="space-y-6">
      <WindowClassifyJobForm />
      <WindowClassifyJobTable
        jobs={activeJobs}
        regionJobs={regionJobs}
        title="Active Jobs"
      />
      <WindowClassifyJobTable
        jobs={previousJobs}
        regionJobs={regionJobs}
        title="Previous Jobs"
        showReview
        onReview={onReview}
      />
    </div>
  );
}
