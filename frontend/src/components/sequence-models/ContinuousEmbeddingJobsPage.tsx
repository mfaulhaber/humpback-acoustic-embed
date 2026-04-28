import { useMemo } from "react";
import {
  isContinuousEmbeddingJobActive,
  useContinuousEmbeddingJobs,
} from "@/api/sequenceModels";
import { ContinuousEmbeddingCreateForm } from "./ContinuousEmbeddingCreateForm";
import { ContinuousEmbeddingJobTablePanel } from "./ContinuousEmbeddingJobTable";

export function ContinuousEmbeddingJobsPage() {
  const { data: jobs = [], isLoading } = useContinuousEmbeddingJobs();

  const { active, previous } = useMemo(() => {
    const a = jobs.filter(isContinuousEmbeddingJobActive);
    const p = jobs.filter((j) => !isContinuousEmbeddingJobActive(j));
    return { active: a, previous: p };
  }, [jobs]);

  if (isLoading) {
    return (
      <div className="space-y-6" data-testid="cej-jobs-page">
        <ContinuousEmbeddingCreateForm />
        <div className="text-sm text-slate-500">Loading…</div>
      </div>
    );
  }

  return (
    <div className="space-y-6" data-testid="cej-jobs-page">
      <ContinuousEmbeddingCreateForm />
      <ContinuousEmbeddingJobTablePanel
        title="Active Jobs"
        jobs={active}
        mode="active"
      />
      <ContinuousEmbeddingJobTablePanel
        title="Previous Jobs"
        jobs={previous}
        mode="previous"
      />
    </div>
  );
}
