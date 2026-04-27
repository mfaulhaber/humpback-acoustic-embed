import { useMemo } from "react";
import {
  isContinuousEmbeddingJobActive,
  useContinuousEmbeddingJobs,
} from "@/api/sequenceModels";
import { ContinuousEmbeddingCreateForm } from "./ContinuousEmbeddingCreateForm";
import { ContinuousEmbeddingJobCard } from "./ContinuousEmbeddingJobCard";

export function ContinuousEmbeddingJobsPage() {
  const { data: jobs = [], isLoading } = useContinuousEmbeddingJobs();

  const { active, previous } = useMemo(() => {
    const a = jobs.filter(isContinuousEmbeddingJobActive);
    const p = jobs.filter((j) => !isContinuousEmbeddingJobActive(j));
    return { active: a, previous: p };
  }, [jobs]);

  return (
    <div className="space-y-6" data-testid="cej-jobs-page">
      <ContinuousEmbeddingCreateForm />

      <section data-testid="cej-active-section">
        <h2 className="text-base font-semibold mb-2">Active</h2>
        {isLoading ? (
          <div className="text-sm text-slate-500">Loading…</div>
        ) : active.length === 0 ? (
          <div className="text-sm text-slate-500">No active jobs.</div>
        ) : (
          <div className="grid gap-2">
            {active.map((j) => (
              <ContinuousEmbeddingJobCard key={j.id} job={j} />
            ))}
          </div>
        )}
      </section>

      <section data-testid="cej-previous-section">
        <h2 className="text-base font-semibold mb-2">Previous</h2>
        {previous.length === 0 ? (
          <div className="text-sm text-slate-500">No previous jobs.</div>
        ) : (
          <div className="grid gap-2">
            {previous.map((j) => (
              <ContinuousEmbeddingJobCard key={j.id} job={j} />
            ))}
          </div>
        )}
      </section>
    </div>
  );
}
