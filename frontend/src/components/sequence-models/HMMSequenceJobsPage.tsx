import { useMemo } from "react";
import {
  isHMMSequenceJobActive,
  useHMMSequenceJobs,
} from "@/api/sequenceModels";
import { HMMSequenceCreateForm } from "./HMMSequenceCreateForm";
import { HMMSequenceJobTablePanel } from "./HMMSequenceJobTable";

export function HMMSequenceJobsPage() {
  const { data: jobs = [], isLoading } = useHMMSequenceJobs();

  const { active, previous } = useMemo(() => {
    const a = jobs.filter(isHMMSequenceJobActive);
    const p = jobs.filter((j) => !isHMMSequenceJobActive(j));
    return { active: a, previous: p };
  }, [jobs]);

  if (isLoading) {
    return (
      <div className="space-y-6" data-testid="hmm-jobs-page">
        <HMMSequenceCreateForm />
        <div className="text-sm text-slate-500">Loading…</div>
      </div>
    );
  }

  return (
    <div className="space-y-6" data-testid="hmm-jobs-page">
      <HMMSequenceCreateForm />
      <HMMSequenceJobTablePanel
        title="Active Jobs"
        jobs={active}
        mode="active"
      />
      <HMMSequenceJobTablePanel
        title="Previous Jobs"
        jobs={previous}
        mode="previous"
      />
    </div>
  );
}
