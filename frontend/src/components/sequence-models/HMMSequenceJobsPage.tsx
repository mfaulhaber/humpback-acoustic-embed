import { useMemo } from "react";
import {
  isHMMSequenceJobActive,
  useHMMSequenceJobs,
} from "@/api/sequenceModels";
import { HMMSequenceCreateForm } from "./HMMSequenceCreateForm";
import { HMMSequenceJobCard } from "./HMMSequenceJobCard";

export function HMMSequenceJobsPage() {
  const { data: jobs = [], isLoading } = useHMMSequenceJobs();

  const { active, previous } = useMemo(() => {
    const a = jobs.filter(isHMMSequenceJobActive);
    const p = jobs.filter((j) => !isHMMSequenceJobActive(j));
    return { active: a, previous: p };
  }, [jobs]);

  return (
    <div className="space-y-6" data-testid="hmm-jobs-page">
      <HMMSequenceCreateForm />

      <section data-testid="hmm-active-section">
        <h2 className="text-base font-semibold mb-2">Active</h2>
        {isLoading ? (
          <div className="text-sm text-slate-500">Loading…</div>
        ) : active.length === 0 ? (
          <div className="text-sm text-slate-500">No active jobs.</div>
        ) : (
          <div className="grid gap-2">
            {active.map((j) => (
              <HMMSequenceJobCard key={j.id} job={j} />
            ))}
          </div>
        )}
      </section>

      <section data-testid="hmm-previous-section">
        <h2 className="text-base font-semibold mb-2">Previous</h2>
        {previous.length === 0 ? (
          <div className="text-sm text-slate-500">No previous jobs.</div>
        ) : (
          <div className="grid gap-2">
            {previous.map((j) => (
              <HMMSequenceJobCard key={j.id} job={j} />
            ))}
          </div>
        )}
      </section>
    </div>
  );
}
