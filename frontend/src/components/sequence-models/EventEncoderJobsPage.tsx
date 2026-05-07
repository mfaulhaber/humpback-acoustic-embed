import { useMemo } from "react";
import {
  isEventEncoderJobActive,
  useEventEncoderJobs,
} from "@/api/sequenceModels";
import { EventEncoderCreateForm } from "./EventEncoderCreateForm";
import { EventEncoderJobTablePanel } from "./EventEncoderJobTable";

export function EventEncoderJobsPage() {
  const { data: jobs = [], isLoading } = useEventEncoderJobs();

  const { active, previous } = useMemo(() => {
    const a = jobs.filter(isEventEncoderJobActive);
    const p = jobs.filter((j) => !isEventEncoderJobActive(j));
    return { active: a, previous: p };
  }, [jobs]);

  if (isLoading) {
    return (
      <div className="space-y-6" data-testid="eej-jobs-page">
        <EventEncoderCreateForm />
        <div className="text-sm text-slate-500">Loading...</div>
      </div>
    );
  }

  return (
    <div className="space-y-6" data-testid="eej-jobs-page">
      <EventEncoderCreateForm />
      <EventEncoderJobTablePanel
        title="Active Jobs"
        jobs={active}
        mode="active"
      />
      <EventEncoderJobTablePanel
        title="Previous Jobs"
        jobs={previous}
        mode="previous"
      />
    </div>
  );
}
