import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  useClassificationJobsWithCorrectionCounts,
  useCreateClassifierTrainingJob,
} from "@/hooks/queries/useCallParsing";
import { useHydrophones } from "@/hooks/queries/useClassifier";
import { toast } from "@/components/ui/use-toast";
import type { ClassificationJobWithCorrectionCount } from "@/api/types";
import { formatUtcShort } from "@/utils/format";

function sourceLabel(
  job: ClassificationJobWithCorrectionCount,
  hydrophones: { id: string; name: string }[],
): string {
  if (job.hydrophone_id) {
    const h = hydrophones.find((hp) => hp.id === job.hydrophone_id);
    return h?.name ?? job.hydrophone_id;
  }
  return "file";
}

export function ClassificationJobPicker() {
  const { data: jobs = [] } = useClassificationJobsWithCorrectionCounts();
  const { data: hydrophones = [] } = useHydrophones();
  const createTraining = useCreateClassifierTrainingJob();
  const [selected, setSelected] = useState<Set<string>>(new Set());

  const jobsWithCorrections = jobs.filter((j) => j.correction_count > 0);

  const toggleSelect = (id: string) =>
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  const toggleAll = () =>
    setSelected((prev) =>
      prev.size === jobsWithCorrections.length
        ? new Set()
        : new Set(jobsWithCorrections.map((j) => j.id)),
    );

  const handleTrain = () => {
    if (selected.size === 0) return;
    createTraining.mutate(
      { source_job_ids: Array.from(selected) },
      {
        onSuccess: () => {
          setSelected(new Set());
          toast({
            title: "Training job created",
            description: `Training from ${selected.size} classification job${selected.size !== 1 ? "s" : ""}.`,
          });
        },
        onError: (err) => {
          toast({
            title: "Failed to create training job",
            description: (err as Error).message,
            variant: "destructive",
          });
        },
      },
    );
  };

  return (
    <div className="border rounded-md">
      <div className="flex items-center justify-between px-4 py-3 border-b">
        <h3 className="text-sm font-semibold">
          Train from Corrections
        </h3>
        <Button
          size="sm"
          onClick={handleTrain}
          disabled={selected.size === 0 || createTraining.isPending}
        >
          {createTraining.isPending
            ? "Creating…"
            : `Train Model${selected.size > 0 ? ` (${selected.size})` : ""}`}
        </Button>
      </div>

      {jobsWithCorrections.length === 0 ? (
        <div className="px-4 py-6 text-center text-sm text-muted-foreground">
          No classification jobs with corrections yet.
        </div>
      ) : (
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b bg-muted/50">
              <th className="w-8 px-2 py-2">
                <Checkbox
                  checked={
                    selected.size === jobsWithCorrections.length &&
                    jobsWithCorrections.length > 0
                  }
                  onCheckedChange={toggleAll}
                />
              </th>
              <th className="px-3 py-2 text-left font-medium">Source</th>
              <th className="px-3 py-2 text-left font-medium">Date Range</th>
              <th className="px-3 py-2 text-right font-medium">Corrections</th>
            </tr>
          </thead>
          <tbody>
            {jobsWithCorrections.map((job) => (
              <tr key={job.id} className="border-b hover:bg-muted/30">
                <td className="w-8 px-2 py-2">
                  <Checkbox
                    checked={selected.has(job.id)}
                    onCheckedChange={() => toggleSelect(job.id)}
                  />
                </td>
                <td className="px-3 py-2">
                  {sourceLabel(job, hydrophones)}
                  <span className="text-xs text-muted-foreground ml-2">
                    {job.id.slice(0, 8)}
                  </span>
                </td>
                <td className="px-3 py-2 text-xs text-muted-foreground">
                  {job.start_timestamp != null && job.end_timestamp != null
                    ? `${formatUtcShort(job.start_timestamp)} – ${formatUtcShort(job.end_timestamp)}`
                    : "—"}
                </td>
                <td className="px-3 py-2 text-right font-medium">
                  {job.correction_count}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
