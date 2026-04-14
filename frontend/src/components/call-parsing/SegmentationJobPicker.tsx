import { useMemo, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import {
  useSegmentationJobsWithCorrectionCounts,
  useCreateSegmentationTrainingDataset,
} from "@/hooks/queries/useCallParsing";
import { useHydrophones } from "@/hooks/queries/useClassifier";
import { toast } from "@/components/ui/use-toast";
import type { SegmentationJobWithCorrectionCount } from "@/api/types";

const PAGE_SIZE = 10;

function hydrophoneLabel(
  job: SegmentationJobWithCorrectionCount,
  hydrophones: { id: string; name: string }[],
): string {
  if (!job.hydrophone_id) return "—";
  const h = hydrophones.find((hp) => hp.id === job.hydrophone_id);
  return h?.name ?? job.hydrophone_id;
}

function dateRange(job: SegmentationJobWithCorrectionCount): string {
  if (job.start_timestamp == null || job.end_timestamp == null) return "—";
  const fmt = (ts: number) =>
    new Date(ts * 1000).toISOString().slice(0, 16).replace("T", " ") + "Z";
  return `${fmt(job.start_timestamp)} – ${fmt(job.end_timestamp)}`;
}

export function SegmentationJobPicker() {
  const { data: jobs = [] } = useSegmentationJobsWithCorrectionCounts();
  const { data: hydrophones = [] } = useHydrophones();
  const createDataset = useCreateSegmentationTrainingDataset();

  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [page, setPage] = useState(0);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");

  const completedJobs = useMemo(
    () => jobs.filter((j) => j.status === "complete"),
    [jobs],
  );

  const totalPages = Math.max(1, Math.ceil(completedJobs.length / PAGE_SIZE));
  const pageJobs = completedJobs.slice(
    page * PAGE_SIZE,
    (page + 1) * PAGE_SIZE,
  );

  const toggleJob = (jobId: string) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(jobId)) next.delete(jobId);
      else next.add(jobId);
      return next;
    });
  };

  const toggleAll = () => {
    const allPageIds = pageJobs.map((j) => j.id);
    const allSelected = allPageIds.every((id) => selected.has(id));
    setSelected((prev) => {
      const next = new Set(prev);
      for (const id of allPageIds) {
        if (allSelected) next.delete(id);
        else next.add(id);
      }
      return next;
    });
  };

  const handleCreate = () => {
    const ids = Array.from(selected);
    if (ids.length === 0) return;
    createDataset.mutate(
      {
        segmentation_job_ids: ids,
        name: name.trim() || undefined,
        description: description.trim() || undefined,
      },
      {
        onSuccess: (data) => {
          toast({
            title: "Training dataset created",
            description: `${data.sample_count} samples from ${ids.length} job${ids.length > 1 ? "s" : ""}`,
          });
          setSelected(new Set());
          setName("");
          setDescription("");
        },
        onError: (err) => {
          toast({
            title: "Failed to create dataset",
            description: (err as Error).message,
            variant: "destructive",
          });
        },
      },
    );
  };

  const allPageSelected =
    pageJobs.length > 0 && pageJobs.every((j) => selected.has(j.id));

  return (
    <div className="border rounded-md">
      <div className="flex items-center justify-between px-4 py-3 border-b">
        <div className="flex items-center gap-2">
          <h3 className="text-sm font-semibold">Create Training Dataset</h3>
          <Badge variant="secondary">
            {completedJobs.length} job{completedJobs.length !== 1 ? "s" : ""}
          </Badge>
        </div>
        {selected.size > 0 && (
          <span className="text-xs text-muted-foreground">
            {selected.size} selected
          </span>
        )}
      </div>

      {completedJobs.length === 0 ? (
        <div className="px-4 py-6 text-center text-sm text-muted-foreground">
          No completed segmentation jobs with corrections available.
        </div>
      ) : (
        <>
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b bg-muted/50">
                <th className="px-3 py-2 w-8">
                  <Checkbox
                    checked={allPageSelected}
                    onCheckedChange={toggleAll}
                    aria-label="Select all on this page"
                  />
                </th>
                <th className="px-3 py-2 text-left font-medium">Hydrophone</th>
                <th className="px-3 py-2 text-left font-medium">Date Range</th>
                <th className="px-3 py-2 text-right font-medium">
                  Corrections
                </th>
              </tr>
            </thead>
            <tbody>
              {pageJobs.map((job) => (
                <tr
                  key={job.id}
                  className="border-b hover:bg-muted/30 cursor-pointer"
                  onClick={() => toggleJob(job.id)}
                >
                  <td className="px-3 py-2">
                    <Checkbox
                      checked={selected.has(job.id)}
                      onCheckedChange={() => toggleJob(job.id)}
                      onClick={(e) => e.stopPropagation()}
                      aria-label={`Select job ${job.id.slice(0, 8)}`}
                    />
                  </td>
                  <td className="px-3 py-2 text-xs">
                    {hydrophoneLabel(job, hydrophones)}
                  </td>
                  <td className="px-3 py-2 text-xs text-muted-foreground whitespace-nowrap">
                    {dateRange(job)}
                  </td>
                  <td className="px-3 py-2 text-right font-mono text-xs">
                    {job.correction_count}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          {totalPages > 1 && (
            <div className="flex items-center justify-between px-4 py-2 border-t">
              <Button
                variant="outline"
                size="sm"
                disabled={page === 0}
                onClick={() => setPage((p) => p - 1)}
              >
                Prev
              </Button>
              <span className="text-xs text-muted-foreground">
                Page {page + 1} of {totalPages}
              </span>
              <Button
                variant="outline"
                size="sm"
                disabled={page >= totalPages - 1}
                onClick={() => setPage((p) => p + 1)}
              >
                Next
              </Button>
            </div>
          )}

          <div className="px-4 py-3 border-t space-y-2">
            <div className="flex gap-2">
              <Input
                placeholder="Dataset name (optional)"
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="max-w-xs text-sm"
              />
              <Input
                placeholder="Description (optional)"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                className="flex-1 text-sm"
              />
            </div>
            <Button
              size="sm"
              disabled={selected.size === 0 || createDataset.isPending}
              onClick={handleCreate}
            >
              {createDataset.isPending
                ? "Creating..."
                : `Create Training Dataset (${selected.size} job${selected.size !== 1 ? "s" : ""})`}
            </Button>
          </div>
        </>
      )}
    </div>
  );
}
