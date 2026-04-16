import { useState, useMemo } from "react";
import { ChevronDown } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { StatusBadge } from "@/components/shared/StatusBadge";
import { ComputeDeviceBadge } from "@/components/shared/ComputeDeviceBadge";
import { BulkDeleteDialog } from "@/components/classifier/BulkDeleteDialog";
import {
  useDeleteClassificationJob,
  useTypedEvents,
} from "@/hooks/queries/useCallParsing";
import type {
  EventClassificationJob,
  EventSegmentationJob,
  RegionDetectionJob,
  HydrophoneInfo,
  EventClassifierModel,
  TypedEventRow,
} from "@/api/types";
import { formatUtcShort } from "@/utils/format";

function hydrophoneName(
  hydrophoneId: string | null,
  hydrophones: HydrophoneInfo[],
): string {
  if (!hydrophoneId) return "—";
  const h = hydrophones.find((hp) => hp.id === hydrophoneId);
  return h ? h.name : hydrophoneId;
}

function sourceLabel(
  job: EventClassificationJob,
  segJobs: EventSegmentationJob[],
  regionJobs: RegionDetectionJob[],
  hydrophones: HydrophoneInfo[],
): string {
  const shortId = job.id.slice(0, 8);
  const sj = segJobs.find((s) => s.id === job.event_segmentation_job_id);
  if (!sj) return shortId;
  const rj = regionJobs.find((r) => r.id === sj.region_detection_job_id);
  if (!rj) return shortId;
  const name = hydrophoneName(rj.hydrophone_id, hydrophones);
  if (rj.start_timestamp != null && rj.end_timestamp != null) {
    return `${name} · ${formatUtcShort(rj.start_timestamp)}–${formatUtcShort(rj.end_timestamp)} — ${shortId}`;
  }
  return `${name} — ${shortId}`;
}

function modelName(
  modelId: string | null,
  models: EventClassifierModel[],
): string {
  if (!modelId) return "—";
  const m = models.find((x) => x.id === modelId);
  return m ? m.name : modelId.slice(0, 8);
}

/** Aggregate typed events into per-type summary */
function typeSummary(
  rows: TypedEventRow[],
): { type_name: string; count: number; mean_score: number; pct: number }[] {
  // Deduplicate events: for each event_id pick the best above_threshold type
  const eventBest = new Map<
    string,
    { type_name: string; score: number }
  >();
  for (const r of rows) {
    if (!r.above_threshold) continue;
    const existing = eventBest.get(r.event_id);
    if (!existing || r.score > existing.score) {
      eventBest.set(r.event_id, { type_name: r.type_name, score: r.score });
    }
  }
  const totalEvents = eventBest.size;
  const byType = new Map<string, { count: number; scoreSum: number }>();
  for (const { type_name, score } of eventBest.values()) {
    const cur = byType.get(type_name) ?? { count: 0, scoreSum: 0 };
    cur.count++;
    cur.scoreSum += score;
    byType.set(type_name, cur);
  }
  return Array.from(byType.entries())
    .map(([type_name, { count, scoreSum }]) => ({
      type_name,
      count,
      mean_score: scoreSum / count,
      pct: totalEvents > 0 ? (count / totalEvents) * 100 : 0,
    }))
    .sort((a, b) => b.count - a.count);
}

function ExpandableRow({ jobId }: { jobId: string }) {
  const { data: typedEvents = [] } = useTypedEvents(jobId);
  const summary = useMemo(() => typeSummary(typedEvents), [typedEvents]);

  if (summary.length === 0) {
    return (
      <p className="text-xs text-slate-500 px-4 py-2">
        No typed events available
      </p>
    );
  }

  return (
    <div className="px-4 py-2">
      <table className="w-full text-xs">
        <thead>
          <tr className="text-slate-500">
            <th className="text-left py-1">Type</th>
            <th className="text-right py-1">Count</th>
            <th className="text-right py-1">Mean Score</th>
            <th className="text-right py-1">% of Events</th>
          </tr>
        </thead>
        <tbody>
          {summary.map((s) => (
            <tr key={s.type_name} className="border-t border-slate-100">
              <td className="py-1">
                <Badge variant="outline" className="text-xs">
                  {s.type_name}
                </Badge>
              </td>
              <td className="text-right py-1">{s.count}</td>
              <td className="text-right py-1">{s.mean_score.toFixed(2)}</td>
              <td className="text-right py-1">{s.pct.toFixed(1)}%</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

interface ClassifyJobTableProps {
  title: string;
  jobs: EventClassificationJob[];
  segJobs: EventSegmentationJob[];
  regionJobs: RegionDetectionJob[];
  hydrophones: HydrophoneInfo[];
  models: EventClassifierModel[];
  mode: "active" | "previous";
  onReview?: (jobId: string) => void;
}

export function ClassifyJobTablePanel({
  title,
  jobs,
  segJobs,
  regionJobs,
  hydrophones,
  models,
  mode,
  onReview,
}: ClassifyJobTableProps) {
  const deleteMutation = useDeleteClassificationJob();
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [showBulkDelete, setShowBulkDelete] = useState(false);

  const toggleExpand = (id: string) =>
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  const toggleSelect = (id: string) =>
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  const toggleAll = () =>
    setSelected((prev) =>
      prev.size === jobs.length ? new Set() : new Set(jobs.map((j) => j.id)),
    );

  if (jobs.length === 0) {
    return (
      <div className="border rounded-lg p-4">
        <h3 className="text-sm font-medium text-slate-500">{title}</h3>
        <p className="text-xs text-slate-400 mt-1">No jobs</p>
      </div>
    );
  }

  return (
    <div className="border rounded-lg">
      <div className="flex items-center justify-between px-4 py-2 border-b">
        <h3 className="text-sm font-medium">{title}</h3>
        {mode === "previous" && selected.size > 0 && (
          <Button
            variant="destructive"
            size="sm"
            onClick={() => setShowBulkDelete(true)}
          >
            Delete {selected.size}
          </Button>
        )}
      </div>
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b text-slate-500 text-xs">
            {mode === "previous" && (
              <th className="w-8 px-2 py-2">
                <Checkbox
                  checked={selected.size === jobs.length && jobs.length > 0}
                  onCheckedChange={toggleAll}
                />
              </th>
            )}
            <th className="text-left px-4 py-2">Source</th>
            <th className="text-left px-4 py-2">Model</th>
            <th className="text-left px-4 py-2">Status</th>
            <th className="text-right px-4 py-2">Events</th>
            <th className="text-right px-4 py-2">Created</th>
            <th className="text-right px-4 py-2">Actions</th>
          </tr>
        </thead>
        <tbody>
          {jobs.map((job) => (
            <JobRow
              key={job.id}
              job={job}
              segJobs={segJobs}
              regionJobs={regionJobs}
              hydrophones={hydrophones}
              models={models}
              mode={mode}
              isExpanded={expanded.has(job.id)}
              isSelected={selected.has(job.id)}
              onToggleExpand={() => toggleExpand(job.id)}
              onToggleSelect={() => toggleSelect(job.id)}
              onReview={onReview}
              onDelete={() => deleteMutation.mutate(job.id)}
            />
          ))}
        </tbody>
      </table>
      <BulkDeleteDialog
        open={showBulkDelete}
        onOpenChange={setShowBulkDelete}
        count={selected.size}
        entityName="job"
        isPending={deleteMutation.isPending}
        onConfirm={async () => {
          for (const id of selected) {
            await deleteMutation.mutateAsync(id);
          }
          setSelected(new Set());
          setShowBulkDelete(false);
        }}
      />
    </div>
  );
}

function JobRow({
  job,
  segJobs,
  regionJobs,
  hydrophones,
  models,
  mode,
  isExpanded,
  isSelected,
  onToggleExpand,
  onToggleSelect,
  onReview,
  onDelete,
}: {
  job: EventClassificationJob;
  segJobs: EventSegmentationJob[];
  regionJobs: RegionDetectionJob[];
  hydrophones: HydrophoneInfo[];
  models: EventClassifierModel[];
  mode: "active" | "previous";
  isExpanded: boolean;
  isSelected: boolean;
  onToggleExpand: () => void;
  onToggleSelect: () => void;
  onReview?: (jobId: string) => void;
  onDelete: () => void;
}) {
  return (
    <>
      <tr className="border-b hover:bg-slate-50">
        {mode === "previous" && (
          <td className="w-8 px-2 py-2">
            <Checkbox checked={isSelected} onCheckedChange={onToggleSelect} />
          </td>
        )}
        <td className="px-4 py-2">
          {sourceLabel(job, segJobs, regionJobs, hydrophones)}
        </td>
        <td className="px-4 py-2">
          {modelName(job.vocalization_model_id, models)}
        </td>
        <td className="px-4 py-2">
          <div className="flex items-center gap-1.5 flex-wrap">
            <StatusBadge status={job.status} />
            {mode === "active" && (
              <ComputeDeviceBadge
                device={job.compute_device}
                fallbackReason={job.gpu_fallback_reason}
              />
            )}
          </div>
        </td>
        <td className="text-right px-4 py-2">
          {job.typed_event_count ?? "—"}
        </td>
        <td className="text-right px-4 py-2 text-xs text-slate-500">
          {new Date(job.created_at).toLocaleDateString()}
        </td>
        <td className="text-right px-4 py-2 space-x-1">
          {mode === "previous" && job.status === "complete" && (
            <>
              <Button
                variant="ghost"
                size="sm"
                className="h-7 text-xs"
                onClick={onToggleExpand}
              >
                <ChevronDown
                  className={`h-3 w-3 transition-transform ${isExpanded ? "rotate-180" : ""}`}
                />
              </Button>
              {onReview && (
                <Button
                  variant="outline"
                  size="sm"
                  className="h-7 text-xs"
                  onClick={() => onReview(job.id)}
                >
                  Review
                </Button>
              )}
            </>
          )}
          <Button
            variant="ghost"
            size="sm"
            className="h-7 text-xs text-red-600 hover:text-red-700"
            onClick={onDelete}
          >
            Delete
          </Button>
        </td>
      </tr>
      {isExpanded && job.status === "complete" && (
        <tr>
          <td colSpan={mode === "previous" ? 7 : 6} className="bg-slate-50">
            <ExpandableRow jobId={job.id} />
          </td>
        </tr>
      )}
    </>
  );
}
