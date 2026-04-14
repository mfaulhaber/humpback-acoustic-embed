import { useState, useMemo, useEffect, useCallback } from "react";
import { Link } from "react-router-dom";
import { SegmentJobDetail } from "./SegmentJobDetail";
import {
  ArrowUp,
  ArrowDown,
  ChevronLeft,
  ChevronRight,
  ChevronDown,
  Search,
  X,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { StatusBadge } from "@/components/shared/StatusBadge";
import { BulkDeleteDialog } from "@/components/classifier/BulkDeleteDialog";
import { useDeleteSegmentationJob } from "@/hooks/queries/useCallParsing";
import type {
  EventSegmentationJob,
  RegionDetectionJob,
  HydrophoneInfo,
  SegmentationModel,
} from "@/api/types";

function hydrophoneName(
  hydrophoneId: string | null,
  hydrophones: HydrophoneInfo[],
): string {
  if (!hydrophoneId) return "—";
  const h = hydrophones.find((hp) => hp.id === hydrophoneId);
  return h ? h.name : hydrophoneId;
}

function formatUtcShort(epoch: number): string {
  const d = new Date(epoch * 1000);
  const months = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
  ];
  return `${months[d.getUTCMonth()]} ${d.getUTCDate()}`;
}

function sourceLabel(
  job: EventSegmentationJob,
  regionJobs: RegionDetectionJob[],
  hydrophones: HydrophoneInfo[],
): string {
  const shortId = job.id.slice(0, 8);
  const rj = regionJobs.find((r) => r.id === job.region_detection_job_id);
  if (!rj) return shortId;
  const name = hydrophoneName(rj.hydrophone_id, hydrophones);
  if (rj.start_timestamp != null && rj.end_timestamp != null) {
    return `${name} · ${formatUtcShort(rj.start_timestamp)}–${formatUtcShort(rj.end_timestamp)} - ${shortId}`;
  }
  return `${name} - ${shortId}`;
}

function modelName(
  job: EventSegmentationJob,
  models: SegmentationModel[],
): string {
  if (!job.segmentation_model_id) return "—";
  const m = models.find((mo) => mo.id === job.segmentation_model_id);
  return m ? m.name : job.segmentation_model_id.slice(0, 8);
}

function parseThresholds(
  job: EventSegmentationJob,
): { high: number | null; low: number | null } {
  if (!job.config_json) return { high: null, low: null };
  try {
    const cfg = JSON.parse(job.config_json) as Record<string, unknown>;
    return {
      high:
        typeof cfg.high_threshold === "number" ? cfg.high_threshold : null,
      low: typeof cfg.low_threshold === "number" ? cfg.low_threshold : null,
    };
  } catch {
    return { high: null, low: null };
  }
}

type SortKey = "status" | "created" | "source" | "model" | "events";
type SortDir = "asc" | "desc";

interface SegmentJobTableProps {
  jobs: EventSegmentationJob[];
  regionJobs: RegionDetectionJob[];
  hydrophones: HydrophoneInfo[];
  models: SegmentationModel[];
  mode: "active" | "previous";
}

export function SegmentJobTable({
  jobs,
  regionJobs,
  hydrophones,
  models,
  mode,
}: SegmentJobTableProps) {
  const deleteMutation = useDeleteSegmentationJob();

  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);
  const [bulkDeleting, setBulkDeleting] = useState(false);
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const [filterText, setFilterText] = useState("");
  const [sortKey, setSortKey] = useState<SortKey>("created");
  const [sortDir, setSortDir] = useState<SortDir>("desc");
  const [page, setPage] = useState(1);
  const PAGE_SIZE = 20;

  const toggleSort = useCallback(
    (key: SortKey) => {
      if (sortKey === key) {
        setSortDir((d) => (d === "asc" ? "desc" : "asc"));
      } else {
        setSortKey(key);
        setSortDir("asc");
      }
    },
    [sortKey],
  );

  const filteredJobs = useMemo(() => {
    if (mode === "active" || !filterText) return jobs;
    const q = filterText.toLowerCase();
    return jobs.filter((j) =>
      sourceLabel(j, regionJobs, hydrophones).toLowerCase().includes(q),
    );
  }, [jobs, filterText, regionJobs, hydrophones, mode]);

  const sortedJobs = useMemo(() => {
    if (mode === "active") return filteredJobs;
    const sorted = [...filteredJobs];
    const dir = sortDir === "asc" ? 1 : -1;
    sorted.sort((a, b) => {
      switch (sortKey) {
        case "status":
          return dir * a.status.localeCompare(b.status);
        case "created":
          return (
            dir *
            (new Date(a.created_at).getTime() -
              new Date(b.created_at).getTime())
          );
        case "source":
          return (
            dir *
            sourceLabel(a, regionJobs, hydrophones).localeCompare(
              sourceLabel(b, regionJobs, hydrophones),
            )
          );
        case "model":
          return (
            dir *
            modelName(a, models).localeCompare(modelName(b, models))
          );
        case "events":
          return dir * ((a.event_count ?? 0) - (b.event_count ?? 0));
        default:
          return 0;
      }
    });
    return sorted;
  }, [filteredJobs, sortKey, sortDir, regionJobs, hydrophones, models, mode]);

  const totalPages = Math.max(1, Math.ceil(sortedJobs.length / PAGE_SIZE));
  const effectivePage = Math.min(page, totalPages);

  const paginatedJobs = useMemo(() => {
    if (mode === "active") return sortedJobs;
    const start = (effectivePage - 1) * PAGE_SIZE;
    return sortedJobs.slice(start, start + PAGE_SIZE);
  }, [sortedJobs, effectivePage, mode]);

  useEffect(() => {
    setPage(1);
  }, [filterText, sortKey, sortDir]);

  useEffect(() => {
    setSelectedIds((prev) => {
      const jobIds = new Set(jobs.map((j) => j.id));
      const next = new Set([...prev].filter((id) => jobIds.has(id)));
      return next.size === prev.size ? prev : next;
    });
  }, [jobs]);

  const handleBulkDelete = async () => {
    setBulkDeleting(true);
    try {
      for (const id of selectedIds) {
        await deleteMutation.mutateAsync(id);
      }
      setSelectedIds(new Set());
      setShowDeleteDialog(false);
    } finally {
      setBulkDeleting(false);
    }
  };

  const SortIcon = ({ col }: { col: SortKey }) => {
    if (sortKey !== col) return null;
    return sortDir === "asc" ? (
      <ArrowUp className="h-3 w-3" />
    ) : (
      <ArrowDown className="h-3 w-3" />
    );
  };

  const sortableHeader = (label: string, col: SortKey) => (
    <th
      className="px-3 py-2 text-left font-medium cursor-pointer select-none hover:bg-muted/80"
      onClick={() => toggleSort(col)}
    >
      <span className="inline-flex items-center gap-1">
        {label}
        <SortIcon col={col} />
      </span>
    </th>
  );

  const colCount = mode === "active" ? 6 : 8;

  return (
    <>
      {mode === "previous" && (
        <div className="flex items-center justify-between px-4 py-2 border-b bg-muted/30">
          <div className="flex items-center gap-2">
            <div className="relative">
              <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
              <Input
                type="search"
                placeholder="Filter by source…"
                value={filterText}
                onChange={(e) => setFilterText(e.target.value)}
                className="h-8 w-64 pl-8 text-xs"
                autoComplete="off"
              />
            </div>
            <Button
              variant="destructive"
              size="sm"
              disabled={selectedIds.size === 0}
              onClick={() => setShowDeleteDialog(true)}
            >
              Delete ({selectedIds.size})
            </Button>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-muted-foreground text-xs">
              {sortedJobs.length > 0
                ? `${(effectivePage - 1) * PAGE_SIZE + 1}–${Math.min(effectivePage * PAGE_SIZE, sortedJobs.length)} of ${sortedJobs.length}`
                : "0 items"}
            </span>
            <Button
              variant="outline"
              size="sm"
              className="h-7 px-2"
              disabled={effectivePage <= 1}
              onClick={() => setPage((p) => Math.max(1, p - 1))}
            >
              <ChevronLeft className="h-3.5 w-3.5" />
            </Button>
            <Button
              variant="outline"
              size="sm"
              className="h-7 px-2"
              disabled={effectivePage >= totalPages}
              onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
            >
              <ChevronRight className="h-3.5 w-3.5" />
            </Button>
          </div>
        </div>
      )}

      <table className="w-full text-sm">
        <thead>
          <tr className="border-b bg-muted/50">
            {mode === "previous" && (
              <th className="w-10 px-3 py-2">
                <Checkbox
                  checked={
                    paginatedJobs.length > 0 &&
                    paginatedJobs.every((j) => selectedIds.has(j.id))
                      ? true
                      : paginatedJobs.some((j) => selectedIds.has(j.id))
                        ? "indeterminate"
                        : false
                  }
                  onCheckedChange={() => {
                    const allSel = paginatedJobs.every((j) =>
                      selectedIds.has(j.id),
                    );
                    if (allSel) {
                      setSelectedIds((prev) => {
                        const next = new Set(prev);
                        paginatedJobs.forEach((j) => next.delete(j.id));
                        return next;
                      });
                    } else {
                      setSelectedIds((prev) => {
                        const next = new Set(prev);
                        paginatedJobs.forEach((j) => next.add(j.id));
                        return next;
                      });
                    }
                  }}
                />
              </th>
            )}
            {mode === "previous"
              ? sortableHeader("Status", "status")
              : (
                  <th className="px-3 py-2 text-left font-medium">Status</th>
                )}
            {mode === "previous"
              ? sortableHeader("Created", "created")
              : (
                  <th className="px-3 py-2 text-left font-medium">Created</th>
                )}
            {mode === "previous"
              ? sortableHeader("Source", "source")
              : (
                  <th className="px-3 py-2 text-left font-medium">Source</th>
                )}
            {mode === "previous"
              ? sortableHeader("Model", "model")
              : (
                  <th className="px-3 py-2 text-left font-medium">Model</th>
                )}
            {mode === "previous"
              ? sortableHeader("Events", "events")
              : (
                  <th className="px-3 py-2 text-left font-medium">Events</th>
                )}
            {mode === "previous" && (
              <th className="px-3 py-2 text-left font-medium">Thresholds</th>
            )}
            <th className="px-3 py-2 text-left font-medium">
              {mode === "active" ? "Actions" : ""}
            </th>
          </tr>
        </thead>
        <tbody>
          {paginatedJobs.map((job) => {
            const isExpanded = expandedId === job.id;
            return (
              <SegmentJobRow
                key={job.id}
                job={job}
                regionJobs={regionJobs}
                hydrophones={hydrophones}
                models={models}
                mode={mode}
                isExpanded={isExpanded}
                onToggleExpand={() =>
                  setExpandedId(isExpanded ? null : job.id)
                }
                onDelete={() => deleteMutation.mutate(job.id)}
                deleteDisabled={deleteMutation.isPending}
                selected={selectedIds.has(job.id)}
                onToggleSelect={() =>
                  setSelectedIds((prev) => {
                    const next = new Set(prev);
                    if (next.has(job.id)) next.delete(job.id);
                    else next.add(job.id);
                    return next;
                  })
                }
                colCount={colCount}
              />
            );
          })}
          {paginatedJobs.length === 0 && (
            <tr>
              <td
                colSpan={colCount + (mode === "previous" ? 1 : 0)}
                className="px-3 py-4 text-center text-muted-foreground text-xs"
              >
                No jobs found.
              </td>
            </tr>
          )}
        </tbody>
      </table>

      {mode === "previous" && (
        <BulkDeleteDialog
          open={showDeleteDialog}
          onOpenChange={setShowDeleteDialog}
          count={selectedIds.size}
          entityName="segmentation job"
          onConfirm={handleBulkDelete}
          isPending={bulkDeleting}
        />
      )}
    </>
  );
}

interface SegmentJobRowProps {
  job: EventSegmentationJob;
  regionJobs: RegionDetectionJob[];
  hydrophones: HydrophoneInfo[];
  models: SegmentationModel[];
  mode: "active" | "previous";
  isExpanded: boolean;
  onToggleExpand: () => void;
  onDelete: () => void;
  deleteDisabled: boolean;
  selected: boolean;
  onToggleSelect: () => void;
  colCount: number;
}

function SegmentJobRow({
  job,
  regionJobs,
  hydrophones,
  models,
  mode,
  isExpanded,
  onToggleExpand,
  onDelete,
  deleteDisabled,
  selected,
  onToggleSelect,
  colCount,
}: SegmentJobRowProps) {
  const src = sourceLabel(job, regionJobs, hydrophones);
  const mName = modelName(job, models);
  const thresh = parseThresholds(job);

  return (
    <>
      <tr
        className={`border-b hover:bg-muted/30 ${mode === "previous" && job.status === "complete" ? "cursor-pointer" : ""}`}
        onClick={
          mode === "previous" && job.status === "complete"
            ? onToggleExpand
            : undefined
        }
      >
        {mode === "previous" && (
          <td className="px-3 py-2" onClick={(e) => e.stopPropagation()}>
            <Checkbox checked={selected} onCheckedChange={onToggleSelect} />
          </td>
        )}
        <td className="px-3 py-2">
          <StatusBadge status={job.status} />
        </td>
        <td className="px-3 py-2 text-xs whitespace-nowrap">
          {new Date(job.created_at).toLocaleString()}
        </td>
        <td className="px-3 py-2 text-xs">
          {mode === "previous" ? (
            <Link
              to="/app/call-parsing/detection"
              className="text-blue-600 underline"
              onClick={(e) => e.stopPropagation()}
            >
              {src}
            </Link>
          ) : (
            src
          )}
        </td>
        <td className="px-3 py-2 text-xs">
          {mode === "previous" ? (
            <Link
              to="/app/call-parsing/segment-training"
              className="text-blue-600 underline"
              onClick={(e) => e.stopPropagation()}
            >
              {mName}
            </Link>
          ) : (
            mName
          )}
        </td>
        <td className="px-3 py-2 text-xs">
          {job.event_count != null ? job.event_count : "—"}
        </td>
        {mode === "previous" && (
          <td className="px-3 py-2 text-xs whitespace-nowrap">
            {thresh.high != null && thresh.low != null
              ? `${thresh.high.toFixed(2)} / ${thresh.low.toFixed(2)}`
              : "—"}
          </td>
        )}
        <td className="px-3 py-2" onClick={(e) => e.stopPropagation()}>
          {mode === "active" ? (
            <Button
              variant="ghost"
              size="sm"
              onClick={onDelete}
              disabled={deleteDisabled}
            >
              <X className="h-3.5 w-3.5 mr-1" />
              Cancel
            </Button>
          ) : (
            <div className="flex items-center gap-2">
              {job.status === "complete" && (
                <>
                  <Link
                    to={`/app/call-parsing/segment?tab=review&reviewJobId=${job.id}`}
                    className="rounded-md border px-2 py-1 text-xs hover:bg-accent"
                  >
                    Review
                  </Link>
                  <button
                    type="button"
                    className="p-0.5 rounded hover:bg-muted"
                    onClick={onToggleExpand}
                    aria-label="Toggle event details"
                  >
                    <ChevronDown
                      className={`h-3.5 w-3.5 text-muted-foreground transition-transform ${isExpanded ? "rotate-180" : ""}`}
                    />
                  </button>
                </>
              )}
              <Button
                variant="ghost"
                size="sm"
                className="text-red-600 hover:text-red-700"
                onClick={onDelete}
                disabled={deleteDisabled}
              >
                Delete
              </Button>
            </div>
          )}
        </td>
      </tr>
      {isExpanded && job.status === "complete" && (
        <tr className="border-b">
          <td colSpan={colCount + 1} className="px-3 py-0">
            <SegmentJobDetail jobId={job.id} />
          </td>
        </tr>
      )}
    </>
  );
}

interface SegmentJobTablePanelProps {
  title: string;
  jobs: EventSegmentationJob[];
  regionJobs: RegionDetectionJob[];
  hydrophones: HydrophoneInfo[];
  models: SegmentationModel[];
  mode: "active" | "previous";
}

export function SegmentJobTablePanel({
  title,
  jobs,
  regionJobs,
  hydrophones,
  models,
  mode,
}: SegmentJobTablePanelProps) {
  if (jobs.length === 0 && mode === "active") return null;

  return (
    <div className="border rounded-md">
      <div className="flex items-center justify-between px-4 py-3 border-b">
        <div className="flex items-center gap-2">
          <h3 className="text-sm font-semibold">{title}</h3>
          <Badge variant="secondary">{jobs.length}</Badge>
        </div>
      </div>
      <SegmentJobTable
        jobs={jobs}
        regionJobs={regionJobs}
        hydrophones={hydrophones}
        models={models}
        mode={mode}
      />
    </div>
  );
}
