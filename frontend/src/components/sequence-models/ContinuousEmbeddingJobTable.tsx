import { useState, useMemo, useEffect, useCallback } from "react";
import { Link } from "react-router-dom";
import {
  ArrowUp,
  ArrowDown,
  ChevronLeft,
  ChevronRight,
  Search,
  X,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { StatusBadge } from "@/components/shared/StatusBadge";
import { BulkDeleteDialog } from "@/components/classifier/BulkDeleteDialog";
import {
  type ContinuousEmbeddingJob,
  isContinuousEmbeddingJobActive,
  useCancelContinuousEmbeddingJob,
  useDeleteContinuousEmbeddingJob,
} from "@/api/sequenceModels";

type SortKey = "status" | "created" | "region" | "spans" | "windows";
type SortDir = "asc" | "desc";

interface TableProps {
  jobs: ContinuousEmbeddingJob[];
  mode: "active" | "previous";
}

export function ContinuousEmbeddingJobTable({ jobs, mode }: TableProps) {
  const cancelMutation = useCancelContinuousEmbeddingJob();
  const deleteMutation = useDeleteContinuousEmbeddingJob();

  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);
  const [bulkDeleting, setBulkDeleting] = useState(false);

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
    return jobs.filter(
      (j) =>
        j.region_detection_job_id.toLowerCase().includes(q) ||
        j.model_version.toLowerCase().includes(q),
    );
  }, [jobs, filterText, mode]);

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
        case "region":
          return (
            dir *
            a.region_detection_job_id.localeCompare(b.region_detection_job_id)
          );
        case "spans":
          return dir * ((a.merged_spans ?? 0) - (b.merged_spans ?? 0));
        case "windows":
          return dir * ((a.total_windows ?? 0) - (b.total_windows ?? 0));
        default:
          return 0;
      }
    });
    return sorted;
  }, [filteredJobs, sortKey, sortDir, mode]);

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

  const colCount = mode === "active" ? 7 : 7;

  return (
    <>
      {mode === "previous" && (
        <div className="flex items-center justify-between px-4 py-2 border-b bg-muted/30">
          <div className="flex items-center gap-2">
            <div className="relative">
              <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
              <Input
                type="search"
                placeholder="Filter by region or model…"
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
            {mode === "previous" ? (
              sortableHeader("Status", "status")
            ) : (
              <th className="px-3 py-2 text-left font-medium">Status</th>
            )}
            {mode === "previous" ? (
              sortableHeader("Created", "created")
            ) : (
              <th className="px-3 py-2 text-left font-medium">Created</th>
            )}
            {mode === "previous" ? (
              sortableHeader("Region Job", "region")
            ) : (
              <th className="px-3 py-2 text-left font-medium">Region Job</th>
            )}
            <th className="px-3 py-2 text-left font-medium">Model</th>
            {mode === "previous" ? (
              sortableHeader("Spans", "spans")
            ) : (
              <th className="px-3 py-2 text-left font-medium">Spans</th>
            )}
            {mode === "previous" ? (
              sortableHeader("Windows", "windows")
            ) : (
              <th className="px-3 py-2 text-left font-medium">Windows</th>
            )}
            <th className="px-3 py-2 text-left font-medium">
              {mode === "active" ? "Actions" : ""}
            </th>
          </tr>
        </thead>
        <tbody>
          {paginatedJobs.map((job) => (
            <tr key={job.id} className="border-b hover:bg-muted/30">
              {mode === "previous" && (
                <td
                  className="px-3 py-2"
                  onClick={(e) => e.stopPropagation()}
                >
                  <Checkbox
                    checked={selectedIds.has(job.id)}
                    onCheckedChange={() =>
                      setSelectedIds((prev) => {
                        const next = new Set(prev);
                        if (next.has(job.id)) next.delete(job.id);
                        else next.add(job.id);
                        return next;
                      })
                    }
                  />
                </td>
              )}
              <td className="px-3 py-2">
                <StatusBadge status={job.status} />
              </td>
              <td className="px-3 py-2 text-xs whitespace-nowrap">
                {new Date(job.created_at).toLocaleString()}
              </td>
              <td className="px-3 py-2 text-xs">
                {job.region_detection_job_id.slice(0, 8)}
              </td>
              <td className="px-3 py-2 text-xs">{job.model_version}</td>
              <td className="px-3 py-2 text-xs">
                {job.merged_spans ?? "—"}
              </td>
              <td className="px-3 py-2 text-xs">
                {job.total_windows ?? "—"}
              </td>
              <td
                className="px-3 py-2"
                onClick={(e) => e.stopPropagation()}
              >
                {mode === "active" ? (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => cancelMutation.mutate(job.id)}
                    disabled={cancelMutation.isPending}
                  >
                    <X className="h-3.5 w-3.5 mr-1" />
                    Cancel
                  </Button>
                ) : (
                  <div className="flex items-center gap-2">
                    {job.status === "complete" && (
                      <Link
                        to={`/app/sequence-models/continuous-embedding/${job.id}`}
                        className="rounded-md border px-2 py-1 text-xs hover:bg-accent"
                      >
                        Review
                      </Link>
                    )}
                    <Button
                      variant="ghost"
                      size="sm"
                      className="text-red-600 hover:text-red-700"
                      onClick={() => deleteMutation.mutate(job.id)}
                      disabled={deleteMutation.isPending}
                    >
                      Delete
                    </Button>
                  </div>
                )}
              </td>
            </tr>
          ))}
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
          entityName="continuous embedding job"
          onConfirm={handleBulkDelete}
          isPending={bulkDeleting}
        />
      )}
    </>
  );
}

interface PanelProps {
  title: string;
  jobs: ContinuousEmbeddingJob[];
  mode: "active" | "previous";
}

export function ContinuousEmbeddingJobTablePanel({
  title,
  jobs,
  mode,
}: PanelProps) {
  if (jobs.length === 0 && mode === "active") return null;

  return (
    <div className="border rounded-md">
      <div className="flex items-center justify-between px-4 py-3 border-b">
        <div className="flex items-center gap-2">
          <h3 className="text-sm font-semibold">{title}</h3>
          <Badge variant="secondary">{jobs.length}</Badge>
        </div>
      </div>
      <ContinuousEmbeddingJobTable jobs={jobs} mode={mode} />
    </div>
  );
}
