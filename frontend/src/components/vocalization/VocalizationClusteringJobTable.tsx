import { useState, useMemo, useEffect, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { ArrowUp, ArrowDown, ChevronLeft, ChevronRight, Search, X } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { StatusBadge } from "@/components/shared/StatusBadge";
import { BulkDeleteDialog } from "@/components/classifier/BulkDeleteDialog";
import { useDeleteVocalizationClusteringJob } from "@/hooks/queries/useVocalization";
import type { ClusteringJob } from "@/api/types";

function detectionJobsSummary(job: ClusteringJob): string {
  const ids = job.detection_job_ids;
  if (!ids || ids.length === 0) return "—";
  return `${ids.length} detection job${ids.length !== 1 ? "s" : ""}`;
}

function clusterCount(job: ClusteringJob): number | null {
  if (!job.metrics) return null;
  const n = job.metrics.n_clusters;
  return typeof n === "number" ? n : null;
}

type SortKey = "status" | "created" | "clusters";
type SortDir = "asc" | "desc";

interface VocalizationClusteringJobTableProps {
  jobs: ClusteringJob[];
  mode: "active" | "previous";
}

export function VocalizationClusteringJobTable({
  jobs,
  mode,
}: VocalizationClusteringJobTableProps) {
  const deleteMutation = useDeleteVocalizationClusteringJob();
  const navigate = useNavigate();

  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);
  const [bulkDeleting, setBulkDeleting] = useState(false);
  const [filterText, setFilterText] = useState("");
  const [sortKey, setSortKey] = useState<SortKey>("created");
  const [sortDir, setSortDir] = useState<SortDir>("desc");
  const [page, setPage] = useState(1);
  const [pageSize] = useState(20);

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
    return jobs.filter((j) => {
      const haystack = [
        detectionJobsSummary(j),
        j.status,
        j.error_message ?? "",
        j.id,
      ].join(" ").toLowerCase();
      return haystack.includes(q);
    });
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
          return dir * (new Date(a.created_at).getTime() - new Date(b.created_at).getTime());
        case "clusters":
          return dir * ((clusterCount(a) ?? -1) - (clusterCount(b) ?? -1));
        default:
          return 0;
      }
    });
    return sorted;
  }, [filteredJobs, sortKey, sortDir, mode]);

  const totalPages = Math.max(1, Math.ceil(sortedJobs.length / pageSize));
  const effectivePage = Math.min(page, totalPages);

  const paginatedJobs = useMemo(() => {
    if (mode === "active") return sortedJobs;
    const start = (effectivePage - 1) * pageSize;
    return sortedJobs.slice(start, start + pageSize);
  }, [sortedJobs, effectivePage, pageSize, mode]);

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

  return (
    <>
      {mode === "previous" && (
        <div className="flex items-center justify-between px-4 py-2 border-b bg-muted/30">
          <div className="flex items-center gap-2">
            <div className="relative">
              <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
              <Input
                type="search"
                placeholder="Filter…"
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
                ? `${(effectivePage - 1) * pageSize + 1}–${Math.min(effectivePage * pageSize, sortedJobs.length)} of ${sortedJobs.length}`
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
                    const allSel = paginatedJobs.every((j) => selectedIds.has(j.id));
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
            {mode === "previous" ? sortableHeader("Status", "status") : <th className="px-3 py-2 text-left font-medium">Status</th>}
            {mode === "previous" ? sortableHeader("Created", "created") : <th className="px-3 py-2 text-left font-medium">Created</th>}
            <th className="px-3 py-2 text-left font-medium">Detection Jobs</th>
            {mode === "previous" && sortableHeader("Clusters", "clusters")}
            {mode === "previous" && <th className="px-3 py-2 text-left font-medium">Error</th>}
            <th className="px-3 py-2 text-left font-medium">Actions</th>
          </tr>
        </thead>
        <tbody>
          {paginatedJobs.map((job) => (
            <tr key={job.id} className="border-b hover:bg-muted/30">
              {mode === "previous" && (
                <td className="px-3 py-2">
                  <Checkbox
                    checked={selectedIds.has(job.id)}
                    onCheckedChange={() => {
                      setSelectedIds((prev) => {
                        const next = new Set(prev);
                        if (next.has(job.id)) next.delete(job.id);
                        else next.add(job.id);
                        return next;
                      });
                    }}
                  />
                </td>
              )}
              <td className="px-3 py-2">
                <StatusBadge status={job.status} />
              </td>
              <td className="px-3 py-2 text-xs whitespace-nowrap">
                {new Date(job.created_at).toISOString().replace("T", " ").replace(/\.\d+Z$/, " UTC")}
              </td>
              <td className="px-3 py-2 text-xs">{detectionJobsSummary(job)}</td>
              {mode === "previous" && (
                <td className="px-3 py-2 text-xs">
                  {clusterCount(job) ?? "—"}
                </td>
              )}
              {mode === "previous" && (
                <td className="px-3 py-2 text-xs text-red-600 max-w-[200px] truncate">
                  {job.error_message ?? ""}
                </td>
              )}
              <td className="px-3 py-2">
                {mode === "active" ? (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => deleteMutation.mutate(job.id)}
                    disabled={deleteMutation.isPending}
                  >
                    <X className="h-3.5 w-3.5 mr-1" />
                    Cancel
                  </Button>
                ) : (
                  job.status === "complete" && (
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() =>
                        navigate(`/app/vocalization/clustering/${job.id}`)
                      }
                    >
                      View
                    </Button>
                  )
                )}
              </td>
            </tr>
          ))}
          {paginatedJobs.length === 0 && (
            <tr>
              <td
                colSpan={mode === "active" ? 4 : 7}
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
          entityName="vocalization clustering job"
          onConfirm={handleBulkDelete}
          isPending={bulkDeleting}
        />
      )}
    </>
  );
}

interface VocalizationClusteringJobTablePanelProps {
  title: string;
  jobs: ClusteringJob[];
  mode: "active" | "previous";
}

export function VocalizationClusteringJobTablePanel({
  title,
  jobs,
  mode,
}: VocalizationClusteringJobTablePanelProps) {
  if (jobs.length === 0 && mode === "active") return null;

  return (
    <div className="border rounded-md">
      <div className="flex items-center justify-between px-4 py-3 border-b">
        <div className="flex items-center gap-2">
          <h3 className="text-sm font-semibold">{title}</h3>
          <Badge variant="secondary">{jobs.length}</Badge>
        </div>
      </div>
      <VocalizationClusteringJobTable jobs={jobs} mode={mode} />
    </div>
  );
}
