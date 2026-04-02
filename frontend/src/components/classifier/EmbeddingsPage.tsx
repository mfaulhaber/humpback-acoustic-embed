import { useState } from "react";
import { Link } from "react-router-dom";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { useEmbeddingJobs } from "@/hooks/queries/useVocalization";
import type { EmbeddingJobListItem } from "@/api/types";

const PAGE_SIZE = 20;

function statusVariant(status: string): "default" | "secondary" | "destructive" | "outline" {
  switch (status) {
    case "complete": return "default";
    case "running": return "secondary";
    case "queued": return "outline";
    case "failed": return "destructive";
    default: return "outline";
  }
}

function formatDuration(created: string, updated: string): string {
  const ms = new Date(updated).getTime() - new Date(created).getTime();
  if (ms < 0) return "-";
  const sec = Math.round(ms / 1000);
  if (sec < 60) return `${sec}s`;
  const min = Math.floor(sec / 60);
  const rem = sec % 60;
  return `${min}m ${rem}s`;
}

function SyncSummary({ summary }: { summary: string }) {
  try {
    const s = JSON.parse(summary);
    const parts: string[] = [];
    if (s.added > 0) parts.push(`+${s.added}`);
    if (s.removed > 0) parts.push(`-${s.removed}`);
    if (s.unchanged > 0) parts.push(`=${s.unchanged}`);
    if (s.skipped > 0) parts.push(`skip ${s.skipped}`);
    return <span className="text-xs text-muted-foreground">{parts.join("  ")}</span>;
  } catch {
    return null;
  }
}

function FullSummary({ summary }: { summary: string }) {
  try {
    const s = JSON.parse(summary);
    return <span className="text-xs text-muted-foreground">{s.total} embeddings</span>;
  } catch {
    return null;
  }
}

function ProgressCell({ job }: { job: EmbeddingJobListItem }) {
  if (job.status === "complete" && job.result_summary) {
    if (job.mode === "sync") return <SyncSummary summary={job.result_summary} />;
    return <FullSummary summary={job.result_summary} />;
  }
  if (job.status === "running" || job.status === "queued") {
    if (job.progress_total != null && job.progress_total > 0) {
      return (
        <span className="text-xs text-muted-foreground">
          {job.progress_current ?? 0} / {job.progress_total}
        </span>
      );
    }
    return <span className="text-xs text-muted-foreground">pending</span>;
  }
  if (job.status === "failed") {
    return (
      <span className="text-xs text-destructive truncate max-w-[200px] inline-block" title={job.error_message ?? ""}>
        {job.error_message ?? "unknown error"}
      </span>
    );
  }
  return <span className="text-xs text-muted-foreground">-</span>;
}

function DetectionJobLink({ job }: { job: EmbeddingJobListItem }) {
  const label = job.hydrophone_name ?? job.audio_folder ?? job.detection_job_id.slice(0, 8);
  return (
    <Link
      to={`/app/classifier/timeline/${job.detection_job_id}`}
      className="text-xs text-blue-600 hover:underline"
    >
      {label}
    </Link>
  );
}

export function EmbeddingsPage() {
  const [page, setPage] = useState(0);
  const offset = page * PAGE_SIZE;
  const { data: jobs = [], isLoading } = useEmbeddingJobs(offset, PAGE_SIZE);

  return (
    <div className="p-6 max-w-5xl">
      <h1 className="text-lg font-semibold mb-4">Embedding Jobs</h1>

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm text-muted-foreground">
            Detection embedding generation and sync history
          </CardTitle>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <p className="text-sm text-muted-foreground py-4">Loading...</p>
          ) : jobs.length === 0 ? (
            <p className="text-sm text-muted-foreground py-4">
              {page === 0 ? "No embedding jobs yet." : "No more jobs."}
            </p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b text-left text-muted-foreground">
                    <th className="py-2 pr-3 font-medium">Status</th>
                    <th className="py-2 pr-3 font-medium">Detection Job</th>
                    <th className="py-2 pr-3 font-medium">Mode</th>
                    <th className="py-2 pr-3 font-medium">Progress / Summary</th>
                    <th className="py-2 pr-3 font-medium">Created</th>
                    <th className="py-2 pr-3 font-medium">Duration</th>
                  </tr>
                </thead>
                <tbody>
                  {jobs.map((job) => (
                    <tr key={job.id} className="border-b last:border-0">
                      <td className="py-2 pr-3">
                        <Badge variant={statusVariant(job.status)} className="text-[10px]">
                          {job.status}
                        </Badge>
                      </td>
                      <td className="py-2 pr-3">
                        <DetectionJobLink job={job} />
                      </td>
                      <td className="py-2 pr-3 text-muted-foreground">
                        {job.mode === "sync" ? "Sync" : "Full"}
                      </td>
                      <td className="py-2 pr-3">
                        <ProgressCell job={job} />
                      </td>
                      <td className="py-2 pr-3 text-muted-foreground whitespace-nowrap">
                        {new Date(job.created_at).toLocaleString()}
                      </td>
                      <td className="py-2 pr-3 text-muted-foreground">
                        {job.status === "complete" || job.status === "failed"
                          ? formatDuration(job.created_at, job.updated_at)
                          : "-"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* Pagination */}
          <div className="flex items-center justify-between mt-3 pt-2 border-t">
            <Button
              variant="ghost"
              size="sm"
              disabled={page === 0}
              onClick={() => setPage((p) => Math.max(0, p - 1))}
            >
              <ChevronLeft className="h-4 w-4 mr-1" /> Prev
            </Button>
            <span className="text-xs text-muted-foreground">Page {page + 1}</span>
            <Button
              variant="ghost"
              size="sm"
              disabled={jobs.length < PAGE_SIZE}
              onClick={() => setPage((p) => p + 1)}
            >
              Next <ChevronRight className="h-4 w-4 ml-1" />
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
