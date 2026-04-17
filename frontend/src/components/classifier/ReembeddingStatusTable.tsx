import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Loader2, RefreshCw, AlertCircle } from "lucide-react";
import type { DetectionEmbeddingJobStatus, DetectionJob } from "@/api/types";

function statusVariant(
  status: string,
): "default" | "secondary" | "destructive" | "outline" {
  switch (status) {
    case "complete":
      return "default";
    case "running":
      return "secondary";
    case "queued":
      return "outline";
    case "failed":
      return "destructive";
    default:
      return "outline";
  }
}

function progressText(row: DetectionEmbeddingJobStatus): string {
  if (row.status === "running" && row.rows_total != null && row.rows_total > 0) {
    const pct = Math.round((row.rows_processed / row.rows_total) * 100);
    return `${row.rows_processed}/${row.rows_total} (${pct}%)`;
  }
  if (row.status === "complete" && row.rows_total != null) {
    return `${row.rows_total}`;
  }
  return "—";
}

function fmtJobLabel(
  detJobId: string,
  detectionJobs: DetectionJob[],
): string {
  const dj = detectionJobs.find((j) => j.id === detJobId);
  if (!dj) return detJobId.slice(0, 8);
  return dj.hydrophone_name ?? dj.audio_folder ?? dj.id.slice(0, 8);
}

interface ReembeddingStatusTableProps {
  rows: DetectionEmbeddingJobStatus[];
  detectionJobs: DetectionJob[];
  onReembed: (detectionJobId: string) => void;
  isReembedding?: boolean;
}

export function ReembeddingStatusTable({
  rows,
  detectionJobs,
  onReembed,
  isReembedding,
}: ReembeddingStatusTableProps) {
  if (rows.length === 0) return null;

  return (
    <div className="border rounded-md overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b bg-muted/40 text-left text-muted-foreground">
            <th className="py-1.5 px-2 font-medium">Detection Job</th>
            <th className="py-1.5 px-2 font-medium">Status</th>
            <th className="py-1.5 px-2 font-medium">Rows</th>
            <th className="py-1.5 px-2 font-medium" />
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.detection_job_id} className="border-b last:border-0">
              <td className="py-1.5 px-2">
                {fmtJobLabel(row.detection_job_id, detectionJobs)}
              </td>
              <td className="py-1.5 px-2">
                <div className="flex items-center gap-1.5">
                  <Badge
                    variant={statusVariant(row.status)}
                    className="text-[10px]"
                  >
                    {row.status === "not_started"
                      ? "Not started"
                      : row.status}
                  </Badge>
                  {row.status === "running" && (
                    <Loader2 className="h-3 w-3 animate-spin text-muted-foreground" />
                  )}
                  {row.status === "failed" && row.error_message && (
                    <Popover>
                      <PopoverTrigger asChild>
                        <button className="text-destructive">
                          <AlertCircle className="h-3 w-3" />
                        </button>
                      </PopoverTrigger>
                      <PopoverContent className="text-xs max-w-sm">
                        {row.error_message}
                      </PopoverContent>
                    </Popover>
                  )}
                </div>
              </td>
              <td className="py-1.5 px-2">{progressText(row)}</td>
              <td className="py-1.5 px-2">
                {(row.status === "not_started" || row.status === "failed") && (
                  <Button
                    size="sm"
                    variant="ghost"
                    className="h-6 px-2 text-xs"
                    disabled={isReembedding}
                    onClick={() => onReembed(row.detection_job_id)}
                  >
                    <RefreshCw className="h-3 w-3 mr-1" />
                    {row.status === "failed" ? "Retry" : "Re-embed"}
                  </Button>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
