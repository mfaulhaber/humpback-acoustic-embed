import { Link } from "react-router-dom";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  type ContinuousEmbeddingJob,
  isContinuousEmbeddingJobActive,
  useCancelContinuousEmbeddingJob,
} from "@/api/sequenceModels";

interface Props {
  job: ContinuousEmbeddingJob;
}

function formatTimeAgoUtc(iso: string): string {
  const created = new Date(iso).getTime();
  const seconds = Math.max(0, Math.floor((Date.now() - created) / 1000));
  if (seconds < 60) return `${seconds}s ago (UTC)`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago (UTC)`;
  return `${Math.floor(seconds / 3600)}h ago (UTC)`;
}

export function ContinuousEmbeddingJobCard({ job }: Props) {
  const cancelMutation = useCancelContinuousEmbeddingJob();
  const active = isContinuousEmbeddingJobActive(job);

  return (
    <Card data-testid={`cej-card-${job.id}`}>
      <CardContent className="py-3 space-y-2">
        <div className="flex justify-between items-start">
          <div>
            <Link
              className="font-medium text-sm text-blue-700 hover:underline"
              to={`/app/sequence-models/continuous-embedding/${job.id}`}
            >
              {job.id.slice(0, 8)}
            </Link>
            <span
              className="ml-2 inline-block px-2 py-0.5 text-xs rounded bg-slate-100"
              data-testid={`cej-status-${job.id}`}
            >
              {job.status}
            </span>
          </div>
          <div className="text-xs text-slate-500">
            {formatTimeAgoUtc(job.created_at)}
          </div>
        </div>

        <div className="grid grid-cols-4 gap-2 text-xs text-slate-600">
          <div>
            <div className="font-medium text-slate-500">regions</div>
            <div>{job.total_regions ?? "—"}</div>
          </div>
          <div>
            <div className="font-medium text-slate-500">spans</div>
            <div>{job.merged_spans ?? "—"}</div>
          </div>
          <div>
            <div className="font-medium text-slate-500">windows</div>
            <div>{job.total_windows ?? "—"}</div>
          </div>
          <div>
            <div className="font-medium text-slate-500">vector_dim</div>
            <div>{job.vector_dim ?? "—"}</div>
          </div>
        </div>

        {job.error_message ? (
          <div
            className="text-xs text-red-700 break-all"
            data-testid={`cej-error-${job.id}`}
          >
            {job.error_message}
          </div>
        ) : null}

        {active ? (
          <Button
            size="sm"
            variant="outline"
            disabled={cancelMutation.isPending}
            onClick={() => cancelMutation.mutate(job.id)}
            data-testid={`cej-cancel-${job.id}`}
          >
            Cancel
          </Button>
        ) : null}
      </CardContent>
    </Card>
  );
}
