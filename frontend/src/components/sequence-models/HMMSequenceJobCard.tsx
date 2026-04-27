import { Link } from "react-router-dom";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  type HMMSequenceJob,
  isHMMSequenceJobActive,
  useCancelHMMSequenceJob,
} from "@/api/sequenceModels";

interface Props {
  job: HMMSequenceJob;
}

function formatTimeAgoUtc(iso: string): string {
  const created = new Date(iso).getTime();
  const seconds = Math.max(0, Math.floor((Date.now() - created) / 1000));
  if (seconds < 60) return `${seconds}s ago (UTC)`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago (UTC)`;
  return `${Math.floor(seconds / 3600)}h ago (UTC)`;
}

export function HMMSequenceJobCard({ job }: Props) {
  const cancelMutation = useCancelHMMSequenceJob();
  const active = isHMMSequenceJobActive(job);

  return (
    <Card data-testid={`hmm-card-${job.id}`}>
      <CardContent className="py-3 space-y-2">
        <div className="flex justify-between items-start">
          <div>
            <Link
              className="font-medium text-sm text-blue-700 hover:underline"
              to={`/app/sequence-models/hmm-sequence/${job.id}`}
            >
              {job.id.slice(0, 8)}
            </Link>
            <span
              className="ml-2 inline-block px-2 py-0.5 text-xs rounded bg-slate-100"
              data-testid={`hmm-status-${job.id}`}
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
            <div className="font-medium text-slate-500">states</div>
            <div>{job.n_states}</div>
          </div>
          <div>
            <div className="font-medium text-slate-500">pca_dims</div>
            <div>{job.pca_dims}</div>
          </div>
          <div>
            <div className="font-medium text-slate-500">train_seqs</div>
            <div>{job.n_train_sequences ?? "—"}</div>
          </div>
          <div>
            <div className="font-medium text-slate-500">log_likelihood</div>
            <div>
              {job.train_log_likelihood != null
                ? job.train_log_likelihood.toFixed(1)
                : "—"}
            </div>
          </div>
        </div>

        {job.error_message ? (
          <div
            className="text-xs text-red-700 break-all"
            data-testid={`hmm-error-${job.id}`}
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
            data-testid={`hmm-cancel-${job.id}`}
          >
            Cancel
          </Button>
        ) : null}
      </CardContent>
    </Card>
  );
}
