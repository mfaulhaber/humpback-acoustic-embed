import { Link, useParams } from "react-router-dom";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useContinuousEmbeddingJob } from "@/api/sequenceModels";

export function ContinuousEmbeddingDetailPage() {
  const { jobId } = useParams<{ jobId: string }>();
  const { data, isLoading, error } = useContinuousEmbeddingJob(jobId ?? null);

  if (isLoading) {
    return (
      <div className="text-sm text-slate-500" data-testid="cej-detail-loading">
        Loading…
      </div>
    );
  }
  if (error || !data) {
    return (
      <div className="text-sm text-red-700" data-testid="cej-detail-error">
        Job not found.
      </div>
    );
  }

  const { job, manifest } = data;

  return (
    <div className="space-y-4" data-testid="cej-detail-page">
      <div>
        <Link
          to="/app/sequence-models/continuous-embedding"
          className="text-sm text-blue-700 hover:underline"
        >
          ← Back to jobs
        </Link>
      </div>

      <Card>
        <CardHeader>
          <CardTitle data-testid="cej-detail-id">{job.id}</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2 text-sm">
          <div>
            <span className="font-medium">Status:</span>{" "}
            <span data-testid="cej-detail-status">{job.status}</span>
          </div>
          <div>
            <span className="font-medium">Region Detection Job:</span>{" "}
            {job.region_detection_job_id}
          </div>
          <div>
            <span className="font-medium">Model:</span> {job.model_version}
          </div>
          <div>
            <span className="font-medium">hop / window / pad (s):</span>{" "}
            {job.hop_seconds} / {job.window_size_seconds} / {job.pad_seconds}
          </div>
          <div>
            <span className="font-medium">Created (UTC):</span> {job.created_at}
          </div>
          {job.error_message ? (
            <div className="text-red-700" data-testid="cej-detail-error-message">
              <span className="font-medium">Error:</span> {job.error_message}
            </div>
          ) : null}
        </CardContent>
      </Card>

      {manifest ? (
        <Card>
          <CardHeader>
            <CardTitle>Manifest</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 text-sm">
            <div className="grid grid-cols-4 gap-2">
              <div>
                <div className="font-medium text-slate-500">vector_dim</div>
                <div>{manifest.vector_dim}</div>
              </div>
              <div>
                <div className="font-medium text-slate-500">regions</div>
                <div>{manifest.total_regions}</div>
              </div>
              <div>
                <div className="font-medium text-slate-500">spans</div>
                <div>{manifest.merged_spans}</div>
              </div>
              <div>
                <div className="font-medium text-slate-500">windows</div>
                <div>{manifest.total_windows}</div>
              </div>
            </div>

            <table
              className="w-full text-xs mt-2"
              data-testid="cej-detail-spans-table"
            >
              <thead>
                <tr className="text-left">
                  <th className="pr-2">span</th>
                  <th className="pr-2">start (UTC sec)</th>
                  <th className="pr-2">end (UTC sec)</th>
                  <th className="pr-2">windows</th>
                  <th>source regions</th>
                </tr>
              </thead>
              <tbody>
                {manifest.spans.map((s) => (
                  <tr key={s.merged_span_id} className="border-t">
                    <td className="pr-2 py-1">{s.merged_span_id}</td>
                    <td className="pr-2 py-1">{s.start_timestamp.toFixed(2)}</td>
                    <td className="pr-2 py-1">{s.end_timestamp.toFixed(2)}</td>
                    <td className="pr-2 py-1">{s.window_count}</td>
                    <td className="py-1">{s.source_region_ids.length}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </CardContent>
        </Card>
      ) : null}
    </div>
  );
}
