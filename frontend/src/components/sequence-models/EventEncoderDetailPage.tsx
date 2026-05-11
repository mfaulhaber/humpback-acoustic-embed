import { Link, useParams } from "react-router-dom";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useEventEncoderJob } from "@/api/sequenceModels";
import { EventEncoderTimelinePanel } from "./EventEncoderTimelinePanel";

interface SummaryPayload {
  total_events?: number;
  encoded_events?: number;
  skipped_events?: number;
  valid_k_values?: number[];
  invalid_k_values?: number[];
}

interface DescriptorStats {
  mean?: number;
  min?: number;
  max?: number;
}

export function EventEncoderDetailPage() {
  const { jobId } = useParams<{ jobId: string }>();
  const { data, isLoading, error } = useEventEncoderJob(jobId ?? null);

  if (isLoading) {
    return (
      <div className="text-sm text-slate-500" data-testid="eej-detail-loading">
        Loading...
      </div>
    );
  }
  if (error || !data) {
    return (
      <div className="text-sm text-red-700" data-testid="eej-detail-error">
        Job not found.
      </div>
    );
  }

  const { job, manifest, report } = data;
  const summary = asSummary(report?.summary);
  const sequencePreview = asStringArrayMap(report?.sequence_preview);
  const tokenization = asRecord(report?.tokenization);
  const tokenExamples = asRecord(report?.token_examples);
  const descriptorSummary = asDescriptorSummary(report?.descriptor_summary);
  const manifestRecord = asRecord(manifest);

  return (
    <div className="space-y-4" data-testid="eej-detail-page">
      <div>
        <Link
          to="/app/sequence-models/event-encoder"
          className="text-sm text-blue-700 hover:underline"
        >
          Back to jobs
        </Link>
      </div>

      <Card data-testid="eej-summary-panel">
        <CardHeader>
          <CardTitle data-testid="eej-detail-id">{job.id}</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2 text-sm">
          <div>
            <span className="font-medium">Status:</span>{" "}
            <span data-testid="eej-detail-status">{job.status}</span>
          </div>
          <div>
            <span className="font-medium">Segmentation Job:</span>{" "}
            {job.event_segmentation_job_id}
          </div>
          <div>
            <span className="font-medium">Continuous Embedding:</span>{" "}
            {job.continuous_embedding_job_id}
          </div>
          <div>
            <span className="font-medium">Event Source:</span>{" "}
            {job.event_source_mode}
          </div>
          <div>
            <span className="font-medium">Tokenizer:</span>{" "}
            {job.tokenizer_version}
          </div>
          <div>
            <span className="font-medium">Created (UTC):</span> {job.created_at}
          </div>
          {job.error_message ? (
            <div className="text-red-700" data-testid="eej-detail-error-message">
              <span className="font-medium">Error:</span> {job.error_message}
            </div>
          ) : null}
          {job.status === "complete" ? (
            <div className="pt-2">
              <Link
                to={`/app/sequence-models/event-encoder/${job.id}/piano-roll`}
                className="inline-flex h-8 items-center rounded-md border px-3 text-xs font-medium text-blue-700 hover:bg-slate-50"
                data-testid="eej-piano-roll-link"
              >
                Open Piano Roll
              </Link>
            </div>
          ) : null}
        </CardContent>
      </Card>

      <EventEncoderTimelinePanel job={job} />

      <Card data-testid="eej-report-panel">
        <CardHeader>
          <CardTitle>Report</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4 text-sm">
          <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
            <Stat label="total_events" value={summary?.total_events ?? job.total_events} />
            <Stat
              label="encoded_events"
              value={summary?.encoded_events ?? job.encoded_events}
            />
            <Stat
              label="skipped_events"
              value={summary?.skipped_events ?? job.skipped_events}
            />
            <Stat label="event_vector_dim" value={job.event_vector_dim} />
            <Stat
              label="valid_k"
              value={
                summary?.valid_k_values?.join(", ") ??
                parseJsonArray(job.k_values_json).join(", ")
              }
            />
          </div>

          {summary?.invalid_k_values?.length ? (
            <div className="flex items-center gap-2">
              <span className="font-medium text-slate-500">invalid_k</span>
              <Badge variant="secondary">
                {summary.invalid_k_values.join(", ")}
              </Badge>
            </div>
          ) : null}

          <div>
            <h3 className="mb-2 text-sm font-semibold">Sequence Preview</h3>
            <div className="space-y-2" data-testid="eej-sequence-preview">
              {Object.entries(sequencePreview).map(([k, tokens]) => (
                <div key={k} className="rounded-md border p-3">
                  <div className="mb-2 text-xs font-medium text-slate-500">
                    k={k}
                  </div>
                  <div className="flex flex-wrap gap-1.5">
                    {tokens.map((token, idx) => (
                      <Badge key={`${k}-${idx}`} variant="outline">
                        {token}
                      </Badge>
                    ))}
                  </div>
                </div>
              ))}
              {Object.keys(sequencePreview).length === 0 ? (
                <div className="text-xs text-muted-foreground">
                  No sequence preview is available yet.
                </div>
              ) : null}
            </div>
          </div>

          {Object.keys(tokenization).length ? (
            <table
              className="w-full text-xs"
              data-testid="eej-tokenization-table"
            >
              <thead>
                <tr className="border-b text-left">
                  <th className="px-2 py-1 font-medium">k</th>
                  <th className="px-2 py-1 font-medium">inertia</th>
                  <th className="px-2 py-1 font-medium">tokens</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(tokenization).map(([k, payload]) => {
                  const tokenizationRow = asRecord(payload);
                  return (
                    <tr key={k} className="border-b">
                      <td className="px-2 py-1">{k}</td>
                      <td className="px-2 py-1">
                        {formatNumber(tokenizationRow.inertia)}
                      </td>
                      <td className="px-2 py-1">
                        {formatTokenCounts(
                          asRecord(tokenizationRow.token_counts),
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          ) : null}

          {Object.keys(tokenExamples).length ? (
            <table className="w-full text-xs" data-testid="eej-exemplar-table">
              <thead>
                <tr className="border-b text-left">
                  <th className="px-2 py-1 font-medium">k</th>
                  <th className="px-2 py-1 font-medium">token</th>
                  <th className="px-2 py-1 font-medium">example_events</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(tokenExamples).flatMap(([k, payload]) =>
                  Object.entries(asRecord(payload)).map(([token, examples]) => (
                    <tr key={`${k}-${token}`} className="border-b">
                      <td className="px-2 py-1">{k}</td>
                      <td className="px-2 py-1">{token}</td>
                      <td className="px-2 py-1">
                        {formatExampleEventIds(examples)}
                      </td>
                    </tr>
                  )),
                )}
              </tbody>
            </table>
          ) : null}
        </CardContent>
      </Card>

      {Object.keys(descriptorSummary).length ? (
        <Card>
          <CardHeader>
            <CardTitle>Acoustic Descriptors</CardTitle>
          </CardHeader>
          <CardContent>
            <table
              className="w-full text-xs"
              data-testid="eej-descriptor-table"
            >
              <thead>
                <tr className="border-b text-left">
                  <th className="px-2 py-1 font-medium">descriptor</th>
                  <th className="px-2 py-1 font-medium">mean</th>
                  <th className="px-2 py-1 font-medium">min</th>
                  <th className="px-2 py-1 font-medium">max</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(descriptorSummary).map(([name, stats]) => (
                  <tr key={name} className="border-b">
                    <td className="px-2 py-1">{name}</td>
                    <td className="px-2 py-1">{formatNumber(stats.mean)}</td>
                    <td className="px-2 py-1">{formatNumber(stats.min)}</td>
                    <td className="px-2 py-1">{formatNumber(stats.max)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </CardContent>
        </Card>
      ) : null}

      {Object.keys(manifestRecord).length ? (
        <Card>
          <CardHeader>
            <CardTitle>Artifacts</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 text-xs">
            <Artifact label="event_vectors" value={job.event_vectors_path} />
            <Artifact label="event_tokens" value={job.event_tokens_path} />
            <Artifact label="token_sequences" value={job.token_sequences_path} />
            <Artifact
              label="signature"
              value={manifestRecord.continuous_embedding_signature}
            />
          </CardContent>
        </Card>
      ) : null}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: unknown }) {
  return (
    <div>
      <div className="font-medium text-slate-500">{label}</div>
      <div>{value == null || value === "" ? "-" : String(value)}</div>
    </div>
  );
}

function Artifact({ label, value }: { label: string; value: unknown }) {
  return (
    <div className="grid grid-cols-[9rem_minmax(0,1fr)] gap-2">
      <span className="font-medium text-slate-500">{label}</span>
      <span className="break-all">{value == null ? "-" : String(value)}</span>
    </div>
  );
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}

function asSummary(value: unknown): SummaryPayload | null {
  const record = asRecord(value);
  return Object.keys(record).length ? (record as SummaryPayload) : null;
}

function asStringArrayMap(value: unknown): Record<string, string[]> {
  const record = asRecord(value);
  const result: Record<string, string[]> = {};
  for (const [key, maybeTokens] of Object.entries(record)) {
    if (Array.isArray(maybeTokens)) {
      result[key] = maybeTokens.map((token) => String(token));
    }
  }
  return result;
}

function asDescriptorSummary(value: unknown): Record<string, DescriptorStats> {
  const record = asRecord(value);
  const result: Record<string, DescriptorStats> = {};
  for (const [key, maybeStats] of Object.entries(record)) {
    result[key] = asRecord(maybeStats) as DescriptorStats;
  }
  return result;
}

function parseJsonArray(value: string): unknown[] {
  try {
    const parsed = JSON.parse(value);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function formatNumber(value: unknown): string {
  return typeof value === "number" && Number.isFinite(value)
    ? value.toFixed(3)
    : "-";
}

function formatTokenCounts(counts: Record<string, unknown>): string {
  return Object.entries(counts)
    .map(
      ([token, count]) =>
        `T${Number(token).toString().padStart(2, "0")}: ${count}`,
    )
    .join(", ");
}

function formatExampleEventIds(value: unknown): string {
  if (!Array.isArray(value)) return "-";
  return value
    .map((item) => {
      const record = asRecord(item);
      return record.event_id == null ? null : String(record.event_id);
    })
    .filter((eventId): eventId is string => eventId != null)
    .join(", ");
}
